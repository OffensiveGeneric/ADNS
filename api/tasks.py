from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Iterable, Sequence

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from app import Flow, Prediction, app, db
from model_runner import DetectionEngine

logger = logging.getLogger(__name__)
detector = DetectionEngine()

SCORING_FETCH_CHUNK = int(os.environ.get("ADNS_SCORING_FETCH_CHUNK", "256"))


def _chunked(ids: Sequence[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(ids), size):
        yield list(ids[start : start + size])


def _insert_predictions(records: list[dict]) -> int:
    if not records:
        return 0

    bind = db.session.get_bind()
    dialect = getattr(bind, "dialect", None)
    if dialect and dialect.name == "postgresql":
        stmt = pg_insert(Prediction).values(records).on_conflict_do_nothing(index_elements=["flow_id"])
        result = db.session.execute(stmt)
        return result.rowcount or 0

    # Fallback for dev databases without PostgreSQL features.
    flow_ids = [rec["flow_id"] for rec in records]
    existing = {
        row.flow_id
        for row in db.session.query(Prediction.flow_id).filter(Prediction.flow_id.in_(flow_ids)).all()
    }
    to_insert = [rec for rec in records if rec["flow_id"] not in existing]
    if not to_insert:
        return 0
    try:
        db.session.bulk_insert_mappings(Prediction, to_insert)
        return len(to_insert)
    except IntegrityError:
        db.session.rollback()
        logger.warning("bulk insert hit integrity error on non-postgres backend; retrying row by row")
        inserted = 0
        for rec in to_insert:
            try:
                db.session.add(Prediction(**rec))
                db.session.flush()
                inserted += 1
            except IntegrityError:
                db.session.rollback()
        return inserted


def score_flow_batch(flow_ids: Sequence[int]) -> int:
    """
    Background job entrypoint for scoring a batch of newly ingested flows.
    """
    if not flow_ids:
        return 0

    with app.app_context():
        ids = [int(fid) for fid in flow_ids if fid]
        if not ids:
            return 0

        detector.reload_if_stale()
        scored = 0
        session = db.session

        try:
            for chunk_ids in _chunked(ids, SCORING_FETCH_CHUNK):
                flows: Iterable[Flow] = (
                    session.query(Flow)
                    .filter(Flow.id.in_(chunk_ids))
                    .order_by(Flow.id.asc())
                    .all()
                )
                if not flows:
                    continue

                predictions = detector.predict_many(session, flows)
                if len(predictions) != len(flows):
                    raise RuntimeError("detection engine returned mismatched prediction count")
                now = datetime.now(timezone.utc)
                records = [
                    {
                        "flow_id": flow.id,
                        "score": score,
                        "label": label,
                        "created_at": now,
                    }
                    for flow, (score, label) in zip(flows, predictions)
                ]

                inserted = _insert_predictions(records)
                scored += inserted

            if scored:
                session.commit()
                logger.info("scored %d flow(s) via RQ job", scored)
            else:
                session.rollback()
            return scored
        except Exception:
            session.rollback()
            raise
        finally:
            session.remove()
