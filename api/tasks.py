from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, Sequence

from app import Flow, Prediction, app, db
from model_runner import DetectionEngine

logger = logging.getLogger(__name__)
detector = DetectionEngine()


def _missing_prediction_ids(flow_ids: Sequence[int]) -> set[int]:
    existing = (
        Prediction.query.with_entities(Prediction.flow_id)
        .filter(Prediction.flow_id.in_(flow_ids))
        .all()
    )
    return {row.flow_id for row in existing}


def score_flow_batch(flow_ids: Sequence[int]) -> int:
    """
    Background job entrypoint for scoring a batch of newly ingested flows.
    """
    if not flow_ids:
        return 0

    with app.app_context():
        ids = [int(fid) for fid in flow_ids if fid]
        already_scored = _missing_prediction_ids(ids)
        flows: Iterable[Flow] = Flow.query.filter(Flow.id.in_(ids)).all()

        scored = 0
        for flow in flows:
            if flow.id in already_scored:
                continue

            score, label = detector.predict(db.session, flow)
            db.session.add(
                Prediction(
                    flow_id=flow.id,
                    score=score,
                    label=label,
                    created_at=datetime.now(timezone.utc),
                )
            )
            scored += 1

        if scored:
            db.session.commit()
            logger.info("scored %d flow(s) via RQ job", scored)
        else:
            db.session.rollback()
        return scored
