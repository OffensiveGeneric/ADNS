#!/usr/bin/env python3
"""
Background worker that reads new Flow rows, computes a placeholder anomaly score,
and writes results to the Prediction table. Replace `score_flow` with a real ML
model when available.
"""
import logging
import os
import time
from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app import Flow, Prediction, db
from scoring import FlowScorer

DB_URL = os.environ.get("ADNS_DATABASE_URL", "postgresql://adns:adns_password@127.0.0.1/adns")
POLL_INTERVAL = float(os.environ.get("ADNS_WORKER_POLL_SECONDS", "5"))
BATCH_SIZE = int(os.environ.get("ADNS_WORKER_BATCH_SIZE", "50"))


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [worker] %(message)s",
    )


scorer = FlowScorer()


def fetch_unscored_flows(session: Session, limit: int) -> list[Flow]:
    stmt = (
        select(Flow)
        .outerjoin(Prediction)
        .where(Prediction.id.is_(None))
        .order_by(Flow.timestamp.desc())
        .limit(limit)
    )
    return session.scalars(stmt).all()


def process_batch(session: Session) -> int:
    flows = fetch_unscored_flows(session, BATCH_SIZE)
    if not flows:
        return 0

    for flow in flows:
        score, label = scorer.predict(session, flow)
        prediction = Prediction(
            flow_id=flow.id,
            score=score,
            label=label,
            created_at=datetime.now(timezone.utc),
        )
        session.add(prediction)

    session.commit()
    return len(flows)


def main() -> None:
    configure_logging()
    engine = create_engine(DB_URL)
    logging.info("worker connected to %s", DB_URL)

    while True:
        with Session(engine) as session:
            try:
                processed = process_batch(session)
                if processed:
                    logging.info("scored %d flow(s)", processed)
                else:
                    logging.debug("no unscored flows found")
            except Exception as exc:  # pragma: no cover
                session.rollback()
                logging.exception("error while scoring flows: %s", exc)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
