#!/usr/bin/env python3
"""
RQ worker bootstrap for asynchronous flow scoring.
"""

import logging
import os

from redis import Redis
from rq import Worker


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [worker] %(message)s",
    )


def main() -> None:
    configure_logging()

    redis_url = os.environ.get("ADNS_REDIS_URL", "redis://127.0.0.1:6379/0")
    queue_names = [
        queue.strip()
        for queue in os.environ.get("ADNS_RQ_QUEUE", "flow_scores").split(",")
        if queue.strip()
    ]

    connection = Redis.from_url(redis_url)
    logging.info("starting RQ worker for queues=%s redis=%s", queue_names, redis_url)

    worker = Worker(queue_names, connection=connection)
    worker.work()


if __name__ == "__main__":
    main()
