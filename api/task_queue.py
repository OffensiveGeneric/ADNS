from __future__ import annotations

import logging
import os
from typing import Sequence

from redis import Redis
from rq import Queue

logger = logging.getLogger(__name__)

_queue: Queue | None = None
_redis_url = os.environ.get("ADNS_REDIS_URL", "redis://127.0.0.1:6379/0")


def _queue_kwargs() -> dict:
    return {
        "default_timeout": int(os.environ.get("ADNS_RQ_JOB_TIMEOUT", "120")),
        "connection": Redis.from_url(_redis_url),
    }


def _get_queue() -> Queue:
    global _queue
    if _queue is None:
        queue_name = os.environ.get("ADNS_RQ_QUEUE", "flow_scores")
        kwargs = _queue_kwargs()
        _queue = Queue(queue_name, **kwargs)
        logger.info("initialized RQ queue '%s' using redis %s", queue_name, _redis_url)
    return _queue


def enqueue_flow_scoring(flow_ids: Sequence[int]) -> int:
    """
    Enqueue one or more flow IDs for asynchronous scoring.
    Returns the number of IDs enqueued.
    """
    ids = [int(fid) for fid in flow_ids if fid]
    if not ids:
        return 0

    batch_size = int(os.environ.get("ADNS_RQ_BATCH_SIZE", "100"))
    queue = _get_queue()
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        queue.enqueue("tasks.score_flow_batch", chunk)
    return len(ids)
