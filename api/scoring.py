from __future__ import annotations

import hashlib
import ipaddress
from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple

from sqlalchemy.orm import Session

from app import Flow

PROTO_BONUS = {
    "ICMP": 0.08,
    "UDP": 0.05,
    "SCTP": 0.04,
    "GRE": 0.03,
}


def is_private_ip(value: str) -> bool:
    try:
        ip_obj = ipaddress.ip_address(value)
        return ip_obj.is_private
    except ValueError:
        return False


@dataclass
class FlowScorer:
    burst_window_seconds: int = 90
    max_bytes_baseline: int = 20000

    def predict(self, session: Session, flow: Flow) -> Tuple[float, str]:
        score = (
            0.15
            + 0.45 * self._bytes_score(flow.bytes)
            + 0.25 * self._burst_score(session, flow)
            + self._direction_bonus(flow)
            + self._proto_bonus(flow.proto)
            + self._stable_jitter(flow)
        )
        score = max(0.0, min(1.0, score))

        if score >= 0.85:
            label = "anomaly"
        elif score >= 0.65:
            label = "watch"
        else:
            label = "normal"
        return score, label

    def _bytes_score(self, byte_count: int) -> float:
        if byte_count <= 0:
            return 0.0
        normalized = byte_count / float(self.max_bytes_baseline)
        return min(1.0, normalized ** 0.6)

    def _burst_score(self, session: Session, flow: Flow) -> float:
        if not flow.src_ip:
            return 0.0
        lower_bound = flow.timestamp - timedelta(seconds=self.burst_window_seconds)

        recent_count = (
            session.query(Flow.id)
            .filter(
                Flow.src_ip == flow.src_ip,
                Flow.timestamp >= lower_bound,
                Flow.id != flow.id,
            )
            .count()
        )

        if recent_count == 0:
            return 0.0
        return min(1.0, recent_count / 60.0)

    def _direction_bonus(self, flow: Flow) -> float:
        if not flow.src_ip or not flow.dst_ip:
            return 0.0
        src_private = is_private_ip(flow.src_ip)
        dst_private = is_private_ip(flow.dst_ip)

        if src_private and not dst_private:
            return 0.07
        if not src_private and dst_private:
            return 0.05
        if not src_private and not dst_private:
            return 0.03
        return 0.0

    def _proto_bonus(self, proto: str) -> float:
        proto_upper = (proto or "").upper()
        return PROTO_BONUS.get(proto_upper, 0.0)

    def _stable_jitter(self, flow: Flow) -> float:
        key = f"{flow.src_ip}-{flow.dst_ip}-{flow.proto}-{int(flow.timestamp.timestamp())}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        jitter = int(digest[:6], 16) / float(0xFFFFFF)
        return jitter * 0.05
