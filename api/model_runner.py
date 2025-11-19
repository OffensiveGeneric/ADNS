from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "model_artifacts" / "flow_detector.joblib"


class FlowModel:
    """Wrapper around the trained sklearn pipeline."""

    def __init__(self, model_path: str | os.PathLike | None = None) -> None:
        resolved = Path(model_path or os.environ.get("ADNS_MODEL_PATH", DEFAULT_MODEL_PATH))
        if not resolved.exists():
            raise FileNotFoundError(f"model artifact not found at {resolved}")
        payload = joblib.load(resolved)
        self.pipeline = payload["model"]
        self.anomaly_threshold = float(payload.get("threshold_anomaly", 0.6))
        self.watch_threshold = float(
            payload.get("threshold_watch", max(0.2, self.anomaly_threshold * 0.65))
        )

    def _feature_dict(self, bytes_count: int, proto: str) -> dict:
        total_bytes = max(0.0, float(bytes_count or 0))
        proto_norm = (proto or "OTHER").upper()
        return {
            "total_bytes": total_bytes,
            "log_total_bytes": math.log1p(total_bytes),
            "proto": proto_norm,
        }

    def score(self, bytes_count: int, proto: str) -> Tuple[float, str]:
        row = pd.DataFrame([self._feature_dict(bytes_count, proto)])
        prob = float(self.pipeline.predict_proba(row)[0][1])
        if prob >= self.anomaly_threshold:
            label = "anomaly"
        elif prob >= self.watch_threshold:
            label = "watch"
        else:
            label = "normal"
        return prob, label


class DetectionEngine:
    """
    Attempts to load the trained FlowModel; falls back to FlowScorer heuristics
    if the artifact has not been provisioned yet.
    """

    def __init__(self) -> None:
        try:
            self.model = FlowModel()
            self._mode = "ml"
        except FileNotFoundError:
            from scoring import FlowScorer  # deferred import to avoid optional deps at import time

            self.model = FlowScorer()
            self._mode = "heuristic"

    @property
    def mode(self) -> str:
        return self._mode

    def predict(self, session, flow) -> Tuple[float, str]:
        if self._mode == "ml":
            return self.model.score(flow.bytes, flow.proto)
        return self.model.predict(session, flow)
