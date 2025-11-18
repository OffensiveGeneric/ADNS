#!/usr/bin/env python3
"""
train_flow_detector.py
----------------------

Trains a lightweight anomaly detector that only relies on the features our live
pipeline actually collects from tshark (`bytes` + `proto`).  It ingests the
UNSW-NB15 training/testing CSVs, engineers a small feature frame that mirrors
the streaming payload (total bytes, log bytes, protocol), fits a calibrated
logistic regression model, and exports a Joblib artifact that can be consumed
by the scoring worker.

Example:
    python train_flow_detector.py \
        --raw_train data/DataSet/UNSW-NB15/'Training and Testing Sets'/UNSW_NB15_training-set.csv \
        --raw_test  data/DataSet/UNSW-NB15/'Training and Testing Sets'/UNSW_NB15_testing-set.csv \
        --model_out api/model_artifacts/flow_detector.joblib \
        --metrics_out api/model_artifacts/flow_detector_metrics.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUM_FEATURES = ["total_bytes", "log_total_bytes"]
CAT_FEATURES = ["proto"]


def _best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f1 = 2 * (prec * rec) / np.maximum(prec + rec, 1e-9)
    idx = int(np.nanargmax(f1))
    if len(thr) == 0:
        return 0.5
    return float(np.clip(thr[max(0, idx - 1)], 0.05, 0.95))


def _metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    y_pred = (prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "threshold": float(threshold),
    }


def _load_csv(path: Path, sample: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if sample is not None and sample > 0 and len(df) > sample:
        df = df.sample(n=sample, random_state=42)
    return df


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    sbytes = pd.to_numeric(df.get("sbytes", 0), errors="coerce").fillna(0)
    dbytes = pd.to_numeric(df.get("dbytes", 0), errors="coerce").fillna(0)
    total_bytes = (sbytes + dbytes).clip(lower=0).astype(float)

    proto_series = (
        df.get("proto", pd.Series(["OTHER"] * len(df)))
        .astype(str)
        .str.upper()
        .replace({"-": "OTHER"})
    )

    feat = pd.DataFrame(
        {
            "total_bytes": total_bytes,
            "log_total_bytes": np.log1p(total_bytes),
            "proto": proto_series,
        }
    )
    return feat


def _build_pipeline() -> CalibratedClassifierCV:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, NUM_FEATURES),
            ("cat", cat_pipe, CAT_FEATURES),
        ]
    )
    base = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)


@dataclass
class TrainResult:
    model: CalibratedClassifierCV
    threshold: float
    watch_threshold: float
    val_metrics: dict
    test_metrics: dict


def train_detector(
    raw_train: Path,
    raw_test: Path,
    random_state: int = 7,
    holdout: float = 0.2,
    sample: int | None = None,
) -> TrainResult:
    train_df = _load_csv(raw_train, sample=sample)
    test_df = _load_csv(raw_test, sample=sample)

    X = _feature_frame(train_df)
    y = pd.to_numeric(train_df["label"], errors="coerce").fillna(0).astype(int).values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=holdout,
        random_state=random_state,
        stratify=y,
    )

    clf = _build_pipeline()
    clf.fit(X_train, y_train)

    val_prob = clf.predict_proba(X_val)[:, 1]
    threshold = _best_f1_threshold(y_val, val_prob)
    watch_threshold = max(0.2, min(threshold * 0.65, threshold))

    val_metrics = _metrics(y_val, val_prob, threshold)

    test_features = _feature_frame(test_df)
    y_test = pd.to_numeric(test_df["label"], errors="coerce").fillna(0).astype(int).values
    test_prob = clf.predict_proba(test_features)[:, 1]
    test_metrics = _metrics(y_test, test_prob, threshold)
    test_metrics["watch_threshold"] = watch_threshold

    return TrainResult(
        model=clf,
        threshold=threshold,
        watch_threshold=watch_threshold,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Train bytes+protocol anomaly detector")
    ap.add_argument("--raw_train", required=True, type=Path)
    ap.add_argument("--raw_test", required=True, type=Path)
    ap.add_argument("--model_out", default="outputs/flow_detector/flow_detector.joblib", type=Path)
    ap.add_argument(
        "--metrics_out",
        default="outputs/flow_detector/flow_detector_metrics.json",
        type=Path,
    )
    ap.add_argument("--random_state", type=int, default=7)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--sample", type=int, default=None, help="Optional random sample size for quick iteration")
    args = ap.parse_args()

    result = train_detector(
        raw_train=args.raw_train,
        raw_test=args.raw_test,
        random_state=args.random_state,
        holdout=args.holdout,
        sample=args.sample,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": result.model,
        "threshold_anomaly": result.threshold,
        "threshold_watch": result.watch_threshold,
        "feature_version": 1,
    }
    dump(payload, args.model_out)

    metrics = {
        "validation": result.val_metrics,
        "test": result.test_metrics,
    }
    args.metrics_out.write_text(json.dumps(metrics, indent=2))

    print("Model saved to", args.model_out)
    print("Metrics saved to", args.metrics_out)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
