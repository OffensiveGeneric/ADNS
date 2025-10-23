#!/usr/bin/env python3
"""
detectors/run_detectors.py

Load train_clean.csv and test_clean.csv (defaults resolve relative to repo root),
select numeric features, optionally subsample training rows for speed,
train IsolationForest & LOF (novelty=True), compute normalized anomaly scores,
and write train_with_scores.csv and test_with_scores.csv.

Usage:
    python detectors/run_detectors.py --data_dir ./outputs/preprocessed --out_dir ./outputs/preprocessed --sample 5000
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, MinMaxScaler

# -------------------------------
# I/O utilities
# -------------------------------
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, low_memory=False)

def select_numeric(df: pd.DataFrame) -> list[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove supervision label if present
    if "label" in cols:
        cols = [c for c in cols if c != "label"]
    return cols

# -------------------------------
# Model fitting & scoring
# -------------------------------
def fit_models(X_train_s: np.ndarray, n_neighbors: int | None = None, sample_limit: int = 5000, random_state: int = 42):
    """Fit IsolationForest and LOF (novelty) on *scaled* training features."""
    n_train = X_train_s.shape[0]

    # IsolationForest: constrain max_samples
    max_samples = min(sample_limit, n_train) if sample_limit else n_train
    max_samples = max(1, max_samples)

    iso = IsolationForest(
        n_estimators=100,
        max_samples=max_samples,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_train_s)

    # LOF with novelty=True requires n_neighbors < n_train
    if n_neighbors is None:
        # choose up to 20, but strictly less than n_train and >= 2
        n_neighbors = max(2, min(20, n_train - 1))
    else:
        n_neighbors = max(2, min(n_neighbors, n_train - 1))

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(X_train_s)

    return iso, lof

def score_models(iso, lof, X_train_s: np.ndarray, X_test_s: np.ndarray):
    """Return raw (unnormalized) anomaly scores for train and test (higher = more anomalous)."""
    iso_train_raw = -iso.decision_function(X_train_s)
    iso_test_raw  = -iso.decision_function(X_test_s)

    lof_train_raw = -lof.score_samples(X_train_s)
    lof_test_raw  = -lof.score_samples(X_test_s)

    return (iso_train_raw, lof_train_raw), (iso_test_raw, lof_test_raw)

# -------------------------------
# Main
# -------------------------------
def main():
    # --- repo-root-relative defaults (portable for everyone) ---
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]  # one level above detectors/
    default_data_dir = repo_root / "outputs" / "preprocessed"
    default_out_dir  = repo_root / "outputs" / "preprocessed"

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None,
                   help=f"Directory containing train_clean.csv/test_clean.csv (default: {default_data_dir})")
    p.add_argument("--out_dir", default=None,
                   help=f"Output directory for *_with_scores.csv (default: {default_out_dir})")
    p.add_argument("--sample", type=int, default=5000,
                   help="Max train rows to fit models on (for speed). Full train/test still get scored.")
    p.add_argument("--lof_neighbors", type=int, default=None,
                   help="Override n_neighbors for LOF (must be < train_sample_size).")
    args = p.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else default_data_dir
    out_dir  = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = data_dir / "train_clean.csv"
    test_csv  = data_dir / "test_clean.csv"

    print(f"[paths] data_dir = {data_dir}")
    print(f"[paths] out_dir  = {out_dir}")
    print(f"[info] reading: {train_csv.name}, {test_csv.name}")

    train_df = load_csv(train_csv)
    test_df  = load_csv(test_csv)

    numeric = select_numeric(train_df)
    if not numeric:
        raise SystemExit("[ERROR] No numeric features found in train_clean.csv - check preprocessing.")

    # -------------------------------
    # Build train sample for *fitting* (speed), but score full train & test
    # -------------------------------
    if args.sample and len(train_df) > args.sample:
        train_fit_df = train_df.sample(n=args.sample, random_state=42)
        print(f"[info] fitting on train sample: {len(train_fit_df)} rows (of {len(train_df)})")
    else:
        train_fit_df = train_df
        print(f"[info] fitting on full train: {len(train_fit_df)} rows")

    # Feature matrices
    X_fit   = train_fit_df[numeric].fillna(0).values
    X_train = train_df[numeric].fillna(0).values
    X_test  = test_df[numeric].fillna(0).values

    # Robust scaling (fit on the same data used to fit detectors)
    scaler = RobustScaler()
    X_fit_s   = scaler.fit_transform(X_fit)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Fit detectors on scaled *fit* matrix
    iso, lof = fit_models(X_fit_s, n_neighbors=args.lof_neighbors, sample_limit=args.sample, random_state=42)

    # Score full train & test
    (iso_tr_raw, lof_tr_raw), (iso_te_raw, lof_te_raw) = score_models(iso, lof, X_train_s, X_test_s)

    # Normalize with MinMax fitted on *train* raw scores (per-detector)
    mm_iso = MinMaxScaler()
    mm_lof = MinMaxScaler()

    iso_tr = mm_iso.fit_transform(iso_tr_raw.reshape(-1, 1)).ravel()
    iso_te = mm_iso.transform(iso_te_raw.reshape(-1, 1)).ravel()

    lof_tr = mm_lof.fit_transform(lof_tr_raw.reshape(-1, 1)).ravel()
    lof_te = mm_lof.transform(lof_te_raw.reshape(-1, 1)).ravel()

    # Attach scores
    train_scored = train_df.copy()
    test_scored  = test_df.copy()

    train_scored["iso_score"] = iso_tr
    train_scored["lof_score"] = lof_tr

    test_scored["iso_score"] = iso_te
    test_scored["lof_score"] = lof_te

    # Outputs
    out_train = out_dir / "train_with_scores.csv"
    out_test  = out_dir / "test_with_scores.csv"

    train_scored.to_csv(out_train, index=False)
    test_scored.to_csv(out_test, index=False)

    # Summary
    def _summary(df: pd.DataFrame, name: str):
        return f"{name}: shape={df.shape}, iso∈[{df['iso_score'].min():.3f},{df['iso_score'].max():.3f}], " \
               f"lof∈[{df['lof_score'].min():.3f},{df['lof_score'].max():.3f}]"

    print("[done] wrote:", out_train, out_test)
    print("       " + _summary(train_scored, "train_with_scores"))
    print("       " + _summary(test_scored,  "test_with_scores"))

if __name__ == "__main__":
    main()
