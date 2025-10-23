#!/usr/bin/env python3
"""
repr_learning/ae_mlpreg.py

Train a lightweight MLPRegressor autoencoder substitute on train_with_scores.csv
(or fall back to train_clean.csv) and compute per-row reconstruction MSE as ae_score.
Write train_with_ae.csv and test_with_ae.csv.

Usage:
    python repr_learning/ae_mlpreg.py --in_dir ./outputs/preprocessed --out_dir ./outputs/preprocessed --sample 5000
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# ----------------------------
# I/O helpers
# ----------------------------
def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, low_memory=False)

def _find_inputs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    """Select numeric columns present in BOTH train and test, excluding labels/scores/ids."""
    train_num = set(train_df.select_dtypes(include=[np.number]).columns)
    test_num  = set(test_df.select_dtypes(include=[np.number]).columns)
    common    = list(train_num.intersection(test_num))
    exclude   = {"iso_score", "lof_score", "ae_score", "label", "id"}
    input_cols = [c for c in common if c not in exclude]
    return sorted(input_cols)

def _safe_minmax_fit(y: np.ndarray) -> MinMaxScaler:
    """Fit MinMax on vector y; handle constant vectors gracefully."""
    scaler = MinMaxScaler()
    y = y.reshape(-1, 1)
    finite = np.isfinite(y).ravel()
    if not finite.any():
        # extreme fallback: everything is NaN/inf -> make a degenerate scaler on zeros
        scaler.fit(np.zeros((1, 1)))
        return scaler
    y_f = y[finite].reshape(-1, 1)
    if np.nanmax(y_f) - np.nanmin(y_f) < 1e-12:
        # constant vector -> fit on a 2-point tiny spread to avoid division-by-zero
        c = float(np.nanmean(y_f))
        scaler.fit(np.array([[c - 1e-9], [c + 1e-9]]))
    else:
        scaler.fit(y_f)
    return scaler

# ----------------------------
# Main
# ----------------------------
def main():
    # --- repo-root-relative defaults (portable) ---
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]  # one level above repr_learning/
    default_in_dir  = repo_root / "outputs" / "preprocessed"
    default_out_dir = repo_root / "outputs" / "preprocessed"

    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",  default=None, help=f"Input dir (default: {default_in_dir})")
    p.add_argument("--out_dir", default=None, help=f"Output dir (default: {default_out_dir})")
    p.add_argument("--sample", type=int, default=5000, help="Max train rows to fit AE on (for speed)")
    args = p.parse_args()

    in_dir  = Path(args.in_dir).expanduser().resolve() if args.in_dir else default_in_dir
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer *_with_scores, fallback to *_clean
    train_path_scores = in_dir / "train_with_scores.csv"
    test_path_scores  = in_dir / "test_with_scores.csv"
    train_path_clean  = in_dir / "train_clean.csv"
    test_path_clean   = in_dir / "test_clean.csv"

    if train_path_scores.exists() and test_path_scores.exists():
        train = _load(train_path_scores)
        test  = _load(test_path_scores)
        print(f"[info] loaded with-scores: {train_path_scores.name}, {test_path_scores.name}")
    elif train_path_clean.exists() and test_path_clean.exists():
        train = _load(train_path_clean)
        test  = _load(test_path_clean)
        print(f"[info] loaded clean: {train_path_clean.name}, {test_path_clean.name}")
    else:
        raise SystemExit("[ERROR] Missing inputs: expected train_with_scores.csv/test_with_scores.csv "
                         "or fallback train_clean.csv/test_clean.csv in --in_dir")

    # Choose input features robustly
    input_cols = _find_inputs(train, test)
    if not input_cols:
        raise SystemExit("[ERROR] No numeric overlap between train & test (after excluding labels/scores).")

    # Training sample for speed (fit), but we will score FULL train & test
    if args.sample and len(train) > args.sample:
        fit_df = train.sample(n=args.sample, random_state=42)
        print(f"[info] fitting AE on sample: {len(fit_df)} rows (of {len(train)}) with {len(input_cols)} features")
    else:
        fit_df = train
        print(f"[info] fitting AE on full train: {len(fit_df)} rows with {len(input_cols)} features")

    # Matrices
    X_fit   = fit_df[input_cols].fillna(0).values
    X_train = train[input_cols].fillna(0).values
    X_test  = test[input_cols].fillna(0).values

    # Scale features (robust to outliers), fit on same data as AE fit
    rscaler = RobustScaler()
    X_fit_s   = rscaler.fit_transform(X_fit)
    X_train_s = rscaler.transform(X_train)
    X_test_s  = rscaler.transform(X_test)

    # Lightweight MLPRegressor as AE (train to reconstruct scaled inputs)
    hidden = max(8, min(64, X_fit_s.shape[1] // 2))
    ae = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        random_state=42,
        verbose=False,
    )
    print(f"[info] training MLPRegressor AE (hidden={hidden}) on scaled features...")
    ae.fit(X_fit_s, X_fit_s)

    # Reconstruct FULL train/test on scaled space
    recon_train_s = ae.predict(X_train_s)
    recon_test_s  = ae.predict(X_test_s)

    # Per-row reconstruction MSE in scaled space
    mse_train = np.mean((X_train_s - recon_train_s) ** 2, axis=1)
    mse_test  = np.mean((X_test_s  - recon_test_s)  ** 2, axis=1)

    # Normalize MSE to [0,1] using train distribution
    mm = _safe_minmax_fit(mse_train)
    ae_train = mm.transform(mse_train.reshape(-1, 1)).ravel()
    ae_test  = mm.transform(mse_test.reshape(-1, 1)).ravel()

    # Attach & write
    train_out = train.copy()
    test_out  = test.copy()
    train_out["ae_score"] = ae_train
    test_out["ae_score"]  = ae_test

    out_train_path = out_dir / "train_with_ae.csv"
    out_test_path  = out_dir / "test_with_ae.csv"
    train_out.to_csv(out_train_path, index=False)
    test_out.to_csv(out_test_path, index=False)

    # Summary
    def _summary(df: pd.DataFrame, name: str):
        m = df["ae_score"]
        return f"{name}: shape={df.shape}, ae∈[{np.nanmin(m):.3f},{np.nanmax(m):.3f}]"

    print("[done] wrote:", out_train_path, out_test_path)
    print("       " + _summary(train_out, "train_with_ae"))
    print("       " + _summary(test_out,  "test_with_ae"))

if __name__ == "__main__":
    main()
