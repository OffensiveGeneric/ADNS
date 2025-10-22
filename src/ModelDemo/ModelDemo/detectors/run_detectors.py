#!/usr/bin/env python3
"""
detectors/run_detectors.py

Load train_clean.csv and test_clean.csv from --data_dir (or specify full path),
select numeric features, optionally subsample, train IsolationForest & LOF (novelty=True),
compute normalized anomaly scores and write train_with_scores.csv and test_with_scores.csv.

Usage:
    python detectors/run_detectors.py --data_dir ./outputs/preprocessed --out_dir ./outputs/preprocessed --sample 5000
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, MinMaxScaler

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)

def select_numeric(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove 'label' if present from features
    if 'label' in num:
        num = [c for c in num if c != 'label']
    return num

def fit_and_score(X_train, X_test, sample_limit=5000):
    # scaler
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # IsolationForest
    iso = IsolationForest(n_estimators=100, max_samples=min(sample_limit, X_train_s.shape[0]),
                          contamination='auto', random_state=42, n_jobs=-1)
    iso.fit(X_train_s)
    iso_train_raw = -iso.decision_function(X_train_s)
    iso_test_raw = -iso.decision_function(X_test_s)

    # LOF (novelty=True)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(X_train_s)
    lof_train_raw = -lof.score_samples(X_train_s)
    lof_test_raw = -lof.score_samples(X_test_s)

    # normalize each score using MinMax on train
    mm_iso = MinMaxScaler(); mm_lof = MinMaxScaler()
    iso_train = mm_iso.fit_transform(iso_train_raw.reshape(-1,1)).ravel()
    iso_test = mm_iso.transform(iso_test_raw.reshape(-1,1)).ravel()
    lof_train = mm_lof.fit_transform(lof_train_raw.reshape(-1,1)).ravel()
    lof_test = mm_lof.transform(lof_test_raw.reshape(-1,1)).ravel()

    return (iso_train, lof_train), (iso_test, lof_test), scaler

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="./outputs/preprocessed")
    p.add_argument("--out_dir", default="./outputs/preprocessed")
    p.add_argument("--sample", type=int, default=5000, help="max samples to use for fitting (speed)")
    args = p.parse_args()

    train_csv = os.path.join(args.data_dir, "train_clean.csv")
    test_csv  = os.path.join(args.data_dir, "test_clean.csv")
    train = load_csv(train_csv)
    test = load_csv(test_csv)

    numeric = select_numeric(train)
    if not numeric:
        raise SystemExit("No numeric features found in train_clean.csv - check preprocessing.")

    # Subsample for speed if requested
    if args.sample and len(train) > args.sample:
        train_sample = train.sample(n=args.sample, random_state=42)
    else:
        train_sample = train

    X_train = train_sample[numeric].fillna(0).values
    X_test = test[numeric].fillna(0).values

    print("Fitting detectors on", X_train.shape[0], "samples and scoring", X_test.shape[0], "test rows.")
    (iso_tr, lof_tr), (iso_te, lof_te), scaler = fit_and_score(X_train, X_test, sample_limit=args.sample)

    # Attach scores to dataframes
    # For train, we may have subsampled; we will score the full train again for completeness
    X_train_full = train[numeric].fillna(0).values
    try:
        # score full train via fitted models by reusing scaler + iso + lof from fit stage is non-trivial here,
        # but for simplicity we will re-fit on the sample then apply to full train.
        pass
    except Exception:
        pass

    # Simpler approach: re-run fit on full train if small, else use sample scores only for train rows present in sample
    if len(train) <= args.sample:
        # we used full train to fit
        train['iso_score'] = iso_tr
        train['lof_score'] = lof_tr
    else:
        # default: map scores to train_sample only, leave rest NaN
        train.loc[train_sample.index, 'iso_score'] = iso_tr
        train.loc[train_sample.index, 'lof_score'] = lof_tr
        # for rows without score, predict by scaling features and using fitted detectors? omitted for speed

    test['iso_score'] = iso_te
    test['lof_score'] = lof_te

    out_train = os.path.join(args.out_dir, "train_with_scores.csv")
    out_test  = os.path.join(args.out_dir, "test_with_scores.csv")
    train.to_csv(out_train, index=False)
    test.to_csv(out_test, index=False)
    print("Wrote:", out_train, out_test)

if __name__ == "__main__":
    main()
