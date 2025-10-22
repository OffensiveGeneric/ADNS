#!/usr/bin/env python3
"""
repr_learning/ae_mlpreg.py

Train a lightweight MLPRegressor autoencoder substitute on train_with_scores.csv (or train_clean.csv)
and compute per-row reconstruction MSE as ae_score. Write train_with_ae.csv and test_with_ae.csv.

Usage:
    python repr_learning/ae_mlpreg.py --in_dir ./outputs/preprocessed --out_dir ./outputs/preprocessed --sample 5000
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="./outputs/preprocessed")
    p.add_argument("--out_dir", default="./outputs/preprocessed")
    p.add_argument("--sample", type=int, default=5000)
    args = p.parse_args()

    train_path = os.path.join(args.in_dir, "train_with_scores.csv")
    test_path  = os.path.join(args.in_dir, "test_with_scores.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise SystemExit("Missing train_with_scores.csv or test_with_scores.csv - run detectors first.")

    train = pd.read_csv(train_path, low_memory=False)
    test  = pd.read_csv(test_path, low_memory=False)

    # choose input cols: numeric excluding existing score columns and label
    all_numeric = train.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set(["iso_score","lof_score","ae_score","label","id"])
    input_cols = [c for c in all_numeric if c not in exclude]
    if not input_cols:
        raise SystemExit("No numeric columns available for AE input.")

    # sample for speed
    if args.sample and len(train) > args.sample:
        train_sample = train.sample(n=args.sample, random_state=42)
    else:
        train_sample = train

    X_train = train_sample[input_cols].fillna(0).values
    X_test = test[input_cols].fillna(0).values

    # train lightweight MLPRegressor
    hidden = max(8, min(64, X_train.shape[1] // 2))
    ae = MLPRegressor(hidden_layer_sizes=(hidden,), activation='relu', solver='adam',
                     max_iter=200, early_stopping=True, random_state=42)
    print("Training MLPRegressor autoencoder (hidden size = {})...".format(hidden))
    ae.fit(X_train, X_train)

    # get recon errors for full datasets (predict on all)
    try:
        recon_train = ae.predict(train[input_cols].fillna(0).values)
        mse_train = np.mean((train[input_cols].fillna(0).values - recon_train)**2, axis=1)
    except Exception:
        # fallback: compute on sampled subset only
        recon_train = ae.predict(X_train)
        mse_train = np.full(len(train), np.nan)
        mse_train[train_sample.index] = np.mean((X_train - recon_train)**2, axis=1)

    recon_test = ae.predict(X_test)
    mse_test = np.mean((X_test - recon_test)**2, axis=1)

    mm = MinMaxScaler()
    # fit mm on train's present mse values (drop NA)
    valid_train_mse = np.nan_to_num(mse_train, nan=np.nanmean(mse_train))
    mm.fit(valid_train_mse.reshape(-1,1))
    train['ae_score'] = mm.transform(valid_train_mse.reshape(-1,1)).ravel()
    test['ae_score'] = mm.transform(mse_test.reshape(-1,1)).ravel()

    out_train = os.path.join(args.out_dir, "train_with_ae.csv")
    out_test  = os.path.join(args.out_dir, "test_with_ae.csv")
    train.to_csv(out_train, index=False)
    test.to_csv(out_test, index=False)
    print("Wrote:", out_train, out_test)

if __name__ == "__main__":
    main()
