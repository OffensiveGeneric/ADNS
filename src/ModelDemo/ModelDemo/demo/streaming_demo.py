#!/usr/bin/env python3
"""
demo/streaming_demo.py

Reads a parquet or CSV (test_with_ae.csv or aggregated_10m.csv) and streams rows one-by-one,
computes combined_score if not present (mean of iso_score, lof_score, ae_score), and prints alerts
for rows where combined_score >= --threshold. Can optionally POST alerts to a webhook URL.

Usage:
    python demo/streaming_demo.py --parquet ./outputs/preprocessed/test_with_ae.csv --threshold 0.85 --delay 0.05 --limit 1000
"""
import os
import argparse
import time
import json
import pandas as pd
import requests

def load_rows(path, limit=None):
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    if limit:
        df = df.head(limit)
    return df

def stream(df, threshold=0.85, delay=0.05, webhook=None):
    # Ensure combined_score exists
    if 'combined_score' not in df.columns:
        score_cols = [c for c in ['iso_score','lof_score','ae_score'] if c in df.columns]
        if not score_cols:
            raise SystemExit("Input lacks iso_score/lof_score/ae_score - run detectors & AE first.")
        df['combined_score'] = df[score_cols].mean(axis=1)

    for idx, row in df.iterrows():
        cs = float(row['combined_score'])
        if cs >= threshold:
            alert = {
                "index": int(idx),
                "combined_score": cs,
                "iso_score": float(row.get('iso_score', 0)),
                "lof_score": float(row.get('lof_score', 0)),
                "ae_score": float(row.get('ae_score', 0)),
                "src": row.get('src'),
                "dst": row.get('dst'),
                "proto": row.get('protocol') or row.get('proto') or row.get('service'),
            }
            print("[ALERT] ", json.dumps(alert))
            if webhook:
                try:
                    requests.post(webhook, json=alert, timeout=3)
                except Exception as e:
                    print("[WARN] webhook post failed:", e)
        time.sleep(delay)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="./outputs/preprocessed/test_with_ae.csv", help="path to test_with_ae.csv or aggregated_10m.csv")
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--delay", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--webhook", default=None)
    args = p.parse_args()

    if not os.path.exists(args.parquet):
        raise SystemExit(f"Input file not found: {args.parquet}")
    df = load_rows(args.parquet, limit=args.limit)
    print(f"Streaming {len(df)} rows from {args.parquet} with threshold {args.threshold}")
    stream(df, threshold=args.threshold, delay=args.delay, webhook=args.webhook)

if __name__ == "__main__":
    main()
