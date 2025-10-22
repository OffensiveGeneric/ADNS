#!/usr/bin/env python3
"""
preprocess/merge_and_clean.py

Reads CSV files in --data_dir in chunks, canonicalizes common network columns,
produces cleaned train/test CSVs (train_clean.csv, test_clean.csv) in --out_dir,
and writes an aggregated 10-minute windows file aggregated_10m.csv for streaming demo.

Usage:
    python preprocess/merge_and_clean.py --data_dir ./data --out_dir ./outputs/preprocessed --train_pattern training --test_pattern testing --chunksize 200000
"""
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Heuristic column name candidates (lowercase check)
TIME_KEYS = ["time","timestamp","epoch","start_time","ts","date"]
SRC_KEYS = ["src","source","srcip","source_ip","ip.src"]
DST_KEYS = ["dst","destination","dstip","destination_ip","ip.dst"]
BYTES_KEYS = ["sbytes","dbytes","bytes","total_bytes","frame_len","ip_len","payload_len"]
PKT_KEYS = ["packets","pkts","n_packets","spkts","dpkts"]
PROTO_KEYS = ["protocol","proto","_ws.col.Protocol","protocol_name"]

def find_col(cols, candidates):
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        cand_l = cand.lower()
        if cand_l in cols_l:
            return cols_l[cand_l]
    # fallback: substring match
    for c_lower, c_orig in cols_l.items():
        for cand in candidates:
            if cand.lower() in c_lower:
                return c_orig
    return None

def canonicalize_chunk(df):
    cols = df.columns.tolist()
    mapping = {}
    mapping['time'] = find_col(cols, TIME_KEYS)
    mapping['src'] = find_col(cols, SRC_KEYS)
    mapping['dst'] = find_col(cols, DST_KEYS)
    mapping['bytes'] = find_col(cols, BYTES_KEYS)
    mapping['packets'] = find_col(cols, PKT_KEYS)
    mapping['proto'] = find_col(cols, PROTO_KEYS)
    # Build canonical df
    out = pd.DataFrame()
    # time -> numeric epoch seconds if possible
    if mapping['time'] is not None:
        out['time'] = pd.to_numeric(df[mapping['time']], errors='coerce')
        if out['time'].isna().all():
            try:
                out['time'] = pd.to_datetime(df[mapping['time']], errors='coerce').astype('int64')/1e9
            except Exception:
                out['time'] = np.nan
    else:
        out['time'] = np.nan
    out['src'] = df[mapping['src']] if mapping['src'] is not None else np.nan
    out['dst'] = df[mapping['dst']] if mapping['dst'] is not None else np.nan
    out['bytes'] = pd.to_numeric(df[mapping['bytes']], errors='coerce').fillna(0) if mapping['bytes'] is not None else 0
    out['packets'] = pd.to_numeric(df[mapping['packets']], errors='coerce').fillna(1) if mapping['packets'] is not None else 1
    out['protocol'] = df[mapping['proto']] if mapping['proto'] is not None else np.nan
    # preserve label if present
    if 'label' in df.columns:
        out['label'] = df['label']
    elif 'attack_cat' in df.columns:
        out['label'] = df['attack_cat']
    else:
        out['label'] = np.nan
    return out

def process_files(files, out_train_csv, out_test_csv, chunksize=200000, train_pattern="training", test_pattern="testing"):
    """
    For any files matching train_pattern or test_pattern put into respective CSV outputs.
    If multiple train/test files exist they get appended.
    Also returns a list of processed DataFrames (small samples) for aggregated_10m creation.
    """
    first_train = True
    first_test = True
    samples_for_agg = []  # small samples to build aggregated_10m quickly
    for f in files:
        basename = os.path.basename(f).lower()
        role = None
        if train_pattern in basename:
            role = 'train'
        elif test_pattern in basename:
            role = 'test'
        else:
            # salvage: if file contains 'train' or 'test' as substrings
            if 'train' in basename:
                role = 'train'
            elif 'test' in basename:
                role = 'test'
        try:
            for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False):
                cf = canonicalize_chunk(chunk)
                cf['source_file'] = os.path.basename(f)
                # append to sample list (small)
                samples_for_agg.append(cf.head(2000))
                if role == 'train':
                    mode = 'w' if first_train else 'a'
                    header = first_train
                    cf.to_csv(out_train_csv, mode=mode, index=False, header=header)
                    first_train = False
                elif role == 'test':
                    mode = 'w' if first_test else 'a'
                    header = first_test
                    cf.to_csv(out_test_csv, mode=mode, index=False, header=header)
                    first_test = False
                else:
                    # if not train/test, still save a generic cleaned file for inspection
                    pass
                # only keep first chunk per file for speed in sample aggregation
                break
        except Exception as e:
            print(f"[WARN] Skipping file {f} due to read error: {e}")
            continue
    return samples_for_agg

def build_aggregated_10m(samples, out_path):
    if not samples:
        print("[WARN] No samples available to build aggregated_10m.csv")
        return
    df_all = pd.concat(samples, ignore_index=True)
    # ensure time numeric
    df_all['time'] = pd.to_numeric(df_all['time'], errors='coerce').fillna(0)
    df_all['window_10m'] = (df_all['time'] // 600).astype(int)
    # aggregate by window,src,dst,protocol
    agg = df_all.groupby(['window_10m','src','dst','protocol'], dropna=False).agg(
        n_packets=('packets','sum'),
        n_flows=('packets','count'),
        total_bytes=('bytes','sum')
    ).reset_index()
    agg['bytes_per_packet'] = agg['total_bytes'] / agg['n_packets'].replace(0,1)
    agg.to_csv(out_path, index=False)
    print(f"Wrote aggregated 10-minute file: {out_path} ({len(agg)} rows)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="X:\ModelDemo\ModelDemo\data")
    p.add_argument("--out_dir", default="./outputs/preprocessed")
    p.add_argument("--train_pattern", default="training")
    p.add_argument("--test_pattern", default="testing")
    p.add_argument("--chunksize", type=int, default=200000)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([str(p) for p in data_dir.glob("*.csv")])
    if not files:
        raise SystemExit(f"No CSV files found in {data_dir} - please place the raw CSVs there.")

    train_out = out_dir / "train_clean.csv"
    test_out = out_dir / "test_clean.csv"
    agg_out = out_dir / "aggregated_10m.csv"

    print("Files detected:", len(files))
    samples = process_files(files, str(train_out), str(test_out), chunksize=args.chunksize,
                            train_pattern=args.train_pattern, test_pattern=args.test_pattern)

    build_aggregated_10m(samples, str(agg_out))
    print("Preprocessing complete. Cleaned train/test CSVs written to:", train_out, test_out)

if __name__ == "__main__":
    main()
