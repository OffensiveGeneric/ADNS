#!/usr/bin/env python3
"""
preprocess/merge_and_clean.py

Reads CSV files in --data_dir in chunks, canonicalizes common network columns,
produces cleaned train/test CSVs (train_clean.csv, test_clean.csv) in --out_dir,
and writes an aggregated 10-minute windows file aggregated_10m.csv for streaming demo.

Usage:
    python preprocess/merge_and_clean.py \
        --data_dir ./data \
        --out_dir ./outputs/preprocessed \
        --train_pattern training \
        --test_pattern testing \
        --chunksize 200000
"""
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Heuristic column name candidates (lowercase check)
SBYTES_KEYS = ["sbytes", "src_bytes", "bytes_src"]
DBYTES_KEYS = ["dbytes", "dst_bytes", "bytes_dst"]
SPKTS_KEYS  = ["spkts", "src_pkts", "packets_src"]
DPKTS_KEYS  = ["dpkts", "dst_pkts", "packets_dst"]
DUR_KEYS    = ["dur", "duration", "flow_duration", "flow_dur"]
SLOAD_KEYS  = ["sload"]
DLOAD_KEYS  = ["dload"]
SJIT_KEYS   = ["sjit"]
DJIT_KEYS   = ["djit"]
ACKDAT_KEYS = ["ackdat", "ack_data"]

EPS = 1e-9

def _num_series(df, col):
    """Return a float Series aligned to df.index; if col is None, return NaN series."""
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype='float64')
    return pd.to_numeric(df[col], errors='coerce')

def safe_div(num, den):
    den = np.where(den == 0, EPS, den)
    return num / den

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
    import numpy as np
    import pandas as pd

    # --- helpers (local so this is self-contained) ---
    def _find(cols, candidates):
        return find_col(cols, candidates)

    def _num_series_local(df_, col):
        """Float Series aligned to df_.index; NaN series if col is None."""
        if col is None:
            return pd.Series(np.nan, index=df_.index, dtype="float64")
        return pd.to_numeric(df_[col], errors="coerce")

    def _sdiv(a, b):
        return a / np.where(b == 0, EPS, b)

    # --- candidate keys (lowercase matching) ---
    TIME_KEYS = ["time","timestamp","epoch","start_time","ts","date"]
    SRC_KEYS  = ["src","source","srcip","source_ip","ip.src"]
    DST_KEYS  = ["dst","destination","dstip","destination_ip","ip.dst"]
    BYTES_KEYS = ["sbytes","dbytes","bytes","total_bytes","frame_len","ip_len","payload_len"]
    PKT_KEYS   = ["packets","pkts","n_packets","spkts","dpkts"]
    PROTO_KEYS = ["protocol","proto","_ws.col.Protocol","protocol_name"]

    # optional directional/temporal
    SBYTES_KEYS = ["sbytes","src_bytes","bytes_src"]
    DBYTES_KEYS = ["dbytes","dst_bytes","bytes_dst"]
    SPKTS_KEYS  = ["spkts","src_pkts","packets_src"]
    DPKTS_KEYS  = ["dpkts","dst_pkts","packets_dst"]
    DUR_KEYS    = ["dur","duration","flow_duration","flow_dur"]
    SLOAD_KEYS  = ["sload"]
    DLOAD_KEYS  = ["dload"]
    SJIT_KEYS   = ["sjit"]
    DJIT_KEYS   = ["djit"]
    ACKDAT_KEYS = ["ackdat","ack_data"]

    # --- mappings ---
    cols = df.columns.tolist()
    mapping = {
        "time": _find(cols, TIME_KEYS),
        "src": _find(cols, SRC_KEYS),
        "dst": _find(cols, DST_KEYS),
        "bytes": _find(cols, BYTES_KEYS),
        "packets": _find(cols, PKT_KEYS),
        "proto": _find(cols, PROTO_KEYS),
    }
    m_sbytes = _find(cols, SBYTES_KEYS)
    m_dbytes = _find(cols, DBYTES_KEYS)
    m_spkts  = _find(cols, SPKTS_KEYS)
    m_dpkts  = _find(cols, DPKTS_KEYS)
    m_dur    = _find(cols, DUR_KEYS)
    m_sload  = _find(cols, SLOAD_KEYS)
    m_dload  = _find(cols, DLOAD_KEYS)
    m_sjit   = _find(cols, SJIT_KEYS)
    m_djit   = _find(cols, DJIT_KEYS)
    m_ackdat = _find(cols, ACKDAT_KEYS)

    # --- base canonical columns (original behavior) ---
    out = pd.DataFrame(index=df.index)

    # time -> numeric epoch seconds if possible; fallback to parsed datetime
    if mapping["time"] is not None:
        t = pd.to_numeric(df[mapping["time"]], errors="coerce")
        if t.isna().all():
            try:
                t = pd.to_datetime(df[mapping["time"]], errors="coerce").astype("int64") / 1e9
            except Exception:
                t = pd.Series(np.nan, index=df.index)
        out["time"] = t
    else:
        out["time"] = np.nan

    out["src"]      = df[mapping["src"]] if mapping["src"] is not None else np.nan
    out["dst"]      = df[mapping["dst"]] if mapping["dst"] is not None else np.nan
    out["bytes"]    = pd.to_numeric(df[mapping["bytes"]], errors="coerce").fillna(0) if mapping["bytes"] is not None else 0
    out["packets"]  = pd.to_numeric(df[mapping["packets"]], errors="coerce").fillna(1) if mapping["packets"] is not None else 1
    out["protocol"] = df[mapping["proto"]] if mapping["proto"] is not None else np.nan

    # preserve label if present
    if "label" in df.columns:
        out["label"] = df["label"]
    elif "attack_cat" in df.columns:
        out["label"] = df["attack_cat"]
    else:
        out["label"] = np.nan

    # --- engineered features (robust & aligned) ---
    sbytes = _num_series_local(df, m_sbytes)
    dbytes = _num_series_local(df, m_dbytes)
    spkts  = _num_series_local(df, m_spkts)
    dpkts  = _num_series_local(df, m_dpkts)
    dur    = _num_series_local(df, m_dur)
    sload  = _num_series_local(df, m_sload)
    dload  = _num_series_local(df, m_dload)
    sjit   = _num_series_local(df, m_sjit)
    djit   = _num_series_local(df, m_djit)
    ackdat = _num_series_local(df, m_ackdat)

    # totals with fallback to canonical columns
    total_bytes = np.where(sbytes.isna() & dbytes.isna(),
                           out["bytes"].astype(float),
                           sbytes.fillna(0.0) + dbytes.fillna(0.0))
    total_pkts  = np.where(spkts.isna() & dpkts.isna(),
                           out["packets"].astype(float),
                           spkts.fillna(0.0) + dpkts.fillna(0.0))

    out["bytes_per_packet"]   = _sdiv(total_bytes, np.maximum(total_pkts, 1))
    out["packets_per_second"] = np.where(dur.isna(), np.nan, _sdiv(total_pkts, np.maximum(dur, EPS)))
    out["bytes_per_second"]   = np.where(dur.isna(), np.nan, _sdiv(total_bytes, np.maximum(dur, EPS)))

    out["packet_ratio"]   = np.where(spkts.isna() | dpkts.isna(), np.nan, _sdiv(spkts, np.maximum(dpkts, 1)))
    out["byte_ratio"]     = np.where(sbytes.isna() | dbytes.isna(), np.nan, _sdiv(sbytes, np.maximum(dbytes, 1)))
    out["symmetry_index"] = np.where(spkts.isna() | dpkts.isna(), np.nan,
                                     _sdiv(np.minimum(spkts, dpkts), np.maximum(spkts, dpkts)))

    # row-wise means (skipna) avoid empty-slice warnings
    out["load_mean"]   = pd.concat([sload, dload], axis=1).mean(axis=1, skipna=True)
    out["jitter_mean"] = pd.concat([sjit,  djit ], axis=1).mean(axis=1, skipna=True)

    out["ack_per_packet"] = np.where(ackdat.isna(), np.nan, _sdiv(ackdat, np.maximum(total_pkts, 1)))

    # lightweight numeric protocol code (keeps feature numeric for detectors)
    proto_series = out["protocol"].astype(str).str.upper()
    proto_map = {"TCP": 1, "UDP": 2, "ICMP": 3}
    out["proto_code"] = proto_series.map(proto_map).astype("float32")  # others -> NaN

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

    # normalize patterns once (case-insensitive substring)
    train_pattern = (train_pattern or "").lower()
    test_pattern  = (test_pattern  or "").lower()

    for f in files:
        basename = os.path.basename(f).lower()
        role = None
        if train_pattern and train_pattern in basename:
            role = 'train'
        elif test_pattern and test_pattern in basename:
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
                    # not train/test; skip writing but still sampled for aggregation
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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    print(f"Wrote aggregated 10-minute file: {out_path} ({len(agg)} rows)")

def main():
    # --- Compute repo-root-relative defaults (portable for all users) ---
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]  # repo root (one level above preprocess/)
    default_data_dir = repo_root / "data"
    default_out_dir  = repo_root / "outputs" / "preprocessed"

    # --- CLI ---
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None,
                   help=f"Directory with raw CSVs (default: {default_data_dir})")
    p.add_argument("--out_dir", default=None,
                   help=f"Output directory (default: {default_out_dir})")
    p.add_argument("--train_pattern", default="training",
                   help="Substring to identify training files (case-insensitive)")
    p.add_argument("--test_pattern", default="testing",
                   help="Substring to identify test files (case-insensitive)")
    p.add_argument("--chunksize", type=int, default=200000,
                   help="Rows per chunk when reading large CSVs")
    args = p.parse_args()

    # --- Resolve paths robustly (supports ~ and relative inputs) ---
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else default_data_dir
    out_dir  = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paths] data_dir = {data_dir}")
    print(f"[paths] out_dir  = {out_dir}")

    # --- Discover input files ---
    files = sorted(str(p) for p in data_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"[ERROR] No CSV files found in {data_dir}\n"
                         f"        Put your raw CSVs in this folder or pass --data_dir.")

    # --- Output targets ---
    train_out = out_dir / "train_clean.csv"
    test_out  = out_dir / "test_clean.csv"
    agg_out   = out_dir / "aggregated_10m.csv"

    print(f"[info] Files detected: {len(files)}")
    print(f"[info] train_pattern='{args.train_pattern}'  test_pattern='{args.test_pattern}'")

    # --- Process & aggregate ---
    samples = process_files(
        files,
        str(train_out),
        str(test_out),
        chunksize=args.chunksize,
        train_pattern=str(args.train_pattern).lower(),
        test_pattern=str(args.test_pattern).lower(),
    )

    build_aggregated_10m(samples, str(agg_out))

    # --- Post-run summary (existence + sizes) ---
    def _finfo(p: Path):
        return f"{p.name} ({p.stat().st_size:,} bytes)" if p.exists() else f"{p.name} [not created]"

    print("[done] Preprocessing complete. Outputs:")
    print(f"       - {_finfo(train_out)}")
    print(f"       - {_finfo(test_out)}")
    print(f"       - {_finfo(agg_out)}")

    # Helpful hints if nothing got written
    if not train_out.exists() and not test_out.exists():
        print("[warn] No train/test outputs were created. "
              "Check that your filename patterns actually match your CSVs.")

if __name__ == "__main__":
    main()
