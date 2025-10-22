#ADD HERE: !/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
OUT_DIR="./outputs/preprocessed"
mkdir -p "${OUT_DIR}"

echo "[1/5] Preprocessing CSVs -> parquet..."
python preprocess/merge_and_clean.py --data_dir "${DATA_DIR}" --out_dir "${OUT_DIR}"

echo "[2/5] Running classical detectors (IsolationForest, LOF)..."
python detectors/run_detectors.py --data_dir "${OUT_DIR}" --out_dir "${OUT_DIR}" --sample 5000

echo "[3/5] Training lightweight AE (MLPRegressor) and writing ae_score..."
python repr_learning/ae_mlpreg.py --in_dir "${OUT_DIR}" --out_dir "${OUT_DIR}" --sample 5000

echo "[4/5] Training meta-model and evaluating..."
python meta/meta_train.py --in_dir "${OUT_DIR}" --out_dir "${OUT_DIR}"

echo "[5/5] Starting streaming demo (printing alerts) -- press Ctrl-C to stop"
python demo/streaming_demo.py --parquet "${OUT_DIR}/test_with_ae.parquet" --threshold 0.85 --delay 0.05
