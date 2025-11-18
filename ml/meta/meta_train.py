#!/usr/bin/env python3
"""
meta_train.py — Implements the new meta pipeline:

  RAW ──► ExtraTrees ┐
                      ├──► Weighted ensemble ─► Evaluate
  CLEAN ─► XGBoost ──┘

- Trains **ExtraTrees** on raw UNSW-NB15 CSVs:
    UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv
- Trains **XGBoost** on preprocessed CSVs:
    train_clean.csv / test_clean.csv
- Calibrates both on a validation split (from their respective training sets)
- Picks decision thresholds by maximizing F1 on each validation split
- Optionally **combines** probabilities (weighted by validation PR-AUC) to produce
  an ensemble score on the test set (requires row alignment by keys or index)

Outputs (under --out_dir):
  et_raw.model, xgb_clean.model
  metrics_raw_et.json, metrics_clean_xgb.json, metrics_ensemble.json
  features_clean.json, encoders_raw.joblib (for raw label encodings),
  preds_raw.csv, preds_clean.csv, preds_ensemble.csv (if --save_preds)

Usage
-----
python meta_train.py \
  --raw_train data/UNSW_NB15_training-set.csv \
  --raw_test  data/UNSW_NB15_testing-set.csv \
  --clean_dir outputs/preprocessed \
  --out_dir   outputs/meta_new \
  --ensemble_keys time src dst     # optional join keys to align rows

Notes
-----
- If --ensemble_keys are not provided, ensemble alignment uses the test row order.
  If lengths differ, the script will skip the ensemble and still save per-model results.
- ExtraTrees is robust to raw features; XGBoost benefits from engineered clean features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

# xgboost is optional but required for the clean track
try:
    import xgboost as xgb
except Exception as e:  # pragma: no cover
    xgb = None

# ---------------------------
# Utilities
# ---------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _coerce_binary_label(y: pd.Series) -> np.ndarray:
    if np.issubdtype(y.dtype, np.number):
        return y.fillna(0).astype(int).values
    s = y.astype(str).str.strip().str.lower()
    normal_tokens = {"benign","normal","clean","legit","non-attack","non_attack","0"}
    return np.where(s.isin(normal_tokens), 0, 1)


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    P, R, F1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "f1": float(F1),
        "precision": float(P),
        "recall": float(R),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": float(thr),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    p, r, thr = precision_recall_curve(y_true, prob)
    f1 = 2 * (p * r) / np.maximum(p + r, 1e-9)
    idx = int(np.nanargmax(f1))
    return float(thr[max(0, idx - 1)]) if len(thr) else 0.5


# ---------------------------
# RAW track (ExtraTrees)
# ---------------------------

def _load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def _encode_categoricals_train(train: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encs: Dict[str, LabelEncoder] = {}
    out = train.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        if col == "label":
            continue
        ser = out[col].astype(str).fillna("NA").replace({"-": "NA"})
        classes = sorted(set(ser.unique()) | {"UNK"})
        le = LabelEncoder().fit(list(classes))
        out[col] = le.transform(ser)
        encs[col] = le
    return out, encs


def _encode_categoricals_apply(df: pd.DataFrame, encs: Dict[str, LabelEncoder]) -> pd.DataFrame:
    out = df.copy()
    for col, le in encs.items():
        if col in out.columns:
            ser = out[col].astype(str).fillna("NA").replace({"-": "NA"})
            known = set(le.classes_)
            ser = ser.where(ser.isin(known), "UNK")
            out[col] = le.transform(ser)
    return out


def run_extra_trees_raw(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 42):
    y_tr = _coerce_binary_label(train_df["label"]) if "label" in train_df.columns else None
    y_te = _coerce_binary_label(test_df["label"]) if "label" in test_df.columns else None

    X_tr = train_df.drop(columns=["label"]) if "label" in train_df.columns else train_df.copy()
    X_te = test_df.drop(columns=["label"]) if "label" in test_df.columns else test_df.copy()

    X_tr_enc, encs = _encode_categoricals_train(X_tr)
    X_te_enc = _encode_categoricals_apply(X_te, encs)

    X_tr_np, X_te_np = X_tr_enc.fillna(0).values, X_te_enc.fillna(0).values

    X0, Xv, y0, yv = train_test_split(X_tr_np, y_tr, test_size=0.15, random_state=seed, stratify=y_tr)

    et = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    ).fit(X0, y0)

    et_cal = CalibratedClassifierCV(et, method="isotonic", cv="prefit").fit(Xv, yv)

    va_prob = et_cal.predict_proba(Xv)[:, 1]
    thr = _best_f1_threshold(yv, va_prob)

    te_prob = et_cal.predict_proba(X_te_np)[:, 1]

    return {
        "model": et,
        "calibrator": et_cal,
        "encoders": encs,
        "y_test": y_te,
        "test_prob": te_prob,
        "val_prob": va_prob,
        "val_y": yv,
        "threshold": thr,
    }


# ---------------------------
# CLEAN track (XGBoost)
# ---------------------------

def _select_numeric_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
    num_train = set(train_df.select_dtypes(include=[np.number]).columns)
    num_test = set(test_df.select_dtypes(include=[np.number]).columns)
    feats = sorted((num_train & num_test) - {"label", "id"})
    if not feats:
        raise SystemExit("[ERROR] No shared numeric features in clean data.")
    return feats


def run_xgb_clean(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 42,
                  max_depth: int = 6, n_estimators: int = 1000, learning_rate: float = 0.3,
                  subsample: float = 0.9, colsample: float = 0.9):
    if xgb is None:
        raise SystemExit("xgboost is required for the clean track. Install with `pip install xgboost`. ")

    y_tr = _coerce_binary_label(train_df["label"]) if "label" in train_df.columns else None
    y_te = _coerce_binary_label(test_df["label"]) if "label" in test_df.columns else None

    feats = _select_numeric_overlap(train_df, test_df)
    X_tr = train_df[feats].fillna(0).values
    X_te = test_df[feats].fillna(0).values

    X0, Xv, y0, yv = train_test_split(X_tr, y_tr, test_size=0.15, random_state=seed, stratify=y_tr)

    # imbalance handling
    pos = max(1, int((y0 == 1).sum()))
    neg = max(1, int((y0 == 0).sum()))
    spw = float(neg / pos)

    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample,
        reg_lambda=1.0,
        reg_alpha=0.0,
        max_delta_step=1,
        tree_method="hist",
        random_state=seed,
        scale_pos_weight=spw,
    )
    clf.fit(X0, y0, eval_set=[(Xv, yv)], verbose=False)

    # calibrate & threshold on validation
    cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit").fit(Xv, yv)
    va_prob = cal.predict_proba(Xv)[:, 1]
    thr = _best_f1_threshold(yv, va_prob)

    te_prob = cal.predict_proba(X_te)[:, 1]

    return {
        "model": clf,
        "calibrator": cal,
        "features": feats,
        "y_test": y_te,
        "test_prob": te_prob,
        "val_prob": va_prob,
        "val_y": yv,
        "threshold": thr,
    }


# ---------------------------
# Ensemble
# ---------------------------

def _weighted_soft_vote(p1_val: np.ndarray, p2_val: np.ndarray, y_val: np.ndarray,
                        p1_test: np.ndarray, p2_test: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, float]]:
    w1 = average_precision_score(y_val, p1_val)
    w2 = average_precision_score(y_val, p2_val)
    denom = max(1e-9, w1 + w2)
    p_val = (w1 * p1_val + w2 * p2_val) / denom
    p_te = (w1 * p1_test + w2 * p2_test) / denom
    thr = _best_f1_threshold(y_val, p_val)
    return p_te, thr, {"w_et": float(w1), "w_xgb": float(w2)}


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Train ExtraTrees on RAW + XGBoost on CLEAN and evaluate ensemble")
    ap.add_argument("--raw_train", default="data/UNSW_NB15_training-set.csv")
    ap.add_argument("--raw_test",  default="data/UNSW_NB15_testing-set.csv")
    ap.add_argument("--clean_dir", default="outputs/preprocessed")
    ap.add_argument("--out_dir",   default="outputs/meta_new")
    ap.add_argument("--ensemble_keys", nargs='*', default=None,
                    help="Optional column names to join RAW and CLEAN test sets for ensemble alignment (e.g., time src dst)")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); _ensure_dir(out_dir)

    # -------- RAW (ET) --------
    raw_tr = _load_raw(Path(args.raw_train))
    raw_te = _load_raw(Path(args.raw_test))
    common = sorted(set(raw_tr.columns) & set(raw_te.columns))
    raw_tr = raw_tr[common]
    raw_te = raw_te[common]

    et_res = run_extra_trees_raw(raw_tr, raw_te, seed=int(args.random_state))

    # -------- CLEAN (XGB) --------
    clean_dir = Path(args.clean_dir)
    tr_clean = pd.read_csv(clean_dir / "train_clean.csv", low_memory=False)
    te_clean = pd.read_csv(clean_dir / "test_clean.csv", low_memory=False)

    xgb_res = run_xgb_clean(tr_clean, te_clean, seed=int(args.random_state))

    # -------- Per-model metrics --------
    metrics_et = _metrics(et_res["y_test"], et_res["test_prob"], et_res["threshold"])
    metrics_xgb = _metrics(xgb_res["y_test"], xgb_res["test_prob"], xgb_res["threshold"])

    # Save models & artifacts
    dump(et_res["model"], out_dir / "et_raw.model")
    dump(et_res["calibrator"], out_dir / "et_calibrator.joblib")
    dump(et_res["encoders"], out_dir / "encoders_raw.joblib")

    xgb_res["model"].save_model(str(out_dir / "xgb_clean.model"))
    dump(xgb_res["calibrator"], out_dir / "xgb_calibrator.joblib")
    (out_dir / "features_clean.json").write_text(json.dumps({"features": xgb_res["features"]}, indent=2))

    (out_dir / "metrics_raw_et.json").write_text(json.dumps(metrics_et, indent=2))
    (out_dir / "metrics_clean_xgb.json").write_text(json.dumps(metrics_xgb, indent=2))

    # -------- Ensemble alignment --------
    ensemble_done = False
    if args.ensemble_keys:
        keys = args.ensemble_keys
        # try to align by keys
        try:
            te_raw_k = raw_te[keys].copy()
            te_clean_k = te_clean[keys].copy()
            te_raw_k["__row"] = np.arange(len(te_raw_k))
            te_clean_k["__row"] = np.arange(len(te_clean_k))
            merged = te_raw_k.merge(te_clean_k, on=keys, how="inner", suffixes=("_r","_c"))
            if not merged.empty:
                idx_r = merged["__row_r"].to_numpy()
                idx_c = merged["__row_c"].to_numpy()
                p_et_test = et_res["test_prob"][idx_r]
                p_xgb_test = xgb_res["test_prob"][idx_c]

                # validation combine on XGB validation labels/probs (same length by construction)
                p_ens_test, thr_ens, weights = _weighted_soft_vote(
                    et_res["val_prob"], xgb_res["val_prob"], xgb_res["val_y"],
                    p_et_test, p_xgb_test
                )
                metrics_ens = _metrics(xgb_res["y_test"][idx_c], p_ens_test, thr_ens)
                metrics_ens.update({"weights": weights})
                (out_dir / "metrics_ensemble.json").write_text(json.dumps(metrics_ens, indent=2))

                # optional predictions
                preds = pd.DataFrame({
                    "p_et": p_et_test,
                    "p_xgb": p_xgb_test,
                    "p_ensemble": p_ens_test,
                })
                preds.to_csv(out_dir / "preds_ensemble.csv", index=False)
                ensemble_done = True
        except Exception:
            ensemble_done = False

    if not ensemble_done:
        # Fallback: if lengths match, align by index
        if len(et_res["test_prob"]) == len(xgb_res["test_prob"]) and np.array_equal(et_res["y_test"], xgb_res["y_test"]):
            p_ens_test, thr_ens, weights = _weighted_soft_vote(
                et_res["val_prob"], xgb_res["val_prob"], xgb_res["val_y"],
                et_res["test_prob"], xgb_res["test_prob"]
            )
            metrics_ens = _metrics(xgb_res["y_test"], p_ens_test, thr_ens)
            metrics_ens.update({"weights": weights})
            (out_dir / "metrics_ensemble.json").write_text(json.dumps(metrics_ens, indent=2))
            ensemble_done = True

    # Save per-model predictions if requested via env flag or by default
    pd.DataFrame({"p_et": et_res["test_prob"]}).to_csv(out_dir / "preds_raw.csv", index=False)
    pd.DataFrame({"p_xgb": xgb_res["test_prob"]}).to_csv(out_dir / "preds_clean.csv", index=False)

    # -------- Pretty print --------
    def _fmt(x):
        return "-" if x is None else (f"{x:.4f}" if isinstance(x, (int, float)) else str(x))

    print("\n=========== Meta Results ===========")
    print("[ExtraTrees on RAW]")
    for k in ["f1","precision","recall","accuracy","roc_auc","pr_auc","threshold"]:
        print(f"{k:>10s}: {_fmt(metrics_et[k])}")
    print("\n[XGBoost on CLEAN]")
    for k in ["f1","precision","recall","accuracy","roc_auc","pr_auc","threshold"]:
        print(f"{k:>10s}: {_fmt(metrics_xgb[k])}")
    if (out_dir / "metrics_ensemble.json").exists():
        m = json.loads((out_dir / "metrics_ensemble.json").read_text())
        print("\n[Ensemble (ET+XGB)]")
        for k in ["f1","precision","recall","accuracy","roc_auc","pr_auc","threshold"]:
            print(f"{k:>10s}: {_fmt(m[k])}")
        if "weights" in m:
            print(f" weights: {m['weights']}")
    else:
        print("\n[Ensemble] skipped (unable to align RAW and CLEAN test rows).")
    print("===================================\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
