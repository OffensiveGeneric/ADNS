#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
meta/meta_train.py

Combine iso_score, lof_score, ae_score and a small set of engineered features to train a meta-model.
Prefer XGBoost if installed; otherwise fallback to RandomForest. Saves model and metrics.

Usage:
    python meta/meta_train.py --in_dir ./outputs/preprocessed --out_dir ./outputs/preprocessed
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# -------------------------------
# Helpers
# -------------------------------
def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, low_memory=False)

def _coerce_binary_label(y: pd.Series) -> np.ndarray:
    """
    Return a binary np.array. If numeric, cast to int {0,1}.
    If string/categorical, map common tokens to {0,1}.
    """
    if np.issubdtype(y.dtype, np.number):
        return y.fillna(0).astype(int).values

    # string mapping heuristics
    s = y.astype(str).str.strip().str.lower()
    map1 = {
        "benign": 0, "normal": 0, "clean": 0, "legit": 0, "non-attack": 0, "non_attack": 0, "0": 0,
        "attack": 1, "anomaly": 1, "malicious": 1, "malware": 1, "1": 1
    }
    mapped = s.map(map1)
    # unseen tokens -> try fallback: anything not equal to known "normal" becomes 1
    if mapped.isna().any():
        mapped = np.where(s.isin(["benign","normal","clean","legit","non-attack","non_attack","0"]), 0, 1)
    return mapped.astype(int)

def _select_features(train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: np.ndarray, k_engineered: int = 8):
    """
    Build feature list:
      - Base scores present: [iso_score, lof_score, ae_score]
      - Up to k_engineered additional numeric features with highest |corr| to y_train.
    Only keep engineered features that exist in BOTH train and test. Returns an ordered list.
    """
    base_order = ["iso_score", "lof_score", "ae_score"]
    base_scores = [c for c in base_order if c in train_df.columns and c in test_df.columns]
    if not base_scores:
        raise SystemExit("[ERROR] None of {iso_score, lof_score, ae_score} present in inputs. Run previous pipeline steps.")

    # candidate engineered features: numeric, excluding base scores, label/id
    numeric_train = set(train_df.select_dtypes(include=[np.number]).columns)
    numeric_test  = set(test_df.select_dtypes(include=[np.number]).columns)
    common_numeric = numeric_train.intersection(numeric_test)

    exclude = set(base_scores) | {"label", "id"}
    engineered_candidates = sorted([c for c in common_numeric if c not in exclude])

    if not engineered_candidates:
        return base_scores  # only scores available

    # rank engineered by absolute Pearson correlation with label
    # (coerce to float, ignore NaNs)
    corrs = []
    y = pd.Series(y_train)
    for c in engineered_candidates:
        v = pd.to_numeric(train_df[c], errors="coerce")
        if v.isna().all():
            continue
        # align/dropna jointly
        df_pair = pd.concat([v, y], axis=1).dropna()
        if len(df_pair) < 2:
            continue
        try:
            corr = float(df_pair.iloc[:,0].corr(df_pair.iloc[:,1]))
            corrs.append((c, abs(corr)))
        except Exception:
            continue

    corrs.sort(key=lambda x: x[1], reverse=True)
    top_eng = [c for c,_ in corrs[:k_engineered]]

    return base_scores + top_eng

def _ensure_test_columns(test_df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """
    Ensure test has all feature columns; if not, create with zeros and warn.
    """
    missing = [c for c in feature_list if c not in test_df.columns]
    if missing:
        print(f"[warn] test is missing {len(missing)} feature(s): {missing} — creating zero-filled columns.")
        for c in missing:
            test_df[c] = 0.0
    return test_df

def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -------------------------------
# Main
# -------------------------------
def main():
    # Repo-root-relative defaults (portable)
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]  # one level above meta/
    default_in_dir  = repo_root / "outputs" / "preprocessed"
    default_out_dir = repo_root / "outputs" / "preprocessed"

    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",  default=None, help=f"Input dir (default: {default_in_dir})")
    p.add_argument("--out_dir", default=None, help=f"Output dir (default: {default_out_dir})")
    p.add_argument("--xgb_rounds", type=int, default=200, help="Num boosting rounds for XGBoost")
    p.add_argument("--xgb_depth", type=int, default=5, help="Max depth for XGBoost trees")
    args = p.parse_args()

    in_dir  = Path(args.in_dir).expanduser().resolve() if args.in_dir else default_in_dir
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = in_dir / "train_with_ae.csv"
    test_path  = in_dir / "test_with_ae.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit("[ERROR] Missing train_with_ae.csv or test_with_ae.csv - run repr_learning first.")

    print(f"[paths] in_dir = {in_dir}")
    print(f"[paths] out_dir= {out_dir}")
    print(f"[info] reading: {train_path.name}, {test_path.name}")

    train = _load_csv(train_path)
    test  = _load_csv(test_path)

    if "label" not in train.columns:
        raise SystemExit("[ERROR] Training set missing 'label' (required for supervised meta-model).")

    # Coerce label to binary ints
    y_train = _coerce_binary_label(train["label"])
    y_test  = _coerce_binary_label(test["label"]) if "label" in test.columns else None

    # Feature selection
    features = _select_features(train, test, y_train, k_engineered=8)
    print(f"[info] meta features ({len(features)}): {features}")

    # Ensure test has all chosen columns
    test = _ensure_test_columns(test, features)

    # Build matrices
    X_train = train[features].fillna(0).values
    X_test  = test[features].fillna(0).values

    model_name = None
    y_prob = None
    y_pred = None

    if XGBOOST_AVAILABLE:
        # class imbalance handling: scale_pos_weight = (neg / pos)
        pos = max(1, int((y_train == 1).sum()))
        neg = max(1, int((y_train == 0).sum()))
        spw = float(neg / pos)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": int(args.xgb_depth),
            "eta": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "lambda": 1.0,
            "alpha": 0.0,
            "seed": 42,
            "scale_pos_weight": spw,
        }

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=[str(i) for i in range(X_train.shape[1])])

        # If test has labels, use for early stopping; otherwise just train
        if y_test is not None:
            dvalid = xgb.DMatrix(X_test, label=y_test, feature_names=[str(i) for i in range(X_test.shape[1])])
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=int(args.xgb_rounds),
                evals=[(dtrain, "train"), (dvalid, "valid")],
                early_stopping_rounds=30,
                verbose_eval=False,
            )
            y_prob = bst.predict(dvalid)
        else:
            bst = xgb.train(params, dtrain, num_boost_round=int(args.xgb_rounds), verbose_eval=False)
            dtest = xgb.DMatrix(X_test, feature_names=[str(i) for i in range(X_test.shape[1])])
            y_prob = bst.predict(dtest)

        y_pred = (y_prob > 0.5).astype(int)
        model_name = "xgboost"
        model_obj = bst

    else:
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        )
        rf.fit(X_train, y_train)
        y_prob = rf.predict_proba(X_test)[:, 1]
        y_pred = rf.predict(X_test)
        model_name = "sklearn_rf"
        model_obj = rf

    # Metrics
    metrics = {}
    if y_test is not None:
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        metrics["precision"] = float(p)
        metrics["recall"]    = float(r)
        metrics["f1"]        = float(f)
        try:
            metrics["auc"] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            metrics["auc"] = None
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        metrics["positives_train"]  = int((y_train == 1).sum())
        metrics["negatives_train"]  = int((y_train == 0).sum())
        metrics["positives_test"]   = int((y_test  == 1).sum())
        metrics["negatives_test"]   = int((y_test  == 0).sum())
    else:
        metrics["note"] = "test set missing labels"
        metrics["positives_train"]  = int((y_train == 1).sum())
        metrics["negatives_train"]  = int((y_train == 0).sum())

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "meta_metrics.json"
    _save_json(metrics, metrics_path)

    features_path = out_dir / "meta_features.json"
    _save_json({"features": features}, features_path)

    if model_name == "xgboost":
        model_path = out_dir / "meta_xgb.model"
        model_obj.save_model(str(model_path))
    else:
        model_path = out_dir / "meta_rf.joblib"
        dump(model_obj, model_path)

    # Final prints
    print(f"[done] Saved model to   {model_path}")
    print(f"[done] Saved metrics to {metrics_path}")
    print(f"[done] Saved features to {features_path}")
    print(f"[summary] features ({len(features)}): {features}")
    if "auc" in metrics:
        print(f"[summary] AUC={metrics.get('auc')}, F1={metrics.get('f1')}, P={metrics.get('precision')}, R={metrics.get('recall')}")

if __name__ == "__main__":
    main()
