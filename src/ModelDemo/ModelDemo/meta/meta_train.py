#!/usr/bin/env python3
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="./outputs/preprocessed")
    p.add_argument("--out_dir", default="./outputs/preprocessed")
    args = p.parse_args()

    train_path = os.path.join(args.in_dir, "train_with_ae.csv")
    test_path  = os.path.join(args.in_dir, "test_with_ae.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise SystemExit("Missing train_with_ae.csv or test_with_ae.csv - run repr_learning first.")

    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)

    if 'label' not in train.columns:
        raise SystemExit("Training set missing 'label' column required for supervised meta-model.")

    # meta features: the three scores plus a small set of engineered numeric features
    base_scores = [c for c in ['iso_score','lof_score','ae_score'] if c in train.columns]
    numeric = train.select_dtypes(include=[np.number]).columns.tolist()
    engineered = [c for c in numeric if c not in base_scores + ['label','id']][:8]

    features = base_scores + engineered
    print("Meta features:", features)

    X_train = train[features].fillna(0).values
    y_train = train['label'].astype(int).values
    X_test = test[features].fillna(0).values
    y_test = test['label'].astype(int).values if 'label' in test.columns else None

    model = None
    if XGBOOST_AVAILABLE:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {"objective":"binary:logistic","eval_metric":"auc","max_depth":5,"eta":0.1}
        bst = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
        model = ("xgboost", bst)
        # predictions
        dtest = xgb.DMatrix(X_test)
        y_prob = bst.predict(dtest)
        y_pred = (y_prob > 0.5).astype(int)
    else:
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        model = ("sklearn_rf", rf)
        y_prob = rf.predict_proba(X_test)[:,1]
        y_pred = rf.predict(X_test)

    metrics = {}
    if y_test is not None:
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        metrics['precision'] = float(p); metrics['recall'] = float(r); metrics['f1'] = float(f)
        try:
            metrics['auc'] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            metrics['auc'] = None
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    else:
        metrics['note'] = "test set missing labels"

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "meta_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # save model
    if model[0] == "xgboost":
        model_path = os.path.join(out_dir, "meta_xgb.model")
        model[1].save_model(model_path)
    else:
        model_path = os.path.join(out_dir, "meta_rf.joblib")
        dump(model[1], model_path)

    print("Saved model to", model_path)
    print("Saved metrics to", metrics_path)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
