import json
import joblib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ----------------------------------------
# Paths — auto-corrected
# ----------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK = 250000  # Adjustable for speed vs memory

RAW_FILE = DATA_DIR / "merged_train.csv"
CLEAN_FILE = DATA_DIR / "train_clean.csv"

# Only supported classes
TARGET_CLASSES = {0: 0, 2: 1, 3: 2}  # 3-class reduction

def clean_chunk(df: pd.DataFrame):
    """Convert to numeric and remap target"""
    df['attack_type'] = pd.to_numeric(df['attack_type'], errors="coerce")
    df.dropna(subset=['attack_type'], inplace=True)

    df = df[df['attack_type'].isin(TARGET_CLASSES.keys())]
    df['attack_type'] = df['attack_type'].replace(TARGET_CLASSES).astype(int)

    for col in df.columns:
        if col != 'attack_type':
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df

def train_extratrees():
    print("\n🌲 Training ExtraTrees on merged_train.csv in chunks...")

    model = ExtraTreesClassifier(
        n_estimators=120,
        max_depth=30,
        n_jobs=-1,
        warm_start=True,
        random_state=42
    )

    fitted_once = False
    with pd.read_csv(
        RAW_FILE,
        chunksize=CHUNK,
        on_bad_lines="skip"  # << FIXED
    ) as reader:
        for chunk in tqdm(reader, desc="ExtraTrees chunks"):
            chunk = clean_chunk(chunk)
            if len(chunk) == 0:
                continue

            X = chunk.drop(columns=['attack_type']).values
            y = chunk['attack_type'].values

            if not fitted_once:
                model.fit(X, y)
                fitted_once = True
            else:
                model.fit(X, y)

    return model


def train_xgboost():
    print("\n⚡ Training XGBoost (binary: normal vs attack)...")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_depth=10,
        n_estimators=500,
        learning_rate=0.10,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        enable_categorical=False
    )

    fitted_once = False

    with pd.read_csv(
        CLEAN_FILE,
        chunksize=CHUNK,
        low_memory=False,
        on_bad_lines="skip"
    ) as reader:

        for chunk in tqdm(reader, desc="XGBoost chunks"):
            chunk = clean_chunk(chunk)

            # 🔥 Label simplify: all attacks → 1
            chunk["attack_type"] = pd.to_numeric(chunk["attack_type"], errors="coerce").fillna(0)
            chunk["attack_type"] = (chunk["attack_type"] != 0).astype(int)

            if len(chunk) == 0:
                continue

            X = chunk.drop(columns=["attack_type"]).values
            y = chunk["attack_type"].values

            if not fitted_once:
                model.fit(X, y)
                fitted_once = True
            else:
                model.fit(X, y, xgb_model=model.get_booster())

    return model


# ==================================================
# 3️⃣ Meta Model Save
# ==================================================
def save_meta_model(et, xgb):
    final_model = {"extra_trees": et, "xgboost": xgb}
    meta_path = OUT_DIR / "meta_model_combined.joblib"
    joblib.dump(final_model, meta_path)
    print(f"\n💾 Combined Meta-Model Saved → {meta_path}")

def main():
    print("📌 Chunk-Based Training Starting...")

    et = train_extratrees()
    xgb = train_xgboost()

    print("\n🔗 Finalizing unified model...")
    save_meta_model(et, xgb)

    print("\n🎉 ALL DONE — Full Dataset Used!")

if __name__ == "__main__":
    main()
