#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"X:\ADNSvadzim\ADNS\src\ModelDemo\ModelDemo\data")
CHUNKSIZE = 250_000  # adjust if needed

OUT_TRAIN = DATA_DIR / "train_clean.csv"
OUT_TEST  = DATA_DIR / "test_clean.csv"

DROP_COLS = ["label", "attack_cat", "dataset_source", "type"]

def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # Ensure attack_type column exists
    if "attack_type" not in chunk.columns:
        chunk["attack_type"] = 99  # unknown fallback

    # Replace invalid values
    chunk = chunk.replace(["-", "?", ""], np.nan).fillna(0)

    # Remove unused columns if present
    for c in DROP_COLS:
        if c in chunk.columns:
            chunk = chunk.drop(columns=c)

    return chunk


def stream_clean(in_csv, out_csv):
    first = True
    for chunk in pd.read_csv(
        in_csv,
        chunksize=CHUNKSIZE,
        engine="python",
        on_bad_lines="skip"
    ):
        chunk = process_chunk(chunk)
        chunk.to_csv(out_csv, mode="a", index=False, header=first)
        first = False


def main():
    print("🚀 Streaming Clean Preprocessing...")
    print(f"📌 Using data folder: {DATA_DIR}")

    # Delete old outputs if exist
    if OUT_TRAIN.exists(): OUT_TRAIN.unlink()
    if OUT_TEST.exists(): OUT_TEST.unlink()

    print("📥 Cleaning Train Set...")
    stream_clean(DATA_DIR / "merged_train.csv", OUT_TRAIN)

    print("📥 Cleaning Test Set...")
    stream_clean(DATA_DIR / "merged_test.csv", OUT_TEST)

    print("✔ Completed successfully!")
    print(f"💾 Saved train_clean.csv → {OUT_TRAIN}")
    print(f"💾 Saved test_clean.csv → {OUT_TEST}")


if __name__ == "__main__":
    main()
