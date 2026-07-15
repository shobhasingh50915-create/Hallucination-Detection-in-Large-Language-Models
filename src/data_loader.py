"""
Step 3: Clean processed data - remove duplicates/nulls, build train/val/test splits.
"""

import pandas as pd
import os

PROCESSED_DIR = "data/processed"
BENCHMARK_DIR = "data/benchmarks"


def clean_and_split(filename, dedup_col="input_text", drop_null_cols=None):
    path = os.path.join(PROCESSED_DIR, filename)
    df = pd.read_csv(path)
    before = len(df)

    df = df.drop_duplicates(subset=[dedup_col])
    if drop_null_cols:
        df = df.dropna(subset=drop_null_cols)

    after = len(df)
    print(f"{filename}: {before} -> {after} rows after cleaning ({before - after} removed)")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train, val, test = df[:train_end], df[train_end:val_end], df[val_end:]

    base = filename.replace(".csv", "")
    train.to_csv(os.path.join(BENCHMARK_DIR, f"{base}_train.csv"), index=False)
    val.to_csv(os.path.join(BENCHMARK_DIR, f"{base}_val.csv"), index=False)
    test.to_csv(os.path.join(BENCHMARK_DIR, f"{base}_test.csv"), index=False)
    print(f"  -> train: {len(train)}, val: {len(val)}, test: {len(test)}")


if __name__ == "__main__":
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    clean_and_split("truthfulqa.csv", dedup_col="input_text", drop_null_cols=["input_text", "reference_answer"])
    clean_and_split("halueval.csv", dedup_col="input_text", drop_null_cols=["input_text", "reference_answer"])
    clean_and_split("fever.csv", dedup_col="input_text", drop_null_cols=["input_text", "context", "label"])
    print("\nDone! Clean splits saved in data/benchmarks/")