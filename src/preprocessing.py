"""
Step 2: Clean and normalize raw datasets into a unified schema.
Unified schema: id, source_dataset, input_text, reference_answer, context, label
"""

from datasets import load_from_disk
import pandas as pd
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def process_truthfulqa():
    print("Processing TruthfulQA...")
    ds = load_from_disk(os.path.join(RAW_DIR, "truthfulqa"))["validation"]
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "id": f"truthfulqa_{i}",
            "source_dataset": "truthfulqa",
            "input_text": ex["question"],
            "reference_answer": ex["best_answer"],
            "context": None,
            "label": None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROCESSED_DIR, "truthfulqa.csv"), index=False)
    print(f"Saved {len(df)} rows -> data/processed/truthfulqa.csv")


def process_halueval():
    print("Processing HaluEval...")
    ds = load_from_disk(os.path.join(RAW_DIR, "halueval_qa"))["data"]
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "id": f"halueval_{i}",
            "source_dataset": "halueval",
            "input_text": ex["question"],
            "reference_answer": ex["right_answer"],
            "context": ex.get("knowledge", None),
            "label": None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROCESSED_DIR, "halueval.csv"), index=False)
    print(f"Saved {len(df)} rows -> data/processed/halueval.csv")


def process_fever():
    print("Processing FEVER (nli_fever)...")
    ds = load_from_disk(os.path.join(RAW_DIR, "fever"))["train"]
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "id": f"fever_{i}",
            "source_dataset": "fever",
            "input_text": ex["hypothesis"],
            "reference_answer": None,
            "context": ex["premise"],
            "label": label_map[ex["label"]],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PROCESSED_DIR, "fever.csv"), index=False)
    print(f"Saved {len(df)} rows -> data/processed/fever.csv")


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    process_truthfulqa()
    process_halueval()
    process_fever()
    print("\nDone! All datasets normalized into data/processed/")