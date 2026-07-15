"""
Evaluate the NLI-based detector against real ground-truth labels from FEVER.
FEVER already has (context, claim, label) triples, so this gives genuine
precision/recall/F1/AUROC numbers - no generation needed, pure detector evaluation.

Since FEVER has 3 classes (entailment/neutral/contradiction) but our
hallucination detector is fundamentally binary (hallucinated vs not), we
treat 'contradiction' as the positive hallucination class and collapse
entailment+neutral into the negative class.
"""

import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, "src")

from detection.nli_based_detection import NLIDetector
from evaluation.metrics import evaluate_detector, sweep_thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="number of FEVER test examples to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading FEVER test set...")
    df = pd.read_csv("data/benchmarks/fever_test.csv")
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"Evaluating on {len(df)} examples")

    detector = NLIDetector()

    risk_scores = []
    true_labels = []

    for i, row in df.iterrows():
        if i % 25 == 0:
            print(f"  [{i}/{len(df)}]")
        result = detector.check(context=row["context"], claim=row["input_text"])
        risk_scores.append(result["hallucination_risk"])
        # true label: 1 if actually a contradiction (hallucination-equivalent), else 0
        true_labels.append(1 if row["label"] == "contradiction" else 0)

    print("\n=== Results at default threshold (0.5) ===")
    metrics = evaluate_detector(risk_scores, true_labels, threshold=0.5)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Threshold sweep (best F1) ===")
    sweep = sweep_thresholds(risk_scores, true_labels)
    print(f"  Best threshold: {sweep['best']}")

    # save results
    out_df = pd.DataFrame({
        "claim": df["input_text"],
        "context": df["context"],
        "true_label": df["label"],
        "hallucination_risk": risk_scores,
    })
    out_df.to_csv("results/logs/nli_fever_eval.csv", index=False)
    print("\nSaved detailed results to results/logs/nli_fever_eval.csv")


if __name__ == "__main__":
    main()