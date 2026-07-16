"""
Evaluate the semantic similarity detector against FEVER ground truth.
Same setup as evaluate_nli_on_fever.py, for direct comparison.
"""

import sys
import argparse
import pandas as pd

sys.path.insert(0, "src")

from detection.semantic_similarity import SemanticSimilarityDetector
from evaluation.metrics import evaluate_detector, sweep_thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading FEVER test set...")
    df = pd.read_csv("data/benchmarks/fever_test.csv")
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"Evaluating on {len(df)} examples")

    detector = SemanticSimilarityDetector()

    risk_scores = []
    true_labels = []

    for i, row in df.iterrows():
        if i % 25 == 0:
            print(f"  [{i}/{len(df)}]")
        result = detector.check(context=row["context"], claim=row["input_text"])
        risk_scores.append(result["hallucination_risk"])
        true_labels.append(1 if row["label"] == "contradiction" else 0)

    print("\n=== Results at default threshold (0.5) ===")
    metrics = evaluate_detector(risk_scores, true_labels, threshold=0.5)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Threshold sweep (best F1) ===")
    sweep = sweep_thresholds(risk_scores, true_labels)
    print(f"  Best threshold: {sweep['best']}")

    out_df = pd.DataFrame({
        "claim": df["input_text"],
        "context": df["context"],
        "true_label": df["label"],
        "hallucination_risk": risk_scores,
    })
    out_df.to_csv("results/logs/semantic_similarity_fever_eval.csv", index=False)
    print("\nSaved detailed results to results/logs/semantic_similarity_fever_eval.csv")


if __name__ == "__main__":
    main()
