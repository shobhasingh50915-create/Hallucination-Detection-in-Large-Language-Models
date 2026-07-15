"""
Ensemble evaluation: NLI detector + Self-Consistency detector on HaluEval.

HaluEval's 'label' column is unpopulated in this dataset build, so ground
truth is derived via a standard QA-hallucination proxy: token-overlap F1
between the model's generated answer and HaluEval's reference_answer.
Low overlap (< threshold) = treated as hallucination. This proxy is
independent of both detectors (NLI uses context+generated-answer entailment,
self-consistency uses embedding agreement across samples), so it's a fair
ground truth to evaluate both against.
"""

import sys
import argparse
import re
import pandas as pd
import numpy as np

sys.path.insert(0, "src")

from detection.nli_based_detection import NLIDetector
from detection.self_consistency import SelfConsistencyDetector
from models.llm_wrapper import get_llm_wrapper
from evaluation.metrics import evaluate_detector, sweep_thresholds


def token_f1(a, b):
    """Simple token-overlap F1 between two strings (SQuAD-style proxy)."""
    a_toks = re.findall(r"\w+", a.lower())
    b_toks = re.findall(r"\w+", b.lower())
    if not a_toks or not b_toks:
        return 0.0
    common = set(a_toks) & set(b_toks)
    if not common:
        return 0.0
    precision = len(common) / len(set(a_toks))
    recall = len(common) / len(set(b_toks))
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of HaluEval examples")
    parser.add_argument("--n_samples", type=int, default=3, help="generations per example for self-consistency")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--f1_threshold", type=float, default=0.3, help="below this = treated as hallucination")
    args = parser.parse_args()

    print("Loading HaluEval test set...")
    df = pd.read_csv("data/benchmarks/halueval_test.csv")
    df = df.dropna(subset=["input_text", "reference_answer", "context"])
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"Evaluating on {len(df)} examples, {args.n_samples} generations each")

    print("Loading LLM wrapper (OpenAI gpt-4o-mini)...")
    llm = get_llm_wrapper("groq", "llama-3.1-8b-instant")

    print("Loading detectors...")
    nli_detector = NLIDetector()
    sc_detector = SelfConsistencyDetector()

    rows = []
    for i, row in df.iterrows():
        print(f"  [{i+1}/{len(df)}] {row['input_text'][:60]}...")
        prompt = f"Context: {row['context']}\n\nQuestion: {row['input_text']}\nAnswer concisely."

        # generate multiple responses for self-consistency
        sc_result = sc_detector.score(llm, prompt, n_samples=args.n_samples, max_tokens=64)
        if sc_result["hallucination_risk"] is None:
            continue

        # use the first generated response as "the" answer for NLI + ground truth
        main_answer = sc_result["responses"][0]

        nli_result = nli_detector.check(context=row["context"], claim=main_answer)

        # ground truth proxy: does generated answer overlap with reference answer?
        f1 = token_f1(main_answer, row["reference_answer"])
        true_label = 1 if f1 < args.f1_threshold else 0

        rows.append({
            "question": row["input_text"],
            "generated_answer": main_answer,
            "reference_answer": row["reference_answer"],
            "answer_f1": f1,
            "true_label": true_label,
            "nli_risk": nli_result["hallucination_risk"],
            "sc_risk": sc_result["hallucination_risk"],
            "ensemble_risk": (nli_result["hallucination_risk"] + sc_result["hallucination_risk"]) / 2,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv("results/logs/ensemble_halueval_eval.csv", index=False)

    for method in ["nli_risk", "sc_risk", "ensemble_risk"]:
        print(f"\n=== {method} ===")
        metrics = evaluate_detector(out_df[method].tolist(), out_df["true_label"].tolist(), threshold=0.5)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        sweep = sweep_thresholds(out_df[method].tolist(), out_df["true_label"].tolist())
        print(f"  best: {sweep['best']}")

    print("\nSaved detailed results to results/logs/ensemble_halueval_eval.csv")


if __name__ == "__main__":
    main()
