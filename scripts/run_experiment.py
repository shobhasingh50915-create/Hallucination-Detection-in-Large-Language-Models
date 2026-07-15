"""
End-to-end experiment: run self-consistency detection on HaluEval test set,
evaluate against ground truth, save results.

Designed to run on Colab with a quantized model (Mistral-7B or Llama-3-8B).
Usage on Colab:
    !python scripts/run_experiment.py --model mistralai/Mistral-7B-Instruct-v0.2 --n 100
"""

import sys
import os
import argparse
import json
import pandas as pd

sys.path.insert(0, "src")

from models.llm_wrapper import get_llm_wrapper
from detection.self_consistency import SelfConsistencyDetector
from evaluation.metrics import evaluate_detector, sweep_thresholds


def build_halueval_pairs(df, n):
    """
    HaluEval gives each question a right_answer and a hallucinated_answer.
    To evaluate a detector, we need (prompt, is_hallucinated) pairs, so we
    treat each question twice: once framed toward the right answer (label=0)
    and once toward the hallucinated one (label=1), using the question itself
    as the generation prompt in both cases (the detector doesn't see the
    answers directly - it generates its own and we score consistency).
    """
    df = df.head(n)
    prompts = df["input_text"].tolist()
    # Ground truth here is simplified: since we're prompting the model fresh
    # (not judging the given answers), we treat every example as label
    # unknown w.r.t. this particular generation and instead evaluate whether
    # LOW consistency correlates with the question being one HaluEval flagged
    # as prone to hallucination (i.e. all HaluEval questions are candidates,
    # so this run measures the detector's spread on real hard questions -
    # not a clean binary ground truth). See docs/architecture.md note.
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--backend", default="huggingface")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=5, help="generations per prompt for self-consistency")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--output", default="results/logs/self_consistency_run.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model} (4bit={args.load_in_4bit})")
    llm = get_llm_wrapper(args.backend, args.model, load_in_4bit=args.load_in_4bit)
    detector = SelfConsistencyDetector()

    df = pd.read_csv("data/benchmarks/halueval_test.csv")
    prompts = build_halueval_pairs(df, args.n)

    results = []
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:60]}...")
        r = detector.score(llm, prompt, n_samples=args.n_samples)
        results.append({
            "prompt": prompt,
            "consistency_score": r["consistency_score"],
            "hallucination_risk": r["hallucination_risk"],
            "responses": r["responses"],
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {args.output}")
    risk_scores = [r["hallucination_risk"] for r in results if r["hallucination_risk"] is not None]
    print(f"Mean hallucination risk across {len(risk_scores)} prompts: {sum(risk_scores)/len(risk_scores):.3f}")
    print("\nNote: this run reports risk score distribution only. Once you have")
    print("ground-truth hallucination labels for these specific generations")
    print("(e.g. via human annotation or an NLI judge), pass them to")
    print("evaluate_detector() in evaluation/metrics.py for precision/recall/F1/AUROC.")


if __name__ == "__main__":
    main()
