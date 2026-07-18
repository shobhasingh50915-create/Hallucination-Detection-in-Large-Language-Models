"""
Generates comparison plots from the 5-way HaluEval ensemble results CSV.
Reads results/logs/ensemble_5way_halueval_eval.csv (already contains per-example
risk scores and true labels), recomputes AUROC/F1 per method fresh via the
existing evaluation.metrics module, and saves bar charts to results/plots/.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from evaluation.metrics import evaluate_detector

os.makedirs("results/plots", exist_ok=True)

df = pd.read_csv("results/logs/ensemble_5way_halueval_eval.csv")
true_labels = df["true_label"].tolist()

methods = {
    "NLI": "nli_risk",
    "Self-Consistency": "sc_risk",
    "Semantic Similarity": "sem_risk",
    "Uncertainty": "unc_risk",
    "Retrieval Verification": "rv_risk",
    "Ensemble: NLI+SC": "ensemble_2way",
    "Ensemble: NLI+SC+Uncertainty": "ensemble_nli_sc_unc",
    "Ensemble: NLI+SC+Retrieval": "ensemble_nli_sc_rv",
    "Ensemble: NLI+SC+Unc+Retrieval": "ensemble_nli_sc_unc_rv",
    "Ensemble: All 5-way": "ensemble_5way",
}

results = {}
for label, col in methods.items():
    metrics = evaluate_detector(df[col].tolist(), true_labels, threshold=0.5)
    results[label] = metrics["auroc"]

sorted_items = sorted(results.items(), key=lambda x: x[1])
labels = [k for k, v in sorted_items]
values = [v for k, v in sorted_items]
colors = ["#d62728" if "Ensemble" not in l else "#2ca02c" for l in labels]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(labels, values, color=colors)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (AUROC 0.5)")
ax.set_xlabel("AUROC")
ax.set_title("Hallucination Detector Comparison on HaluEval (n=60)")
ax.set_xlim(0, 1)
ax.legend(loc="lower right")

for bar, val in zip(bars, values):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("results/plots/halueval_auroc_comparison.png", dpi=150)
print("Saved plot to results/plots/halueval_auroc_comparison.png")
print("\nComputed AUROC values used in the plot:")
for label, val in sorted_items:
    print(f"  {label}: {val:.3f}")
