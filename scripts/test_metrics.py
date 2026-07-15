"""Sanity check for evaluation/metrics.py using synthetic scores/labels."""
import sys
sys.path.insert(0, "src")

from evaluation.metrics import evaluate_detector, sweep_thresholds

# synthetic example: 6 samples, risk scores and true hallucination labels
risk_scores = [0.9, 0.8, 0.2, 0.1, 0.6, 0.4]
true_labels = [1,   1,   0,   0,   1,   0]

result = evaluate_detector(risk_scores, true_labels, threshold=0.5)
print("Metrics at threshold 0.5:", result)

sweep = sweep_thresholds(risk_scores, true_labels)
print("\nBest threshold:", sweep["best"])