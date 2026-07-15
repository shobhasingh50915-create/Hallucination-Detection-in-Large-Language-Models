"""
Evaluation metrics for hallucination detection.
Takes continuous risk scores + binary ground truth labels, computes
standard classification metrics plus threshold-free AUROC.
"""

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, confusion_matrix
)
import numpy as np


def evaluate_detector(risk_scores, true_labels, threshold=0.5):
    """
    risk_scores: list/array of floats in [0,1], higher = more likely hallucinated
    true_labels: list/array of 0/1, where 1 = actually hallucinated
    threshold: cutoff above which a risk_score counts as a positive (hallucinated) prediction

    Returns a dict of metrics. Also returns AUROC, which does not depend on
    the threshold and is the most robust single number to report when
    comparing detection methods.
    """
    risk_scores = np.array(risk_scores, dtype=float)
    true_labels = np.array(true_labels, dtype=int)

    predictions = (risk_scores >= threshold).astype(int)

    results = {
        "threshold": threshold,
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, zero_division=0),
        "recall": recall_score(true_labels, predictions, zero_division=0),
        "f1": f1_score(true_labels, predictions, zero_division=0),
    }

    # AUROC requires both classes present; guard against edge cases
    if len(np.unique(true_labels)) == 2:
        results["auroc"] = roc_auc_score(true_labels, risk_scores)
    else:
        results["auroc"] = None
        results["note"] = "AUROC undefined: true_labels contains only one class"

    tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()
    results["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    return results


def sweep_thresholds(risk_scores, true_labels, thresholds=None):
    """
    Compute F1 across a range of thresholds, to help pick the best cutoff
    instead of assuming 0.5. Returns the threshold with the highest F1
    along with the full sweep results.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    sweep = []
    for t in thresholds:
        m = evaluate_detector(risk_scores, true_labels, threshold=t)
        sweep.append({"threshold": t, "f1": m["f1"], "precision": m["precision"], "recall": m["recall"]})

    best = max(sweep, key=lambda x: x["f1"])
    return {"sweep": sweep, "best": best}
