# Hallucination Detection in Large Language Models

## Overview
This project detects hallucinations in LLM outputs using two implemented detection methods — NLI-based contradiction detection and self-consistency checking — combined into a normalized ensemble, evaluated on FEVER and HaluEval.

## Implemented methods
- **NLI-based detection**: uses `facebook/bart-large-mnli` to check if a generated claim is entailed by, contradicted by, or neutral to its source context. Contradiction probability = hallucination risk.
- **Self-consistency detection**: generates multiple LLM responses to the same prompt (Groq `llama-3.1-8b-instant`) and measures embedding agreement across them. Low agreement = high hallucination risk.
- **Ensemble**: both detectors' scores are min-max normalized, then averaged.

Planned but not yet implemented: retrieval-augmented verification, semantic similarity, uncertainty-based methods (stub files exist in `src/detection/`).

## Results

### NLI detector on FEVER (n=200)
| Metric | Value |
|---|---|
| Accuracy | 0.79 |
| F1 | 0.52 |
| AUROC | 0.77 |

### NLI + Self-Consistency ensemble on HaluEval (n=60)
| Method | Accuracy | F1 | AUROC |
|---|---|---|---|
| NLI only | 0.68 | 0.30 | 0.59 |
| Self-consistency only | 0.70 | 0.18 | 0.72 |
| **Ensemble (normalized)** | **0.72** | 0.32 | **0.77** |

**Key finding**: the normalized ensemble achieves the best AUROC (0.77), outperforming either detector alone — evidence the two methods capture complementary signal.

### Limitations
- Sample sizes (60–200) are small; treat metric differences as suggestive, not conclusive.
- HaluEval ground truth uses a token-overlap proxy against reference answers, not human labels.
- Only 2 of 5 planned detection methods are implemented.

## Project structure
## Installation
```bash
pip install -r requirements.txt
```

## Run evaluations
```bash
python scripts/evaluate_nli_on_fever.py
python scripts/evaluate_ensemble_on_halueval.py
```

## Author
**Shobha Singh**
