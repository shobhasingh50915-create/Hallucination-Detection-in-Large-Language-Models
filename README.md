# Hallucination Detection in Large Language Models

## Overview
This project detects hallucinations in LLM outputs using three implemented detection methods — NLI-based contradiction detection, self-consistency checking, and semantic similarity — combined into a normalized ensemble, evaluated on FEVER and HaluEval.

## Implemented methods
- **NLI-based detection**: uses `facebook/bart-large-mnli` to check if a generated claim is entailed by, contradicted by, or neutral to its source context. Contradiction probability = hallucination risk.
- **Self-consistency detection**: generates multiple LLM responses to the same prompt (Groq `llama-3.1-8b-instant`) and measures embedding agreement across them. Low agreement = high hallucination risk.
- **Semantic similarity detection**: embeds the generated claim and source context, uses cosine distance as hallucination risk. Works well on FEVER; does not transfer to HaluEval-style QA grounding (see Results below).
- **Ensemble**: detector scores are min-max normalized, then averaged. The best-performing ensemble on HaluEval combines NLI + self-consistency only (see below).

Planned but not yet implemented: retrieval-augmented verification, uncertainty-based methods (stub files exist in `src/detection/`).

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

### Semantic similarity detector — a third method, with a notable negative result

A third detector was added: cosine similarity between the embedded generated claim and its source context (low similarity → high hallucination risk).

**On FEVER (n=200)**, this works reasonably well: AUROC 0.71, best F1 0.45 — comparable to the other detectors.

**On HaluEval (n=60)**, it fails badly: AUROC **0.12** (worse than random, i.e. inverted signal), and adding it to the ensemble *hurt* overall performance — the 3-way ensemble (NLI + self-consistency + semantic similarity) scored AUROC 0.29, well below the 2-way ensemble's 0.60.

**Why**: FEVER compares a claim sentence against a context sentence of similar scope, so similarity tracks agreement well. HaluEval compares a short QA answer (e.g. "Bram Stoker") against a long multi-sentence context passage — a correct short answer is naturally *dissimilar* in embedding space to a long paragraph, while a rambling incorrect answer that echoes more of the context's wording can score artificially higher similarity. Raw embedding similarity, as implemented, doesn't transfer well to QA-answer grounding without adjustment (e.g., comparing against the single most relevant sentence rather than the full context).

**Conclusion used going forward**: semantic similarity is retained as a standalone, FEVER-validated detector, but excluded from the HaluEval ensemble. This is reported as a deliberate, methodologically-grounded finding rather than a bug — a useful illustration that detector transferability across task types can't be assumed.

### Limitations
- Sample sizes (60–200) are small; treat metric differences as suggestive, not conclusive.
- HaluEval ground truth uses a token-overlap proxy against reference answers, not human labels.
- Semantic similarity does not transfer from FEVER-style claim verification to HaluEval-style QA grounding (see above) — it is task-dependent, not a general-purpose signal in its current form.
- 3 of 5 originally planned detection methods are implemented (NLI, self-consistency, semantic similarity); retrieval-augmented verification and uncertainty-based methods remain stubs.

## Project structure
```
hallucination-detection-llm/
├── config/
├── data/
├── src/
│   └── detection/       # nli_based_detection.py, self_consistency.py, semantic_similarity.py (implemented);
│                         # retrieval_verification.py, uncertainity_methods.py (stubs)
├── api/
├── notebooks/
├── tests/
├── scripts/              # evaluate_nli_on_fever.py, evaluate_ensemble_on_halueval.py,
│                          # evaluate_semantic_similarity_on_fever.py, evaluate_ensemble_3way_halueval.py
├── results/logs/          # nli_fever_eval.csv, ensemble_halueval_eval.csv,
│                          # semantic_similarity_fever_eval.csv, ensemble_3way_halueval_eval.csv
└── docs/
```

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
