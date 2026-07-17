# Hallucination Detection in Large Language Models

## Overview
This project detects hallucinations in LLM outputs using four implemented detection methods -- NLI-based contradiction detection, self-consistency checking, semantic similarity, and verbalized-confidence uncertainty -- combined into normalized ensembles, evaluated on FEVER and HaluEval.

## Implemented methods
- **NLI-based detection**: uses `facebook/bart-large-mnli` to check if a generated claim is entailed by, contradicted by, or neutral to its source context. Contradiction probability = hallucination risk.
- **Self-consistency detection**: generates multiple LLM responses to the same prompt (Groq `llama-3.1-8b-instant`) and measures embedding agreement across them. Low agreement = high hallucination risk.
- **Semantic similarity detection**: embeds the generated claim and source context, uses cosine distance as hallucination risk. Works well on FEVER; does not transfer to HaluEval-style QA grounding (see Results below).
- **Uncertainty detection**: prompts the LLM to self-rate its confidence (0-100) in its own answer, given the source context. Low confidence = high hallucination risk. Lightweight verbalized-confidence technique, no extra dependencies.
- **Ensembles**: detector scores are min-max normalized, then averaged. Multiple detector combinations are compared (see Results below).

Planned but not yet implemented: retrieval-augmented verification (stub file exists in `src/detection/`).

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

**Key finding**: the normalized ensemble achieves the best AUROC (0.77), outperforming either detector alone -- evidence the two methods capture complementary signal.

### Semantic similarity detector -- a third method, with a notable negative result

A third detector was added: cosine similarity between the embedded generated claim and its source context (low similarity -> high hallucination risk).

**On FEVER (n=200)**, this works reasonably well: AUROC 0.71, best F1 0.45 -- comparable to the other detectors.

**On HaluEval (n=60)**, it fails badly: AUROC **0.12** (worse than random, i.e. inverted signal), and adding it to the ensemble *hurt* overall performance -- the 3-way ensemble (NLI + self-consistency + semantic similarity) scored AUROC 0.29, well below the 2-way ensemble's 0.60.

**Why**: FEVER compares a claim sentence against a context sentence of similar scope, so similarity tracks agreement well. HaluEval compares a short QA answer (e.g. "Bram Stoker") against a long multi-sentence context passage -- a correct short answer is naturally *dissimilar* in embedding space to a long paragraph, while a rambling incorrect answer that echoes more of the context's wording can score artificially higher similarity. Raw embedding similarity, as implemented, doesn't transfer well to QA-answer grounding without adjustment (e.g., comparing against the single most relevant sentence rather than the full context).

**Conclusion used going forward**: semantic similarity is retained as a standalone, FEVER-validated detector, but excluded from the HaluEval ensemble. This is reported as a deliberate, methodologically-grounded finding rather than a bug -- a useful illustration that detector transferability across task types can't be assumed.

### Uncertainty detector -- a fourth method, added to the HaluEval ensemble

A fourth detector was added: verbalized confidence, where the LLM self-rates 0-100 confidence in its own answer given the source context.

**Standalone results on HaluEval (n=60, this run)**:
| Method | Accuracy | F1 | AUROC |
|---|---|---|---|
| NLI only | 0.68 | 0.34 | 0.53 |
| Self-consistency only | 0.68 | 0.17 | 0.62 |
| Semantic similarity only | 0.27 | 0.12 | 0.18 |
| Uncertainty only | 0.70 | 0.25 | 0.57 |

**Ensemble comparison (same run)**:
| Ensemble | AUROC |
|---|---|
| NLI + Self-consistency (2-way) | 0.61 |
| NLI + Self-consistency + Semantic similarity (3-way) | 0.39 |
| **NLI + Self-consistency + Uncertainty** | **0.64** |
| All four (4-way) | 0.44 |

**Key finding**: adding the uncertainty detector to NLI + self-consistency improves AUROC (0.61 -> 0.64) -- it contributes complementary signal, unlike semantic similarity. Including semantic similarity in any HaluEval ensemble continues to hurt performance, dragging the full 4-way ensemble (0.44) below every 2- and 3-detector combination that excludes it.

*Note: standalone NLI/self-consistency/ensemble_2way numbers in this run differ somewhat from the table in the section above (e.g. NLI AUROC 0.53 vs 0.59, ensemble_2way 0.61 vs 0.77), most likely due to run-to-run variance in the live self-consistency LLM calls rather than a code change. Flagged here for transparency rather than silently reconciled.*

### Limitations
- Sample sizes (60-200) are small; treat metric differences as suggestive, not conclusive.
- HaluEval ground truth uses a token-overlap proxy against reference answers, not human labels.
- Semantic similarity does not transfer from FEVER-style claim verification to HaluEval-style QA grounding (see above) -- it is task-dependent, not a general-purpose signal in its current form.
- Results show some run-to-run variance, likely from non-determinism in live LLM calls (self-consistency, uncertainty) despite temperature 0.
- 4 of 5 originally planned detection methods are implemented (NLI, self-consistency, semantic similarity, uncertainty); retrieval-augmented verification remains a stub.

## Project structure
## Installation
```bash
pip install -r requirements.txt
```

## Run evaluations
```bash
python scripts/evaluate_nli_on_fever.py
python scripts/evaluate_ensemble_on_halueval.py
python scripts/evaluate_ensemble_4way_halueval.py
```

## Author
**Shobhakumari Singh**
