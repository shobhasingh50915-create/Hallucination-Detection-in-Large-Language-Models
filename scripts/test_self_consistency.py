"""Sanity check for self_consistency.py using gpt2 (already cached from previous test)."""
import sys
sys.path.insert(0, "src")

from models.llm_wrapper import get_llm_wrapper
from detection.self_consistency import SelfConsistencyDetector

llm = get_llm_wrapper("huggingface", "gpt2")
detector = SelfConsistencyDetector()

result = detector.score(llm, "The capital of France is", n_samples=4, max_tokens=15)
print("Consistency score:", result["consistency_score"])
print("Hallucination risk:", result["hallucination_risk"])
print("Responses:")
for r in result["responses"]:
    print(" -", r)