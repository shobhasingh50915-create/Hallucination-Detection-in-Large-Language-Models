"""Sanity check for nli_based_detection.py with clear entailment vs contradiction cases."""
import sys
sys.path.insert(0, "src")

from detection.nli_based_detection import NLIDetector

detector = NLIDetector()

context = "Paris is the capital of France. It is located on the Seine river."

print("=== Testing entailment (should be low hallucination risk) ===")
result = detector.check(context, "Paris is the capital city of France.")
print(result)

print("\n=== Testing contradiction (should be high hallucination risk) ===")
result = detector.check(context, "London is the capital of France.")
print(result)

print("\n=== Testing neutral (unrelated claim) ===")
result = detector.check(context, "France has excellent cuisine.")
print(result)