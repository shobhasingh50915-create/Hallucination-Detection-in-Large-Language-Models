"""
NLI-based hallucination detection.
Checks whether a generated response is entailed by, contradicted by, or
neutral to a source context/evidence passage. Contradiction = likely
hallucination. Uses the same 3-way schema as the FEVER dataset
(entailment / neutral / contradiction).
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class NLIDetector:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # bart-large-mnli label order: 0=contradiction, 1=neutral, 2=entailment
        self.id2label = self.model.config.id2label

    def check(self, context, claim):
        """
        context: the source evidence/knowledge passage (the premise)
        claim: the generated response to check (the hypothesis)

        Returns a dict with:
          - label: predicted relationship ("entailment" / "neutral" / "contradiction")
          - scores: full probability distribution over the 3 labels
          - hallucination_risk: probability of contradiction (0-1)
        """
        inputs = self.tokenizer(context, claim, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

        scores = {self.id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
        predicted_label = max(scores, key=scores.get)
        contradiction_score = scores.get("contradiction", 0.0)

        return {
            "label": predicted_label,
            "scores": scores,
            "hallucination_risk": contradiction_score,
        }

    def check_batch(self, context, claims):
        """Convenience method: check multiple claims against the same context."""
        return [self.check(context, claim) for claim in claims]
