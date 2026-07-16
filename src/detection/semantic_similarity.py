"""
Semantic similarity-based hallucination detection.
Embeds the generated claim and the source context, uses cosine distance
between them as the hallucination risk score — low similarity to the
source context suggests the claim may not be grounded in it.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SemanticSimilarityDetector:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)

    def check(self, context, claim):
        """
        context: source evidence/knowledge passage
        claim: generated response to check

        Returns a dict with:
          - similarity: cosine similarity between context and claim embeddings (0-1)
          - hallucination_risk: 1 - similarity (higher = more likely hallucinated)
        """
        embeddings = self.embedder.encode([context, claim])
        sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        sim = max(0.0, min(1.0, sim))  # clip to [0,1] in case of small negative cosine values
        return {
            "similarity": sim,
            "hallucination_risk": 1 - sim,
        }

    def check_batch(self, context, claims):
        """Convenience method: check multiple claims against the same context."""
        return [self.check(context, claim) for claim in claims]
