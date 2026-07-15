"""
Self-consistency hallucination detection.
Generate multiple responses to the same prompt; low semantic agreement
across responses signals possible hallucination.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SelfConsistencyDetector:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)

    def score(self, llm_wrapper, prompt, n_samples=5, max_tokens=256, temperature=0.9):
        """
        Returns a dict with:
          - consistency_score: mean pairwise cosine similarity (0-1, higher = more consistent)
          - hallucination_risk: 1 - consistency_score (higher = more likely hallucinated)
          - responses: the raw generated responses (for inspection/logging)
        """
        responses = llm_wrapper.generate_multiple(
            prompt, n=n_samples, max_tokens=max_tokens, temperature=temperature
        )

        # filter out empty responses (can happen with weak models / edge prompts)
        responses = [r for r in responses if r.strip()]
        if len(responses) < 2:
            return {
                "consistency_score": None,
                "hallucination_risk": None,
                "responses": responses,
                "note": "fewer than 2 non-empty responses; cannot compute consistency",
            }

        embeddings = self.embedder.encode(responses)
        sim_matrix = cosine_similarity(embeddings)

        # average of upper triangle (excluding diagonal) = mean pairwise similarity
        n = len(responses)
        upper_triangle = sim_matrix[np.triu_indices(n, k=1)]
        consistency_score = float(np.mean(upper_triangle))

        return {
            "consistency_score": consistency_score,
            "hallucination_risk": 1 - consistency_score,
            "responses": responses,
        }
