"""
Retrieval-augmented verification.
Narrows the source context down to its single most relevant sentence
(via TF-IDF cosine similarity to the claim) before running NLI-based
contradiction checking, rather than checking against the full context.
Tests whether narrowing context fixes the FEVER -> HaluEval transfer
problem seen with the standalone semantic similarity detector.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from detection.nli_based_detection import NLIDetector


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


class RetrievalVerificationDetector:
    def __init__(self, nli_detector=None, top_k=1):
        self.nli_detector = nli_detector or NLIDetector()
        self.top_k = top_k

    def retrieve(self, context, claim):
        """Return the top_k most relevant sentences from context for claim."""
        sentences = split_sentences(context)
        if len(sentences) <= self.top_k:
            return sentences
        vectorizer = TfidfVectorizer().fit(sentences + [claim])
        sent_vecs = vectorizer.transform(sentences)
        claim_vec = vectorizer.transform([claim])
        sims = cosine_similarity(claim_vec, sent_vecs)[0]
        top_idx = sorted(sims.argsort()[::-1][:self.top_k])
        return [sentences[i] for i in top_idx]

    def check(self, context, claim):
        """
        context: the source evidence/knowledge passage
        claim: the generated response to check

        Returns a dict with:
          - retrieved_context: the narrowed context actually used
          - label / scores / hallucination_risk: from the underlying NLI check
        """
        relevant_sentences = self.retrieve(context, claim)
        narrowed_context = " ".join(relevant_sentences) if relevant_sentences else context
        result = self.nli_detector.check(context=narrowed_context, claim=claim)
        result["retrieved_context"] = narrowed_context
        return result

    def check_batch(self, context, claims):
        return [self.check(context, claim) for claim in claims]
