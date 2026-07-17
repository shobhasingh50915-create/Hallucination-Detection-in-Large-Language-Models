"""
Uncertainty-based hallucination detection via verbalized confidence.
Prompts the LLM to self-rate its confidence (0-100) in its own answer's
correctness, given the source context. Low confidence = high hallucination
risk. This is a standard lightweight uncertainty-estimation technique that
avoids needing raw token log-probabilities (which many hosted APIs don't
expose), based on the finding that LLMs have some calibrated sense of what
they know (see: "Language Models (Mostly) Know What They Know").
"""

import re


class UncertaintyDetector:
    def __init__(self):
        pass

    def check(self, llm_wrapper, context, question, answer):
        """
        llm_wrapper: an LLMWrapper instance (e.g. GroqWrapper)
        context: source evidence/knowledge passage
        question: the original question
        answer: the answer to rate confidence for

        Returns a dict with:
          - confidence: self-reported confidence (0-100)
          - hallucination_risk: 1 - (confidence / 100)
          - raw_response: the raw text response (for inspection/debugging)
        """
        prompt = (
            f"Context: {context}\n\n"
            f"Question: {question}\n"
            f"Proposed answer: {answer}\n\n"
            "On a scale of 0 to 100, how confident are you that this answer is "
            "factually correct and well-supported by the context? "
            "Respond with ONLY a single number from 0 to 100, nothing else."
        )
        raw = llm_wrapper.generate(prompt, max_tokens=10, temperature=0.0)
        match = re.search(r"\d+", raw)
        confidence = min(100, max(0, int(match.group()))) if match else 50

        return {
            "confidence": confidence,
            "hallucination_risk": 1 - (confidence / 100),
            "raw_response": raw,
        }
