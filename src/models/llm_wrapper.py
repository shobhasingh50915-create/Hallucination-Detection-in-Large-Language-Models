"""
LLM abstraction layer - unified interface over different backends.
Supports: local HuggingFace models (quantized, for Colab), OpenAI API,
and Groq API (free tier, OpenAI-compatible, fast inference).
"""

from abc import ABC, abstractmethod
import os


class LLMWrapper(ABC):
    @abstractmethod
    def generate(self, prompt, max_tokens=256, temperature=0.7):
        """Generate a single response."""
        pass

    def generate_multiple(self, prompt, n=5, max_tokens=256, temperature=0.9):
        """Generate n responses (used for self-consistency detection). Default
        implementation just calls generate() n times; subclasses can override
        for more efficient batched generation."""
        return [self.generate(prompt, max_tokens, temperature) for _ in range(n)]


class HuggingFaceWrapper(LLMWrapper):
    """Local HF model, e.g. Mistral-7B-Instruct or Llama-3-8B-Instruct.
    Use load_in_4bit=True on Colab with a GPU + bitsandbytes installed."""

    def __init__(self, model_name, load_in_4bit=False, device_map="auto"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        kwargs = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    def generate(self, prompt, max_tokens=256, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text[len(prompt):].strip()


class OpenAIWrapper(LLMWrapper):
    """GPT-3.5-turbo / GPT-4o-mini via OpenAI API. Requires OPENAI_API_KEY env var."""

    def __init__(self, model_name="gpt-4o-mini"):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable before using OpenAIWrapper.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt, max_tokens=256, temperature=0.7):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


class GroqWrapper(LLMWrapper):
    """Free-tier, fast inference via Groq's OpenAI-compatible API.
    Requires GROQ_API_KEY env var. Get a free key at console.groq.com/keys."""

    def __init__(self, model_name="llama-3.1-8b-instant"):
        from openai import OpenAI
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Set GROQ_API_KEY environment variable before using GroqWrapper.")
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model_name = model_name

    def generate(self, prompt, max_tokens=256, temperature=0.7):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


def get_llm_wrapper(backend, model_name, **kwargs):
    """Factory function. backend: 'huggingface', 'openai', or 'groq'."""
    if backend == "huggingface":
        return HuggingFaceWrapper(model_name, **kwargs)
    elif backend == "openai":
        return OpenAIWrapper(model_name)
    elif backend == "groq":
        return GroqWrapper(model_name)
    else:
        raise ValueError(f"Unknown backend: {backend}")
