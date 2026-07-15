"""Quick sanity check for llm_wrapper.py using a tiny local model (no GPU needed)."""
import sys
sys.path.insert(0, "src")

from models.llm_wrapper import get_llm_wrapper

llm = get_llm_wrapper("huggingface", "gpt2")
response = llm.generate("The capital of France is", max_tokens=10)
print("Single generate():", response)

responses = llm.generate_multiple("The capital of France is", n=2, max_tokens=10)
print("generate_multiple():", responses)