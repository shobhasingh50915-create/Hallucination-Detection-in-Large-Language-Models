"""
Step 1: Download benchmark datasets for hallucination detection.
"""

from datasets import load_dataset
import os

RAW_DIR = "data/raw"

def download_truthfulqa():
    print("Downloading TruthfulQA...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation")
    save_path = os.path.join(RAW_DIR, "truthfulqa")
    ds.save_to_disk(save_path)
    print(f"Saved to {save_path}")
    print(ds)

    print("\n--- Let's look at 2 real examples ---")
    for i in range(2):
        example = ds["validation"][i]
        print(f"\nExample {i+1}:")
        print("Question:", example["question"])
        print("Best answer:", example["best_answer"])
        print("Incorrect answers:", example["incorrect_answers"])


def download_halueval():
    print("Downloading HaluEval...")
    ds = load_dataset("pminervini/HaluEval", "qa")
    save_path = os.path.join(RAW_DIR, "halueval_qa")
    ds.save_to_disk(save_path)
    print(f"Saved to {save_path}")
    print(ds)


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    download_truthfulqa()
    download_halueval()
    print("\nDone!")