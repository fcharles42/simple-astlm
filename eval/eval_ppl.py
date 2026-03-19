import os
import sys
import math
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"

LORA_PATH = os.path.join(REPO_ROOT, "checkpoints", "lora", "checkpoint-690")
TEST_TOKENIZED_PATH = os.path.join(REPO_ROOT, "data", "processed", "tokenized_test.pt")
TOKENIZER_PATH = os.path.join(REPO_ROOT, "data", "processed", "tokenizer")


def main():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    print("[INFO] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    device = next(model.parameters()).device

    print("[INFO] Loading test tokens...")
    data = torch.load(TEST_TOKENIZED_PATH)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for ex in tqdm(data):
            input_ids = ex["input_ids"].unsqueeze(0).to(device)
            labels = ex["labels"].unsqueeze(0).to(device)

            outputs = model(input_ids=input_ids, labels=labels)

            loss = outputs.loss  # already averaged over valid tokens

            # Count valid (non -100) tokens
            valid_tokens = (labels != -100).sum().item()

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    print("\n========== EVAL RESULTS ==========")
    print("Total tokens:", total_tokens)
    print("Avg loss:", round(avg_loss, 4))
    print("Perplexity:", round(ppl, 4))


if __name__ == "__main__":
    main()