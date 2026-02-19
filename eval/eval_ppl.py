import os, sys, math, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from peft import PeftModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"

LORA_PATH = os.path.join(REPO_ROOT, "checkpoints", "lora", "checkpoint-277")
TEST_TOKENIZED_PATH = os.path.join(REPO_ROOT, "data", "processed", "tokenized_test.pt")


def main():
    print("[INFO] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("[INFO] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    print("[INFO] Loading test tokens...")
    data = torch.load(TEST_TOKENIZED_PATH)

    losses = []
    total_loss = 0.0

    with torch.no_grad():
        for ex in tqdm(data):
            input_ids = ex["input_ids"].unsqueeze(0).to(model.device)
            labels = ex["labels"].unsqueeze(0).to(model.device)

            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss.item()

            losses.append(loss)
            total_loss += loss

    avg_loss = total_loss / len(losses)
    ppl = math.exp(avg_loss)

    print("\n========== EVAL RESULTS ==========")
    print("Blocks evaluated:", len(losses))
    print("Avg loss:", round(avg_loss, 4))
    print("Perplexity:", round(ppl, 4))


if __name__ == "__main__":
    main()
