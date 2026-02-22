import os, sys, ast, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = os.path.join(REPO_ROOT, "checkpoints", "lora", "checkpoint-277")

PROMPT = "Module(body=[FunctionDef(name="
MAX_NEW_TOKENS = 300
NUM_SAMPLES = 20
SHOW_FAILURES = 20


def generate_once(model, tokenizer):
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

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

    success = 0
    fail = 0
    failures_shown = 0

    for i in range(NUM_SAMPLES):
        text = generate_once(model, tokenizer)

        try:
            tree = eval(text, {"__builtins__": {}}, vars(ast))
            ast.unparse(tree)
            success += 1

        except Exception as e:
            fail += 1

            if failures_shown < SHOW_FAILURES:
                print("\n--- FAILURE SAMPLE ---")
                print(text[:300])
                print("Error:", type(e).__name__)
                failures_shown += 1

    print("\nRESULTS")
    print("Total:", NUM_SAMPLES)
    print("Success:", success)
    print("Fail:", fail)
    print("Success Rate:", round(success / NUM_SAMPLES * 100, 2), "%")


if __name__ == "__main__":
    main()