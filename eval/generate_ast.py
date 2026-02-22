import os, sys, ast, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = os.path.join(REPO_ROOT, "checkpoints", "lora", "checkpoint-277")

PROMPT = "Module(body=[FunctionDef("
MAX_NEW_TOKENS = 600

NUM_SAMPLES = 50          
SHOW_FAILURES = 3         

def generate_once(model, tokenizer, do_sample=True):
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=do_sample,
            temperature=0.8 if do_sample else None,
            top_p=0.95 if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def evaluate(model, tokenizer, do_sample=True):
    success = 0
    fail = 0
    ename_count = 0
    failures_shown = 0

    for i in range(NUM_SAMPLES):
        text = generate_once(model, tokenizer, do_sample=do_sample)

        if "ename=" in text:
            ename_count += 1

        try:
            tree = eval(text, {"__builtins__": {}}, vars(ast))
            ast.fix_missing_locations(tree)
            ast.unparse(tree)
            success += 1

        except Exception as e:
            fail += 1

            if failures_shown < SHOW_FAILURES:
                print("\n----- FAILURE SAMPLE -----")
                print(text[:500])
                print("Error:", repr(e))
                failures_shown += 1

    return success, fail, ename_count


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

    print("[INFO] Loading LoRA...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    # Sampling evaluation
    print("\n========== SAMPLING MODE ==========")
    s_success, s_fail, s_ename = evaluate(model, tokenizer, do_sample=True)

    print("\n[Sampling Results]")
    print("Total:", NUM_SAMPLES)
    print("Success:", s_success)
    print("Fail:", s_fail)
    print("ename occurrences:", s_ename)

    # Greedy evaluation
    print("\n========== GREEDY MODE ==========")
    g_success, g_fail, g_ename = evaluate(model, tokenizer, do_sample=False)

    print("\n[Greedy Results]")
    print("Total:", NUM_SAMPLES)
    print("Success:", g_success)
    print("Fail:", g_fail)
    print("ename occurrences:", g_ename)


if __name__ == "__main__":
    main()