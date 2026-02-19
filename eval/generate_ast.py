import os, sys, ast, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = os.path.join(REPO_ROOT, "checkpoints", "lora", "checkpoint-277")

PROMPT = "Module(body=[FunctionDef("
MAX_NEW_TOKENS = 600


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    print("\n========== GENERATED AST DUMP ==========\n")
    print(text)

    # Try reconstructing AST object
    print("\n========== TRY PARSING AST DUMP ==========\n")

    try:
        tree = eval(text, {"__builtins__": {}}, vars(ast))
        ast.fix_missing_locations(tree)

        code = ast.unparse(tree)

        print("[OK] Successfully parsed AST + unparsed to Python code!\n")
        print("========== GENERATED PYTHON CODE ==========\n")
        print(code)

    except Exception as e:
        print("[FAIL] Could not convert generated AST dump back to code.")
        print("Error:", repr(e))


if __name__ == "__main__":
    main()
