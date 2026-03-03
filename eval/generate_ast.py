import os, sys, ast, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    print("[WARN] json-repair not installed. Run: pip install json-repair")
    print("[WARN] Falling back to raw json.loads() only.")
    HAS_JSON_REPAIR = False

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = os.path.join(REPO_ROOT, "checkpoints", "lora", "checkpoint-277")  

# JSON AST prompt
PROMPT = '{"_type": "Module", "body": [{"_type": "FunctionDef",'
MAX_NEW_TOKENS = 512
NUM_SAMPLES = 20
SHOW_FAILURES = 5


def dict_to_ast(obj): #Recursively reconstruct an AST node from a JSON dict produced by ast_to_json().
    if isinstance(obj, dict) and "_type" in obj:
        cls = getattr(ast, obj["_type"], None)
        if cls is None:
            raise ValueError(f"Unknown AST node type: {obj['_type']}")
        node = cls()
        for key, val in obj.items():
            if key != "_type":
                setattr(node, key, dict_to_ast(val))
        ast.fix_missing_locations(node)
        return node
    elif isinstance(obj, list):
        return [dict_to_ast(item) for item in obj]
    else:
        return obj


def try_parse_output(text: str):
    """
    Try to parse the model's output as a JSON AST dict, then reconstruct the AST tree.
    Returns the AST tree on success, or None on failure.
    """
    # The model sees the prompt as part of input; reconstruct full JSON
    # If the prompt is not already in the output, prepend it
    if not text.strip().startswith("{"):
        # Find where JSON starts
        start = text.find("{")
        if start == -1:
            return None
        text = text[start:]

    # Attempt JSON repair if available
    if HAS_JSON_REPAIR:
        try:
            text = repair_json(text)
        except Exception:
            pass

    try:
        ast_dict = json.loads(text)
    except json.JSONDecodeError:
        return None
    
    try:
        tree = dict_to_ast(ast_dict)
        ast.unparse(tree)  # Validates the AST is well-formed
        return tree
    except Exception:
        return None


def generate_once(model, tokenizer) -> str:
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
        tree = try_parse_output(text)

        if tree is not None:
            success += 1
        else:
            fail += 1
            if failures_shown < SHOW_FAILURES:
                print("\n--- FAILURE SAMPLE ---")
                print(text[:400])
                print("(Could not parse as valid JSON AST)")
                failures_shown += 1

    print("\n=== RESULTS ===")
    print(f"Total:        {NUM_SAMPLES}")
    print(f"Success:      {success}")
    print(f"Fail:         {fail}")
    print(f"Success Rate: {round(success / NUM_SAMPLES * 100, 2)}%")


if __name__ == "__main__":
    main()