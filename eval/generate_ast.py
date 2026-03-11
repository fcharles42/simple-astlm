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

AST_START = "<ast_start>"
AST_END = "<ast_end>"

PROMPT = f'{AST_START}{{"_type": "Module", "body": [{{"_type": "FunctionDef",'
MAX_NEW_TOKENS = 512
NUM_SAMPLES = 20
SHOW_FAILURES = 5


def dict_to_ast(obj):
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


# Returns (outcome, tree)
# outcome: "success" | "truncated" | "invalid"
def try_parse_output(text: str):
    # Extract between boundary tokens
    start_idx = text.find(AST_START)
    end_idx = text.find(AST_END)

    had_end_token = end_idx != -1

    if start_idx != -1:
        content_start = start_idx + len(AST_START)
        text = text[content_start:end_idx] if had_end_token else text[content_start:]
    else:
        # Fall back: find raw JSON start
        raw_start = text.find("{")
        if raw_start == -1:
            return "invalid", None
        text = text[raw_start:end_idx] if had_end_token else text[raw_start:]

    if HAS_JSON_REPAIR:
        try:
            text = repair_json(text)
        except Exception:
            pass

    try:
        ast_dict = json.loads(text)
    except json.JSONDecodeError:
        return "truncated" if had_end_token else "invalid", None

    try:
        tree = dict_to_ast(ast_dict)
        ast.unparse(tree)
        return "success", tree
    except Exception:
        return "truncated" if had_end_token else "invalid", None


def generate_once(model, tokenizer) -> str:
    ast_end_id = tokenizer.convert_tokens_to_ids(AST_END)

    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.2,
            eos_token_id=[tokenizer.eos_token_id, ast_end_id],
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0], skip_special_tokens=False)


def main():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [AST_START, AST_END]})

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

    counts = {"success": 0, "truncated": 0, "invalid": 0}
    failures_shown = 0

    for i in range(NUM_SAMPLES):
        text = generate_once(model, tokenizer)
        outcome, tree = try_parse_output(text)
        counts[outcome] += 1

        if outcome != "success" and failures_shown < SHOW_FAILURES:
            had_end = AST_END in text
            print(f"\n--- {outcome.upper()} (end_token={'yes' if had_end else 'no'}) ---")
            print(text[:400])
            failures_shown += 1

    print("\n=== RESULTS ===")
    print(f"Total:        {NUM_SAMPLES}")
    print(f"Success:      {counts['success']}  ({round(counts['success']/NUM_SAMPLES*100, 1)}%)")
    print(f"Truncated:    {counts['truncated']}  ({round(counts['truncated']/NUM_SAMPLES*100, 1)}%)")
    print(f"Invalid:      {counts['invalid']}  ({round(counts['invalid']/NUM_SAMPLES*100, 1)}%)")


if __name__ == "__main__":
    main()