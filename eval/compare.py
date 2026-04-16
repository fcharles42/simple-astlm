import os, ast, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except:
    HAS_JSON_REPAIR = False


BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_PATH = "checkpoints/lora/checkpoint-277"

K = 10
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7

PREFIXES = [
    "def is_prime(n):",
    "def factorial(n):",
    "def reverse_string(s):",
]


# ------------------------
# Utils
# ------------------------

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


def is_valid(code):
    try:
        ast.parse(code)
        return True
    except:
        return False


def ast_to_code(tree):
    try:
        code = ast.unparse(tree)
        ast.parse(code)
        return code
    except:
        return None


# ------------------------
# AST helpers
# ------------------------

def build_ast_prompt(prefix):
    name = prefix.split("(")[0].replace("def", "").strip()

    return f'''{{
  "_type": "Module",
  "body": [{{
    "_type": "FunctionDef",
    "name": "{name}",
    "args": {{
      "_type": "arguments",
      "posonlyargs": [],
      "args": [],
      "kwonlyargs": [],
      "kw_defaults": [],
      "defaults": []
    }},
    "body": ['''


def dict_to_ast(obj):
    if isinstance(obj, dict) and "_type" in obj:
        cls = getattr(ast, obj["_type"], None)
        if cls is None:
            return None
        node = cls()
        for k, v in obj.items():
            if k != "_type":
                setattr(node, k, dict_to_ast(v))
        return node
    elif isinstance(obj, list):
        return [dict_to_ast(x) for x in obj]
    return obj


def try_parse_ast(text):
    if not text.strip().startswith("{"):
        idx = text.find("{")
        if idx == -1:
            return None
        text = text[idx:]

    if HAS_JSON_REPAIR:
        try:
            text = repair_json(text)
        except:
            pass

    try:
        obj = json.loads(text)
    except:
        return None

    try:
        tree = dict_to_ast(obj)
        ast.fix_missing_locations(tree)
        ast.unparse(tree)
        return tree
    except:
        return None


# ------------------------
# Evaluation
# ------------------------

def evaluate_mode(mode, model, tokenizer, prefix):
    results = []
    samples = []

    for _ in range(K):

        if mode == "base":
            out = generate(model, tokenizer, prefix)
            valid = is_valid(out)
            samples.append(out)

        elif mode == "ast_prompt":
            ast_prompt = build_ast_prompt(prefix)
            out = generate(model, tokenizer, ast_prompt)

            tree = try_parse_ast(out)
            if tree is None:
                valid = False
                samples.append(None)
            else:
                code = ast_to_code(tree)
                valid = code is not None
                samples.append(code)

        elif mode == "ast_code_prompt":
            out = generate(model, tokenizer, prefix)
            valid = is_valid(out)
            samples.append(out)

        results.append(valid)

    return {
        "pass@k": any(results),
        "valid_rate": sum(results) / K,
        "samples": samples
    }


# ------------------------
# Main
# ------------------------

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    ast_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    ast_model.eval()

    for prefix in PREFIXES:
        print("\n" + "="*60)
        print(f"PROMPT: {prefix}")
        print("="*60)

        base_res = evaluate_mode("base", base_model, tokenizer, prefix)
        ast_res = evaluate_mode("ast_prompt", ast_model, tokenizer, prefix)
        ast_code_res = evaluate_mode("ast_code_prompt", ast_model, tokenizer, prefix)

        # ---- SHOW EXAMPLES ----
        print("\n[Base Model Example]")
        print(base_res["samples"][0])

        print("\n[AST Model (AST prompt) Example]")
        print(ast_res["samples"][0])

        print("\n[AST Model (Code prompt) Example]")
        print(ast_code_res["samples"][0])

        # ---- METRICS ----
        print("\n--- METRICS ---")
        print("Base:", base_res["pass@k"], base_res["valid_rate"])
        print("AST (AST prompt):", ast_res["pass@k"], ast_res["valid_rate"])
        print("AST (Code prompt):", ast_code_res["pass@k"], ast_code_res["valid_rate"])


if __name__ == "__main__":
    main()