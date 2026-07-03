#!/usr/bin/env python3
"""Eval all checkpoints: valid AST rate, pass@k, JSON rate, truncation, sig match, body, node count."""
import os, sys, ast, json, math, csv, re, argparse, traceback, time, inspect, tempfile
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from tqdm import tqdm

REPO_ROOT       = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
OUTPUT_CSV      = REPO_ROOT / "eval" / "results.csv"

BASE_MODELS = {
    "base":  "Qwen/Qwen2.5-0.5B",
    "coder": "Qwen/Qwen2.5-Coder-0.5B",
}

TRAIN_BS = {"ft2": 16, "ft15": 8, "ftfull": 4}
SEQ_LEN  = 2048

# intended total steps per run type (for correct % computation)
TOTAL_STEPS = {
    ("ft2",    "BT"):   3000,
    ("ft2",    "NoBT"): 3000,
    ("ft15",   "BT"):   6000,
    ("ft15",   "NoBT"): 6000,
    ("ftfull", "BT"):   12000,
    ("ftfull", "NoBT"): 3000,
}

K              = 10
MAX_NEW_TOKENS = 300
TEMPERATURE    = 0.2

PREFIXES = [
    "def is_prime(n):",
    "def factorial(n):",
    "def reverse_string(s):",
    "def fibonacci(n):",
    "def binary_search(arr, target):",
    "def merge_sort(lst):",
    "def count_vowels(s):",
    "def flatten(nested):",
    "def gcd(a, b):",
    "def power(base, exp):",
    "def is_palindrome(s):",
    "def sum_digits(n):",
    "def remove_duplicates(lst):",
    "def capitalize_words(sentence):",
    "def matrix_multiply(a, b):",
    "def caesar_cipher(text, shift):",
    "def count_words(text):",
    "def find_max(lst):",
    "def is_sorted(lst):",
    "def zip_lists(a, b):",
]


# ── AST helpers ───────────────────────────────────────────────────────────────

def build_prompt(prefix: str) -> str:
    name = re.match(r"def\s+(\w+)", prefix).group(1)
    m = re.search(r"\(([^)]*)\)", prefix)
    raw_args = [a.strip() for a in m.group(1).split(",") if a.strip()] if m else []
    args_json = ", ".join(
        '{{"_type": "arg", "arg": "{}", "annotation": null}}'.format(a.split(":")[0].strip())
        for a in raw_args
    )
    return (
        f'{{\n  "_type": "Module",\n  "body": [{{\n'
        f'    "_type": "FunctionDef",\n    "name": "{name}",\n'
        f'    "args": {{\n      "_type": "arguments",\n      "posonlyargs": [],\n'
        f'      "args": [{args_json}],\n      "kwonlyargs": [],\n'
        f'      "kw_defaults": [],\n      "defaults": []\n    }},\n'
        f'    "body": ['
    )


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


def try_parse(text: str):
    """Return (json_ok, tree_or_None)."""
    idx = text.find("{")
    if idx == -1:
        return False, None
    text = text[idx:]
    try:
        obj = json.loads(text)
    except Exception:
        return False, None
    try:
        tree = dict_to_ast(obj)
        ast.fix_missing_locations(tree)
        ast.unparse(tree)
        return True, tree
    except Exception:
        return True, None


def count_nodes(tree) -> int:
    return sum(1 for _ in ast.walk(tree))


def sig_matches(tree, prefix: str) -> bool:
    m_name = re.match(r"def\s+(\w+)", prefix)
    m_args = re.search(r"\(([^)]*)\)", prefix)
    if not m_name:
        return False
    expected_name = m_name.group(1)
    raw_args = [a.strip() for a in m_args.group(1).split(",") if a.strip()] if m_args else []
    expected_argc = len(raw_args)
    try:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name == expected_name and len(node.args.args) == expected_argc
    except Exception:
        pass
    return False


def body_nontrivial(tree) -> bool:
    try:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                body = node.body
                if len(body) == 0:
                    return False
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    return False
                if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    return False
                return True
    except Exception:
        pass
    return False


# ── model loading ─────────────────────────────────────────────────────────────

def is_lora(ckpt_dir: Path) -> bool:
    return (ckpt_dir / "adapter_config.json").exists()


def sanitized_adapter_dir(ckpt_dir: Path) -> Path:
    """Checkpoints were saved across sessions with different peft versions, and
    the installed peft's LoraConfig may not recognize fields a newer peft wrote
    into adapter_config.json (e.g. 'alora_invocation_tokens') — strip anything
    unrecognized rather than crash. Symlinks the (large) weight files into a temp
    dir and only materializes a filtered copy of the small config json, so this
    never touches the original checkpoint on shared storage."""
    valid_keys = set(inspect.signature(LoraConfig.__init__).parameters) - {"self"}
    cfg_path = ckpt_dir / "adapter_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    unrecognized = set(cfg) - valid_keys
    print(f"  [sanitize] {ckpt_dir.name}: unrecognized fields = {sorted(unrecognized) or 'none'}", flush=True)
    if not unrecognized:
        return ckpt_dir  # nothing unrecognized, use as-is

    filtered = {k: v for k, v in cfg.items() if k in valid_keys}
    tmp_dir = Path(tempfile.mkdtemp(prefix="peft_cfg_"))
    for item in ckpt_dir.iterdir():
        if item.name == "adapter_config.json" or not item.is_file():
            continue
        os.symlink(item.resolve(), tmp_dir / item.name)
    with open(tmp_dir / "adapter_config.json", "w") as f:
        json.dump(filtered, f)
    print(f"  [sanitize] wrote filtered config to {tmp_dir}", flush=True)
    return tmp_dir


def load_model(base_model_id: str, ckpt_dir: Path, hf_cache: str | None):
    kw = {"cache_dir": hf_cache} if hf_cache else {}

    run_dir = ckpt_dir.parent
    tok_src = str(run_dir) if (run_dir / "tokenizer.json").exists() else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True, **kw)
    tokenizer.pad_token = tokenizer.eos_token

    if is_lora(ckpt_dir):
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.float16, device_map={"": "cuda:0"},
            trust_remote_code=True, **kw,
        )
        # read exact vocab size from saved adapter weights to avoid mismatch
        adapter_path = ckpt_dir / "adapter_model.safetensors"
        if adapter_path.exists():
            with safe_open(str(adapter_path), framework="pt") as f:
                keys = list(f.keys())
                vocab_key = next((k for k in keys if (
                    ("embed_tokens" in k or "lm_head" in k) and
                    ("weight" in k or "lora_B" in k)
                )), None)
                if vocab_key:
                    ckpt_vocab = f.get_tensor(vocab_key).shape[0]
                    if ckpt_vocab != base.config.vocab_size:
                        base.resize_token_embeddings(ckpt_vocab)
        model = PeftModel.from_pretrained(base, str(sanitized_adapter_dir(ckpt_dir)))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_dir), torch_dtype=torch.float16, device_map={"": "cuda:0"},
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


# ── eval ─────────────────────────────────────────────────────────────────────

def eval_checkpoint(model, tokenizer) -> dict:
    per_prefix = defaultdict(lambda: {
        "valid": [], "json_ok": [], "truncated": [],
        "sig_ok": [], "nontrivial": [], "node_counts": [],
    })

    for prefix in tqdm(PREFIXES, desc="  prefixes", leave=False):
        prompt = build_prompt(prefix)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        expanded   = input_ids.expand(K, -1)
        attn_mask  = torch.ones(K, input_ids.shape[1], device=input_ids.device)

        with torch.no_grad():
            out = model.generate(
                expanded,
                attention_mask=attn_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        n_new     = out.shape[1] - input_ids.shape[1]
        truncated = n_new >= MAX_NEW_TOKENS
        p = per_prefix[prefix]

        for idx in range(K):
            text = tokenizer.decode(out[idx], skip_special_tokens=True)
            json_ok, tree = try_parse(text)
            valid = tree is not None
            p["json_ok"].append(json_ok)
            p["valid"].append(valid)
            p["truncated"].append(truncated)
            p["sig_ok"].append(sig_matches(tree, prefix) if valid else False)
            p["nontrivial"].append(body_nontrivial(tree) if valid else False)
            if valid:
                p["node_counts"].append(count_nodes(tree))

    all_valid      = [v for p in per_prefix.values() for v in p["valid"]]
    all_json       = [v for p in per_prefix.values() for v in p["json_ok"]]
    all_trunc      = [v for p in per_prefix.values() for v in p["truncated"]]
    all_sig        = [v for p in per_prefix.values() for v in p["sig_ok"]]
    all_nontrivial = [v for p in per_prefix.values() for v in p["nontrivial"]]
    all_nodes      = [v for p in per_prefix.values() for v in p["node_counts"]]

    pass_at_k = sum(any(p["valid"]) for p in per_prefix.values()) / len(per_prefix)

    def pct(lst): return round(100 * sum(lst) / len(lst), 2) if lst else 0.0

    return {
        "valid_rate":      pct(all_valid),
        "pass_at_k":       round(pass_at_k * 100, 2),
        "json_rate":       pct(all_json),
        "truncation_rate": pct(all_trunc),
        "sig_match_rate":  pct(all_sig),
        "nontrivial_rate": pct(all_nontrivial),
        "mean_node_count": round(sum(all_nodes) / len(all_nodes), 1) if all_nodes else 0.0,
        "n_samples":       len(all_valid),
    }


# ── checkpoint enumeration ────────────────────────────────────────────────────

def enumerate_checkpoints():
    for run_dir in sorted(CHECKPOINTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpts = sorted(
            [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not ckpts:
            continue
        for ckpt_dir in ckpts:
            yield run_dir.name, ckpt_dir.name, ckpt_dir


def parse_run_name(run_name: str) -> dict:
    bt     = "NoBT" if run_name.endswith("-nob") else "BT"
    base   = run_name.replace("-nob", "")
    parts  = base.split("-")
    family = parts[0]
    ft     = parts[1] if len(parts) > 1 else "?"
    return {"family": family, "ft": ft, "bt": bt}


def pct_of_run(ft: str, bt: str, steps: int) -> float:
    total = TOTAL_STEPS.get((ft, bt), None)
    if total is None:
        return 0.0
    return round(steps / total * 100, 1)


def tokens_seen(ft: str, steps: int) -> int:
    return steps * TRAIN_BS.get(ft, 4) * SEQ_LEN


# ── main ─────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "run", "checkpoint", "steps", "pct_of_run", "tokens_seen",
    "family", "ft", "bt", "lora",
    "valid_rate", "pass_at_k", "json_rate", "truncation_rate",
    "sig_match_rate", "nontrivial_rate", "mean_node_count", "n_samples",
    "eval_time_s", "eval_finished_at",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-cache", default=None)
    parser.add_argument("--filter",   default=None, help="only runs matching this substring")
    parser.add_argument("--shard",    default=None, help="i/n e.g. 0/4 — process i-th shard of n")
    parser.add_argument("--output",   default=str(OUTPUT_CSV))
    parser.add_argument("--k",        type=int, default=10)
    args = parser.parse_args()

    global K
    K = args.k

    checkpoints = list(enumerate_checkpoints())
    if args.filter:
        checkpoints = [(r, c, p) for r, c, p in checkpoints if args.filter in r]
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        checkpoints = checkpoints[i::n]

    print(f"Found {len(checkpoints)} checkpoints")
    for r, c, _ in checkpoints:
        print(f"  {r}/{c}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    rows = []
    if out_path.exists():
        with open(out_path) as f:
            for row in csv.DictReader(f):
                # only skip if eval actually completed (not an ERROR row)
                if row.get("valid_rate", "ERROR") not in ("ERROR", ""):
                    rows.append(row)
                    done.add((row["run"], row["checkpoint"]))
                else:
                    print(f"[redo] {row['run']}/{row['checkpoint']} — previous run errored")
        print(f"Resuming — {len(done)} already done")

    for run_name, ckpt_name, ckpt_path in checkpoints:
        if (run_name, ckpt_name) in done:
            print(f"[skip] {run_name}/{ckpt_name}")
            continue

        specs    = parse_run_name(run_name)
        steps    = int(ckpt_name.replace("checkpoint-", ""))
        pct      = pct_of_run(specs["ft"], specs["bt"], steps)
        tok_seen = tokens_seen(specs["ft"], steps)

        print(f"\n{'='*60}")
        print(f"{run_name}/{ckpt_name}  |  {pct}% of run  |  ~{tok_seen/1e6:.1f}M tokens")

        row = {
            "run": run_name, "checkpoint": ckpt_name, "steps": steps,
            "pct_of_run": pct, "tokens_seen": tok_seen, **specs, "lora": is_lora(ckpt_path),
            "valid_rate": "ERROR", "pass_at_k": "", "json_rate": "",
            "truncation_rate": "", "sig_match_rate": "", "nontrivial_rate": "",
            "mean_node_count": "", "n_samples": "", "eval_time_s": "", "eval_finished_at": "",
        }

        try:
            base_id = BASE_MODELS[specs["family"]]
            print(f"  Loading {base_id}...")
            t0 = time.time()
            model, tokenizer = load_model(base_id, ckpt_path, args.hf_cache)
            metrics = eval_checkpoint(model, tokenizer)
            elapsed = round(time.time() - t0, 1)
            row.update(metrics)
            row["eval_time_s"] = elapsed
            row["eval_finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"  valid={metrics['valid_rate']}%  pass@k={metrics['pass_at_k']}%  "
                  f"json={metrics['json_rate']}%  trunc={metrics['truncation_rate']}%  "
                  f"sig={metrics['sig_match_rate']}%  nodes={metrics['mean_node_count']}")
        except Exception:
            traceback.print_exc()
        finally:
            try: del model
            except: pass
            torch.cuda.empty_cache()

        rows.append(row)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  → {out_path}")

    print(f"\nDone. {out_path}")


if __name__ == "__main__":
    main()
