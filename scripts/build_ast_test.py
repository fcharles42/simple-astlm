import os, sys, json, ast
from tqdm import tqdm
from datasets import load_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

OUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "astdump_test.jsonl")

DATASET_NAME = "bigcode/the-stack"
DATA_DIR = "data/python"
SPLIT = "train"

SKIP_FIRST = 20000          # skip training samples
MAX_TEST_SAMPLES = 2000
MAX_CHARS = 8000


def extract_function_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield ast.Module(body=[node], type_ignores=[])


def ast_to_dump(tree: ast.AST) -> str:
    return ast.dump(tree, include_attributes=False, indent=None)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    dataset = load_dataset(
        DATASET_NAME,
        data_dir=DATA_DIR,
        split=SPLIT,
        streaming=True,
    )

    written = 0
    seen = 0
    skipped = 0

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in tqdm(dataset):
            code = ex.get("content", "")
            if not code.strip():
                continue

            try:
                mod = ast.parse(code)
            except Exception:
                continue

            for fn_mod in extract_function_modules(mod):
                try:
                    dump_str = ast_to_dump(fn_mod)
                except Exception:
                    continue

                if len(dump_str) > MAX_CHARS:
                    continue

                seen += 1

                # skip first 20k to avoid overlap with training data
                if seen <= SKIP_FIRST:
                    continue

                obj = {"ast_dump": dump_str}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1

                if written >= MAX_TEST_SAMPLES:
                    break

            if written >= MAX_TEST_SAMPLES:
                break

    print(f"[OK] Wrote {written} test samples to {OUT_PATH}")
    print(f"[INFO] Skipped overlap samples: {SKIP_FIRST}")
    os._exit(0)


if __name__ == "__main__":
    main()
