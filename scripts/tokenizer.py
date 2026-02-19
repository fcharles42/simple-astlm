import os, sys, json, torch
from tqdm import tqdm
from transformers import AutoTokenizer

# This script tokenizes ast data, and PACKS multiple samples together into full 2048-token blocks to avoid padding waste.
# Saves fixed-length (2048) input_ids/labels tensors so training is faster and more GPU-efficient.

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

IN_PATH = os.path.join(REPO_ROOT, "data", "processed", "astdump.jsonl")
OUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "tokenized.pt")

MAX_SEQ_LEN = 2048


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    examples = []
    skipped = 0

    buffer = []

    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            obj = json.loads(line)
            dump_str = obj["ast_dump"]

            ids = tokenizer.encode(dump_str, add_special_tokens=False)

            if len(ids) < 8:
                skipped += 1
                continue

            # add EOS separator between samples
            ids.append(tokenizer.eos_token_id)

            buffer.extend(ids)

            # slicing out full blocks
            while len(buffer) >= MAX_SEQ_LEN:
                chunk = buffer[:MAX_SEQ_LEN]
                buffer = buffer[MAX_SEQ_LEN:]

                input_ids = chunk
                labels = chunk.copy()

                examples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                })

    # pad leftover buffer
    if len(buffer) > 8:
        if len(buffer) > MAX_SEQ_LEN:
            buffer = buffer[:MAX_SEQ_LEN]

        if len(buffer) < MAX_SEQ_LEN:
            buffer = buffer + [pad_id] * (MAX_SEQ_LEN - len(buffer))

        input_ids = buffer
        labels = [tok if tok != pad_id else -100 for tok in buffer]

        examples.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        })

    torch.save(examples, OUT_PATH)

    print(f"[OK] Saved {len(examples)} packed examples to {OUT_PATH}")
    print(f"[INFO] Skipped {skipped}")


if __name__ == "__main__":
    main()