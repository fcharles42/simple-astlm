import os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
import env  # noqa: E402

import argparse, json, ast
import torch
for _n in ["int1","int2","int3","int4","int5","int6","int7","uint1","uint2","uint3","uint4","uint5","uint6","uint7"]:
    if not hasattr(torch, _n): setattr(torch, _n, torch.int8)
import transformers
if not hasattr(transformers, "AutoProcessor"):
    from transformers.models.auto.processing_auto import AutoProcessor
    transformers.AutoProcessor = AutoProcessor

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model
from torch.utils.data import IterableDataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from run_matrix import MODEL_HUB, LORA_CFGS  # noqa: E402

_SHARED_HUB = os.environ.get("SHARED_HUB", "")

def resolve_model(name):
    """Use local snapshot from shared hub if readable, else fall back to HF download."""
    slug = "models--" + name.replace("/", "--")
    snapshots = os.path.join(_SHARED_HUB, slug, "snapshots")
    if os.path.isdir(snapshots):
        hashes = os.listdir(snapshots)
        if hashes:
            return os.path.join(snapshots, hashes[0])
    return name

DATASET_NAME = "bigcode/the-stack"
DATA_DIR     = "data/python"
MAX_CHARS    = 16000
MAX_SEQ_LEN  = 2048
AST_START    = "<ast_start>"
AST_END      = "<ast_end>"

LR           = 1e-4
BATCH_SIZE   = 4    # standalone-invocation fallback defaults; orchestrator.py always
GRAD_ACCUM   = 4    # passes explicit --batch-size/--grad-accum/--max-steps/--optim
MAX_STEPS    = 3000 # per run, sourced from run_matrix.py's per-(ft,nob) _CONFIG
LORA_DROPOUT = 0.05


class TokenTrackingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not state.is_world_process_zero:
            return
        step = state.global_step
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        logs["train/tokens_seen"] = step * args.per_device_train_batch_size * args.gradient_accumulation_steps * MAX_SEQ_LEN * world_size
        logs["train/examples_seen"] = step * args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size


class StepPrintCallback(TrainerCallback):
    """Line-based progress print every 100 steps — tqdm's carriage-return updates
    don't render usefully once piped through tee to a log file, so print plainly too."""

    def _print(self, args, state, label):
        pct = round(100 * state.global_step / state.max_steps, 1)
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        tokens_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps * MAX_SEQ_LEN * world_size
        tokens_seen = state.global_step * tokens_per_step
        tokens_total = state.max_steps * tokens_per_step
        print(f"[progress]{label} step {state.global_step}/{state.max_steps} ({pct}%) | "
              f"tokens {tokens_seen/1e6:.1f}M/{tokens_total/1e6:.1f}M", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        # first line printed — shows the actual resumed step (not 0) so a resume can
        # be confirmed at a glance instead of waiting for the next 100-step tick
        if state.is_world_process_zero:
            self._print(args, state, " STARTING at")

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or state.global_step % 100 != 0:
            return
        self._print(args, state, "")


def ast_to_json(node):
    if isinstance(node, ast.AST):
        result = {"_type": type(node).__name__}
        for field, value in ast.iter_fields(node):
            result[field] = ast_to_json(value)
        return result
    elif isinstance(node, list):
        return [ast_to_json(item) for item in node]
    elif isinstance(node, (str, int, float, bool, type(None))):
        return node
    else:
        return str(node)


def extract_function_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield ast.Module(body=[node], type_ignores=[])


class ASTStreamingDataset(IterableDataset):
    def __init__(self, tokenizer, use_boundary_tokens=True):
        self.tokenizer = tokenizer
        self.use_boundary_tokens = use_boundary_tokens

    def __iter__(self):
        import env  # re-apply in DataLoader worker processes  # noqa: F401

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        dataset = load_dataset(DATASET_NAME, data_dir=DATA_DIR, split="train", streaming=True)
        if world_size > 1:
            from datasets.distributed import split_dataset_by_node
            dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

        buffer = []
        for ex in dataset:
            code = ex.get("content", "")
            if not code.strip():
                continue
            try:
                mod = ast.parse(code)
            except Exception:
                continue
            for fn_mod in extract_function_modules(mod):
                try:
                    dump_str = json.dumps(ast_to_json(fn_mod), ensure_ascii=False)
                except Exception:
                    continue
                if len(dump_str) > MAX_CHARS:
                    continue
                text = dump_str if not self.use_boundary_tokens else f"{AST_START}{dump_str}{AST_END}"
                text = text.replace('\x00', '')
                try:
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                except Exception:
                    continue
                if len(ids) < 8:
                    continue
                ids.append(self.tokenizer.eos_token_id)
                buffer.extend(ids)
                while len(buffer) >= MAX_SEQ_LEN:
                    chunk = buffer[:MAX_SEQ_LEN]
                    buffer = buffer[MAX_SEQ_LEN:]
                    yield {"input_ids": torch.tensor(chunk, dtype=torch.long),
                           "labels":    torch.tensor(chunk, dtype=torch.long)}


def collate(batch):
    return {"input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels":    torch.stack([b["labels"]    for b in batch])}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      choices=["base", "coder"], default="base")
    p.add_argument("--ft",         choices=["2", "15", "full"], default="2")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--max-steps",  type=int, default=MAX_STEPS)
    p.add_argument("--optim", choices=["adamw_torch_fused", "paged_adamw_8bit"], default="paged_adamw_8bit",
                    help="MUST match whatever optim= wrote the checkpoint being resumed — the "
                         "optimizer.pt state dict shape is tied to it, mismatches break resume. "
                         "orchestrator.py always passes this explicitly per run_matrix.py's _CONFIG.")
    p.add_argument("--no-boundary-tokens", action="store_true",
                    help="omit <ast_start>/<ast_end> wrapping — produces the '-nob' run variant")
    return p.parse_args()


def main():
    args = parse_args()
    model_name = resolve_model(MODEL_HUB[args.model])
    run_name   = f"{args.model}-ft{args.ft}" + ("-nob" if args.no_boundary_tokens else "")
    out_dir    = os.path.join(REPO_ROOT, "checkpoints", run_name)
    os.makedirs(out_dir, exist_ok=True)

    batch_size = args.batch_size
    max_steps  = args.max_steps
    save_steps_val = max_steps // 8  # saves at 12.5/25/.../100% of total token budget

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if not args.no_boundary_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [AST_START, AST_END]})
    tokenizer.pad_token = tokenizer.eos_token

    bf16 = torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if bf16 else torch.float16, trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.ft == "full":
        model.print_trainable_parameters() if hasattr(model, "print_trainable_parameters") else None
        print(f"[full fine-tune] all {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params trainable")
    else:
        cfg = LORA_CFGS[args.ft]
        model = get_peft_model(model, LoraConfig(
            r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=LORA_DROPOUT,
            target_modules=cfg["target_modules"],
            bias="none", task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        train_dataset=ASTStreamingDataset(tokenizer, use_boundary_tokens=not args.no_boundary_tokens),
        data_collator=collate,
        callbacks=[TokenTrackingCallback(), StepPrintCallback()],
        args=TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=max_steps,
            learning_rate=LR,
            bf16=bf16, fp16=not bf16,
            logging_steps=50,
            save_steps=save_steps_val,
            save_total_limit=8,
            report_to="none",
            optim=args.optim,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            ddp_find_unused_parameters=False,
        ),
    )

    resume = True if any(
        f.name.startswith("checkpoint-") for f in os.scandir(out_dir) if f.is_dir()
    ) else None
    trainer.train(resume_from_checkpoint=resume)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[OK] Saved to {out_dir}")


if __name__ == "__main__":
    main()
