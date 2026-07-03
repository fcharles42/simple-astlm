"""Shared run-matrix config for train_multigpu.py and orchestrator.py.

Deliberately has zero heavy dependencies (no torch/transformers/etc) so
orchestrator.py can import it without pulling in the whole ML stack just to
read constants. Single source of truth for which (model, ft, nob) settings
exist and what batch_size/grad_accum/max_steps/optim they train with — the
batch_size mismatch bug (2026-07-02) happened because orchestrator.py
hand-duplicated this instead of sharing it with train_multigpu.py.

2026-07-02, reverted back to original per-run values: these MUST match
exactly what's already saved in checkpoints/<name>/ on disk (batch size,
optimizer type) or resuming breaks — optimizer.pt's state dict shape is tied
to the exact optim= used when it was written (proven the hard way: switching
paged_adamw_8bit <-> adamw_torch_fused, or changing batch size, both broke
resume with real errors this session). Values below were read directly out
of each run's trainer_state.json (`train_batch_size` field), not assumed.
"""

MODEL_HUB = {
    "base":  "Qwen/Qwen2.5-0.5B",
    "coder": "Qwen/Qwen2.5-Coder-0.5B",
}

FT_TYPES = ["2", "15", "full"]

# LoRA configs: r, alpha, target_modules (ft="full" means all-params fine-tune, no LoRA)
LORA_CFGS = {
    "2":  dict(r=16,  lora_alpha=16,  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    "15": dict(r=128, lora_alpha=128, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]),
}

GRAD_ACCUM = 4  # flat, matches every run's original training_args (effective batch = bs * 4 * world_size(4))

# (batch_size, max_steps, optim) per (ft, nob) — NOT symmetric between BT/NoBT for
# ftfull (ftfull-nob was actually trained at bs=16/3000 steps, not bs=4/12000 like
# the BT variant), confirmed via trainer_state.json audit, not assumed from docs.
_CONFIG = {
    ("2",    False): {"batch_size": 16, "max_steps": 3000,  "optim": "paged_adamw_8bit"},
    ("2",    True):  {"batch_size": 16, "max_steps": 3000,  "optim": "paged_adamw_8bit"},
    ("15",   False): {"batch_size": 8,  "max_steps": 6000,  "optim": "paged_adamw_8bit"},
    ("15",   True):  {"batch_size": 8,  "max_steps": 6000,  "optim": "paged_adamw_8bit"},
    ("full", False): {"batch_size": 4,  "max_steps": 12000, "optim": "adamw_torch_fused"},
    ("full", True):  {"batch_size": 16, "max_steps": 3000,  "optim": "adamw_torch_fused"},
}


def setting_name(model, ft, nob):
    return f"{model}-ft{ft}" + ("-nob" if nob else "")


RUNS = [
    {
        "model": model, "ft": ft, "nob": nob, "grad_accum": GRAD_ACCUM,
        **_CONFIG[(ft, nob)],
    }
    for model in MODEL_HUB
    for ft in FT_TYPES
    for nob in (False, True)
]
