# phase1_train.py
"""
Phase 1 smoke-test training script.

Trains the 3-layer SLM on ~2.5M tokens (2M FineWeb-Edu + 500K Wikipedia).
Goal: prove the pipeline is correct, not to produce a useful model.

Usage:
  # Step 1 — download data (once)
  python -c "from slm_project.data.download import download_phase1; download_phase1()"

  # Step 2 — run training
  python phase1_train.py

Sequence length is capped at 256 for the 4GB VRAM dev machine.
On the training machine (RTX 5060 / 12 GB+) restore max_seq_len=8192
and increase physical_batch_seqs back to 4.
"""

import glob
import torch
from torch.utils.data import DataLoader

from slm_project.config import ModelConfig, TrainConfig, Phase1Config
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights
from slm_project.training.optimizer import build_optimizer
from slm_project.training.trainer import Trainer
from slm_project.data.dataset import ShardedDataset

# ── Device ────────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device : {device}")
if device == 'cuda':
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / 1e9
    print(f"GPU    : {props.name}")
    print(f"VRAM   : {vram:.1f} GB")

# ── Config ────────────────────────────────────────────────────────────────────
cfg   = ModelConfig()
tcfg  = TrainConfig()
p1cfg = Phase1Config()

# 🔥 CRITICAL for 4 GB GPU — reduce seq_len so full forward fits in VRAM
# Restore max_seq_len=8192 on the training machine.
cfg.max_seq_len = 256

# Also tighten batch size to avoid OOM on 4 GB dev GPU
tcfg.physical_batch_seqs = 1    # 1 seq × 8 accum = 8 seqs per optimizer step
tcfg.grad_accum_steps    = 8    # keep effective batch reasonable

# ── Model ─────────────────────────────────────────────────────────────────────
model = SLM(cfg, tcfg).to(device)
init_model_weights(model)

n_params = model.get_num_params()
print(f"\n3-layer SLM: {n_params:,} params ({n_params/1e6:.1f}M)")
print(f"seq_len    : {cfg.max_seq_len}")
print(f"phys_batch : {tcfg.physical_batch_seqs} seqs")
print(f"grad_accum : {tcfg.grad_accum_steps}")

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = build_optimizer(model, tcfg)

# ── Data ──────────────────────────────────────────────────────────────────────
all_shards = (
    sorted(glob.glob('data/shards/fineweb_edu_shard*.bin'))
    + sorted(glob.glob('data/shards/wikipedia_en_shard*.bin'))
)
if not all_shards:
    raise FileNotFoundError(
        "No data shards found.\n"
        "Run:  python -c \"from slm_project.data.download import download_phase1; "
        "download_phase1()\""
    )
print(f"\nData shards: {len(all_shards)} file(s) found")

dataset = ShardedDataset('data/shards/*.bin', seq_len=cfg.max_seq_len)

loader = DataLoader(
    dataset,
    batch_size=tcfg.physical_batch_seqs,
    shuffle=False,
    num_workers=0,       # 0 on Windows — avoids multiprocessing spawn issues
    pin_memory=(device == 'cuda'),
)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = Trainer(model, optimizer, tcfg, loader, device=device)

print(f"\nToken budget : {p1cfg.total_tokens:,}")
print(f"Shard total  : {dataset.total_tokens:,}")
print("=" * 60)

success = trainer.run(max_tokens=p1cfg.total_tokens)

print()
print("=" * 60)
if success:
    print("PHASE 1 COMPLETE — All Go/No-Go gates passed.")
    print("Pipeline verified. Ready to scale to 125M.")
else:
    print("PHASE 1 FAILED — Check gate diagnostics above.")
    print("Debug before scaling.")
print("=" * 60)
