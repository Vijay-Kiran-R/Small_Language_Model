# slm_project/config.py
"""
Single source of truth for all hyperparameters.
No magic numbers anywhere else in the codebase — always import from here.
"""

from dataclasses import dataclass, field
from typing import Tuple


# ── MODEL CONFIGURATION ──────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # ── Vocabulary ──────────────────────────────────────────────────────────
    vocab_size: int = 32_010          # Mistral 32K base + 10 special tokens

    # ── Dimensions ──────────────────────────────────────────────────────────
    d_model:    int = 768
    n_layers:   int = 3               # TEST VERSION: 3  (production = 16)
    n_heads_q:  int = 12              # 12 × 64 = 768 ✓
    n_heads_kv: int = 4               # GQA 3:1 ratio — KV cache 3× smaller
    d_head:     int = 64              # Power-of-2; FlashAttention optimal
    ffn_hidden: int = 2048            # SwiGLU: round(8/3 × 768 / 256) × 256

    # ── Context ─────────────────────────────────────────────────────────────
    max_seq_len: int = 8192
    swa_window:  int = 2048           # 25 % of max_seq_len for local blocks
    rope_base:   int = 500_000        # Supports up to 32 K+ context

    # ── Architecture flags ───────────────────────────────────────────────────
    # 3-layer test: only the last layer is global
    #   (NoPE, full-seq causal attention)
    # Pattern: local, local, GLOBAL  → global_layers = (2,)
    # Production 16-layer: (3, 7, 11, 15)  — 3 local + 1 global, cycling
    global_layers:    Tuple[int, ...] = (2,)   # TEST: layer index 2 only
    use_full_attnres: bool = True
    weight_tying:     bool = True

    # ── Regularisation ───────────────────────────────────────────────────────
    dropout_pretrain: float = 0.0    # Disabled during pre-training
    dropout_finetune: float = 0.1    # W_O and W_down only during fine-tuning

    # ── Special token IDs (must match tokenizer exactly) ─────────────────────
    eos_token_id: int = 32_005       # <|end|>

    # ── Derived properties (auto-computed — do NOT change) ───────────────────

    @property
    def n_attnres_sublayers(self) -> int:
        """Number of AttnRes sublayers = n_layers × 2."""
        return self.n_layers * 2      # 3 × 2 = 6 for test model

    @property
    def n_layer_outputs(self) -> int:
        """Total layer outputs fed into MTP head = v₀ + v₁…v_{2L}."""
        return 1 + self.n_attnres_sublayers   # 1 + 6 = 7 for test model


# ── TRAINING CONFIGURATION ───────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Learning rate (WSD schedule) ────────────────────────────────────────
    peak_lr:      float = 3e-4
    min_lr:       float = 3e-5        # 10 % of peak — final decay target
    warmup_steps: int   = 2000        # Phase 1 of WSD: linear ramp 0 → peak_lr

    # WSD STABLE PHASE: hold peak_lr until plateau_steps of no val-PPL
    # improvement AND step > min_pretrain_steps, then decay over ~2000 steps.
    # ⚠️  DO NOT use a fixed total_steps — keep training if still improving.
    min_pretrain_steps: int = 70_000  # Never trigger decay before this
    plateau_steps:      int = 3000    # Steps of no improvement → trigger decay

    # ── Batch / gradient ────────────────────────────────────────────────────
    physical_batch_seqs: int   = 4    # 4 sequences fit in 12 GB VRAM
    grad_accum_steps:    int   = 8    # effective batch = 32 seqs = 262,144 tok
    grad_clip:           float = 1.0

    # ── Precision ────────────────────────────────────────────────────────────
    precision: str = 'bfloat16'       # NEVER float16; no GradScaler needed

    # ── Logging / checkpointing ──────────────────────────────────────────────
    ckpt_freq: int = 500              # Save every 500 steps
    log_freq:  int = 10
    eval_freq: int = 200

    # ── MTP weight annealing schedule ────────────────────────────────────────
    mtp_weight_start: float = 0.3
    mtp_weight_end:   float = 0.1
    mtp_anneal_start: int   = 50_000
    mtp_anneal_end:   int   = 60_000

    # ── Optimizer (AdamW) ────────────────────────────────────────────────────
    adam_beta1:   float = 0.9
    adam_beta2:   float = 0.95
    adam_eps:     float = 1e-8
    weight_decay: float = 0.1
    pseudo_query_lr_multiplier: float = 2.0   # pseudo_query LR = 2× base LR


# ── PHASE 1 SMOKE-TEST CONFIGURATION ─────────────────────────────────────────

@dataclass
class Phase1Config:
    """Overrides / gates for the 3-layer smoke test (SMALL download version)."""

    # ── Data budget (SMALL — dev/smoke test only) ─────────────────────────────
    # Production values: fineweb=240M, wikipedia=60M, total=300M
    # These small values are intentional — NEVER download the full dataset.
    total_tokens:     int = 2_500_000     # 2.5 M total (dev smoke test)
    fineweb_tokens:   int = 2_000_000     # 2 M from FineWeb-Edu
    wikipedia_tokens: int =   500_000     # 0.5 M from Wikipedia EN
    sft_examples:     int =     2_000     # SmolTalk format test only

    # ── Shard config ──────────────────────────────────────────────────────────
    shard_size: int = 200_000             # 200 K tokens per binary shard

    # ── Go / No-Go gates (values only — logic lives in trainer) ──────────────
    expected_loss_step0_lo:  float = 10.3  # log(32010) ≈ 10.37
    expected_loss_step0_hi:  float = 10.5
    expected_loss_step300:   float = 7.0   # Must be below this at step 300
    layer_outputs_expected:  int   = 7     # 1 + 2×3 for 3-layer model
    pseudo_query_norm_lo:    float = 0.001
    pseudo_query_norm_hi:    float = 1.0
