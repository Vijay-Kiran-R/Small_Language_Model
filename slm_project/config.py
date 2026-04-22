# slm_project/config.py
"""
Single source of truth for all hyperparameters.
No magic numbers anywhere else in the codebase — always import from here.
"""

from dataclasses import dataclass
from typing import Tuple


# ── MODEL CONFIGURATION ──────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    # ── Vocabulary ──────────────────────────────────────────
    vocab_size: int = 32_010          # Mistral 32K base + 10 special tokens

    # ── Dimensions ──────────────────────────────────────────
    d_model:    int = 768
    n_layers:   int = 16              # ← CHANGED: 3 → 16  (production scale)
    n_heads_q:  int = 12
    n_heads_kv: int = 4               # GQA 3:1 ratio
    d_head:     int = 64
    ffn_hidden: int = 2048            # round(8/3 × 768 / 256) × 256

    # ── Context ─────────────────────────────────────────────
    max_seq_len: int = 8192
    swa_window:  int = 2048           # 25% of max_seq_len
    rope_base:   int = 500_000        # Supports up to 32K+

    # ── Architecture flags ───────────────────────────────────
    # Pattern: 3 local (SWA+RoPE) + 1 global (NoPE, full-seq), cycling × 4
    # Layers: 0,1,2→local | 3→global | 4,5,6→local | 7→global |
    #         8,9,10→local | 11→global | 12,13,14→local | 15→global
    global_layers:    Tuple[int, ...] = (3, 7, 11, 15)  # ← CHANGED: (2,) → (3,7,11,15)
    use_full_attnres: bool = True
    weight_tying:     bool = True

    # ── Regularisation ───────────────────────────────────────
    dropout_pretrain: float = 0.0
    dropout_finetune: float = 0.1    # W_O and W_down only

    # ── Special token IDs ────────────────────────────────────
    eos_token_id: int = 32_005       # <|end|>

    # ── Derived values (auto-computed) ────────────────────────
    @property
    def n_attnres_sublayers(self) -> int:
        return self.n_layers * 2      # 16 × 2 = 32

    @property
    def n_layer_outputs(self) -> int:
        return 1 + self.n_attnres_sublayers   # 1 + 32 = 33


# ── TRAINING CONFIGURATION ───────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── WSD Learning rate schedule ──────────────────────────
    peak_lr:   float = 3e-4
    min_lr:    float = 3e-5
    warmup_steps: int = 2000          # Linear ramp 0 → peak_lr

    # WSD: after warmup, hold peak_lr in stable phase.
    # Trigger decay when BOTH: val PPL plateau ≥ plateau_steps
    # AND step > min_pretrain_steps.
    # Then decay peak_lr → min_lr over ~2000 steps.
    min_pretrain_steps: int = 70_000
    plateau_steps:      int = 3_000

    # ── Batch / gradient ────────────────────────────────────
    physical_batch_seqs: int = 4      # 4 seqs fit in 12 GB
    grad_accum_steps:    int = 8      # 4×8×8192 = 262,144 tokens/step
    grad_clip:           float = 1.0

    # ── Precision ────────────────────────────────────────────
    precision: str = 'bfloat16'       # NEVER float16; no GradScaler

    # ── Logging / checkpointing ──────────────────────────────
    ckpt_freq:  int = 500             # Each checkpoint ≈ 1.5–2 GB
    log_freq:   int = 10
    eval_freq:  int = 500

    # ── MTP weight annealing ─────────────────────────────────
    mtp_weight_start: float = 0.3
    mtp_weight_end:   float = 0.1
    mtp_anneal_start: int   = 50_000
    mtp_anneal_end:   int   = 60_000

    # ── Optimizer ────────────────────────────────────────────
    adam_beta1:   float = 0.9
    adam_beta2:   float = 0.95
    adam_eps:     float = 1e-8
    weight_decay: float = 0.1
    pseudo_query_lr_multiplier: float = 2.0  # Group 3 LR = 6e-4


# ── FINE-TUNE CONFIGURATIONS ─────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    """Hyperparameters for Phase 4a (General SFT)."""
    lr:              float = 3e-5
    warmup_steps:    int   = 100
    min_epochs:      float = 1.5
    max_epochs:      float = 2.5
    physical_batch:  int   = 4
    grad_accum:      int   = 4        # 4×4×8192 = 131,072 tokens/step
    grad_clip:       float = 1.0
    dropout:         float = 0.1      # After W_O and W_down only


@dataclass
class CoTConfig:
    """Hyperparameters for Phase 4b (Chain-of-Thought FT)."""
    lr:             float = 1e-5
    warmup_steps:   int   = 100
    epochs:         float = 2.0
    physical_batch: int   = 4
    grad_accum:     int   = 4
    cot_direct_split: float = 0.5    # 50% CoT examples, 50% direct response
    # Loss computed on ALL tokens including <|think|> content AND final answer


@dataclass
class DomainFTConfig:
    """Hyperparameters for Phase 4c (Emotion + Intent Domain FT)."""
    lr:             float = 5e-6
    warmup_steps:   int   = 50
    epochs_max:     int   = 3        # 2–3 epochs max; early stop on val
    physical_batch: int   = 4
    grad_accum:     int   = 4
    # Primary metric: 150–200 hand-crafted test examples, NOT benchmarks


@dataclass
class DPOConfig:
    """Hyperparameters for Phase 4d (DPO Alignment)."""
    beta:           float = 0.05     # Start here — NOT 0.1 (too aggressive at 125M)
    lr:             float = 5e-7
    max_epochs:     int   = 1        # NEVER more than 1 epoch
    warmup_steps:   int   = 50
    # KL monitoring every 100 steps:
    #   KL < 0.2 after step 200 → may increase beta to 0.08
    #   KL > 0.3 at ANY step    → reduce to 0.02 immediately and restart


@dataclass
class LongContextConfig:
    """Phase 5.5: 8K → 16K context extension."""
    max_seq_len:    int   = 16384
    swa_window:     int   = 4096     # 25% of 16384 — read from config in FlashAttn call
    physical_batch: int   = 1        # Only 1 seq fits at 16K in 12 GB
    grad_accum:     int   = 16       # 1×16×16384 = 262,144 tokens/step = matches pretrain
    warmup_steps:   int   = 200      # CRITICAL: prevents destabilising positions
    lr:             float = 1e-5     # WSD: warmup → stable → decay to 1e-6 on plateau
    lr_min:         float = 1e-6
    total_tokens:   int   = 1_000_000_000   # 1B tokens
    precision:      str   = 'bfloat16'
    # FlashAttention local call MUST use (swa_window, 0) from config — NEVER hardcode


@dataclass
class GRPOConfig:
    """
    Phase 4b.5: Post-SFT GRPO for reasoning tasks (math + code only).
    Based on DeepSeek-R1 methodology, adapted for 125M scale.

    CRITICAL: NEVER use GRPO on writing, empathy, or open-ended tasks.
    Rule-based rewards ONLY (no neural reward model — susceptible to reward hacking).
    """
    # ── Group sampling ────────────────────────────────────────
    G:              int   = 8         # Group size per question (R1 uses 16; 8 safer at 125M)
    max_gen_len:    int   = 4096      # Max tokens per generated response
    temperature:    float = 1.0       # Sampling temperature for rollout
    top_p:          float = 0.95

    # ── GRPO objective ────────────────────────────────────────
    clip_eps:       float = 0.2       # PPO-style clipping ε (R1 first stage uses 10; 0.2 safer at 125M)
    kl_coef:        float = 0.001     # β: KL penalty from reference model
    ref_update_freq: int  = 400       # Steps before reference model = latest policy (same as R1)

    # ── Training ──────────────────────────────────────────────
    lr:             float = 3e-6      # Same as R1 (paper confirmed)
    max_steps:      int   = 700       # 500–1000; cap at 700 to avoid reward hacking
    batch_questions: int  = 32        # Unique questions per step (G rollouts each)
    warmup_steps:   int   = 50        # Short warmup for GRPO stage

    # ── Reward weights (sum = 1.0) ────────────────────────────
    accuracy_weight:  float = 0.8     # Correct final answer (rule-based verification)
    format_weight:    float = 0.1     # Uses <|think|>...</|think|> correctly
    language_weight:  float = 0.1     # Target language consistency (from R1 paper eq. 7)

    # ── Domain scope (GRPO only on these) ────────────────────
    # math_datasets: OpenR1-Math, GSM8K, MATH, MetaMathQA
    # code_datasets: Stack-Edu subset, Python-Edu problems with tests
    # NEVER: emotion, writing, open-domain QA

    # ── Safety ───────────────────────────────────────────────
    min_response_tokens: int = 64     # Ignore responses shorter than this (collapsed outputs)
    reward_hacking_kl_threshold: float = 0.5  # If avg KL > this, halt training immediately


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
