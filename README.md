# SLM — Small Language Model: Complete Technical Reference

> **Who this is for:** Beginners who want to understand how language models are built from scratch,
> AND experienced ML engineers / PhD researchers who need every architectural and implementation
> decision explained with full precision. No assumptions are made. Nothing is glossed over.

---

## TABLE OF CONTENTS

1. [What This Project Is](#1-what-this-project-is)
2. [Directory Structure](#2-directory-structure)
3. [The Vocabulary & Tokenizer](#3-the-vocabulary--tokenizer)
4. [Config — The Single Source of Truth](#4-config--the-single-source-of-truth)
5. [Model Architecture Overview](#5-model-architecture-overview)
6. [RMSNorm](#6-rmsnorm)
7. [RoPE — Rotary Position Embedding](#7-rope--rotary-position-embedding)
8. [AttnRes — Full Attention Residuals](#8-attnres--full-attention-residuals)
9. [Grouped Query Attention + QK-Norm + FlashAttention](#9-grouped-query-attention--qk-norm--flashattention)
10. [IHA-Global — Interleaved Head Attention](#10-iha-global--interleaved-head-attention) ← **NEW**
11. [SwiGLU Feed-Forward Network](#11-swiglu-feed-forward-network)
12. [TransformerBlock — Wiring Everything Together](#12-transformerblock--wiring-everything-together)
13. [MTP Head — Multi-Token Prediction](#13-mtp-head--multi-token-prediction)
14. [Full Model — SLM Class](#14-full-model--slm-class)
15. [Weight Initialisation](#15-weight-initialisation)
16. [Data Pipeline — Download & Shards](#16-data-pipeline--download--shards)
17. [ShardedDataset — Reading Shards During Training](#17-shardeddataset--reading-shards-during-training)
18. [Optimizer — Three Param Groups](#18-optimizer--three-param-groups)
19. [LR Schedule — WSD (Warmup → Stable → Decay)](#19-lr-schedule--wsd-warmup--stable--decay)
20. [Trainer Loop — The Training Engine](#20-trainer-loop--the-training-engine)
21. [Phase 1 Go/No-Go Gates](#21-phase-1-gonogo-gates)
22. [SFT Format & Loss Masking](#22-sft-format--loss-masking)
23. [GRPO — Phase 4b.5 Reasoning RL](#23-grpo--phase-4b5-reasoning-rl) ← **NEW**
24. [Complete Training Pipeline — End to End](#24-complete-training-pipeline--end-to-end)
25. [How to Run Everything](#25-how-to-run-everything)

---

## 1. What This Project Is

This codebase builds a **Small Language Model (SLM)** from scratch in PyTorch.
It is not a fine-tuning wrapper around an existing model. Every component —
the tokenizer, the attention mechanism, the residual connections, the optimizer
setup, the data pipeline — is written and understood from first principles.

### Why build from scratch instead of fine-tuning?

Fine-tuning gives you a model that already knows language. Building from scratch
gives you complete control over the architecture, the data, the training dynamics,
and the deployment target. This matters when:

- You want a model that runs on a specific hardware budget (e.g. a phone or
  embedded device) and no existing model fits
- You need custom architecture features (like AttnRes, described later)
- You want to understand exactly what every parameter does

### Two-phase design

The project is built in two phases:

|          Phase          | Model Size     | Layers    |                    Purpose                          |
|---                      |---             |---        |                                                  ---|
| Phase 1 (this codebase) | ~44M params    | 3 layers  | **Smoke test** — proves the full pipeline works     |
| Phase 5 (scale-up)      | ~125.9M params | 16 layers | Production model — identical code, different config |

The 3-layer model is not useful for real language tasks. It exists purely to
validate that the pipeline (tokenizer → data → model → optimizer → trainer →
checkpoint) is correct before committing GPU-hours to the full run.
**Every line of code in Phase 1 runs identically in Phase 5.**

### Architecture novelties

This model contains three components that go beyond a standard transformer:

**AttnRes** (Full Attention Residuals) — replaces the fixed `+x` residual
connection with learned softmax attention over ALL previous layer outputs.
By layer 16, instead of signal being diluted 1/16th, the model can
dynamically weight every previous layer's representation. Section 8 explains
this in full.

**IHA-Global** (Interleaved Head Attention, $P=2$) — integrated into the 4
global NoPE layers. Each of the 12 query heads now computes $P=2$
pseudo-queries by linearly mixing all head outputs, doubling the number of
distinct attention patterns available without increasing depth. The paper
(arXiv:2602.21371) proves IHA can realise polynomial-order reasoning filters
in the same number of layers as exponential-order MHA. Adds only 2,560
parameters total. Sections 10 and 15 explain this in detail.

**GRPO** (Group Relative Policy Optimization, DeepSeek-R1 methodology) —
Phase 4b.5 of the training pipeline. After SFT, the model is refined on
math and code tasks using rule-based rewards (no neural reward models).
This teaches the model to self-reflect and verify answers. Section 23
explains the full implementation.

### Hardware targets

| Machine | GPU | VRAM | Role |
|---|---|---|---|
| Dev laptop | RTX 3050 Laptop | 4.3 GB | Code, small tests only |
| Training machine | RTX 5060 Ti (or better) | 12 GB+ | Full Phase 1 and Phase 5 |

Everything in this codebase is written to run safely on the dev machine
(seq_len=256, batch=1) and scale without code changes to the training machine
(seq_len=8192, batch=4).

---

## 2. Directory Structure

```
v:\code\SLM\
│
├── slm_project/                  ← All source code lives here
│   ├── config.py                 ← Single source of truth for ALL hyperparameters
│   │                               (ModelConfig, TrainConfig, GRPOConfig, ...)
│   ├── tokenizer_utils.py        ← Build, save, load, and verify the tokenizer
│   │
│   ├── model/                    ← Every model component
│   │   ├── rms_norm.py           ← RMSNorm (used everywhere)
│   │   ├── rope.py               ← Rotary Position Embedding
│   │   ├── attn_res.py           ← AttnRes (novel residual mechanism)
│   │   ├── attention.py          ← GQA + QK-Norm + FlashAttn
│   │   │                           + IHAGlobalAttention (P=2, global layers only)
│   │   ├── ffn.py                ← SwiGLU Feed-Forward Network
│   │   ├── block.py              ← TransformerBlock — routes IHA vs GQA per layer
│   │   ├── mtp.py                ← Multi-Token Prediction head
│   │   ├── model.py              ← Full SLM class (assembles everything)
│   │   ├── init_weights.py       ← 3-pass weight init (standard → zero_pq → IHA_id)
│   │   └── generate.py           ← Greedy / sampling decode for inference
│   │
│   ├── data/                     ← Data pipeline
│   │   ├── download.py           ← Stream and tokenise from HuggingFace
│   │   ├── download_pretrain.py  ← Full 20B token pretraining download
│   │   ├── dataset.py            ← ShardedDataset (reads binary shards)
│   │   └── phase55_data.py       ← Long-context 16K data for Phase 5.5
│   │
│   └── training/                 ← Training engine
│       ├── optimizer.py          ← AdamW with 3 mandatory param groups
│       ├── lr_schedule.py        ← WSD learning rate schedule
│       ├── trainer.py            ← Pretraining loop + WSD + checkpoint pruning
│       ├── finetune.py           ← SFTTrainer for Phases 4a/4b/4c
│       ├── dpo_trainer.py        ← DPO trainer for Phase 4d
│       └── grpo_trainer.py       ← GRPO trainer for Phase 4b.5 (NEW)
│
├── tests/                        ← Automated test suite (22 tests)
│   ├── test_tokenizer.py         ← Tokenizer verification (6 tests)
│   ├── test_attnres.py           ← AttnRes correctness (5 tests)
│   ├── test_model.py             ← Full model assembly — asserts 125,931,008 params
│   ├── test_iha.py               ← IHA-Global: shape, init, grad flow (NEW)
│   ├── test_grpo.py              ← GRPO reward functions (NEW)
│   ├── test_backward_ckpt.py     ← Backward + checkpointing
│   └── test_trainer.py           ← Trainer dry-run (7 tests)
│
├── tokenizer/                    ← Saved tokenizer files (auto-generated)
│   ├── tokenizer.json            ← Full vocabulary (32,010 tokens)
│   ├── tokenizer_config.json     ← Model type, special token config
│   └── special_tokens_map.json   ← Mapping of special token names to IDs
│
├── data/
│   └── shards/                   ← Pre-tokenised binary data (auto-generated)
│       ├── fineweb_edu_shard0000.bin   ← 200K tokens, uint16, ~391 KB each
│       └── ...                         ← 10 shards = 2M tokens (test set)
│
├── checkpoints/                  ← Saved model checkpoints (auto-generated)
│   └── step_NNNNNNN.pt           ← Model + optimizer + data position + IHA state
│
├── eval/
│   └── domain_eval.py            ← Domain-specific evaluation metrics
│
├── pretrain_stage1.py            ← Stage 1: 0–8B token pretraining
├── pretrain_stage2.py            ← Stage 2: 8B–16B token pretraining
├── pretrain_stage3.py            ← Stage 3: 16B–20B (WSD decay phase)
├── finetune_4a_sft.py            ← Phase 4a: General SFT
├── finetune_4b_cot.py            ← Phase 4b: CoT SFT (with <|think|> tokens)
├── finetune_4c_domain.py         ← Phase 4c: Domain fine-tuning
├── phase4b5_grpo.py              ← Phase 4b.5: GRPO reasoning RL (NEW)
├── phase55_extend.py             ← Phase 5.5: Long-context extension to 16K
├── mini_e2e_pipeline.py          ← Full pipeline verification on test shards
├── final_verify.py               ← Final production-readiness check
├── health_check_pretrain.py      ← Health diagnostics during pretraining
├── sft_format_test.py            ← Gate 7: SFT chat template + loss mask
├── stage0_verify.py              ← Stage 0: environment check
└── README.md                     ← This file
```

### Why is the code structured this way?

Each subdirectory corresponds to a clean separation of concerns:

- **`model/`** — pure PyTorch `nn.Module` classes. No data loading, no training
  logic. Each file is independently testable.
- **`data/`** — everything about getting tokens from disk into the model. No
  model code here.
- **`training/`** — the optimizer, schedule, and training loops. Three separate
  trainers for three different regimes: pretraining (`trainer.py`), supervised
  fine-tuning (`finetune.py`), and reinforcement learning (`grpo_trainer.py`).
- **`config.py`** — sits at the root of `slm_project/` because everything imports
  from it. It is the single source of truth. There are **zero magic numbers**
  anywhere else in the codebase.
- **`tests/`** — every stage has a corresponding test that must pass before the
  next stage is attempted. **22 tests, 0 skips**.

---

*Part 1 complete.*

---

## 3. The Vocabulary & Tokenizer

**File:** `slm_project/tokenizer_utils.py`

### What a tokenizer does

Before a language model can process text, text must become numbers. A tokenizer
splits raw text into *tokens* — chunks that can be single characters, word
fragments, whole words, or special symbols — and maps each chunk to an integer ID.
The model never sees letters. It only ever sees integers.

Example (approximate):
```
"Hello world" → ["Hello", "▁world"] → [15043, 1526]
```

The model learns to predict the next integer in a sequence. After training,
you reverse the mapping: integers → tokens → text.

### Why Mistral's tokenizer?

The Mistral-7B tokenizer uses **SentencePiece BPE** (Byte-Pair Encoding),
which was trained on a large multilingual corpus. Its 32,000 base tokens cover
English very efficiently (most common words are single tokens). We reuse it
rather than training our own because:

1. Training a tokenizer requires billions of tokens and significant compute
2. Vocabulary quality directly affects model quality — a poor tokenizer
   means the model wastes capacity on poor splits
3. Starting from Mistral's vocabulary makes our model compatible with
   Mistral's pre-trained embeddings if we ever want to use them for
   warm-starting

### The 10 special tokens (IDs 32000–32009)

We extend the 32,000 base tokens with exactly 10 special tokens. These are
**structural delimiters** — they mark the boundaries of turns in a conversation,
tool calls, and reasoning traces. The model must learn what follows each one.

```
ID 32000  <|system|>      — starts a system-level instruction turn
ID 32001  <|user|>        — starts a user message turn
ID 32002  <|assistant|>   — starts an assistant response turn
ID 32003  <|think|>       — starts an internal chain-of-thought block
ID 32004  <|/think|>      — ends the chain-of-thought block
ID 32005  <|end|>         — END OF SEQUENCE (EOS) — model must emit this to stop
ID 32006  <|tool_call|>   — starts a tool/function call
ID 32007  <|/tool_call|>  — ends a tool call
ID 32008  <|tool_result|> — starts the result returned by a tool
ID 32009  <|/tool_result|>— ends the tool result block
```

**Why these exact IDs matter:**
The model config has `eos_token_id = 32005`. If `<|end|>` was accidentally
assigned a different ID (e.g. 32006), the model would never stop generating
during inference. The verification loop below exists to catch this.

### The co-verification loop — why it is mandatory

SentencePiece can split tokens that weren't in the original training vocabulary
into multiple sub-pieces. For example, if `<|system|>` got split, encoding it
might return `[32000, 45, 12]` instead of `[32000]`. Then during training, the
model would see three separate tokens for what should be one structural delimiter.
The loss mask (Section 21) would be wrong. Inference would break.

The verification loop checks **every single special token** before any model
code runs:

```python
# From tokenizer_utils.py — runs for each of the 10 special tokens
ids = tokenizer.encode(token, add_special_tokens=False)
assert ids == [expected_id], f"FAIL: {token!r} → {ids}, expected [{expected_id}]"
```

If any assertion fails, training cannot proceed. Fix the tokenizer first.

### How build_and_save_tokenizer() works — step by step

```python
# Step 1: Download just the tokenizer files from HuggingFace (~500 KB)
# No model weights are downloaded — just the vocabulary files.
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
# tokenizer now has vocab_size = 32,000

# Step 2: Add our 10 special tokens
num_added = tokenizer.add_special_tokens(
    {'additional_special_tokens': SPECIAL_TOKENS}
)
assert num_added == 10  # if < 10, some already existed in base vocab
# tokenizer now has vocab_size = 32,010

# Step 3: Wire EOS explicitly
tokenizer.eos_token    = '<|end|>'
tokenizer.eos_token_id = 32005
# Without this, the tokenizer's default EOS would be Mistral's </s> (ID 2)

# Step 4: Save to disk — creates tokenizer/, tokenizer.json, etc.
tokenizer.save_pretrained('tokenizer/')
```

### How load_tokenizer() works

Every script that needs the tokenizer calls:
```python
from slm_project.tokenizer_utils import load_tokenizer
tok = load_tokenizer()
```

This loads from the local `tokenizer/` directory — no internet required after
the first build. The download is a one-time operation.

### The tokenizer files on disk

After `build_and_save_tokenizer()`, the `tokenizer/` directory contains:

| File | Contents |
|---|---|
| `tokenizer.json` | Full vocabulary: 32,010 entries, merge rules, special token config |
| `tokenizer_config.json` | Model type, padding, EOS token name |
| `special_tokens_map.json` | Maps "eos_token" → `<\|end\|>`, etc. |

`tokenizer.json` is ~3.5 MB. It is the only file actually needed at runtime.

### What happens during tokenisation of a full conversation

During SFT (Supervised Fine-Tuning), a conversation looks like this on disk:

```
<|system|>You are a helpful assistant.<|end|>
<|user|>What is 2+2?<|end|>
<|assistant|>4<|end|>
```

After tokenisation:
```
[32000, 1976, 460, 264, 10297, 13892, 29889, 32005,   ← system turn
 32001, 1724, 338, 29871, 29906, 29974, 29906, 29973, 32005,  ← user turn
 32002, 29946, 32005]                                  ← assistant turn
```

The **loss mask** (Section 21) then marks which of these token IDs count toward
the training loss. Only assistant content + EOS = 1. System and user = 0.

### The vocab_size in config

`ModelConfig.vocab_size = 32_010` must match the tokenizer exactly. If you
change the number of special tokens, you must update this value AND re-run the
verification. The embedding table is `[vocab_size, d_model]` — if vocab_size
is wrong, you get a shape mismatch crash when you try to embed a token with
ID ≥ the table size.

---

*Part 2 complete.*

---

## 4. Config — The Single Source of Truth

**File:** `slm_project/config.py`

### The philosophy: zero magic numbers

Every number in this codebase that affects model behaviour or training dynamics
lives in `config.py`. Nowhere else. When you read `attention.py` and see
`cfg.d_head`, you know exactly where `64` came from and you can change it in
one place. When you read `trainer.py` and see `tcfg.grad_clip`, you know it's
`1.0` and why. This makes the codebase safe to modify and impossible to
accidentally get out of sync.

There are **three config dataclasses**:

---

### ModelConfig — architecture hyperparameters

```python
@dataclass
class ModelConfig:
    # ── Vocabulary ────────────────────────────────────────
    vocab_size:  int   = 32_010   # Embedding table rows. MUST match tokenizer.

    # ── Dimensions ────────────────────────────────────────
    d_model:     int   = 768      # Hidden dimension — every tensor's last dim.
    n_layers:    int   = 16       # Production: 16. Smoke-test: 3.
    n_heads_q:   int   = 12       # Query heads. 12 × 64 = 768 = d_model. ✓
    n_heads_kv:  int   = 4        # Key/Value heads. GQA 3:1 ratio.
    d_head:      int   = 64       # Per-head dimension. Power of 2 (FlashAttn).
    ffn_hidden:  int   = 2048     # FFN intermediate width. Derived below.
    max_seq_len: int   = 8192     # Max tokens per sequence.
    swa_window:  int   = 2048     # SWA window = 25% of max_seq_len.
    rope_base:   int   = 500_000  # RoPE frequency base. Explained in Sec 7.

    # ── Layer type assignment ─────────────────────────────
    # Global layers use IHAGlobalAttention (NoPE, full sequence).
    # All other layers use GroupedQueryAttention (SWA + RoPE).
    global_layers: tuple = (3, 7, 11, 15)  # Production: one every 4 layers.

    # ── Regularisation ────────────────────────────────────
    dropout_pretrain: float = 0.0  # No dropout during pre-training.
    dropout_finetune: float = 0.1  # Applied to W_O and W_down during SFT only.

    # ── Generation ────────────────────────────────────────
    eos_token_id: int  = 32_005   # Must match tokenizer ID for <|end|>.
```

**Why these specific numbers:**

- **`d_model = 768`** — same as GPT-2 Medium and BERT-base. Well understood,
  fits in 4 GB VRAM for testing, and scales cleanly to 125M parameters at
  n_layers=16.

- **`n_heads_q=12, n_heads_kv=4`** — Grouped Query Attention (GQA) with 3:1
  ratio. 4 KV heads instead of 12 means the KV cache (the memory that grows
  with sequence length during inference) is **3× smaller**. This is the same
  trick used by Mistral and LLaMA 3. Section 9 explains GQA fully.

- **`d_head = 64`** — derived: `d_model / n_heads_q = 768 / 12 = 64`. Must be
  a power of 2 for FlashAttention's internal tiling to work correctly.

- **`ffn_hidden = 2048`** — derived formula: `round(8/3 × d_model / 256) × 256`.
  For d_model=768: `8/3 × 768 = 2048.0` exactly. This formula comes from the
  SwiGLU paper and targets a specific compute-to-parameter ratio. The `/ 256`
  then `× 256` rounds to the nearest multiple of 256, which is optimal for
  GPU tensor cores.

- **`swa_window = 2048`** — Sliding Window Attention window = 25% of max_seq_len.
  Local blocks only attend to the nearest 2,048 tokens; global blocks see all.
  This is the same sliding window used in Mistral-7B.

- **`rope_base = 500_000`** — critical for long context. The standard value
  from the original RoPE paper is 10,000. At base 10,000, position encodings
  repeat (aliase) at ~8,000 tokens. At base 500,000, they stay unique up to
  32K+ tokens. This is the value used by LLaMA 3. Section 7 explains why.

- **`global_layers = (3, 7, 11, 15)`** — production layout: one global NoPE
  layer every 4 layers. These are the 4 layers where `IHAGlobalAttention`
  replaces `GroupedQueryAttention`. Local layers (all others) use SWA + RoPE
  and are NEVER given IHA — doing so would collapse the SWA window from 2,048
  to ~14 tokens with P=2. Section 10 covers this constraint in detail.

**Derived properties (auto-computed, read-only):**

```python
@property
def n_attnres_sublayers(self) -> int:
    return self.n_layers * 2        # 3 × 2 = 6 AttnRes instances

@property
def n_layer_outputs(self) -> int:
    return 1 + self.n_attnres_sublayers  # 1 + 6 = 7 total layer_outputs
```

These are properties (not fields) so they cannot be accidentally overridden.
The value `7` (for the 3-layer model) is checked by an assertion in
`model.forward()` on every single forward pass.

---

### TrainConfig — training dynamics

```python
@dataclass
class TrainConfig:
    # Learning rate
    peak_lr:      float = 3e-4    # Maximum LR during stable phase
    min_lr:       float = 3e-5    # 10% of peak — floor during decay phase
    warmup_steps: int   = 2000    # Steps to linearly ramp from 0 → peak_lr

    # When to stop training (WSD — no fixed total_steps)
    min_pretrain_steps: int = 70_000  # NEVER trigger decay before this
    plateau_steps:      int = 3000    # Steps of no val-PPL improvement → decay

    # Batching
    physical_batch_seqs: int   = 4    # Sequences per GPU step (12GB machine)
    grad_accum_steps:    int   = 8    # Accumulate before one optimizer step
    grad_clip:           float = 1.0  # Max gradient norm

    # Precision
    precision: str = 'bfloat16'   # NEVER float16. No GradScaler needed.

    # Logging & checkpointing
    ckpt_freq: int = 500   # Save checkpoint every N optimizer steps
    log_freq:  int = 10    # Print metrics every N steps
    eval_freq: int = 200   # Run validation every N steps

    # MTP weight annealing
    mtp_weight_start: float = 0.3    # MTP loss weight from step 0
    mtp_weight_end:   float = 0.1    # MTP weight after annealing
    mtp_anneal_start: int   = 50_000 # Step to begin annealing
    mtp_anneal_end:   int   = 60_000 # Step to finish annealing

    # Optimizer
    adam_beta1:   float = 0.9   # AdamW momentum (standard)
    adam_beta2:   float = 0.95  # AdamW second moment (slightly higher than 0.999)
    adam_eps:     float = 1e-8  # Numerical stability in Adam denominator
    weight_decay: float = 0.1   # L2 regularisation on linear weights
    pseudo_query_lr_multiplier: float = 2.0  # AttnRes gets 2× base LR
```

**Key decisions explained:**

- **`physical_batch_seqs=4, grad_accum_steps=8`** — effective batch =
  4 × 8 × 8192 = **262,144 tokens per step**. This matches the "chinchilla
  optimal" batch size range for models in the 100M parameter range. On the
  dev machine, `physical_batch_seqs=1` and `seq_len=256` is used instead.

- **`bfloat16`, never `float16`** — bfloat16 has the same dynamic range as
  float32 (8 exponent bits) but half the precision (7 mantissa bits vs 23).
  float16 has only 5 exponent bits, causing overflow/underflow with gradients,
  which requires GradScaler to compensate. bfloat16 does not need GradScaler.
  Using GradScaler with bfloat16 is wrong and adds noise.

- **`adam_beta2=0.95` not 0.999`** — higher beta2 (closer to 1) means the
  second moment estimate is slower to change, which smooths out gradient
  variance. For large batch training with strong regularisation, 0.95 provides
  more aggressive adaptation than 0.999. Used by GPT-3 and PaLM.

- **`weight_decay=0.1`** — applied only to 2D+ tensors (Linear weights,
  Embedding). Never to RMSNorm gammas or biases (those are in Group 2 of the
  optimizer). Section 17 covers this in detail.

- **`pseudo_query_lr_multiplier=2.0`** — AttnRes pseudo_query parameters get
  2× the base learning rate. This is because pseudo_query starts at zero and
  needs to move faster to break symmetry. Section 8 and 17 explain this.

---

### Phase1Config — smoke-test data gates

```python
@dataclass
class Phase1Config:
    total_tokens:     int = 2_500_000   # Stop training at 2.5M tokens
    fineweb_tokens:   int = 2_000_000   # Stream 2M from FineWeb-Edu
    wikipedia_tokens: int =   500_000   # Stream 500K from Wikipedia EN
    sft_examples:     int =     2_000   # SmolTalk examples for Gate 7 test
    shard_size:       int =   200_000   # 200K tokens per .bin shard file

    # Go/No-Go gate thresholds — checked at step 0 and step 300
    expected_loss_step0_lo:  float = 10.3   # log(32010) ≈ 10.37
    expected_loss_step0_hi:  float = 10.5
    expected_loss_step300:   float = 7.0    # Must beat this by step 300
    layer_outputs_expected:  int   = 7      # 1 + 2 × 3 for 3-layer model
    pseudo_query_norm_lo:    float = 0.001
    pseudo_query_norm_hi:    float = 1.0
```

**Important:** `total_tokens = 2.5M` is intentionally tiny — this is the
dev-machine smoke test. The production Phase 1 run uses 20B tokens. The
`fineweb_tokens` and `wikipedia_tokens` values are **budgets**, not full
downloads. The streamer stops reading as soon as the budget is hit.

---

### GRPOConfig — Phase 4b.5 reinforcement learning  *(NEW)*

```python
@dataclass
class GRPOConfig:
    # ── Sampling ─────────────────────────────────────────────────────
    G:               int   = 8       # Group size — outputs sampled per question
    max_gen_len:     int   = 4096    # Max tokens per generated response
    temperature:     float = 0.9     # Sampling temperature
    top_p:           float = 0.95    # Nucleus sampling threshold

    # ── Loss ─────────────────────────────────────────────────────────
    clip_eps:        float = 0.2     # PPO clip range (same as DeepSeek-R1)
    kl_coef:         float = 0.001   # KL penalty coefficient

    # ── Training ─────────────────────────────────────────────────────
    max_steps:       int   = 700     # Hard ceiling — stop even if reward < 0.75
    batch_questions: int   = 4       # Questions per optimizer step
    lr:              float = 5e-7    # Very low LR — RL is sensitive

    # ── Reward weights ────────────────────────────────────────────────
    w_accuracy:      float = 1.0     # Weight for correct answer reward
    w_format:        float = 0.2     # Weight for <|think|> format reward
    w_language:      float = 0.1     # Weight for language consistency reward

    # ── Safety ───────────────────────────────────────────────────────
    reward_hacking_kl_threshold: float = 0.5   # Halt if KL exceeds this
    early_stop_reward:           float = 0.75  # Stop early if mean reward ≥ this
```

**Key GRPO design decisions:**

- **`G=8` (not 16):** At 125M scale the model needs less exploration diversity
  than DeepSeek-R1-Zero (671B). 8 samples per question gives sufficient
  advantage signal with reasonable memory cost.

- **`max_gen_len=4096` (not 32,768):** The 125M model cannot reliably produce
  longer reasoning chains. Setting this higher wastes compute and increases
  reward hacking risk.

- **`reward_hacking_kl_threshold=0.5`:** If KL-divergence between the current
  policy and the reference model exceeds 0.5, training halts immediately.
  The reference model is a frozen copy of the SFT checkpoint — the policy
  must not drift too far from it or it loses the SFT quality.

- **`early_stop_reward=0.75`:** If `mean_reward` reaches 0.75+ before 700
  steps, stop early. More steps beyond peak = reward hacking, not learning.
  700 steps is a **ceiling**, not a target.

- **`w_format=0.2, w_language=0.1`:** These are small compared to
  `w_accuracy=1.0`. The accuracy reward drives the real learning. Format
  and language consistency are regularisers — they prevent degenerate
  outputs but don't dominate the update.

---

There are now **five config dataclasses** in total:
`ModelConfig`, `TrainConfig`, `Phase1Config`, `FinetuneConfig` (SFT/CoT), and `GRPOConfig`.
Every training phase draws exclusively from these — zero magic numbers elsewhere.

## 5. Model Architecture Overview

**File:** `slm_project/model/model.py` (assembly)

Before diving into individual components, here is the complete picture of what
the model does to one batch of token sequences.

### Input → Output

```
Input:   token IDs  [B, T]         (B=batch size, T=sequence length)
Output:  logits     [B, T, 32010]  (a probability score for each vocab token)
         loss       scalar          (if labels provided)
```

### The complete forward pass, step by step

```
Step 1 — EMBED
  token IDs [B, T]
      ↓  nn.Embedding(32010, 768)
  x = v₀   [B, T, 768]
  layer_outputs = [v₀]             ← grows by 2 with every block

Step 2 — BLOCKS (3 times, once per TransformerBlock)
  ┌─────────────────────────────────────────────────────────┐
  │  For each block b = 0, 1, 2:                            │
  │                                                         │
  │  ATTENTION SUB-LAYER:                                   │
  │    AttnRes([v₀..v_{2b}])    → h          [B,T,768]     │
  │    RMSNorm(h)                → h_normed  [B,T,768]     │
  │    GroupedQueryAttention(h_normed) → attn_out [B,T,768] │
  │    layer_outputs.append(attn_out)   → v_{2b+1}         │
  │                                                         │
  │  FFN SUB-LAYER:                                         │
  │    AttnRes([v₀..v_{2b+1}])  → h2         [B,T,768]    │
  │    RMSNorm(h2)               → h2_normed [B,T,768]    │
  │    SwiGLUFFN(h2_normed)     → mlp_out   [B,T,768]     │
  │    layer_outputs.append(mlp_out)    → v_{2b+2}         │
  └─────────────────────────────────────────────────────────┘

  After block 0: layer_outputs = [v₀, v₁, v₂]      (len=3)
  After block 1: layer_outputs = [v₀, v₁, v₂, v₃, v₄] (len=5)
  After block 2: layer_outputs = [v₀..v₆]           (len=7)

Step 3 — FINAL NORM
  RMSNorm(v₆)  → final_h   [B, T, 768]

Step 4 — MAIN LOGITS (weight-tied)
  final_h @ embedding.weight.T  → logits  [B, T, 32010]

Step 5 — MTP LOGITS (weight-tied, same embedding table)
  MTPHead(v₆, embedding.weight) → mtp_logits  [B, T, 32010]

Step 6 — LOSSES (only if labels provided)
  main_loss = cross_entropy(logits[:,:-1],  labels[:,1:])   predict t+1
  mtp_loss  = cross_entropy(mtp_logits[:,:-2], labels[:,2:]) predict t+2
  total_loss = main_loss + mtp_weight(step) × mtp_loss
```

### Parameter count breakdown (3-layer model, ~44M total)

| Component | Formula | Count |
|---|---|---|
| Embedding table | 32,010 × 768 | **24,583,680** |
| Attention per layer (×3) | W_Q+W_K+W_V+W_O+QK-Norm | 1,573,888 |
| FFN per layer (×3) | W_gate+W_up+W_down | 4,718,592 |
| RMSNorm gammas (×7) | 7 × 768 | 5,376 |
| AttnRes pseudo_queries (×6) | 6 × 768 | 4,608 |
| AttnRes key_norms (×6) | 6 × 768 | 4,608 |
| MTP head | W_mtp (768×768) + RMSNorm | 590,592 |
| **Output projection** | *tied to embedding — no extra params* | 0 |
| **MTP vocab projection** | *tied to embedding — no extra params* | 0 |
| **TOTAL** | | **~44,066,304** |

**Note on weight tying:** The output projection (`logits = final_h @ embedding.weight.T`)
and the MTP vocab projection both reuse `embedding.weight`. This means the
same 24.6M parameters serve three purposes simultaneously:
1. Look up embeddings for input tokens
2. Project hidden states to vocabulary logits
3. Project MTP hidden states to vocabulary logits

This is standard practice (GPT-2, LLaMA, etc.) and reduces parameter count
significantly while actually helping quality — the embedding and unembedding
directions become consistent.

### Block type assignment

| Block index | Type | Attention | Position encoding |
|---|---|---|---|
| 0 | LOCAL | SWA (window=2048) | RoPE |
| 1 | LOCAL | SWA (window=2048) | RoPE |
| 2 | GLOBAL | Full sequence | NoPE (no position encoding) |

In the 16-layer production model, the pattern repeats every 4 layers:
local, local, local, GLOBAL — at indices 0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15.
Global layers are at indices 3, 7, 11, 15.

The reason for mixing local and global: local SWA blocks are cheaper (they
only compute attention over 2,048 tokens instead of all 8,192), but they cannot
propagate information from far away in the sequence. Global blocks can, but are
more expensive. The mix gets the best of both worlds.

---

*Parts 3 & 4 complete.*

---

## 6. RMSNorm

**File:** `slm_project/model/rms_norm.py`

### What normalisation does and why we need it

During training, the values flowing through a neural network can grow or shrink
dramatically as they pass through layer after layer of multiplications. If a
hidden state has values like `[1500, -2000, 800]`, the gradient through it will
be enormous and training explodes. Normalisation rescales these values back to a
stable range before each sub-layer.

### LayerNorm vs RMSNorm — what's the difference?

**LayerNorm** (used in GPT-2, BERT):
```
LayerNorm(x) = (x - mean(x)) / sqrt(variance(x) + ε) × γ + β
```
This requires computing both the mean and the variance across the feature
dimension, then subtracting the mean (re-centring), then scaling.

**RMSNorm** (used in LLaMA, Mistral, this model):
```
RMSNorm(x) = x / sqrt(mean(x²) + ε) × γ
```
This skips the mean subtraction entirely. It only computes the **Root Mean
Square** of the values and scales by that. No β (bias) parameter needed.

**Why RMSNorm?**
1. **30% faster** — one fewer reduction operation (no mean subtraction)
2. **Fewer parameters** — no β bias term (saves `dim` floats per instance)
3. **Empirically equivalent quality** — the re-centring step of LayerNorm turns
   out to not be important; the re-scaling is what matters
4. **Standard in modern LLMs** — LLaMA, Mistral, Falcon all use RMSNorm

### The implementation line by line

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(dim))  # γ, learnable scale
        # No β (bias) — RMSNorm doesn't need it
```

The `eps = 1e-6` prevents division by zero when the RMS is very close to 0.
`gamma` starts at 1.0 (identity — no scaling initially), and the model learns
the right scale during training.

```python
def forward(self, x):
    # x shape: (..., dim) — works for any number of leading dims
    rms = x.pow(2)              # Square every element
            .mean(dim=-1,        # Average across the feature (last) dimension
                  keepdim=True)  # Keep the dim so broadcast works
            .add(self.eps)       # Add ε for numerical stability
            .sqrt()              # Take the square root → RMS value
    return (x / rms) * self.gamma  # Rescale and apply learned gamma
```

The result has values roughly in the range `[-1, 1] × gamma`. Since `gamma`
initialises to 1.0, the output range starts at roughly `[-1, 1]`.

### Why gamma initialises to 1.0, not 0.2887

In standard transformer architectures with residual connections that add with
weight 1.0, the hidden state magnitude grows as `O(√L)` where L is the layer
depth. To compensate for this, some implementations initialise gamma to
`1/√L ≈ 0.2887` for a 12-layer model (to pre-scale down the expected growth).

**AttnRes eliminates this problem entirely.** Because AttnRes uses a learnable
softmax to weight all previous layer outputs, no single layer's output compounds
multiplicatively. The magnitude stays bounded. Therefore `gamma = 1.0` is
correct here. Using 0.2887 would under-scale the outputs and slow learning.

### Where RMSNorm is used (complete map)

```
Pre-attention norm:     3 instances (one per TransformerBlock, before GQA)
Pre-FFN norm:           3 instances (one per TransformerBlock, before FFN)
AttnRes key_norm:       6 instances (two per block — one in each AttnRes)
QK-Norm (Q heads):     12 instances per layer × 3 layers = 36
QK-Norm (K heads):      4 instances per layer × 3 layers = 12
Final norm:             1 instance (before output projection)
MTP head norm:          1 instance (inside MTPHead)
─────────────────────────────────────────────────────────
TOTAL:                 72 RMSNorm instances, each with d_head or d_model params
```

Each instance is **independent** — they do not share weights. The 12+4 = 16
QK-Norm instances per attention layer are particularly important: they normalise
per-head (each head's 64-dim vector), not per-token across all heads.

---

## 7. RoPE — Rotary Position Embedding

**File:** `slm_project/model/rope.py`

### The problem: attention has no idea what position tokens are at

The core attention operation — `softmax(Q × Kᵀ / √d) × V` — computes
relationships between tokens purely based on their content. There is no
information about whether token A is at position 3 or position 300. A sentence
where word order matters would be processed identically regardless of word order
without position information.

The classic solution (GPT, BERT) was to add a fixed or learned position vector
to each token's embedding before the first layer. RoPE takes a different approach:
it encodes position **into the attention scores directly**, by rotating the Q
and K vectors based on their absolute position before computing the dot product.

### The key insight: rotation makes relative positions appear in dot products

If we rotate query vector `q` at position `m` by angle `mθ`, and key vector `k`
at position `n` by angle `nθ`, then when we compute `q · k`:

```
q_rotated · k_rotated = f(q, k, m-n)
```

The dot product only depends on the **relative position** `m-n`, not on the
absolute positions m and n individually. This means attention naturally and
correctly learns "how much does a token 5 positions ago matter?" without needing
to learn absolute positions.

### How RoPE works mechanically

Each d_head=64 dimensional Q or K vector is split into 32 pairs: `(x₁, x₂)`.
Each pair is rotated by an angle that depends on:
1. The token's absolute position in the sequence
2. Which pair it is (pairs get different frequencies)

The rotation formula for pair `i` at position `m`:
```
[x₁, x₂] → [x₁·cos(mθᵢ) - x₂·sin(mθᵢ),
              x₁·sin(mθᵢ) + x₂·cos(mθᵢ)]
```

In matrix form (and in the code):
```
x_rotated = x * cos(mθ) + rotate_half(x) * sin(mθ)
```
where `rotate_half([x₁..x₃₂, x₃₃..x₆₄]) = [-x₃₃..-x₆₄, x₁..x₃₂]`.

### The frequency bands — why rope_base = 500,000 matters

The frequencies `θᵢ` are set by:
```
θᵢ = 1 / (rope_base ^ (2i / d_head))   for i = 0, 1, ..., d_head/2 - 1
```

With `rope_base = 10,000` (original RoPE paper):
```
θ₀ = 1/1      = 1.0       (fastest — completes a full rotation every ~6 tokens)
θ₁ = 1/1.5    = 0.667
...
θ₃₁ = 1/10000 = 0.0001    (slowest — completes a full rotation every 62,832 tokens)
```

At `rope_base = 10,000`, positions repeat (alias) around `2π / θ_min = 62,832`
tokens — but the **effective** repetition for attention quality starts much
earlier, around 8,000 tokens, because the high-frequency dimensions don't
distinguish positions well beyond that point.

With `rope_base = 500,000` (this model, same as LLaMA 3):
```
θ₃₁ = 1/500000 = 0.000002   (slowest pair completes a rotation every 3.14M tokens)
```

Position encoding stays unique and distinguishable up to 32K+ tokens. For an
8K max_seq_len, this gives abundant headroom and means the model can be extended
to longer contexts later without position aliasing.

### The cos/sin cache — why pre-compute?

Computing `cos(mθᵢ)` and `sin(mθᵢ)` for every position on every forward pass
would be wasteful — these values never change. The cache is built once:

```python
def _build_cache(self, seq_len, device):
    t     = torch.arange(seq_len)            # [0, 1, 2, ..., T-1]
    freqs = torch.einsum('i,j->ij', t, inv_freq)  # [T, 32] — outer product
    emb   = torch.cat([freqs, freqs], dim=-1)     # [T, 64] — duplicated for rotate_half
    self.cos_cached = emb.cos()[None, None, :, :] # [1, 1, T, 64] — broadcast-ready
    self.sin_cached = emb.sin()[None, None, :, :]
```

During the forward pass, only a slice `[:, :, :seq_len, :]` is used, so short
sequences don't pay the cost of the full cache.

### RoPE is applied only in LOCAL blocks — why not global?

**Local (SWA) blocks:** attend to nearby tokens within a 2,048-token window.
Relative position matters enormously here — a token 5 positions ago is very
different from one 2,000 positions ago. RoPE encodes this.

**Global (NoPE) blocks:** attend to the entire sequence. These blocks use
**NoPE** — No Position Encoding. The idea is that by the time information
reaches a global block, it has already been enriched with positional content
by the local blocks below it. Global blocks focus on **content-based** global
aggregation, not positional relationships. This is the architecture used by
Gemma 2 and several other recent models.

The assignment happens in `TransformerBlock.__init__`:
```python
# In model.py when constructing blocks:
TransformerBlock(cfg, is_global=True,  rope=None)    # global: no RoPE
TransformerBlock(cfg, is_global=False, rope=self.rope)  # local: shared RoPE
```

**One shared RoPE instance** is created in the SLM class and passed to all
local blocks. They all use the same cached cos/sin tables, saving memory.

---

*Parts 5 & 6 complete.*

---

## 8. AttnRes — Full Attention Residuals

**File:** `slm_project/model/attn_res.py`

This is the most novel component in this codebase. Understanding it deeply is
essential to understanding why the model trains the way it does.

### The problem with standard residual connections

In a standard transformer, the residual connection is:
```
output = sub_layer(x) + x
```

This adds the original input `x` back with weight **1.0** every time. After
N layers, the original embedding vector `v₀` has been added N times — but so
has every intermediate layer output. The effective contribution of `v₀` to the
final representation is diluted. In a 16-layer model, `v₀` is just 1 of 17
terms being averaged (roughly). The model cannot easily decide "use mostly the
embedding" vs "use mostly layer 8's output."

Another problem: all intermediate outputs are combined with equal weighting.
Layer 15's output is worth exactly as much as layer 2's output in the residual
sum, regardless of what the task requires.

### The AttnRes solution: learned softmax attention over all previous layers

Instead of `output = sub_layer(x) + x`, AttnRes computes:

```
h = Σᵢ αᵢ · vᵢ      where i ∈ {0, 1, ..., N-1}

αᵢ = softmax_over_layers( wᵀ · RMSNorm(vᵢ) )

w ∈ ℝ^{d_model}   — the pseudo_query, one per AttnRes instance
```

In plain words:
- Every previous layer output `vᵢ` is treated as a "key"
- A learned vector `w` (the pseudo_query) is dot-producted with each key
- The results are passed through softmax (which forces them to sum to 1.0)
- The final hidden state is a **weighted average** of all previous outputs

The critical word is **softmax**: the weights `αᵢ` must sum to 1.0 over all
layers. This means the model is always choosing a convex combination — it
cannot amplify signals, only redistribute weight among existing representations.

### What happens at initialisation: zero = uniform

The pseudo_query is initialised to **exactly zero** (this is mandatory and
enforced in `init_weights.py`):

```python
self.pseudo_query = nn.Parameter(torch.zeros(d_model))
```

When `w = 0`:
```
logits = wᵀ · RMSNorm(vᵢ) = 0 · (anything) = 0   for all i
softmax([0, 0, ..., 0]) = [1/N, 1/N, ..., 1/N]
```

Every layer output gets exactly equal weight: `1/N`. This is equivalent to
averaging all previous layer outputs — a stable, well-conditioned baseline.
Training then specialises from this baseline. The model learns which layers
to weight more for which positions and tasks.

If pseudo_query were initialised randomly instead of zero, different AttnRes
instances would start with wildly different attention patterns, creating
asymmetry and making training unstable in early steps.

### The softmax is over the LAYER dimension, not the sequence dimension

This is the most common source of bugs when reimplementing AttnRes. There are
three dimensions after stacking: `[N, B, T, d_model]`.

- `dim=0` is the **N (layer)** dimension — THIS is where softmax goes
- `dim=1` is the batch dimension — wrong, would give random weights per sample
- `dim=2` is the sequence (T) dimension — wrong, would weight positions, not layers
- `dim=3` is the feature dimension — nonsensical for softmax here

```python
# From attn_res.py
V = torch.stack(layer_outputs, dim=0)    # [N, B, T, d_model]
K = self.key_norm(V)                     # [N, B, T, d_model] — normalised keys

# logits: dot product of pseudo_query with each normalised key vector
# pseudo_query: [d_model]   K: [N, B, T, d_model]
logits = torch.einsum('d, nbtd -> nbt', self.pseudo_query, K)  # [N, B, T]

# CRITICAL: softmax over dim=0 (the N/layer dimension)
alpha = torch.softmax(logits, dim=0)     # [N, B, T], sums to 1 over dim=0

# Weighted sum → [B, T, d_model]
h = torch.einsum('nbt, nbtd -> btd', alpha, V)
```

The assertion `alpha.sum(dim=0).allclose(ones)` is always active (not just in
debug mode) to catch any accidental change of the softmax dimension.

### The key_norm — why each AttnRes has its own RMSNorm

Without key normalisation, a layer that happens to have large-magnitude outputs
(e.g. layer 5 produces vectors with mean norm 50) would dominate the attention
scores purely by magnitude, not by relevance. `key_norm` normalises each layer's
output before computing the dot product, so the attention scores reflect the
**angular similarity** between pseudo_query and each layer's representation,
not its magnitude.

This is analogous to QK-Norm in attention (Section 9) — normalise before
computing dot products so scale doesn't dominate content.

Each AttnRes instance has **its own independent key_norm** (a separate RMSNorm
with its own gamma). This is correct: the pre-attention AttnRes and the pre-FFN
AttnRes see different distributions of layer outputs and need different scales.

### The first AttnRes: a special case

Block 0's **pre-attention AttnRes** only ever sees `layer_outputs = [v₀]` (the
embedding). With N=1:
```
softmax([anything]) = [1.0]
output = 1.0 × v₀ = v₀
```

No matter what pseudo_query is, this AttnRes outputs `v₀` unchanged.
Its gradient is always zero. Its norm stays at 0.0 throughout training.
**This is correct behaviour, not a bug.** It becomes meaningful only after
`layer_outputs` has at least 2 entries to choose between.

### AttnRes placement: two per block, one per sub-layer

```
Block 0:
  attn_res_attn  ← sees [v₀] at first call           (N=1, no choice)
  attn_res_ffn   ← sees [v₀, attn_out_0]             (N=2, first real choice)

Block 1:
  attn_res_attn  ← sees [v₀, attn_out_0, ffn_out_0]  (N=3)
  attn_res_ffn   ← sees [v₀, ..., attn_out_1]        (N=4)

Block 2:
  attn_res_attn  ← sees [v₀, ..., ffn_out_1]         (N=5)
  attn_res_ffn   ← sees [v₀, ..., attn_out_2]        (N=6)
```

By block 2's FFN, the AttnRes is choosing from 6 previous representations.
In the 16-layer production model, the final AttnRes chooses from 32 entries.

### pseudo_query and the 2× learning rate

Because pseudo_query starts at zero and all gradients cancel at initialisation
(softmax([0,0,...]) is a saddle point), it needs a **higher learning rate than
normal weights** to break symmetry quickly. This is why Group 3 in the optimizer
gives pseudo_query parameters `2× base_lr = 6e-4`.

If pseudo_query is accidentally placed in the same optimizer group as normal
weights (3e-4 LR), it learns 2× too slowly. Gate 4 in Phase 1 catches this:
norms should reach [0.001, 2.0] by step 300. If they're all near zero, the LR
is wrong.

---

## 9. Grouped Query Attention + QK-Norm + FlashAttention

**File:** `slm_project/model/attention.py`

### Standard Multi-Head Attention (MHA) — the starting point

In original attention (Vaswani et al., 2017):
```
Q = x @ W_Q   [B, T, d_model]
K = x @ W_K   [B, T, d_model]
V = x @ W_V   [B, T, d_model]

Reshape Q/K/V to [B, T, n_heads, d_head]
For each head h:
    scores_h = Q_h @ K_hᵀ / √d_head    [B, T, T]
    attn_h   = softmax(scores_h) @ V_h  [B, T, d_head]
Concatenate all heads → [B, T, d_model]
Output = concat @ W_O
```

With n_heads=12, d_head=64: Q, K, V all have 12 heads. This means **12 separate
K and V matrices per layer** during inference. The KV cache (which stores K and V
for all previously generated tokens) grows as `2 × T × n_heads × d_head` per layer.

### Grouped Query Attention (GQA) — 3× smaller KV cache

GQA uses **fewer KV heads than Q heads**. Here: 12 Q heads, 4 KV heads (3:1 ratio).

```
Q = x @ W_Q   [B, T, 12 × 64] → reshaped to [B, T, 12, 64]
K = x @ W_K   [B, T,  4 × 64] → reshaped to [B, T,  4, 64]
V = x @ W_V   [B, T,  4 × 64] → reshaped to [B, T,  4, 64]
```

Each group of 3 Q heads shares one K head and one V head. During attention:
- Q head 0, 1, 2 all attend to K/V head 0
- Q head 3, 4, 5 all attend to K/V head 1
- etc.

**KV cache size:** `2 × T × 4 × 64 = 512T` floats per layer (vs `2 × T × 12 × 64 = 1536T`
for MHA). **3× reduction.** This is what lets the model serve longer sequences
without running out of GPU memory during inference.

**Parameter savings:**
```
W_K: 768 × (4 × 64) = 768 × 256 = 196,608   (vs 768 × 768 = 589,824 in MHA)
W_V: 768 × (4 × 64) = 196,608                (same saving)
Total saving per layer: ~800K parameters
```

### QK-Norm — per-head RMSNorm on Q and K

Before computing attention scores, both Q and K are normalised per-head:

```python
# 12 separate RMSNorm(64) instances for Q heads
self.q_norms = nn.ModuleList([RMSNorm(cfg.d_head) for _ in range(cfg.n_heads_q)])

# 4 separate RMSNorm(64) instances for K heads
self.k_norms = nn.ModuleList([RMSNorm(cfg.d_head) for _ in range(cfg.n_heads_kv)])
```

**Why per-head, not per-token?** If we used RMSNorm(d_model=768), we'd normalise
across all 12 heads at once, destroying the per-head specialisation. Each head
is meant to learn different things; their scale should be normalised independently.

**Why normalise Q and K at all?** In bfloat16, large dot products between
unnormalised Q and K vectors can cause attention scores to overflow (become
±inf), making softmax output either 0 or 1 everywhere (attention "spikes").
QK-Norm keeps the scores in a stable range regardless of input magnitude.
This is the same technique used in Gemma, Cohere, and other recent models.

**Parameter count:** `(12 + 4) × 64 = 1,024` parameters per attention layer.

### FlashAttention vs SDPA fallback

The attention kernel (the actual `softmax(QKᵀ/√d) × V` computation) has two
implementations:

**FlashAttention (production — training machine):**
```python
from flash_attn import flash_attn_func

# Local block: SWA with explicit causal mask
out = flash_attn_func(q, k, v, causal=True, window_size=(2048, 0))

# Global block: full sequence with explicit causal mask
out = flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))
```

FlashAttention never materialises the full `[B, T, T]` attention matrix in HBM.
It computes attention in tiles that fit in SRAM, reducing memory from `O(T²)` to
`O(T)`. For T=8192, this is `8192²×2 bytes = 128 MB` saved per layer per call.

**CRITICAL:** `causal=True` is always passed explicitly. `window_size=(-1,-1)`
disables SWA but does NOT automatically set causal masking — that must be
passed separately. Getting this wrong causes the model to attend to future tokens
during training, which silently makes loss appear lower than it really is.

**SDPA fallback (dev machine — no real FlashAttention):**
```python
# Expand KV heads to match Q heads (GQA simulation)
k_exp = k.repeat_interleave(3, dim=2)   # [B,T,4,64] → [B,T,12,64]
v_exp = v.repeat_interleave(3, dim=2)

# Build the correct mask for local vs global
if self.is_global:
    causal_mask = torch.tril(torch.ones(T, T))  # full causal
else:
    # SWA: causal AND within 2048-token window
    causal_mask = tril AND (position_diff <= 2048)

out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
```

The SDPA fallback produces identical numerical results to FlashAttention for
the dev-machine tests. The real FlashAttention package will automatically
replace the shim when installed on the training machine.

### The full attention forward pass

```
x [B, T, 768]
    ↓ W_Q                      → [B, T, 768] → reshape → [B, T, 12, 64]
    ↓ W_K                      → [B, T, 256] → reshape → [B, T,  4, 64]
    ↓ W_V                      → [B, T, 256] → reshape → [B, T,  4, 64]
    ↓ QK-Norm (per head)       → q [B,T,12,64], k [B,T,4,64] normalised
    ↓ RoPE (local blocks only) → q, k rotated by position angle
    ↓ FlashAttention / SDPA    → [B, T, 12, 64]
    ↓ reshape                  → [B, T, 768]
    ↓ W_O                      → [B, T, 768]
```

---

*Parts 7 & 8 complete.*

---

## 10. IHA-Global — Interleaved Head Attention  *(NEW)*

**File:** `slm_project/model/attention.py` → class `IHAGlobalAttention`

### What problem IHA solves

Standard Multi-Head Attention (MHA) is expressive but has a fundamental
limitation in **multi-hop reasoning**. To reason over a chain of $k$ facts
(e.g. "A→B, B→C, therefore A→C"), MHA in a single layer can only make 1
attention "hop". Getting $k$ hops requires $k$ layers or $O(k)$ heads.

The paper *"Interleaved Head Attention"* (arXiv:2602.21371) proves that IHA
can realise **polynomial-order ($O(\sqrt{k})$) reasoning filters** in the same
number of layers and heads by allowing heads to communicate with each other
during the attention computation.

### What IHA actually does

In standard MHA, each head $h$ computes attention independently:
```
head_h = softmax(Q_h × K_hᵀ / √d) × V_h
```
Head $h$ uses only its own $Q_h$, $K_h$, $V_h$. No cross-head communication.

In IHA, before computing attention, **pseudo-queries, pseudo-keys,
and pseudo-values** are formed as learned linear combinations of all heads:

```
For each head h, for each pseudo-slot p ∈ {0, 1, ..., P-1}:
  Q̃_{h,p} = Σᵢ α_Q[h, i, p] · Q_i      # mix across all H_Q heads
  K̃_{h,p} = Σᵢ α_K[h, i, p] · K_i      # mix across all H_KV heads
  Ṽ_{h,p}  = Σᵢ α_V[h, i, p] · V_i      # mix across all H_KV heads
```

Each head can now realise $P^2$ distinct attention patterns (one for each
$(p_q, p_k)$ pair) instead of just 1. The paper proves IHA strictly
**contains** MHA — any MHA output can be reproduced by IHA.

### The P=2 choice and why it is fixed

This model uses **P=2** (two pseudo-slots per head). This is a deliberate,
non-negotiable constraint:

| P value | Extra params | Within-attention sequence | Where? |
|---------|-------------|--------------------------|--------|
| P=1     | 0           | N×1 (= standard MHA)    | —      |
| **P=2** | **+2,560**  | **N×2**                  | **Global only** |
| P=12    | +61,440     | N×12                     | ❌ Never |

**P=12 in local layers would collapse the SWA window:**
Local blocks use Sliding Window Attention with `swa_window=2048`. The IHA
expansion creates a `N×P` sequence internally. If a local block had P=12,
the effective SWA window would become `2048/12 ≈ 170` tokens — catastrophically
small. **IHA is therefore barred from all local blocks.** Only global layers
(NoPE, full-sequence attention) use IHA, where there is no SWA window to collapse.

P=2 in the 4 global layers adds:
```
alpha_Q:  [12, 12, 2]  = 288 params  per global layer
alpha_K:  [ 4,  4, 2]  =  32 params  per global layer
alpha_V:  [ 4,  4, 2]  =  32 params  per global layer
R (rot):  [ 2,  2]     =   4 params  per global layer (optional rotation)
─────────────────────────────────────────────────────
356 params × 4 global layers = 2,560 params total (IHA overhead)
```

This brings the production model to exactly **125,931,008 parameters**
(= 125,928,448 base + 2,560 IHA).

### The forward pass of IHAGlobalAttention

```
x [B, N, D]
    ↓ W_Q, W_K, W_V
Q [B,N,12,64]   K [B,N,4,64]   V [B,N,4,64]
    ↓ QK-Norm (16 RMSNorm(64), identical to GQA)
    ↓ IHA Step 1: Mix across heads
        Q̃ [B,N,12,2,64]   K̃ [B,N,4,2,64]   Ṽ [B,N,4,2,64]
    ↓ IHA Step 2: Interleave pseudo-dim into sequence dimension
        Q_exp [B, N×2, 12, 64]
        K_exp [B, N×2,  4, 64]
        V_exp [B, N×2,  4, 64]
    ↓ FlashAttention (full causal, NoPE — no RoPE)
        out [B, N×2, 12, 64]
    ↓ IHA Step 3: Fold pseudo-dim back out of sequence
        out [B, N, 12, 2, 64]  →  mean over P dim  →  [B, N, 12, 64]
    ↓ reshape + W_O
        [B, N, D]
```

The output is always `[B, N, D]` — identical interface to `GroupedQueryAttention`.
The `N×P` expansion is entirely internal. AttnRes, MTP, and all downstream
modules see the same shape.

### Identity initialisation (mandatory)

The mixing matrices `alpha_Q`, `alpha_K`, `alpha_V` are initialised to
**identity** (diagonal = 1.0, off-diagonal = 0.0):

```python
nn.init.zeros_(self.alpha_Q)                # zero all first
for h in range(self.n_heads_q):             # then set diagonal to 1
    self.alpha_Q.data[h, h, :] = 1.0
```

This means at step 0, IHA is exactly equivalent to standard MHA — each head
mixes only with itself, no cross-head communication. Training then specialises
the mixing from this stable baseline.

**Why identity, not random?** With random mixing, pseudo-heads start with
wild patterns. The loss at step 0 would be unstable and Gate 1 would fail.
Identity initialisation guarantees the model starts at the same stable
point as a plain GQA model, and IHA's expressivity is only activated by training.

### The 3-pass initialisation order

Because `init_model_weights()` calls `model.apply(standard_init)` first
(which overwrites everything with Gaussian random values), the IHA identity
must be **re-applied after** `standard_init`. This is the third pass:

```
Pass 1: model.apply(standard_init)           → Gaussian(0, 0.02) for all Linear
Pass 2: model.apply(zero_attnres_queries)    → AttnRes.pseudo_query = 0
Pass 3: model.apply(reinit_iha_identity)     → alpha_Q/K/V back to identity
```

If Pass 3 is omitted, `alpha_Q` stays Gaussian-random and the model starts
in an unstable IHA state. Section 15 documents this in full.

### Optimizer group assignment for IHA params

`alpha_Q`, `alpha_K`, `alpha_V`, and `R` are all 2D or 3D tensors —
they fall into **Group 1 (weight-decay, base LR)** by the `dim >= 2`
heuristic in `build_optimizer()`. This is correct:

- They should NOT be in Group 3 (pseudo_query 2× LR) — they are not
  starting from zero and don't need the accelerated escape from a saddle point.
- They should NOT be in Group 2 (no weight decay) — they are learnable weight
  matrices and benefit from regularisation.

Verified at startup: optimizer Group 1 tensor count increases by the number
of IHA weight matrices when the production model is built.

---

## 11. SwiGLU Feed-Forward Network

**File:** `slm_project/model/ffn.py`

### What the FFN does

After attention, each token has a new representation that has gathered
information from other tokens. The FFN's job is **per-token processing** —
it applies the same learned transformation to each token's vector independently,
with no communication between positions. Think of attention as "gather
information from context" and FFN as "process that information into the right
representation."

### Standard FFN vs SwiGLU

**Standard FFN** (used in original Transformer, GPT-2):
```
FFN(x) = ReLU(x @ W₁ + b₁) @ W₂ + b₂
```
Two linear layers with a ReLU in between. Simple but suboptimal.

**GLU (Gated Linear Unit):**
```
GLU(x) = (x @ W₁) ⊙ σ(x @ W₂)
```
Two parallel projections: one is passed through a sigmoid gate, then they're
element-wise multiplied. The sigmoid gate learns to "open" or "close" each
dimension of the hidden state, giving the network a soft switch for each feature.

**SwiGLU** (Noam Shazeer, 2020 — used in LLaMA, Mistral, this model):
```
SwiGLU(x) = SiLU(x @ W_gate) ⊙ (x @ W_up)
```
Replaces the sigmoid gate with **SiLU** (Sigmoid Linear Unit, also called Swish):
```
SiLU(z) = z × σ(z)    where σ is sigmoid
```
SiLU is smoother than ReLU (no hard zero at negative values), differentiable
everywhere, and empirically outperforms ReLU and GeLU in large models.

The full SwiGLU FFN:
```
gate = SiLU(x @ W_gate)    [B, T, ffn_hidden]  — gating signal
up   = x @ W_up            [B, T, ffn_hidden]  — content signal
out  = (gate ⊙ up) @ W_down [B, T, d_model]   — project back
```

Three weight matrices: W_gate, W_up, W_down. No biases (they add minimal
value and cost parameters). Two of the three matrices (W_gate, W_up) expand
from d_model=768 to ffn_hidden=2048. W_down contracts back.

### The ffn_hidden = 2048 formula

In a standard 2-matrix FFN, the hidden dim is typically `4 × d_model = 3072`.
SwiGLU uses 3 matrices instead of 2, so the hidden dim is adjusted to keep
parameter count roughly equal:

```
Standard FFN:    2 × (768 × 3072)  = 4,718,592 params (same!)
SwiGLU:          3 × (768 × 2048)  = 4,718,592 params
```

The formula `round(8/3 × d_model / 256) × 256` gives `2048` for d_model=768.
The `8/3` factor comes from: `(4/3) × 2 = 8/3` — you need 4/3 of the original
hidden dim when using 3 matrices to match 2-matrix parameter count, and the
`×2` is because SwiGLU needs two "up" matrices (gate + up). The `round(./256)×256`
snaps to the nearest multiple of 256, which is optimal for GPU tensor core tiling.

### Dropout placement

```python
out = self.dropout(self.W_down(gate * up))
```

Dropout goes **after W_down**, before the result is passed back. During
pre-training, `dropout=0.0` (no dropout — the model needs full gradient signal).
During SFT, `dropout=0.1` is applied. This is set from config: `cfg.dropout_pretrain`
or `cfg.dropout_finetune`. Never hardcode the rate.

---

## 11. TransformerBlock — Wiring Everything Together

**File:** `slm_project/model/block.py`

### What the block contains

Each TransformerBlock holds exactly 6 sub-modules:

```
attn_res_attn:   AttnRes     — residual gating before attention
norm_attn:       RMSNorm     — pre-attention normalisation
attention:       GroupedQueryAttention — the actual attention
attn_res_ffn:    AttnRes     — residual gating before FFN
norm_ffn:        RMSNorm     — pre-FFN normalisation
ffn:             SwiGLUFFN   — the feed-forward network
```

### The forward pass in detail

```python
def forward(self, layer_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
```

Note the signature: it takes the **entire list** of previous outputs and
**returns that same list extended by 2 new tensors**. This is unusual —
most transformer blocks take a single tensor `x` and return a single tensor.
Here the block mutates a list because AttnRes needs access to every previous
output, not just the immediately preceding one.

**Attention sub-layer:**
```python
# Step 1: AttnRes decides which previous representations to combine
h = self.attn_res_attn(layer_outputs)      # [B, T, 768]
# At block 0: layer_outputs = [v₀],         h = v₀ (forced, only one choice)
# At block 1: layer_outputs = [v₀,v₁,v₂],  h = softmax-weighted sum of 3

# Step 2: Pre-norm then attention
attn_out = self.attention(self.norm_attn(h))  # [B, T, 768]
# Note: norm_attn(h) is NOT stored — only attn_out is

# Step 3: Append to the growing list
layer_outputs.append(attn_out)             # v_{2b+1} added
```

**FFN sub-layer:**
```python
# Step 4: AttnRes NOW INCLUDES attn_out in its pool
h2 = self.attn_res_ffn(layer_outputs)     # [B, T, 768]
# This is why we append before computing the FFN AttnRes

# Step 5: Pre-norm then FFN
mlp_out = self.ffn(self.norm_ffn(h2))     # [B, T, 768]

# Step 6: Append
layer_outputs.append(mlp_out)             # v_{2b+2} added

return layer_outputs
```

### Pre-Norm vs Post-Norm — and why Pre-Norm is used here

**Post-Norm** (original Transformer, 2017): `output = LayerNorm(x + sub_layer(x))`
The normalisation happens after the residual addition. This was notoriously
difficult to train without warmup and careful LR schedules.

**Pre-Norm** (GPT-2, LLaMA, this model): `output = sub_layer(LayerNorm(x)) + x`
The input is normalised before the sub-layer. This is much more stable: the
sub-layer always receives a normalised input, so its outputs are bounded, and
gradients flow more cleanly through the residual path.

In this model, the Pre-Norm pattern is:
```
h   = AttnRes(layer_outputs)       — combine previous layers
out = attention(RMSNorm(h))        — normalise h before feeding to attention
```

The AttnRes output `h` is used as the residual baseline. The sub-layer output
is added directly to `layer_outputs` — not added back to `h`. This is a subtle
but important difference from standard Post-Norm and Pre-Norm:

- Standard: `output = LayerNorm(x) → sub_layer → + x`
- AttnRes:  `output = sub_layer(RMSNorm(AttnRes(all_previous)))`
             `stored separately, not added to h`

Each sub-layer produces a **fresh representation** that goes into `layer_outputs`.
The "residual" is the attention that AttnRes pays to all previous outputs —
not an explicit `+x` addition.

### Gradient checkpointing interaction

The list mutation pattern (`layer_outputs.append(...)`) is safe during normal
forward passes. During gradient checkpointing, PyTorch re-runs the forward pass
during backward to recompute activations. The CRITICAL constraint is that
`layer_outputs` cannot be passed as a Python list to `torch.utils.checkpoint.checkpoint()`,
because checkpoint() cannot track mutations to a list object between the forward
and recompute passes.

The solution (implemented in `model.py`) is to **pack the list as a tuple**
before crossing the checkpoint boundary:

```python
# In model._forward_with_checkpoint:
def _block_fn(*lo_tensors, _block=block):
    lo = list(lo_tensors)      # unpack tuple → list
    result = _block(lo)        # run block (mutates list)
    return tuple(result)       # pack result → tuple

new_outputs = ckpt.checkpoint(_block_fn, *tuple(layer_outputs))
layer_outputs = list(new_outputs)
```

This prevents the "RuntimeError: Trying to backward through the graph a
second time" and "freed buffer" errors that occur if the list is passed directly.

---

## 12. MTP Head — Multi-Token Prediction

**File:** `slm_project/model/mtp.py`

### Why predict more than one token?

The main training objective is **next-token prediction**: from position `t`,
predict token `t+1`. The loss gradient flows backward through the entire model
for each token predicted.

**Multi-Token Prediction (MTP)** adds an auxiliary objective: from position `t`,
also predict token `t+2` (two steps ahead). This provides:

1. **Extra gradient signal** — the same hidden state now has two tasks,
   doubling the gradient information flowing back to early layers
2. **Richer representations** — the model must capture not just "what comes
   next" but "what tends to follow what comes next"
3. **Faster early training** — token `t` gets feedback from two future positions
   instead of one

The MTP loss is added to the main loss with a learnable weight that anneals
over training (see below). At inference, the MTP head is unused (or used for
speculative decoding).

### Architecture — intentionally lightweight

```python
class MTPHead(nn.Module):
    def __init__(self, cfg):
        self.norm  = RMSNorm(cfg.d_model)          # stabilise input
        self.W_mtp = nn.Linear(d_model, d_model)   # 768×768 = 589,824 params
        # No W_vocab — weight-tied to embedding

    def forward(self, hidden, embedding_weight):
        x = F.gelu(self.W_mtp(self.norm(hidden)))  # [B, T, 768]
        return x @ embedding_weight.T              # [B, T, 32010]
```

**GeLU not SiLU:** The MTP head uses GeLU (Gaussian Error Linear Unit) rather
than SiLU. GeLU is smoother and more commonly used in head/projection layers.
The main FFN uses SiLU because it's part of SwiGLU; these are different design
choices for different architectural roles.

**Weight tying:** `embedding_weight` is passed in by the caller (the SLM class)
as `self.embedding.weight`. The MTP head owns no separate vocabulary projection
— it reuses the same 24.6M parameter table. This keeps the parameter count low
and ensures consistency between the main logits and MTP logits.

### The loss computation

```python
# Main task: predict t+1 from position t
main_loss = cross_entropy(
    logits[:, :-1],    # positions 0..T-2 predict
    labels[:, 1:]      # tokens 1..T-1
)

# MTP task: predict t+2 from position t
mtp_loss = cross_entropy(
    mtp_logits[:, :-2],  # positions 0..T-3 predict
    labels[:, 2:]        # tokens 2..T-1
)

total_loss = main_loss + mtp_weight(step) × mtp_loss
```

Note the index shifts: `[:, :-1]` drops the last position (it has nothing to
predict for t+1), `[:, :-2]` drops the last two positions (they have nothing
to predict for t+2). The `ignore_index=-100` in cross_entropy allows SFT loss
masking — user/system tokens are set to -100 in `labels` and are skipped.

### MTP weight annealing schedule

```
Step 0       → 50,000:  mtp_weight = 0.3  (constant)
Step 50,000  → 60,000:  mtp_weight linearly 0.3 → 0.1
Step 60,000+ →    ∞:    mtp_weight = 0.1  (constant)
```

Why reduce over time? Early in training, the model benefits most from the
extra gradient signal. As the model matures and the main task loss is well-behaved,
the MTP signal can be downweighted to focus on the primary objective. At 0.1,
it continues to provide a regularisation signal without dominating.

---

*Parts 9, 10 & 11 complete.*

---

## 15. Weight Initialisation

**File:** `slm_project/model/init_weights.py`

### Why initialisation matters so much

A neural network's weights before any training is done determine:
1. How large the initial outputs are (too large → gradient explosion)
2. How symmetric or asymmetric the starting point is (too symmetric → all neurons learn the same thing)
3. Whether special components like AttnRes start in the correct mode

Bad initialisation can make training impossible to recover. The ordering of
initialisation steps in this codebase is **non-negotiable** — swapping them
silently breaks the model.

### The three-step init protocol

```python
def init_model_weights(model):
    # STEP 1 — Gaussian init for ALL Linear and Embedding weights
    model.apply(standard_init)

    # STEP 2 — Zero ALL AttnRes.pseudo_query
    model.apply(zero_attnres_pseudo_queries)

    # STEP 3 — Re-init IHA mixing tensors to identity
    model.apply(reinit_iha_identity)

    # STEP 4 — Verify overrides survived
    for name, module in model.named_modules():
        if isinstance(module, AttnRes):
            assert module.pseudo_query.allclose(torch.zeros_like(...))
```

**Why the order must be strictly 1 → 2 → 3:** `model.apply(fn)` visits every module
recursively and calls `fn` on each. If you do Step 2 or 3 first and then
call `model.apply(standard_init)` afterward, `standard_init` will overwrite the
zeros and identities with Gaussian random values. The overrides must happen after
ALL other `apply()` calls have completed.

**Why Step 4 (verification) is not optional:** Without the assertion, a
programmer could accidentally add another `model.apply()` call after
`init_model_weights()`, reordering would be silent, and the training would
proceed with randomly-initialised pseudo_queries — producing unstable behaviour
that is very hard to diagnose.

### standard_init — why std=0.02?

```python
def standard_init(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

`std=0.02` is the GPT-2 / LLaMA convention. Here is why this specific value:

- Too small (e.g. 0.001): neurons start nearly identical; gradients are tiny
  in early training; slow convergence.
- Too large (e.g. 0.1): with 768-dimensional dot products, the attention
  logits would be in the range `0.1 × √768 ≈ 2.77`, pushing softmax toward
  one-hot before training even begins.
- `std=0.02`: attention logits ≈ `0.02 × √768 ≈ 0.55` — in the range where
  softmax is smooth and all positions receive meaningful gradient.

The formula for the "right" std from the Xavier/He initialisation literature
is approximately `1/√fan_in`. For a 768→768 linear, `1/√768 ≈ 0.036`. The
0.02 value is slightly more conservative (smaller outputs at init), which has
been empirically validated for language models across GPT-2, LLaMA, Falcon, etc.

**Biases:** All linear layers in this model are `bias=False` (explicitly set
in every `nn.Linear` call). The `if module.bias is not None` branch exists for
completeness but never fires. Biases in attention and FFN layers add minimal
value and cost parameters.

**RMSNorm gamma:** `standard_init` intentionally does NOT touch RMSNorm gammas.
They are initialised to `torch.ones(dim)` in `RMSNorm.__init__()` and left at
1.0. This is correct — see Section 6.

### zero_attnres_pseudo_queries

```python
def zero_attnres_pseudo_queries(module):
    from slm_project.model.attn_res import AttnRes
    if isinstance(module, AttnRes):
        nn.init.zeros_(module.pseudo_query)
```

This is a one-liner but it is the most critical initialisation decision in the
entire codebase. With `pseudo_query = 0`:
- AttnRes outputs uniform weighted average of all previous layers
- Training starts from a provably stable, balanced baseline
- Gradient through AttnRes is well-behaved from step 1

With `pseudo_query = random`:
- Different AttnRes instances start with wildly different attention patterns
- Some layers might immediately dominate; others might receive near-zero weight
- Early gradients are noisy and unbalanced; recovery is possible but slow and
  depends heavily on luck

### The verification print

After `init_model_weights()`, you will always see:
```
Weight init verified: all pseudo_queries = 0.0
```
If you don't see this line, `init_model_weights()` was not called. If training
starts and Gate 3 fails (pseudo_query norms ≠ 0 at step 0), the init was called
in the wrong order or overridden by something else.

---

## 16. Data Pipeline — Download & Shards

**Files:** `slm_project/data/download.py` + `slm_project/data/dataset.py`

### The problem this pipeline solves

A language model needs to see text. But:
- **FineWeb-Edu** has ~10 billion tokens. We only need 2 million.
- **Wikipedia EN** has ~4 billion tokens. We only need 500,000.
- Downloading the full datasets would take hours and hundreds of GB.
- Re-tokenising on every training run would waste time.
- We need to be able to resume training from exactly where we stopped.

The solution: **stream a tiny slice** → **tokenise once** → **save as binary shards** → **read deterministically during training**.

### Step 1: Streaming download (download.py)

HuggingFace's `datasets` library supports streaming mode — it downloads
examples one at a time without fetching the whole dataset:

```python
ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                  streaming=True, split='train')

for example in ds:          # downloads one example at a time
    text = example['text']
    ids  = tokenizer.encode(text)
    ids.append(eos_token_id)  # EOS marks document boundaries
    # ... accumulate tokens ...
    if tokens_collected >= 2_000_000:
        break               # stop mid-stream — rest is never downloaded
```

The HuggingFace server sends a socket error when we close the connection early.
This is expected and harmless — the data we need is already on disk.

### Why EOS between documents?

Each document ends with `<|end|>` (ID 32005). Without this, the last token
of document A and the first token of document B are adjacent — the model would
learn to predict inter-document transitions that don't exist in real text.

The EOS also teaches the model that sequences end: during inference, the model
must emit EOS to stop generating. If it never sees EOS in training data, it
will keep generating forever.

### The shard format: uint16 binary

```python
def _save_shard(tokens: list, prefix: str, idx: int):
    arr = np.array(tokens, dtype=np.uint16)   # 2 bytes per token
    arr.tofile(f"{prefix}_shard{idx:04d}.bin")
```

Why `uint16`? Our vocabulary has 32,010 tokens. The maximum token ID is 32,009,
which fits in `uint16` (max value 65,535). Using `uint16` instead of `int32`
halves the storage:
- 200,000 tokens × 2 bytes = **391 KB per shard** (vs 781 KB with int32)
- 10 shards × 391 KB = ~3.9 MB total for Phase 1 data

The file naming uses zero-padded 4-digit indices (`shard0000.bin`, `shard0001.bin`)
so that alphabetical sort equals numerical sort — critical for deterministic
shard ordering.

### Step 2: ShardedDataset (dataset.py)

This is a standard PyTorch `Dataset` that reads the binary shards:

```python
dataset = ShardedDataset('data/shards/*.bin', seq_len=256)
loader  = DataLoader(dataset, batch_size=1, shuffle=False)
```

**What it returns per item:**
```python
input_ids = tokens[t   : t + seq_len]   # [seq_len] — model input
labels    = tokens[t+1 : t + seq_len+1] # [seq_len] — shifted by 1 (targets)
```

The label is shifted by 1 because the task is next-token prediction: input
position `t` should predict label position `t+1`.

**How the window-to-shard mapping works (O(1) lookup):**

At startup, the dataset scans all shard file sizes and builds a cumulative
index:
```
shard_paths  = ['shard0000.bin', 'shard0001.bin', ...]
shard_lengths = [200000, 200000, ...]        # tokens per shard
shard_offsets = [0, 200000, 400000, ...]     # cumulative start position
```

For window index `idx`:
```
token_start = idx × seq_len
shard_idx   = binary_search(shard_offsets, token_start)
local_start = token_start - shard_offsets[shard_idx]
```

The binary search is `O(log n)` in the number of shards — effectively O(1)
for the small numbers of shards we use.

**Cross-shard stitching:**

When a window of 256 tokens starts at position 199,900 in shard 0 (which has
200,000 tokens), the window needs tokens 199,900–200,155 — but shard 0 only
goes to 200,000. The dataset stitches from two shards:

```python
part1 = shard_0[199900:]          # 100 tokens from end of shard 0
part2 = shard_1[:156]             # 156 tokens from start of shard 1
chunk = concatenate([part1, part2])
```

This ensures every window has exactly `seq_len` tokens, even at shard boundaries.

**Resume safety — why `data_global_idx` must be in every checkpoint:**

If training crashes at step 500 and resumes from a checkpoint, the DataLoader
needs to know to start from window 500 × grad_accum_steps × batch_size, NOT
from window 0. If it starts from 0:
- The model has already seen those tokens
- Their loss will be artificially low (memorisation)
- The training curve will look better than it really is
- The model will be implicitly trained on some data 2× more than other data

Every checkpoint saves:
```python
'data_global_idx': step × physical_batch_seqs × grad_accum_steps
```

On resume, pass this as `start_global_idx` to `ShardedDataset`.

---

*Parts 12 & 13 complete.*

---

## 18. Optimizer — Three Param Groups

**File:** `slm_project/training/optimizer.py`

### Why AdamW?

**SGD** (Stochastic Gradient Descent) updates weights proportional to the
gradient. Simple but requires careful tuning of the learning rate for every
layer, and it struggles with sparse gradients and saddle points.

**Adam** tracks a moving average of both the gradient (`m`, first moment) and
the squared gradient (`v`, second moment). The update is:
```
m_t = β₁ × m_{t-1} + (1-β₁) × g_t        # momentum — smoothed gradient
v_t = β₂ × v_{t-1} + (1-β₂) × g_t²       # smoothed squared gradient
θ_t = θ_{t-1} - lr × m̂_t / (√v̂_t + ε)   # normalise by expected gradient scale
```
The `v` term acts as a per-parameter adaptive learning rate: parameters with
consistently large gradients get a smaller effective step; parameters with small
or sparse gradients get a relatively larger step. This makes Adam much less
sensitive to the initial LR than SGD.

**AdamW** is Adam with **decoupled weight decay**. In standard L2-regularised
Adam, the weight decay is added to the gradient before the Adam update, which
means the decay is scaled by the Adam denominator. In AdamW, the decay is
applied directly to the weights after the Adam step:
```
θ_t = (1 - lr × wd) × θ_{t-1} - lr × adam_update(g_t)
```
This is more principled: weight decay acts as a constant shrinkage toward zero,
independent of the gradient history. GPT-3, LLaMA, and all modern LLMs use AdamW.

### Why THREE param groups — not one or two?

Most implementations use two groups (decay and no-decay). This model requires
a mandatory third group for a fundamental architectural reason.

**Group 1 — Weight-decay params (2D+ tensors: Linear weights, Embedding table)**

```
LR = peak_lr = 3e-4
weight_decay = 0.1
```

Weight decay (L2 regularisation) is applied to all 2-dimensional+ parameters.
The logic: weight matrices store information by their direction and magnitude.
Penalising large magnitudes prevents the model from over-relying on any single
weight path, which improves generalisation.

The `dim >= 2` heuristic correctly captures:
- `nn.Linear` weight: `[out, in]` → 2D ✓
- `nn.Embedding` weight: `[vocab, d_model]` → 2D ✓
- `IHAGlobalAttention` mixing tensors (`alpha_Q/K/V`): 3D ✓
- `IHAGlobalAttention` rotation tensor (`R`): 2D ✓

**Group 2 — No-decay params (1D tensors: RMSNorm gammas, biases)**

```
LR = peak_lr = 3e-4
weight_decay = 0.0
```

These are 1D vectors. Applying weight decay to RMSNorm gammas would push them
toward zero — meaning the normalisation would produce near-zero outputs,
destroying training. The rule is: **never apply weight decay to scale/shift
parameters**.

The `dim < 2` heuristic correctly captures:
- RMSNorm gamma: `[d_model]` → 1D ✓
- Any bias: `[out_features]` → 1D ✓ (none in this model, but the code is robust)

**Group 3 — pseudo_query ONLY (AttnRes parameters)**

```
LR = 2 × peak_lr = 6e-4
weight_decay = 0.1
```

This is the critical group. `pseudo_query` is a 1D tensor (shape `[d_model]`),
so the `dim >= 2` heuristic would place it in Group 2 with no weight decay and
base LR. **This would be wrong on both counts:**

- In Group 2 (no decay, base LR): `pseudo_query` learns too slowly to break
  symmetry. The zero-init saddle point is not strongly escaped — AttnRes stays
  near-uniform far too long.
- In Group 1 (decay, base LR): weight decay would pull `pseudo_query` toward
  zero, fighting against the gradient that's trying to specialise it. AttnRes
  collapses back toward uniform weighting — exactly what it's trying to escape.

Group 3 gives `pseudo_query` **2× base LR** (6e-4) AND weight decay (to prevent
runaway growth). This is the only combination that allows quick specialisation
from the zero-init baseline while keeping the parameters bounded.

### How the code identifies pseudo_query parameters

```python
# Step 1: collect the Python object id() of every pseudo_query tensor
pseudo_query_ids = set()
for module in model.modules():
    if isinstance(module, AttnRes):
        pseudo_query_ids.add(id(module.pseudo_query))

# Step 2: iterate named_parameters and route each to the right group
for name, param in model.named_parameters():
    if id(param) in pseudo_query_ids:
        pseudo_q_params.append(param)   # Group 3
    elif param.dim() >= 2:
        decay_params.append(param)      # Group 1
    else:
        no_decay_params.append(param)   # Group 2
```

Using `id(param)` for identity matching is important because `param.dim() == 1`
is true for both pseudo_query AND RMSNorm gammas — we can't use shape alone to
distinguish them. The `id()` check is exact identity, not value comparison.

### The assertion

```python
assert len(pseudo_q_params) > 0, "No pseudo_query parameters found."
```

If the model has no AttnRes modules (e.g. if `use_full_attnres=False` were
added), this assertion catches it at startup rather than silently training with
a wrong optimizer setup.

### The printed summary

At startup you will see:
```
Optimizer param groups:
  Group 1 (weight-decay):    47 tensors  lr=0.0003  wd=0.1
  Group 2 (no-decay):        73 tensors  lr=0.0003  wd=0.0
  Group 3 (pseudo_query):     6 tensors  lr=0.0006  wd=0.1  [2x base LR]
```

For the 3-layer test model: 6 AttnRes instances → 6 pseudo_query tensors in
Group 3. In the 16-layer production model: 32 AttnRes instances → 32 in Group 3.

---

## 19. LR Schedule — WSD (Warmup → Stable → Decay)

**File:** `slm_project/training/lr_schedule.py`

### Why not use cosine annealing?

The most common LR schedule for transformers is **cosine annealing**: the LR
follows a cosine curve from peak to near-zero over a fixed number of steps, then
training stops. This is simple but has a critical flaw: you must know the total
number of training steps in advance.

In practice, you don't know the optimal stopping point before training starts.
If you set `total_steps` too small, the model hasn't converged when you stop.
If you set it too large, you waste compute at a near-zero LR where learning
has essentially stopped.

**WSD (Warmup-Stable-Decay)** solves this by separating the schedule into three
independent phases with the decay triggered by a data-driven criterion (plateau
in validation loss), not a fixed step count.

### Phase 1: Linear Warmup

```python
if step < tcfg.warmup_steps:           # warmup_steps = 2000
    return peak_lr * step / warmup_steps
```

```
Step    0:  LR = 0.0
Step  500:  LR = 0.000075  (25% of peak)
Step 1000:  LR = 0.00015   (50% of peak)
Step 2000:  LR = 0.0003    (100% of peak = peak_lr)
```

**Why warmup?** At step 0, the model is randomly initialised. Gradients are
large and noisy. Starting with a high LR at this point causes large chaotic
weight updates and often sends the model to a very bad local minimum from which
recovery is slow. Warmup gives the model time to "get its bearings" with small
steps, then accelerates once the gradient direction is more reliable.

2000 warmup steps at batch size 262,144 tokens = 512M tokens seen before
reaching peak LR. For a 125M parameter model, this is appropriate.

### Phase 2: Stable Phase

```python
if decay_triggered_at is None:
    return peak_lr                     # hold at peak indefinitely
```

This is the "train until you stop improving" phase. The LR stays at `peak_lr`
for as long as validation loss (perplexity) keeps decreasing. The trainer
monitors val-PPL every `eval_freq = 200` steps. If it hasn't improved for
`plateau_steps = 3000` consecutive steps, AND the model has been trained for
at least `min_pretrain_steps = 70,000` steps, the decay phase is triggered.

**Why min_pretrain_steps = 70,000?**
Early in training, validation loss can plateau temporarily (the model is
figuring out a particular concept and the loss stagnates before the next
drop). Triggering decay too early would cap the LR before the model has
a chance to recover from these plateaus. 70,000 steps guarantees the model
has seen a substantial amount of data first.

### Phase 3: Linear Decay

```python
steps_since_decay = step - decay_triggered_at
progress = steps_since_decay / DECAY_DURATION   # DECAY_DURATION = 2000

if progress >= 1.0:
    return min_lr                      # clamp — return directly, no fp math
return peak_lr - (peak_lr - min_lr) * progress
```

Once triggered, LR decays linearly over exactly 2,000 steps:
```
Trigger + 0:    LR = 0.0003  (peak_lr)
Trigger + 500:  LR = 0.000225
Trigger + 1000: LR = 0.00015
Trigger + 1500: LR = 0.000075
Trigger + 2000: LR = 0.00003 (min_lr = 10% of peak)
Trigger + any:  LR = 0.00003 (holds here forever)
```

**Why return `min_lr` directly at `progress >= 1.0`?**

Floating-point arithmetic accumulates rounding errors. If you compute:
```python
peak_lr - (peak_lr - min_lr) × 1.0
```
you might get `3e-5 - (3e-4 - 3e-5) × 1.0 = 2.9999999e-5` instead of exactly
`3e-5`. Over thousands of steps at min_lr, this tiny error would compound and
cause the LR to drift slightly. The direct `return min_lr` guard prevents this.

### apply_lr() — maintaining the 2× ratio at every step

```python
def apply_lr(optimizer, lr: float):
    for i, group in enumerate(optimizer.param_groups):
        if i == 2:             # Group 3: pseudo_query
            group['lr'] = lr * 2.0
        else:
            group['lr'] = lr
```

Every time the LR is updated (every step), `apply_lr` writes the new value to
all three param groups, maintaining the `2×` ratio for pseudo_query at every
point in training. If this function were skipped for even one step, the LR
ratio would be momentarily wrong.

The trainer calls this at the start of every optimizer step:
```python
lr = get_lr(step, tcfg, decay_triggered_at)
apply_lr(optimizer, lr)
```

### The `decay_triggered_at` state

This is saved in every checkpoint:
```python
torch.save({
    ...
    'decay_triggered_at': trainer.decay_triggered_at,  # None or int
}, ckpt_path)
```

On resume, `decay_triggered_at` is loaded and passed back to `get_lr()`. This
ensures the LR curve is continuous across restarts — the model doesn't "restart"
the warmup or stable phase after a checkpoint reload.

---

*Parts 14 & 15 complete.*

---

## 20. Trainer Loop — The Training Engine

**File:** `slm_project/training/trainer.py`

### What the Trainer class manages

The `Trainer` class owns the full training loop state:
- `global_step` — optimizer steps completed (each step = `grad_accum_steps` micro-batches)
- `tokens_seen` — cumulative tokens processed (for budget enforcement)
- `decay_triggered_at` — step when WSD decay phase began (None = still stable)
- `best_val_ppl` — best validation perplexity seen (for plateau detection)
- `plateau_counter` — steps since last val-PPL improvement

### The micro-batch vs optimizer step distinction

Training uses **gradient accumulation** — a technique that simulates a large
batch on hardware that can't fit the full batch in memory:

```
For each optimizer step:
    for micro_step in range(grad_accum_steps=8):    # 8 micro-batches
        loss = model(batch) / grad_accum_steps      # DIVIDE FIRST
        loss.backward()                             # accumulate gradients
    clip_gradients()
    optimizer.step()
    optimizer.zero_grad()
```

**Why divide loss before backward?** PyTorch's `.backward()` accumulates
gradients into `param.grad`. If you call it 8 times without dividing by 8,
`param.grad` will contain the sum of 8 gradients. The optimizer interprets
this as a single gradient 8× too large — equivalent to having an effective
LR that is 8× too high. Dividing before backward makes the accumulated
gradient the **mean** across the 8 micro-batches, which is what we want.

Effective batch size:
```
physical_batch_seqs × grad_accum_steps × max_seq_len
= 1 × 8 × 256 (dev machine)  = 2,048 tokens/step
= 4 × 8 × 8192 (prod)        = 262,144 tokens/step
```

### bfloat16 autocast — why NO GradScaler

```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    _, loss = model(input_ids, labels=labels)
```

`torch.autocast` automatically casts tensor operations to bfloat16 where safe
(matrix multiplications, attention) while keeping others in float32 (softmax,
loss computation). It does NOT touch the model weights — those stay in whatever
dtype they were created in (float32 by default).

**GradScaler is for float16 ONLY.** float16 has only 5 exponent bits, meaning
gradients smaller than ~6e-5 underflow to zero. GradScaler works around this by
scaling the loss up before backward (so gradients are large enough to represent
in float16), then scaling the optimizer step back down.

**bfloat16 has 8 exponent bits** (same as float32). Gradients never underflow
in bfloat16. GradScaler is not only unnecessary — using it with bfloat16 would
add noise by artificially inflating then deflating all gradients.

### Gradient clipping

```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This computes the global L2 norm across all parameters and, if it exceeds 1.0,
scales all gradients down proportionally so the total norm equals exactly 1.0.

**Why clip?** Occasionally, a particularly "surprised" batch (e.g. a document
with an unusual token sequence) produces very large gradients that would send
the weights to a bad place in one step. Clipping at 1.0 limits the maximum
"damage" any single batch can do. The threshold 1.0 is empirically validated
across GPT-2, LLaMA, and the vast majority of LLM training runs.

After clipping, if `total_norm` is NaN or Inf, the model is in an unstable
state — this is logged as a warning with a specific diagnostic hint (usually:
are you accidentally using GradScaler with bfloat16?).

### Checkpoint format — everything needed to resume

```python
payload = {
    'global_step':        self.global_step,      # which step we're at
    'tokens_seen':        self.tokens_seen,       # total tokens processed
    'model_state':        model.state_dict(),     # all ~44M weight tensors
    'optimizer_state':    optimizer.state_dict(), # Adam m, v moment buffers
    'val_ppl':            val_ppl,               # last validation PPL
    'decay_triggered_at': self.decay_triggered_at,# LR schedule state
    'data_global_idx':    step × batch × accum,  # where to resume data from
}
```

The `optimizer_state` is often overlooked but is critical. AdamW's `m` (first
moment) and `v` (second moment) buffers are the same size as the model weights.
Without them, resuming training means the optimizer "forgets" its gradient
history — the first ~2000 steps after resume will have an incorrect effective
LR because the moment buffers are cold.

Checkpoint files are saved to `checkpoints/step_0000500.pt` with zero-padded
step numbers. This ensures alphabetical and numerical ordering are the same,
making it trivial to find the latest checkpoint.

### The log line — what every field means

```
step=   300 | loss=9.8914 | lr=3.00e-04 | grad=1.000 | pq_mean=0.0427 | tok=0.60M | t=28s
```

| Field | Meaning |
|---|---|
| `step` | Optimizer steps completed (not micro-steps) |
| `loss` | Mean loss over last `grad_accum_steps` micro-batches (unscaled) |
| `lr` | Current base LR (from param_groups[0] — Groups 1 & 2) |
| `grad` | Pre-clip gradient norm. If always = 1.000, clipping every step (normal for early training) |
| `pq_mean` | Mean pseudo_query L2 norm across all 6 AttnRes instances. Should grow from 0.0 |
| `tok` | Millions of tokens seen so far |
| `t` | Seconds elapsed since training started |

### The `zero_grad(set_to_none=True)` detail

```python
optimizer.zero_grad(set_to_none=True)
```

`set_to_none=True` deallocates the gradient tensors entirely after each optimizer
step, rather than filling them with zeros. This saves memory (one less full-size
copy of the model parameters) and is marginally faster because the subsequent
`backward()` allocates new gradient tensors rather than writing to existing ones.

---

## 21. Phase 1 Go/No-Go Gates

**File:** `slm_project/training/trainer.py` → `evaluate_phase1_gates()`

The gates are a structured diagnostic framework that makes it impossible to
accidentally proceed to the full 125M training with a broken pipeline. All 8
gates must pass before scaling up.

### Gate 1 — Initial loss = ln(vocab_size)

**Checked at:** step 0, before any weight update.

**Criterion:** `10.3 ≤ loss ≤ 10.5`

**Why this specific range:** With a randomly initialised model, the output
logits are approximately uniform across all 32,010 vocabulary entries. The
cross-entropy loss for uniform predictions is:
```
loss = -log(1/32010) = log(32010) ≈ 10.372
```

The range [10.3, 10.5] accounts for the slight non-uniformity from std=0.02
Gaussian initialisation.

**If loss < 9.0:** The tokenizer has a bug — some tokens are appearing far
more often in the output than others. Usually caused by a vocab_size mismatch
(the embedding table covers fewer tokens than the tokenizer produces).

**If loss > 10.5:** The model is producing skewed logits even before training.
Usually caused by wrong `std` in `standard_init` — if `std` is too large, some
logits are very large, making softmax nearly one-hot.

### Gate 2 — Loss decreasing by step 300

**Checked at:** step 300.

**Criterion:** `loss < 10.3` (must beat the initial random loss)

**What it proves:** The gradient is flowing correctly and the model is
learning something from the data. If loss at step 300 is still ≥ 10.3, the
model is not improving. Most likely causes:

- **Gradient accumulation bug:** Loss not divided by `grad_accum_steps` before
  backward → effective LR is `grad_accum_steps × peak_lr` → immediate gradient
  explosion followed by NaN weights
- **Data bug:** Training on garbage data (e.g. all-zero token sequences from
  a corrupt shard)
- **Wrong optimizer:** GradScaler applied to bfloat16 → gradients are
  artificially deflated → near-zero updates

### Gate 3 — All pseudo_query norms = 0.0 at step 0

**Checked at:** step 0, before any weight update.

**Criterion:** All 6 pseudo_query norms must be exactly 0.0 (within 1e-8).

**What it proves:** `init_model_weights()` was called in the correct order
with `zero_attnres_pseudo_queries` last.

**If any norm > 0:** `init_model_weights()` was not called, was called in
the wrong order, or another `model.apply()` call ran afterward and
overwrote the zeros with random values.

### Gate 4 — pseudo_query norms growing by step 300

**Checked at:** step 300.

**Criterion:** All active norms (indices 1–5) in range `[0.001, 2.0]`.
Index 0 is excluded — the first AttnRes only ever sees `[v₀]` (one input),
so `softmax([x]) = 1.0` regardless of pseudo_query. Its gradient is always
zero. **Norm staying at 0.0 for index 0 is correct, not a bug.**

**If norms stuck near 0.0 (indices 1–5):** pseudo_query is not in Group 3
of the optimizer. It's getting the wrong LR (3e-4 instead of 6e-4). The
zero-init saddle point is not being escaped fast enough.

**If norms > 2.0:** The key_norm in AttnRes is not functioning correctly,
or softmax is being applied on the wrong dimension (dim=1 or dim=2 instead
of dim=0). Without normalisation, large attention logits → softmax spikes →
pseudo_query explodes trying to reach the spike.

### Gates 5 & 6 — Inline checks during training

**Gate 5 (gradient finiteness):** Checked inside `optimizer_step()`. If
`total_norm` is NaN or Inf, a warning is printed. Training continues (the
gradient is clipped to 1.0 which handles the NaN), but repeated NaN gradients
indicate a deep problem.

**Gate 6 (layer_outputs count = 7):** Enforced by the assertion in
`model.forward()` on every single forward pass:
```python
assert len(layer_outputs) == 1 + 2 * n_layers   # 7 for 3-layer model
```
If this assertion fires, a block is not appending exactly 2 tensors to
`layer_outputs`. This catches mis-wired blocks immediately rather than
producing subtly wrong outputs.

### What to do if a gate fails

| Gate Failed | Where to Look | What to Check |
|---|---|---|
| 1 (loss range) | `tokenizer_utils.py` | Re-run Stage 2 10-token verification |
| 2 (loss decreasing) | `trainer.py` | Is loss divided by `grad_accum_steps` before `.backward()`? |
| 3 (pq norms at step 0) | `init_weights.py` | Is `zero_attnres_pseudo_queries` the LAST `apply()` call? |
| 4 (pq norms stuck) | `optimizer.py` | Are 6 tensors in Group 3? Print `len(pseudo_q_params)` |
| 4 (pq norms exploding) | `attn_res.py` | Is softmax on `dim=0`? Is `key_norm` applied? |
| 5 (NaN gradient) | `trainer.py` | Remove any GradScaler. Confirm `dtype=torch.bfloat16` |
| 6 (layer_outputs count) | `block.py` | Does each block append exactly 2 tensors? |

---

*Parts 16 & 17 complete.*

---

## 22. SFT Format & Loss Masking

**File:** `sft_format_test.py`

### What SFT is

**Supervised Fine-Tuning (SFT)** is the step after pre-training where the model
learns to follow instructions and have conversations. During pre-training, the
model learns language from raw text. During SFT, it learns how to respond to
users, follow system prompts, use tools, and know when to stop.

SFT uses **labelled conversations** — datasets where every turn is explicitly
marked with a role (system / user / assistant) and the correct response is known.
The model only trains on the **assistant responses** — it would be pointless
(and harmful) to train it on the user's words, because we don't want the model
to predict what the user will say next; we want it to predict what a helpful
assistant would say.

### The chat template

Every conversation is serialised using the special tokens defined in Section 3:

```
<|system|>You are a helpful assistant.<|end|>
<|user|>What is 2+2?<|end|>
<|assistant|>The answer is 4.<|end|>
<|user|>Explain why.<|end|>
<|assistant|>Because 2+2=4 by definition of addition.<|end|>
```

Each turn ends with `<|end|>` (EOS, ID 32005). The model learns:
- After `<|assistant|>`, emit the correct response, then emit `<|end|>`
- The `<|end|>` in an assistant turn is the "stop signal" — the model must learn to generate it

### The loss mask

The loss mask controls which tokens contribute to the training loss:

```
Token type            loss_mask value
─────────────────────────────────────
<|system|>            0  (structural — not generated by model)
system text           0  (model doesn't generate system prompts)
<|end|> (system)      0  (not generated by model)
<|user|>              0  (structural)
user text             0  (model doesn't generate user messages)
<|end|> (user)        0  (not generated)
<|assistant|>         0  (structural — model knows it's its turn)
assistant text        1  ← LEARN THIS
<|end|> (assistant)   1  ← LEARN TO STOP HERE
<|tool_call|> block   1  (model generates tool calls)
<|tool_result|> block 0  (tool results come from the environment, not the model)
```

During training, the cross-entropy loss is multiplied by this mask:
```python
loss = cross_entropy(logits, labels, ignore_index=-100)
# Tokens with mask=0 have their label set to -100
# ignore_index=-100 means cross_entropy skips those positions entirely
```

**Why `<|end|>` in assistant turns has mask=1:** If the model doesn't see a
training signal for the EOS token, it will never learn to stop generating.
During inference, after the assistant completes its response, you check if the
output token is `<|end|>` (ID 32005). If the model never trained to predict it,
it will generate forever.

**Why `<|assistant|>` itself has mask=0:** The role token is structural — it
marks that we're now in the assistant's turn. The model didn't choose to start
speaking; the format tells it to. However, everything the model says after that
token is its responsibility.

### Verification results (Stage 17)

The `sft_format_test.py` script verified over 2,000 SmolTalk conversations:

```
All 2,000 examples: len(input_ids) == len(loss_mask) ✓
Total assistant tokens with loss=1 : 1,340,789 tokens
EOS tokens with loss=1             : 3,695 EOS tokens
Every example has ≥ 1 assistant token with loss=1 ✓
```

**3,695 EOS tokens** means the model will see 3,695 explicit training signals to
stop generating — across 2,000 conversations. This is sufficient for reliable
stopping behaviour.

---

---

## 23. GRPO — Phase 4b.5 Reasoning RL  *(NEW)*

**File:** `slm_project/training/grpo_trainer.py`

### Why GRPO instead of PPO?

Proximal Policy Optimization (PPO) requires maintaining an active Value Model
that estimates the expected reward from any given state. This model takes up
as much VRAM as the policy model itself, effectively doubling the memory footprint
during RL.

**Group Relative Policy Optimization (GRPO)** eliminates the Value Model entirely.
Instead of comparing a generated token to a baseline predicted by a separate model,
GRPO generates a group of $G$ complete answers (we use $G=8$) for the same prompt.
It scores all 8 answers using rule-based rewards, computes the mean and standard
deviation of those 8 scores, and then normalises them:
```
advantage = (reward_i - mean(rewards)) / std(rewards)
```
Answers that scored above the group average get a positive advantage (reinforce);
answers below the average get a negative advantage (penalise). This provides a
stable, variance-reduced baseline without requiring a Value Model.

### Rule-based Rewards vs Neural Reward Models

This project strictly uses **rule-based rewards**, avoiding the "reward hacking"
vulnerabilities of neural reward models. We only train GRPO on verifiable
domains: Math and Code.

The total reward is the sum of three components:

1. **Accuracy Reward (Weight 1.0):**
   - Extract the final answer from the `<|end_think|>` block.
   - For Math: compare against the ground truth using SymPy algebraic equivalence.
   - For Code: execute the generated function against the dataset's unit tests.
   - 1.0 if correct, 0.0 if wrong.

2. **Format Reward (Weight 0.2):**
   - Does the response contain exactly one `<|think|>` and one `<|end_think|>`?
   - Is the think block placed before the final answer?
   - Provides a dense, early signal to force the model into the CoT format
     even before it learns to solve the problems correctly.

3. **Language Consistency Reward (Weight 0.1):**
   - Penalises the model if it generates non-English characters or excessive
     repetition.

### KL Divergence Penalty

While learning to maximise rewards, the model must not collapse into a narrow
distribution or "forget" its SFT capabilities. The `grpo_trainer.py` maintains
a frozen copy of the SFT checkpoint (the `ref_model`).

For every token generated, we compute:
```
kl_div = log(prob_policy / prob_ref)
```
This KL divergence is subtracted from the advantage. If the policy diverges too
far from the SFT reference, the penalty overwhelms the reward. If the mean KL
divergence exceeds `reward_hacking_kl_threshold = 0.5`, training halts early.

---

## 24. Complete Training Pipeline — End to End

Here is the complete picture from raw text to trained model:

```
┌─────────────────────────────────────────────────────────────────────┐
│  SETUP (one-time)                                                   │
│                                                                     │
│  1. build_and_save_tokenizer()                                      │
│     Mistral-7B base (32K) + 10 special tokens → 32,010 vocab       │
│     Saved to tokenizer/ directory                                   │
│                                                                     │
│  2. verify_all_special_tokens()                                     │
│     Each special token → exactly 1 ID, matching EXPECTED_IDS       │
│     FAIL → stop, do not proceed                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  DATA PREPARATION (one-time per phase)                              │
│                                                                     │
│  3. download_phase1()                                               │
│     Stream FineWeb-Edu (budget: 20B tokens for production)         │
│     Stream Wikipedia EN (budget: 4B tokens)                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  MODEL CONSTRUCTION                                                 │
│                                                                     │
│  4. SLM(cfg, tcfg)                                                  │
│     Embedding:     [32010, 768]                                     │
│     RoPE:          cos/sin cache for 8192 positions                 │
│     Block 0,1,2:   LOCAL  (AttnRes → GQA-SWA → AttnRes → FFN)     │
│     Block 3:       GLOBAL (AttnRes → IHA-NoPE → AttnRes → FFN)    │
│     [... repeats 1 every 4 ...]                                     │
│     mtp_head:      RMSNorm → W_mtp(768×768) → GeLU → embedding.T  │
│                                                                     │
│  5. init_model_weights(model)                                       │
│     model.apply(standard_init)          → Gaussian(0, 0.02)       │
│     model.apply(zero_attnres_queries)   → all pseudo_query = 0    │
│     model.apply(reinit_iha_identity)    → alpha_Q/K/V = identity  │
│     assert all pseudo_query.norm() == 0 → verified                 │
│                                                                     │
│  6. build_optimizer(model, tcfg)                                    │
│     Group 1: 2D+ tensors & IHA     → LR=3e-4, wd=0.1              │
│     Group 2: 1D tensors (Norms)    → LR=3e-4, wd=0.0              │
│     Group 3: pseudo_query ONLY     → LR=6e-4, wd=0.1  [MANDATORY] │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  TRAINING PHASES (Sequential Execution)                             │
│                                                                     │
│  PHASE 1–3: PRETRAINING (trainer.py)                               │
│     - Next token prediction + MTP loss                             │
│     - 20B tokens on FineWeb + Wikipedia                            │
│     - WSD schedule (decay triggers when val-PPL plateaus)          │
│                                                                     │
│  PHASE 4A & 4B: SUPERVISED FINE-TUNING (finetune.py)               │
│     - 4a: General instruction following (SmolTalk)                 │
│     - 4b: Chain-of-Thought SFT (NuminaMath CoT)                    │
│     - Strict formatting with <|system|>, <|user|>, <|assistant|>  │
│     - Loss mask = 1 ONLY on assistant turns and <|end|> token      │
│                                                                     │
│  PHASE 4B.5: GRPO REASONING RL (grpo_trainer.py)                   │
│     - Load Phase 4b checkpoint. Freeze copy as reference model.    │
│     - Math & Code only. Rule-based rewards. G=8.                   │
│     - Train for 700 steps max, or until reward > 0.75.             │
│                                                                     │
│  PHASE 4C & 4D: DOMAIN ALIGNMENT                                   │
│     - 4c: Domain specific SFT (loads GRPO checkpoint)              │
│     - 4d: DPO (Direct Preference Optimization) for safety/tone     │
│                                                                     │
│  PHASE 5.5: LONG CONTEXT EXTENSION (phase55_extend.py)             │
│     - SWA window increased. RoPE base handles 32K naturally.       │
│     - Train on 16K length documents.                               │
└─────────────────────────────────────────────────────────────────────┘
```

### What the gradient signal teaches each component

| Component | What gradient from main loss teaches | What gradient from MTP teaches |
|---|---|---|
| Embedding | Which token representations are useful | Which representations help predict 2-ahead |
| AttnRes pseudo_query | Which layers to weight for each token type | Same signal, 2-step lookahead |
| GQA (W_Q, W_K, W_V) | Which tokens attend to which | Longer-range attention patterns |
| SwiGLU FFN | Which features to transform and how | Feature transformations for prediction |
| MTP head (W_mtp) | — | Directly trained on t+2 prediction |

---

## 25. How to Run Everything

### Prerequisites

```powershell
# Create virtual environment
python -m venv slm
slm\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets numpy pytest

# On training machine only (12GB+ VRAM):
pip install flash-attn --no-build-isolation
```

### Step-by-step from scratch

```powershell
# ── Stage 0: Environment check ──────────────────────────────────────────
python stage0_verify.py
# Expected: All imports OK, CUDA available (or CPU fallback)

# ── Stage 1: Build tokenizer (one-time, needs internet) ─────────────────
python -c "
from slm_project.tokenizer_utils import build_and_save_tokenizer, verify_all_special_tokens
tok = build_and_save_tokenizer()
verify_all_special_tokens(tok)
"
# Expected: tokenizer/ directory created, all 10 tokens verified

# ── Stage 2: Run full test suite ────────────────────────────────────────
cd slm_project
pytest tests/ -v --tb=short
# Expected: 22 tests, ALL PASS, 0 failures

# ── Stage 3: Download Phase 1 data (needs internet, ~5-10 min) ──────────
python -c "from slm_project.data.download import download_phase1; download_phase1()"
# Expected: 10 shards fineweb + 3 shards wikipedia saved to data/shards/

# ── Stage 4: Run Phase 1 training ───────────────────────────────────────
python phase1_train.py
# Expected output:
#   Weight init verified: all pseudo_queries = 0.0
#   Optimizer param groups: ... Group 3 (pseudo_query): 6 tensors lr=0.0006
#   PHASE 1 GO/NO-GO GATE CHECK — step 0
#     [PASS] Gate 1: loss @ step 0 = 10.xxxx  (10.3 – 10.5)
#     [PASS] Gate 3: all 6 pseudo_query norms = 0.0
#   step=   10 | loss=10.xxxx | lr=1.50e-06 | grad=1.000 | pq_mean=0.0000 | tok=0.00M
#   ...
#   PHASE 1 GO/NO-GO GATE CHECK — step 300
#     [PASS] Gate 2: loss @ step 300 = 9.xxxx  (< 10.3)
#     [PASS] Gate 4: pseudo_query norms (indices 1-5) in [0.001, 2.0]
#   ...
#   Token budget reached: 2,500,000 tokens.
#   Training complete.

# ── Stage 5: SFT format verification ────────────────────────────────────
python sft_format_test.py
# Expected:
#   [OK] All role tokens encode to single IDs
#   [OK] All 2000 examples: len(input_ids) == len(loss_mask)
#   [OK] Total assistant tokens with loss=1 : 1,340,789
#   STAGE 17 PASSED  (SFT format + loss mask verified)
```

### Scaling to 125M (production)

Change only these values in `config.py`:

```python
# ModelConfig
n_layers:    int = 16         # was 3
global_layers: Tuple = (3, 7, 11, 15)  # one global per 4 layers

# TrainConfig
physical_batch_seqs: int = 4   # was 1 (requires 12GB+ VRAM)
# max_seq_len stays at 8192 (was reduced to 256 for dev)
```

Set `max_seq_len=8192` back in `phase1_train.py`:
```python
mcfg = ModelConfig()           # max_seq_len=8192 (default)
tcfg = TrainConfig(physical_batch_seqs=4)
```

Everything else — the training loop, the optimizer, the checkpoint format,
the data pipeline — is identical. This is the design principle: **the only
difference between the smoke test and the production run is a config change.**

### Resuming from a checkpoint

```python
ckpt = torch.load('checkpoints/step_0000500.pt')

model = SLM(mcfg, tcfg).to(device)
model.load_state_dict(ckpt['model_state'])

optimizer = build_optimizer(model, tcfg)
optimizer.load_state_dict(ckpt['optimizer_state'])

dataset = ShardedDataset(
    'data/shards/*.bin',
    seq_len=mcfg.max_seq_len,
    start_global_idx=ckpt['data_global_idx'],  # ← CRITICAL
)

trainer = Trainer(model, optimizer, tcfg, loader, device)
trainer.global_step        = ckpt['global_step']
trainer.tokens_seen        = ckpt['tokens_seen']
trainer.decay_triggered_at = ckpt['decay_triggered_at']
trainer.run(max_tokens=Phase1Config().total_tokens)
```

### Key things that will break the model silently

These mistakes produce no error but corrupt training:

| Mistake | Symptom | How to catch |
|---|---|---|
| GradScaler with bfloat16 | Gradients near-zero, loss stalls | Gate 2 fails |
| No loss / grad_accum | Effective LR 8× too high, NaN | Gate 2 fails, NaN gradients |
| pseudo_query in Group 1 or 2 | pq norms stuck or decaying | Gate 4 fails |
| init order wrong (apply after zero) | pq norms ≠ 0 at step 0 | Gate 3 fails |
| softmax(dim=1) in AttnRes | Attention across batch not layers | Gate 4 exploding |
| No EOS between documents | Model never learns to stop | Inference runs forever |
| Not saving data_global_idx | Data replayed on resume | Loss curve looks too good |
| gamma=0.2887 in RMSNorm | Outputs under-scaled, slow learning | Gradual quality loss |
| Wrong rope_base (10K not 500K) | Context > 8K broken | Long-context tasks fail |

---

## ✅ Phase 1 Results (Verified)

The 3-layer smoke test passed all 8 gates on the dev machine (RTX 3050 4.3 GB VRAM):

| Gate | Criterion | Result |
|---|---|---|
| 1 | Initial loss = 10.50 (expected 10.3–10.5) | **PASS** |
| 2 | Loss @ step 300 = 9.89 (< 10.3) | **PASS** |
| 3 | All 6 pseudo_query norms = 0.0 at step 0 | **PASS** |
| 4 | pq norms 1–5 = [0.069, 0.066, 0.058, 0.050, 0.051] | **PASS** |
| 5 | Gradient norms finite throughout (clipped at 1.0) | **PASS** |
| 6 | layer_outputs == 7 on every forward pass | **PASS** |
| 7 | 1,340,789 assistant tokens masked correctly | **PASS** |
| 8 | No OOM — 4.3 GB VRAM throughout 976 steps | **PASS** |

**Total training:** 976 steps, 2,500,000 tokens, ~641 seconds (~10.7 minutes).

**The codebase is verified. All systems are go for Phase 5: 125M parameter production training (125,931,008 base + IHA parameters).**

---

*README complete. All 25 sections written.*

---

## 🚀 Getting Started — Installation & Training Guide

This section walks you through every step from a fresh clone to a verified
Phase 1 training run. Follow the steps **in order** — each step is a
prerequisite for the next.

---

### Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| **Python** | 3.10 | 3.11 |
| **GPU** | NVIDIA with ≥ 4 GB VRAM (bfloat16 capable) | RTX 5060 Ti / 12 GB+ VRAM |
| **CUDA** | 12.1 | 12.4 |
| **Disk space** | 1 GB (smoke-test shards only) | 5 GB (room for checkpoints) |
| **RAM** | 8 GB | 16 GB |

> **⚠ bfloat16 is mandatory.**  
> This model uses `torch.autocast(dtype=torch.bfloat16)` with NO GradScaler.  
> bfloat16 requires an NVIDIA Ampere GPU (RTX 30xx) or newer.  
> Turing (RTX 20xx) and older cards do NOT support bfloat16 — training will crash.

---

### Step 0 — Clone the repository

```bash
git clone <your-repo-url>
cd SLM
```

---

### Step 1 — Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv slm_env
slm_env\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python -m venv slm_env
source slm_env/bin/activate
```

> **Note:** The `slm/` folder in this repo is a pre-existing venv used during
> development. It is listed in `.gitignore`. Always create your own fresh venv
> as shown above.

---

### Step 2 — Install all dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs PyTorch, HuggingFace Transformers, Datasets, NumPy,
SentencePiece, and the test suite. No other installs are needed.

**Verify PyTorch sees your GPU:**
```python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:
```
True
NVIDIA GeForce RTX XXXX
```

If `False` is printed, your PyTorch build does not match your CUDA version.
Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and
install the correct wheel for your CUDA version.

---

### Step 3 — Build and verify the tokenizer (one-time, ~30 seconds)

The tokenizer extends Mistral v0.1's vocabulary with 10 special tokens.
This downloads only the tokenizer files (~500 KB), **not** any model weights.

```bash
python -c "
from slm_project.tokenizer_utils import build_and_save_tokenizer, verify_all_special_tokens
tok = build_and_save_tokenizer()
verify_all_special_tokens(tok)
"
```

Expected output:
```
Loading base tokenizer from mistralai/Mistral-7B-v0.1 ...
Adding 10 special tokens ...
Tokenizer saved to 'tokenizer/'
Running full 10-token co-verification ...
All 10 special tokens verified OK
Total vocab size: 32010 OK
```

> **⚠ Do NOT skip the verification step.**  
> If any special token gets split by SentencePiece into multiple sub-pieces,
> training will silently produce wrong loss masks. The assertion will catch this
> before you waste any GPU time.

The saved files will appear in `tokenizer/`:
```
tokenizer/
├── tokenizer.json          ← 32,010-token vocabulary (~3.5 MB)
├── tokenizer_config.json
└── special_tokens_map.json
```

---

### Step 4 — Download and tokenise the training data (one-time, ~2–5 minutes)

This streams from HuggingFace and stops **early** at the token budget.
It does **not** download the full dataset (which would be ~10B tokens).

```bash
python -c "from slm_project.data.download import download_phase1; download_phase1()"
```

Expected output:
```
Streaming HuggingFaceFW/fineweb-edu [sample-10BT]  (budget: 2,000,000 tokens) ...
  Saved shard 0000: 200,000 tokens → data/shards/fineweb_edu_shard0000.bin
  Saved shard 0001: 200,000 tokens → data/shards/fineweb_edu_shard0001.bin
  ...
Done: 2,000,000 tokens in 10 shard(s).

Streaming wikimedia/wikipedia [20241101.en]  (budget: 500,000 tokens) ...
  Saved shard 0000: 200,000 tokens → data/shards/wikipedia_en_shard0000.bin
  ...
Done: 500,000 tokens in 3 shard(s).

Phase 1 data download complete.
```

After this you will have **13 binary shard files** in `data/shards/`, totalling
~2.5 million tokens (~5 MB on disk).

> **⚠ Never call `load_dataset()` without `streaming=True` on these datasets.**  
> FineWeb-Edu sample-10BT is 10 billion tokens (~20 GB). The streamer stops
> after exactly 2M tokens. With `streaming=False` it would attempt to download
> the entire dataset.

---

### Step 5 — Run the environment smoke-test (optional but recommended)

```bash
python stage0_verify.py
```

This checks that all imports resolve, CUDA is available, bfloat16 autocast
works, and the tokenizer loads correctly. All checks should pass before training.

---

### Step 6 — Run the automated test suite

```bash
pytest tests/ -v
```

All **24 tests** must pass before starting training. If any fail, fix the
issue before proceeding — the tests catch shape bugs, weight-init errors, and
gradient flow problems that would silently corrupt training.

Expected:
```
24 passed in Xs
```

---

### Step 7 — Start Phase 1 training

```bash
python phase1_train.py
```

The script will:
1. Print device and VRAM info
2. Build the model (~44M params, 3 layers)
3. Run the Phase 1 Gate 3 check (pseudo_query norms = 0)
4. Start the training loop
5. Check Gate 1 (initial loss ≈ 10.37) at step 0
6. Check Gate 2 + Gate 4 (loss < 10.3, pq norms healthy) at step 300
7. Save checkpoints every 500 steps to `checkpoints/`
8. Print a PASS/FAIL summary when the 2.5M token budget is exhausted

**Expected log output (first few lines):**
```
Device : cuda
GPU    : NVIDIA GeForce RTX XXXX
VRAM   : 12.0 GB

3-layer SLM: 44,066,304 params (44.1M)
seq_len    : 256
phys_batch : 1 seqs
grad_accum : 8

Optimizer param groups:
  Group 1 (weight-decay):   XX tensors  lr=0.0003  wd=0.1
  Group 2 (no-decay):       XX tensors  lr=0.0003  wd=0.0
  Group 3 (pseudo_query):    6 tensors  lr=0.0006  wd=0.1  [2x base LR]

Data shards: 13 file(s) found
...
Training started.  Device: cuda
Effective batch: 1 seqs × 8 accum × 256 tokens = 2,048 tokens/step

============================================================
PHASE 1 GO/NO-GO GATE CHECK — step 0
============================================================
  [PASS] Gate 3: all 6 pseudo_query norms = 0.0
  [PASS] Gate 1: loss @ step 0 = 10.XXXX  (10.3 – 10.5)
============================================================

step=    10 | loss=10.XXXX | lr=1.50e-06 | grad=X.XXX | ...
step=    20 | loss=10.XXXX | lr=3.00e-06 | grad=X.XXX | ...
```

**Expected training duration (Phase 1 smoke test):**

| Machine | GPU | seq_len | ~Time |
|---|---|---|---|
| Dev laptop | RTX 3050 4 GB | 256 | ~10 minutes |
| Training machine | RTX 5060 Ti 12 GB | 8192 | ~30–60 minutes |

**At step 300, gates are evaluated:**
```
============================================================
PHASE 1 GO/NO-GO GATE CHECK — step 300
============================================================
  [PASS] Gate 2: loss @ step 300 = X.XXXX  (< 10.3)
  [PASS] Gate 4: pseudo_query norms (indices 1-5) in [0.001, 2.0]: [...]
============================================================
```

**Final output (success):**
```
============================================================
PHASE 1 COMPLETE — All Go/No-Go gates passed.
Pipeline verified. Ready to scale to 125M.
============================================================
```

---

### Resuming from a checkpoint

If training is interrupted, resume by loading the latest checkpoint.
Add this to `phase1_train.py` before `trainer.run()`:

```python
import glob, torch

ckpts = sorted(glob.glob('checkpoints/step_*.pt'))
if ckpts:
    ckpt = torch.load(ckpts[-1], map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    trainer.global_step        = ckpt['global_step']
    trainer.tokens_seen        = ckpt['tokens_seen']
    trainer.decay_triggered_at = ckpt['decay_triggered_at']
    dataset.seek(ckpt['data_global_idx'])   # resume from correct shard position
    print(f"Resumed from {ckpts[-1]}  (step {ckpt['global_step']})")
```

> **⚠ `data_global_idx` must be restored.**  
> Without it, the dataloader restarts from shard 0 and the model sees the same
> data twice, making the loss curve look artificially good while validation
> perplexity stays high.

---

### Scaling to Phase 5 (125M parameters)

To go from the 3-layer smoke test to the 16-layer production model, change
**only** the config values — the training code is identical:

```python
# In phase1_train.py, replace the config section:
cfg.n_layers      = 16
cfg.global_layers = (3, 7, 11, 15)
cfg.max_seq_len   = 8192

tcfg.physical_batch_seqs = 4    # requires ≥ 12 GB VRAM
tcfg.grad_accum_steps    = 8    # effective batch = 262,144 tokens

# Data: increase to 300M tokens
#   fineweb_tokens  = 240_000_000
#   wikipedia_tokens =  60_000_000
```

---

### Common errors and fixes

| Error | Likely cause | Fix |
|---|---|---|
| `No data shards found` | Step 4 not run | Run `download_phase1()` |
| `AssertionError: No pseudo_query parameters` | Wrong model class | Ensure `from slm_project.model.model import SLM` |
| `CUDA out of memory` | VRAM too small | Reduce `physical_batch_seqs=1` and `max_seq_len=128` |
| `loss @ step 0 = 0.XXXX` (too low) | `init_model_weights()` not called | Call `init_model_weights(model)` after construction |
| `loss @ step 0 = 10.37` but Gate 1 FAILS | Gate thresholds wrong | Check `Phase1Config.expected_loss_step0_lo/hi` |
| `[FAIL] Gate 4: norms stuck ≈ 0` | pseudo_query not in optimizer Group 3 | Use `build_optimizer()` — do not build AdamW manually |
| `AssertionError: Expected vocab_size=32,010` | Tokenizer not built | Run Step 3 |
| `bfloat16 not supported` | GPU is pre-Ampere | Use a RTX 30xx or newer GPU |








