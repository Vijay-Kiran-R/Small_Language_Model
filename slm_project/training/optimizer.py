# slm_project/training/optimizer.py
"""
Hybrid Muon + AdamW optimizer builder.

Param group split
─────────────────
  Group 0 — MUON:  All 2D hidden weight matrices
                   (W_Q, W_K, W_V, W_O all 16 layers; FFN gate/up/down; W_mtp)
                   EXCLUDES: embedding.weight (weight-tied, filtered by 'embed' in name)
                   EXCLUDES: IHA alpha_Q/K/V (3D tensors — Muon undefined for 3D)
                   EXCLUDES: IHA R (kept with IHA group in AdamW for consistency)
                   lr = peak_lr,  momentum = muon_momentum

  Group 1 — AdamW (no weight decay):
                   All 1D params: RMSNorm gammas, QK-norm gammas, key_norm gammas
                   IHA params (alpha_Q, alpha_K, alpha_V, R) — 3D/2D, AdamW
                   embedding.weight (weight-tied — MUST NOT go to Muon)
                   lr = peak_lr,  wd = 0.0

  Group 2 — AdamW (weight decay):
                   All remaining 2D params not caught above (edge-case catch-all)
                   lr = peak_lr,  wd = weight_decay

  Group 3 — AdamW (pseudo_query, 2× LR):
                   AttnRes.pseudo_query — zero-init 1D [768] vectors
                   CRITICAL: must be in Group 3 at 2× LR — Gate 4 checks this
                   lr = 2 × peak_lr,  wd = weight_decay

Design decisions
────────────────
1. Embedding filter: `'embed' in name` — catches `embedding.weight` which is 2D
   but MUST use AdamW (tied to LM head, sparse gradients, token manifold).
2. IHA filter: isinstance(parent_module, IHAGlobalAttention) — catches alpha_Q,
   alpha_K, alpha_V (3D tensors — Muon undefined), and R (2D but kept with group
   for consistency; only 48 params — Muon gain negligible).
3. pseudo_query: filtered first by AttnRes module ID — gets its own 2× LR group.
4. LR schedule compatibility: apply_lr() reads group index — Group 3 = pseudo_query
   at 2× always. MuonWithAuxAdam exposes the same .param_groups list.

CRITICAL — do NOT change the group ordering. The lr_schedule.apply_lr() function
addresses group index 3 (0-indexed) for the pseudo_query 2× LR. Everything
before it must match.
"""

import torch
from slm_project.config import TrainConfig
from slm_project.model.attn_res import AttnRes
from slm_project.model.attention import IHAGlobalAttention
from slm_project.training.muon import MuonWithAuxAdam


def build_optimizer(model: torch.nn.Module, tcfg: TrainConfig) -> MuonWithAuxAdam:
    """
    Build the hybrid Muon+AdamW optimizer with the four mandatory param groups.

    Args:
        model: SLM instance (after init_model_weights()).
        tcfg:  TrainConfig — source of all optimiser hyperparameters.

    Returns:
        MuonWithAuxAdam instance with groups correctly separated.

    Raises:
        AssertionError: if no pseudo_query or Muon parameters are found.
    """
    # ── Collect special parameter IDs ────────────────────────────────────────

    # pseudo_query: AttnRes.pseudo_query — needs its own 2× LR AdamW group
    pseudo_query_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, AttnRes):
            pseudo_query_ids.add(id(module.pseudo_query))

    # IHA params: alpha_Q, alpha_K, alpha_V, R — all go to AdamW
    # (alpha_Q/K/V are 3D; Muon is undefined for 3D tensors)
    iha_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, IHAGlobalAttention):
            iha_ids.add(id(module.alpha_Q))
            iha_ids.add(id(module.alpha_K))
            iha_ids.add(id(module.alpha_V))
            iha_ids.add(id(module.R))

    # ── Partition parameters ──────────────────────────────────────────────────

    muon_params:      list[torch.nn.Parameter] = []  # Group 0: Muon 2D hidden weights
    adamw_nodecay:    list[torch.nn.Parameter] = []  # Group 1: AdamW no-decay (1D + embed + IHA)
    adamw_decay:      list[torch.nn.Parameter] = []  # Group 2: AdamW decay (extra 2D catch-all)
    pseudo_q_params:  list[torch.nn.Parameter] = []  # Group 3: pseudo_query at 2× LR

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        pid = id(param)

        # Priority 1: pseudo_query → Group 3 (2× LR AdamW)
        if pid in pseudo_query_ids:
            pseudo_q_params.append(param)
            continue

        # Priority 2: IHA params → Group 1 (AdamW, no decay — small init near identity)
        if pid in iha_ids:
            adamw_nodecay.append(param)
            continue

        # Priority 3: Embedding — MUST be AdamW (weight-tied, sparse grads)
        # 'embed' catches: embedding.weight (nn.Embedding)
        if 'embed' in name:
            adamw_nodecay.append(param)
            continue

        # Priority 4: 1D params → AdamW no-decay
        # (RMSNorm gammas, QK-norm gammas, key_norm gammas)
        if param.ndim == 1:
            adamw_nodecay.append(param)
            continue

        # Priority 5: 2D (or higher) non-embed params → Muon
        # These are the linear weight matrices: W_Q, W_K, W_V, W_O, W_gate, W_up, W_down, W_mtp
        if param.ndim >= 2:
            muon_params.append(param)
            continue

        # Fallback (should never hit for this architecture — no biases, no 0D scalars)
        adamw_decay.append(param)

    # ── Sanity checks ────────────────────────────────────────────────────────
    assert len(pseudo_q_params) > 0, (
        "No pseudo_query parameters found. "
        "Ensure AttnRes modules are present in the model."
    )
    assert len(muon_params) > 0, (
        "No Muon parameters found. "
        "Check that 2D weight matrices exist and 'embed' filter is not too broad."
    )

    pq_lr = tcfg.peak_lr * tcfg.pseudo_query_lr_multiplier

    # ── Print group summary ───────────────────────────────────────────────────
    print("\nOptimizer param groups (Muon + AdamW hybrid):")
    print(f"  Group 0 (Muon):          {len(muon_params):>4} tensors  "
          f"lr={tcfg.peak_lr}  momentum={tcfg.muon_momentum}  wd={tcfg.muon_weight_decay}")
    print(f"  Group 1 (AdamW no-wd):   {len(adamw_nodecay):>4} tensors  "
          f"lr={tcfg.peak_lr}  wd=0.0  "
          f"[embed + 1D norms + IHA]")
    if adamw_decay:
        print(f"  Group 2 (AdamW decay):   {len(adamw_decay):>4} tensors  "
              f"lr={tcfg.peak_lr}  wd={tcfg.weight_decay}  [catch-all]")
    print(f"  Group 3 (pseudo_query):  {len(pseudo_q_params):>4} tensors  "
          f"lr={pq_lr}  wd={tcfg.weight_decay}  [2x base LR]")

    # ── Build param groups ────────────────────────────────────────────────────
    param_groups = [
        # Group 0: Muon — 2D hidden weight matrices
        dict(
            params=muon_params,
            use_muon=True,
            lr=tcfg.peak_lr,
            momentum=tcfg.muon_momentum,
            nesterov=True,
            ns_steps=5,
            weight_decay=tcfg.muon_weight_decay,
        ),
        # Group 1: AdamW no-decay — embeddings, 1D norms, IHA
        dict(
            params=adamw_nodecay,
            use_muon=False,
            lr=tcfg.peak_lr,
            betas=(tcfg.adam_beta1, tcfg.adam_beta2),
            eps=tcfg.adam_eps,
            weight_decay=0.0,
        ),
        # Group 2: AdamW decay — catch-all (normally empty for this architecture)
        dict(
            params=adamw_decay,
            use_muon=False,
            lr=tcfg.peak_lr,
            betas=(tcfg.adam_beta1, tcfg.adam_beta2),
            eps=tcfg.adam_eps,
            weight_decay=tcfg.weight_decay,
        ),
        # Group 3: pseudo_query at 2× LR — CRITICAL for Gate 4
        dict(
            params=pseudo_q_params,
            use_muon=False,
            lr=pq_lr,
            betas=(tcfg.adam_beta1, tcfg.adam_beta2),
            eps=tcfg.adam_eps,
            weight_decay=tcfg.weight_decay,
        ),
    ]

    optimizer = MuonWithAuxAdam(param_groups)
    return optimizer
