# slm_project/model/init_weights.py
"""
Weight initialisation for the SLM.

CRITICAL ORDER — never swap:
  1. model.apply(standard_init)              — all Linear + Embedding weights
  2. model.apply(zero_attnres_pseudo_queries) — LAST — zeros all pseudo_query

If pseudo_queries are zeroed first and then apply() runs again, they get
overwritten with random weights and AttnRes breaks permanently.
"""

import torch
import torch.nn as nn


def standard_init(module: nn.Module) -> None:
    """
    Standard Gaussian init for Linear and Embedding layers.
    Called via model.apply(standard_init) as the FIRST step.

    std=0.02 matches GPT-2/LLaMA convention.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def zero_attnres_pseudo_queries(module: nn.Module) -> None:
    """
    Zero all pseudo_query vectors in every AttnRes instance.

    MUST run AFTER standard_init — i.e. AFTER model.apply(standard_init).

    Effect of zero init:
      logits = pseudo_query · K = 0  →  softmax = uniform 1/N
      AttnRes reduces to standard averaging — a stable, well-conditioned
      baseline from which training specialises.
    """
    from slm_project.model.attn_res import AttnRes
    if isinstance(module, AttnRes):
        nn.init.zeros_(module.pseudo_query)


def reinit_iha_identity(module) -> None:
    """
    Re-initialize IHA mixing tensors to identity after standard_init().
    MUST run AFTER model.apply(standard_init) has overwritten them.

    Called as the THIRD pass in init_model_weights().
    Order: standard_init → zero_attnres → reinit_iha_identity
    """
    from slm_project.model.attention import IHAGlobalAttention
    if isinstance(module, IHAGlobalAttention):
        module._init_iha_identity()


def init_model_weights(model) -> None:
    """
    Full initialisation sequence. UPDATED to include IHA identity re-init.

    CRITICAL ORDER — do not change:
    Step 1: standard_init (all linear/embedding weights)
    Step 2: zero_attnres_pseudo_queries (MUST be after step 1)
    Step 3: reinit_iha_identity (MUST be after step 1 — re-fixes IHA params)
    """
    # Step 1: Standard init for all layers
    model.apply(standard_init)

    # Step 2: Zero all AttnRes pseudo_queries — MUST BE AFTER standard_init
    model.apply(zero_attnres_pseudo_queries)

    # Step 3: Re-initialize IHA params to identity — MUST BE AFTER standard_init
    model.apply(reinit_iha_identity)

    # ── Verification ─────────────────────────────────────────
    from slm_project.model.attn_res import AttnRes
    from slm_project.model.attention import IHAGlobalAttention
    import torch

    n_pq_verified = 0
    n_iha_verified = 0

    for name, module in model.named_modules():
        if isinstance(module, AttnRes):
            assert module.pseudo_query.allclose(torch.zeros_like(module.pseudo_query)), \
                f"pseudo_query not zero in {name} after init!"
            n_pq_verified += 1

        if isinstance(module, IHAGlobalAttention):
            # Verify α_Q identity: αQ[h,h,p] = 1.0 for all h,p
            for h in range(module.n_heads_q):
                for p in range(module.P):
                    assert module.alpha_Q.data[h, h, p] == 1.0, \
                        f"alpha_Q[{h},{h},{p}] not 1.0 in {name} after init!"
            # Verify R selects pseudo j=0
            for h in range(module.n_heads_q):
                assert module.R.data[h, h * module.P] == 1.0, \
                    f"R identity not set in {name} after init!"
            n_iha_verified += 1

    print(f"Weight init verified:")
    print(f"  {n_pq_verified} pseudo_queries = 0.0 [OK]")
    print(f"  {n_iha_verified} IHA modules = identity [OK]")
