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


def init_model_weights(model: nn.Module) -> None:
    """
    Full initialisation sequence. Call exactly once before training begins.

    Step 1 — standard_init  : Gaussian(0, 0.02) for all Linear + Embedding.
    Step 2 — zero_pseudo     : Zero every AttnRes.pseudo_query.  MUST BE LAST.
    Step 3 — verification    : Assert all pseudo_queries are zero.
    """
    # Step 1: standard Gaussian init for all layers
    model.apply(standard_init)

    # Step 2: zero all AttnRes pseudo_queries — MUST be the final apply() call
    model.apply(zero_attnres_pseudo_queries)

    # Step 3: verify the zeroing held (catches any ordering mistakes)
    from slm_project.model.attn_res import AttnRes
    for name, module in model.named_modules():
        if isinstance(module, AttnRes):
            assert module.pseudo_query.allclose(
                torch.zeros_like(module.pseudo_query)
            ), f"pseudo_query not zero in '{name}' after init!"

    print("Weight init verified: all pseudo_queries = 0.0")
