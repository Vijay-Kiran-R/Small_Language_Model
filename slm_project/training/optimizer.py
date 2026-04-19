# slm_project/training/optimizer.py
"""
AdamW optimizer with three mandatory param groups.

Group 1 — Weight-decay params  (Linear weights, Embedding table)
           LR = peak_lr,  weight_decay = 0.1

Group 2 — No-weight-decay params  (RMSNorm gammas, biases)
           LR = peak_lr,  weight_decay = 0.0

Group 3 — pseudo_query params only  (AttnRes instances)
           LR = 2 × peak_lr,  weight_decay = 0.1

CRITICAL: pseudo_query MUST land in Group 3.
  - In Group 1: it decays toward zero — AttnRes collapses to uniform weights.
  - In Group 2: it learns at the wrong rate — specialisation is too slow.
  Group 3 is not optional.
"""

import torch
from torch.optim import AdamW

from slm_project.config import TrainConfig
from slm_project.model.attn_res import AttnRes


def build_optimizer(model: torch.nn.Module, tcfg: TrainConfig) -> AdamW:
    """
    Build AdamW with the three mandatory param groups.

    Args:
        model: SLM instance (after init_model_weights()).
        tcfg:  TrainConfig — source of all optimiser hyperparameters.

    Returns:
        AdamW optimiser with groups correctly separated.

    Raises:
        AssertionError: if no pseudo_query parameters are found.
    """
    # Collect IDs of all pseudo_query tensors across AttnRes instances
    pseudo_query_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, AttnRes):
            pseudo_query_ids.add(id(module.pseudo_query))

    decay_params:    list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    pseudo_q_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in pseudo_query_ids:
            pseudo_q_params.append(param)
        elif param.dim() >= 2:
            # Linear weights (2-D) and Embedding table (2-D) → weight decay
            decay_params.append(param)
        else:
            # RMSNorm gammas (1-D) and any biases → no weight decay
            no_decay_params.append(param)

    assert len(pseudo_q_params) > 0, (
        "No pseudo_query parameters found. "
        "Ensure AttnRes modules are present in the model."
    )

    pq_lr = tcfg.peak_lr * tcfg.pseudo_query_lr_multiplier
    print("Optimizer param groups:")
    print(f"  Group 1 (weight-decay):  {len(decay_params):>4} tensors  "
          f"lr={tcfg.peak_lr}  wd={tcfg.weight_decay}")
    print(f"  Group 2 (no-decay):      {len(no_decay_params):>4} tensors  "
          f"lr={tcfg.peak_lr}  wd=0.0")
    print(f"  Group 3 (pseudo_query):  {len(pseudo_q_params):>4} tensors  "
          f"lr={pq_lr}  wd={tcfg.weight_decay}  [2x base LR]")

    optimizer = AdamW(
        [
            {
                'params':       decay_params,
                'weight_decay': tcfg.weight_decay,
                'lr':           tcfg.peak_lr,
            },
            {
                'params':       no_decay_params,
                'weight_decay': 0.0,
                'lr':           tcfg.peak_lr,
            },
            {
                'params':       pseudo_q_params,
                'weight_decay': tcfg.weight_decay,
                'lr':           pq_lr,
            },
        ],
        betas=(tcfg.adam_beta1, tcfg.adam_beta2),
        eps=tcfg.adam_eps,
    )
    return optimizer
