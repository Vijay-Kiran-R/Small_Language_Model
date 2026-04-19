# slm_project/training/lr_schedule.py
"""
WSD (Warmup → Stable → Decay) learning rate schedule.

Phase 1 — Warmup (steps 0 → warmup_steps):
  Linear ramp from 0 → peak_lr.

Phase 2 — Stable (warmup_steps → decay trigger):
  Hold peak_lr indefinitely while val PPL is still improving.
  NEVER trigger decay before min_pretrain_steps.

Phase 3 — Decay (decay_triggered_at → decay_triggered_at + 2000):
  Linear decay from peak_lr → min_lr over 2,000 steps, then hold min_lr.

Decay trigger criteria (caller / trainer loop is responsible):
  - step > tcfg.min_pretrain_steps  (70 K default)
  - val PPL has not improved for tcfg.plateau_steps consecutive steps (3 K)
  BOTH conditions must be true simultaneously.
  Record the step at which they first both hold as decay_triggered_at.

This module only computes the LR given the current step + trigger state.
It does NOT decide when to trigger decay.
"""

from typing import Optional
from slm_project.config import TrainConfig

# Fixed decay duration — number of steps from peak_lr → min_lr
DECAY_DURATION = 2_000


def get_lr(
    step:               int,
    tcfg:               TrainConfig,
    decay_triggered_at: Optional[int] = None,
) -> float:
    """
    Compute the base learning rate for the given training step.

    Args:
        step:               Current training step (0-indexed).
        tcfg:               TrainConfig — source of all schedule parameters.
        decay_triggered_at: Step at which the decay phase was triggered.
                            None → still in Warmup or Stable phase.

    Returns:
        Scalar float base LR.  Multiply by 2.0 for the pseudo_query group
        (apply_lr() does this automatically).
    """
    # ── Phase 1: Linear warmup ────────────────────────────────────────────────
    if step < tcfg.warmup_steps:
        return tcfg.peak_lr * step / tcfg.warmup_steps

    # ── Phase 2: Stable — hold until decay is triggered ──────────────────────
    if decay_triggered_at is None:
        return tcfg.peak_lr

    # ── Phase 3: Linear decay → hold at min_lr ───────────────────────────────
    steps_since_decay = step - decay_triggered_at
    progress = steps_since_decay / DECAY_DURATION
    if progress >= 1.0:
        return tcfg.min_lr
    return tcfg.peak_lr - (tcfg.peak_lr - tcfg.min_lr) * progress


def apply_lr(optimizer, lr: float) -> None:
    """
    Write a new base LR to all param groups.

    Group index 2 (pseudo_query) always receives 2× the base LR to match
    the ratio established at optimizer creation.  Groups 0 and 1 receive
    the base LR directly.

    Args:
        optimizer: AdamW instance built by build_optimizer().
        lr:        Base LR (the value returned by get_lr()).
    """
    for i, group in enumerate(optimizer.param_groups):
        if i == 2:   # Group 3: pseudo_query — always 2× base LR
            group['lr'] = lr * 2.0
        else:
            group['lr'] = lr
