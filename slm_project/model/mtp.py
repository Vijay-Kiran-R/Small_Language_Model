# slm_project/model/mtp.py
"""
Multi-Token Prediction (MTP) head.

Predicts token at position t+2 from the hidden state at position t,
providing extra gradient signal during pre-training.
At inference it is unused unless speculative decoding is active.

Architecture
────────────
  hidden [B, T, 768]
    → RMSNorm          [B, T, 768]
    → W_mtp [768→768] + GeLU  [B, T, 768]
    → @ EmbeddingTable.T      [B, T, 32010]

Params: 768×768 (W_mtp) + 768 (RMSNorm gamma) = 590,592

Weight tying
────────────
The projection into vocabulary space reuses the main model's embedding
table (transposed). The caller passes embedding_weight for this tying;
MTPHead owns no separate vocab projection weight.

MTP weight annealing schedule (from TrainConfig)
─────────────────────────────────────────────────
  Steps 0       → mtp_anneal_start (50k): weight = mtp_weight_start (0.3)
  Steps 50k     → mtp_anneal_end   (60k): linearly 0.3 → 0.1
  Steps 60k+                            : weight = mtp_weight_end   (0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_project.config import ModelConfig, TrainConfig
from slm_project.model.rms_norm import RMSNorm


class MTPHead(nn.Module):
    """Lightweight Multi-Token Prediction head with weight-tied vocab projection."""

    def __init__(self, cfg: ModelConfig) -> None:
        """
        Args:
            cfg: ModelConfig — source of d_model and vocab_size.
        """
        super().__init__()
        self.norm  = RMSNorm(cfg.d_model)
        self.W_mtp = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden:           [B, T, d_model] — last layer output
                              (v₆ for the 3-layer test model; v₃₂ in production).
            embedding_weight: [vocab_size, d_model] — tied embedding table.

        Returns:
            mtp_logits: [B, T, vocab_size]
        """
        x = F.gelu(self.W_mtp(self.norm(hidden)))  # [B, T, d_model]
        return x @ embedding_weight.T               # [B, T, vocab_size]


def get_mtp_weight(step: int, tcfg: TrainConfig) -> float:
    """
    Return the MTP loss coefficient for the given training step.

    Schedule:
      [0,           mtp_anneal_start] → mtp_weight_start  (0.3, constant)
      [anneal_start, mtp_anneal_end]  → linear 0.3 → 0.1
      [mtp_anneal_end,       ∞)       → mtp_weight_end    (0.1, constant)

    Args:
        step:  Current training step (0-indexed).
        tcfg:  TrainConfig — source of all schedule parameters.

    Returns:
        Scalar float MTP loss weight.
    """
    if step <= tcfg.mtp_anneal_start:
        return tcfg.mtp_weight_start

    if step <= tcfg.mtp_anneal_end:
        progress = (step - tcfg.mtp_anneal_start) / (
            tcfg.mtp_anneal_end - tcfg.mtp_anneal_start
        )
        return tcfg.mtp_weight_start - (
            tcfg.mtp_weight_start - tcfg.mtp_weight_end
        ) * progress

    return tcfg.mtp_weight_end
