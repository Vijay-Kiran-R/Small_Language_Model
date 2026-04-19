# slm_project/model/rms_norm.py
"""
Root Mean Square Layer Normalization.

Used everywhere normalisation is needed:
  - Pre-attention norm          (3×, one per layer)
  - Pre-FFN norm                (3×, one per layer)
  - AttnRes key_norm            (6×, two per layer)
  - QK-Norm per head            (16 instances per attention layer)
  - Final RMSNorm before LM head (1×)
  - MTP head norm

Design notes
────────────
* 30 % faster than LayerNorm — no mean subtraction.
* Applied Pre-Sub-Layer (Pre-Norm pattern).
* gamma always initialised to 1.0.
  DO NOT use 0.2887 — that value compensates the O(L) magnitude growth that
  occurs in standard residual streams. AttnRes eliminates that growth, so
  gamma=1.0 is the correct initialisation here.

Formula
───────
  RMSNorm(x) = x / sqrt(mean(x²) + ε) × γ
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """
        Args:
            dim: Feature dimension (last axis of input tensor).
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(dim))   # Always init 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalised tensor, same shape as x.
        """
        # Compute RMS over the last dimension, keep dim for broadcasting
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.gamma

    def extra_repr(self) -> str:
        return f"dim={self.gamma.shape[0]}, eps={self.eps}"
