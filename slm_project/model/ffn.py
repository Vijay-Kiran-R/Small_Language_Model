# slm_project/model/ffn.py
"""
SwiGLU Feed-Forward Network.

ffn_hidden = 2048 is derived from round(8/3 × 768 / 256) × 256.
Do not change this value.

Param count per layer: 3 × (768 × 2048) = 4,718,592
  W_gate: [768, 2048]
  W_up:   [768, 2048]
  W_down: [2048, 768]

Dropout: 0.0 during pre-training, 0.1 (after W_down) during fine-tuning.
Rates always come from config — never hardcode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_project.config import ModelConfig


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    FFN(x) = Dropout( (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down )
    """

    def __init__(self, cfg: ModelConfig, dropout: float = 0.0) -> None:
        """
        Args:
            cfg:     ModelConfig — source of d_model and ffn_hidden.
            dropout: Applied after W_down.  Pass cfg.dropout_pretrain (0.0)
                     during pre-training and cfg.dropout_finetune (0.1)
                     during fine-tuning.
        """
        super().__init__()
        self.W_gate = nn.Linear(cfg.d_model,    cfg.ffn_hidden, bias=False)
        self.W_up   = nn.Linear(cfg.d_model,    cfg.ffn_hidden, bias=False)
        self.W_down = nn.Linear(cfg.ffn_hidden, cfg.d_model,    bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model]
        """
        gate = F.silu(self.W_gate(x))   # [B, T, ffn_hidden]
        up   = self.W_up(x)             # [B, T, ffn_hidden]
        out  = self.W_down(gate * up)   # [B, T, d_model]
        return self.dropout(out)
