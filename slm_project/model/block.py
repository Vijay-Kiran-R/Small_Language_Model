# slm_project/model/block.py
"""
TransformerBlock — wires AttnRes + RMSNorm + GQA + AttnRes + RMSNorm + SwiGLUFFN.

Each block appends exactly 2 tensors to layer_outputs:
  v_{2b+1} = attn_out   (attention sub-layer output)
  v_{2b+2} = mlp_out    (FFN sub-layer output)

After N blocks, layer_outputs has 1 + 2N entries (v₀ is the embedding).

Gradient checkpointing note
───────────────────────────
Do NOT pass layer_outputs directly to torch.utils.checkpoint.checkpoint()
as a single list — that causes freed-buffer RuntimeErrors.
Use the pack/unpack tuple pattern in model.py (Stage 10).
"""

import torch
import torch.nn as nn
from typing import List, Optional

from slm_project.config import ModelConfig
from slm_project.model.attn_res import AttnRes
from slm_project.model.rms_norm import RMSNorm
from slm_project.model.attention import GroupedQueryAttention
from slm_project.model.ffn import SwiGLUFFN
from slm_project.model.rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    """
    One transformer block with two AttnRes-gated sub-layers.

    Sub-layer layout:
      1. attn_res_attn(all previous outputs) → h
      2. norm_attn(h) → attention(·) → attn_out      [appended to layer_outputs]
      3. attn_res_ffn(all outputs incl. attn_out) → h2
      4. norm_ffn(h2) → ffn(·) → mlp_out             [appended to layer_outputs]
    """

    def __init__(
        self,
        cfg: ModelConfig,
        is_global: bool = False,
        rope: Optional[RotaryEmbedding] = None,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            cfg:       ModelConfig — single source of truth.
            is_global: True  → full-sequence NoPE attention (no SWA, no RoPE).
                       False → local SWA block with RoPE.
            rope:      Shared RotaryEmbedding; must be None when is_global=True.
            dropout:   FFN dropout rate. cfg.dropout_pretrain during pre-training,
                       cfg.dropout_finetune during fine-tuning.
        """
        super().__init__()
        self.is_global = is_global

        # ── Attention sub-layer ──────────────────────────────────────────────
        self.attn_res_attn = AttnRes(cfg.d_model)
        self.norm_attn     = RMSNorm(cfg.d_model)
        self.attention     = GroupedQueryAttention(cfg, is_global=is_global, rope=rope)

        # ── FFN sub-layer ────────────────────────────────────────────────────
        self.attn_res_ffn = AttnRes(cfg.d_model)
        self.norm_ffn     = RMSNorm(cfg.d_model)
        self.ffn          = SwiGLUFFN(cfg, dropout=dropout)

    def forward(self, layer_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Run one transformer block, appending 2 tensors to layer_outputs.

        Args:
            layer_outputs: Accumulated sub-layer outputs so far.
                           Must have ≥ 1 entry (v₀ = embedding).

        Returns:
            The same list with 2 new tensors appended in-place.
        """
        # ── ATTENTION SUB-LAYER ──────────────────────────────────────────────
        # AttnRes: softmax-weighted combination of all previous outputs
        h = self.attn_res_attn(layer_outputs)       # [B, T, d_model]

        # Pre-norm → attention
        attn_out = self.attention(self.norm_attn(h)) # [B, T, d_model]

        # Record v_{2b+1}
        layer_outputs.append(attn_out)

        # ── FFN SUB-LAYER ────────────────────────────────────────────────────
        # AttnRes: now includes attn_out in the pool
        h2 = self.attn_res_ffn(layer_outputs)       # [B, T, d_model]

        # Pre-norm → FFN
        mlp_out = self.ffn(self.norm_ffn(h2))        # [B, T, d_model]

        # Record v_{2b+2}
        layer_outputs.append(mlp_out)

        return layer_outputs
