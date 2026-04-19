# slm_project/model/attention.py
"""
Grouped Query Attention with per-head QK-Norm, RoPE/NoPE, and FlashAttention.

Key design decisions
────────────────────
* GQA: 12 Q heads, 4 KV heads (3:1 ratio) — KV cache is 3× smaller.
* QK-Norm: (n_heads_q + n_heads_kv) SEPARATE RMSNorm(d_head=64) instances.
  NOT RMSNorm(d_model=768) — that is wrong in both parameter count and
  behaviour (would normalise per-token across all heads instead of per-head).
  Correct: 12×RMSNorm(64) for Q + 4×RMSNorm(64) for K = 1,024 params/layer.
* RoPE: applied in local (SWA) blocks only; global (NoPE) blocks skip it.
* FlashAttention: causal=True is ALWAYS set explicitly on EVERY call.
  window_size=(-1,-1) disables SWA but does NOT imply causal — must be
  passed explicitly.
* SDPA fallback: used when FlashAttention is unavailable (dev machine).

Param counts per attention layer
─────────────────────────────────
  W_Q :  768 × 768  =   589,824
  W_K :  768 × 256  =   196,608   (n_heads_kv=4, d_head=64 → 4×64=256)
  W_V :  768 × 256  =   196,608
  W_O :  768 × 768  =   589,824
  QK-Norm: 16 × 64  =     1,024
  TOTAL  :           = 1,573,888
"""

import torch
import torch.nn as nn
from typing import Optional

from slm_project.model.rms_norm import RMSNorm
from slm_project.model.rope import RotaryEmbedding
from slm_project.config import ModelConfig

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class GroupedQueryAttention(nn.Module):
    """GQA + per-head QK-Norm + RoPE/NoPE + FlashAttention (SWA or full-seq)."""

    def __init__(
        self,
        cfg: ModelConfig,
        is_global: bool = False,
        rope: Optional[RotaryEmbedding] = None,
    ) -> None:
        """
        Args:
            cfg:       ModelConfig — single source of truth for all dimensions.
            is_global: True → full-sequence NoPE (no RoPE, no SWA window).
                       False → local SWA block with RoPE.
            rope:      Shared RotaryEmbedding instance.  Must be None when
                       is_global=True; ignored anyway but explicit is clearer.
        """
        super().__init__()
        self.n_heads_q  = cfg.n_heads_q
        self.n_heads_kv = cfg.n_heads_kv
        self.d_head     = cfg.d_head
        self.d_model    = cfg.d_model
        self.swa_window = cfg.swa_window
        self.is_global  = is_global
        self.rope       = rope   # None for global (NoPE) blocks

        # ── Projections (no bias) ────────────────────────────────────────────
        self.W_Q = nn.Linear(cfg.d_model, cfg.n_heads_q  * cfg.d_head, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.n_heads_kv * cfg.d_head, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.n_heads_kv * cfg.d_head, bias=False)
        self.W_O = nn.Linear(cfg.n_heads_q * cfg.d_head, cfg.d_model,  bias=False)

        # ── QK-Norm: SEPARATE RMSNorm(d_head) per head ──────────────────────
        # CRITICAL: NOT RMSNorm(d_model).  Each head has its own normaliser.
        # 12 Q norms + 4 K norms = 16 instances × 64 params = 1,024 total.
        self.q_norms = nn.ModuleList(
            [RMSNorm(cfg.d_head) for _ in range(cfg.n_heads_q)]
        )
        self.k_norms = nn.ModuleList(
            [RMSNorm(cfg.d_head) for _ in range(cfg.n_heads_kv)]
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _apply_qk_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ):
        """
        Apply per-head RMSNorm to Q and K.

        Args:
            q: [B, T, n_heads_q,  d_head]
            k: [B, T, n_heads_kv, d_head]

        Returns:
            (q_normed, k_normed) with identical shapes.
        """
        q_normed = torch.stack(
            [self.q_norms[h](q[..., h, :]) for h in range(self.n_heads_q)],
            dim=2,
        )
        k_normed = torch.stack(
            [self.k_norms[h](k[..., h, :]) for h in range(self.n_heads_kv)],
            dim=2,
        )
        return q_normed, k_normed

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        FlashAttention path.  flash_attn_func expects [B, T, H, d_head].
        GQA (n_heads_q ≠ n_heads_kv) is handled natively by flash_attn ≥ 2.1.
        causal=True is ALWAYS EXPLICIT — window_size alone does not set it.
        """
        if self.is_global:
            # Full-sequence causal attention — no SWA window, no RoPE
            out = flash_attn_func(
                q, k, v,
                causal=True,          # ← ALWAYS EXPLICIT
                window_size=(-1, -1), # Full sequence (disables SWA)
            )
        else:
            # Sliding Window Attention — still causal=True explicitly
            out = flash_attn_func(
                q, k, v,
                causal=True,                       # ← ALWAYS EXPLICIT
                window_size=(self.swa_window, 0),  # e.g. (2048, 0)
            )
        return out   # [B, T, n_heads_q, d_head]

    def _sdpa_fallback(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        PyTorch SDPA fallback when FlashAttention is unavailable (dev machine).

        LOCAL  → sliding-window causal mask.
        GLOBAL → standard lower-triangular causal mask.
        These are DIFFERENT masks — not interchangeable.
        """
        T = q.shape[1]

        # Expand KV heads to match Q heads: [B,T,n_heads_kv,d] → [B,T,n_heads_q,d]
        expand_factor = self.n_heads_q // self.n_heads_kv
        k_exp = k.repeat_interleave(expand_factor, dim=2)
        v_exp = v.repeat_interleave(expand_factor, dim=2)

        # SDPA layout: [B, H, T, d_head]
        q_t   = q.permute(0, 2, 1, 3)
        k_t   = k_exp.permute(0, 2, 1, 3)
        v_t   = v_exp.permute(0, 2, 1, 3)

        if self.is_global:
            # Standard full-sequence causal mask
            causal_mask = torch.tril(
                torch.ones(T, T, device=q.device, dtype=torch.bool)
            )
        else:
            # SWA: causal AND within swa_window positions
            positions = torch.arange(T, device=q.device)
            causal_mask = (
                torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
                & (positions.unsqueeze(0) >= (positions.unsqueeze(1) - self.swa_window))
            )

        # Convert bool mask to additive mask (0 or -inf)
        attn_mask = torch.zeros(T, T, device=q.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~causal_mask, float('-inf'))

        out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=attn_mask,
        )
        return out.permute(0, 2, 1, 3)   # [B, T, n_heads_q, d_head]

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model]
        """
        B, T, _ = x.shape

        # Project and reshape to per-head tensors
        q = self.W_Q(x).view(B, T, self.n_heads_q,  self.d_head)   # [B,T,12,64]
        k = self.W_K(x).view(B, T, self.n_heads_kv, self.d_head)   # [B,T, 4,64]
        v = self.W_V(x).view(B, T, self.n_heads_kv, self.d_head)   # [B,T, 4,64]

        # Per-head QK-Norm
        q, k = self._apply_qk_norm(q, k)

        # RoPE — local blocks only; global (NoPE) blocks skip entirely
        if not self.is_global and self.rope is not None:
            q, k = self.rope(q, k, seq_len=T)

        # Attention (FlashAttention preferred; SDPA fallback on dev machine)
        if HAS_FLASH_ATTN:
            attn_out = self._flash_attention(q, k, v)
        else:
            attn_out = self._sdpa_fallback(q, k, v)

        # Merge heads and project back to d_model
        return self.W_O(attn_out.reshape(B, T, self.n_heads_q * self.d_head))
