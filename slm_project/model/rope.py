# slm_project/model/rope.py
"""
Rotary Position Embedding (RoPE).

Applied to Q and K in LOCAL (SWA) blocks only.
Global (NoPE) blocks skip RoPE entirely — the attention layer is responsible
for gating this call based on the layer type.

Design notes
────────────
* rope_base = 500,000  (LOCKED — supports 8K pretrain and 16K Phase 5.5).
  DO NOT use rope_base = 10,000 — too small for 8K context.
* Operates on d_head = 64 dimensions.
* cos/sin tables are cached up to max_seq_len to avoid recomputation.
* sin/cos outputs are bounded [-1, 1]; bfloat16 precision is sufficient.
  DO NOT add GradScaler or float32 casts here.
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with configurable base frequency."""

    def __init__(
        self,
        d_head: int = 64,
        max_seq_len: int = 8192,
        rope_base: int = 500_000,
        device: str = 'cpu',
    ) -> None:
        """
        Args:
            d_head:      Head dimension (must be even).
            max_seq_len: Maximum sequence length to pre-cache.
            rope_base:   Base frequency for the rotation schedule.
            device:      Device for the cached tables.
        """
        super().__init__()
        self.d_head      = d_head
        self.max_seq_len = max_seq_len
        self.rope_base   = rope_base

        # Frequency bands — [d_head // 2]
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len, device)

    def _build_cache(self, seq_len: int, device) -> None:
        """Pre-compute and cache cos/sin tables for positions 0..seq_len-1."""
        t     = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)   # [T, d_head//2]
        emb   = torch.cat([freqs, freqs], dim=-1)            # [T, d_head]
        # Shape: [1, 1, T, d_head] — broadcasts over batch and heads
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Swap and negate halves of the last dimension: [x1, x2] → [-x2, x1]."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.

        Args:
            q: Query tensor  [B, T, n_heads_q,  d_head]
            k: Key tensor    [B, T, n_heads_kv, d_head]
            seq_len: Actual sequence length (≤ max_seq_len).

        Returns:
            (q_rotated, k_rotated) with identical shapes to inputs.
        """
        # Slice cached tables to actual seq_len: [1, 1, T, d_head]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        # Transpose to [1, T, 1, d_head] so it broadcasts over the heads dim
        cos = cos.transpose(1, 2)   # [1, T, 1, d_head]
        sin = sin.transpose(1, 2)   # [1, T, 1, d_head]

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot
