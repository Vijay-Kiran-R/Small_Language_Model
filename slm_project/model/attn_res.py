# slm_project/model/attn_res.py
"""
Full Attention Residual (AttnRes) module.

Why it exists
─────────────
Standard residuals add with weight 1.0. By layer 16, v₀ (the embedding)
is diluted to 1/16th influence. AttnRes replaces the residual path with
learned softmax attention over ALL previous sub-layer outputs:

    h_l = Σᵢ αᵢ→l · vᵢ       where i ∈ [0, l-1]
    αᵢ→l = softmax( wₗᵀ · RMSNorm(vᵢ) )   ← softmax over N layers (dim=0)
    wₗ ∈ ℝ^d_model  — learned pseudo-query, INITIALISED TO ZERO

When wₗ = 0: all logits = 0 → softmax gives uniform weights
             → AttnRes is equivalent to standard averaging.
Training specialises from this stable, well-conditioned baseline.

Placement
─────────
One AttnRes instance per SUB-LAYER (both the attention sub-layer and
the FFN sub-layer each get their own AttnRes with separate pseudo_query
and key_norm weights).

Critical rules
──────────────
1. pseudo_query MUST be initialised to ZERO (done in init_weights.py,
   called LAST, after model.apply()).
2. softmax MUST be over dim=0 (the N-layers dimension) — never dim=1/2.
3. key_norm is a SEPARATE RMSNorm per AttnRes instance.
4. The softmax-sum assertion is always active (not just in debug mode).
5. _last_alpha is stored for downstream test verification.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from slm_project.model.rms_norm import RMSNorm


class AttnRes(nn.Module):
    """
    Full Attention Residual: replaces standard residual addition with
    learned, position-wise softmax attention over all previous layer outputs.
    """

    def __init__(self, d_model: int = 768) -> None:
        """
        Args:
            d_model: Hidden dimension (must match the model's d_model).
        """
        super().__init__()
        self.d_model = d_model

        # Learned pseudo-query: one vector per sub-layer.
        # MUST be initialised to zero — init_weights.py handles this LAST.
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))

        # Separate key normalisation — prevents any single layer from
        # dominating attention by having a much larger activation magnitude.
        self.key_norm = RMSNorm(d_model)

        # Stored for test verification (detached, no memory overhead in prod)
        self._last_alpha: Optional[torch.Tensor] = None

    def forward(self, layer_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted combination of all previous sub-layer outputs.

        Args:
            layer_outputs: List of N tensors, each [B, T, d_model].
                           N grows as we go deeper (v₀ is always the embedding).

        Returns:
            h: [B, T, d_model] — softmax-weighted sum of layer_outputs.

        Memory note:
            All N tensors must remain alive during the full forward pass.
            For gradient checkpointing, use the pack/unpack tuple pattern
            in block.py — do NOT free layer_outputs prematurely.
        """
        N = len(layer_outputs)
        assert N >= 1, (
            "layer_outputs must contain at least 1 entry (v₀ = embedding)."
        )

        # Stack: [N, B, T, d_model]
        V = torch.stack(layer_outputs, dim=0)

        # Normalise keys to prevent any single layer from dominating
        K = self.key_norm(V)   # [N, B, T, d_model]

        # Attention logits: dot product of pseudo_query with each normalised key
        # pseudo_query: [d_model]   K: [N, B, T, d_model]  →  logits: [N, B, T]
        logits = torch.einsum('d, nbtd -> nbt', self.pseudo_query, K)

        # Softmax over dim=0 (the layer/N dimension) — weights sum to 1
        alpha = torch.softmax(logits, dim=0)   # [N, B, T]

        # Hard assertion — always active, not debug-only
        sum_err = (alpha.sum(dim=0) - torch.ones_like(alpha[0])).abs().max()
        assert sum_err < 1e-5, (
            f"AttnRes softmax must sum to 1 over dim=0 (layer dimension). "
            f"Got max error {sum_err:.2e}. "
            f"Check: softmax(dim=0), not softmax(dim=1) or dim=2)."
        )

        # Store for downstream test verification (detached — no graph retained)
        self._last_alpha = alpha.detach()

        # Weighted combination → [B, T, d_model]
        h = torch.einsum('nbt, nbtd -> btd', alpha, V)
        return h

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"
