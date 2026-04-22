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


class IHAGlobalAttention(nn.Module):
    """
    Interleaved Head Attention for Global (NoPE) Layers ONLY.
    GQA-adapted: H_Q=12 Q heads, H_KV=4 KV heads, P=2 pseudo-heads.
    
    Parameter overhead per layer:
      αQ: 12×12×2 = 288
      αK: 4×4×2   = 32
      αV: 4×4×2   = 32
      R:  12×24   = 288
      Total: 640 params per global layer × 4 layers = 2,560 total
    
    CRITICAL RULES:
    1. This class is ONLY for global layers — never use in local (SWA) layers.
    2. NO RoPE — global NoPE layers, same as before.
    3. causal=True always explicit in FlashAttention calls.
    4. window_size=(-1,-1) for full-sequence global attention.
    5. After IHA collapse, output shape [B, N, D] is identical to standard attn.
       AttnRes buffer sees no difference.
    6. αQ, αK, αV initialized to one-hot identity → IHA reduces to standard MHA.
       Training then learns cross-head mixing from this stable baseline.
    7. R initialized to select pseudo-head j=1 from each head → clean identity start.
    """

    def __init__(self, cfg, P: int = 2):
        super().__init__()
        self.n_heads_q  = cfg.n_heads_q    # 12
        self.n_heads_kv = cfg.n_heads_kv   # 4
        self.d_head     = cfg.d_head       # 64
        self.d_model    = cfg.d_model      # 768
        self.P          = P                # 2 pseudo-heads per head

        # ── Standard GQA projections (same as GroupedQueryAttention) ──
        self.W_Q = nn.Linear(cfg.d_model, cfg.n_heads_q  * cfg.d_head, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.n_heads_kv * cfg.d_head, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.n_heads_kv * cfg.d_head, bias=False)
        self.W_O = nn.Linear(cfg.n_heads_q * cfg.d_head, cfg.d_model,  bias=False)

        # ── Per-head QK-Norm (same as GroupedQueryAttention — do NOT skip) ──
        self.q_norms = nn.ModuleList([RMSNorm(cfg.d_head) for _ in range(cfg.n_heads_q)])
        self.k_norms = nn.ModuleList([RMSNorm(cfg.d_head) for _ in range(cfg.n_heads_kv)])

        # ── IHA mixing tensors ────────────────────────────────────────
        # αQ: for each Q head h, for each pseudo p, linear combination
        #     over all H_Q original Q heads
        self.alpha_Q = nn.Parameter(torch.zeros(cfg.n_heads_q, cfg.n_heads_q, P))
        # αK: for each KV head h, for each pseudo p, mix over H_KV K heads
        self.alpha_K = nn.Parameter(torch.zeros(cfg.n_heads_kv, cfg.n_heads_kv, P))
        # αV: same structure as αK
        self.alpha_V = nn.Parameter(torch.zeros(cfg.n_heads_kv, cfg.n_heads_kv, P))
        # R: collapse map from (H_Q × P) pseudo-outputs back to H_Q outputs
        # Shape: [H_Q, H_Q * P]
        self.R = nn.Parameter(torch.zeros(cfg.n_heads_q, cfg.n_heads_q * P))

        # ── Initialize to identity (IHA reduces to standard MHA at init) ──
        self._init_iha_identity()

    def _init_iha_identity(self):
        """
        Initialize IHA parameters so that at step 0, IHA = standard MHA.
        This is the stable baseline required before training.

        αQ[i, i, 0] = 1.0, all others = 0  → pseudo j=0 of head i = head i's Q
        αQ[i, i, 1] = 1.0, all others = 0  → pseudo j=1 of head i = head i's Q (copy)
        Same for αK, αV.
        R selects only pseudo j=0 from each head → exact MHA output.

        After training: αQ learns cross-head mixtures; R learns optimal collapse.
        """
        # αQ: identity along head dimension for both pseudo-heads
        for h in range(self.n_heads_q):
            for p in range(self.P):
                self.alpha_Q.data[h, h, p] = 1.0

        # αK, αV: identity along KV head dimension
        for h in range(self.n_heads_kv):
            for p in range(self.P):
                self.alpha_K.data[h, h, p] = 1.0
                self.alpha_V.data[h, h, p] = 1.0

        # R: select only pseudo j=0 from each head h (identity collapse)
        # R[h, h*P + 0] = 1.0, all others = 0
        self.R.data.zero_()
        for h in range(self.n_heads_q):
            self.R.data[h, h * self.P] = 1.0

    def _apply_qk_norm(self, q, k):
        """Per-head QK-Norm. Identical to GroupedQueryAttention._apply_qk_norm."""
        q_normed = torch.stack([
            self.q_norms[h](q[..., h, :]) for h in range(self.n_heads_q)
        ], dim=2)
        k_normed = torch.stack([
            self.k_norms[h](k[..., h, :]) for h in range(self.n_heads_kv)
        ], dim=2)
        return q_normed, k_normed

    def _build_pseudo_heads(self, Q, K, V):
        """
        IHA Step 1: Build P pseudo-heads via learned mixing across original heads.

        Q: [B, N, H_Q, d_head]    K: [B, N, H_KV, d_head]    V: [B, N, H_KV, d_head]

        Returns:
          Q_pseudo: [B, N, H_Q,  P, d_head]
          K_pseudo: [B, N, H_KV, P, d_head]
          V_pseudo: [B, N, H_KV, P, d_head]

        IHA formula (from Algorithm 1):
          Q_pseudo[b,n,h,p,:] = sum_m (alpha_Q[h,m,p] * Q[b,n,m,:])
          Equivalent to einsum: 'hmp, bnmd -> bnhpd', alpha_Q, Q
        """
        # einsum: alpha_Q[h,m,p], Q[b,n,m,d] → Q_pseudo[b,n,h,p,d]
        Q_pseudo = torch.einsum('hmp, bnmd -> bnhpd', self.alpha_Q, Q)
        K_pseudo = torch.einsum('hmp, bnmd -> bnhpd', self.alpha_K, K)
        V_pseudo = torch.einsum('hmp, bnmd -> bnhpd', self.alpha_V, V)
        return Q_pseudo, K_pseudo, V_pseudo

    def _interleave_pseudos(self, X_pseudo):
        """
        IHA Step 2: Interleave pseudo-head dimension into sequence dimension.

        X_pseudo: [B, N, H, P, d_head]
        Returns:  [B, N*P, H, d_head]

        Interleaving means token n's P pseudo-tokens are placed consecutively:
        (n=0, p=0), (n=0, p=1), (n=1, p=0), (n=1, p=1), ...
        This gives each pseudo-token a distinct position index for RoPE.
        For NoPE global layers, RoPE is not applied, but interleaving still
        separates the pseudo-tokens so they can attend differently to each other.
        """
        B, N, H, P, d = X_pseudo.shape
        # Reshape to interleave P into N: [B, N, H, P, d] → [B, N*P, H, d]
        # Interleaved order: (0,0),(0,1),(1,0),(1,1),...
        X_interleaved = X_pseudo.permute(0, 1, 3, 2, 4)  # [B, N, P, H, d]
        X_interleaved = X_interleaved.reshape(B, N * P, H, d)
        return X_interleaved

    def _collapse_pseudos(self, attn_out):
        """
        IHA Step 4: Collapse pseudo-heads back to H heads using learned R.

        attn_out: [B, N*P, H_Q, d_head]
        Returns:  [B, N, H_Q, d_head]

        R: [H_Q, H_Q * P] — learned collapse map
        """
        B, NP, H_Q, d = attn_out.shape
        N = NP // self.P
        # Reshape: [B, N*P, H_Q, d] → [B, N, P, H_Q, d] → [B, N, H_Q*P, d]
        attn_out = attn_out.reshape(B, N, self.P, H_Q, d)
        attn_out = attn_out.permute(0, 1, 3, 2, 4)         # [B, N, H_Q, P, d]
        attn_out = attn_out.reshape(B, N, H_Q * self.P, d)  # [B, N, H_Q*P, d]

        # Collapse: R[h, h'] weights sum over h' ∈ [H_Q*P]
        # output[b,n,h,d] = sum_hp (R[h,hp] * attn_out[b,n,hp,d])
        # einsum: 'hhp, bnhpd -> bnhd' where the combined hp index
        out = torch.einsum('ij, bnjd -> bnid', self.R, attn_out)  # [B, N, H_Q, d]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        Returns: [B, N, D]

        Steps:
        1. Project → Q[B,N,H_Q,d], K[B,N,H_KV,d], V[B,N,H_KV,d]
        2. QK-Norm per head
        3. IHA: build pseudo-heads, interleave → expanded seqs
        4. FlashAttention on expanded seqs (causal=True, full-seq)
        5. Collapse pseudo-heads back
        6. Output projection
        """
        B, N, _ = x.shape

        # Step 1: Standard GQA projections
        Q = self.W_Q(x).view(B, N, self.n_heads_q,  self.d_head)   # [B,N,12,64]
        K = self.W_K(x).view(B, N, self.n_heads_kv, self.d_head)   # [B,N, 4,64]
        V = self.W_V(x).view(B, N, self.n_heads_kv, self.d_head)   # [B,N, 4,64]

        # Step 2: QK-Norm (same as standard GQA — no change here)
        Q, K = self._apply_qk_norm(Q, K)

        # Step 3: IHA pseudo-head construction + interleaving
        Q_pseudo, K_pseudo, V_pseudo = self._build_pseudo_heads(Q, K, V)
        # Q_pseudo: [B, N, H_Q=12, P=2, 64], K/V_pseudo: [B, N, H_KV=4, P=2, 64]

        Q_exp = self._interleave_pseudos(Q_pseudo)  # [B, N*P, H_Q=12,  64]
        K_exp = self._interleave_pseudos(K_pseudo)  # [B, N*P, H_KV=4,  64]
        V_exp = self._interleave_pseudos(V_pseudo)  # [B, N*P, H_KV=4,  64]

        # Step 4: FlashAttention on expanded sequence
        # CRITICAL: causal=True ALWAYS explicit
        # CRITICAL: window_size=(-1,-1) for full-sequence global attention (NoPE)
        attn_out = self._flash_global(Q_exp, K_exp, V_exp)  # [B, N*P, H_Q=12, 64]

        # Step 5: Collapse pseudo-heads
        out = self._collapse_pseudos(attn_out)   # [B, N, H_Q=12, 64]

        # Step 6: Output projection (same as standard GQA)
        out = out.reshape(B, N, self.n_heads_q * self.d_head)
        return self.W_O(out)                     # [B, N, D=768]

    def _flash_global(self, Q_exp, K_exp, V_exp):
        """
        FlashAttention call for IHA global layer.
        Q_exp: [B, N*P, H_Q,  d_head]
        K_exp: [B, N*P, H_KV, d_head]
        V_exp: [B, N*P, H_KV, d_head]
        Returns: [B, N*P, H_Q, d_head]

        CRITICAL: causal=True MUST be explicit (window_size=(-1,-1) does NOT set it)
        CRITICAL: window_size=(-1,-1) = full sequence (no SWA — correct for global layers)
        CRITICAL: GQA ratio H_Q/H_KV = 3 is handled by FlashAttention natively
        """
        if HAS_FLASH_ATTN:
            return flash_attn_func(
                Q_exp, K_exp, V_exp,
                causal=True,         # ALWAYS EXPLICIT
                window_size=(-1, -1) # Full sequence, no SWA
            )
        else:
            return self._sdpa_global_fallback(Q_exp, K_exp, V_exp)

    def _sdpa_global_fallback(self, Q_exp, K_exp, V_exp):
        """PyTorch SDPA fallback for global attention on expanded sequence."""
        B, NP, H_Q, d  = Q_exp.shape
        B, NP, H_KV, d = K_exp.shape

        # Expand KV heads to match Q heads
        expand_factor = H_Q // H_KV
        K_exp_full = K_exp.repeat_interleave(expand_factor, dim=2)
        V_exp_full = V_exp.repeat_interleave(expand_factor, dim=2)

        # [B, H, NP, d] for SDPA
        q = Q_exp.permute(0, 2, 1, 3)
        k = K_exp_full.permute(0, 2, 1, 3)
        v = V_exp_full.permute(0, 2, 1, 3)

        # Standard lower-triangular causal mask (global — full sequence)
        mask = torch.tril(torch.ones(NP, NP, device=q.device, dtype=torch.bool))
        attn_bias = torch.zeros(NP, NP, device=q.device, dtype=q.dtype)
        attn_bias = attn_bias.masked_fill(~mask, float('-inf'))

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        return out.permute(0, 2, 1, 3)  # [B, NP, H_Q, d]
