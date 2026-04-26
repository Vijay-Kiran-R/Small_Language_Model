# slm_project/training/muon.py
"""
Single-GPU Muon optimizer with AdamW auxiliary.

Adapted from KellerJordan/Muon (MIT License, https://github.com/KellerJordan/Muon)
This is the standalone single-GPU variant — no torch.distributed required.

Theory
──────
Muon (MomentUm Orthogonalized by Newton-Schulz) applies orthogonalized gradient
updates to 2D hidden weight matrices using Newton-Schulz iteration.

For a weight matrix W with gradient G and momentum buffer M:
  M  ← β·M + G                          (momentum accumulation)
  U  ← newton_schulz(M)                 (orthogonalize: singular values → ~1)
  W  ← W - lr · U                       (update)

This ensures balanced update magnitude across ALL singular value directions,
eliminating the spectral imbalance that AdamW's second moment causes.

Eligibility rules (HARD)
────────────────────────
  MUON:  p.ndim >= 2  AND  'embed' NOT in name
  ADAMW: everything else (1D params, embeddings, pseudo_query, IHA αQ/αK/αV)

Usage
─────
  from slm_project.training.muon import MuonWithAuxAdam
  optimizer = MuonWithAuxAdam(param_groups)

  where param_groups is a list of dicts, each with:
    use_muon (bool)  — True → Muon update; False → AdamW update
    params           — list of parameters
    lr               — learning rate
    + Muon keys:  momentum (float, default 0.95), nesterov (bool, default True)
    + AdamW keys: betas (tuple), eps (float), weight_decay (float)
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import List, Optional


# ── Newton-Schulz orthogonalization ──────────────────────────────────────────

def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute G / ||G||_spec  (semi-orthogonal).

    Uses the degree-5 minimax polynomial from the Muon paper:
        X_{k+1} = a·X_k + b·X_k·X_k^T·X_k + c·X_k·(X_k^T·X_k)^2

    Coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) are chosen to
    minimise the approximation error on [-1, 1].

    Args:
        G:     2D matrix [m, n] in float32 or bfloat16.
        steps: Number of Newton-Schulz iterations (5 is sufficient).

    Returns:
        Orthogonalized matrix with same shape as G.

    Note: operates in float32 internally for numerical stability, even if
    input is bfloat16 — bfloat16 momentum buffers are cast up before here.
    """
    assert G.ndim >= 2, f"newton_schulz requires ndim >= 2, got {G.ndim}"

    # Always compute in float32 for numerical stability
    X = G.float()

    # Normalize so spectral norm ≈ 1 before iterating
    X = X / (X.norm() + 1e-7)

    # Reshape to 2D if higher-dimensional (shouldn't happen in our use case)
    orig_shape = X.shape
    if X.ndim > 2:
        X = X.view(X.shape[0], -1)

    # Ensure m >= n (transpose if needed)
    transposed = X.shape[0] < X.shape[1]
    if transposed:
        X = X.T

    # Newton-Schulz iterations — degree-5 Chebyshev polynomial
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ A @ X)

    if transposed:
        X = X.T

    # Restore original shape
    X = X.view(orig_shape)
    return X.to(G.dtype)


# ── MuonWithAuxAdam ───────────────────────────────────────────────────────────

class MuonWithAuxAdam(Optimizer):
    """
    Combined optimizer: Muon for 2D hidden weights, AdamW for everything else.

    Implements the standard PyTorch Optimizer interface — supports:
      optimizer.step()
      optimizer.zero_grad()
      optimizer.state_dict() / optimizer.load_state_dict()
      optimizer.param_groups (for LR schedule compatibility)

    Parameter group format
    ──────────────────────
    Each group must contain:
      use_muon (bool): True → Muon update path; False → AdamW update path
      params (list):   Parameter tensors

    Muon-specific keys (only for use_muon=True groups):
      lr         (float): Learning rate, default 3e-4
      momentum   (float): Momentum coefficient, default 0.95
      nesterov   (bool):  Use Nesterov momentum, default True
      ns_steps   (int):   Newton-Schulz iterations, default 5
      weight_decay (float): Weight decay (applied after orthogonalization)

    AdamW-specific keys (only for use_muon=False groups):
      lr           (float): Learning rate
      betas        (tuple): (beta1, beta2), default (0.9, 0.95)
      eps          (float): Epsilon, default 1e-8
      weight_decay (float): Weight decay coefficient
    """

    def __init__(self, param_groups: List[dict]) -> None:
        # Fill defaults before passing to Optimizer base
        defaults: dict = {}  # Base class requires defaults dict (unused here)
        processed = []
        for g in param_groups:
            group = dict(g)  # copy
            if group.get('use_muon', False):
                group.setdefault('lr',           3e-4)
                group.setdefault('momentum',     0.95)
                group.setdefault('nesterov',     True)
                group.setdefault('ns_steps',     5)
                group.setdefault('weight_decay', 0.0)
            else:
                group.setdefault('lr',           3e-4)
                group.setdefault('betas',        (0.9, 0.95))
                group.setdefault('eps',          1e-8)
                group.setdefault('weight_decay', 0.1)
            processed.append(group)
        super().__init__(processed, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """
        Perform one optimizer step.

        Iterates over all param groups:
          - use_muon=True  → Newton-Schulz orthogonalized momentum update
          - use_muon=False → Standard AdamW (decoupled weight decay + Adam moments)

        Args:
            closure: Optional closure that re-evaluates the model (not used).

        Returns:
            Loss from closure if provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get('use_muon', False):
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group: dict) -> None:
        """Muon update: orthogonalized momentum for 2D weight matrices."""
        lr           = group['lr']
        momentum     = group['momentum']
        nesterov     = group['nesterov']
        ns_steps     = group['ns_steps']
        weight_decay = group['weight_decay']

        for p in group['params']:
            if p.grad is None:
                continue

            g = p.grad
            state = self.state[p]

            # Initialize momentum buffer on first step
            if 'momentum_buffer' not in state:
                state['momentum_buffer'] = torch.zeros_like(p.data)

            buf = state['momentum_buffer']

            # Update momentum buffer: buf = β·buf + g
            buf.mul_(momentum).add_(g)

            # Nesterov lookahead: effective gradient = β·buf + g
            if nesterov:
                effective_g = buf.mul(momentum).add(g)
            else:
                effective_g = buf

            # Orthogonalize via Newton-Schulz (handles ndim >= 2)
            if effective_g.ndim >= 2:
                update = zeropower_via_newtonschulz5(effective_g, steps=ns_steps)
            else:
                # Fallback for unexpected 1D params in a Muon group (shouldn't happen)
                update = effective_g

            # Scale update norm to match gradient norm (preserve update magnitude)
            # This ensures the effective LR is consistent regardless of matrix size
            g_norm = g.norm()
            u_norm = update.norm()
            if u_norm > 1e-8:
                update = update * (g_norm / u_norm)

            # Decoupled weight decay (applied to param, not gradient)
            if weight_decay != 0.0:
                p.data.mul_(1.0 - lr * weight_decay)

            # Parameter update
            p.data.add_(update, alpha=-lr)

    def _adamw_step(self, group: dict) -> None:
        """AdamW update: per-element adaptive rates with decoupled weight decay."""
        lr           = group['lr']
        beta1, beta2 = group['betas']
        eps          = group['eps']
        weight_decay = group['weight_decay']

        for p in group['params']:
            if p.grad is None:
                continue

            g = p.grad.float()  # compute in float32 for stability
            state = self.state[p]

            # Initialize state on first step
            if 'step' not in state:
                state['step'] = 0
                state['exp_avg']    = torch.zeros_like(p.data, dtype=torch.float32)
                state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

            state['step'] += 1
            step = state['step']
            m    = state['exp_avg']
            v    = state['exp_avg_sq']

            # Update biased first and second moment estimates
            m.mul_(beta1).add_(g, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

            # Bias-corrected moment estimates
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            m_hat = m / bc1
            v_hat = v / bc2

            # Decoupled weight decay
            if weight_decay != 0.0:
                p.data.mul_(1.0 - lr * weight_decay)

            # Adam update (cast back to param dtype)
            update = (m_hat / (v_hat.sqrt() + eps)).to(p.dtype)
            p.data.add_(update, alpha=-lr)
