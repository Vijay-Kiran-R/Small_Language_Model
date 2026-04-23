# slm_project/model/model.py
"""
SLM — Small Language Model (3-layer test version, ~44M params).

Architecture overview
─────────────────────
  Embedding  (weight-tied with output projection)
  3 × TransformerBlock:
      Block 0, 1 : LOCAL  — SWA (window=2048) + RoPE
      Block 2    : GLOBAL — full-sequence causal NoPE
  Final RMSNorm  (gamma=1.0 — DO NOT change to 0.2887)
  Output projection  (tied to embedding.weight.T)
  MTPHead  (predicts t+2; tied to embedding.weight.T)

Forward pass
─────────────
  1. Embed tokens → v₀;  layer_outputs = [v₀]
  2. For each block b in [0..2]:
       a. AttnRes → RMSNorm → Attn → append v_{2b+1}
       b. AttnRes → RMSNorm → FFN  → append v_{2b+2}
     After all blocks: len(layer_outputs) == 1 + 2×n_layers == 7
  3. Final RMSNorm on layer_outputs[-1]
  4. Main logits: final_h @ embedding.weight.T
  5. MTP  logits: MTPHead(layer_outputs[-1], embedding.weight)
  6. If labels provided: compute cross-entropy losses and combine.

Loss formula
─────────────
  total_loss = cross_entropy(t+1) + mtp_weight(step) × cross_entropy(t+2)

Weight tying
─────────────
  Both output projection and MTPHead share embedding.weight — no separate
  parameter is allocated.  embedding.weight appears exactly once in
  model.named_parameters().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from slm_project.config import ModelConfig, TrainConfig
from slm_project.model.rms_norm import RMSNorm
from slm_project.model.rope import RotaryEmbedding
from slm_project.model.block import TransformerBlock
from slm_project.model.mtp import MTPHead, get_mtp_weight


class SLM(nn.Module):
    """Small Language Model — 3-layer test version (~44 M params)."""

    def __init__(
        self,
        cfg: ModelConfig,
        tcfg: Optional[TrainConfig] = None,
    ) -> None:
        """
        Args:
            cfg:  ModelConfig — architecture hyperparameters.
            tcfg: TrainConfig — schedule / loss hyperparameters.
                  Defaults to TrainConfig() if not supplied.
        """
        super().__init__()
        self.cfg  = cfg
        self.tcfg = tcfg or TrainConfig()

        # ── Embedding (weight-tied with output projection) ───────────────────
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # ── Shared RoPE (local blocks only) ──────────────────────────────────
        self.rope = RotaryEmbedding(
            d_head=cfg.d_head,
            max_seq_len=cfg.max_seq_len,
            rope_base=cfg.rope_base,
        )

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList()
        for layer_idx in range(cfg.n_layers):
            is_global = layer_idx in cfg.global_layers
            self.blocks.append(
                TransformerBlock(
                    cfg,
                    is_global=is_global,
                    rope=None if is_global else self.rope,
                    dropout=cfg.dropout_pretrain,
                )
            )

        # ── Final RMSNorm — gamma=1.0 (RMSNorm default) ──────────────────────
        # DO NOT change to 0.2887 — that compensates O(L) growth in standard
        # residuals; AttnRes eliminates that growth.
        self.final_norm = RMSNorm(cfg.d_model)

        # ── MTP head (weight-tied to embedding) ───────────────────────────────
        self.mtp_head = MTPHead(cfg)

        # Output projection is implemented as:  final_h @ self.embedding.weight.T
        # No separate nn.Linear — weight tying via shared parameter reference.

    # ── Utility ──────────────────────────────────────────────────────────────

    def get_num_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters())

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,                   # [B, T]
        labels:         Optional[torch.Tensor] = None,  # [B, T]
        global_step:    int  = 0,
        use_checkpoint: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids:      Token IDs [B, T].
            labels:         Target IDs [B, T].  None → inference (loss=None).
            global_step:    Used for MTP weight annealing schedule.
            use_checkpoint: Enable gradient checkpointing (saves memory).

        Returns:
            (logits [B, T, vocab_size], total_loss or None)
        """
        # ── 1. Embed ─────────────────────────────────────────────────────────
        x = self.embedding(input_ids)   # [B, T, d_model]
        layer_outputs = [x]             # v₀ = raw embedding output

        # ── 2. Blocks ────────────────────────────────────────────────────────
        if use_checkpoint:
            layer_outputs = self._forward_with_checkpoint(layer_outputs)
        else:
            for block in self.blocks:
                layer_outputs = block(layer_outputs)

        # Sanity-check output count (fast assertion; disable in prod if needed)
        expected_n = 1 + 2 * self.cfg.n_layers   # 7 for 3-layer model
        assert len(layer_outputs) == expected_n, (
            f"Expected {expected_n} layer_outputs, got {len(layer_outputs)}. "
            f"Each block must append exactly 2 tensors."
        )

        # ── 3. Final norm ─────────────────────────────────────────────────────
        final_h = self.final_norm(layer_outputs[-1])   # [B, T, d_model]

        # ── 4. Main logits (weight-tied) ──────────────────────────────────────
        logits = final_h @ self.embedding.weight.T     # [B, T, vocab_size]

        # ── 5. MTP logits (weight-tied) ───────────────────────────────────────
        mtp_logits = self.mtp_head(layer_outputs[-1], self.embedding.weight)

        if labels is None:
            return logits, None

        # ── 6. Losses ────────────────────────────────────────────────────────
        # Main: predict token t+1 from position t
        # CRITICAL FIX: dataset.py already shifts labels (input_ids=chunk[:-1], labels=chunk[1:])
        # so logits[t] should predict labels[t]. No further shifting needed!
        main_loss = F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )

        # MTP: predict token t+2 from position t
        # mtp_logits[t] predicts t+2. labels[t+1] is t+2.
        mtp_loss = F.cross_entropy(
            mtp_logits[:, :-1].reshape(-1, self.cfg.vocab_size),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

        mtp_w      = get_mtp_weight(global_step, self.tcfg)
        total_loss = main_loss + mtp_w * mtp_loss

        return logits, total_loss

    # ── Gradient checkpointing ───────────────────────────────────────────────

    def _forward_with_checkpoint(self, layer_outputs: list) -> list:
        """
        Run all blocks with gradient checkpointing.

        CRITICAL — do NOT pass layer_outputs as a single list argument to
        checkpoint(); that causes freed-buffer RuntimeErrors because
        checkpoint() cannot track list mutation.

        Correct pattern: pack the list as a tuple before crossing the
        checkpoint boundary, unpack back to list on the other side.
        """
        import torch.utils.checkpoint as ckpt

        for block in self.blocks:
            def _block_fn(*lo_tensors, _block=block):
                lo = list(lo_tensors)
                result = _block(lo)
                return tuple(result)

            new_outputs = ckpt.checkpoint(
                _block_fn,
                *tuple(layer_outputs),
                use_reentrant=False,   # Required for PyTorch ≥ 2.0
            )
            layer_outputs = list(new_outputs)

        return layer_outputs
