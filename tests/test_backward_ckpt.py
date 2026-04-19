# tests/test_backward_ckpt.py
"""
Stage 11 test gate — backward pass + gradient checkpointing safety.
Run as:  python tests/test_backward_ckpt.py
     or: python -m pytest tests/test_backward_ckpt.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from slm_project.config import ModelConfig, TrainConfig
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights
from slm_project.model.attn_res import AttnRes

cfg   = ModelConfig()
tcfg  = TrainConfig()
model = SLM(cfg, tcfg)
init_model_weights(model)

# Shorter seq for speed on dev machine — still exercises all 3 blocks
B, T = 2, 64


def test_11_backward_with_checkpoint():
    """Full backward with gradient checkpointing must not raise."""
    model.train()
    model.zero_grad()
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))

    try:
        logits, loss = model(x, labels=y, use_checkpoint=True)
        loss.backward()
    except RuntimeError as e:
        raise AssertionError(
            f"FAIL: backward with grad checkpointing raised RuntimeError:\n{e}\n"
            f"Fix: check the pack/unpack tuple pattern in "
            f"model._forward_with_checkpoint()"
        )

    # Every parameter must have a gradient
    no_grad = [(n, p) for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, (
        f"Params with no grad after checkpointed backward: "
        f"{[n for n, _ in no_grad[:5]]}"
    )

    # All gradients must be finite
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    print(f"test_11 PASS: backward with gradient checkpointing clean  "
          f"(loss={loss.item():.4f})")


def test_12_attnres_softmax_dim0():
    """_last_alpha must sum to 1 over dim=0 (the layer dimension)."""
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        model(x)

    checked = 0
    for name, module in model.named_modules():
        if isinstance(module, AttnRes) and module._last_alpha is not None:
            alpha = module._last_alpha   # [N, B, T]
            err = (alpha.sum(dim=0) - 1.0).abs().max().item()
            assert err < 1e-5, (
                f"FAIL test_12: softmax not summing to 1 over dim=0 in {name} "
                f"(max_err={err:.2e})"
            )
            checked += 1

    assert checked > 0, "No AttnRes module produced _last_alpha — check forward()"
    print(f"test_12 PASS: AttnRes softmax dim=0 verified  ({checked} modules checked)")


def test_13_qk_norm_per_head_params():
    """QK-Norm must have exactly 1,024 params per attention layer (16 heads × 64)."""
    for i, block in enumerate(model.blocks):
        qk_q = sum(p.numel() for p in block.attention.q_norms.parameters())
        qk_k = sum(p.numel() for p in block.attention.k_norms.parameters())
        total = qk_q + qk_k
        assert total == 1024, (
            f"Block {i}: QK-Norm params = {total}, expected 1024 "
            f"(12+4 heads × 64)"
        )
    print("test_13 PASS: QK-Norm per-head = 1,024 params across all blocks")


if __name__ == '__main__':
    print("=" * 60)
    print("STAGE 11 TEST GATE — Backward + Checkpointing Safety")
    print("=" * 60)
    print()

    test_11_backward_with_checkpoint()
    test_12_attnres_softmax_dim0()
    test_13_qk_norm_per_head_params()

    print()
    print("=" * 60)
    print("STAGE 11 PASSED  (All safety checks clean)")
    print("MODEL IS READY FOR TRAINING")
