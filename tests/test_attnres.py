# tests/test_attnres.py
"""
Stage 5 test gate — AttnRes exhaustive verification.
Run as:  python tests/test_attnres.py
     or: python -m pytest tests/test_attnres.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from slm_project.model.attn_res import AttnRes

d = 768


def test_5_1_zero_init():
    """pseudo_query must be initialised to zero."""
    ar = AttnRes(d)
    assert ar.pseudo_query.allclose(torch.zeros(d)), \
        "pseudo_query must init to ZERO"
    print("5-1 PASS: zero init")


def test_5_2_forward_shape():
    """Output shape must match [B, T, d_model]."""
    ar = AttnRes(d)
    B, T = 2, 64
    layer_outputs = [torch.randn(B, T, d) for _ in range(3)]
    h = ar(layer_outputs)
    assert h.shape == (B, T, d), f"Wrong shape: {h.shape}"
    print("5-2 PASS: forward shape")


def test_5_3_softmax_sums_to_one():
    """Alpha must sum to 1.0 over dim=0 (the layer dimension)."""
    ar = AttnRes(d)
    B, T = 2, 64
    layer_outputs = [torch.randn(B, T, d) for _ in range(3)]
    ar(layer_outputs)
    alpha = ar._last_alpha   # [N, B, T]
    assert (alpha.sum(dim=0) - 1.0).abs().max() < 1e-5, \
        "softmax dim=0 must sum to 1"
    print("5-3 PASS: softmax sums to 1 over dim=0")


def test_5_4_uniform_attention_with_zero_query():
    """With zero pseudo_query, logits=0 everywhere → alpha must be uniform 1/N."""
    ar = AttnRes(d)
    B, T = 2, 64
    layer_outputs = [torch.randn(B, T, d) for _ in range(3)]
    N = len(layer_outputs)
    ar(layer_outputs)
    alpha = ar._last_alpha   # [N, B, T]
    expected = torch.ones(N, B, T) / N
    assert (alpha - expected).abs().max() < 1e-5, \
        f"With zero pseudo_query, alpha must be uniform 1/{N}"
    print(f"5-4 PASS: uniform attention (1/{N}) with zero pseudo_query")


def test_5_5_gradients_flow():
    """Gradients must reach pseudo_query, key_norm.gamma, and all inputs."""
    ar = AttnRes(d)
    B, T = 2, 64
    layer_outputs = [torch.randn(B, T, d, requires_grad=True) for _ in range(4)]
    h = ar(layer_outputs)
    h.sum().backward()
    assert ar.pseudo_query.grad is not None, "pseudo_query has no gradient"
    assert ar.key_norm.gamma.grad is not None, "key_norm.gamma has no gradient"
    for i, lo in enumerate(layer_outputs):
        assert lo.grad is not None, f"layer_output[{i}] has no gradient"
    print("5-5 PASS: gradients flow through pseudo_query, key_norm, and all inputs")


# ── Standalone runner ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("STAGE 5 TEST GATE — AttnRes")
    print("=" * 60)
    print()

    test_5_1_zero_init()
    test_5_2_forward_shape()
    test_5_3_softmax_sums_to_one()
    test_5_4_uniform_attention_with_zero_query()
    test_5_5_gradients_flow()

    print()
    print("=" * 60)
    print("STAGE 5 PASSED  (AttnRes fully verified)")
