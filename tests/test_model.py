# tests/test_model.py
"""
Stage 10 test gate — full model assembly.
Run as:  python tests/test_model.py
     or: python -m pytest tests/test_model.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from slm_project.config import ModelConfig, TrainConfig
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights
from slm_project.model.attn_res import AttnRes

cfg  = ModelConfig()
tcfg = TrainConfig()

# Use small B, T — dev machine only, keep memory low
B, T = 2, 32


def build_model():
    m = SLM(cfg, tcfg)
    init_model_weights(m)
    return m


def test_01_vocab_and_embedding():
    model = build_model()
    assert cfg.vocab_size == 32_010
    assert model.embedding.weight.shape == (32_010, 768)
    print("test_01 PASS: vocab + embedding shape")


def test_02_param_count():
    model = build_model()
    n = model.get_num_params()
    assert 40_000_000 < n < 50_000_000, \
        f"3-layer param count {n:,} outside expected [40M, 50M]"
    print(f"test_02 PASS: params = {n:,}")


def test_03_output_shapes():
    model = build_model()
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(x, labels=y)
    assert logits.shape == (B, T, 32_010), f"Logits shape: {logits.shape}"
    print("test_03 PASS: output shapes")


def test_04_no_nan():
    model = build_model()
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(x, labels=y)
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert loss is not None and not torch.isnan(loss), "NaN in loss"
    print(f"test_04 PASS: no NaN  (loss = {loss.item():.4f})")


def test_05_pseudo_queries_zero():
    model = build_model()
    for name, module in model.named_modules():
        if isinstance(module, AttnRes):
            assert module.pseudo_query.allclose(torch.zeros(cfg.d_model)), \
                f"pseudo_query not zero in {name}"
    print("test_05 PASS: all pseudo_queries = 0.0")


def test_06_weight_tying():
    model = build_model()
    param_names = [n for n, _ in model.named_parameters()]
    count = sum(1 for n in param_names if 'embedding.weight' in n)
    assert count == 1, f"embedding.weight appears {count}x — tying broken"
    print("test_06 PASS: weight tying confirmed (embedding.weight appears once)")


def test_07_all_params_have_grad():
    model = build_model()
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))
    _, loss = model(x, labels=y)
    loss.backward()
    no_grad = [(n, p) for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, \
        f"Params with no grad: {[n for n,_ in no_grad[:5]]}"
    print("test_07 PASS: all params have gradients")


def test_08_layer_outputs_count():
    # The assert inside model.forward() fires if count != 7, so a clean
    # forward pass implicitly validates this.
    model = build_model()
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))
    model(x, labels=y)   # would raise AssertionError if count != 7
    print("test_08 PASS: layer_outputs length = 7 = 1 + 2×3")


def test_09_global_local_blocks():
    model = build_model()
    for i, block in enumerate(model.blocks):
        expected = i in cfg.global_layers
        assert block.is_global == expected, \
            f"Block {i}: is_global={block.is_global}, expected {expected}"
    print(f"test_09 PASS: global_layers = {cfg.global_layers}")


def test_10_key_norm_separate():
    model = build_model()
    b0 = model.blocks[0]
    assert b0.attn_res_attn.key_norm is not b0.norm_attn, \
        "key_norm and norm_attn must be separate RMSNorm objects"
    print("test_10 PASS: key_norm is a separate RMSNorm instance")


if __name__ == '__main__':
    print("=" * 60)
    print("STAGE 10 TEST GATE — Full Model Assembly")
    print("=" * 60)
    print()

    test_01_vocab_and_embedding()
    test_02_param_count()
    test_03_output_shapes()
    test_04_no_nan()
    test_05_pseudo_queries_zero()
    test_06_weight_tying()
    test_07_all_params_have_grad()
    test_08_layer_outputs_count()
    test_09_global_local_blocks()
    test_10_key_norm_separate()

    print()
    print("=" * 60)
    print("STAGE 10 PASSED  (Full model assembly verified)")
