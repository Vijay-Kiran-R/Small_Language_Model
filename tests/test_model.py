import torch
from slm_project.config import ModelConfig, TrainConfig
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights
from slm_project.model.attn_res import AttnRes

def test_model_16_layer():
    cfg   = ModelConfig()   # n_layers=16
    tcfg  = TrainConfig()
    model = SLM(cfg, tcfg)
    init_model_weights(model)

    B, T = 2, 128
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))

    # ── test_01: Vocab and embedding shape ───────────────────────
    assert cfg.vocab_size == 32_010
    assert model.embedding.weight.shape == (32_010, 768)
    print("test_01 PASS ✓")

    # ── test_02: Parameter count (16-layer = 125,931,008 with IHA) ─────────
    n_params = model.get_num_params()
    assert n_params == 125_931_008, \
        f"Params = {n_params:,}, expected 125,931,008. Check component counts."
    print(f"test_02 PASS ✓  (params = {n_params:,} = 125.9M)")

    # ── test_03: Output shapes ────────────────────────────────────
    logits, loss = model(x, labels=y)
    assert logits.shape == (B, T, 32_010)
    print("test_03 PASS ✓")

    # ── test_04: No NaN ───────────────────────────────────────────
    assert not torch.isnan(logits).any()
    assert not torch.isnan(loss)
    print("test_04 PASS ✓")

    # ── test_05: ALL 32 pseudo_queries zero after init ────────────
    n_pq = 0
    for name, module in model.named_modules():
        if isinstance(module, AttnRes):
            assert module.pseudo_query.allclose(torch.zeros(cfg.d_model)), \
                f"pseudo_query not zero in {name}"
            n_pq += 1
    assert n_pq == 32, f"Expected 32 AttnRes instances (16 layers × 2), found {n_pq}"
    print(f"test_05 PASS ✓  (all {n_pq} pseudo_queries = 0.0)")

    # ── test_06: Weight tying ─────────────────────────────────────
    param_names = [n for n, _ in model.named_parameters()]
    assert sum(1 for n in param_names if 'embedding.weight' in n) == 1
    print("test_06 PASS ✓")

    # ── test_07: Gradients reach all params ──────────────────────
    loss.backward()
    no_grad = [(n, p) for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"Params with no grad: {[n for n,_ in no_grad[:5]]}"
    print("test_07 PASS ✓")

    # ── test_08: layer_outputs length = 33 (not 7) ───────────────
    # The assert inside model.forward() verifies this; no assertion fire = correct
    model2 = SLM(cfg, tcfg)
    init_model_weights(model2)
    _ = model2(x, labels=y)   # assert inside checks len == 1+2*16 = 33
    print("test_08 PASS ✓  (layer_outputs = 33 = 1 + 2×16)")

    # ── test_09: Correct global/local assignment ──────────────────
    for i, block in enumerate(model.blocks):
        expected = (i in cfg.global_layers)
        assert block.is_global == expected, \
            f"Block {i}: is_global={block.is_global}, expected {expected}"
    # Verify global pattern: local,local,local,GLOBAL × 4
    assert [b.is_global for b in model.blocks] == [
        False, False, False, True,   # layers 0-3
        False, False, False, True,   # layers 4-7
        False, False, False, True,   # layers 8-11
        False, False, False, True    # layers 12-15
    ]
    print("test_09 PASS ✓  (global_layers = {3,7,11,15}; pattern verified)")

    # ── test_10: key_norm ≠ norm_attn ────────────────────────────
    for i, block in enumerate(model.blocks):
        assert block.attn_res_attn.key_norm is not block.norm_attn
    print("test_10 PASS ✓  (key_norm is separate in all 16 blocks)")

if __name__ == '__main__':
    test_model_16_layer()
