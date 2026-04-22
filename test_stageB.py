import torch
from slm_project.config import ModelConfig
from slm_project.model.attention import IHAGlobalAttention, GroupedQueryAttention
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights

def test_stageB():
    cfg = ModelConfig()

    # ── TEST B-1: IHA global attn shape ──────────────────────────
    iha = IHAGlobalAttention(cfg, P=2)
    x = torch.randn(2, 64, 768)
    out = iha(x)
    assert out.shape == (2, 64, 768), f"Wrong shape: {out.shape}"
    assert not torch.isnan(out).any()
    print("B-1 PASS ✓: IHAGlobalAttention output shape")

    # ── TEST B-2: At init, IHA ≈ standard MHA (identity start) ──
    # With identity initialization, pseudo-heads are copies → output ≈ MHA
    # Exact equality not guaranteed (R collapses P pseudos → averaging effect)
    # but output should not be pathological
    assert out.norm() > 0, "Output is zero — init broken"
    assert out.norm() < 1e4, "Output norm exploding — init broken"
    print("B-2 PASS ✓: identity initialization gives stable output")

    # ── TEST B-3: α init is identity-like ────────────────────────
    # αQ[h, h, p] should be 1.0 for all h and p
    for h in range(cfg.n_heads_q):
        for p in range(2):
            assert iha.alpha_Q.data[h, h, p] == 1.0, \
                f"alpha_Q[{h},{h},{p}] = {iha.alpha_Q.data[h,h,p]}, expected 1.0"
    print("B-3 PASS ✓: alpha_Q identity initialization")

    # ── TEST B-4: R init selects pseudo j=0 ──────────────────────
    for h in range(cfg.n_heads_q):
        assert iha.R.data[h, h * 2] == 1.0, \
            f"R[{h},{h*2}] = {iha.R.data[h, h*2]}, expected 1.0 (select pseudo 0)"
        assert iha.R.data[h, h * 2 + 1] == 0.0, \
            f"R[{h},{h*2+1}] = {iha.R.data[h,h*2+1]}, expected 0.0"
    print("B-4 PASS ✓: R initialization selects pseudo j=0")

    # ── TEST B-5: Parameter count ─────────────────────────────────
    # αQ(288) + αK(32) + αV(32) + R(288) + standard GQA params
    iha_extra = sum(p.numel() for p in [iha.alpha_Q, iha.alpha_K, iha.alpha_V, iha.R])
    assert iha_extra == 640, f"IHA overhead = {iha_extra}, expected 640"
    print(f"B-5 PASS ✓: IHA overhead = {iha_extra} params per global layer")

    # ── TEST B-6: Full model still 125,931,008 params ────────────
    model = SLM(cfg)
    init_model_weights(model)
    n_params = model.get_num_params()
    assert n_params == 125_931_008, \
        f"Expected 125,931,008 params, got {n_params:,}\n" \
        f"Difference: {n_params - 125_928_448:+,} (expected +2,560)"
    print(f"B-6 PASS ✓: Full model params = {n_params:,} (125.9M + IHA overhead)")

    # ── TEST B-7: Global blocks use IHAGlobalAttention ───────────
    for i, block in enumerate(model.blocks):
        if i in cfg.global_layers:
            assert isinstance(block.attention, IHAGlobalAttention), \
                f"Block {i} should use IHAGlobalAttention, got {type(block.attention)}"
        else:
            assert isinstance(block.attention, GroupedQueryAttention), \
                f"Block {i} should use GroupedQueryAttention, got {type(block.attention)}"
    print("B-7 PASS ✓: correct attention class in all 16 blocks")

    # ── TEST B-8: layer_outputs still length 33 ───────────────────
    x_ids = torch.randint(0, 32010, (2, 128))
    y_ids = torch.randint(0, 32010, (2, 128))
    logits, loss = model(x_ids, labels=y_ids)
    # assert in model.forward() checks len == 33 without firing
    print("B-8 PASS ✓: layer_outputs = 33 (AttnRes sees no change)")

    # ── TEST B-9: Backward through IHA global layers ─────────────
    loss.backward()
    no_grad = [(n, p) for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"Params with no grad: {[n for n,_ in no_grad[:5]]}"
    # IHA params should have gradients
    for n, p in model.named_parameters():
        if 'alpha_Q' in n or 'alpha_K' in n or 'alpha_V' in n or '.R' in n:
            assert p.grad is not None, f"IHA param {n} has no gradient"
    print("B-9 PASS ✓: gradients flow through IHA global layers")

    # ── TEST B-10: IHA global != local (different classes) ────────
    assert type(model.blocks[0].attention) != type(model.blocks[3].attention)
    print("B-10 PASS ✓: local (GQA) ≠ global (IHA) attention classes")

    print("\nSTAGE B FULLY PASSED ✓ — IHAGlobalAttention integrated")

if __name__ == '__main__':
    test_stageB()
