import torch
from slm_project.config import ModelConfig
from slm_project.model.attention import IHAGlobalAttention, GroupedQueryAttention
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights

def test_B_1_through_B_10():
    cfg = ModelConfig()

    # ── TEST B-1: IHA global attn shape ──────────────────────────
    iha = IHAGlobalAttention(cfg, P=2)
    x = torch.randn(2, 64, 768)
    out = iha(x)
    assert out.shape == (2, 64, 768), f"Wrong shape: {out.shape}"
    assert not torch.isnan(out).any()

    # ── TEST B-2: At init, IHA ≈ standard MHA (identity start) ──
    assert out.norm() > 0, "Output is zero — init broken"
    assert out.norm() < 1e4, "Output norm exploding — init broken"

    # ── TEST B-3: α init is identity-like ────────────────────────
    for h in range(cfg.n_heads_q):
        for p in range(2):
            assert iha.alpha_Q.data[h, h, p] == 1.0

    # ── TEST B-4: R init selects pseudo j=0 ──────────────────────
    for h in range(cfg.n_heads_q):
        assert iha.R.data[h, h * 2] == 1.0
        assert iha.R.data[h, h * 2 + 1] == 0.0

    # ── TEST B-5: Parameter count ─────────────────────────────────
    iha_extra = sum(p.numel() for p in [iha.alpha_Q, iha.alpha_K, iha.alpha_V, iha.R])
    assert iha_extra == 640

    # ── TEST B-6: Full model params ────────────────────────────
    model = SLM(cfg)
    init_model_weights(model)
    n_params = model.get_num_params()
    assert n_params == 125_931_008

    # ── TEST B-7: Global blocks use IHAGlobalAttention ───────────
    for i, block in enumerate(model.blocks):
        if i in cfg.global_layers:
            assert isinstance(block.attention, IHAGlobalAttention)
        else:
            assert isinstance(block.attention, GroupedQueryAttention)

    # ── TEST B-8: layer_outputs still length 33 ───────────────────
    x_ids = torch.randint(0, 32010, (2, 128))
    y_ids = torch.randint(0, 32010, (2, 128))
    logits, loss = model(x_ids, labels=y_ids)

    # ── TEST B-9: Backward through IHA global layers ─────────────
    loss.backward()
    no_grad = [(n, p) for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0
    for n, p in model.named_parameters():
        if 'alpha_Q' in n or 'alpha_K' in n or 'alpha_V' in n or '.R' in n:
            assert p.grad is not None

    # ── TEST B-10: IHA global != local (different classes) ────────
    assert type(model.blocks[0].attention) != type(model.blocks[3].attention)
