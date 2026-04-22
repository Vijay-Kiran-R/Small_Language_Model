import torch
from slm_project.config import ModelConfig, TrainConfig
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights
from slm_project.model.attn_res import AttnRes

def test_backward_ckpt_16_layer():
    cfg   = ModelConfig()   # 16 layers
    tcfg  = TrainConfig()
    model = SLM(cfg, tcfg)
    init_model_weights(model)

    B, T = 2, 256   # Short seq for speed — all 16 blocks still exercised
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))

    # ── test_11: Backward with gradient checkpointing ────────────
    print("test_11: Running backward with gradient checkpointing ...")
    try:
        logits, loss = model(x, labels=y, use_checkpoint=True)
        loss.backward()
    except RuntimeError as e:
        raise AssertionError(
            f"FAIL test_11: backward with grad checkpointing raised: {e}\n"
            f"Fix: pack/unpack tuple pattern in model._forward_with_checkpoint()"
        )
    no_grad = [(n, p) for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"No-grad params: {[n for n,_ in no_grad[:5]]}"
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf grad in {name}"
    print("test_11 PASS ✓  (backward + grad checkpointing: 16 layers clean)")

    # ── test_12: AttnRes softmax is over dim=0 ───────────────────
    model.eval()
    with torch.no_grad():
        _ = model(x)
    checked = 0
    for name, module in model.named_modules():
        if isinstance(module, AttnRes) and module._last_alpha is not None:
            alpha = module._last_alpha
            assert (alpha.sum(dim=0) - 1.0).abs().max() < 1e-5, \
                f"Softmax not summing to 1 in {name}"
            checked += 1
    assert checked >= 32, f"Only checked {checked} AttnRes modules, expected 32"
    print(f"test_12 PASS ✓  (softmax dim=0 verified in all {checked} AttnRes modules)")

    # ── test_13: QK-Norm per-head = 1,024 params ─────────────────
    for i, block in enumerate(model.blocks):
        qk_q = sum(p.numel() for p in block.attention.q_norms.parameters())
        qk_k = sum(p.numel() for p in block.attention.k_norms.parameters())
        assert qk_q + qk_k == 1024, \
            f"Block {i}: QK-Norm params = {qk_q+qk_k}, expected 1024"
    print("test_13 PASS ✓  (QK-Norm = 1,024 params in all 16 blocks)")

if __name__ == '__main__':
    test_backward_ckpt_16_layer()
