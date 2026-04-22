from slm_project.config import ModelConfig
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights
from slm_project.model.attention import IHAGlobalAttention

def test_stageC():
    cfg = ModelConfig()
    model = SLM(cfg)
    init_model_weights(model)  # Must print both verification lines

    # Verify AttnRes pseudos still zero
    from slm_project.model.attn_res import AttnRes
    import torch
    for name, mod in model.named_modules():
        if isinstance(mod, AttnRes):
            assert mod.pseudo_query.allclose(torch.zeros_like(mod.pseudo_query)), \
                f"pseudo_query not zero in {name}"

    # Verify IHA identity not overwritten
    for name, mod in model.named_modules():
        if isinstance(mod, IHAGlobalAttention):
            for h in range(mod.n_heads_q):
                assert mod.alpha_Q.data[h, h, 0] == 1.0
                assert mod.alpha_Q.data[h, h, 1] == 1.0
            for h in range(mod.n_heads_kv):
                assert mod.alpha_K.data[h, h, 0] == 1.0
                assert mod.alpha_V.data[h, h, 0] == 1.0

    print("STAGE C PASSED ✓  (init order verified: standard → zero_pq → IHA_identity)")

if __name__ == '__main__':
    test_stageC()
