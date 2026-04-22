from slm_project.config import ModelConfig, TrainConfig, LongContextConfig

def test_stage0_config():
    cfg  = ModelConfig()
    tcfg = TrainConfig()
    lc   = LongContextConfig()

    # Architecture
    assert cfg.n_layers         == 16
    assert cfg.global_layers    == (3, 7, 11, 15)
    assert cfg.n_attnres_sublayers == 32    # 16 × 2
    assert cfg.n_layer_outputs     == 33    # 1 + 32

    # Batch math
    pretrain_batch = cfg.max_seq_len * tcfg.physical_batch_seqs * tcfg.grad_accum_steps
    assert pretrain_batch == 262_144, f"Pretrain batch {pretrain_batch} ≠ 262,144"

    p55_batch = lc.max_seq_len * lc.physical_batch * lc.grad_accum
    assert p55_batch == 262_144, f"Phase 5.5 batch {p55_batch} ≠ 262,144"

    # WSD thresholds
    assert tcfg.min_pretrain_steps == 70_000
    assert tcfg.plateau_steps       == 3_000

    # Decay rates
    assert tcfg.peak_lr == 3e-4
    assert tcfg.min_lr  == 3e-5
    assert abs(tcfg.min_lr / tcfg.peak_lr - 0.1) < 1e-9   # min = 10% of peak

    print(f"n_layers         = {cfg.n_layers}  ✓")
    print(f"global_layers    = {cfg.global_layers}  ✓")
    print(f"layer_outputs    = {cfg.n_layer_outputs}  ✓")
    print(f"pretrain batch   = {pretrain_batch:,} tokens  ✓")
    print(f"Phase 5.5 batch  = {p55_batch:,} tokens  ✓")
    print("STAGE 0 PASSED ✓")

if __name__ == '__main__':
    test_stage0_config()
