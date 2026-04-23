import pytest
from slm_project.config import ModelConfig, TrainConfig, LongContextConfig, GRPOConfig

def test_stage0_config():
    cfg  = ModelConfig()
    tcfg = TrainConfig()
    lc   = LongContextConfig()

    # Architecture
    assert cfg.n_layers         == 16
    assert cfg.global_layers    == (3, 7, 11, 15)
    assert cfg.n_attnres_sublayers == 32    # 16 A- 2
    assert cfg.n_layer_outputs     == 33    # 1 + 32

    # Batch math
    pretrain_batch = cfg.max_seq_len * tcfg.physical_batch_seqs * tcfg.grad_accum_steps
    assert pretrain_batch == 262_144, f"Pretrain batch {pretrain_batch} != 262,144"

    p55_batch = lc.max_seq_len * lc.physical_batch * lc.grad_accum
    assert p55_batch == 262_144, f"Phase 5.5 batch {p55_batch} != 262,144"

    # WSD thresholds
    assert tcfg.min_pretrain_steps == 70_000
    assert tcfg.plateau_steps       == 3_000

    # Decay rates
    assert tcfg.peak_lr == 3e-4
    assert tcfg.min_lr  == 3e-5
    assert abs(tcfg.min_lr / tcfg.peak_lr - 0.1) < 1e-9   # min = 10% of peak

def test_stage11_long_context_config():
    lc = LongContextConfig()

    # Verify all corrected values
    assert lc.max_seq_len    == 16384,    f"Wrong max_seq_len: {lc.max_seq_len}"
    assert lc.swa_window     == 4096,     f"Wrong swa_window: {lc.swa_window}"
    assert lc.physical_batch == 1,        f"Wrong physical_batch: {lc.physical_batch}"
    assert lc.grad_accum     == 16,       f"Wrong grad_accum: {lc.grad_accum} (was 32)"
    assert lc.warmup_steps   == 200,      f"Missing warmup_steps: {lc.warmup_steps}"
    assert lc.lr             == 1e-5
    assert lc.lr_min         == 1e-6

    # Batch token check
    batch_tokens = lc.physical_batch * lc.grad_accum * lc.max_seq_len
    assert batch_tokens == 262_144, f"Phase 5.5 batch {batch_tokens:,} != 262,144"

    # AttnRes VRAM at Phase 5.5 (batch=1)
    attnres_vram = 33 * 1 * 16384 * 768 * 2   # bf16
    assert attnres_vram == 830_472_192, f"AttnRes buffer = {attnres_vram/1e9:.3f} GB"

def test_stageA_grpo_config():
    gcfg = GRPOConfig()
    assert gcfg.G               == 8
    assert gcfg.kl_coef         == 0.001
    assert gcfg.max_steps       == 700
    assert gcfg.ref_update_freq == 400
    # Reward weights must sum to 1.0
    assert abs(gcfg.accuracy_weight + gcfg.format_weight + gcfg.language_weight - 1.0) < 1e-6
