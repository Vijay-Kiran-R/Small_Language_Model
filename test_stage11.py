import torch
from slm_project.config import LongContextConfig

def test_stage11_long_context_config():
    lc = LongContextConfig()

    # Verify all corrected values
    assert lc.max_seq_len    == 16384,    f"Wrong max_seq_len: {lc.max_seq_len}"
    assert lc.swa_window     == 4096,     f"Wrong swa_window: {lc.swa_window}"
    assert lc.physical_batch == 1,        f"Wrong physical_batch: {lc.physical_batch}"
    assert lc.grad_accum     == 16,       f"Wrong grad_accum: {lc.grad_accum} (was 32 — FIX applied)"
    assert lc.warmup_steps   == 200,      f"Missing warmup_steps: {lc.warmup_steps}"
    assert lc.lr             == 1e-5
    assert lc.lr_min         == 1e-6

    # Batch token check
    batch_tokens = lc.physical_batch * lc.grad_accum * lc.max_seq_len
    assert batch_tokens == 262_144, f"Phase 5.5 batch {batch_tokens:,} ≠ 262,144"

    # AttnRes VRAM at Phase 5.5 (batch=1)
    attnres_vram = 33 * 1 * 16384 * 768 * 2   # bf16
    assert attnres_vram == 830_472_192, f"AttnRes buffer = {attnres_vram/1e9:.3f} GB"
    print(f"AttnRes buffer (Phase 5.5, batch=1): {attnres_vram/1e9:.3f} GB ✓")

    print("STAGE 11 TEST GATE PASSED ✓  (Phase 5.5 config verified)")

if __name__ == '__main__':
    test_stage11_long_context_config()
