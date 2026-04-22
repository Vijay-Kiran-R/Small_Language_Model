import pytest
from slm_project.config import TrainConfig
from slm_project.training.lr_schedule import get_lr

tcfg = TrainConfig()

def test_stage5_wsd_decay():
    # Stable phase — always holds peak_lr
    assert get_lr(2000,  tcfg, None) == 3e-4   # warmup just ended
    assert get_lr(70000, tcfg, None) == 3e-4   # deep in stable phase

    # Decay phase — linear from peak → min over 2000 steps
    decay_at = 72000
    assert get_lr(72000, tcfg, decay_at) == 3e-4   # just triggered
    assert get_lr(73000, tcfg, decay_at) == pytest.approx(1.65e-4, rel=1e-3)  # midway
    assert get_lr(74000, tcfg, decay_at) == 3e-5   # post-decay = min_lr

    # End of training: loss should be approximately 2.9–3.1
    # If still above 3.1 at 20B tokens: extend stable phase; do NOT decay yet
    print("STAGE 5 PASSED ✓  (WSD decay logic verified)")

if __name__ == '__main__':
    test_stage5_wsd_decay()
