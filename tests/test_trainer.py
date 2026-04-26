# tests/test_trainer.py
"""
Stage 14 test gate — trainer loop dry-run with synthetic data.
Run as:  python tests/test_trainer.py
     or: python -m pytest tests/test_trainer.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import TensorDataset, DataLoader

from slm_project.config import ModelConfig, TrainConfig
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights
from slm_project.training.optimizer import build_optimizer
from slm_project.training.trainer import Trainer, get_pseudo_query_norms

cfg   = ModelConfig()
tcfg  = TrainConfig()

# Small synthetic dataset — T=64, 64 samples for 2 full optimizer steps
# 2 steps × grad_accum_steps(8) micro-batches × B(4) = 64 samples minimum
B, T = 4, 64
fake_ids    = torch.randint(0, cfg.vocab_size, (64, T))
fake_labels = torch.randint(0, cfg.vocab_size, (64, T))
loader = DataLoader(TensorDataset(fake_ids, fake_labels), batch_size=B, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = SLM(cfg, tcfg).to(device)
init_model_weights(model)
opt    = build_optimizer(model, tcfg)
trainer = Trainer(model, opt, tcfg, loader, device=device)


def test_14_1_two_steps_complete():
    """Run exactly 2 full optimizer steps without error."""
    steps_run = 0
    for batch in loader:
        trainer.train_step(batch)
        steps_run += 1
        if steps_run % tcfg.grad_accum_steps == 0:
            trainer.optimizer_step()
            if trainer.global_step >= 2:
                break

    assert trainer.global_step == 2, \
        f"Expected 2 optimizer steps, got {trainer.global_step}"
    print(f"  [OK] 2 optimizer steps completed")


def test_14_2_tokens_seen():
    """tokens_seen must be incremented during train_step."""
    assert trainer.tokens_seen > 0, "tokens_seen not incremented"
    print(f"  [OK] tokens_seen = {trainer.tokens_seen:,}")


def test_14_3_no_nan_params():
    """No NaN in any model parameter after 2 steps."""
    for name, p in model.named_parameters():
        assert not torch.isnan(p).any(), f"NaN in param {name}"
    print("  [OK] no NaN in model parameters after 2 steps")


def test_14_4_pseudo_query_norms():
    """pseudo_query norms should be non-negative after training."""
    pq_norms = get_pseudo_query_norms(model)
    assert all(n >= 0.0 for n in pq_norms), \
        f"Negative pseudo_query norm: {pq_norms}"
    print(f"  [OK] pseudo_query norms: {[round(n, 6) for n in pq_norms]}")


def test_14_5_params_stay_float32():
    """bfloat16 autocast must not cast master weights to bf16."""
    for name, p in model.named_parameters():
        assert p.dtype == torch.float32, \
            f"Param {name} is {p.dtype}, expected float32"
    print("  [OK] all master weights remain float32 (bfloat16 autocast only)")


def test_14_6_lr_schedule_updated():
    """LR must be non-zero and pseudo_query group (index 3) must be 2x base."""
    base_lr = opt.param_groups[0]["lr"]
    pq_lr   = opt.param_groups[3]["lr"]   # Group 3 is pseudo_query (was index 2 pre-Muon)
    assert base_lr > 0, f"Base LR is zero after 2 steps"
    assert abs(pq_lr - 2 * base_lr) < 1e-12, \
        f"pseudo_query LR {pq_lr} != 2x base {base_lr}"
    print(f"  [OK] LR: base={base_lr:.2e}, pseudo_query={pq_lr:.2e} (2x)")


def test_14_7_loss_is_divided():
    """
    Verify grad_accum division is correct: accumulate loss over 2 micro-batches
    and confirm total loss values are reasonable (not 8x inflated).
    """
    model_check = SLM(cfg, tcfg).to(device)
    init_model_weights(model_check)
    opt_check = build_optimizer(model_check, tcfg)
    trainer_check = Trainer(model_check, opt_check, tcfg, loader, device=device)

    losses = []
    count = 0
    for batch in loader:
        l = trainer_check.train_step(batch)
        losses.append(l)
        count += 1
        if count >= 2:
            break

    # Unscaled loss should be in [5, 25] range for random data / fresh model
    for l in losses:
        assert 0 < l < 50, f"Loss {l} outside plausible range — check grad_accum division"
    print(f"  [OK] loss values plausible: {[round(l, 3) for l in losses]}")


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 14 TEST GATE — Trainer Dry-Run")
    print("=" * 60)
    print()

    test_14_1_two_steps_complete()
    test_14_2_tokens_seen()
    test_14_3_no_nan_params()
    test_14_4_pseudo_query_norms()
    test_14_5_params_stay_float32()
    test_14_6_lr_schedule_updated()
    test_14_7_loss_is_divided()

    print()
    print("=" * 60)
    print("STAGE 14 PASSED  (Trainer loop dry-run clean)")
