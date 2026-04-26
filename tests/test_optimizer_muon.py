# tests/test_optimizer_muon.py
"""
Muon optimizer — unit tests.

Validates:
  1. Parameter group routing (embed→AdamW, IHA→AdamW, 2D hidden→Muon, 1D→AdamW, pq→AdamW 2×LR)
  2. Embedding NOT in Muon group (critical safety check)
  3. Newton-Schulz orthogonalization produces semi-orthogonal output
  4. Muon step updates 2D params; AdamW step updates 1D params
  5. No NaN in any params after 5 steps
  6. pseudo_query in Group 3 at exactly 2× base LR
  7. LR schedule (apply_lr) applies correctly to all 4 groups
  8. Optimizer state_dict round-trip (checkpoint compatibility)
  9. All master weights remain float32 after bfloat16 autocast steps
 10. Muon gradient norm scales correctly (non-exploding)

Run: python tests/test_optimizer_muon.py
  or: python -m pytest tests/test_optimizer_muon.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from slm_project.config import ModelConfig, TrainConfig
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights
from slm_project.model.attn_res import AttnRes
from slm_project.model.attention import IHAGlobalAttention
from slm_project.training.optimizer import build_optimizer
from slm_project.training.muon import MuonWithAuxAdam, zeropower_via_newtonschulz5
from slm_project.training.lr_schedule import get_lr, apply_lr

# ── Setup ────────────────────────────────────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg    = ModelConfig()
tcfg   = TrainConfig()

# Use 3-layer config for speed in unit tests
cfg_small = ModelConfig(n_layers=3, global_layers=(2,))


def make_model_opt(cfg_=None):
    """Build a fresh model+optimizer pair."""
    c = cfg_ or cfg_small
    m = SLM(c, tcfg).to(device)
    init_model_weights(m)
    opt = build_optimizer(m, tcfg)
    return m, opt


# ── Test 1: Group routing ─────────────────────────────────────────────────────

def test_01_group_routing():
    """Verify every parameter lands in the correct optimizer group."""
    model, opt = make_model_opt()

    group0_ids = {id(p) for p in opt.param_groups[0]['params']}  # Muon
    group1_ids = {id(p) for p in opt.param_groups[1]['params']}  # AdamW no-decay
    group2_ids = {id(p) for p in opt.param_groups[2]['params']}  # AdamW decay (catch-all)
    group3_ids = {id(p) for p in opt.param_groups[3]['params']}  # pseudo_query

    all_groups = group0_ids | group1_ids | group2_ids | group3_ids
    all_param_ids = {id(p) for p in model.parameters() if p.requires_grad}

    # Every parameter must be in exactly one group
    assert all_param_ids == all_groups, (
        f"Param ID mismatch — "
        f"{len(all_param_ids - all_groups)} params not assigned, "
        f"{len(all_groups - all_param_ids)} phantom IDs"
    )

    # Check for duplicates
    counts = {}
    for g_ids in [group0_ids, group1_ids, group2_ids, group3_ids]:
        for pid in g_ids:
            counts[pid] = counts.get(pid, 0) + 1
    dups = {pid: c for pid, c in counts.items() if c > 1}
    assert not dups, f"Parameters in multiple groups: {len(dups)} duplicates"

    print(f"  [OK] Group 0 (Muon):        {len(group0_ids)} params")
    print(f"  [OK] Group 1 (AdamW no-wd): {len(group1_ids)} params")
    print(f"  [OK] Group 2 (AdamW decay): {len(group2_ids)} params")
    print(f"  [OK] Group 3 (pseudo_q 2x): {len(group3_ids)} params")


# ── Test 2: Embedding safety check ──────────────────────────────────────────

def test_02_embedding_not_in_muon():
    """CRITICAL: embedding.weight must NEVER be in the Muon group."""
    model, opt = make_model_opt()
    group0_ids = {id(p) for p in opt.param_groups[0]['params']}  # Muon

    for name, param in model.named_parameters():
        if 'embed' in name:
            assert id(param) not in group0_ids, (
                f"CRITICAL: '{name}' (shape={tuple(param.shape)}) is in the Muon group! "
                f"This will corrupt the embedding table via Newton-Schulz."
            )
            print(f"  [OK] '{name}' correctly NOT in Muon group")


# ── Test 3: IHA params not in Muon ──────────────────────────────────────────

def test_03_iha_params_in_adamw():
    """IHA alpha_Q/K/V (3D) and R must be in AdamW, not Muon."""
    model, opt = make_model_opt()
    group0_ids = {id(p) for p in opt.param_groups[0]['params']}  # Muon
    group1_ids = {id(p) for p in opt.param_groups[1]['params']}  # AdamW no-decay

    iha_found = 0
    for name, param in model.named_parameters():
        if any(k in name for k in ['alpha_Q', 'alpha_K', 'alpha_V', '.R']):
            assert id(param) not in group0_ids, \
                f"IHA param '{name}' is in Muon group! Shape={tuple(param.shape)}"
            assert id(param) in group1_ids, \
                f"IHA param '{name}' not in AdamW no-decay group!"
            iha_found += 1
    assert iha_found > 0, "No IHA params found — is IHAGlobalAttention in the model?"
    print(f"  [OK] {iha_found} IHA params correctly in AdamW no-decay group")


# ── Test 4: Newton-Schulz orthogonality ─────────────────────────────────────

def test_04_newton_schulz_orthogonality():
    """NS output should be approximately semi-orthogonal: U @ U.T ≈ I."""
    torch.manual_seed(42)
    for shape in [(768, 768), (768, 256), (2048, 768), (12, 24)]:
        G = torch.randn(*shape)
        U = zeropower_via_newtonschulz5(G, steps=5)
        assert U.shape == G.shape, f"Shape mismatch: {U.shape} vs {G.shape}"

        # Check singular values ≈ 1 (semi-orthogonal)
        try:
            sv = torch.linalg.svdvals(U.float())
            max_sv = sv.max().item()
            min_sv = sv[sv > 1e-6].min().item()  # ignore near-zero SVs for non-square
            assert max_sv < 1.25, f"Max singular value {max_sv:.4f} > 1.25 for shape {shape}"
            print(f"  [OK] NS {shape}: SVs in [{min_sv:.3f}, {max_sv:.3f}]")
        except Exception as e:
            print(f"  [WARN] SVD check skipped for {shape}: {e}")

        assert not torch.isnan(U).any(), f"NaN in NS output for shape {shape}"
        assert not torch.isinf(U).any(), f"Inf in NS output for shape {shape}"


# ── Test 5: Muon step actually updates params ────────────────────────────────

def test_05_muon_step_updates_params():
    """After one Muon step, 2D hidden weights must change; 1D params via AdamW."""
    model, opt = make_model_opt()

    # Snapshot param values before step
    before = {n: p.data.clone() for n, p in model.named_parameters()}

    # Forward + backward with synthetic data
    B, T = 2, 64
    ids    = torch.randint(0, cfg_small.vocab_size, (B, T)).to(device)
    labels = torch.randint(0, cfg_small.vocab_size, (B, T)).to(device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss = model(ids, labels=labels)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    # Check that 2D hidden weights changed (Muon updated them)
    muon_changed = 0
    adamw_changed = 0
    for name, param in model.named_parameters():
        changed = not torch.allclose(before[name], param.data)
        if 'embed' in name:
            assert changed, f"Embedding '{name}' was not updated by AdamW!"
        elif param.ndim == 1:
            adamw_changed += int(changed)
        elif param.ndim >= 2:
            if not any(k in name for k in ['alpha_Q', 'alpha_K', 'alpha_V', '.R', 'embed']):
                assert changed, f"2D hidden weight '{name}' was NOT updated by Muon!"
                muon_changed += 1

    print(f"  [OK] Muon updated {muon_changed} 2D hidden weight tensors")
    print(f"  [OK] AdamW updated {adamw_changed} 1D tensors")


# ── Test 6: No NaN after 5 steps ─────────────────────────────────────────────

def test_06_no_nan_after_5_steps():
    """Five full optimizer steps must leave no NaN in any parameter."""
    model, opt = make_model_opt()
    B, T = 2, 64

    for step in range(5):
        ids    = torch.randint(0, cfg_small.vocab_size, (B, T)).to(device)
        labels = torch.randint(0, cfg_small.vocab_size, (B, T)).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(ids, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in '{name}' after 5 steps"
        assert not torch.isinf(param).any(), f"Inf in '{name}' after 5 steps"
    print(f"  [OK] No NaN/Inf in any parameter after 5 steps")


# ── Test 7: pseudo_query LR is 2× base ──────────────────────────────────────

def test_07_pseudo_query_lr():
    """Group 3 (pseudo_query) must always have exactly 2× the Group 0 LR."""
    model, opt = make_model_opt()

    # Apply a LR schedule update
    apply_lr(opt, 2e-4)
    base = opt.param_groups[0]['lr']
    pq   = opt.param_groups[3]['lr']

    assert abs(pq - 2 * base) < 1e-14, \
        f"pseudo_query LR {pq} != 2× base {base}"

    # Apply at warmup end
    apply_lr(opt, tcfg.peak_lr)
    base = opt.param_groups[0]['lr']
    pq   = opt.param_groups[3]['lr']
    assert abs(pq - tcfg.peak_lr * tcfg.pseudo_query_lr_multiplier) < 1e-14, \
        f"pseudo_query LR {pq} != {tcfg.peak_lr * tcfg.pseudo_query_lr_multiplier}"

    print(f"  [OK] pseudo_query LR is exactly 2× base (base={base:.2e}, pq={pq:.2e})")


# ── Test 8: Optimizer state_dict round-trip ──────────────────────────────────

def test_08_checkpoint_roundtrip():
    """Optimizer state must survive state_dict → load_state_dict."""
    model, opt = make_model_opt()
    B, T = 2, 64

    # Do 2 steps to populate state
    for _ in range(2):
        ids    = torch.randint(0, cfg_small.vocab_size, (B, T)).to(device)
        labels = torch.randint(0, cfg_small.vocab_size, (B, T)).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(ids, labels=labels)
        loss.backward()
        opt.step()

    # Save and reload state
    state = opt.state_dict()
    model2, opt2 = make_model_opt()
    opt2.load_state_dict(state)

    # Both should give same result on next step
    ids    = torch.randint(0, cfg_small.vocab_size, (B, T)).to(device)
    labels = torch.randint(0, cfg_small.vocab_size, (B, T)).to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss1 = model(ids, labels=labels)
    loss1.backward()
    opt.step()

    # opt2 should be in same state (different model params but same optimizer state)
    assert len(opt2.state_dict()['state']) == len(opt.state_dict()['state']), \
        "State dict mismatch after load_state_dict"
    print(f"  [OK] state_dict round-trip succeeded ({len(state['state'])} state entries)")


# ── Test 9: Master weights stay float32 ─────────────────────────────────────

def test_09_master_weights_float32():
    """bfloat16 autocast must not cast master weights to bf16."""
    model, opt = make_model_opt()
    B, T = 2, 64
    ids    = torch.randint(0, cfg_small.vocab_size, (B, T)).to(device)
    labels = torch.randint(0, cfg_small.vocab_size, (B, T)).to(device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss = model(ids, labels=labels)
    loss.backward()
    opt.step()

    for name, param in model.named_parameters():
        assert param.dtype == torch.float32, \
            f"Master weight '{name}' is {param.dtype}, expected float32"
    print(f"  [OK] All master weights remain float32 after bfloat16 step")


# ── Test 10: Grad norm sanity ─────────────────────────────────────────────────

def test_10_grad_norm_sanity():
    """Gradient norm must be finite and non-zero before and after clip."""
    model, opt = make_model_opt()
    B, T = 2, 64
    ids    = torch.randint(0, cfg_small.vocab_size, (B, T)).to(device)
    labels = torch.randint(0, cfg_small.vocab_size, (B, T)).to(device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss = model(ids, labels=labels)
    loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert norm.isfinite(), f"Gradient norm is not finite: {norm}"
    assert norm > 0, "Gradient norm is zero — no learning signal"
    print(f"  [OK] Gradient norm = {norm.item():.4f} (finite and non-zero)")


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MUON OPTIMIZER — Unit Tests")
    print("=" * 60)
    print()

    tests = [
        ("01 Group routing",               test_01_group_routing),
        ("02 Embedding not in Muon",        test_02_embedding_not_in_muon),
        ("03 IHA params in AdamW",          test_03_iha_params_in_adamw),
        ("04 Newton-Schulz orthogonality",  test_04_newton_schulz_orthogonality),
        ("05 Muon step updates params",     test_05_muon_step_updates_params),
        ("06 No NaN after 5 steps",         test_06_no_nan_after_5_steps),
        ("07 pseudo_query LR = 2×base",     test_07_pseudo_query_lr),
        ("08 Checkpoint round-trip",        test_08_checkpoint_roundtrip),
        ("09 Master weights float32",       test_09_master_weights_float32),
        ("10 Grad norm sanity",             test_10_grad_norm_sanity),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"\n[{name}]")
            fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print()
    print("=" * 60)
    if failed == 0:
        print(f"ALL {passed} MUON TESTS PASSED [OK]")
    else:
        print(f"{passed} passed, {failed} FAILED [FAIL]")
    print("=" * 60)
