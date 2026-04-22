# Verify DPO loss implementation
import torch
from slm_project.training.dpo_trainer import dpo_loss, DPOState

def test_stage10_dpo_loss():
    # Synthetic test: when chosen is clearly better, loss should be low
    pol_chosen   = torch.tensor([-1.0, -1.2, -0.8])   # high prob under policy
    pol_rejected = torch.tensor([-3.0, -3.5, -2.8])   # low prob under policy
    ref_chosen   = torch.tensor([-2.0, -2.0, -2.0])   # same reference for all
    ref_rejected = torch.tensor([-2.0, -2.0, -2.0])

    loss, acc = dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=0.05)
    assert loss.item() > 0, "DPO loss must be positive"
    assert acc.item() > 0.5, "Reward accuracy should be > 50% when chosen is better"

    # Beta auto-adjustment
    state = DPOState(beta=0.05)
    state.step = 201
    for _ in range(5):
        state.update_beta(0.15)   # KL < 0.2, should increase beta
    assert state.beta > 0.05, "Beta should increase when KL is consistently < 0.2"

    state2 = DPOState(beta=0.05)
    state2.update_beta(0.4)   # KL > 0.3, should immediately reduce beta
    assert state2.beta < 0.05, "Beta must reduce immediately when KL > 0.3"

    print("STAGE 10 TEST GATE PASSED ✓  (DPO loss + beta control verified)")

if __name__ == '__main__':
    test_stage10_dpo_loss()
