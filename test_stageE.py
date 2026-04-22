# Quick dry-run to verify GRPO machinery without real data
import torch
from slm_project.config import ModelConfig, TrainConfig, GRPOConfig
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights
from slm_project.training.grpo_trainer import GRPOTrainer

def test_stageE():
    cfg   = ModelConfig()
    tcfg  = TrainConfig()
    gcfg  = GRPOConfig(max_steps=3, batch_questions=2, G=2)  # Tiny for dry-run

    model = SLM(cfg, tcfg).cuda()
    init_model_weights(model)

    trainer = GRPOTrainer(model, gcfg, device='cuda')

    # Check reference model is separate object
    assert trainer.ref_model is not model
    assert trainer.ref_model is not trainer.model
    assert not any(p.requires_grad for p in trainer.ref_model.parameters())
    print("Reference model correctly frozen ✓")

    # Check reward functions return correct types
    from slm_project.training.grpo_trainer import accuracy_reward_math
    r = accuracy_reward_math("\\boxed{42}", "42")
    assert isinstance(r, float)
    assert 0.0 <= r <= 1.0
    print("Reward functions return valid floats ✓")

    # Verify model still has correct params
    assert model.get_num_params() == 125_931_008
    print("Model params correct after GRPO init ✓")

    print("STAGE E PASSED ✓  (GRPO dry-run clean)")

if __name__ == '__main__':
    test_stageE()
