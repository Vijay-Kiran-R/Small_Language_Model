"""
Direct Preference Optimization for 125M SLM.

CRITICAL PARAMETERS:
  beta = 0.05   START HERE — NOT 0.1 (too aggressive at 125M scale)
  lr   = 5e-7   Very conservative — DPO is sensitive
  max_epochs = 1  NEVER more than 1 epoch (reward hacking at 125M)

KL MONITORING (every 100 steps):
  KL < 0.2 sustained after step 200 → may carefully increase beta to 0.08
  KL > 0.3 at ANY step              → reduce beta to 0.02 immediately
  If KL exceeds 0.3 twice           → reduce to 0.01 and restart epoch

LOG: wandb.log({'dpo/kl': kl_div, 'dpo/beta': beta, 'dpo/reward_acc': acc})
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class DPOState:
    beta:             float = 0.05
    step:             int   = 0
    kl_history:       list  = None  # last 10 KL values

    def __post_init__(self):
        if self.kl_history is None:
            self.kl_history = []

    def update_beta(self, kl: float) -> None:
        """Auto-adjust beta based on KL divergence monitoring."""
        self.kl_history.append(kl)
        if len(self.kl_history) > 10:
            self.kl_history.pop(0)

        if kl > 0.3:
            old_beta = self.beta
            self.beta = max(0.01, self.beta / 2)
            print(f"⚠ KL={kl:.4f} > 0.3 → beta {old_beta:.3f} → {self.beta:.3f}")
        elif (self.step > 200 and
              len(self.kl_history) >= 5 and
              all(k < 0.2 for k in self.kl_history[-5:]) and
              self.beta < 0.1):
            old_beta = self.beta
            self.beta = min(0.1, self.beta * 1.2)
            print(f"✓ KL stable < 0.2 → beta {old_beta:.3f} → {self.beta:.3f}")


def dpo_loss(
    policy_logps_chosen:   torch.Tensor,   # log-probs of chosen under policy
    policy_logps_rejected: torch.Tensor,   # log-probs of rejected under policy
    ref_logps_chosen:      torch.Tensor,   # log-probs of chosen under reference
    ref_logps_rejected:    torch.Tensor,   # log-probs of rejected under reference
    beta:                  float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DPO loss and reward accuracy.

    DPO objective:
      log σ(β × (log π(y_w|x) - log π_ref(y_w|x))
              - β × (log π(y_l|x) - log π_ref(y_l|x)))

    Returns (loss, reward_accuracy).
    """
    chosen_logratios   = policy_logps_chosen   - ref_logps_chosen
    rejected_logratios = policy_logps_rejected - ref_logps_rejected
    logits = beta * (chosen_logratios - rejected_logratios)
    loss   = -F.logsigmoid(logits).mean()
    reward_acc = (logits > 0).float().mean()
    return loss, reward_acc


def compute_kl_divergence(
    policy_logps: torch.Tensor,
    ref_logps:    torch.Tensor,
) -> torch.Tensor:
    """KL(policy || reference) = E[log π - log π_ref]"""
    return (policy_logps - ref_logps).mean()
