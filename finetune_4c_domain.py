# finetune_4c_domain.py
import torch, glob
from slm_project.config import ModelConfig, TrainConfig, DomainFTConfig
from slm_project.model.model import SLM
from slm_project.training.optimizer import build_optimizer
from slm_project.training.finetune import SFTTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg   = ModelConfig()
tcfg  = TrainConfig()
dfcfg = DomainFTConfig()

# Load CoT checkpoint (or latest pretrained/SFT if CoT not done)
ckpts = sorted(glob.glob('checkpoints/step_*.pt'))
if ckpts:
    ckpt  = torch.load(ckpts[-1], map_location='cpu')
    model = SLM(cfg, tcfg).to(device)
    state_dict = {k: v for k, v in ckpt['model_state'].items() if not k.endswith('_cached')}
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint for Domain FT: step {ckpt['global_step']:,}, "
          f"{ckpt['tokens_seen']/1e9:.3f}B tokens")
else:
    print("WARNING: No checkpoints found. Using untrained model for testing.")
    model = SLM(cfg, tcfg).to(device)

# Domain FT optimizer: LR=5e-6 (lower than CoT)
import torch.optim as optim
optimizer = optim.AdamW(
    model.parameters(),
    lr=dfcfg.lr,          # 5e-6
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)

trainer = SFTTrainer(model, optimizer, dfcfg, device)

# TODO: load Domain dataset (EmpatheticDialogues, AugESC, ESConv, ESCoT, GoEmotions, BANKING77, etc.)
# run trainer.train_step() for 2-3 epochs with early stopping on domain eval

print("Domain Phase 4c ready. Connect Domain dataloader and run.")
