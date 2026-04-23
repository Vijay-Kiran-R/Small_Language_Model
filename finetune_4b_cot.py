# finetune_4b_cot.py
import torch, glob
from slm_project.config import ModelConfig, TrainConfig, CoTConfig
from slm_project.model.model import SLM
from slm_project.training.optimizer import build_optimizer
from slm_project.training.finetune import SFTTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg   = ModelConfig()
tcfg  = TrainConfig()
ccfg  = CoTConfig()

# Load SFT checkpoint (or latest pretrained if SFT not done)
ckpts = sorted(glob.glob('trained_models/step_*.pt'))
if ckpts:
    ckpt  = torch.load(ckpts[-1], map_location='cpu')
    model = SLM(cfg, tcfg).to(device)
    state_dict = {k: v for k, v in ckpt['model_state'].items() if not k.endswith('_cached')}
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint for CoT: step {ckpt['global_step']:,}, "
          f"{ckpt['tokens_seen']/1e9:.3f}B tokens")
else:
    print("WARNING: No checkpoints found. Using untrained model for testing.")
    model = SLM(cfg, tcfg).to(device)

# CoT optimizer: 3× lower LR than SFT (1e-5 vs 3e-5)
import torch.optim as optim
optimizer = optim.AdamW(
    model.parameters(),
    lr=ccfg.lr,           # 1e-5
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)

trainer = SFTTrainer(model, optimizer, ccfg, device)

# TODO: load CoT dataset (OpenR1-Math-220K, etc.)
# 50/50 split: half the batch = CoT examples (with <|think|> reasoning)
#              half the batch = direct response examples (no <|think|>)
# run trainer.train_step() for 2.0 epochs

print("CoT Phase 4b ready. Connect CoT dataloader and run.")
