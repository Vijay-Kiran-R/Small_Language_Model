# finetune_4a_sft.py
import torch, glob
from slm_project.config import ModelConfig, TrainConfig, FinetuneConfig
from slm_project.model.model import SLM
from slm_project.training.optimizer import build_optimizer
from slm_project.training.finetune import SFTTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg   = ModelConfig()
tcfg  = TrainConfig()
fcfg  = FinetuneConfig()

# Load pretrained checkpoint
ckpts = sorted(glob.glob('trained_models/step_*.pt'))
if ckpts:
    ckpt  = torch.load(ckpts[-1], map_location='cpu')
    model = SLM(cfg, tcfg).to(device)
    # Filter out cached rope parameters in case seq len changed
    state_dict = {k: v for k, v in ckpt['model_state'].items() if not k.endswith('_cached')}
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained model: step {ckpt['global_step']:,}, "
          f"{ckpt['tokens_seen']/1e9:.3f}B tokens")
else:
    print("WARNING: No checkpoints found. Using untrained model for testing.")
    model = SLM(cfg, tcfg).to(device)

# Fine-tuning optimizer: 10× lower LR, no WSD schedule
import torch.optim as optim
optimizer = optim.AdamW(
    model.parameters(),
    lr=fcfg.lr,           # 3e-5
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)

trainer = SFTTrainer(model, optimizer, fcfg, device)

# TODO: load SFT dataset (SmolTalk + Magpie-Ultra + OASST2 + etc.)
# and run trainer.train_step() for 1.5–2.5 epochs with early stopping

print("SFT Phase 4a ready. Connect SFT dataloader and run.")
