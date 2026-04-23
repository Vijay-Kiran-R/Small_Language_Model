"""
CRITICAL STEPS for Phase 5.5:
1. Update swa_window in model from 2048 → 4096 (read from LongContextConfig)
2. Update max_seq_len in model from 8192 → 16384
3. FlashAttention local blocks: window_size = (config.swa_window, 0) = (4096, 0)
   NOT hardcoded to (2048, 0) — read from config EVERY time
4. Use grad_accum=16 (NOT 32) for 262,144 tokens/step
5. warmup_steps=200 (prevents destabilising positions)
"""
import torch, glob
from slm_project.config import ModelConfig, TrainConfig, LongContextConfig
from slm_project.model.model import SLM
from slm_project.training.optimizer import build_optimizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
lc_cfg = LongContextConfig()

# Load DPO checkpoint (or latest pretrained if DPO not done)
ckpts = sorted(glob.glob('trained_models/step_*.pt') + glob.glob('trained_models/dpo_*.pt'))
if not ckpts:
    print("WARNING: No checkpoints found. Using untrained model for Phase 5.5")
    model_state = None
else:
    ckpt = torch.load(ckpts[-1], map_location='cpu')
    model_state = ckpt.get('model_state')

# Update model config for Phase 5.5
cfg_p55 = ModelConfig()
cfg_p55.max_seq_len = lc_cfg.max_seq_len   # 16384
cfg_p55.swa_window  = lc_cfg.swa_window    # 4096

model = SLM(cfg_p55, TrainConfig()).to(device)

if model_state is not None:
    # Filter out cached rope parameters in case seq len changed (it did!)
    state_dict = {k: v for k, v in model_state.items() if not k.endswith('_cached')}
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint for Phase 5.5: {ckpts[-1]}")

print(f"Phase 5.5: seq_len {cfg_p55.max_seq_len}, swa_window {cfg_p55.swa_window}")
print(f"Batch tokens: {lc_cfg.physical_batch * lc_cfg.grad_accum * lc_cfg.max_seq_len:,}")
assert lc_cfg.physical_batch * lc_cfg.grad_accum * lc_cfg.max_seq_len == 262_144

# Update FlashAttention window_size in all local blocks
# CRITICAL: must read from config, never hardcode (2048, 0)
for block in model.blocks:
    if not block.is_global:
        block.attention.swa_window = lc_cfg.swa_window  # 4096
        # FlashAttention call in attention.py now uses:
        # window_size=(self.swa_window, 0) which reads 4096

print("Phase 5.5 model ready.")
