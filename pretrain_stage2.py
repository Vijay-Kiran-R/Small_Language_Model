import torch, glob
from slm_project.config import ModelConfig, TrainConfig
from slm_project.model.model import SLM
from slm_project.training.optimizer import build_optimizer
from slm_project.training.trainer import Trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg    = ModelConfig()
tcfg   = TrainConfig()

latest_ckpt = sorted(glob.glob('checkpoints/step_*.pt'))
if latest_ckpt:
    print(f"Loading Stage 1 checkpoint: {latest_ckpt[-1]}")
    ckpt  = torch.load(latest_ckpt[-1], map_location='cpu')
    model = SLM(cfg, tcfg).to(device)
    model.load_state_dict(ckpt['model_state'])
    optimizer = build_optimizer(model, tcfg)
    optimizer.load_state_dict(ckpt['optimizer_state'])

    trainer = Trainer(model, optimizer, tcfg, None, device)
    trainer.global_step        = ckpt['global_step']
    trainer.tokens_seen        = ckpt['tokens_seen']
    trainer.decay_triggered_at = ckpt.get('decay_triggered_at')
    trainer.best_val_ppl       = ckpt.get('val_ppl', float('inf'))

    print(f"Resuming at step {trainer.global_step}, "
          f"{trainer.tokens_seen/1e9:.3f}B tokens seen")
else:
    print("WARNING: No checkpoints found. Stage 2 expects to resume from Stage 1.")
    model = SLM(cfg, tcfg).to(device)
    optimizer = build_optimizer(model, tcfg)
    trainer = Trainer(model, optimizer, tcfg, None, device)

# -------------------------------------------------------------
# NEW CODE (FULL SCALE) - COMMENTED OUT FOR TESTING
# -------------------------------------------------------------
# trainer.run_stage(
#     stage_name   = 'stage2',
#     shard_glob   = 'data/shards/stage2/**/*.bin',
#     token_budget  = 8_000_000_000,
#     allow_decay   = False   # Stage 2: still no WSD decay
# )

# -------------------------------------------------------------
# OLD CODE (SMALL/DEV SCALE) - UNCOMMENTED FOR TESTING
# -------------------------------------------------------------
print("Running with existing data for testing as requested...")
trainer.run_stage(
    stage_name   = 'phase1_smoke_stage2',
    shard_glob   = 'data/shards/*_shard*.bin',
    token_budget = 50_000, # Run just a tiny bit for test
    allow_decay  = False
)

print("Stage 2 complete. Proceed to Stage 3.")
