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
    print(f"Loading checkpoint: {latest_ckpt[-1]}")
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

    print(f"Stage 3 start: step {trainer.global_step}, "
          f"tokens {trainer.tokens_seen/1e9:.3f}B")
    print(f"WSD decay allowed from step {tcfg.min_pretrain_steps:,} "
          f"(current: {trainer.global_step:,})")
else:
    print("WARNING: No checkpoints found. Expected checkpoint for Stage 3.")
    model = SLM(cfg, tcfg).to(device)
    optimizer = build_optimizer(model, tcfg)
    trainer = Trainer(model, optimizer, tcfg, None, device)


# -------------------------------------------------------------
# NEW CODE (FULL SCALE) - COMMENTED OUT FOR TESTING
# -------------------------------------------------------------
# trainer.run_stage(
#     stage_name   = 'stage3',
#     shard_glob   = 'data/shards/stage3/**/*.bin',
#     token_budget  = 4_000_000_000,
#     allow_decay   = True    # Stage 3: WSD decay fires on plateau
# )

# -------------------------------------------------------------
# OLD CODE (SMALL/DEV SCALE) - UNCOMMENTED FOR TESTING
# -------------------------------------------------------------
print("Running with existing data for testing as requested...")
trainer.run_stage(
    stage_name   = 'phase1_smoke_stage3',
    shard_glob   = 'data/shards/*_shard*.bin',
    token_budget = 50_000, # Run just a tiny bit for test
    allow_decay  = True
)

print("\nPRETRAINING COMPLETE")
print(f"Final step: {trainer.global_step:,}")
print(f"Total tokens: {trainer.tokens_seen/1e9:.3f}B")
print("If val PPL is still improving at 20B: KEEP TRAINING in stable phase.")
print("Only trigger decay when 3000+ steps with no improvement AND step > 70,000.")
