import torch
import os, glob
from slm_project.config import ModelConfig, TrainConfig
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights
from slm_project.training.optimizer import build_optimizer
from slm_project.training.trainer import Trainer

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg    = ModelConfig()   # n_layers=16
    tcfg   = TrainConfig()

    # Build model (or load checkpoint if resuming)
    latest_ckpt = sorted(glob.glob('trained_models/step_*.pt'))
    if latest_ckpt:
        print(f"Resuming from: {latest_ckpt[-1]}")
        ckpt  = torch.load(latest_ckpt[-1], map_location='cpu')
        model = SLM(cfg, tcfg).to(device)
        model.load_state_dict(ckpt['model_state'])
        optimizer = build_optimizer(model, tcfg)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        trainer = Trainer(model, optimizer, tcfg, None, device)
        trainer.global_step         = ckpt['global_step']
        trainer.tokens_seen         = ckpt['tokens_seen']
        trainer.decay_triggered_at  = ckpt.get('decay_triggered_at')
        trainer.best_val_ppl        = ckpt.get('val_ppl', float('inf'))
    else:
        print("Starting fresh training ...")
        model = SLM(cfg, tcfg).to(device)
        init_model_weights(model)   # zeros pseudo_queries LAST — non-negotiable
        n_params = model.get_num_params()
        assert n_params == 125_931_008, f"Wrong param count: {n_params:,}"
        print(f"Model: {n_params:,} parameters (125.9M + IHA) [OK]")
        optimizer = build_optimizer(model, tcfg)
        trainer = Trainer(model, optimizer, tcfg, None, device)

    # -------------------------------------------------------------
    # NEW CODE (FULL SCALE) - COMMENTED OUT FOR TESTING
    # -------------------------------------------------------------
    # trainer.run_stage(
    #     stage_name  = 'stage1',
    #     shard_glob  = 'data/shards/stage1/**/*.bin',
    #     token_budget = 8_000_000_000,
    #     allow_decay  = False   # Stage 1: never trigger WSD decay
    # )

    # -------------------------------------------------------------
    # OLD CODE (SMALL/DEV SCALE) - UNCOMMENTED FOR TESTING
    # -------------------------------------------------------------
    print("Running with existing data for testing as requested...")
    trainer.run_stage(
        stage_name   = 'phase1_smoke_stage1',
        shard_glob   = 'data/shards/*_shard*.bin',
        token_budget = 50_000, # Run just a tiny bit for test
        allow_decay  = False
    )
    print("Stage 1 complete. Proceed to Stage 2 while Stage 2 data downloads.")

if __name__ == '__main__':
    main()
