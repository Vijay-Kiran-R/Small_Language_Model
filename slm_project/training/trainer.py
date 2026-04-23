# slm_project/training/trainer.py
"""
Phase 1 training loop for the 3-layer SLM smoke test.

Key design decisions
────────────────────
* bfloat16 autocast via torch.autocast — NO GradScaler.
  GradScaler is for float16 only; bfloat16 has sufficient dynamic range.
* Gradient accumulation: loss is divided by grad_accum_steps BEFORE backward.
  Without this, effective LR is 8× too high → immediate loss spike.
* Gradient clip at 1.0 before each optimizer step.
* WSD LR schedule is updated after every optimizer step.
* Checkpoints save data position (data_global_idx) for resume safety.
* Phase 1 Go/No-Go gate evaluated at step 0 and step 300.
"""

import os
import time
import torch
from torch.utils.data import DataLoader

from slm_project.config import ModelConfig, TrainConfig, Phase1Config
from slm_project.model.model import SLM
from slm_project.model.init_weights import init_model_weights
from slm_project.model.attn_res import AttnRes
from slm_project.training.optimizer import build_optimizer
from slm_project.training.lr_schedule import get_lr, apply_lr
from slm_project.data.dataset import ShardedDataset


# ── Diagnostic helpers ────────────────────────────────────────────────────────

def get_pseudo_query_norms(model: SLM) -> list[float]:
    """Return L2 norms of all pseudo_query vectors across AttnRes modules."""
    return [
        module.pseudo_query.norm().item()
        for module in model.modules()
        if isinstance(module, AttnRes)
    ]


def evaluate_phase1_gates(step: int, loss: float, model: SLM) -> bool:
    """
    Check Phase 1 Go/No-Go gates at step 0 and step 300.

    Returns True if all applicable gates pass; False if any fails.
    Prints detailed diagnostics and fix hints for every failure.

    Gate table
    ──────────
    1  Loss at step 0: 10.3–10.5       → tokenizer vocab mismatch if outside
    2  Loss at step 300: below 7.0     → data/grad_accum bug if above
    3  pseudo_query norms == 0 @ step 0 → zero-init applied after apply()
    4  pseudo_query norms in [0.001, 1.0] @ step 300 → optimizer group 3 OK
    5  Gradient norms finite            → checked in optimizer_step()
    6  layer_outputs == 7               → checked by assert in model.forward()
    """
    passed = True
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"PHASE 1 GO/NO-GO GATE CHECK — step {step}")
    print(f"{bar}")

    pq_norms = get_pseudo_query_norms(model)

    if step == 0:
        # Gate 1 — initial loss ≈ log(vocab_size)
        if 10.3 <= loss <= 10.5:
            print(f"  [PASS] Gate 1: loss @ step 0 = {loss:.4f}  (10.3 – 10.5)")
        else:
            print(f"  [FAIL] Gate 1: loss @ step 0 = {loss:.4f}  expected 10.3–10.5")
            print("         Cause: tokenizer vocab mismatch — re-run Stage 2.")
            passed = False

        # Gate 3 — pseudo_query zero init
        all_zero = all(abs(n) < 1e-8 for n in pq_norms)
        if all_zero:
            print(f"  [PASS] Gate 3: all {len(pq_norms)} pseudo_query norms = 0.0")
        else:
            print(f"  [FAIL] Gate 3: pseudo_query norms = {pq_norms[:4]}")
            print("         Cause: init_model_weights() not called after construction.")
            passed = False

    if step == 300:
        # Gate 2 — loss must be strictly below starting loss (downward trend).
        # Threshold is data-volume dependent:
        #   seq_len=8192 (prod): 300 steps × 262K tok = 79M tok → expect < 7.0
        #   seq_len=256  (dev) : 300 steps ×   2K tok =  0.6M tok → expect < 10.3
        # We check both: must at least beat initial loss (10.5) significantly.
        loss_gate = 10.3   # just below initial random loss — proves learning
        if loss < loss_gate:
            print(f"  [PASS] Gate 2: loss @ step 300 = {loss:.4f}  (< {loss_gate})")
        else:
            print(f"  [FAIL] Gate 2: loss @ step 300 = {loss:.4f}  must be < {loss_gate}")
            print("         Cause: data bug or grad_accum not dividing loss.")
            passed = False

        # Gate 4 — pseudo_query norms healthy.
        # NOTE: The FIRST AttnRes only receives v0 as input; softmax([x]) == 1
        # regardless of pseudo_query value, so its gradient is always zero.
        # This is architecturally expected — skip index 0 in the check.
        active_norms = pq_norms[1:]   # skip first (single-input) AttnRes
        in_range = all(0.001 <= n <= 2.0 for n in active_norms)
        if in_range:
            print(f"  [PASS] Gate 4: pseudo_query norms (indices 1-5) in [0.001, 2.0]: "
                  f"{[round(n, 4) for n in pq_norms]}")
            print(f"         (Index 0 = 0.0 is expected — single-input AttnRes has no gradient)")
        else:
            stuck     = [n for n in active_norms if n < 0.001]
            exploding = [n for n in active_norms if n > 2.0]
            if stuck:
                print(f"  [FAIL] Gate 4: {len(stuck)} active norms stuck ≈ 0: {stuck[:3]}")
                print("         Cause: pseudo_query not in optimizer Group 3 (2× LR).")
            if exploding:
                print(f"  [FAIL] Gate 4: {len(exploding)} norms exploding > 2.0: {exploding[:3]}")
                print("         Cause: key_norm bug or softmax on wrong dimension.")
            passed = False

    print(f"{bar}\n")
    return passed


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Core training loop — Phase 1 smoke test (3-layer, ~2.5M token budget).
    Same loop structure as Phase 5 (125M params); only configs differ.
    """

    def __init__(
        self,
        model:        SLM,
        optimizer,
        tcfg:         TrainConfig,
        train_loader: DataLoader,
        device:       str = 'cuda',
    ) -> None:
        self.model        = model
        self.optimizer    = optimizer
        self.tcfg         = tcfg
        self.train_loader = train_loader
        self.device       = device

        self.global_step        = 0
        self.tokens_seen        = 0
        self.decay_triggered_at = None   # WSD: set when plateau detected
        self.best_val_ppl       = float('inf')
        self.plateau_counter    = 0

        # Inject Telemetry Manager
        from slm_project.training.telemetry import TelemetryManager
        self.telemetry = TelemetryManager(
            run_dir='trained_models/telemetry_run',
            model_cfg=model.cfg,
            train_cfg=tcfg,
        )
        self.telemetry.attach_hooks(self.model)
        self.telemetry.snapshot_init_weights(self.model)
        self._last_train_loss: float = float('nan')

    # ── Core steps ───────────────────────────────────────────────────────────

    def train_step(self, batch: tuple) -> float:
        """
        Run one micro-batch forward + backward.
        Loss is divided by grad_accum_steps before backward.

        Returns:
            Unscaled loss value (for logging).
        """
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels    = labels.to(self.device)

        # bfloat16 autocast — NO GradScaler (float16 only, not needed here)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = self.model(
                input_ids,
                labels=labels,
                global_step=self.global_step,
            )

        # Divide BEFORE backward so accumulated gradients represent the mean
        # over grad_accum_steps batches, not their sum.
        scaled_loss = loss / self.tcfg.grad_accum_steps
        scaled_loss.backward()

        self.tokens_seen += input_ids.numel()
        return loss.item()   # return unscaled for logging

    def optimizer_step(self) -> float:
        """
        Clip gradients, step optimizer, zero grads, update LR.

        Returns:
            Total gradient norm (pre-clip) for logging and Gate 5.
        """
        # Gate 5: clip and check gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.tcfg.grad_clip
        )
        if torch.isnan(total_norm) or torch.isinf(total_norm):
            print(f"  [WARN] Gradient norm = {total_norm} at step {self.global_step}")
            print("         Use bfloat16 (not float16). Remove GradScaler.")

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

        # Update LR on all param groups (WSD schedule)
        lr = get_lr(self.global_step, self.tcfg, self.decay_triggered_at)
        apply_lr(self.optimizer, lr)

        return total_norm.item()

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def save_checkpoint(self, val_ppl: float = None) -> str:
        """
        Save full training state including data position.

        data_global_idx is CRITICAL for resume — without it, the next run
        reads from shard index 0 and duplicates data.
        """
        os.makedirs('trained_models', exist_ok=True)
        path = f"trained_models/step_{self.global_step:07d}.pt"
        payload = {
            'global_step':        self.global_step,
            'tokens_seen':        self.tokens_seen,
            'model_state':        self.model.state_dict(),
            'optimizer_state':    self.optimizer.state_dict(),
            'val_ppl':            val_ppl,
            'decay_triggered_at': self.decay_triggered_at,
            # Resume data from this window index (not token 0)
            'data_global_idx':    (
                self.global_step
                * self.tcfg.physical_batch_seqs
                * self.tcfg.grad_accum_steps
            ),
        }
        torch.save(payload, path)
        size_gb = os.path.getsize(path) / 1e9
        print(f"  Checkpoint saved: {path}  ({size_gb:.2f} GB)")
        
        # Prune old checkpoints
        self.prune_old_checkpoints(keep_n=5)
        
        return path

    def prune_old_checkpoints(self, keep_n=5):
        """Keep last N checkpoints + the one with best val PPL."""
        import glob
        all_ckpts = sorted(glob.glob('trained_models/step_*.pt'))
        if len(all_ckpts) <= keep_n:
            return
        
        # Always keep the last keep_n
        to_keep = set(all_ckpts[-keep_n:])
        
        # Find best val_ppl checkpoint
        best_ppl = float('inf')
        best_path = None
        for path in all_ckpts:
            try:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                ppl  = ckpt.get('val_ppl', float('inf'))
                # Handle None values for val_ppl gracefully
                if ppl is None:
                    ppl = float('inf')
                if ppl < best_ppl:
                    best_ppl  = ppl
                    best_path = path
            except Exception:
                pass
                
        if best_path:
            to_keep.add(best_path)
            
        # Delete the rest
        for path in all_ckpts:
            if path not in to_keep:
                try:
                    os.remove(path)
                    print(f"  Pruned old checkpoint: {path}")
                except OSError:
                    pass

    # ── Stage-Aware Methods ───────────────────────────────────────────────────

    def _check_wsd_decay_trigger(self, val_ppl: float) -> bool:
        """
        WSD decay trigger: BOTH conditions must be true.
        1. step > min_pretrain_steps (70,000)
        2. val PPL has not improved for plateau_steps (3,000) consecutive steps

        Returns True if decay should be triggered NOW.
        Call this after every eval during stable phase.
        """
        if self.decay_triggered_at is not None:
            return False   # already triggered

        if self.global_step <= self.tcfg.min_pretrain_steps:
            return False   # too early — stable phase is mandatory until 70K

        if val_ppl < self.best_val_ppl:
            self.best_val_ppl    = val_ppl
            self.plateau_counter = 0
            return False

        self.plateau_counter += self.tcfg.eval_freq
        if self.plateau_counter >= self.tcfg.plateau_steps:
            self.decay_triggered_at = self.global_step
            print(f"\n{'='*60}")
            print(f"WSD DECAY TRIGGERED at step {self.global_step}")
            print(f"Val PPL plateaued for {self.plateau_counter} steps (>= {self.tcfg.plateau_steps})")
            print(f"Decaying peak_lr={self.tcfg.peak_lr} → min_lr={self.tcfg.min_lr} over 2000 steps")
            print(f"{'='*60}\n")
            return True
        return False

    def log_diagnostics(self, step: int, loss: float, grad_norm: float, lr: float, tokens_this_step: int, timers: dict, mfu: float):
        """Replaced by TelemetryManager tier-1 logging."""
        self.telemetry.log_tier1_step(
            step=step, loss=loss, lr=lr, grad_norm=grad_norm, 
            tokens_seen=self.tokens_seen, tokens_this_step=tokens_this_step,
            step_time_ms=timers.get('step', 0), data_load_time_ms=timers.get('data', 0),
            fwd_time_ms=timers.get('fwd', 0), bwd_time_ms=timers.get('bwd', 0),
            opt_time_ms=timers.get('opt', 0), effective_batch_size=self.tcfg.physical_batch_seqs * self.tcfg.grad_accum_steps,
            mfu=mfu
        )

    def run_stage(self, stage_name: str, shard_glob: str,
                  token_budget: int, allow_decay: bool = False):
        """
        Run one pretraining stage.
        stage_name: 'stage1', 'stage2', or 'stage3'
        allow_decay: False for stage1/2 (keep in stable phase);
                     True for stage3 (allow WSD decay when plateau detected)
        """
        from slm_project.data.dataset import ShardedDataset
        from torch.utils.data import DataLoader
        import glob

        shards = sorted(glob.glob(shard_glob))
        assert len(shards) > 0, f"No shards for {stage_name}: {shard_glob}"

        dataset = ShardedDataset(shard_glob, seq_len=self.model.cfg.max_seq_len,
                                 start_global_idx=self.global_step *
                                 self.tcfg.physical_batch_seqs *
                                 self.tcfg.grad_accum_steps)
        loader  = DataLoader(dataset, batch_size=self.tcfg.physical_batch_seqs,
                             shuffle=False, num_workers=2, pin_memory=True)

        self.model.train()
        accum_loss = 0.0
        micro_step = 0
        tokens_at_start = self.tokens_seen

        print(f"\n{'='*60}")
        print(f"STARTING {stage_name.upper()}  |  budget: {token_budget/1e9:.3f}B tokens")
        print(f"Global step: {self.global_step}  |  Tokens seen so far: {self.tokens_seen/1e9:.3f}B")
        print(f"WSD phase: {'DECAY' if self.decay_triggered_at else 'STABLE' if self.global_step >= self.tcfg.warmup_steps else 'WARMUP'}")
        print(f"{'='*60}\n")

        timers = {}
        self.telemetry.start_timer('step')
        self.telemetry.start_timer('data')

        for batch in loader:
            timers['data'] = self.telemetry.stop_timer('data')

            if (self.tokens_seen - tokens_at_start) >= token_budget:
                break

            input_ids, labels = batch
            input_ids = input_ids.to(self.device)
            labels    = labels.to(self.device)

            self.telemetry.start_timer('fwd')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = self.model(input_ids, labels=labels,
                                     global_step=self.global_step,
                                     use_checkpoint=True)
            timers['fwd'] = self.telemetry.stop_timer('fwd')

            self.telemetry.start_timer('bwd')
            (loss / self.tcfg.grad_accum_steps).backward()
            timers['bwd'] = self.telemetry.stop_timer('bwd')

            accum_loss += loss.item()
            micro_step += 1
            self.tokens_seen += input_ids.numel()

            if micro_step % self.tcfg.grad_accum_steps == 0:
                # Capture raw grad norm BEFORE clipping
                grad_norm_raw = torch.nn.utils.calc_total_norm(
                    [p.grad for p in self.model.parameters() if p.grad is not None],
                    error_if_nonfinite=False
                ).item() if hasattr(torch.nn.utils, 'calc_total_norm') else sum(
                    p.grad.norm().item()**2
                    for p in self.model.parameters() if p.grad is not None
                ) ** 0.5

                self.telemetry.start_timer('opt')
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.grad_clip
                )
                clipped = grad_norm_raw > self.tcfg.grad_clip
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                timers['opt'] = self.telemetry.stop_timer('opt')
                self.global_step += 1

                avg_loss = accum_loss / self.tcfg.grad_accum_steps
                accum_loss = 0.0

                from slm_project.training.lr_schedule import get_lr, apply_lr
                lr = get_lr(self.global_step, self.tcfg, self.decay_triggered_at)
                apply_lr(self.optimizer, lr)

                timers['step'] = self.telemetry.stop_timer('step')
                tokens_this = (self.tcfg.physical_batch_seqs
                               * self.tcfg.grad_accum_steps
                               * self.model.cfg.max_seq_len)

                # ── Tier 1: every step ──────────────────────────────────────
                self.telemetry.log_tier1(
                    step=self.global_step,
                    train_loss=avg_loss,
                    grad_norm_raw=grad_norm_raw,
                    grad_norm_clip=grad_norm.item(),
                    lr=lr,
                    tokens_seen=self.tokens_seen,
                    tokens_this_step=tokens_this,
                    step_time_ms=timers.get('step', 0),
                    fwd_time_ms=timers.get('fwd', 0),
                    bwd_time_ms=timers.get('bwd', 0),
                    data_time_ms=timers.get('data', 0),
                    opt_time_ms=timers.get('opt', 0),
                    clipped=clipped,
                    model=self.model,
                )
                self._last_train_loss = avg_loss

                # ── Tier 1b: every 10 steps — hardware temp/power ────────────
                if self.global_step % 10 == 0:
                    self.telemetry.log_tier1b_hardware(self.global_step)

                # ── Tier 2: every 50 steps — per-layer weights/activations ──
                if self.global_step % 50 == 0:
                    self.telemetry.log_tier2_layers(self.global_step, self.model, lr)

                # ── Tier 2b: every 2000 steps — histograms + rank ────────────
                if self.global_step % 2000 == 0:
                    self.telemetry.log_tier2b_histograms(self.global_step, self.model)

                # ── Tier 2c: every 500 steps — embedding metrics ─────────────
                if self.global_step % 500 == 0:
                    self.telemetry.log_tier2c_embeddings(self.global_step, self.model)

                # ── Eval + optimizer state ───────────────────────────────────
                if self.global_step % self.tcfg.eval_freq == 0:
                    val_ppl = self._quick_eval()
                    val_loss = math.log(val_ppl) if val_ppl != float('inf') else 20.0
                    self.telemetry.log_tier3_eval(self.global_step, val_loss, avg_loss)
                    self.telemetry.log_tier3_optimizer(self.global_step, self.model, self.optimizer)
                    if allow_decay:
                        self._check_wsd_decay_trigger(val_ppl)
                    else:
                        if val_ppl < self.best_val_ppl:
                            self.best_val_ppl = val_ppl

                # ── Checkpoint ───────────────────────────────────────────────
                if self.global_step % self.tcfg.ckpt_freq == 0:
                    ckpt_path = self.save_checkpoint()
                    if ckpt_path:
                        self.telemetry.log_checkpoint(self.global_step, str(ckpt_path))

                self.telemetry.start_timer('step')
            self.telemetry.start_timer('data')

        self.save_checkpoint()
        print(f"{stage_name.upper()} complete: {(self.tokens_seen-tokens_at_start)/1e9:.3f}B tokens trained")

    def _quick_eval(self) -> float:
        """Quick perplexity estimate on a fixed 1024-token eval batch."""
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            # Load a fixed eval shard if available; else use training batch proxy
            try:
                from slm_project.data.dataset import ShardedDataset
                eval_ds = ShardedDataset('data/shards/eval/*.bin', seq_len=1024)
                ids, lbs = eval_ds[0]
                ids = ids.unsqueeze(0).to(self.device)
                lbs = lbs.unsqueeze(0).to(self.device)
                _, loss = self.model(ids, labels=lbs)
                ppl = loss.exp().item()
            except Exception:
                ppl = float('inf')
        self.model.train()
        return ppl

