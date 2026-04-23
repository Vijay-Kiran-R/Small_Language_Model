"""
slm_project/training/telemetry.py
Production-grade Telemetry Engine — covers all 15 metric categories.
Built for 12 GB VRAM, single-GPU, bfloat16 training.

Tiered schedule:
  Tier 1  every step          → loss, grad_norm, lr, tokens, hardware flags
  Tier 2  every 10 steps      → gpu temp, power, memory fragmentation
  Tier 3  every 50-100 steps  → per-layer weights, grads, residuals, FFN, norms
  Tier 4  every 500-1000 steps→ eval, optimizer state, embedding metrics, generation
  Tier 5  every 1000 steps    → MTP metrics, grad flow analysis, checkpoint
  Tier 6  every 2000-5000 steps→ weight histograms, attention pattern viz
  Tier 7  every 5000 steps    → head specialisation, neuron-level FFN stats
  Tier 8  end of run          → full downstream eval, calibration, timeline
"""

from __future__ import annotations

import os
import json
import math
import time
import subprocess
import collections
import sys
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import numpy as np

# ── Optional backends ──────────────────────────────────────────────────────────
try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_OK = True
except ImportError:
    _TB_OK = False
    SummaryWriter = None  # type: ignore

try:
    import wandb as _wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
    _wandb = None  # type: ignore


# ==============================================================================
# 1.  LOGGER INTERFACE  (Cat. 22 — swappable backends)
# ==============================================================================

class BaseLogger:
    """All loggers implement this interface."""
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None: pass
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None: pass
    def log_text(self, tag: str, text: str, step: int) -> None: pass
    def flush(self) -> None: pass
    def close(self) -> None: pass


class JSONLLogger(BaseLogger):
    """Flat-file safety net — always write here regardless of other backends."""
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._path = path

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        entry = {"step": step, **{k: v for k, v in metrics.items()
                                   if isinstance(v, (int, float, bool, str))}}
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")


class TensorBoardLogger(BaseLogger):
    def __init__(self, log_dir: str):
        self._writer = SummaryWriter(log_dir=log_dir) if _TB_OK else None
        if not self._writer:
            print("[TelemetryWARN] tensorboard not installed — TB logging disabled.")

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        if not self._writer: return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, v, global_step=step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        if self._writer:
            try:
                self._writer.add_histogram(tag, values.cpu().float(), global_step=step)
            except Exception:
                pass

    def flush(self) -> None:
        if self._writer: self._writer.flush()

    def close(self) -> None:
        if self._writer: self._writer.close()


class WandbLogger(BaseLogger):
    def __init__(self, project: str, run_name: str, config: dict):
        self._active = False
        if _WANDB_OK and _wandb.run is None:
            try:
                _wandb.init(project=project, name=run_name, config=config,
                            resume="allow")
                self._active = True
            except Exception as e:
                print(f"[TelemetryWARN] wandb init failed: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        if self._active:
            _wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._active: _wandb.finish()


# ==============================================================================
# 2.  TELEMETRY MANAGER
# ==============================================================================

class TelemetryManager:
    """
    Central orchestrator.  Owns all loggers, hooks, and rolling-window state.
    Construct once in Trainer.__init__ and call the appropriate tier methods
    from the training loop.
    """

    def __init__(
        self,
        run_dir: str,
        model_cfg,          # ModelConfig dataclass
        train_cfg,          # TrainConfig dataclass
        use_wandb: bool = False,
        wandb_project: str = "slm-project",
    ):
        self.run_dir = run_dir
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        os.makedirs(run_dir, exist_ok=True)

        # ── Logger pool ───────────────────────────────────────────────────────
        self._loggers: List[BaseLogger] = [
            JSONLLogger(os.path.join(run_dir, "training_logs.jsonl")),
            TensorBoardLogger(run_dir),
        ]
        if use_wandb:
            self._loggers.append(
                WandbLogger(wandb_project,
                            os.path.basename(run_dir),
                            self._build_cfg_dict())
            )

        # ── Hook storage ──────────────────────────────────────────────────────
        self._fwd_hooks: list = []
        self._bwd_hooks: list = []
        # Sampled activation stats (filled by hooks, consumed by log_tier3)
        self._act_cache: Dict[str, Dict[str, float]] = {}
        # Residual stream snapshots (pre/post attn, pre/post FFN)
        self._res_cache: Dict[str, float] = {}

        # ── Rolling windows ───────────────────────────────────────────────────
        self._loss_window: collections.deque = collections.deque(maxlen=100)
        self._grad_window: collections.deque = collections.deque(maxlen=100)
        self._spike_steps: List[int] = []
        self._spike_count: int = 0
        self._last_loss: float = float("nan")

        # ── Weight snapshot for delta tracking ───────────────────────────────
        self._prev_weights: Dict[int, torch.Tensor] = {}   # id(param) → clone
        self._init_weights: Dict[int, torch.Tensor] = {}   # id(param) → clone @ step 0
        self._prev_grads:   Dict[int, torch.Tensor] = {}   # id(param) → grad clone

        # ── Timers ────────────────────────────────────────────────────────────
        self._timers: Dict[str, float] = {}
        self._step_tokens_seen: int = 0   # tokens at start of current step

        # ── Grad clipping tracker ────────────────────────────────────────────
        self._total_steps: int = 0
        self._clipped_steps: int = 0

        # ── Layer convergence tracker ─────────────────────────────────────────
        self._layer_converge_step: Dict[str, Optional[int]] = {}

        # Dump metadata once
        self._log_run_metadata()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _dispatch(self, metrics: Dict[str, Any], step: int) -> None:
        for lg in self._loggers:
            lg.log_metrics(metrics, step)

    def _dispatch_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        for lg in self._loggers:
            lg.log_histogram(tag, values, step)

    def _build_cfg_dict(self) -> dict:
        out = {}
        for cfg in (self.model_cfg, self.train_cfg):
            try:
                import dataclasses
                out.update(dataclasses.asdict(cfg))
            except Exception:
                out.update(vars(cfg))
        return out

    def _git_hash(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode("ascii").strip()
        except Exception:
            return "unknown"

    def _log_run_metadata(self) -> None:
        """Cat. 21: System & Run Metadata — logged once at run start."""
        import platform
        cfg = self._build_cfg_dict()

        gpu_name, gpu_count, vram_gb = "unknown", 0, 0
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name  = torch.cuda.get_device_name(0)
            vram_gb   = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        meta = {
            # Run identity
            "run_id":            os.path.basename(self.run_dir),
            "timestamp_start":   time.time(),
            "git_commit_hash":   self._git_hash(),
            # Environment
            "python_version":    sys.version,
            "torch_version":     torch.__version__,
            "cuda_version":      torch.version.cuda or "cpu",
            "hostname":          platform.node(),
            # Hardware
            "gpu_name":          gpu_name,
            "gpu_count":         gpu_count,
            "gpu_vram_gb":       round(vram_gb, 2),
            # Architecture (Cat. 20)
            "num_layers":        self.model_cfg.n_layers,
            "d_model":           self.model_cfg.d_model,
            "n_heads_q":         self.model_cfg.n_heads_q,
            "n_heads_kv":        self.model_cfg.n_heads_kv,
            "d_head":            self.model_cfg.d_head,
            "ffn_hidden":        self.model_cfg.ffn_hidden,
            "vocab_size":        self.model_cfg.vocab_size,
            "max_seq_len":       self.model_cfg.max_seq_len,
            "rope_base":         self.model_cfg.rope_base,
            "global_layers":     list(self.model_cfg.global_layers),
            "weight_tying":      self.model_cfg.weight_tying,
            "activation_fn":     "SwiGLU",
            "norm_type":         "RMSNorm",
            "positional_enc":    "RoPE",
            "attention_type":    "FlashAttention-SWA+NoPE",
            "dtype":             self.train_cfg.precision,
            # Training
            "peak_lr":           self.train_cfg.peak_lr,
            "min_lr":            self.train_cfg.min_lr,
            "warmup_steps":      self.train_cfg.warmup_steps,
            "physical_batch":    self.train_cfg.physical_batch_seqs,
            "grad_accum_steps":  self.train_cfg.grad_accum_steps,
            "grad_clip":         self.train_cfg.grad_clip,
            "effective_batch_tokens": (
                self.train_cfg.physical_batch_seqs
                * self.train_cfg.grad_accum_steps
                * self.model_cfg.max_seq_len
            ),
            # Full config snapshot
            "full_config":       cfg,
        }

        path = os.path.join(self.run_dir, "run_metadata.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"[Telemetry] Run metadata written → {path}")
        print(f"[Telemetry] git commit: {meta['git_commit_hash']}")

    def snapshot_init_weights(self, model: nn.Module) -> None:
        """Call once after init_model_weights() to record starting point."""
        for param in model.parameters():
            if param.requires_grad:
                self._init_weights[id(param)] = param.data.detach().cpu().clone()

    # ──────────────────────────────────────────────────────────────────────────
    # Timer helpers
    # ──────────────────────────────────────────────────────────────────────────

    def start_timer(self, name: str) -> None:
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Returns elapsed milliseconds."""
        if name in self._timers:
            ms = (time.perf_counter() - self._timers.pop(name)) * 1000.0
            return ms
        return 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Hook management
    # ──────────────────────────────────────────────────────────────────────────

    def attach_hooks(self, model: nn.Module) -> None:
        """
        Attach forward hooks to capture activation, residual, attention stats.
        Hooks store results in self._act_cache / self._res_cache.
        Called once from Trainer.__init__.
        """
        self.remove_hooks()

        for full_name, module in model.named_modules():
            layer_tag = full_name if full_name else "root"

            # ── Activation + inf detection for FFN / attention ─────────────────
            if any(k in full_name for k in ("ffn", "attention", "attn", "mlp")):
                def _act_hook(mod, inp, out, tag=layer_tag):
                    try:
                        t = out[0] if isinstance(out, tuple) else out
                        if not isinstance(t, torch.Tensor): return
                        with torch.no_grad():
                            self._act_cache[tag] = {
                                "mean":     t.mean().item(),
                                "std":      t.std().item(),
                                "max":      t.abs().max().item(),
                                "sparsity": (t.abs() < 1e-3).float().mean().item(),
                                "inf":      int(t.isinf().any().item()),
                                "nan":      int(t.isnan().any().item()),
                            }
                    except Exception:
                        pass
                h = module.register_forward_hook(_act_hook)
                self._fwd_hooks.append(h)

            # ── Attention output entropy proxy (reshape by heads) ──────────────
            # We compute output variance per head as a proxy for attention entropy.
            # High variance per head → head is attending selectively (good).
            # Near-zero variance → head has collapsed to uniform attention (bad).
            if "attention" in full_name or "attn" in full_name:
                def _attn_hook(mod, inp, out, tag=layer_tag,
                               n_heads=model.cfg.n_heads_q,
                               d_head=model.cfg.d_head):
                    try:
                        t = out[0] if isinstance(out, tuple) else out
                        if not isinstance(t, torch.Tensor) or t.dim() < 3:
                            return
                        # t: (B, T, n_heads * d_head) → (B, n_heads, T, d_head)
                        B, T, D = t.shape
                        if D != n_heads * d_head:
                            return
                        with torch.no_grad():
                            heads = t.reshape(B, T, n_heads, d_head).permute(0, 2, 1, 3)  # (B,H,T,d)
                            # Variance across T per head → proxy for selectivity
                            head_var = heads.var(dim=2).mean(dim=(0, 3))  # (n_heads,)
                            # Treat var as proxy for entropy: low var = collapsed
                            head_entropy_proxy = head_var.tolist()
                            self._act_cache[tag + "__attn_head_var"] = {
                                "mean":     float(head_var.mean().item()),
                                "std":      float(head_var.std().item()),
                                "max":      float(head_var.max().item()),
                                "sparsity": float((head_var < 1e-4).float().mean().item()),  # dead head fraction
                                "inf": 0, "nan": 0,
                                "per_head": head_entropy_proxy,
                            }
                    except Exception:
                        pass
                h2 = module.register_forward_hook(_attn_hook)
                self._fwd_hooks.append(h2)

    def remove_hooks(self) -> None:
        for h in self._fwd_hooks + self._bwd_hooks:
            h.remove()
        self._fwd_hooks.clear()
        self._bwd_hooks.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # TIER 1 — every optimizer step
    # ──────────────────────────────────────────────────────────────────────────

    def log_tier1(
        self,
        step: int,
        train_loss: float,
        grad_norm_raw: float,   # before clipping
        grad_norm_clip: float,  # after clipping
        lr: float,
        tokens_seen: int,
        tokens_this_step: int,
        step_time_ms: float,
        fwd_time_ms: float,
        bwd_time_ms: float,
        data_time_ms: float,
        opt_time_ms: float,
        clipped: bool,
        model: nn.Module,
    ) -> None:
        """Every-step metrics: loss, grads, optimizer, hardware flags."""

        self._total_steps += 1
        if clipped:
            self._clipped_steps += 1

        # Rolling windows
        self._loss_window.append(train_loss)
        self._grad_window.append(grad_norm_clip)
        loss_mean = float(np.mean(self._loss_window))
        loss_std  = float(np.std(self._loss_window))  if len(self._loss_window) > 1 else 0.0
        grad_mean = float(np.mean(self._grad_window))

        # Spike detection (>20% above rolling mean)
        spike = (not math.isnan(self._last_loss) and
                 train_loss > 1.20 * loss_mean and
                 len(self._loss_window) >= 10)
        if spike:
            self._spike_count += 1
            self._spike_steps.append(step)
        self._last_loss = train_loss

        # Weight norm global (for update ratio)
        w_norm = sum(p.norm().item() for p in model.parameters() if p.requires_grad)
        w_upd  = (lr * grad_norm_clip / w_norm) if w_norm > 1e-8 else 0.0

        # Warmup fraction
        warmup_frac = min(1.0, step / max(1, self.train_cfg.warmup_steps))

        # Adam effective step (global approx)
        adam_step = 0.0
        for g in model.parameters():
            if g.requires_grad and g.grad is not None:
                adam_step = lr / (g.grad.abs().mean().item() + 1e-8)
                break

        # Throughput
        tps = (tokens_this_step / (step_time_ms / 1000.0)) if step_time_ms > 0 else 0.0

        # NaN / Inf checks
        nan_loss = math.isnan(train_loss)
        inf_loss = math.isinf(train_loss)
        nan_grad = math.isnan(grad_norm_raw)

        # Stability flags
        vanish    = grad_norm_clip < 1e-7 and grad_norm_clip > 0
        explosion = grad_norm_raw > self.train_cfg.grad_clip * 3

        # GPU memory
        gpu_mem_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        gpu_res_mb = torch.cuda.memory_reserved()  / 1024**2 if torch.cuda.is_available() else 0.0

        m: Dict[str, Any] = {
            # Loss
            "loss/train_loss":         train_loss,
            "loss/perplexity_train":   math.exp(min(train_loss, 20)),
            "loss/loss_per_token":     train_loss,   # same since CE is per-token
            "loss/rolling_mean_100":   loss_mean,
            "loss/rolling_std_100":    loss_std,
            "loss/spike_flag":         int(spike),
            "loss/spike_count_total":  self._spike_count,
            "loss/spike_step":         step if spike else -1,
            # Gradients
            "grad/norm_before_clip":   grad_norm_raw,
            "grad/norm_after_clip":    grad_norm_clip,
            "grad/clipped_fraction":   self._clipped_steps / max(1, self._total_steps),
            "grad/max_abs_global":     max((p.grad.abs().max().item()
                                           for p in model.parameters()
                                           if p.grad is not None), default=0.0),
            "grad/rolling_mean_100":   grad_mean,
            "grad/vanish_flag":        int(vanish),
            "grad/explosion_flag":     int(explosion),
            "grad/nan_flag":           int(nan_grad),
            # Loss flags
            "flags/nan_in_loss":       int(nan_loss),
            "flags/inf_in_loss":       int(inf_loss),
            # Optimizer / LR
            "optim/learning_rate":     lr,
            "optim/weight_update_ratio_global": w_upd,
            "optim/warmup_fraction":   warmup_frac,
            "optim/adam_step_size_global": adam_step,
            # Data
            "data/tokens_seen_b":      tokens_seen / 1e9,
            "data/tokens_per_second":  tps,
            # Perf
            "perf/step_time_ms":       step_time_ms,
            "perf/forward_time_ms":    fwd_time_ms,
            "perf/backward_time_ms":   bwd_time_ms,
            "perf/data_load_time_ms":  data_time_ms,
            "perf/optimizer_time_ms":  opt_time_ms,
            "perf/data_stall_flag":    int(data_time_ms > 0.5 * step_time_ms),
            # Hardware
            "hw/gpu_memory_allocated_mb": gpu_mem_mb,
            "hw/gpu_memory_reserved_mb":  gpu_res_mb,
        }

        # GPU utilization (pynvml)
        if _NVML_OK:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                m["hw/gpu_utilization_percent"] = util.gpu
            except Exception:
                pass

        self._dispatch(m, step)

        # Print concise summary
        print(f"step={step:6d} | loss={train_loss:.4f} | ppl={math.exp(min(train_loss,20)):.2f} "
              f"| lr={lr:.2e} | gnorm={grad_norm_clip:.3f} "
              f"| tok={tokens_seen/1e9:.3f}B | {tps:.0f}tok/s"
              + (" [SPIKE]" if spike else "")
              + (" [NaN]"   if nan_loss else "")
              + (" [EXPL]"  if explosion else ""))

    # ──────────────────────────────────────────────────────────────────────────
    # TIER 1b — every 10 steps: GPU temperature / power
    # ──────────────────────────────────────────────────────────────────────────

    def log_tier1b_hardware(self, step: int) -> None:
        m: Dict[str, Any] = {}
        if _NVML_OK:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                m["hw/gpu_temperature_c"] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                m["hw/gpu_power_watt"]    = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                m["hw/gpu_memory_used_mb"]  = mem.used  / 1024**2
                m["hw/gpu_memory_total_mb"] = mem.total / 1024**2
            except Exception:
                pass
        if psutil:
            m["hw/cpu_utilization_percent"] = psutil.cpu_percent()
            m["hw/ram_used_gb"] = psutil.virtual_memory().used / 1024**3
        if m:
            self._dispatch(m, step)

    # ──────────────────────────────────────────────────────────────────────────
    # TIER 2 — every 50 steps: per-layer weights, activations, residuals,
    #           attention, dead-layer / dead-head flags
    # ──────────────────────────────────────────────────────────────────────────

    def log_tier2_layers(self, step: int, model: nn.Module, lr: float) -> None:
        m: Dict[str, Any] = {}
        dead_layer_flag  = False
        dead_head_flag   = False
        inf_in_act_flag  = False

        # Global grad norm for flow fraction
        global_g_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ) ** 0.5 + 1e-8

        # ── Per-layer weight + grad metrics ───────────────────────────────────
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            tag = name.replace(".", "_")
            w_norm   = param.data.norm().item()
            w_std    = param.data.std().item()
            w_maxabs = param.data.abs().max().item()
            m[f"weight/{tag}_norm"]    = w_norm
            m[f"weight/{tag}_std"]     = w_std
            m[f"weight/{tag}_max_abs"] = w_maxabs
            # weight_decay contribution = wd × norm (actual L2 penalty magnitude)
            wd_contrib = self.train_cfg.weight_decay * w_norm
            m[f"weight/{tag}_wd_contrib"] = wd_contrib

            if param.grad is not None:
                g_norm = param.grad.norm().item()
                g_max  = param.grad.abs().max().item()
                m[f"grad/{tag}_norm"]    = g_norm
                m[f"grad/{tag}_max_abs"] = g_max

                # grad flow fraction: how much of total gradient energy is in this layer
                m[f"grad/{tag}_flow_fraction"] = g_norm / global_g_norm

                # dead gradient fraction: % of params with |grad| < 1e-7
                dead_frac = (param.grad.abs() < 1e-7).float().mean().item()
                m[f"grad/{tag}_dead_fraction"] = dead_frac

                # weight update ratio per layer
                m[f"weight/{tag}_update_ratio"] = lr * g_norm / (w_norm + 1e-8)

                # grad sign consistency vs previous step
                pid = id(param)
                if pid in self._prev_grads:
                    sc = (torch.sign(param.grad) == torch.sign(self._prev_grads[pid])).float().mean().item()
                    m[f"grad/{tag}_sign_consistency"] = sc
                self._prev_grads[pid] = param.grad.detach().cpu().clone()

            # Delta weight vs previous snapshot
            pid = id(param)
            if pid in self._prev_weights:
                delta = (param.data.cpu() - self._prev_weights[pid]).norm().item()
                m[f"weight/{tag}_delta"] = delta
            self._prev_weights[pid] = param.data.detach().cpu().clone()

        # ── Activations + inf/nan + attention entropy proxy from hooks ─────────
        for tag, stats in self._act_cache.items():
            safe = tag.replace(".", "_")
            if "__attn_head_var" in tag:
                # Attention head output variance (entropy proxy)
                safe_base = safe.replace("__attn_head_var", "")
                m[f"attn/{safe_base}_head_var_mean"]        = stats["mean"]
                m[f"attn/{safe_base}_head_var_std"]         = stats["std"]
                m[f"attn/{safe_base}_dead_head_fraction"]   = stats["sparsity"]
                m[f"attn/{safe_base}_max_head_var"]         = stats["max"]
                if stats["sparsity"] > 0.5:   # >50% of heads near-zero variance
                    dead_head_flag = True
            else:
                m[f"act/{safe}_mean"]     = stats["mean"]
                m[f"act/{safe}_std"]      = stats["std"]
                m[f"act/{safe}_max"]      = stats["max"]
                m[f"act/{safe}_sparsity"] = stats["sparsity"]
                if stats["std"] < 1e-6:
                    dead_layer_flag = True
                if stats.get("inf", 0):
                    inf_in_act_flag = True
        self._act_cache.clear()

        # ── Residual norms from cache ──────────────────────────────────────────
        for tag, val in self._res_cache.items():
            m[f"residual/{tag}"] = val
        self._res_cache.clear()

        # ── Stability flags ───────────────────────────────────────────────────
        m["flags/dead_layer_flag"]   = int(dead_layer_flag)
        m["flags/dead_head_flag"]    = int(dead_head_flag)
        m["flags/inf_in_activation"] = int(inf_in_act_flag)

        # ── RMSNorm gamma stats ────────────────────────────────────────────────
        for name, mod in model.named_modules():
            if hasattr(mod, "weight") and "norm" in name.lower():
                g = mod.weight.data
                tag = name.replace(".", "_")
                m[f"norm/{tag}_gamma_mean"] = g.mean().item()
                m[f"norm/{tag}_gamma_std"]  = g.std().item()
                m[f"norm/{tag}_gamma_max"]  = g.max().item()
                m[f"norm/{tag}_explosion"]  = int(g.max().item() > 10.0)

        self._dispatch(m, step)

    # ──────────────────────────────────────────────────────────────────────────
    # TIER 2b — every 100 steps: weight histograms, rank estimates
    # ──────────────────────────────────────────────────────────────────────────

    def log_tier2b_histograms(self, step: int, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._dispatch_histogram(f"hist/weight/{name}", param.data, step)
                # Effective rank: nuclear norm / spectral norm approx
                if param.data.dim() >= 2:
                    try:
                        flat = param.data.reshape(param.data.shape[0], -1).float()
                        s = torch.linalg.svdvals(flat)
                        nuc  = s.sum().item()
                        spec = s[0].item() + 1e-8
                        rank_est = nuc / spec
                        self._dispatch({f"weight/{name.replace('.','_')}_rank_est": rank_est}, step)
                    except Exception:
                        pass

    # ──────────────────────────────────────────────────────────────────────────
    # TIER 2c — embedding-specific metrics
    # ──────────────────────────────────────────────────────────────────────────

    def log_tier2c_embeddings(self, step: int, model: nn.Module) -> None:
        emb = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Embedding):
                emb = mod.weight.data.float()
                break
        if emb is None:
            return

        m: Dict[str, Any] = {}
        m["emb/norm"]  = emb.norm(dim=1).mean().item()
        m["emb/mean"]  = emb.mean().item()
        m["emb/std"]   = emb.std().item()
        m["emb/max_abs"] = emb.abs().max().item()

        # Anisotropy (cosine similarity between random pairs)
        idx = torch.randint(0, emb.shape[0], (256,))
        e   = emb[idx]
        e_n = torch.nn.functional.normalize(e, dim=1)
        cos = (e_n @ e_n.T)
        mask = ~torch.eye(cos.shape[0], dtype=torch.bool)
        m["emb/anisotropy"] = cos[mask].mean().item()
        m["flags/embedding_collapse_flag"] = int(m["emb/anisotropy"] > 0.95)

        # Dead dimensions (near-zero variance across vocab)
        dim_var = emb.var(dim=0)
        m["emb/dead_dimensions"] = int((dim_var < 1e-6).sum().item())

        self._dispatch(m, step)

    # ──────────────────────────────────────────────────────────────────────────
    # TIER 3 — every 500 steps: eval, optimizer state, generation
    # ──────────────────────────────────────────────────────────────────────────

    def log_tier3_eval(self, step: int, val_loss: float, train_loss: float) -> None:
        val_ppl = math.exp(min(val_loss, 20))
        m: Dict[str, Any] = {
            "eval/val_loss":       val_loss,
            "eval/val_perplexity": val_ppl,
            "eval/bpc":            val_loss / math.log(2) / 3.5,  # ~3.5 chars/token
            "eval/bpb":            val_loss / math.log(2),
            "loss/loss_gap":       val_loss - train_loss,
            "eval/loss_plateau_flag": int(val_ppl > (sum(self._loss_window) / max(1, len(self._loss_window)) * 0.999)),
        }
        # Loss trend slope (linear regression over recent window)
        if len(self._loss_window) >= 20:
            x = np.arange(len(self._loss_window), dtype=np.float32)
            y = np.array(self._loss_window, dtype=np.float32)
            slope = float(np.polyfit(x, y, 1)[0])
            m["loss/trend_slope"] = slope
        self._dispatch(m, step)

    def log_tier3_optimizer(self, step: int, model: nn.Module, optimizer) -> None:
        m: Dict[str, Any] = {}
        m1s, m2s = [], []
        opt_bytes = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p, {})
                if "exp_avg" in state:
                    v = state["exp_avg"]
                    m1s.append(v.norm().item())
                    opt_bytes += v.numel() * v.element_size()
                if "exp_avg_sq" in state:
                    v = state["exp_avg_sq"]
                    m2s.append(v.norm().item())
                    opt_bytes += v.numel() * v.element_size()
        if m1s:
            m["optim/adam_m1_global_norm"] = float(np.mean(m1s))
            m["optim/adam_m2_global_norm"] = float(np.mean(m2s))
        m["optim/state_size_mb"] = opt_bytes / 1024**2

        # Bias correction factor
        beta1 = self.train_cfg.adam_beta1
        beta2 = self.train_cfg.adam_beta2
        m["optim/bias_correction_m1"] = 1 - beta1**step
        m["optim/bias_correction_m2"] = 1 - beta2**step
        m["optim/beta1"] = beta1
        m["optim/beta2"] = beta2
        self._dispatch(m, step)

    def log_tier3_generation(self, step: int, tokenizer, model: nn.Module,
                              prompt_ids: torch.Tensor, device: str) -> None:
        """Generate a short sample and compute diversity / repetition metrics."""
        model.eval()
        try:
            with torch.no_grad():
                inp = prompt_ids[:, :32].to(device)
                out = model.generate(inp, max_new_tokens=64) if hasattr(model, "generate") else None
            if out is None:
                return
            toks = out[0, 32:].tolist()
            # Distinct-1 and Distinct-2
            unigrams = set(toks)
            bigrams  = set(zip(toks, toks[1:]))
            d1 = len(unigrams) / max(1, len(toks))
            d2 = len(bigrams)  / max(1, len(toks) - 1)
            # Repetition rate (4-gram)
            ngrams = [tuple(toks[i:i+4]) for i in range(len(toks)-3)]
            rep = 1.0 - len(set(ngrams)) / max(1, len(ngrams))
            m = {
                "gen/distinct1":        d1,
                "gen/distinct2":        d2,
                "gen/repetition_rate":  rep,
                "gen/length":           len(toks),
            }
            self._dispatch(m, step)
        except Exception:
            pass
        finally:
            model.train()

    # ──────────────────────────────────────────────────────────────────────────
    # TIER 4 — val loss (every N steps, called from _quick_eval result)
    # ──────────────────────────────────────────────────────────────────────────

    def log_val_loss(self, step: int, val_loss: float, train_loss: float) -> None:
        self.log_tier3_eval(step, val_loss, train_loss)

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpoint / resume flags
    # ──────────────────────────────────────────────────────────────────────────

    def log_checkpoint(self, step: int, path: str) -> None:
        self._dispatch({"flags/checkpoint_saved": 1, "flags/checkpoint_path": path,
                        "flags/checkpoint_step": step}, step)

    def log_resumed(self, step: int, path: str) -> None:
        self._dispatch({"flags/resumed_from_step": step, "flags/resumed_from_path": path}, step)

    # ──────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.remove_hooks()
        for lg in self._loggers:
            lg.close()
