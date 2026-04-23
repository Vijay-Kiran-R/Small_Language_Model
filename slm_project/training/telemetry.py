"""
slm_project/training/telemetry.py
Comprehensive Telemetry Engine for SLM.
Implements the 13-part specification including modular logger backends,
4-tier logging schedules, and hardware/activation/gradient metrics.
"""
import os
import json
import time
import math
import subprocess
import torch
import torch.nn as nn

# Optional backends / telemetry libs
try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
except ImportError:
    pynvml = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    wandb = None

# ==============================================================================
# 1. LOGGER INTERFACES (Cat. 13)
# ==============================================================================
class BaseLogger:
    def log_metrics(self, metrics: dict, step: int): pass
    def log_histogram(self, name: str, values: torch.Tensor, step: int): pass
    def log_text(self, name: str, text: str, step: int): pass

class JSONLLogger(BaseLogger):
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
    def log_metrics(self, metrics: dict, step: int):
        metrics['step'] = step
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

class TensorBoardLogger(BaseLogger):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir) if SummaryWriter else None
        if not self.writer:
            print("WARNING: tensorboard not installed. Skipping TB logging.")

    def log_metrics(self, metrics: dict, step: int):
        if not self.writer: return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, step)

    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        if self.writer:
            self.writer.add_histogram(name, values, step)

class WandbLogger(BaseLogger):
    def __init__(self, project: str, run_name: str, config: dict):
        self.active = False
        if wandb and wandb.run is None:
            try:
                wandb.init(project=project, name=run_name, config=config)
                self.active = True
            except Exception as e:
                print(f"WARNING: Failed to init wandb: {e}")

    def log_metrics(self, metrics: dict, step: int):
        if self.active:
            wandb.log(metrics, step=step)

# ==============================================================================
# 2. TELEMETRY MANAGER
# ==============================================================================
class TelemetryManager:
    def __init__(self, run_dir: str, config: dict, use_wandb: bool = False):
        self.run_dir = run_dir
        self.loggers = [
            JSONLLogger(os.path.join(run_dir, 'training_logs.jsonl')),
            TensorBoardLogger(run_dir)
        ]
        if use_wandb:
            self.loggers.append(WandbLogger("slm-project", os.path.basename(run_dir), config))

        # Internal state for advanced metrics
        self.prev_weights = {}
        self.prev_grads = {}
        self.activation_stats = {}
        self.hooks = []
        self._timer_starts = {}

        self._log_metadata(config)

    def _dispatch(self, metrics: dict, step: int):
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    # --- TIER 0: METADATA ---
    def _log_metadata(self, config: dict):
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except:
            commit = "unknown"

        meta = {
            "run_id": os.path.basename(self.run_dir),
            "timestamp_start": time.time(),
            "git_commit_hash": commit,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "cpu",
            "hostname": os.uname().nodename if hasattr(os, 'uname') else "windows",
            "config": config
        }
        with open(os.path.join(self.run_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    # --- TIMERS ---
    def start_timer(self, name: str):
        self._timer_starts[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        if name in self._timer_starts:
            return (time.perf_counter() - self._timer_starts.pop(name)) * 1000.0
        return 0.0

    # --- TIER 1: EVERY STEP ---
    def log_tier1_step(self, step: int, loss: float, lr: float, grad_norm: float, 
                       tokens_seen: int, tokens_this_step: int, step_time_ms: float,
                       data_load_time_ms: float, fwd_time_ms: float, bwd_time_ms: float,
                       opt_time_ms: float, effective_batch_size: int, mfu: float):
        
        metrics = {
            # 1. Loss
            "loss/train_loss": loss,
            "loss/loss_per_token": loss / (tokens_this_step if tokens_this_step > 0 else 1),
            "loss/perplexity": math.exp(loss) if loss < 20 else float('inf'),
            
            # 2. Gradients & Optim
            "grad/norm_global": grad_norm,
            "grad/effective_batch_size": effective_batch_size,
            "optim/learning_rate": lr,

            # 6. Data Pipeline
            "data/tokens_seen_b": tokens_seen / 1e9,
            "data/tokens_per_second": (tokens_this_step / (step_time_ms / 1000.0)) if step_time_ms > 0 else 0,
            
            # 7. Hardware & Timers
            "perf/step_time_ms": step_time_ms,
            "perf/forward_time_ms": fwd_time_ms,
            "perf/backward_time_ms": bwd_time_ms,
            "perf/optimizer_time_ms": opt_time_ms,
            "perf/data_load_time_ms": data_load_time_ms,
            "perf/mfu_percent": mfu * 100.0,
        }

        # Stability Flags
        metrics["stability/nan_detected"] = 1.0 if math.isnan(loss) else 0.0
        metrics["stability/inf_detected"] = 1.0 if math.isinf(loss) else 0.0

        # System/Hardware Telemetry (pynvml / psutil)
        if torch.cuda.is_available():
            metrics["hw/gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
            metrics["hw/gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
        
        if pynvml:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics["hw/gpu_utilization_percent"] = util.gpu
                metrics["hw/gpu_temperature_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics["hw/gpu_power_watt"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except:
                pass
                
        if psutil:
            metrics["hw/cpu_utilization_percent"] = psutil.cpu_percent()
            metrics["hw/ram_used_gb"] = psutil.virtual_memory().used / (1024**3)

        self._dispatch(metrics, step)

    # --- TIER 2: EVERY 50-100 STEPS (Deep Weights/Grads) ---
    def log_tier2_weights(self, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, lr: float):
        metrics = {}
        
        # Optimizer States (M1/M2)
        m1_norms, m2_norms = [], []
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p)
                if state and 'exp_avg' in state:
                    m1_norms.append(state['exp_avg'].norm().item())
                    m2_norms.append(state['exp_avg_sq'].norm().item())
        if m1_norms:
            metrics["optim/adam_m1_mean"] = sum(m1_norms) / len(m1_norms)
            metrics["optim/adam_m2_mean"] = sum(m2_norms) / len(m2_norms)

        # Weight & Gradient Introspection
        w_norms, w_maxs, w_stds = [], [], []
        g_norms, g_maxs = [], []
        
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            
            w_norm = param.norm().item()
            w_norms.append(w_norm)
            w_maxs.append(param.abs().max().item())
            w_stds.append(param.std().item())
            
            # Specific embeddings
            if "tok_emb" in name: metrics["weight/embedding_norm"] = w_norm
            if "output" in name or "lm_head" in name: metrics["weight/lm_head_norm"] = w_norm

            if param.grad is not None:
                g_norm = param.grad.norm().item()
                g_norms.append(g_norm)
                g_maxs.append(param.grad.abs().max().item())
                metrics[f"grad_layer/{name}_norm"] = g_norm
                metrics[f"weight_layer/{name}_norm"] = w_norm

                # Track Adam sign consistency & Delta weights
                p_id = id(param)
                # 1. Sign consistency
                if p_id in self.prev_grads:
                    signs_match = (torch.sign(param.grad) == torch.sign(self.prev_grads[p_id])).float().mean().item()
                    metrics[f"grad_layer/{name}_sign_consistency"] = signs_match
                # Store detached copy for next time (VRAM intensive!)
                self.prev_grads[p_id] = param.grad.detach().clone()

                # 2. Weight delta
                if p_id in self.prev_weights:
                    delta = (param.data - self.prev_weights[p_id]).norm().item()
                    metrics[f"weight_layer/{name}_delta"] = delta
                self.prev_weights[p_id] = param.data.detach().clone()

        if w_norms:
            avg_w_norm = sum(w_norms)/len(w_norms)
            avg_g_norm = sum(g_norms)/len(g_norms) if g_norms else 1e-8
            metrics["weight/mean_norm_global"] = avg_w_norm
            metrics["weight/max_abs_global"] = max(w_maxs)
            metrics["grad/max_abs_global"] = max(g_maxs) if g_maxs else 0
            
            # weight_update_ratio = lr * grad_norm / weight_norm
            if avg_w_norm > 0:
                metrics["weight/update_ratio_global"] = (lr * avg_g_norm) / avg_w_norm
                
        self._dispatch(metrics, step)

    # --- TIER 3: EVERY 500 STEPS (Activations) ---
    def attach_activation_hooks(self, model: nn.Module):
        """Attach forward hooks to sample activations. Call once."""
        self.remove_hooks()
        
        def get_hook(name):
            def hook(module, inp, out):
                # Sample stats without keeping graph
                with torch.no_grad():
                    if isinstance(out, tuple):
                        out = out[0]
                    self.activation_stats[f"act/{name}_mean"] = out.mean().item()
                    self.activation_stats[f"act/{name}_max"] = out.max().item()
                    self.activation_stats[f"act/{name}_std"] = out.std().item()
                    # Sparsity: % of activations < 1e-3
                    sparsity = (out.abs() < 1e-3).float().mean().item()
                    self.activation_stats[f"act/{name}_sparsity"] = sparsity
            return hook

        for name, mod in model.named_modules():
            if "ffn" in name or "attention" in name or "norm" in name:
                self.hooks.append(mod.register_forward_hook(get_hook(name)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def log_tier3_activations(self, step: int):
        """Log stats gathered by hooks on the most recent forward pass."""
        if self.activation_stats:
            self._dispatch(self.activation_stats, step)
            self.activation_stats.clear()

    # --- TIER 4: EVAL & SNAPSHOTS ---
    def log_tier4_eval(self, step: int, eval_ppl: float, chars_per_token: float = 3.5):
        metrics = {
            "eval/perplexity": eval_ppl,
            "eval/bpc": (math.log(eval_ppl) / math.log(2)) / chars_per_token if eval_ppl != float('inf') else float('inf')
        }
        self._dispatch(metrics, step)
