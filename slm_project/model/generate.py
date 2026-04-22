# slm_project/model/generate.py
import torch
from slm_project.model.model import SLM

def generate(model: SLM, input_ids: torch.Tensor, max_new_tokens: int, batch_size: int = 1, seq_len: int = 8192):
    """
    AT INFERENCE — AttnRes caching rules:
      batch=1 at 16K context: caching is optional (0.83 GB extra — acceptable)
      batch>1 at 16K context: ALWAYS recompute AttnRes (do NOT cache layer_outputs)
        Reason: 33 × 4 × 16384 × 768 × 2 bytes = 3.32 GB extra → OOM with KV cache
    """
    if batch_size > 1 and seq_len > 8192:
        use_cached_attnres = False   # recompute on every forward
    else:
        use_cached_attnres = True    # can cache safely
    
    # Generation logic using use_cached_attnres flag would go here
    pass
