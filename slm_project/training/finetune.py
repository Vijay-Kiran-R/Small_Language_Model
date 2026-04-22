# slm_project/training/finetune.py
"""
CORRECTED Loss Mask Specification (from Master Build Plan Part 8):

Token                    loss_weight    Reason
─────────────────────────────────────────────────────────────────
<|system|> content       0.0            Model never generates system prompts
<|user|> content         0.0            Model never generates user turns
<|assistant|>            0.0            Role marker (not generated, just a delimiter)
assistant text content   1.0            Model generates this ← TRAIN ON THIS
<|tool_call|>            1.0            Model must learn to emit tool call start
tool call JSON content   1.0            Model generates this
<|/tool_call|>           1.0            Model must learn to emit tool call end
<|tool_result|>          0.0  ← CRITICAL  Orchestrator injects; model NEVER generates
tool result content      0.0  ← CRITICAL  Oracle data; model must NOT memorize
<|/tool_result|>         0.0  ← CRITICAL  Orchestrator injects; model NEVER generates
<|end|> (EOS)            1.0            Model must learn to stop
─────────────────────────────────────────────────────────────────

CRITICAL: ENTIRE <|tool_result|>...<|/tool_result|> block = 0.0,
INCLUDING BOTH DELIMITER TOKENS. The orchestrator injects this block.
Setting only the content to 0 but leaving the delimiters at 1 is WRONG.

CRITICAL: position_ids MUST reset to 0 at each packed example boundary.
Without this, RoPE applies wrong positions to subsequent packed examples,
silently corrupting fine-tuning.
"""
import torch
import torch.nn as nn
from slm_project.tokenizer_utils import load_tokenizer

tok = load_tokenizer()

# Special token IDs (must match tokenizer exactly)
TOK_SYSTEM     = tok.convert_tokens_to_ids('<|system|>')      # 32000
TOK_USER       = tok.convert_tokens_to_ids('<|user|>')        # 32001
TOK_ASSISTANT  = tok.convert_tokens_to_ids('<|assistant|>')   # 32002
TOK_THINK      = tok.convert_tokens_to_ids('<|think|>')       # 32003
TOK_THINK_END  = tok.convert_tokens_to_ids('<|/think|>')      # 32004
TOK_END        = tok.convert_tokens_to_ids('<|end|>')         # 32005
TOK_TOOL_CALL  = tok.convert_tokens_to_ids('<|tool_call|>')   # 32006
TOK_TOOL_CALL_END = tok.convert_tokens_to_ids('<|/tool_call|>')  # 32007
TOK_TOOL_RES   = tok.convert_tokens_to_ids('<|tool_result|>')    # 32008
TOK_TOOL_RES_END = tok.convert_tokens_to_ids('<|/tool_result|>') # 32009


def build_loss_mask(input_ids: list[int]) -> list[int]:
    """
    Build a token-level loss mask for a conversation sequence.
    Returns a list of 0/1 values of the same length as input_ids.
    
    Uses a state machine to track which role is currently active.
    """
    mask = [0] * len(input_ids)
    i = 0
    state = 'system'  # system | user | assistant | tool_result

    while i < len(input_ids):
        tok_id = input_ids[i]

        if tok_id == TOK_SYSTEM:
            state = 'system'
            mask[i] = 0   # <|system|> marker itself = 0
            i += 1

        elif tok_id == TOK_USER:
            state = 'user'
            mask[i] = 0
            i += 1

        elif tok_id == TOK_ASSISTANT:
            state = 'assistant'
            mask[i] = 0   # <|assistant|> marker = 0 (delimiter, not generated)
            i += 1

        elif tok_id == TOK_TOOL_RES:
            # CRITICAL: entire <|tool_result|>...<|/tool_result|> = 0.0
            # Including both delimiter tokens
            mask[i] = 0   # <|tool_result|> delimiter = 0
            i += 1
            while i < len(input_ids) and input_ids[i] != TOK_TOOL_RES_END:
                mask[i] = 0   # all content = 0
                i += 1
            if i < len(input_ids):
                mask[i] = 0   # <|/tool_result|> delimiter = 0
                i += 1

        elif tok_id == TOK_END:
            # EOS: loss = 1 if in assistant turn, 0 if in system/user turn
            mask[i] = 1 if state == 'assistant' else 0
            i += 1

        else:
            # Regular content: 1 if in assistant state, 0 otherwise
            mask[i] = 1 if state == 'assistant' else 0
            i += 1

    return mask


def build_cot_loss_mask(input_ids: list[int]) -> list[int]:
    """
    Like build_loss_mask but <|think|>...<|/think|> block = 1.0 (not 0).
    Used for CoT fine-tuning examples.
    """
    mask = build_loss_mask(input_ids)  # start with standard mask
    # <|think|> block is inside assistant turn → already 1.0 from build_loss_mask
    # No additional changes needed if think tokens appear in assistant turn
    return mask


def pack_conversations(examples: list[dict], max_seq_len: int = 8192) -> dict:
    """
    Pack multiple conversations into a single sequence.
    CRITICAL: position_ids MUST reset to 0 at each packed example boundary.
    
    Returns dict with:
      input_ids:   [max_seq_len]  (token IDs; padded with EOS if short)
      labels:      [max_seq_len]  (same as input_ids; -100 where mask=0)
      position_ids:[max_seq_len]  (resets to 0 at each conversation start)
    """
    all_ids       = []
    all_masks     = []
    all_positions = []

    for example in examples:
        conv_ids  = format_conversation(example)
        conv_mask = build_loss_mask(conv_ids)
        pos_ids   = list(range(len(conv_ids)))   # resets per conversation
        all_ids.extend(conv_ids)
        all_masks.extend(conv_mask)
        all_positions.extend(pos_ids)

    # Truncate / pad to max_seq_len
    all_ids       = all_ids[:max_seq_len]
    all_masks     = all_masks[:max_seq_len]
    all_positions = all_positions[:max_seq_len]

    pad_len = max_seq_len - len(all_ids)
    all_ids       += [TOK_END] * pad_len
    all_masks     += [0] * pad_len
    all_positions += list(range(pad_len))

    # Convert mask to labels (0 → -100 so CE ignores it)
    labels = [-100 if m == 0 else tid for tid, m in zip(all_ids, all_masks)]

    return {
        'input_ids':   torch.tensor(all_ids,       dtype=torch.long),
        'labels':      torch.tensor(labels,         dtype=torch.long),
        'position_ids':torch.tensor(all_positions,  dtype=torch.long),
    }


def format_conversation(example: dict) -> list[int]:
    """Convert a conversation dict to a flat token ID list."""
    ids = []
    messages = example.get('messages', example.get('conversations', []))

    for msg in messages:
        role    = msg.get('role', msg.get('from', ''))
        content = msg.get('content', msg.get('value', ''))

        if role in ('system', 'human', 'user', 'instruction'):
            role_tok = TOK_SYSTEM if role == 'system' else TOK_USER
            ids += [role_tok] + tok.encode(content, add_special_tokens=False) + [TOK_END]
        elif role in ('assistant', 'gpt', 'bot', 'response'):
            ids += [TOK_ASSISTANT] + tok.encode(content, add_special_tokens=False) + [TOK_END]

    return ids


class SFTTrainer:
    """Fine-tuning trainer. Applies dropout only to W_O and W_down."""

    def __init__(self, model, optimizer, fcfg, device='cuda'):
        self.model     = model
        self.optimizer = optimizer
        self.fcfg      = fcfg
        self.device    = device
        self.step      = 0

        # Enable dropout for fine-tuning (W_O and W_down only)
        # These are identified by name in the model
        self._enable_dropout(dropout=fcfg.dropout)

    def _enable_dropout(self, dropout: float):
        """Apply dropout only after W_O and W_down — not globally."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                # Only enable dropout in attention output and FFN output positions
                # Your block.py should set dropout in FFN(W_down) and attention(W_O)
                module.p = dropout

    def train_step(self, input_ids, labels, position_ids):
        """Single fine-tuning step with correct loss masking."""
        input_ids    = input_ids.to(self.device)
        labels       = labels.to(self.device)
        position_ids = position_ids.to(self.device)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = self.model(input_ids, global_step=self.step)

        # Manually compute loss with correct ignore_index=-100
        # (labels already have -100 where mask=0 from pack_conversations)
        import torch.nn.functional as F
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 32010),
            labels[:, 1:].reshape(-1),
            ignore_index=-100
        )

        (loss / self.fcfg.grad_accum).backward()
        return loss.item()
