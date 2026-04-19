# slm_project/tokenizer_utils.py
"""
Tokenizer utilities for the SLM project.

Loads the Mistral v0.1 tokenizer (32,000 base tokens), extends it with
exactly 10 special tokens to reach vocab_size=32,010, saves it locally,
and provides a full verification loop.

CRITICAL: run verify_all_special_tokens() before any model code touches
a token ID. If a special token encodes to >1 ID, EOS and tool tokens will
never be detected correctly during training/inference.
"""

import os
from transformers import AutoTokenizer

# ── Special token table ───────────────────────────────────────────────────────
# IDs are assigned sequentially starting at 32000.
SPECIAL_TOKENS = [
    '<|system|>',       # ID 32000
    '<|user|>',         # ID 32001
    '<|assistant|>',    # ID 32002
    '<|think|>',        # ID 32003
    '<|/think|>',       # ID 32004
    '<|end|>',          # ID 32005  ← EOS token
    '<|tool_call|>',    # ID 32006
    '<|/tool_call|>',   # ID 32007
    '<|tool_result|>',  # ID 32008
    '<|/tool_result|>', # ID 32009
]

EXPECTED_IDS: dict[str, int] = {
    tok: 32000 + i for i, tok in enumerate(SPECIAL_TOKENS)
}

TOKENIZER_SAVE_PATH = 'tokenizer/'


# ── Build & save ──────────────────────────────────────────────────────────────

def build_and_save_tokenizer(
    hf_model_id: str = 'mistralai/Mistral-7B-v0.1',
) -> AutoTokenizer:
    """
    Load Mistral v0.1 tokenizer, add 10 special tokens, save locally.

    Args:
        hf_model_id: HuggingFace model ID to pull the base tokenizer from.

    Returns:
        The extended tokenizer (vocab_size == 32,010).
    """
    print(f"Loading base tokenizer from {hf_model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    print("Adding 10 special tokens ...")
    num_added = tokenizer.add_special_tokens(
        {'additional_special_tokens': SPECIAL_TOKENS}
    )
    assert num_added == 10, (
        f"Expected exactly 10 new tokens, got {num_added}. "
        "Some tokens may already exist in the base vocabulary."
    )

    # Set EOS explicitly
    tokenizer.eos_token    = '<|end|>'
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|end|>')
    assert tokenizer.eos_token_id == 32005, (
        f"EOS token ID must be 32005, got {tokenizer.eos_token_id}. "
        "Check that SPECIAL_TOKENS ordering matches EXPECTED_IDS."
    )

    os.makedirs(TOKENIZER_SAVE_PATH, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
    print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH!r}")
    return tokenizer


# ── Load ──────────────────────────────────────────────────────────────────────

def load_tokenizer() -> AutoTokenizer:
    """Load the already-saved tokenizer from disk."""
    return AutoTokenizer.from_pretrained(TOKENIZER_SAVE_PATH)


# ── Verify ────────────────────────────────────────────────────────────────────

def verify_all_special_tokens(tokenizer: AutoTokenizer) -> None:
    """
    CRITICAL — run the full 10-token co-verification loop.

    Checks:
      1. Every special token encodes to exactly one ID (no SentencePiece splits).
      2. Each ID matches the expected value from EXPECTED_IDS.
      3. Total vocab size == 32,010.

    Raises AssertionError immediately on first failure with a descriptive message.
    """
    print("Running full 10-token co-verification ...")
    for token, expected_id in EXPECTED_IDS.items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        assert ids == [expected_id], (
            f"FAIL: {token!r} → {ids}, expected [{expected_id}].\n"
            f"SentencePiece split this token into multiple pieces.\n"
            f"Fix: call tokenizer.add_special_tokens() again or retrain."
        )

    assert len(tokenizer) == 32_010, (
        f"Expected vocab_size=32,010, got {len(tokenizer)}. "
        "add_special_tokens() may not have committed all tokens."
    )

    print("All 10 special tokens verified OK")
    print(f"Total vocab size: {len(tokenizer)} OK")
