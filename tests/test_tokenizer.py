# tests/test_tokenizer.py
"""
Stage 2 test gate — tokenizer setup & full verification.
Run as:  python -m pytest tests/test_tokenizer.py -v
     or: python tests/test_tokenizer.py
"""

import sys
import os

# Allow running from project root without installing as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from slm_project.tokenizer_utils import (
    build_and_save_tokenizer,
    load_tokenizer,
    verify_all_special_tokens,
    EXPECTED_IDS,
)


def test_build_and_verify():
    """Build tokenizer, add special tokens, verify all 10."""
    tok = build_and_save_tokenizer()
    verify_all_special_tokens(tok)


def test_save_load_roundtrip():
    """Load saved tokenizer from disk and re-verify."""
    tok2 = load_tokenizer()
    verify_all_special_tokens(tok2)


def test_encode_decode_roundtrip():
    """Special tokens survive encode → decode."""
    tok2 = load_tokenizer()
    text = "Hello <|assistant|> world <|end|>"
    ids = tok2.encode(text, add_special_tokens=False)
    decoded = tok2.decode(ids)
    assert '<|assistant|>' in decoded, f"<|assistant|> missing from: {decoded!r}"
    assert '<|end|>' in decoded,       f"<|end|> missing from: {decoded!r}"


def test_eos_is_single_token():
    """EOS must encode to exactly [32005] — never split."""
    tok2 = load_tokenizer()
    assert tok2.eos_token_id == 32005, (
        f"eos_token_id must be 32005, got {tok2.eos_token_id}"
    )
    eos_ids = tok2.encode('<|end|>', add_special_tokens=False)
    assert eos_ids == [32005], (
        f"EOS must encode to single token [32005], got {eos_ids}"
    )


def test_all_special_tokens_single_id():
    """Every special token must encode to exactly one ID."""
    tok2 = load_tokenizer()
    for token, expected_id in EXPECTED_IDS.items():
        ids = tok2.encode(token, add_special_tokens=False)
        assert ids == [expected_id], (
            f"{token!r} encoded to {ids}, expected [{expected_id}]"
        )


def test_vocab_size():
    """Total vocabulary must be exactly 32,010."""
    tok2 = load_tokenizer()
    assert len(tok2) == 32_010, f"Expected 32010, got {len(tok2)}"


# ── Standalone runner ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("STAGE 2 TEST GATE")
    print("=" * 60)

    print("\n[1] Build + verify ...")
    test_build_and_verify()

    print("\n[2] Save/load roundtrip ...")
    test_save_load_roundtrip()

    print("\n[3] Encode/decode roundtrip ...")
    test_encode_decode_roundtrip()

    print("\n[4] EOS is single token ...")
    test_eos_is_single_token()

    print("\n[5] All specials encode to single ID ...")
    test_all_special_tokens_single_id()

    print("\n[6] Vocab size == 32,010 ...")
    test_vocab_size()

    print("\n" + "=" * 60)
    print("STAGE 2 PASSED")
