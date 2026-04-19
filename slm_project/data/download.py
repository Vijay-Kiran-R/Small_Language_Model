# slm_project/data/download.py
"""
Streaming data downloader for Phase 1 smoke test.

⚠ WARNING — NEVER download the full dataset.
  FineWeb-Edu sample-10BT is ~10B tokens; we stop at 2M.
  Wikipedia EN is ~4B tokens; we stop at 500K.
  Both datasets are streamed — no full download to disk.

Token budget is enforced by stopping iteration mid-stream.
Truncation mid-document is intentional and harmless.
"""

import os
import numpy as np
from datasets import load_dataset

from slm_project.tokenizer_utils import load_tokenizer
from slm_project.config import Phase1Config

p1cfg = Phase1Config()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_shard(tokens: list, prefix: str, idx: int) -> None:
    """Write a list of token IDs as a uint16 binary file."""
    arr  = np.array(tokens, dtype=np.uint16)
    path = f"{prefix}_shard{idx:04d}.bin"
    arr.tofile(path)
    print(f"  Saved shard {idx:04d}: {len(tokens):,} tokens → {path}")


# ── Core streaming function ───────────────────────────────────────────────────

def stream_and_tokenize(
    hf_id:               str,
    hf_name:             str | None,
    token_budget:        int,
    output_shard_prefix: str,
    shard_size:          int  = 200_000,
    split:               str  = 'train',
    text_field:          str  = 'text',
) -> None:
    """
    Stream a HuggingFace dataset and tokenise until token_budget is reached.
    Saves uint16 binary shards of exactly shard_size tokens (last shard may
    be smaller).

    Args:
        hf_id:               HuggingFace dataset ID.
        hf_name:             Dataset config name (None if not applicable).
        token_budget:        Stop after collecting this many tokens.
        output_shard_prefix: File path prefix; shard index and .bin appended.
        shard_size:          Tokens per shard (default 200 K).
        split:               Dataset split to stream.
        text_field:          Field name containing the raw text.

    Guarantees:
      - Never loads more than shard_size tokens into RAM at once.
      - Stops exactly at token_budget (may truncate mid-document — fine).
      - Each document ends with eos_token_id as a separator.
    """
    tok = load_tokenizer()
    os.makedirs(os.path.dirname(output_shard_prefix) or '.', exist_ok=True)

    load_kwargs = {'streaming': True, 'split': split}
    if hf_name:
        ds = load_dataset(hf_id, name=hf_name, **load_kwargs)
    else:
        ds = load_dataset(hf_id, **load_kwargs)

    tokens_collected = 0
    shard_idx        = 0
    current_shard    = []

    print(f"\nStreaming {hf_id}"
          + (f" [{hf_name}]" if hf_name else "")
          + f"  (budget: {token_budget:,} tokens) ...")

    for example in ds:
        if tokens_collected >= token_budget:
            break

        text = example.get(text_field, '')
        if not text or len(text.strip()) < 20:
            continue

        ids = tok.encode(text, add_special_tokens=False)
        ids.append(tok.eos_token_id)   # document separator

        # Clip to exactly the remaining budget
        remaining = token_budget - tokens_collected
        ids = ids[:remaining]

        current_shard.extend(ids)
        tokens_collected += len(ids)

        # Flush full shards
        while len(current_shard) >= shard_size:
            _save_shard(current_shard[:shard_size], output_shard_prefix, shard_idx)
            shard_idx     += 1
            current_shard  = current_shard[shard_size:]

    # Flush remaining tokens
    if current_shard:
        _save_shard(current_shard, output_shard_prefix, shard_idx)
        shard_idx += 1

    print(f"Done: {tokens_collected:,} tokens in {shard_idx} shard(s).")


# ── Phase 1 entry point ───────────────────────────────────────────────────────

def download_phase1() -> None:
    """
    Download and tokenise all data for the 3-layer smoke test.

    Budgets (from Phase1Config — intentionally small):
      FineWeb-Edu : 2,000,000 tokens   (streaming, stop early)
      Wikipedia EN:   500,000 tokens   (streaming, stop early)
    """
    # FineWeb-Edu sample-10BT — streaming, stop at 2M tokens
    stream_and_tokenize(
        hf_id='HuggingFaceFW/fineweb-edu',
        hf_name='sample-10BT',
        token_budget=p1cfg.fineweb_tokens,
        output_shard_prefix='data/shards/fineweb_edu',
        shard_size=p1cfg.shard_size,
    )

    # Wikipedia EN 2024 — streaming, stop at 500K tokens
    stream_and_tokenize(
        hf_id='wikimedia/wikipedia',
        hf_name='20241101.en',
        token_budget=p1cfg.wikipedia_tokens,
        output_shard_prefix='data/shards/wikipedia_en',
        shard_size=p1cfg.shard_size,
    )

    print("\nPhase 1 data download complete.")
    print(f"  FineWeb shards : data/shards/fineweb_edu_shard*.bin")
    print(f"  Wikipedia shards: data/shards/wikipedia_en_shard*.bin")


if __name__ == '__main__':
    download_phase1()
