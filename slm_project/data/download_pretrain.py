# slm_project/data/download_pretrain.py
"""
Streaming download manager for large datasets.
Tokenises and saves uint16 shards as it streams.
NEVER loads full FineWeb-Edu or DCLM-Baseline into memory.

Stage 1 datasets and token budgets:
  FineWeb-Edu (score≥3): 3.2B   → shards in data/shards/stage1/fineweb/
  DCLM-Baseline:         2.0B   → shards in data/shards/stage1/dclm/
  Project Gutenberg:     1.6B   → shards in data/shards/stage1/gutenberg/
  Wikipedia EN 2024:     1.2B   → shards in data/shards/stage1/wikipedia/

Stage 2 adds:
  FineWeb-Edu (350BT):   2.64B  → data/shards/stage2/fineweb/
  DCLM-Baseline:         1.2B   → data/shards/stage2/dclm/
  Project Gutenberg:     1.2B   → data/shards/stage2/gutenberg/
  Wikipedia EN 2024:     0.8B   → data/shards/stage2/wikipedia/
  Stack-Edu:             0.8B   → data/shards/stage2/stack_edu/
  OpenWebMath:           0.64B  → data/shards/stage2/openwebmath/
  Cosmopedia v2 (stories only): 0.56B → data/shards/stage2/cosmopedia/
  Python-Edu:            0.16B  → data/shards/stage2/python_edu/

Stage 3 adds:
  FineWeb-Edu (score≥4): 1.0B   → data/shards/stage3/fineweb/
  Cosmopedia v2 (FULL):  0.8B   → data/shards/stage3/cosmopedia/
  DCLM-Baseline:         0.6B   → data/shards/stage3/dclm/
  Stack-Edu:             0.4B   → data/shards/stage3/stack_edu/
  FineMath 4+:           0.4B   → data/shards/stage3/finemath/
  Project Gutenberg:     0.4B   → data/shards/stage3/gutenberg/
  SODA:                  0.3B   → data/shards/stage3/soda/
  OpenSubtitles:         0.1B   → data/shards/stage3/opensubtitles/
"""
from datasets import load_dataset
from slm_project.tokenizer_utils import load_tokenizer
import numpy as np, os, hashlib
try:
    from rbloom import Bloom
except ImportError:
    # Handle the case where it might not be installed yet for local testing
    Bloom = None

SHARD_SIZE = 100_000_000   # 100M tokens per shard (uint16 = 200 MB per shard)

# Bloom filter for Phase 5.5 dedup (built during pretraining preprocessing)
BLOOM_PATH = 'data/pretraining_hashes.bloom'

if Bloom is not None:
    seen_hashes = Bloom(20_000_000, 0.001)   # 20M entries, 0.1% false positive
else:
    seen_hashes = None


def hash_doc(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def stream_tokenize_shard(
    hf_id: str,
    hf_name: str | None,
    token_budget: int,
    output_dir: str,
    split: str = 'train',
    text_field: str = 'text',
    min_doc_tokens: int = 20,
    filter_fn=None,
):
    """
    Stream a HuggingFace dataset, tokenise, and save uint16 shards.
    Also adds document hashes to seen_hashes bloom filter for Phase 5.5 dedup.

    filter_fn: optional callable(example) → bool for dataset-specific filtering
               e.g. FineWeb-Edu score≥3, Project Gutenberg docs > 4096 tokens
    """
    tok = load_tokenizer()
    os.makedirs(output_dir, exist_ok=True)

    load_kwargs = {'streaming': True, 'split': split}
    ds = load_dataset(hf_id, name=hf_name, **load_kwargs) if hf_name else \
         load_dataset(hf_id, **load_kwargs)

    tokens_done = 0
    shard_idx   = 0
    shard_buf   = []

    print(f"\nStreaming: {hf_id}  budget={token_budget/1e9:.2f}B tokens ...")

    for example in ds:
        if tokens_done >= token_budget:
            break
        if filter_fn and not filter_fn(example):
            continue

        text = example.get(text_field, '')
        if not text or len(text.strip()) < min_doc_tokens:
            continue

        # Add to bloom filter (for Phase 5.5 dedup)
        if seen_hashes is not None:
            seen_hashes.add(hash_doc(text))

        ids = tok.encode(text, add_special_tokens=False)
        ids.append(tok.eos_token_id)
        shard_buf.extend(ids)
        tokens_done += len(ids)

        while len(shard_buf) >= SHARD_SIZE:
            _write_shard(shard_buf[:SHARD_SIZE], output_dir, shard_idx)
            shard_idx += 1
            shard_buf  = shard_buf[SHARD_SIZE:]

    if shard_buf:
        _write_shard(shard_buf, output_dir, shard_idx)
        shard_idx += 1

    print(f"  Done: {tokens_done/1e9:.3f}B tokens → {shard_idx} shards")
    return tokens_done


def _write_shard(tokens: list, output_dir: str, idx: int):
    arr  = np.array(tokens, dtype=np.uint16)
    path = os.path.join(output_dir, f"shard_{idx:04d}.bin")
    arr.tofile(path)
    print(f"  Shard {idx}: {len(tokens):,} tokens → {path}")


# ── Stage 1 filters ──────────────────────────────────────────
def fineweb_score_ge3(example):
    return example.get('score', 0) >= 3

def fineweb_score_ge4(example):
    return example.get('score', 0) >= 4

def gutenberg_long_docs(example):
    """Only include docs > 4096 tokens — REQUIRED for NoPE global layer training."""
    text = example.get('text', '')
    # Quick word-count proxy (actual tokens ≈ words × 1.3 for books)
    return len(text.split()) > 3100   # ~4096 tokens


# ── Download functions per stage ──────────────────────────────
def download_stage1():
    """Stage 1: Foundation (0–8B tokens). Run before pretraining Stage 1."""
    print("=== DOWNLOADING STAGE 1 DATA (0–8B tokens) ===")
    stream_tokenize_shard(
        'HuggingFaceFW/fineweb-edu', 'sample-350BT',
        3_200_000_000, 'data/shards/stage1/fineweb',
        filter_fn=fineweb_score_ge3
    )
    stream_tokenize_shard(
        'mlfoundations/dclm-baseline', None,
        2_000_000_000, 'data/shards/stage1/dclm'
    )
    stream_tokenize_shard(
        'storytracer/US-PD-Books', None,
        1_600_000_000, 'data/shards/stage1/gutenberg',
        filter_fn=gutenberg_long_docs
    )
    stream_tokenize_shard(
        'wikimedia/wikipedia', '20241101.en',
        1_200_000_000, 'data/shards/stage1/wikipedia'
    )
    _save_bloom()
    print("Stage 1 data complete: 8.0B tokens")


def download_stage2():
    """Stage 2: Structure (8B–16B tokens). Run while Stage 1 trains."""
    print("=== DOWNLOADING STAGE 2 DATA (8–16B tokens) ===")
    stream_tokenize_shard(
        'HuggingFaceFW/fineweb-edu', 'sample-350BT',
        2_640_000_000, 'data/shards/stage2/fineweb',
        filter_fn=fineweb_score_ge3
    )
    stream_tokenize_shard(
        'mlfoundations/dclm-baseline', None,
        1_200_000_000, 'data/shards/stage2/dclm'
    )
    stream_tokenize_shard(
        'storytracer/US-PD-Books', None,
        1_200_000_000, 'data/shards/stage2/gutenberg',
        filter_fn=gutenberg_long_docs
    )
    stream_tokenize_shard(
        'wikimedia/wikipedia', '20241101.en',
        800_000_000, 'data/shards/stage2/wikipedia'
    )
    stream_tokenize_shard(
        'HuggingFaceTB/stack-edu', None,
        800_000_000, 'data/shards/stage2/stack_edu'
    )
    stream_tokenize_shard(
        'open-web-math/open-web-math', None,
        640_000_000, 'data/shards/stage2/openwebmath'
    )
    # Cosmopedia v2 — STORIES SUBSET ONLY in Stage 2
    stream_tokenize_shard(
        'HuggingFaceTB/cosmopedia-v2', None,
        560_000_000, 'data/shards/stage2/cosmopedia',
        filter_fn=lambda ex: ex.get('format', '') == 'stories'
    )
    stream_tokenize_shard(
        'HuggingFaceTB/python-edu', None,
        160_000_000, 'data/shards/stage2/python_edu'
    )
    _save_bloom()
    print("Stage 2 data complete: 8.0B tokens")


def download_stage3():
    """Stage 3: High-Signal Decay (16B–20B tokens). Run while Stage 2 trains."""
    print("=== DOWNLOADING STAGE 3 DATA (16–20B tokens) ===")
    # FineWeb-Edu with HIGHER quality filter (score≥4) in final stage
    stream_tokenize_shard(
        'HuggingFaceFW/fineweb-edu', 'sample-350BT',
        1_000_000_000, 'data/shards/stage3/fineweb',
        filter_fn=fineweb_score_ge4
    )
    # Cosmopedia v2 — FULL (textbooks + stories) in Stage 3
    stream_tokenize_shard(
        'HuggingFaceTB/cosmopedia-v2', None,
        800_000_000, 'data/shards/stage3/cosmopedia'
    )
    stream_tokenize_shard(
        'mlfoundations/dclm-baseline', None,
        600_000_000, 'data/shards/stage3/dclm'
    )
    stream_tokenize_shard(
        'HuggingFaceTB/stack-edu', None,
        400_000_000, 'data/shards/stage3/stack_edu'
    )
    # FineMath 4+ — Stage 3 ONLY (math peaks in final stage per SmolLM2 ablations)
    stream_tokenize_shard(
        'HuggingFaceTB/finemath', None,
        400_000_000, 'data/shards/stage3/finemath'
    )
    stream_tokenize_shard(
        'storytracer/US-PD-Books', None,
        400_000_000, 'data/shards/stage3/gutenberg',
        filter_fn=gutenberg_long_docs
    )
    # SODA — domain seeding; 1M+ social dialogues; no upsampling needed
    stream_tokenize_shard(
        'allenai/soda', None,
        300_000_000, 'data/shards/stage3/soda'
    )
    # OpenSubtitles — natural emotional speech; domain seeding only
    stream_tokenize_shard(
        'Helsinki-NLP/open_subtitles', None,
        100_000_000, 'data/shards/stage3/opensubtitles',
        text_field='translation'
    )
    _save_bloom()
    print("Stage 3 data complete: 4.0B tokens")


def _save_bloom():
    if seen_hashes is not None:
        seen_hashes.save(BLOOM_PATH)
        print(f"Bloom filter saved: {BLOOM_PATH}")

if __name__ == '__main__':
    # -------------------------------------------------------------
    # NEW CODE (FULL SCALE) - COMMENTED OUT FOR TESTING
    # -------------------------------------------------------------
    # download_stage1()
    # download_stage2()
    # download_stage3()

    # -------------------------------------------------------------
    # OLD CODE (SMALL/DEV SCALE) - UNCOMMENTED FOR TESTING
    # -------------------------------------------------------------
    from slm_project.data.download import download_phase1
    download_phase1()
