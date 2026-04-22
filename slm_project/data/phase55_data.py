"""
Phase 5.5 data: long documents from pretraining sources, deduplicated
against all pretraining shards using the Bloom filter built in Stage 2.

Use: Project Gutenberg (long books > 4096 tokens) + long Wikipedia articles
     + long DCLM-Baseline documents
Filter: any document whose hash is in the pretraining bloom filter is EXCLUDED.
"""
from rbloom import Bloom
import hashlib

BLOOM_PATH = 'data/pretraining_hashes.bloom'

def load_bloom_filter() -> Bloom:
    print(f"Loading Bloom filter from {BLOOM_PATH} ...")
    return Bloom.load(BLOOM_PATH)

def is_new_document(text: str, seen: Bloom) -> bool:
    """True if document was NOT seen during pretraining."""
    doc_hash = hashlib.md5(text.encode()).hexdigest()
    return doc_hash not in seen

def filter_long_documents(dataset, min_tokens: int = 4096, max_docs: int = None):
    """Yield documents > min_tokens that are not in pretraining set."""
    from slm_project.tokenizer_utils import load_tokenizer
    tok  = load_tokenizer()
    seen = load_bloom_filter()
    n    = 0
    for example in dataset:
        text = example.get('text', '')
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < min_tokens:
            continue
        if not is_new_document(text, seen):
            continue   # skip if seen during pretraining
        yield example
        n += 1
        if max_docs and n >= max_docs:
            break
