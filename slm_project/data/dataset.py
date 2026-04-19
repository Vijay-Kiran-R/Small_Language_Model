# slm_project/data/dataset.py
"""
ShardedDataset — reads pre-tokenised uint16 binary shards.

Shard format
────────────
Each shard is a flat binary file of uint16 values (2 bytes per token).
Produced by download.py → stream_and_tokenize() → _save_shard().

Resume safety
─────────────
Save current_global_idx + shuffle_seed in EVERY checkpoint.
Without this, resuming training re-reads from index 0 — effectively
training on the same data twice and distorting the loss curve.

Example checkpoint entry:
  {'data_global_idx': dataset.global_idx, 'shuffle_seed': seed}
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class ShardedDataset(Dataset):
    """
    Map-style dataset over pre-tokenised uint16 binary shards.

    Splits the concatenated token stream into fixed-length windows of
    seq_len tokens.  Input and label are offset by 1:
      input_ids = tokens[t   : t + seq_len]
      labels    = tokens[t+1 : t + seq_len + 1]

    Deterministic: the same global index always returns the same tokens.
    Supports resume: pass start_global_idx from a saved checkpoint.
    """

    def __init__(
        self,
        shard_glob:       str,
        seq_len:          int = 8192,
        start_global_idx: int = 0,
    ) -> None:
        """
        Args:
            shard_glob:       Glob pattern matching shard files,
                              e.g. 'data/shards/fineweb_edu_shard*.bin'.
            seq_len:          Token window length (ModelConfig.max_seq_len).
            start_global_idx: Resume from this window index (from checkpoint).
        """
        self.seq_len    = seq_len
        self.global_idx = start_global_idx

        # Collect and sort shards deterministically
        self.shard_paths = sorted(glob.glob(shard_glob))
        assert len(self.shard_paths) > 0, (
            f"No shards found matching: {shard_glob!r}\n"
            f"Run slm_project/data/download.py first."
        )

        # Build cumulative token-count index for O(1) window → shard lookup
        self.shard_lengths  = []
        self.shard_offsets  = []   # cumulative token start of each shard
        cumulative = 0
        for path in self.shard_paths:
            n = os.path.getsize(path) // 2   # uint16 = 2 bytes per token
            self.shard_offsets.append(cumulative)
            self.shard_lengths.append(n)
            cumulative += n

        self.total_tokens  = cumulative
        self.total_windows = max(0, (self.total_tokens - 1) // seq_len)

        print(
            f"ShardedDataset: {len(self.shard_paths)} shard(s), "
            f"{self.total_tokens:,} tokens, "
            f"{self.total_windows:,} windows of {seq_len} tokens each."
        )

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int):
        """
        Returns:
            (input_ids, labels) each as torch.int64 tensors of length seq_len.
        """
        token_start = idx * self.seq_len
        token_end   = token_start + self.seq_len + 1   # +1 for the label shift

        # Find which shard contains token_start
        shard_idx = self._find_shard(token_start)
        shard_offset = self.shard_offsets[shard_idx]
        local_start  = token_start - shard_offset
        local_end    = local_start + self.seq_len + 1

        data = np.fromfile(self.shard_paths[shard_idx], dtype=np.uint16)

        if local_end <= len(data):
            # Common case: window fits entirely in one shard
            chunk = data[local_start:local_end].astype(np.int64)
        else:
            # Window spans a shard boundary — stitch from two consecutive shards
            part1 = data[local_start:].astype(np.int64)
            if shard_idx + 1 < len(self.shard_paths):
                data2 = np.fromfile(
                    self.shard_paths[shard_idx + 1], dtype=np.uint16
                )
                need  = (self.seq_len + 1) - len(part1)
                part2 = data2[:need].astype(np.int64)
                chunk = np.concatenate([part1, part2])
            else:
                chunk = part1

        # Pad with zeros if we're at the very end of the dataset
        if len(chunk) < self.seq_len + 1:
            chunk = np.pad(chunk, (0, self.seq_len + 1 - len(chunk)))

        input_ids = torch.from_numpy(chunk[:-1])
        labels    = torch.from_numpy(chunk[1:])
        return input_ids, labels

    def _find_shard(self, token_pos: int) -> int:
        """Binary search: return shard index containing token_pos."""
        lo, hi = 0, len(self.shard_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.shard_offsets[mid] <= token_pos:
                lo = mid
            else:
                hi = mid - 1
        return lo
