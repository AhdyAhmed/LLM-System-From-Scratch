"""
Day036: Memory-mapped tokenized shards, EOS-separated document packing within each split,
source-document provenance tracing, padding/loss masks, and resumable deterministic iteration.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch

from scratchlm.data.ingestion import CleanedCorpusRecord
from scratchlm.tokenizer.byte_tokenizer import EOS_TOKEN_ID, IGNORE_INDEX, PAD_TOKEN_ID


class TokenizerProtocol(Protocol):
    vocab_size: int
    pad_token_id: int
    eos_token_id: int

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
    def tokenizer_hash(self) -> str: ...


@dataclass(frozen=True)
class DocumentTokenSpan:
    """Exact token offset span [start_offset, end_offset) for a packed document inside a split shard."""

    doc_id: str
    split_name: str
    start_offset: int
    end_offset: int
    token_count: int
    ends_with_eos: bool


@dataclass(frozen=True)
class TokenShardIndex:
    """Metadata and provenance manifest for a binary `.bin` memory-mapped token shard."""

    schema_version: str
    split_name: str
    dataset_id: str
    split_file_sha256: str
    tokenizer_hash: str
    vocab_size: int
    total_tokens: int
    eos_token_id: int
    pad_token_id: int
    shard_bin_sha256: str
    document_spans: list[DocumentTokenSpan]

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.canonical_json()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: Path) -> TokenShardIndex:
        raw = json.loads(path.read_text(encoding="utf-8"))
        spans = [DocumentTokenSpan(**s) for s in raw["document_spans"]]
        return cls(
            schema_version=raw["schema_version"],
            split_name=raw["split_name"],
            dataset_id=raw["dataset_id"],
            split_file_sha256=raw["split_file_sha256"],
            tokenizer_hash=raw["tokenizer_hash"],
            vocab_size=raw["vocab_size"],
            total_tokens=raw["total_tokens"],
            eos_token_id=raw["eos_token_id"],
            pad_token_id=raw["pad_token_id"],
            shard_bin_sha256=raw["shard_bin_sha256"],
            document_spans=spans,
        )


@dataclass(frozen=True)
class PackedBatch:
    """
    A next-token training/evaluation batch drawn from a single split shard, with full
    source-document provenance and EOS boundary indices.
    """

    split_name: str
    input_ids: torch.Tensor          # [B, T] int64
    targets: torch.Tensor            # [B, T] int64 (with IGNORE_INDEX = -100 on padded positions)
    valid_target_counts: torch.Tensor  # [B] int64
    total_valid_targets: int
    window_indices: list[int]
    source_doc_ids_per_row: list[list[str]]
    eos_positions_per_row: list[list[int]]


def build_packed_split_shard(
    records: Sequence[CleanedCorpusRecord],
    *,
    split_name: str,
    dataset_id: str,
    split_file_sha256: str,
    tokenizer: TokenizerProtocol,
    output_dir: Path,
    allowed_doc_ids: set[str] | None = None,
) -> TokenShardIndex:
    """
    Pack documents within a single split into an EOS-separated uint16 binary shard (`.bin`)
    and a provenance index (`.idx.json`). Rejects cross-split packing if any document ID is
    outside `allowed_doc_ids`.
    """
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split_name '{split_name}'")
    if not records:
        raise ValueError(f"Cannot build shard for empty split '{split_name}'")

    all_tokens: list[int] = []
    spans: list[DocumentTokenSpan] = []
    seen_ids: set[str] = set()

    for rec in records:
        if allowed_doc_ids is not None and rec.doc_id not in allowed_doc_ids:
            raise PermissionError(
                f"Cross-split packing violation: doc_id '{rec.doc_id}' is not authorized for split '{split_name}'"
            )
        if rec.doc_id in seen_ids:
            raise ValueError(f"Duplicate doc_id '{rec.doc_id}' encountered while packing split '{split_name}'")
        seen_ids.add(rec.doc_id)

        doc_tokens = tokenizer.encode(rec.text, add_bos=False, add_eos=True)
        if not doc_tokens or doc_tokens[-1] != tokenizer.eos_token_id:
            raise AssertionError(f"Document '{rec.doc_id}' did not terminate with EOS_TOKEN_ID")

        start_offset = len(all_tokens)
        all_tokens.extend(doc_tokens)
        end_offset = len(all_tokens)

        spans.append(
            DocumentTokenSpan(
                doc_id=rec.doc_id,
                split_name=split_name,
                start_offset=start_offset,
                end_offset=end_offset,
                token_count=len(doc_tokens),
                ends_with_eos=True,
            )
        )

    arr = np.asarray(all_tokens, dtype=np.uint16)
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{split_name}.bin"
    tmp_bin = output_dir / f"{split_name}.bin.tmp"
    raw_bytes = arr.tobytes()
    tmp_bin.write_bytes(raw_bytes)
    tmp_bin.replace(bin_path)

    bin_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    index = TokenShardIndex(
        schema_version="1.0.0",
        split_name=split_name,
        dataset_id=dataset_id,
        split_file_sha256=split_file_sha256,
        tokenizer_hash=tokenizer.tokenizer_hash(),
        vocab_size=tokenizer.vocab_size,
        total_tokens=int(arr.size),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        shard_bin_sha256=bin_sha256,
        document_spans=spans,
    )
    index.save(output_dir / f"{split_name}.idx.json")
    return index


class PackedSplitShard:
    """Memory-mapped reader over a single split's `.bin` and `.idx.json` files."""

    def __init__(self, shards_dir: Path, split_name: str) -> None:
        self.shards_dir = shards_dir
        self.split_name = split_name
        self.index_path = shards_dir / f"{split_name}.idx.json"
        self.bin_path = shards_dir / f"{split_name}.bin"

        self.index = TokenShardIndex.load(self.index_path)
        if self.index.split_name != split_name:
            raise ValueError(f"Shard split mismatch: index declares {self.index.split_name}, expected {split_name}")

        raw_bytes = self.bin_path.read_bytes()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha != self.index.shard_bin_sha256:
            raise ValueError(
                f"Corrupted shard binary {self.bin_path}: expected SHA-256 {self.index.shard_bin_sha256}, got {actual_sha}"
            )

        self._mmap = np.memmap(self.bin_path, dtype=np.uint16, mode="r", shape=(self.index.total_tokens,))

    @property
    def total_tokens(self) -> int:
        return self.index.total_tokens

    def num_windows(self, seq_len: int, include_padded_tail: bool = True) -> int:
        """
        Compute the number of non-overlapping windows of input length `seq_len`.
        Each full window consumes `seq_len` step positions (`[offset : offset + seq_len + 1]`).
        """
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if self.total_tokens < 2:
            raise ValueError("Shard must contain at least 2 tokens")
        full_windows = (self.total_tokens - 1) // seq_len
        remainder = (self.total_tokens - 1) % seq_len
        if include_padded_tail and remainder > 0:
            return full_windows + 1
        return max(1, full_windows)

    def trace_source_documents(self, start_offset: int, end_offset: int) -> list[str]:
        """Return all source `doc_id`s whose token spans overlap `[start_offset, end_offset)`."""
        matched: list[str] = []
        for span in self.index.document_spans:
            if span.end_offset <= start_offset:
                continue
            if span.start_offset >= end_offset:
                break
            matched.append(span.doc_id)
        return matched

    def get_batch(self, window_indices: Sequence[int], seq_len: int, device: str | torch.device = "cpu") -> PackedBatch:
        if not window_indices:
            raise ValueError("window_indices cannot be empty")
        bsz = len(window_indices)
        inputs = torch.full((bsz, seq_len), self.index.pad_token_id, dtype=torch.long)
        targets = torch.full((bsz, seq_len), IGNORE_INDEX, dtype=torch.long)
        valid_counts = torch.zeros(bsz, dtype=torch.long)
        source_docs_per_row: list[list[str]] = []
        eos_positions_per_row: list[list[int]] = []

        for row_idx, win_idx in enumerate(window_indices):
            start = win_idx * seq_len
            end = min(start + seq_len + 1, self.total_tokens)
            slice_tokens = np.asarray(self._mmap[start:end], dtype=np.int64)
            if slice_tokens.size < 2:
                raise IndexError(f"Window index {win_idx} has fewer than 2 tokens at start={start}")

            n_valid = int(slice_tokens.size - 1)
            inputs[row_idx, :n_valid] = torch.from_numpy(slice_tokens[:-1])
            targets[row_idx, :n_valid] = torch.from_numpy(slice_tokens[1:])
            valid_counts[row_idx] = n_valid

            source_docs_per_row.append(self.trace_source_documents(start, end))
            eos_locs = [
                int(pos)
                for pos in range(n_valid)
                if int(slice_tokens[pos]) == self.index.eos_token_id
            ]
            eos_positions_per_row.append(eos_locs)

        return PackedBatch(
            split_name=self.split_name,
            input_ids=inputs.to(device),
            targets=targets.to(device),
            valid_target_counts=valid_counts.to(device),
            total_valid_targets=int(valid_counts.sum().item()),
            window_indices=list(window_indices),
            source_doc_ids_per_row=source_docs_per_row,
            eos_positions_per_row=eos_positions_per_row,
        )


class DeterministicShardIterator:
    """
    Stateful, resumable iterator over a `PackedSplitShard` (Day036 & Day038).
    Supports deterministic per-epoch window shuffling and exact checkpoint state serialization.
    """

    def __init__(
        self,
        shard: PackedSplitShard,
        *,
        seq_len: int,
        batch_size: int,
        seed: int = 1337,
        shuffle: bool = True,
        include_padded_tail: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.shard = shard
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.include_padded_tail = include_padded_tail
        self.device = device

        self.epoch: int = 0
        self.cursor: int = 0
        self._window_order: list[int] = self._build_epoch_order(self.epoch)

    def _build_epoch_order(self, epoch: int) -> list[int]:
        n_win = self.shard.num_windows(self.seq_len, include_padded_tail=self.include_padded_tail)
        order = list(range(n_win))
        if self.shuffle:
            rng = random.Random(self.seed + epoch * 100_003)
            rng.shuffle(order)
        return order

    def next_batch(self) -> PackedBatch:
        selected: list[int] = []
        while len(selected) < self.batch_size:
            if self.cursor >= len(self._window_order):
                self.epoch += 1
                self.cursor = 0
                self._window_order = self._build_epoch_order(self.epoch)
            selected.append(self._window_order[self.cursor])
            self.cursor += 1
        return self.shard.get_batch(selected, self.seq_len, device=self.device)

    def state_dict(self) -> dict:
        return {
            "split_name": self.shard.split_name,
            "shard_bin_sha256": self.shard.index.shard_bin_sha256,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "include_padded_tail": self.include_padded_tail,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "window_order": list(self._window_order),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("shard_bin_sha256") != self.shard.index.shard_bin_sha256:
            raise ValueError("Shard binary hash mismatch when restoring DeterministicShardIterator")
        if state.get("seq_len") != self.seq_len or state.get("batch_size") != self.batch_size:
            raise ValueError("Iterator seq_len or batch_size mismatch on restore")
        self.epoch = int(state["epoch"])
        self.cursor = int(state["cursor"])
        self._window_order = [int(x) for x in state["window_order"]]
