"""
Day034-Day035: Educational byte-level BPE tokenizer with deterministic pair counting,
lexicographical tie-breaking on (token_id_a, token_id_b), train-only split leakage guard,
special-token handling (256..263), stateful UTF-8 streaming decoder, and artifact persistence.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from scratchlm.data.ingestion import CleanedCorpusRecord
from scratchlm.data.splits import SplitsManifest, assert_train_only_texts
from scratchlm.tokenizer.byte_tokenizer import (
    ASSISTANT_TOKEN_ID,
    BOS_TOKEN_ID,
    BYTE_VOCAB_SIZE,
    CONTEXT_TOKEN_ID,
    END_TURN_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    ROLE_TO_SPECIAL_ID,
    SPECIAL_TOKENS,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    ByteTokenizer,
)

# Pre-tokenization regex that keeps leading spaces attached to words while keeping runs of newlines/spaces lossless
_PRETOKENIZE_RE = re.compile(r" ?[^\s\.,!?;:\"'()\[\]{}]+| ?[\.,!?;:\"'()\[\]{}]+|\s+")


def count_adjacent_pairs(sequences: Sequence[Sequence[int]]) -> dict[tuple[int, int], int]:
    """Count adjacent token pairs across pre-tokenized byte/token sequences."""
    counts: dict[tuple[int, int], int] = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def select_best_pair(pair_counts: dict[tuple[int, int], int], min_frequency: int = 2) -> tuple[tuple[int, int], int] | None:
    """
    Deterministic merge selection (Day034):
    Choose the pair with highest frequency >= min_frequency.
    Tie-breaking rule: lexicographically smallest (token_id_a, token_id_b).
    """
    best_pair: tuple[int, int] | None = None
    best_count = -1
    for pair, count in pair_counts.items():
        if count < min_frequency:
            continue
        if count > best_count or (count == best_count and best_pair is not None and pair < best_pair):
            best_pair = pair
            best_count = count
    if best_pair is None:
        return None
    return best_pair, best_count


def apply_merge_to_sequence(seq: Sequence[int], pair: tuple[int, int], new_token_id: int) -> list[int]:
    """Replace non-overlapping occurrences of `pair` left-to-right with `new_token_id`."""
    n = len(seq)
    if n < 2:
        return list(seq)
    out: list[int] = []
    i = 0
    a, b = pair
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_token_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


@dataclass(frozen=True)
class BPEMergeRule:
    """Single learned BPE merge step."""

    rank: int
    token_id_a: int
    token_id_b: int
    new_token_id: int
    training_frequency: int
    merged_bytes_hex: str


@dataclass(frozen=True)
class TokenizerCompressionStats:
    """Comparison metrics between ByteTokenizer (264) and BPETokenizer on a sample."""

    sample_documents: int
    total_utf8_bytes: int
    byte_token_count: int
    bpe_token_count: int
    bytes_per_bpe_token: float
    token_reduction_ratio: float
    encode_time_ms: float
    decode_time_ms: float


class BpeIncrementalUtf8StreamDecoder:
    """Stateful UTF-8 streaming decoder supporting multi-byte BPE tokens and raw byte fallback."""

    def __init__(self, token_id_to_bytes: dict[int, bytes], vocab_size: int) -> None:
        self._token_id_to_bytes = token_id_to_bytes
        self._vocab_size = vocab_size
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def step(self, token_id: int) -> str:
        if token_id < 0 or token_id >= self._vocab_size:
            raise ValueError(f"Token ID {token_id} out of range [0, {self._vocab_size})")
        if 256 <= token_id < BYTE_VOCAB_SIZE:
            return ""
        raw_bytes = self._token_id_to_bytes[token_id]
        return self._decoder.decode(raw_bytes, final=False)

    def flush(self) -> str:
        return self._decoder.decode(b"", final=True)

    def reset(self) -> None:
        self._decoder.reset()


class BPETokenizer:
    """
    Educational deterministic byte-level BPE tokenizer (Day034-Day035).
    - IDs 0..255: raw UTF-8 bytes
    - IDs 256..263: reserved structural special tokens (<|pad|>, <|eos|>, <|bos|>, role markers)
    - IDs 264..vocab_size-1: learned byte-pair merges in rank order
    """

    pad_token_id: int = PAD_TOKEN_ID
    eos_token_id: int = EOS_TOKEN_ID
    bos_token_id: int = BOS_TOKEN_ID

    def __init__(
        self,
        merges: Sequence[BPEMergeRule],
        *,
        training_dataset_id: str = "scratch-curated-tinystories-v1",
        training_split_sha256: str = "",
    ) -> None:
        self.merges: list[BPEMergeRule] = list(merges)
        self.training_dataset_id = training_dataset_id
        self.training_split_sha256 = training_split_sha256
        self.vocab_size: int = BYTE_VOCAB_SIZE + len(self.merges)

        self._token_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._pair_to_merge: dict[tuple[int, int], BPEMergeRule] = {}

        for idx, rule in enumerate(self.merges):
            expected_id = BYTE_VOCAB_SIZE + idx
            if rule.new_token_id != expected_id:
                raise ValueError(f"Merge rank {idx} expected new_token_id={expected_id}, got {rule.new_token_id}")
            if rule.token_id_a not in self._token_to_bytes or rule.token_id_b not in self._token_to_bytes:
                raise ValueError(f"Merge rank {idx} references unknown constituent token IDs")
            merged_bytes = self._token_to_bytes[rule.token_id_a] + self._token_to_bytes[rule.token_id_b]
            if merged_bytes.hex() != rule.merged_bytes_hex:
                raise ValueError(f"Merge rank {idx} hex mismatch: {merged_bytes.hex()} != {rule.merged_bytes_hex}")
            self._token_to_bytes[rule.new_token_id] = merged_bytes
            self._pair_to_merge[(rule.token_id_a, rule.token_id_b)] = rule

    @classmethod
    def train_from_records(
        cls,
        train_records: Iterable[str | CleanedCorpusRecord],
        target_vocab_size: int = 512,
        *,
        min_frequency: int = 2,
        splits_manifest: SplitsManifest | None = None,
    ) -> BPETokenizer:
        """
        Learn deterministic byte-level BPE merges strictly from training split records.
        If `splits_manifest` is supplied, validates that zero validation/test records are present.
        """
        if target_vocab_size < BYTE_VOCAB_SIZE:
            raise ValueError(f"target_vocab_size must be >= {BYTE_VOCAB_SIZE}")

        if splits_manifest is not None:
            texts = assert_train_only_texts(train_records, splits_manifest)
            dataset_id = splits_manifest.dataset_id
            train_sha = splits_manifest.splits["train"].file_sha256
        else:
            texts = [r.text if isinstance(r, CleanedCorpusRecord) else str(r) for r in train_records]
            if not texts:
                raise ValueError("Cannot train BPE tokenizer on empty input")
            dataset_id = "unverified-fixture"
            train_sha = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()

        # Pre-tokenize into chunks and weight identical chunks by count for fast deterministic learning
        chunk_freq: dict[tuple[int, ...], int] = {}
        for text in texts:
            for match in _PRETOKENIZE_RE.finditer(text):
                chunk_bytes = tuple(match.group(0).encode("utf-8"))
                if len(chunk_bytes) >= 2:
                    chunk_freq[chunk_bytes] = chunk_freq.get(chunk_bytes, 0) + 1

        # Maintain sorted list of [sequence, count] for deterministic pair accumulation
        active_chunks: list[tuple[list[int], int]] = [
            (list(seq), freq) for seq, freq in sorted(chunk_freq.items(), key=lambda kv: kv[0])
        ]

        token_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        learned_merges: list[BPEMergeRule] = []
        max_merges = target_vocab_size - BYTE_VOCAB_SIZE

        for rank in range(max_merges):
            pair_counts: dict[tuple[int, int], int] = {}
            for seq, freq in active_chunks:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq

            selected = select_best_pair(pair_counts, min_frequency=min_frequency)
            if selected is None:
                break

            best_pair, best_freq = selected
            new_id = BYTE_VOCAB_SIZE + rank
            merged_bytes = token_to_bytes[best_pair[0]] + token_to_bytes[best_pair[1]]
            token_to_bytes[new_id] = merged_bytes

            learned_merges.append(
                BPEMergeRule(
                    rank=rank,
                    token_id_a=best_pair[0],
                    token_id_b=best_pair[1],
                    new_token_id=new_id,
                    training_frequency=best_freq,
                    merged_bytes_hex=merged_bytes.hex(),
                )
            )

            active_chunks = [
                (apply_merge_to_sequence(seq, best_pair, new_id), freq)
                for seq, freq in active_chunks
            ]

        return cls(
            learned_merges,
            training_dataset_id=dataset_id,
            training_split_sha256=train_sha,
        )

    def _encode_chunk_bytes(self, raw_bytes: bytes) -> list[int]:
        seq = list(raw_bytes)
        if len(seq) < 2 or not self._pair_to_merge:
            return seq

        while len(seq) >= 2:
            # Find the lowest-rank applicable merge among adjacent pairs in `seq`
            best_rule: BPEMergeRule | None = None
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                rule = self._pair_to_merge.get(pair)
                if rule is not None and (best_rule is None or rule.rank < best_rule.rank):
                    best_rule = rule
            if best_rule is None:
                break
            seq = apply_merge_to_sequence(
                seq,
                (best_rule.token_id_a, best_rule.token_id_b),
                best_rule.new_token_id,
            )
        return seq

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        Encode text into BPE token IDs. Literal special-token strings in user text are encoded
        as ordinary UTF-8 bytes/subwords so untrusted user input cannot inject role markers.
        """
        ids: list[int] = []
        if add_bos:
            ids.append(BOS_TOKEN_ID)
        if text:
            for match in _PRETOKENIZE_RE.finditer(text):
                ids.extend(self._encode_chunk_bytes(match.group(0).encode("utf-8")))
        if add_eos:
            ids.append(EOS_TOKEN_ID)
        return ids

    def encode_structural_turn(self, role: str, content: str, *, append_eos: bool = False) -> list[int]:
        if role not in ROLE_TO_SPECIAL_ID:
            raise ValueError(f"Unsupported structural role '{role}'")
        ids = [ROLE_TO_SPECIAL_ID[role]]
        ids.extend(self.encode(content))
        ids.append(END_TURN_TOKEN_ID)
        if append_eos:
            ids.append(EOS_TOKEN_ID)
        return ids

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        special_id_to_str = {v: k for k, v in SPECIAL_TOKENS.items()}
        chunks: list[str] = []
        byte_buf = bytearray()

        for tid in token_ids:
            t_int = int(tid)
            if t_int < 0 or t_int >= self.vocab_size:
                raise ValueError(f"Token ID {t_int} out of range [0, {self.vocab_size})")
            if 256 <= t_int < BYTE_VOCAB_SIZE:
                if byte_buf:
                    chunks.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                    byte_buf.clear()
                if not skip_special_tokens:
                    chunks.append(special_id_to_str[t_int])
            else:
                byte_buf.extend(self._token_to_bytes[t_int])

        if byte_buf:
            chunks.append(bytes(byte_buf).decode("utf-8", errors="replace"))
        return "".join(chunks)

    def new_stream_decoder(self) -> BpeIncrementalUtf8StreamDecoder:
        return BpeIncrementalUtf8StreamDecoder(self._token_to_bytes, self.vocab_size)

    def create_stream_decoder(self) -> BpeIncrementalUtf8StreamDecoder:
        return self.new_stream_decoder()

    def to_dict(self) -> dict:
        return {
            "schema_version": "1.0.0",
            "tokenizer_type": "byte_level_bpe",
            "base_byte_vocab_size": 256,
            "special_tokens": SPECIAL_TOKENS,
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "tie_breaking_rule": "lexicographical_min_token_id_pair",
            "training_dataset_id": self.training_dataset_id,
            "training_split_sha256": self.training_split_sha256,
            "merges": [asdict(m) for m in self.merges],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def tokenizer_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.canonical_json()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: Path, expected_sha256: str | None = None) -> BPETokenizer:
        raw_text = path.read_text(encoding="utf-8")
        actual_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        if expected_sha256 is not None and actual_sha != expected_sha256:
            raise ValueError(
                f"Tokenizer hash mismatch for {path}: expected {expected_sha256}, got {actual_sha}"
            )
        data = json.loads(raw_text)
        if data.get("tokenizer_type") != "byte_level_bpe":
            raise ValueError(f"Unsupported tokenizer_type: {data.get('tokenizer_type')}")
        merges = [BPEMergeRule(**m) for m in data["merges"]]
        return cls(
            merges,
            training_dataset_id=data.get("training_dataset_id", ""),
            training_split_sha256=data.get("training_split_sha256", ""),
        )

    def compare_compression(self, sample_texts: Sequence[str]) -> TokenizerCompressionStats:
        byte_tok = ByteTokenizer()
        total_bytes = sum(len(t.encode("utf-8")) for t in sample_texts)
        byte_count = sum(len(byte_tok.encode(t)) for t in sample_texts)

        t0 = time.perf_counter()
        encoded_lists = [self.encode(t) for t in sample_texts]
        t1 = time.perf_counter()
        decoded_texts = [self.decode(ids) for ids in encoded_lists]
        t2 = time.perf_counter()

        for orig, dec in zip(sample_texts, decoded_texts):
            if orig != dec:
                raise AssertionError("Lossless round-trip failed during compression evaluation")

        bpe_count = sum(len(ids) for ids in encoded_lists)
        bytes_per_token = total_bytes / max(1, bpe_count)
        reduction_ratio = bpe_count / max(1, byte_count)
        return TokenizerCompressionStats(
            sample_documents=len(sample_texts),
            total_utf8_bytes=total_bytes,
            byte_token_count=byte_count,
            bpe_token_count=bpe_count,
            bytes_per_bpe_token=round(bytes_per_token, 4),
            token_reduction_ratio=round(reduction_ratio, 4),
            encode_time_ms=round((t1 - t0) * 1000.0, 3),
            decode_time_ms=round((t2 - t1) * 1000.0, 3),
        )
