"""Day033: Deterministic document-level split assignment, manifest freezing, and zero-leakage guards."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from scratchlm.data.ingestion import CleanedCorpusRecord, compute_canonical_dedup_hash, normalize_document_text


@dataclass(frozen=True)
class SplitFileMetadata:
    """Metadata and checksum for a single split JSONL file."""

    split_name: str
    record_count: int
    utf8_bytes: int
    file_sha256: str
    doc_ids: list[str]
    exact_hashes: list[str]
    canonical_dedup_hashes: list[str]


@dataclass(frozen=True)
class SplitsManifest:
    """Frozen split manifest proving train/val/test disjointness and sealing held-out test data."""

    schema_version: str
    dataset_id: str
    corpus_sha256: str
    split_seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    test_split_sealed: bool
    splits: dict[str, SplitFileMetadata]

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        content = self.canonical_json()
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: Path) -> SplitsManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        splits = {k: SplitFileMetadata(**v) for k, v in raw["splits"].items()}
        return cls(
            schema_version=raw["schema_version"],
            dataset_id=raw["dataset_id"],
            corpus_sha256=raw["corpus_sha256"],
            split_seed=raw["split_seed"],
            train_ratio=raw["train_ratio"],
            val_ratio=raw["val_ratio"],
            test_ratio=raw["test_ratio"],
            test_split_sealed=raw["test_split_sealed"],
            splits=splits,
        )


def verify_split_disjointness(
    train_records: Iterable[CleanedCorpusRecord],
    val_records: Iterable[CleanedCorpusRecord],
    test_records: Iterable[CleanedCorpusRecord],
) -> None:
    """
    Verify zero document-ID, exact-hash, or normalized-canonical-hash overlap across
    train, validation, and test splits (Day033).
    """
    by_split = {
        "train": list(train_records),
        "val": list(val_records),
        "test": list(test_records),
    }

    seen_ids: dict[str, str] = {}
    seen_exact: dict[str, str] = {}
    seen_canonical: dict[str, str] = {}

    for split_name, records in by_split.items():
        if not records:
            raise ValueError(f"Split '{split_name}' cannot be empty")
        for rec in records:
            norm = normalize_document_text(rec.text)
            exact_h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            canon_h = compute_canonical_dedup_hash(norm)

            if rec.doc_id in seen_ids:
                raise ValueError(
                    f"Split leakage detected: doc_id '{rec.doc_id}' in both '{seen_ids[rec.doc_id]}' and '{split_name}'"
                )
            if exact_h in seen_exact:
                raise ValueError(
                    f"Split leakage detected: exact text hash for '{rec.doc_id}' in '{split_name}' "
                    f"matches '{seen_exact[exact_h]}'"
                )
            if canon_h in seen_canonical:
                raise ValueError(
                    f"Split leakage detected: normalized near-duplicate for '{rec.doc_id}' in '{split_name}' "
                    f"matches '{seen_canonical[canon_h]}'"
                )

            seen_ids[rec.doc_id] = split_name
            seen_exact[exact_h] = f"{split_name}:{rec.doc_id}"
            seen_canonical[canon_h] = f"{split_name}:{rec.doc_id}"


def assign_and_freeze_splits(
    records: list[CleanedCorpusRecord],
    dataset_id: str,
    corpus_sha256: str,
    output_dir: Path,
    split_seed: int = 20261016,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> tuple[dict[str, list[CleanedCorpusRecord]], SplitsManifest]:
    """
    Deterministically partition deduplicated records into train/val/test at the document level,
    verify disjointness, write JSONL split files atomically, and emit `splits_manifest.json`.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    if len(records) < 10:
        raise ValueError("At least 10 records required to form non-empty train/val/test splits")

    sorted_records = sorted(records, key=lambda r: r.doc_id)
    rng = random.Random(split_seed)
    shuffled = list(sorted_records)
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = max(1, int(round(total * train_ratio)))
    n_val = max(1, int(round(total * val_ratio)))
    n_test = total - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, total - n_val - n_test)

    train_split = sorted(shuffled[:n_train], key=lambda r: r.doc_id)
    val_split = sorted(shuffled[n_train : n_train + n_val], key=lambda r: r.doc_id)
    test_split = sorted(shuffled[n_train + n_val :], key=lambda r: r.doc_id)

    verify_split_disjointness(train_split, val_split, test_split)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_map = {
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }
    split_meta: dict[str, SplitFileMetadata] = {}

    for split_name, split_records in split_map.items():
        jsonl_path = output_dir / f"{split_name}.jsonl"
        lines = [
            json.dumps(asdict(rec), sort_keys=True, ensure_ascii=False)
            for rec in split_records
        ]
        payload = "\n".join(lines) + "\n"
        tmp_path = jsonl_path.with_suffix(".jsonl.tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(jsonl_path)

        file_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        split_meta[split_name] = SplitFileMetadata(
            split_name=split_name,
            record_count=len(split_records),
            utf8_bytes=sum(r.utf8_bytes for r in split_records),
            file_sha256=file_sha,
            doc_ids=[r.doc_id for r in split_records],
            exact_hashes=[r.exact_sha256 for r in split_records],
            canonical_dedup_hashes=[r.canonical_dedup_hash for r in split_records],
        )

    manifest = SplitsManifest(
        schema_version="1.0.0",
        dataset_id=dataset_id,
        corpus_sha256=corpus_sha256,
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        test_split_sealed=True,
        splits=split_meta,
    )
    manifest.save(output_dir / "splits_manifest.json")
    return split_map, manifest


def load_split_records(splits_dir: Path, split_name: str, allow_sealed_test: bool = False) -> list[CleanedCorpusRecord]:
    """
    Load a frozen split JSONL file and verify its SHA-256 against `splits_manifest.json`.
    Blocks loading the sealed `test` split unless `allow_sealed_test=True` is explicitly passed.
    """
    manifest = SplitsManifest.load(splits_dir / "splits_manifest.json")
    if split_name not in manifest.splits:
        raise ValueError(f"Unknown split '{split_name}'")
    if split_name == "test" and manifest.test_split_sealed and not allow_sealed_test:
        raise PermissionError(
            "The held-out 'test' split is sealed during tokenizer training, pretraining, and hyperparameter tuning."
        )

    meta = manifest.splits[split_name]
    jsonl_path = splits_dir / f"{split_name}.jsonl"
    raw_content = jsonl_path.read_text(encoding="utf-8")
    actual_sha = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    if actual_sha != meta.file_sha256:
        raise ValueError(
            f"Integrity check failed for split '{split_name}': expected SHA-256 {meta.file_sha256}, got {actual_sha}"
        )

    records: list[CleanedCorpusRecord] = []
    for line in raw_content.splitlines():
        if line.strip():
            records.append(CleanedCorpusRecord(**json.loads(line)))
    return records


def assert_train_only_texts(
    texts_or_records: Iterable[str | CleanedCorpusRecord],
    splits_manifest: SplitsManifest,
) -> list[str]:
    """
    Guardrail for Day034 BPE tokenizer training: verifies that every input document belongs
    exclusively to the 'train' split and does not match any validation or test document ID,
    exact hash, or normalized canonical hash.
    """
    forbidden_doc_ids = set(splits_manifest.splits["val"].doc_ids) | set(splits_manifest.splits["test"].doc_ids)
    forbidden_exact = set(splits_manifest.splits["val"].exact_hashes) | set(splits_manifest.splits["test"].exact_hashes)
    forbidden_canonical = set(splits_manifest.splits["val"].canonical_dedup_hashes) | set(
        splits_manifest.splits["test"].canonical_dedup_hashes
    )

    verified_texts: list[str] = []
    for item in texts_or_records:
        if isinstance(item, CleanedCorpusRecord):
            if item.doc_id in forbidden_doc_ids:
                raise PermissionError(
                    f"Data leakage blocked: doc_id '{item.doc_id}' belongs to a held-out evaluation split"
                )
            text = item.text
        else:
            text = str(item)

        norm = normalize_document_text(text)
        exact_h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        canon_h = compute_canonical_dedup_hash(norm)
        if exact_h in forbidden_exact or canon_h in forbidden_canonical:
            raise PermissionError(
                "Data leakage blocked: input text matches a held-out validation or test document"
            )
        verified_texts.append(norm)

    if not verified_texts:
        raise ValueError("Cannot train tokenizer on an empty training corpus")
    return verified_texts
