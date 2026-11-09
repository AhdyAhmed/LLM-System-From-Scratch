"""Day031: Corpus provenance, license policy, storage budgets, and manifest validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StorageBudget:
    """Explicit byte and record caps enforced before corpus ingestion or sharding."""

    max_raw_bytes: int
    max_processed_bytes: int
    max_records: int
    max_cache_bytes: int
    min_record_chars: int
    max_record_chars: int

    def validate(self) -> None:
        if self.max_raw_bytes <= 0 or self.max_raw_bytes > 2 * 1024 * 1024 * 1024:
            raise ValueError("max_raw_bytes must be in (0, 2 GiB]")
        if self.max_processed_bytes <= 0 or self.max_processed_bytes > 2 * 1024 * 1024 * 1024:
            raise ValueError("max_processed_bytes must be in (0, 2 GiB]")
        if self.max_records <= 0:
            raise ValueError("max_records must be positive")
        if self.max_cache_bytes <= 0 or self.max_cache_bytes > 2 * 1024 * 1024 * 1024:
            raise ValueError("max_cache_bytes must be in (0, 2 GiB]")
        if self.min_record_chars < 1 or self.max_record_chars < self.min_record_chars:
            raise ValueError("Invalid record character bounds")


@dataclass(frozen=True)
class CorpusManifest:
    """Immutable provenance record for a bounded pretraining corpus (Day031)."""

    schema_version: str
    dataset_id: str
    source_name: str
    source_url: str
    source_revision: str
    license_spdx: str
    license_reference: str
    redistribution_policy: str
    subset_selection_rule: str
    acquisition_timestamp_utc: str
    filtering_code_version: str
    normalization_policy: str
    storage_budget: StorageBudget
    raw_record_count: int
    accepted_record_count: int
    rejected_record_count: int
    duplicate_record_count: int
    accepted_utf8_bytes: int
    corpus_sha256: str
    exclude_private_chat_and_uploads: bool

    def validate(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError(f"Unsupported manifest schema_version: {self.schema_version}")
        if not self.dataset_id or not self.source_revision or not self.license_spdx:
            raise ValueError("dataset_id, source_revision, and license_spdx are required")
        if not self.exclude_private_chat_and_uploads:
            raise ValueError("Training corpus must explicitly exclude private chat histories and user uploads")
        self.storage_budget.validate()
        if self.accepted_record_count <= 0:
            raise ValueError("accepted_record_count must be > 0")
        if self.accepted_record_count > self.storage_budget.max_records:
            raise ValueError("accepted_record_count exceeds storage_budget.max_records")
        if self.accepted_utf8_bytes > self.storage_budget.max_processed_bytes:
            raise ValueError("accepted_utf8_bytes exceeds storage_budget.max_processed_bytes")
        if self.raw_record_count != (
            self.accepted_record_count + self.rejected_record_count + self.duplicate_record_count
        ):
            raise ValueError(
                "raw_record_count must equal accepted_record_count + rejected_record_count + duplicate_record_count"
            )
        if len(self.corpus_sha256) != 64:
            raise ValueError("corpus_sha256 must be a 64-character hex SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def save(self, path: Path) -> str:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        content = self.canonical_json()
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: Path) -> CorpusManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        budget = StorageBudget(**raw["storage_budget"])
        payload = {**raw, "storage_budget": budget}
        manifest = cls(**payload)
        manifest.validate()
        return manifest
