"""Corpus acquisition, cleaning, deduplication, split freezing, and binary shard packing."""

from scratchlm.data.ingestion import (
    CleanedCorpusRecord,
    IngestionAuditReport,
    RawCorpusRecord,
    build_default_corpus_manifest,
    compute_canonical_dedup_hash,
    generate_curated_story_stream,
    ingest_bounded_corpus,
    normalize_document_text,
)
from scratchlm.data.manifest import CorpusManifest, StorageBudget
from scratchlm.data.shards import (
    DeterministicShardIterator,
    DocumentTokenSpan,
    PackedBatch,
    PackedSplitShard,
    TokenShardIndex,
    build_packed_split_shard,
)
from scratchlm.data.splits import (
    SplitFileMetadata,
    SplitsManifest,
    assert_train_only_texts,
    assign_and_freeze_splits,
    load_split_records,
    verify_split_disjointness,
)

__all__ = [
    "CleanedCorpusRecord",
    "CorpusManifest",
    "DeterministicShardIterator",
    "DocumentTokenSpan",
    "IngestionAuditReport",
    "PackedBatch",
    "PackedSplitShard",
    "RawCorpusRecord",
    "SplitFileMetadata",
    "SplitsManifest",
    "StorageBudget",
    "TokenShardIndex",
    "assert_train_only_texts",
    "assign_and_freeze_splits",
    "build_default_corpus_manifest",
    "build_packed_split_shard",
    "compute_canonical_dedup_hash",
    "generate_curated_story_stream",
    "ingest_bounded_corpus",
    "load_split_records",
    "normalize_document_text",
    "verify_split_disjointness",
]
