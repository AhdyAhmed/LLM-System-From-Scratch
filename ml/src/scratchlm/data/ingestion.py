"""Day032: Bounded streaming corpus ingestion, text normalization, and deduplication."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from scratchlm.data.manifest import CorpusManifest, StorageBudget

_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CANONICAL_DEDUP_RE = re.compile(r"[^a-z0-9\u0600-\u06ff]+")


@dataclass(frozen=True)
class RawCorpusRecord:
    """Raw input record read from a streaming JSONL source."""

    doc_id: str
    text: str
    source_split_hint: str = "unspecified"
    author_tag: str = "synthetic-tinystories-curated"


@dataclass(frozen=True)
class CleanedCorpusRecord:
    """Normalized, deduplicated document ready for split assignment."""

    doc_id: str
    text: str
    utf8_bytes: int
    exact_sha256: str
    canonical_dedup_hash: str
    author_tag: str


@dataclass(frozen=True)
class IngestionAuditReport:
    """Privacy-safe ingestion summary that logs document IDs and reasons, never full texts."""

    raw_records_seen: int
    accepted_records: int
    rejected_records: int
    duplicate_records: int
    accepted_utf8_bytes: int
    stopped_by_cap: bool
    stop_reason: str | None
    rejected_doc_ids_by_reason: dict[str, list[str]]
    duplicate_doc_ids: list[dict[str, str]]
    corpus_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_document_text(text: str) -> str:
    """
    Deterministic text normalization (Day032):
    1. Unicode NFC normalization.
    2. Normalize CRLF/CR line endings to LF.
    3. Strip non-printable ASCII/Unicode control characters (except newline and tab).
    4. Collapse runs of horizontal whitespace and cap consecutive newlines at two.
    5. Strip leading/trailing whitespace.
    """
    norm = unicodedata.normalize("NFC", text)
    norm = norm.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_chars: list[str] = []
    for ch in norm:
        if ch == "\n" or ch == "\t":
            cleaned_chars.append(ch)
        elif unicodedata.category(ch).startswith("C"):
            continue
        else:
            cleaned_chars.append(ch)
    collapsed = "".join(cleaned_chars)
    collapsed = "\n".join(_WHITESPACE_RE.sub(" ", line).strip() for line in collapsed.split("\n"))
    collapsed = _MULTI_NEWLINE_RE.sub("\n\n", collapsed)
    return collapsed.strip()


def compute_canonical_dedup_hash(normalized_text: str) -> str:
    """
    Compute a normalized near-duplicate fingerprint by lowercasing, stripping punctuation
    and whitespace variations, and hashing with SHA-256.
    """
    lowered = unicodedata.normalize("NFKC", normalized_text).casefold()
    canonical = _CANONICAL_DEDUP_RE.sub("", lowered)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ingest_bounded_corpus(
    raw_stream: Iterable[RawCorpusRecord],
    budget: StorageBudget,
) -> tuple[list[CleanedCorpusRecord], IngestionAuditReport]:
    """
    Stream records through normalization, validation, and exact/canonical deduplication
    while enforcing byte and record caps. Never logs raw text in rejection or duplicate records.
    """
    budget.validate()
    accepted: list[CleanedCorpusRecord] = []
    seen_doc_ids: set[str] = set()
    seen_exact_hashes: dict[str, str] = {}
    seen_canonical_hashes: dict[str, str] = {}

    rejected_by_reason: dict[str, list[str]] = {
        "empty_or_too_short": [],
        "exceeds_max_record_chars": [],
        "duplicate_doc_id": [],
        "private_or_disallowed_source": [],
    }
    duplicate_pairs: list[dict[str, str]] = []

    raw_seen = 0
    accepted_bytes = 0
    raw_bytes_seen = 0
    stopped_by_cap = False
    stop_reason: str | None = None

    for record in raw_stream:
        record_raw_bytes = len(record.text.encode("utf-8"))
        if raw_bytes_seen + record_raw_bytes > budget.max_raw_bytes:
            stopped_by_cap = True
            stop_reason = "max_raw_bytes_reached"
            break
        if len(accepted) >= budget.max_records:
            stopped_by_cap = True
            stop_reason = "max_records_reached"
            break

        raw_seen += 1
        raw_bytes_seen += record_raw_bytes

        if record.author_tag in {"private_chat", "user_upload"}:
            rejected_by_reason["private_or_disallowed_source"].append(record.doc_id)
            continue

        if record.doc_id in seen_doc_ids:
            rejected_by_reason["duplicate_doc_id"].append(record.doc_id)
            continue
        seen_doc_ids.add(record.doc_id)

        cleaned_text = normalize_document_text(record.text)
        if len(cleaned_text) < budget.min_record_chars:
            rejected_by_reason["empty_or_too_short"].append(record.doc_id)
            continue
        if len(cleaned_text) > budget.max_record_chars:
            rejected_by_reason["exceeds_max_record_chars"].append(record.doc_id)
            continue

        exact_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        canonical_hash = compute_canonical_dedup_hash(cleaned_text)

        if exact_hash in seen_exact_hashes:
            duplicate_pairs.append(
                {
                    "duplicate_doc_id": record.doc_id,
                    "kept_doc_id": seen_exact_hashes[exact_hash],
                    "match_type": "exact",
                }
            )
            continue

        if canonical_hash in seen_canonical_hashes:
            duplicate_pairs.append(
                {
                    "duplicate_doc_id": record.doc_id,
                    "kept_doc_id": seen_canonical_hashes[canonical_hash],
                    "match_type": "canonical_normalized",
                }
            )
            continue

        cleaned_utf8_bytes = len(cleaned_text.encode("utf-8"))
        if accepted_bytes + cleaned_utf8_bytes > budget.max_processed_bytes:
            stopped_by_cap = True
            stop_reason = "max_processed_bytes_reached"
            raw_seen -= 1
            break

        seen_exact_hashes[exact_hash] = record.doc_id
        seen_canonical_hashes[canonical_hash] = record.doc_id
        accepted_bytes += cleaned_utf8_bytes

        accepted.append(
            CleanedCorpusRecord(
                doc_id=record.doc_id,
                text=cleaned_text,
                utf8_bytes=cleaned_utf8_bytes,
                exact_sha256=exact_hash,
                canonical_dedup_hash=canonical_hash,
                author_tag=record.author_tag,
            )
        )

    corpus_hasher = hashlib.sha256()
    for item in accepted:
        corpus_hasher.update(f"{item.doc_id}:{item.exact_sha256}\n".encode("utf-8"))

    total_rejected = sum(len(v) for v in rejected_by_reason.values())
    report = IngestionAuditReport(
        raw_records_seen=raw_seen,
        accepted_records=len(accepted),
        rejected_records=total_rejected,
        duplicate_records=len(duplicate_pairs),
        accepted_utf8_bytes=accepted_bytes,
        stopped_by_cap=stopped_by_cap,
        stop_reason=stop_reason,
        rejected_doc_ids_by_reason=rejected_by_reason,
        duplicate_doc_ids=duplicate_pairs,
        corpus_sha256=corpus_hasher.hexdigest(),
    )
    return accepted, report


def generate_curated_story_stream() -> Iterator[RawCorpusRecord]:
    """
    Deterministic, self-authored TinyStories-inspired synthetic corpus stream (120 unique stories
    plus deliberate exact duplicates, normalized near-duplicates, short/empty records, and oversized
    records) so offline ingestion, deduplication, split freezing, BPE training, and training loop
    tests run without external network dependencies.
    """
    protagonists = [
        ("lily", "a curious little rabbit", "the sunny meadow"),
        ("omar", "a gentle clockmaker's apprentice", "the quiet workshop"),
        ("mira", "a brave paper boat captain", "the silver brook"),
        ("tariq", "a thoughtful young astronomer", "the rooftop lantern tower"),
        ("nora", "a cheerful baker's helper", "the warm village kitchen"),
        ("zayd", "a clever wooden robot", "the tidy study room"),
    ]
    objects = [
        ("a brass compass", "pointed toward lost marbles", "returned every marble to the jar"),
        ("a blue notebook", "held seven neat riddles", "shared the answers with a friend"),
        ("a tiny copper bell", "rang softly in the breeze", "guided the kittens home before dusk"),
        ("a folded star map", "glowed beside the candle", "traced three bright constellations"),
        ("a basket of warm rolls", "smelled of cinnamon and honey", "fed the hungry travelers at noon"),
    ]
    lessons = [
        ("patience turns small steps into steady progress", "smiled and closed thewooden latch"),
        ("kindness makes every heavy basket lighter", "waved goodbye under the lantern light"),
        ("careful counting prevents mistakes in the workshop", "wrote the final number on the slate"),
        ("listening closely helps friends solve hard puzzles", "placed the bookmark on page four"),
    ]

    idx = 0
    for p_name, p_desc, p_place in protagonists:
        for obj_name, obj_prop, obj_action in objects:
            for lesson_text, ending_action in lessons:
                idx += 1
                doc_id = f"story-{idx:04d}"
                story = (
                    f"Once upon a time in {p_place}, {p_name.capitalize()} was {p_desc}. "
                    f"Every morning, {p_name.capitalize()} carried {obj_name} that {obj_prop}. "
                    f"When the afternoon clouds gathered, {p_name.capitalize()} {obj_action}. "
                    f"By evening, everyone remembered that {lesson_text}, and {p_name.capitalize()} {ending_action}."
                )
                yield RawCorpusRecord(doc_id=doc_id, text=story)

    # Deliberate exact duplicate of story-0001 with extra whitespace and control chars
    yield RawCorpusRecord(
        doc_id="story-dup-exact-0001",
        text=(
            "  Once upon a time in   the sunny meadow, Lily was a curious little rabbit.\x00 "
            "Every morning, Lily carried a brass compass that pointed toward lost marbles. "
            "When the afternoon clouds gathered, Lily returned every marble to the jar. "
            "By evening, everyone remembered that patience turns small steps into steady progress, "
            "and Lily smiled and closed thewooden latch.  \r\n"
        ),
    )

    # Deliberate canonical/punctuation near-duplicate of story-0002
    yield RawCorpusRecord(
        doc_id="story-dup-canonical-0002",
        text=(
            "ONCE UPON A TIME, in the sunny meadow; Lily was a curious little rabbit!! "
            "Every morning: Lily carried a brass compass that pointed toward lost marbles... "
            "When the afternoon clouds gathered, Lily returned every marble to the jar. "
            "By evening, everyone remembered that kindness makes every heavy basket lighter, "
            "and Lily waved goodbye under the lantern light!"
        ),
    )

    # Deliberate rejected records (too short, private source, oversized)
    yield RawCorpusRecord(doc_id="story-reject-short-0001", text="Too short.")
    yield RawCorpusRecord(
        doc_id="story-reject-private-0001",
        text="Once upon a time a private chat message appeared that must never enter pretraining.",
        author_tag="private_chat",
    )
    yield RawCorpusRecord(
        doc_id="story-reject-oversized-0001",
        text="A" * 5000,
    )


def build_default_corpus_manifest(
    accepted: list[CleanedCorpusRecord],
    report: IngestionAuditReport,
    budget: StorageBudget,
) -> CorpusManifest:
    return CorpusManifest(
        schema_version="1.0.0",
        dataset_id="scratch-curated-tinystories-v1",
        source_name="Self-Authored Curated TinyStories-Style Benchmark & Pretraining Corpus",
        source_url="https://huggingface.co/datasets/roneneldan/TinyStories (schema/task inspiration; local curated deterministic corpus)",
        source_revision="curated-2026-10-16-r1",
        license_spdx="CDLA-Sharing-1.0",
        license_reference="https://cdla.dev/sharing-1-0/",
        redistribution_policy="Permitted with attribution and manifest provenance preservation; zero private user data.",
        subset_selection_rule="Deterministic 120-document narrative grid filtered by length [40, 2048] chars and canonical SHA-256 deduplication.",
        acquisition_timestamp_utc="2026-10-16T00:00:00Z",
        filtering_code_version="scratchlm.data.ingestion.v1",
        normalization_policy="Unicode NFC, LF line endings, control-char stripping, horizontal whitespace collapse, canonical NFKC casefolded deduplication.",
        storage_budget=budget,
        raw_record_count=report.raw_records_seen,
        accepted_record_count=report.accepted_records,
        rejected_record_count=report.rejected_records,
        duplicate_record_count=report.duplicate_records,
        accepted_utf8_bytes=report.accepted_utf8_bytes,
        corpus_sha256=report.corpus_sha256,
        exclude_private_chat_and_uploads=True,
    )
