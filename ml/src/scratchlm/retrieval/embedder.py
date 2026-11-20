"""
Day073 & Day074: Deterministic document chunker and pinned 384-dimensional CPU embedding component.

IMPORTANT PROVENANCE NOTE:
This 384-dimensional CPU embedding component (`pretrained-cpu-hash384-v1`) is an external/separate
retrieval representation used exclusively for `pgvector` similarity search and hybrid retrieval.
It is explicitly separate from the from-scratch causal decoder-only transformer (`ScratchGPT`).
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

EMBEDDING_MODEL_REVISION = "pretrained-cpu-hash384-v1"
EMBEDDING_DIMENSION = 384
CHUNKER_VERSION = "paragraph-window-v1"

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


@dataclass(frozen=True)
class DocumentChunkSpec:
    chunk_order: int
    source_location: str
    chunk_text: str
    embedding_revision: str
    embedding: list[float]


def normalize_document_text(raw_text: str) -> str:
    """Normalize line endings and strip NUL characters while preserving paragraph structure."""
    cleaned = raw_text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return cleaned.strip()


def chunk_document_text(
    raw_text: str,
    document_revision: int = 1,
    max_chunk_chars: int = 260,
    overlap_chars: int = 40,
) -> list[DocumentChunkSpec]:
    """
    Day073: Deterministic paragraph-and-window chunker producing stable `chunk_order`,
    `source_location`, and 384-dimensional L2-normalized CPU embeddings.
    """
    normalized = normalize_document_text(raw_text)
    if not normalized:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    raw_spans: list[tuple[int, int, str]] = []
    cursor = 0

    for para in paragraphs:
        idx = normalized.find(para, cursor)
        if idx < 0:
            idx = cursor
        cursor = idx + len(para)

        if len(para) <= max_chunk_chars:
            raw_spans.append((idx, idx + len(para), para))
        else:
            step = max(40, max_chunk_chars - overlap_chars)
            pos = 0
            while pos < len(para):
                window = para[pos : pos + max_chunk_chars].strip()
                if window:
                    raw_spans.append((idx + pos, idx + min(len(para), pos + max_chunk_chars), window))
                if pos + max_chunk_chars >= len(para):
                    break
                pos += step

    # Coalesce very short heading paragraphs with the following paragraph when they fit in max_chunk_chars
    merged_spans: list[tuple[int, int, str]] = []
    i = 0
    while i < len(raw_spans):
        start_char, end_char, text_seg = raw_spans[i]
        if (
            i + 1 < len(raw_spans)
            and len(text_seg) < 60
            and len(text_seg) + 2 + len(raw_spans[i + 1][2]) <= max_chunk_chars
        ):
            next_start, next_end, next_text = raw_spans[i + 1]
            combined = f"{text_seg}\n\n{next_text}"
            merged_spans.append((start_char, next_end, combined))
            i += 2
        else:
            merged_spans.append((start_char, end_char, text_seg))
            i += 1

    chunks: list[DocumentChunkSpec] = []
    for order, (start_c, end_c, chunk_str) in enumerate(merged_spans):
        source_loc = f"rev{document_revision}:chunk{order}:chars{start_c}-{end_c}"
        vec = embed_text_384(chunk_str)
        chunks.append(
            DocumentChunkSpec(
                chunk_order=order,
                source_location=source_loc,
                chunk_text=chunk_str,
                embedding_revision=EMBEDDING_MODEL_REVISION,
                embedding=vec,
            )
        )
    return chunks


def tokenize_retrieval_terms(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    filtered: list[str] = []
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        # Lightweight deterministic suffix normalization so plurals/inflections align across Java & Python
        stem = _light_stem(tok)
        filtered.append(stem)
    return filtered


def _light_stem(word: str) -> str:
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def embed_text_384(text: str) -> list[float]:
    """
    Day074: Deterministic 384-dimensional L2-normalized CPU text embedding (`pretrained-cpu-hash384-v1`).
    Combines weighted unigram stems, bigram phrases, and character 4-grams via SHA-256 signed feature
    projection into R^384. Identical algorithm in Java (`DeterministicCpuEmbedder`) and Python.
    """
    vec = [0.0] * EMBEDDING_DIMENSION
    terms = tokenize_retrieval_terms(text)

    if not terms:
        vec[0] = 1.0
        return vec

    features: list[tuple[str, float]] = []
    for t in terms:
        features.append((f"u:{t}", 2.0))
        if len(t) >= 4:
            for j in range(len(t) - 3):
                features.append((f"c4:{t[j:j+4]}", 0.5))

    for i in range(len(terms) - 1):
        features.append((f"b:{terms[i]}_{terms[i+1]}", 1.5))

    for feat, weight in features:
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        idx1 = int.from_bytes(digest[0:2], "big") % EMBEDDING_DIMENSION
        sign1 = 1.0 if (digest[2] & 1) == 0 else -1.0
        idx2 = int.from_bytes(digest[3:5], "big") % EMBEDDING_DIMENSION
        sign2 = 1.0 if (digest[5] & 1) == 0 else -1.0
        vec[idx1] += sign1 * weight
        vec[idx2] += sign2 * (weight * 0.5)

    norm_sq = sum(v * v for v in vec)
    if norm_sq <= 1e-12:
        vec[0] = 1.0
        return vec
    inv_norm = 1.0 / math.sqrt(norm_sq)
    return [round(v * inv_norm, 6) for v in vec]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
