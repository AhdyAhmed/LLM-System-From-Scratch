"""
Day071–Day080 (Phase 8): Automated unit and benchmark tests for deterministic chunking,
384-d L2-normalized CPU embeddings, workspace-isolated retrieval evaluation (Lexical vs. Vector vs. Hybrid),
and ScratchLlmClient citation inspection / stale citation CLI reporting.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import httpx

from scratchlm.client import ScratchLlmClient
from scratchlm.retrieval.embedder import (
    CHUNKER_VERSION,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_REVISION,
    chunk_document_text,
    cosine_similarity,
    embed_text_384,
    tokenize_retrieval_terms,
)
from scratchlm.retrieval.evaluator import (
    build_in_memory_index,
    evaluate_retrieval_benchmark,
    search_workspace_chunks,
)


def test_deterministic_chunker_and_384d_unit_norm_embeddings() -> None:
    doc_text = (
        "# Chapter 1: Causal Masking\n"
        "The decoder-only ScratchGPT transformer enforces strict upper-triangular causal masking "
        "over 128 context tokens so future tokens never leak into past positions.\n\n"
        "# Chapter 2: Single-GPU Slot\n"
        "A fair Semaphore(1, true) in Java and a threading.BoundedSemaphore(1) in Python enforce "
        "strictly 1 active generation slot on the GTX 1050 (2 GB VRAM)."
    )
    chunks_a = chunk_document_text(doc_text, document_revision=1)
    chunks_b = chunk_document_text(doc_text, document_revision=1)

    assert len(chunks_a) == 2
    assert chunks_a == chunks_b
    assert chunks_a[0].embedding_revision == EMBEDDING_MODEL_REVISION
    assert len(chunks_a[0].embedding) == EMBEDDING_DIMENSION

    norm0 = math.sqrt(sum(v * v for v in chunks_a[0].embedding))
    assert abs(norm0 - 1.0) < 1e-4

    q_mask = embed_text_384("upper-triangular causal masking over 128 context tokens")
    q_gpu = embed_text_384("Semaphore(1, true) single active generation slot on GTX 1050")

    assert cosine_similarity(q_mask, chunks_a[0].embedding) > cosine_similarity(
        q_mask, chunks_a[1].embedding
    )
    assert cosine_similarity(q_gpu, chunks_a[1].embedding) > cosine_similarity(
        q_gpu, chunks_a[0].embedding
    )

    terms_0 = set(tokenize_retrieval_terms(chunks_a[0].chunk_text))
    terms_1 = set(tokenize_retrieval_terms(chunks_a[1].chunk_text))
    query_terms = set(tokenize_retrieval_terms("causal masking 128 tokens"))
    assert len(query_terms.intersection(terms_0)) > len(
        query_terms.intersection(terms_1)
    )


def test_retrieval_benchmark_meets_gate_g8_invariants() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path = repo_root / "evals" / "retrieval" / "corpus_documents.json"
    questions_path = repo_root / "evals" / "retrieval" / "questions_benchmark.json"

    summary = evaluate_retrieval_benchmark(
        corpus_path=corpus_path,
        questions_path=questions_path,
    )

    assert summary["chunker_version"] == CHUNKER_VERSION
    assert summary["embedding_model_revision"] == EMBEDDING_MODEL_REVISION
    assert summary["embedding_dimension"] == 384
    assert summary["total_indexed_chunks"] >= 15

    for split_name in ("dev", "test"):
        hybrid = summary["modes"]["hybrid"][split_name]
        assert hybrid["recall_at_1"] >= 0.90
        assert hybrid["recall_at_3"] == 1.0
        assert hybrid["mrr"] >= 0.95
        assert hybrid["unanswerable_abstention_rate"] == 1.0

    # Verify cross-workspace isolation directly on the benchmark index
    corpus_data = json.loads(corpus_path.read_text(encoding="utf-8"))
    index = build_in_memory_index(corpus_data)
    alpha_hits_for_beta_secret = search_workspace_chunks(
        index=index,
        workspace_scope="workspace-alpha",
        query="obsidian vault badge BETA-991",
        mode="hybrid",
        top_k=3,
    )
    assert alpha_hits_for_beta_secret == []

    beta_hits_for_beta_secret = search_workspace_chunks(
        index=index,
        workspace_scope="workspace-beta",
        query="obsidian vault badge BETA-991",
        mode="hybrid",
        top_k=3,
    )
    assert len(beta_hits_for_beta_secret) >= 1
    assert beta_hits_for_beta_secret[0]["doc_slug"] == "private-beta-vault-notes"


def test_client_document_retrieval_and_stale_citation_cli_rendering() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/csrf":
            return httpx.Response(
                200, json={"header_name": "X-CSRF-TOKEN", "token": "csrf-p8"}
            )
        if (
            request.url.path == "/api/v1/workspaces/ws-alpha/documents"
            and request.method == "POST"
        ):
            body = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                201,
                json={
                    "document_id": "doc-uuid-1",
                    "workspace_id": "ws-alpha",
                    "filename": body["filename"],
                    "ingestion_status": "PENDING",
                    "revision": 1,
                },
            )
        if (
            request.url.path == "/api/v1/workspaces/ws-alpha/documents/search"
            and request.method == "POST"
        ):
            return httpx.Response(
                200,
                json={
                    "workspace_id": "ws-alpha",
                    "mode": "hybrid",
                    "embedding_model_revision": EMBEDDING_MODEL_REVISION,
                    "results": [
                        {
                            "citation_id": "cit-1111222233",
                            "chunk_id": "11112222-3333-4444-5555-666677778888",
                            "document_id": "doc-uuid-1",
                            "filename": "architecture-overview.md",
                            "document_revision": 1,
                            "chunk_order": 0,
                            "source_location": "rev1:chunk0:chars0-120",
                            "snippet": "Canonical lock hierarchy: system_capacity -> usage_buckets -> conversations.",
                            "score": 0.9125,
                            "truncated": False,
                            "availability_status": "AVAILABLE",
                        }
                    ],
                },
            )
        if request.url.path == "/api/v1/runs/run-rag-1/citations":
            return httpx.Response(
                200,
                json={
                    "run_id": "run-rag-1",
                    "citations": [
                        {
                            "citation_id": "cit-1111222233",
                            "chunk_id": "11112222-3333-4444-5555-666677778888",
                            "document_id": "doc-uuid-1",
                            "filename": "architecture-overview.md",
                            "document_revision": 1,
                            "chunk_order": 0,
                            "source_location": "rev1:chunk0:chars0-120",
                            "snippet": "Canonical lock hierarchy: system_capacity -> usage_buckets -> conversations.",
                            "score": 0.9125,
                            "truncated": False,
                            "availability_status": "AVAILABLE",
                        },
                        {
                            "citation_id": "cit-9999888877",
                            "chunk_id": "99998888-7777-6666-5555-444433332222",
                            "document_id": "doc-uuid-deleted",
                            "filename": "deleted-notes.md",
                            "document_revision": 1,
                            "chunk_order": 1,
                            "source_location": "rev1:chunk1:chars121-240",
                            "snippet": "[SOURCE UNAVAILABLE: Document was deleted or superseded]",
                            "score": 0.7420,
                            "truncated": False,
                            "availability_status": "UNAVAILABLE_DELETED",
                        },
                    ],
                },
            )
        if request.url.path == "/api/v1/runs/run-rag-1/citations/validate":
            return httpx.Response(
                200,
                json={
                    "run_id": "run-rag-1",
                    "results": [
                        {
                            "citation_id": "cit-1111222233",
                            "valid": True,
                            "status": "VALIDATED",
                        },
                        {
                            "citation_id": "cit-hallucinated",
                            "valid": False,
                            "status": "REJECTED_UNVERIFIED_CITATION",
                        },
                    ],
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    with ScratchLlmClient(transport=transport) as client:
        uploaded = client.upload_document(
            workspace_id="ws-alpha",
            filename="architecture-overview.md",
            content="# Lock Hierarchy\nsystem_capacity -> usage_buckets -> conversations",
        )
        assert uploaded["document_id"] == "doc-uuid-1"
        assert uploaded["ingestion_status"] == "PENDING"

        search_resp = client.search_documents(
            workspace_id="ws-alpha",
            query="canonical lock hierarchy",
            mode="hybrid",
            top_k=3,
        )
        assert len(search_resp["results"]) == 1
        assert search_resp["results"][0]["citation_id"] == "cit-1111222233"

        citations_resp = client.get_run_citations("run-rag-1")
        cli_output = client.format_citations_for_cli(citations_resp["citations"])
        assert "[cit-1111222233] architecture-overview.md" in cli_output
        assert "Canonical lock hierarchy" in cli_output
        assert "[cit-9999888877] deleted-notes.md" in cli_output
        assert "[UNAVAILABLE: Source document was deleted or superseded]" in cli_output

        val_resp = client.validate_run_citations(
            "run-rag-1", ["cit-1111222233", "cit-hallucinated"]
        )
        assert val_resp["results"][0]["valid"] is True
        assert val_resp["results"][1]["valid"] is False
        assert val_resp["results"][1]["status"] == "REJECTED_UNVERIFIED_CITATION"
