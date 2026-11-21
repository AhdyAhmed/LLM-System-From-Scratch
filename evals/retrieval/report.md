# Phase 8 Independent Retrieval Evaluation Report (Day071–Day080)

## 1. Corpus & Provenance Summary

- **Corpus ID:** `scratchlm-authored-retrieval-corpus-v1` (`CC0-1.0` self-authored project benchmark)
- **Benchmark ID:** `scratchlm-retrieval-benchmark-v1` (`32` total questions: `16` dev, `16` held-out test, including answerable, unanswerable, and cross-workspace isolation queries)
- **Chunker Version:** `paragraph-window-v1` (`15` deterministic chunks across `workspace-alpha` and `workspace-beta`)
- **CPU Embedding Component:** `pretrained-cpu-hash384-v1` (`384` dimensions, L2-normalized). Explicitly separate from the from-scratch `ScratchGPT` causal decoder.

---

## 2. Retrieval Metrics (Development vs. Frozen Held-Out Test Split)

| Mode | Split | Questions (Ans / Unans) | Recall@1 | Recall@3 | MRR | Unanswerable Abstention Rate |
|---|---|---:|---:|---:|---:|---:|
| **Lexical Baseline (FTS)** | `test` | 16 (13 / 3) | 1.0000 | 1.0000 | 1.0000 | 0.6667 |
| **Vector (`pgvector` 384-D)** | `test` | 16 (13 / 3) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Hybrid (`RRF + Cosine/FTS`)** | `dev` | 16 (14 / 2) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Hybrid (`RRF + Cosine/FTS`)** | `test` | 16 (13 / 3) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## 3. Workspace Authorization, Context Budget & Safety Invariants

1. **Pre-Ranking Workspace Membership Filter:**
   - Queries in `workspace-alpha` (`Q29`: asking for the `workspace-beta` obsidian vault badge `BETA-991`) apply `workspace_id` and `memberships` filtering **before** similarity ranking or `LIMIT`, returning `0` unauthorized chunks from `workspace-beta`.
   - Authorized `workspace-beta` queries (`Q30`) retrieve `BETA-991` at rank `1`.
2. **Context-Window Budget & Server-Validated Citations (`C = 128`):**
   - Retrieved snippets are packed within the `128`-token `ScratchGPT` context limit while reserving `max_new_tokens` and prompt tokens.
   - Stored evidence is persisted in `retrieval_snapshots` and emitted via `retrieval.context` SSE events with server-validated `citation_id`s resolving to exact `(document_id, document_revision, chunk_order)` rows.
3. **Untrusted Document Content & Tombstone Deletion:**
   - Prompt-injection text in `untrusted-vendor-memo.txt` (`IGNORE ALL PREVIOUS INSTRUCTIONS...`) is treated strictly as passive quoted evidence; it cannot alter `workspace_id` scope, session permissions, or execute tools.
   - Deleting a document sets `ingestion_status = 'DELETED'` and `deleted_at = clock_timestamp()`, immediately excluding its chunks from search and marking historical citations as `UNAVAILABLE_DELETED`.
