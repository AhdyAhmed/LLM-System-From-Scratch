# Changelog

All notable changes across the 120-day development of the **ScratchLLM Platform** are documented in this file.

## [`v1.0.0`] — 2026-10-17 (`Gate G12 Release`)

### Added
- **Phase 1 (`Day001`–`Day010`, Gate `G1`)**: Pinned Java 21 / Python 3.12 / Docker Compose infrastructure (`PostgreSQL 16 + pgvector`, `Redis 7.4`, `Kafka 3.9 KRaft`, `MLflow 3.17`, `Airflow 3.2`), Flyway `V1` foundation schema, and 5-stage `scripts/check_all.ps1` gate.
- **Phase 2 (`Day011`–`Day020`, Gate `G2`)**: Session + CSRF authentication, owner-isolated workspaces/conversations, Redis atomic Lua rate limiter, idempotent run submission, durable PostgreSQL `run_events` SSE streaming with `Last-Event-ID` replay, and Kafka transactional outbox/inbox contracts.
- **Phase 3 (`Day021`–`Day030`, Gate `G3`)**: From-scratch PyTorch `ScratchGPT` pre-norm Transformer (`ScratchLayerNorm`, `TokenAndPositionEmbedding`, `MultiHeadCausalSelfAttention`, GELU MLP, weight tying, $1/\sqrt{2L}$ residual scaling, masked cross-entropy) with gradient-leakage and toy-memorization verification.
- **Phase 4 (`Day031`–`Day040`, Gate `G4`)**: Deterministic corpus normalization/deduplication/splitting, train-only `ByteLevelBPETokenizer` (`vocab_size=512`), resumable `AdamW` training loop, and `SpoolingMlflowRestTracker` with local outage spool recovery.
- **Phase 5 (`Day041`–`Day050`, Gate `G5`)**: Pretraining and narrow SFT with assistant-only loss masking, `ScratchLmPyfuncAdapter` packaging with SHA-256 verification, and MLflow Model Registry candidate gating.
- **Phase 6 (`Day051`–`Day060`, Gate `G6`)**: FastAPI internal inference service (`X-Internal-Service-Token`), incremental `ScratchKVCache` decoding, `ContextPacker`, Redis `model:rev:*` metadata cache, and batched `SharedSseEventMultiplexer`.
- **Phase 7 (`Day061`–`Day070`, Gate `G7`)**: Leased `work_items` with `fencing_token`, outbox high-water-mark backpressure, DLQ routing, reservation/settlement quota reconciliation, and retention pruning.
- **Phase 8 (`Day071`–`Day080`, Gate `G8`)**: Asynchronous Kafka document ingestion, versioned markdown/text chunker, hybrid `pgvector` + PostgreSQL FTS retrieval with verified citations, prompt-injection defense, and monotonic `release_generation` model promotion/rollback.
- **Phase 9 (`Day081`–`Day090`, Gate `G9`)**: Airflow DAG (`scratchlm_offline_training_dag.py`) to native host GPU worker bridge (`/internal/training-jobs`), singleton `gpu_leases` mutual exclusion (`503 GPU_BUSY_WITH_TRAINING`), STRIDE threat model, and `/api/v1/observability/metrics`.
- **Phase 10 (`Day091`–`Day100`, Gate `G10`)**: Reproducible benchmark harness, PostgreSQL `V6` covering indexes, stateless two-replica Nginx config with unbuffered SSE, real-model KV-cache benchmark matrix, sequential Compose profiles, startup config validation, and graceful replica draining.
- **Phase 11 (`Day101`–`Day110`, Gate `G11`)**: Local TLS reverse-proxy rehearsal, checksummed multi-DB backup & isolated restore drills (`V7`), retention/deduplication horizon enforcement (`336h > 168h`), single-service outage drills, and complete-stack tiny-workload rehearsal.
- **Phase 12 (`Day111`–`Day120`, Gate `G12`)**: Sealed final held-out `test` split evaluation (`day114_sealed_final_test_evaluation.json`), Model/Data/Tokenizer cards, ADRs, Transformer walkthrough, consolidated engineering report, Postman collection, clean-checkout rehearsal, and `GET /api/v1/release/manifest`.
