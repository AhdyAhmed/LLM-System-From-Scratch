# Architecture Decision Records (`ADRs`, `Day112`)

This file records the core architectural decisions across Phases 1–12, including decisions that were evaluated and rejected or deferred based on measured workstation trade-offs.

---

## ADR-001: Modular Monolith in Java 21 + Spring MVC + JDBC/Flyway (`Day001`–`Day015`)
- **Context**: A solo developer building an end-to-end LLM system on a `16 GB RAM` workstation needs strong transactional invariants across auth, conversations, quota reservations, SSE event logs, and outbox publishing without microservice deployment sprawl.
- **Decision**: Use a single Java 21 / Spring Boot 3.4 modular monolith (`auth`, `conversations`, `generation`, `documents`, `modelcatalog`, `mlops`, `observability`, `ops`) with explicit `JdbcTemplate` SQL and Flyway migrations (`V1`–`V7`).
- **Consequences**: Canonical lock ordering (`system_capacity -> usage_buckets -> conversations`), `FOR UPDATE` row locking, and `pgvector` distance queries are explicit and inspectable via `EXPLAIN (FORMAT JSON)`.

## ADR-002: Durable PostgreSQL `run_events` + Shared Batched SSE Multiplexer (`Day019`, `Day060`)
- **Context**: Spawning one dedicated database polling thread per open SSE connection exhausts HikariCP connections under 25–50 concurrent streams.
- **Decision**: Persist ordered `run_events (run_id, sequence)` in PostgreSQL and fan them out via [`SharedSseEventMultiplexer`](../../backend/src/main/java/com/scratchllm/platform/generation/SharedSseEventMultiplexer.java) using a single scheduled batch poller (`WHERE (run_id, sequence) > ...`) and per-subscriber bounded queues.
- **Trade-off / Non-Improvement Note**: We evaluated pushing raw token deltas through Redis Pub/Sub before PostgreSQL commit, but rejected it because a crash between Pub/Sub emission and DB commit breaks `Last-Event-ID` reconnect replay consistency.

## ADR-003: Redis for Stateless Sessions, Atomic Lua Rate Limits & Immutable Revision Cache (`Day017`, `Day055`)
- **Context**: Two stateless API replicas (`replica-1` on `:8080`, `replica-2` on `:8082`) must share authentication sessions, enforce burst rate limits without TOCTOU races, and avoid hitting PostgreSQL on every model alias lookup.
- **Decision**: Store sessions in Redis (`HttpOnly`, `SameSite=Lax`), execute rate-limit increments + TTL healing in a single atomic Lua script ([`RedisRateLimiterService`](../../backend/src/main/java/com/scratchllm/platform/infrastructure/RedisRateLimiterService.java)), and cache model metadata by immutable revision (`model:rev:{revision_tag}`) with transparent PostgreSQL fallback.

## ADR-004: Kafka Transactional Outbox, Idempotent Consumer Inbox & Leased `work_items` (`Day020`, `Day061`–`Day064`)
- **Context**: Direct dual-writes to PostgreSQL and Kafka during HTTP requests lose events or create phantom jobs when a broker or process crashes.
- **Decision**: Commit `outbox_events` inside the same PostgreSQL transaction as domain state, relay asynchronously to Kafka (`generation.requested.v1`, `document.ingestion.requested.v1`, `model.released.v1`), deduplicate via `consumer_inbox (consumer_name, event_id)`, and coordinate long-running execution with leased `work_items` and monotonic `fencing_token` checks.

## ADR-005: Custom PyTorch `ScratchGPT` Boundary & `ScratchKVCache` (`Day021`–`Day030`, `Day057`)
- **Context**: Educational and portfolio value requires writing the tokenizer, embeddings, LayerNorm, causal multi-head attention, MLP, cross-entropy loss, and KV cache explicitly without `transformers.AutoModelForCausalLM`.
- **Decision**: Implement [`ScratchGPT`](../../ml/src/scratchlm/model/gpt.py) and [`ScratchKVCache`](../../ml/src/scratchlm/inference/kv_cache.py) directly on `torch.Tensor` operations using PyTorch autograd, verifying gradient finiteness and cached-vs-uncached logit parity ($\le 10^{-5}$).

## ADR-006: Authoritative Application Release Table + Monotonic `release_generation` (`Day068`, `Day105`)
- **Context**: MLflow Model Registry alias updates (`champion`) are not transactional with PostgreSQL and can arrive out of order during rollbacks.
- **Decision**: PostgreSQL `active_model_releases` + `model_promotions` with a strictly monotonic `release_generation` is the authoritative routing source of truth; rollbacks increment `release_generation` (`N -> N + 1`) while selecting the prior immutable `revision_tag`.

## ADR-007: Airflow in Docker + Native Host GPU Worker via `gpu_leases` (`Day081`–`Day085`)
- **Context**: Passing an older NVIDIA Quadro M2000M (`4 GB VRAM`, Maxwell `sm_50`) into Linux Airflow containers on Windows adds fragile WSL2/CUDA container coupling and risks concurrent GPU OOMs between training and serving.
- **Decision**: Run Airflow `LocalExecutor` in Docker for orchestration while executing GPU training natively on the Windows host via [`native_worker.py`](../../ml/src/scratchlm/pipelines/native_worker.py) behind the authenticated `/internal/training-jobs` bridge and singleton `gpu_leases (id = 1)` table verified by OS `(native_pid, process_create_time_utc)`.

## ADR-008: Exact `pgvector` Search Before Approximate ANN Indexes (`Day073`–`Day074`)
- **Context**: At workspace corpus sizes of $< 10{,}000$ chunks, IVFFlat/HNSW approximate indexes introduce recall degradation on filtered workspace queries without meaningful latency gains over exact `pgvector` cosine distance scan.
- **Decision**: Use exact `pgvector` cosine search (`<=>`) alongside PostgreSQL GIN full-text search (`to_tsvector`) and reciprocal rank fusion, deferring approximate ANN indexes until corpus scale justifies the recall trade-off.
