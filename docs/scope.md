# Phase 1 Scope Checklist: Headless Conversational AI Platform with Scratch Transformer

Status: Active Baseline (Day001 — Phase 1 Gate G1)
Companion documents: [design.md](../design.md) | [roadmap.md](../roadmap.md)

## 1. Product Boundary (Strict Headless Scope)

- **In Scope (Required Release by Day120):**
  - **Java 21 + Spring Boot 3.4.x Modular Monolith Backend:** Session-based authentication with CSRF protection, workspace & creator-private conversation isolation, idempotent run submission (`Idempotency-Key`), SSE token streaming with `Last-Event-ID` cursor replay and Shared SSE Multiplexer, and daily token quota reservation/settlement.
  - **Scratch Decoder-Only Transformer in PyTorch (`ml/src/scratchlm`):** Custom byte-level and educational byte-level BPE tokenizers, explicit `LayerNorm`, manual causal multi-head self-attention, pre-norm residual blocks with $1/\sqrt{2L}$ output projection scaling, tied input/output embeddings, custom AdamW training loop, checkpoint/resume, narrow SFT, KV cache, and stateful incremental UTF-8 streaming decoder.
  - **PostgreSQL 16 (`pgvector`):** Authoritative store for identity, conversations, messages, `generation_runs`, `run_events`, `idempotency_records`, `outbox_events`, `consumer_inbox`, `work_items` (`FOR UPDATE SKIP LOCKED`), usage accounting, and workspace-isolated document chunks. Separate databases (`llm_app`, `mlflow_db`, `airflow_db`) and credentials on the PostgreSQL instance.
  - **Redis 7 (`noeviction`):** Shared HTTP sessions (`Spring Session Data Redis`), atomic Lua-scripted rate limiting, and bounded cache-aside immutable model revision metadata.
  - **Apache Kafka (KRaft single-broker local profile):** Asynchronous event backbone for `generation.requested.v1`, `document.ingestion.requested.v1`, `model.released.v1`, and matching `.dlq` topics.
  - **MLflow Tracking & Model Registry (`--serve-artifacts`):** Custom-loop experiment tracking, reloadable `pyfunc` model packages, `mlflow-artifacts:/` logical URI bridge, and monotonic `release_generation` promotion reconciliation.
  - **Apache Airflow (`LocalExecutor`):** Offline batch DAGs (validate manifest $\rightarrow$ tokenize $\rightarrow$ submit bounded native Windows training job $\rightarrow$ evaluate $\rightarrow$ gate candidate registration) with OS-verified GPU mutual exclusion (`pid` + `process_create_time_utc`).
  - **Headless API Clients:** `curl.exe` cookie-jar scripts, exported Postman collection, and a Python `httpx` SSE CLI client.

- **Out of Scope (Explicit Non-Goals):**
  - Any custom browser UI, HTML/CSS/JS frontend, React/Vue/Angular app, or server-rendered Thymeleaf templates.
  - Claims of frontier LLM reasoning quality, general-purpose safety alignment, or multi-region high availability from a single laptop broker.
  - Wrapping `nn.Transformer`, `nn.MultiheadAttention`, or `AutoModelForCausalLM` as a substitute for the scratch generative decoder.

## 2. Phase 1 Gate (G1) Acceptance Checklist

- [x] **Day001:** Headless scope checklist (`docs/scope.md`) and pinned environment/storage baseline (`docs/environment.md`).
- [x] **Day002:** Non-destructive environment probe (`scripts/probe_environment.py`) executing CPU and GPU (`sm_50`) FP32 forward/backward smoke checks and recording structured evidence (`docs/reports/day002-environment-probe.json`).
- [x] **Day003:** Runnable monorepo skeleton (`backend/`, `ml/`, `contracts/`, `infra/`, `scripts/`) with Maven Wrapper, `pyproject.toml`, `.gitignore`, and single-command verification scripts.
- [x] **Day004:** Bounded Docker Compose profiles (`infra/compose.yml`) with separate PostgreSQL databases/users (`llm_app`, `mlflow_db`, `airflow_db`), resource limits, and Flyway migration integration (`docs/local-setup.md`).
- [x] **Day005:** Versioned REST/SSE OpenAPI specification (`contracts/openapi.yaml`), JSON Schemas (`contracts/internal-inference.schema.json`, `contracts/sse-events.schema.json`), and shared Java/Python contract fixtures.
- [x] **Day006:** Complete foundation schema (`backend/src/main/resources/db/migration/V1__phase1_foundation_schema.sql`) and documentation (`docs/schema.md`) enforcing partial unique indexes (`uq_active_run_per_conversation`, `uq_assistant_msg_per_run`), foreign keys, and deadlock-free lock ordering.
- [x] **Day007:** Java modular package structure, `GenerationProvider` SPI, and `DeterministicFakeGenerationProvider` supporting controllable chunk streaming, error injection, and mid-stream cancellation without a GPU.
- [x] **Day008:** GitHub Actions CI workflow (`.github/workflows/ci.yml`) and local secret/lock/test verification script (`scripts/check_all.ps1`).
- [x] **Day009:** Test strategy (`docs/test-strategy.md`), initial threat model (`docs/threat-model.md`), benchmark protocol (`benchmarks/protocol.md`), and foundational ADR (`docs/adr/0001-phase1-architecture-and-contracts.md`).
- [x] **Day010:** Clean-shell end-to-end verification of Gate **G1** recorded in `docs/reports/g1-gate-evidence.md`.
