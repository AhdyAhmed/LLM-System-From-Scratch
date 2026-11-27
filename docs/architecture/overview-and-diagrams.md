# Architecture Overview & Sequence/State Diagrams (`Day112`)

This document provides the implementation-verified architecture diagrams for the **ScratchLLM Platform (`v1.0.0`)**, matching the Java 21 / Spring Boot (`backend/`) and Python 3.12 / PyTorch (`ml/`) source code.

---

## 1. Synchronous Chat, Durable SSE & Transactional Outbox Sequence

```mermaid
sequenceDiagram
    participant Client as HTTP/SSE Client
    participant API as Spring Boot API (RunService)
    participant Redis as Redis 7.4
    participant PG as PostgreSQL 16 (llm_app)
    participant Kafka as Kafka 3.9 (KRaft)
    participant Worker as Generation Worker / FastAPI Gateway

    Client->>API: POST /api/v1/conversations/{id}/runs (Idempotency-Key, X-CSRF-TOKEN)
    API->>PG: Check idempotency_records (replay before rate-limit/quota!)
    API->>Redis: Eval atomic Lua rate limit (scratchllm:ratelimit:run:{userId})
    API->>Redis: Resolve model:alias:{alias} -> model:rev:{revision} (PG fallback)
    API->>PG: Lock system_capacity -> usage_buckets -> conversations (canonical order)
    API->>PG: Insert USER message, generation_run (QUEUED), usage_reservation, run.queued, outbox_event
    API-->>Client: 202 Accepted (run_id, status=QUEUED, stream_url)

    Client->>API: GET /api/v1/runs/{runId}/events (Last-Event-ID optional)
    API->>PG: SharedSseEventMultiplexer catch-up + batched 50ms polling
    PG->>Kafka: OutboxRelay publishes generation.requested.v1
    Kafka->>Worker: Lease work_item (fencing_token) + stream NDJSON deltas
    Worker->>PG: Persist batched token.delta + ASSISTANT message + settle usage_reservation
    API-->>Client: SSE event: token.delta ... event: run.completed
```

---

## 2. Generation Run Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> QUEUED: POST /runs (202 Accepted, quota reserved)
    QUEUED --> RUNNING: Worker acquires lease & increments attempt_token
    QUEUED --> CANCELLED: POST /cancel or queue_deadline_at expired (quota refunded)
    QUEUED --> FAILED: Dependency or admission fault (quota refunded)
    RUNNING --> COMPLETED: EOS / max_new_tokens reached (ASSISTANT message committed, quota settled)
    RUNNING --> CANCELLED: Client cancel_requested=true (immediate GPU slot release)
    RUNNING --> FAILED: Execution deadline / worker crash (partial tokens settled)
```

---

## 3. Single-GPU Mutual Exclusion & Airflow-to-Native-Worker Bridge (`gpu_leases`)

```mermaid
flowchart LR
    Airflow["Airflow DAG\n(scratchlm_offline_training_dag.py)"] -->|POST /internal/training-jobs\nX-Internal-Service-Token| Bridge["InternalTrainingJobController\n(Spring Boot)"]
    Bridge -->|Lock singleton gpu_leases (id=1)\nVerify (native_pid, process_create_time_utc)| PG[("PostgreSQL\ntraining_jobs + gpu_leases")]
    NativeWorker["Native Host Training Worker\n(native_worker.py)"] -->|Poll & Execute CLI Steps\nHeartbeat + Lease Renewal| PG
    NativeWorker -->|Log Metrics & Register Candidate| MLflow[("MLflow Registry\n(:5001)")]
    OnlineRun["Online Run Request\n(model_alias=scratch-tiny)"] -->|Check gpu_leases (id=1)| PG
    PG -->|If holder_kind=TRAINING| Reject["503 GPU_BUSY_WITH_TRAINING\n(fake-deterministic still 202 OK)"]
```
