# Reproducible Terminal & Postman Demo Script (`Day116`)

This script provides a deterministic end-to-end walkthrough of the **ScratchLLM Platform (`v1.0.0`)** covering persistent chat, real-model SSE streaming, `Last-Event-ID` reconnect replay, two-replica stateless Redis sessions, Kafka outbox/inbox deduplication, workspace-scoped RAG citations, and Airflow/MLflow GPU-exclusive training.

Companion Postman Collection: [`docs/postman/scratchllm-release-collection.json`](postman/scratchllm-release-collection.json).

---

## 1. Environment Setup & Reset

```powershell
# 1. Start core data plane + tracking containers
docker compose --env-file infra\.env.example -f infra\compose.yml --profile platform --profile tracking up -d

# 2. Launch native FastAPI Inference Gateway (:8091) and two Spring Boot API replicas (:8080, :8082)
powershell -ExecutionPolicy Bypass -File scripts\start_native_services.ps1 -Mode two-replica
```

---

## 2. Step-by-Step Terminal Walkthrough

### Step 2.1 — Fetch CSRF Token, Register & Login (`Replica 1 :8080`)
```bash
# 1. Fetch CSRF cookie & token
curl -s -c cookies.txt http://127.0.0.1:8080/auth/csrf
# Expected: {"token":"<csrf-token>","headerName":"X-CSRF-TOKEN"}

# 2. Register user & default workspace
curl -s -b cookies.txt -c cookies.txt -X POST http://127.0.0.1:8080/auth/register \
  -H "Content-Type: application/json" \
  -H "X-CSRF-TOKEN: <csrf-token>" \
  -d '{"email":"demo@scratchllm.local","password":"StrongDemoPass!123","workspace_name":"Demo Workspace"}'
# Expected: 201 Created with user_id and workspace_id

# 3. Login (rotates session ID and sets HttpOnly SESSION cookie in Redis)
curl -s -b cookies.txt -c cookies.txt -X POST http://127.0.0.1:8080/auth/login \
  -H "Content-Type: application/json" \
  -H "X-CSRF-TOKEN: <csrf-token>" \
  -d '{"email":"demo@scratchllm.local","password":"StrongDemoPass!123"}'
```

### Step 2.2 — Create Conversation on `Replica 1 (:8080)` & Submit Run on `Replica 2 (:8082)`
```bash
# Create conversation on Replica 1
curl -s -b cookies.txt -X POST http://127.0.0.1:8080/api/v1/workspaces/<workspace_id>/conversations \
  -H "Content-Type: application/json" \
  -H "X-CSRF-TOKEN: <csrf-token>" \
  -d '{"title":"ScratchGPT & RAG Demo"}'
# Expected: 201 Created {"conversation_id":"<conv_id>", ...}

# Submit generation run on Replica 2 using the same Redis session cookie & Idempotency-Key
curl -s -b cookies.txt -X POST http://127.0.0.1:8082/api/v1/conversations/<conv_id>/runs \
  -H "Content-Type: application/json" \
  -H "X-CSRF-TOKEN: <csrf-token>" \
  -H "Idempotency-Key: demo-run-idem-0001" \
  -d '{
    "model_alias": "scratch-tiny",
    "prompt": "Once upon a time in the sunny meadow",
    "sampling": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "max_new_tokens": 16}
  }'
# Expected: 202 Accepted {"run_id":"<run_id>","status":"QUEUED","model_revision":"scratch-tiny-r1"}
```

### Step 2.3 — Stream Unbuffered SSE & Reconnect Replay with `Last-Event-ID`
```bash
# Stream live SSE events (unbuffered: X-Accel-Buffering: no)
curl -N -b cookies.txt http://127.0.0.1:8080/api/v1/runs/<run_id>/events

# Reconnect mid-stream or post-completion from sequence 2 (skips sequences 1 and 2!)
curl -N -b cookies.txt -H "Last-Event-ID: 2" http://127.0.0.1:8082/api/v1/runs/<run_id>/events
```

### Step 2.4 — Upload Workspace Document via Kafka Outbox & Query Cited RAG
```bash
# Upload Markdown document (writes outbox_event -> Kafka document.ingestion.requested.v1 -> chunks + pgvector)
curl -s -b cookies.txt -X POST http://127.0.0.1:8080/api/v1/workspaces/<workspace_id>/documents \
  -H "Content-Type: application/json" \
  -H "X-CSRF-TOKEN: <csrf-token>" \
  -d '{
    "filename": "architecture-notes.md",
    "mime_type": "text/markdown",
    "content": "ScratchLLM uses a singleton gpu_leases row (id=1) in PostgreSQL to enforce single-GPU mutual exclusion between Airflow training and ScratchGPT inference."
  }'

# Search workspace chunks with verified source citations
curl -s -b cookies.txt -X POST http://127.0.0.1:8080/api/v1/workspaces/<workspace_id>/documents/search \
  -H "Content-Type: application/json" \
  -H "X-CSRF-TOKEN: <csrf-token>" \
  -d '{"query":"How does ScratchLLM enforce single-GPU mutual exclusion?","top_k":3,"mode":"hybrid"}'
```

### Step 2.5 — Trigger Airflow-to-Native-Worker Training Bridge & Verify GPU Mutual Exclusion
```bash
# Submit offline training job holding primary_gpu lease
curl -s -X POST http://127.0.0.1:8080/internal/training-jobs \
  -H "Content-Type: application/json" \
  -H "X-Internal-Service-Token: scratchllm-internal-dev-token-day051" \
  -d '{
    "dag_id": "scratchlm_offline_training_v1",
    "dag_run_id": "demo_run_01",
    "task_id": "submit_and_poll_training",
    "map_index": -1,
    "try_number": 1,
    "config_id": "scratch-tiny-pretrain-v1",
    "config_hash": "a1b2c3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
    "dataset_manifest_sha256": "d1d2d3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
    "tokenizer_sha256": "f1f2f3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
    "hold_gpu_lease_running": true
  }'
# Expected while training holds GPU:
# - Submitting a run with model_alias="scratch-tiny" returns 503 GPU_BUSY_WITH_TRAINING
# - Submitting a run with model_alias="fake-deterministic" returns 202 Accepted
```
