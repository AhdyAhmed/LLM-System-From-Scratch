# Local Workstation Setup & Sequential Profile Guide (`Day097`)

This guide documents how to run the **ScratchLLM Platform** within the single-workstation physical budget (`8 CPU threads`, `16 GB RAM`, `1x 6 GB CUDA GPU`, `80 GB SSD`) without swapping or host OOMs.

---

## 1. Workstation Resource Budget & Sequential Profiles

Running PostgreSQL, Redis, Kafka, MLflow, Airflow, two Spring Boot API replicas, the React UI, and PyTorch CUDA simultaneously exceeds 16 GB of system RAM. Therefore, Docker Compose ([`infra/compose.yml`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/infra/compose.yml)) and native launch scripts use explicit **sequential profiles**:

| Profile Name | Containers Started | Host Native Processes | Peak RAM Budget | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `db` | `postgres` (`pgvector:pg16`) | None / Schema tools | `~1.8 GB` | Schema migrations, SQL query-plan inspection (`Day092`), offline evals |
| `platform` | `postgres`, `redis`, `kafka` | 1–2 Spring Boot replicas + FastAPI Gateway + React UI | `~7.5 GB` | Interactive chat, SSE streaming, RAG ingestion, two-replica scaling (`Day093`–`Day096`) |
| `tracking` | `postgres`, `mlflow` | Python training / eval CLI | `~4.5 GB` | Pretraining, SFT, DPO, and MLflow artifact registration (`Phase 3`–`Phase 5`) |
| `pipelines` | `postgres`, `airflow` | Native Training Worker (`native_worker.py`) | `~5.2 GB` | Offline Airflow DAG orchestration & single-GPU training handoff (`Phase 9`) |
| `full` | All 5 containers | Config validation / high-RAM hosts | `~9.5 GB` | Compose syntax verification (`docker compose --profile full config -q`) |

---

## 2. Step-by-Step Startup Workflows

### 2.1 Interactive Platform Profile (`platform`)

1. Start the core data plane containers:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\start_infra.ps1 -Profile platform
   ```
2. Launch native services (single-replica or two-replica mode):
   ```powershell
   # Single replica (Spring Boot on :8080 + FastAPI Inference Gateway on :8091)
   powershell -ExecutionPolicy Bypass -File scripts\start_native_services.ps1 -Mode single

   # Two-replica mode (Spring Boot on :8080 and :8082 + FastAPI Inference Gateway on :8091)
   powershell -ExecutionPolicy Bypass -File scripts\start_native_services.ps1 -Mode two-replica
   ```
3. Optional: Start the stateless round-robin Nginx reverse proxy on `127.0.0.1:8090` using [`infra/nginx/nginx-two-replica.conf`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/infra/nginx/nginx-two-replica.conf).

### 2.2 CPU-Only Demo Mode (`No-CUDA Fallback`)

For laptops or CI runners without a dedicated NVIDIA GPU, run the self-contained CPU demo using the `scratch-mini` (`~1.2M` parameters) checkpoint and `fake-deterministic` provider:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_cpu_demo.ps1
```

### 2.3 Stopping Services Cleanly

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stop_native_services.ps1
powershell -ExecutionPolicy Bypass -File scripts\stop_infra.ps1
```

---

## 3. Startup Config Validation & Graceful Draining (`Day098`)

- **Startup Validation**: [`RuntimeReadinessAndDrainService`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/backend/src/main/java/com/scratchllm/platform/observability/RuntimeReadinessAndDrainService.java) validates `maxActiveRuns >= 1`, `maxQueuedRuns >= 1`, `ssePollIntervalMs >= 10`, and `dailyAllowanceTokens >= 64` at startup (`@PostConstruct`), failing fast if misconfigured.
- **Readiness Probe**: `GET /api/v1/observability/readiness` returns `200 OK` when PostgreSQL, Redis, and admission state are `READY`, or `503 Service Unavailable` when `DRAINING`.
- **Graceful Drain**: `POST /api/v1/observability/drain` transitions the replica to `DRAINING`, blocking new run admissions with `503 SERVICE_DRAINING` while allowing idempotent replays and releasing idle GPU inference leases cleanly.
