# Release Checklist & Runtime Maintenance Audit (`Day109`)

This checklist audits the **ScratchLLM Platform** against actual pinned dependency versions, hardware constraints, and security/deployment boundaries prior to Gate **G11** and **G12**.

---

## 1. Deployment Scope & Exposure Truthfulness

- [x] **Release Scope**: `local-private-demo` (local workstation execution; no public URL or multi-region cloud HA is claimed).
- [x] **Private Service Bindings**: PostgreSQL (`127.0.0.1:54322`), Redis (`127.0.0.1:63799`), Kafka (`127.0.0.1:9094`), MLflow (`127.0.0.1:5001`), Airflow (`127.0.0.1:8088`), FastAPI Gateway (`127.0.0.1:8091`), and Spring Boot replicas (`127.0.0.1:8080`, `127.0.0.1:8082`) bind strictly to loopback.
- [x] **Fake-Provider vs. Real-Model Separation**: [`benchmarks/api-scaling.md`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/benchmarks/api-scaling.md) (`fake-deterministic` orchestration throughput) and [`benchmarks/model-serving.md`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/benchmarks/model-serving.md) (real `ScratchGPT` KV-cache decode latency & memory) are reported separately and never conflated.

---

## 2. Pinned Dependency & Hardware Maintenance Constraints

| Component | Pinned Version | Maintenance & Hardware Constraint |
| :--- | :--- | :--- |
| Java / Spring Boot | Java `21` / Spring Boot `3.4.3` | Flyway `V1`–`V7` migrations validated on PostgreSQL 16 |
| Python / PyTorch | Python `3.12` / PyTorch `2.x` | Local GPU is an **NVIDIA Quadro M2000M (`4 GB VRAM`, Maxwell `sm_50`)**. Newer PyTorch CUDA wheels deprecate `sm_50` PTX kernels; therefore, the CPU tiny-model path (`scratch-mini`, `scripts/run_cpu_demo.ps1`) is maintained as a first-class, always-runnable release path alongside single-GPU execution |
| PostgreSQL + `pgvector` | `pgvector/pgvector:pg16` | Hosts `llm_app`, `mlflow_db`, and `airflow_db` with isolated credentials and `max_connections=80` |
| Redis | `redis:7.4-alpine` | `384 MB` cap (`noeviction`); used for sessions, atomic Lua rate limiting, and `model:rev:*` cache |
| Apache Kafka | `apache/kafka:3.9.2` (KRaft) | Single-broker KRaft mode; transactional outbox (`outbox_events`) + idempotent `consumer_inbox` |
| MLflow | `pymk-mlflow:3.17.1` | REST tracking & Model Registry with local outage spooling and SHA-256 package verification |
| Apache Airflow | `apache/airflow:3.2.2-python3.12` | `LocalExecutor` in Docker calling host Native Training Worker over `X-Internal-Service-Token` |

---

## 3. Operational & Security Readiness Checklist

- [x] **Secret Scanning**: `scripts/check_all.ps1` step `[1/5]` verifies zero leaked credentials or private keys in Git.
- [x] **Session & CSRF Protection**: `HttpOnly` session cookies, CSRF token enforcement (`403 CSRF_INVALID`), and URL-token rejection (`400 TOKEN_IN_URL_FORBIDDEN`).
- [x] **Single-GPU Mutual Exclusion**: `gpu_leases` (`id = 1`) prevents concurrent Airflow training and `scratch-tiny` GPU inference (`503 GPU_BUSY_WITH_TRAINING` / `409 GPU_EXCLUSION_CONFLICT`) with OS `(native_pid, process_create_time_utc)` stale lease reclamation.
- [x] **Backup, Restore & Rollback Verified**: Timed isolated restore drill (`docs/restore-drill.md`) and monotonic `release_generation` model rollback tested end-to-end.
