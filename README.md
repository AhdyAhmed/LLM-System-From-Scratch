# End-to-End LLM System From Scratch

A production-grade, headless/API-first **Large Language Model System built from scratch**, combining a custom **PyTorch decoder-only Transformer & BPE tokenizer (`ml/`)** with a **Java 21 + Spring Boot 3.5 modular monolith control plane (`backend/`)** backed by **PostgreSQL 16 + `pgvector`**, **Redis 7**, **Kafka KRaft**, **MLflow**, and **Airflow** .

---

## Architecture & Key Design Documents

- **[System Design Specification (`design.md`)](design.md)** — Mathematical formulations, custom BPE & Transformer architecture, Java modular monolith boundaries, PostgreSQL 16 + `pgvector` schema, SSE streaming & backpressure, and operational runbooks.
- **[120-Day Engineering Roadmap (`roadmap.md`)](roadmap.md)** — 12 phases (Day001–Day120) with explicit acceptance gates (`G1`–`G12`).
- **[Scope & Non-Goals (`docs/scope.md`)](docs/scope.md)** — Strict API-only boundary and from-scratch vs. framework rules.
- **[Hardware & Toolchain Baseline (`docs/environment.md`)](docs/environment.md)** — Pinning for Java 21, Python 3.12, PyTorch `2.6.0+cu118` on NVIDIA Quadro M2000M (`sm_50`, 4 GB VRAM).
- **[Local Infrastructure Setup (`docs/local-setup.md`)](docs/local-setup.md)** — Profiled Docker Compose topology (`db`, `cache`, `events`, `mlops`, `orchestration`).
- **[Database Schema & Invariants (`docs/schema.md`)](docs/schema.md)** — 15 core tables, partial unique concurrency indexes, and hybrid `tsvector` + `pgvector` HNSW indexing.
- **[Gate G1 Verification Evidence (`docs/reports/g1-gate-evidence.md`)](docs/reports/g1-gate-evidence.md)** — Full Phase 1 (Day001–Day010) verification report.

---

## Quick Start (Phase 1 — Gate G1 Verified)

### 1. Start the PostgreSQL 16 + `pgvector` Database Profile
```powershell
docker compose --env-file infra/.env.example -f infra/compose.yml --profile db up -d
```

### 2. Run the Hardware & Runtime Probe
```powershell
python scripts/probe_environment.py
```

### 3. Run the Python ML & Shared Contracts Test Suite
```powershell
cd ml
python -m pytest
```

### 4. Run the Java 21 Spring Boot & PostgreSQL Integration Test Suite
```powershell
cd backend
.\mvnw.cmd test
```

### 5. Run the Full Unified Verification Harness
```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_all.ps1
```
