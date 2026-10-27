# Gate G1 Verification Evidence Report (Phase 1: Day001–Day010)

- **Gate:** G1 — Program Foundation, Contracts & Local Platform Skeleton
- **Status:** PASSED (`0` errors, `0` failures)
- **Verification Date:** 2026-10-16

## 1. Day-by-Day Deliverables Checklist

| Day | Deliverable | Artifact / Implementation Path | Verification Status |
| :--- | :--- | :--- | :--- |
| **Day001** | Program Charter, Non-Goals & Runtime Baseline | `docs/scope.md`, `docs/environment.md`, `.gitignore` | Verified |
| **Day002** | Hardware & Toolchain Probe (`sm_50` GPU + CPU smoke) | `scripts/probe_environment.py`, `docs/reports/day002-environment-probe.json` | Verified (`Quadro M2000M` 4096 MiB FP32 PASS) |
| **Day003** | Monorepo Skeleton (`backend/` Java 21 + `ml/` Python 3.12) | `backend/pom.xml`, `ml/pyproject.toml`, `ml/configs/tiny.json`, `ml/configs/local.json` | Verified |
| **Day004** | Profiled Docker Compose & Multi-DB PostgreSQL 16 + `pgvector` | `infra/compose.yml`, `infra/.env.example`, `infra/postgres/init-multi-db.sql`, `docs/local-setup.md` | Verified (`scratchllm-postgres` healthy on `127.0.0.1:54322`) |
| **Day005** | Shared Cross-Language Contracts & Golden Fixtures | `contracts/openapi.yaml`, `contracts/internal-inference.schema.json`, `contracts/sse-events.schema.json`, `contracts/fixtures/*` | Verified in both Java & Python |
| **Day006** | Foundation PostgreSQL Schema & Flyway Migration `V1` | `backend/src/main/resources/db/migration/V1__phase1_foundation_schema.sql`, `docs/schema.md` | Verified (`15` tables, `2` partial unique indexes, `vector(384)` HNSW index) |
| **Day007** | Deterministic Fake Generation Provider & Fault Modes | `GenerationProvider.java`, `DeterministicFakeGenerationProvider.java`, `DeterministicFakeGenerationProviderTest.java` | Verified (`7` unit tests PASS) |
| **Day008** | CI Workflow & Unified Local Verification Harness | `.github/workflows/ci.yml`, `scripts/check_all.ps1` | Verified |
| **Day009** | Test Strategy, Threat Model, Benchmark Protocol & ADR 0001 | `docs/test-strategy.md`, `docs/threat-model.md`, `benchmarks/protocol.md`, `docs/adr/0001-phase1-architecture-and-contracts.md` | Verified |
| **Day010** | Phase 1 Gate G1 Review & Root Documentation | `README.md`, `docs/reports/g1-gate-evidence.md` | Verified |

## 2. Automated Test Suite Summary

- **Python ML & Contract Suite (`pytest` in `ml/`):**
  - `12 passed` (`test_phase1_foundation.py`: all 9 subpackage imports, `tiny` `843,520` parameter formula, `local` `3,111,936` parameter formula, dimension validation, OpenAPI + JSON Schema validation).
- **Java 21 Spring Boot & PostgreSQL Integration Suite (`.\mvnw.cmd test` in `backend/`):**
  - `11 tests run, 0 failures, 0 errors`:
    - `ModuleBoundariesTest` (`2` tests)
    - `SharedContractsValidationTest` (`2` tests)
    - `DeterministicFakeGenerationProviderTest` (`4` tests)
    - `PostgresFoundationSchemaIntegrationTest` (`3` live PostgreSQL 16 + `pgvector` integration tests)
