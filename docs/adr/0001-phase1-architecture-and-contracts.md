# ADR 0001: Phase 1 Architecture, Hardware Envelope, and Boundary Contracts

- **Status:** Accepted
- **Date:** 2026-10-16
- **Gate:** G1 (Phase 1 — Program Foundation, Contracts & Local Platform Skeleton)

## Context

We are building a complete, end-to-end LLM system from scratch on a single developer workstation equipped with an Intel Core i7-6820HQ (4C/8T), 16 GB RAM, and an NVIDIA Quadro M2000M (4 GB VRAM, Maxwell `sm_50`). The system requires both a custom PyTorch decoder-only Transformer stack (`ml/`) and a production-grade Java 21 + Spring Boot 3.5 modular monolith control plane (`backend/`).

## Decisions

1. **Headless API-First Scope:**
   - The system exposes REST + Server-Sent Events (SSE) + OpenAPI 3.1 contracts (`contracts/openapi.yaml`) and CLI tooling. Frontend UI is excluded to focus 100% of engineering rigor on the custom ML stack and backend control plane.
2. **Maxwell `sm_50` FP32 Execution Envelope:**
   - Because Maxwell GPUs lack native FP16/BF16 Tensor Cores and modern PyTorch wheels drop `sm_50` after CUDA 11.8, the Python stack is pinned to `torch==2.6.0+cu118` in FP32 (`torch.float32`) with gradient accumulation and activation checkpointing.
   - Two explicit model profiles are frozen: `tiny` (`843,520` parameters) for rapid CPU/CI verification and `local` (`3,111,936` parameters) for GPU pretraining and SFT within `< 2.5 GB` VRAM.
3. **Database-Backed Concurrency & Idempotency Invariants:**
   - PostgreSQL 16 (`pgvector/pgvector:pg16` on host port `54322`) enforces one active generation run per conversation via the partial unique index `uq_active_run_per_conversation` and one assistant response per run via `uq_assistant_msg_per_run`.
4. **Profiled Infrastructure Compose Topology:**
   - `infra/compose.yml` isolates services into `--profile db`, `--profile cache`, `--profile events`, `--profile mlops`, and `--profile orchestration` so daily development consumes `< 1 GB` container RAM on a 16 GB host.
5. **Deterministic Fake Generation Provider (`GenerationProvider` SPI):**
   - `DeterministicFakeGenerationProvider` implements the full streaming, stop-sequence, latency-injection, and fault-injection contract (`TIMEOUT`, `UPSTREAM_503`, `MALFORMED_STREAM`, `DISCONNECT_MID_STREAM`) so Phases 7–11 can be tested deterministically with or without GPU inference loaded.

## Consequences

- Java and Python workspaces evolve independently against versioned JSON schemas (`contracts/internal-inference.schema.json`, `contracts/sse-events.schema.json`) validated in both `pytest` and JUnit 5.
- Every subsequent phase builds directly on the `V1__phase1_foundation_schema.sql` baseline without destructive schema rewrites.
