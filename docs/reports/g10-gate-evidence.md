# Gate G10 Evidence Report — Measured Performance, Replica Behavior, and Packaging (`Day091`–`Day100`)

## 1. Executive Summary

Gate **G10** verifies that the ScratchLLM Platform meets all performance, multi-replica stateless scaling, real-model KV-cache acceleration, single-GPU mutual exclusion, startup configuration validation, graceful draining, and bounded soak/retention invariants.

| Day | Deliverable | Artifact / Evidence | Status |
| :--- | :--- | :--- | :--- |
| `Day091` | Reproducible Benchmark Harness | [`benchmarks/results/day091_harness_rehearsal.json`](../../benchmarks/results/day091_harness_rehearsal.json) | **PASSED** |
| `Day092` | PostgreSQL Query-Plan & Covering Indexes (`V6`) | [`benchmarks/postgres-query-plans.md`](../../benchmarks/postgres-query-plans.md) | **PASSED** |
| `Day093` | Two-Replica Stateless Routing & Unbuffered SSE | [`infra/nginx/nginx-two-replica.conf`](../../infra/nginx/nginx-two-replica.conf) | **PASSED** |
| `Day094` | 1-Replica vs 2-Replica Scaling & Queue Overload | [`benchmarks/api-scaling.md`](../../benchmarks/api-scaling.md) | **PASSED** |
| `Day095` | Real `ScratchGPT` KV-Cache `on` vs `off` Matrix | [`benchmarks/model-serving.md`](../../benchmarks/model-serving.md) | **PASSED** |
| `Day096` | Mixed Workload (Chat + Cancel + RAG + GPU Exclusion) | [`benchmarks/results/day096_mixed_workload.json`](../../benchmarks/results/day096_mixed_workload.json) | **PASSED** |
| `Day097` | Sequential Compose Profiles & CPU Demo Packaging | [`docs/local-setup.md`](../local-setup.md), [`scripts/run_cpu_demo.ps1`](../../scripts/run_cpu_demo.ps1) | **PASSED** |
| `Day098` | Startup Config Validation, Readiness & Graceful Drain | [`RuntimeReadinessAndDrainService.java`](../../backend/src/main/java/com/scratchllm/platform/observability/RuntimeReadinessAndDrainService.java) | **PASSED** |
| `Day099` | Bounded Soak & Retention Pruning Verification | [`benchmarks/soak.md`](../../benchmarks/soak.md) | **PASSED** |
| `Day100` | Gate G10 Review & Freeze | `scripts/check_all.ps1` (`0 errors, 0 failures`) | **PASSED** |

## 2. Key Quantitative Results

- **PostgreSQL Hot Queries (`Day092`)**: Composite indexes in `V6` improved `messages` tail fetch (`7.82x`), `generation_runs` history (`8.65x`), `outbox_events` relay batch (`6.67x`), and `run_events` retention pruning (`12.96x`).
- **Multi-Replica Scaling (`Day094`)**: Scaling from 1 to 2 stateless API replicas increased `fake-deterministic` throughput from `142.5 RPS` to `248.0 RPS` (`1.74x`) while enforcing the global queue cap (`503 SYSTEM_CAPACITY_EXCEEDED`).
- **Real-Model KV Cache (`Day095`)**: `ScratchKVCache` verified `100%` greedy token parity (`all_parity_checks_passed = True`) across all `(prompt_len, max_new_tokens)` cells within `max_seq_len = 128`.
- **Soak & Retention (`Day099`)**: `40` soak cycles completed with `0` leaked connections, `0` orphaned runs, and verified retention cleanup preserving pinned model lineage.
