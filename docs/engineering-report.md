# Consolidated Engineering Report (`Day115`)

This report consolidates the measured performance, reliability, MLOps lineage, and failure-recovery findings across Phases 1–12 of the **ScratchLLM Platform (`v1.0.0`)**, linking every claim to raw machine-readable artifacts in the repository.

---

## 1. Workstation Hardware & Measurement Context

All measurements were collected on the target local workstation without cloud offload:
- **CPU / RAM**: `8 CPU threads`, `16 GB RAM` (`~7.5–7.8 GB` peak during active `platform` profile)
- **GPU**: `1x NVIDIA Quadro M2000M` (`4 GB VRAM`, Maxwell `sm_50`) + first-class CPU tiny-model execution path
- **Environment Probe**: [`docs/reports/day002-environment-probe.json`](reports/day002-environment-probe.json)

> [!IMPORTANT]
> We strictly separate **application orchestration capacity** (measured with `fake-deterministic` to stress Spring Boot, Redis, PostgreSQL, and SSE fan-out) from **real-model compute capacity** (measured with `ScratchGPT` and `ScratchKVCache`).

---

## 2. Summary of Measured Benchmarks (`Phase 10`)

### 2.1 PostgreSQL Query-Plan & Index Optimization (`Day092`)
Raw artifact: [`benchmarks/results/day092_query_plan_profile.json`](../benchmarks/results/day092_query_plan_profile.json) | Full report: [`benchmarks/postgres-query-plans.md`](../benchmarks/postgres-query-plans.md)

| Hot Query Path | Covering Index Added (`V6`) | `p95` Before | `p95` After | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| Conversation messages tail fetch | `idx_messages_conv_order_desc` | `4.85 ms` | `0.62 ms` | **7.82x** |
| Conversation runs recent history | `idx_generation_runs_conv_created_desc` | `6.40 ms` | `0.74 ms` | **8.65x** |
| Outbox relay unpublished batch | `idx_outbox_unpublished_retry_created` | `3.20 ms` | `0.48 ms` | **6.67x** |
| `run_events` retention pruning | `idx_run_events_created_at` | `14.90 ms` | `1.15 ms` | **12.96x** |

### 2.2 Stateless Two-Replica API Scaling & Overload Protection (`Day093`–`Day094`)
Raw artifact: [`benchmarks/results/day094_api_scaling.json`](../benchmarks/results/day094_api_scaling.json) | Full report: [`benchmarks/api-scaling.md`](../benchmarks/api-scaling.md)

- **1 Replica (`:8080`) vs. 2 Replicas (`:8080` + `:8082`)** on `fake-deterministic`:
  - Throughput increased from `142.5 req/s` to `248.0 req/s` (**1.74x** scaling efficiency) with `p95` run creation latency dropping from `24.8 ms` to `15.6 ms`.
  - Zero sticky sessions required: `SESSION` cookies, CSRF tokens, and atomic Lua rate limits are shared via Redis.
- **Queue Cap Overload Rejection**:
  - When `active_or_waiting_runs` reaches `max_waiting_runs = 8`, excess requests fail fast (`p95 = 3.8 ms`) with `503 SYSTEM_CAPACITY_EXCEEDED`.

### 2.3 Real-Model `ScratchGPT` KV-Cache `on` vs `off` Matrix (`Day095`)
Raw artifact: [`benchmarks/results/day095_model_serving_matrix.json`](../benchmarks/results/day095_model_serving_matrix.json) | Full report: [`benchmarks/model-serving.md`](../benchmarks/model-serving.md)

- Verified across prompt lengths `(16, 64, 96)` and output token caps `(16, 32)` within `max_seq_len = 128`.
- **100% greedy token parity** and logit equivalence ($\max |\Delta \text{logit}| \le 10^{-5}$) between cached incremental decoding and uncached full-prefix recomputation.

---

## 3. Failure Recovery, MLOps & Operational Drills (`Phase 7`–`Phase 11`)

| Subsystem / Scenario | Tested Failure Mode | Verified Recovery Invariant | Evidence |
| :--- | :--- | :--- | :--- |
| **Redis** (`Day055`, `Day107`) | Cache eviction / cold `model:rev:*` lookup | Transparent fallback to PostgreSQL `model_revisions`; auth/rate-limit fails closed (`503 REDIS_UNAVAILABLE`) | [`Phase11DeploymentBackupRestoreRollbackAndOutageIntegrationTest.java`](../backend/src/test/java/com/scratchllm/platform/ops/Phase11DeploymentBackupRestoreRollbackAndOutageIntegrationTest.java) |
| **Kafka Outbox/Inbox** (`Day062`–`Day064`) | Broker outage, duplicate delivery, poison message | Events buffer in `outbox_events` (`503 OUTBOX_BACKPRESSURE_EXCEEDED` at cap); `consumer_inbox` suppresses duplicates; poison payloads route to `.dlq.v1` | [`Phase2KafkaContractsIntegrationTest.java`](../backend/src/test/java/com/scratchllm/platform/infrastructure/Phase2KafkaContractsIntegrationTest.java) |
| **MLflow Registry** (`Day040`, `Day050`, `Day105`) | Tracking outage; model regression rollback | Local outage spool buffers metrics; active inference continues from local verified package; rollback increments monotonic `release_generation` (`N -> N+1`) | [`test_phase5_pretraining_sft_registry.py`](../ml/tests/test_phase5_pretraining_sft_registry.py) |
| **Airflow + GPU Lease** (`Day083`–`Day085`) | Task retry (`try_number++`), worker crash, GPU contention | Stable `submission_key` prevents duplicate jobs; stale lease reclaimed via OS `(native_pid, process_create_time_utc)`; `scratch-tiny` returns `503 GPU_BUSY_WITH_TRAINING` while `fake-deterministic` returns `202` | [`Phase9AirflowBridgeGpuExclusionSecurityAndObservabilityIntegrationTest.java`](../backend/src/test/java/com/scratchllm/platform/mlops/Phase9AirflowBridgeGpuExclusionSecurityAndObservabilityIntegrationTest.java) |
| **Backup & Restore** (`Day103`–`Day104`) | Multi-DB + artifact restore in maintenance mode | SHA-256 manifest verified, Redis sessions invalidated, tombstones re-applied, `gpu_leases` reset to `RELEASED` | [`docs/restore-drill.md`](restore-drill.md) |
