# Service Targets, Alerts & Operational Playbooks (Day089)

## 1. Service-Level Objectives (SLOs) & Latency Targets (Single-Node Workstation Profile)

| Subsystem / Flow | Metric | Target / Threshold | Enforcement / Bounding Mechanism |
| :--- | :--- | :--- | :--- |
| **Control Plane APIs (`/api/v1/**`)** | p95 Non-Streaming Latency | `< 150 ms` | Connection pooling (HikariCP max 10), indexed keyset pagination, cache-aside Redis model catalog (`model:rev:{revision}`) |
| **Generation Admission (`POST /runs`)** | Queue Capacity & Backpressure | Max `8` waiting/active runs; Max `250` unpublished outbox rows | `system_capacity` row lock (`FOR UPDATE`), fails closed with `503 SYSTEM_CAPACITY_EXCEEDED` or `503 OUTBOX_BACKPRESSURE_EXCEEDED` |
| **SSE Streaming (`GET /runs/{id}/events`)** | Multiplexer DB Polling & Subscriber Memory | `1` batch SQL poll per tick regardless of subscriber count; `64` events/subscriber cap | `SharedSseEventMultiplexer` single-flight batch poller; slow readers overflow cleanly with `resync.required` |
| **Single-GPU Mutual Exclusion** | Concurrent GPU Contention | `0` simultaneous training + GPU inference processes | `gpu_leases` (`resource_name = 'primary_gpu'`) + `Semaphore(1, true)`; GPU inference returns `503 GPU_BUSY_WITH_TRAINING` during offline training |
| **Transactional Outbox & Kafka Relay** | Relay Lag & Poison Message Isolation | Unpublished backlog `< 50` steady-state; DLQ `= 0` | `FOR UPDATE SKIP LOCKED` relay + bounded retry (`max 3`) to `llm.dlq.v1` / `dead_letter_events` |

---

## 2. Alert Definitions & Operational Playbooks

### Alert 1: `OutboxBacklogHigh` (`outbox_unpublished_backlog >= 200`)
- **Symptom**: `GET /api/v1/observability/metrics` reports `outbox_unpublished_backlog` approaching `max_unpublished_outbox_events` (`250`). New run submissions begin returning `503 OUTBOX_BACKPRESSURE_EXCEEDED`.
- **Playbook**:
  1. Check Kafka container health (`docker compose ps kafka`).
  2. Inspect `outbox_events` `last_error` column for broker connectivity errors.
  3. Once Kafka recovers, `OutboxRelayAndInboxService` automatically drains pending rows in batches using `FOR UPDATE SKIP LOCKED`.

### Alert 2: `DeadLetterQueueNonZero` (`dlq_unresolved_count > 0`)
- **Symptom**: `dead_letter_events` count increases above `0`.
- **Playbook**:
  1. Query `SELECT id, original_topic, event_type, error_reason FROM dead_letter_events ORDER BY failed_at DESC LIMIT 10;`.
  2. Fix any downstream schema or consumer bug, or inspect if a stale `release_generation` promotion event was rejected.

### Alert 3: `StaleGpuLeaseOrHungTrainingJob` (`gpu_lease.lease_state == 'ACTIVE'` past expected window)
- **Symptom**: Interactive `scratch-tiny` runs return `503 GPU_BUSY_WITH_TRAINING`, or a training task is stuck.
- **Playbook**:
  1. Query `SELECT resource_name, holder_kind, holder_id, native_pid, process_create_time_utc, lease_expires_at FROM gpu_leases;`.
  2. If the job should be preempted immediately, call `POST /internal/training-jobs/{jobId}/cancel` with `X-Internal-Service-Token`.
  3. If the worker process crashed, the next submission or heartbeat check automatically verifies `(native_pid, process_create_time_utc)` against the OS process table and reclaims the stale lease.
