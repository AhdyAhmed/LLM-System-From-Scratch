# Gate G7 Verification Evidence — Kafka Delivery, Durable Execution & Failure Recovery (Day061–Day070)

## 1. Gate G7 Scope & Acceptance Criteria

Gate G7 verifies that the ScratchLLM platform's asynchronous control plane and durable execution layer satisfy all Phase 7 requirements across **Day061–Day070**:

| Day | Requirement | Implementation Artifact | Verification Test |
|---|---|---|---|
| **Day061** | Authoritative run lifecycle state machine, outbox/inbox/work_items/DLQ/promotion schema (`V3__phase7_durable_execution_dlq_and_promotions.sql`) | `V3__phase7_durable_execution_dlq_and_promotions.sql`, `docs/run-lifecycle.md` | Flyway migration + `Phase7OutboxKafkaFencingSseAndPromotionIntegrationTest` |
| **Day062** | Atomic acceptance in one short PostgreSQL transaction (`system_capacity -> usage_buckets -> conversations`) + Outbox High-Water-Mark backpressure (`503 OUTBOX_BACKPRESSURE_EXCEEDED`) | `RunService.java` | `outboxHighWaterMarkBackpressureAndKafkaRelayInboxDeduplicationWorkEndToEnd` |
| **Day063** | Polling Outbox Relay (`FOR UPDATE SKIP LOCKED`) -> Kafka (`generation.requested.v1`, `model.released.v1`) -> Idempotent Consumer Inbox (`ON CONFLICT (consumer_name, event_id) DO NOTHING`) -> `work_items` | `OutboxRelayAndInboxService.java` | `outboxHighWaterMarkBackpressureAndKafkaRelayInboxDeduplicationWorkEndToEnd` |
| **Day064** | Leased execution (`FOR UPDATE SKIP LOCKED`), monotonic `work_items.fencing_token` & `generation_runs.attempt_token`, lease renewal heartbeats, fenced write predicate (`WHERE id = :runId AND attempt_token = :attemptToken AND status = 'RUNNING'`), and orphan/expired-lease reconciliation | `FencedGenerationDispatcher.java` | `fencedDispatcherRejectsStaleAttemptWritesAndReconcilesOrphanedLeases` |
| **Day065** | Batched delta chunk persistence, idempotent terminal transition (`uq_assistant_msg_per_run`), and midnight-rollover quota settlement against original `usage_reservations.bucket_id` | `FencedGenerationDispatcher.java` | `settlesQuotaAgainstOriginalUtcDateBucketAcrossMidnightRollover` |
| **Day066** | Shared SSE Event Multiplexer (1 batched DB poller per replica, bounded 64-event subscriber queue, `Last-Event-ID` catch-up, and `resync.required` control frame on expired cursor) | `SharedSseEventMultiplexer.java`, `RunController.java`, `ml/src/scratchlm/client.py` | `sharedSseMultiplexerSupports50ConcurrentSubscribersAndEmitsResyncRequiredOnExpiredCursor` & `test_phase7_sse_resync_and_reliability.py` |
| **Day067** | Slow-reader queue overflow bounding (`SLOW_READER_BUFFER_OVERFLOW`), client disconnect vs. explicit cancellation separation, poison Kafka event routing to `.dlq.v1` + `dlq_events`, and manual DLQ replay skipping terminal runs | `SharedSseEventMultiplexer.java`, `OutboxRelayAndInboxService.java` | `poisonKafkaEventRoutesToDlqAndManualReplaySkipsCompletedRun` & `test_phase7_sse_resync_and_reliability.py` |
| **Day068** | Durable `model_promotions` + `active_model_releases` with monotonic `release_generation`, `model.released.v1` outbox event, stale release-generation rejection, and rollback to prior immutable revision via a newer `release_generation` | `ModelPromotionService.java`, `ModelCatalogController.java`, `ModelCatalogService.java` | `modelPromotionAdvancesMonotonicReleaseGenerationRejectsStaleEventsAndSupportsRollback` |
| **Day069 & Day070** | Full fault-injection & concurrency verification under 50 concurrent SSE subscribers, two-dispatcher fencing conflicts, broker outage recovery, and UTC midnight settlement | `Phase7OutboxKafkaFencingSseAndPromotionIntegrationTest.java`, `test_phase7_sse_resync_and_reliability.py` | `scripts/check_all.ps1` (0 errors, 0 failures) |

---

## 2. Fault-Injection & Reliability Summary

1. **Prolonged Kafka Outage & Outbox High-Water-Mark Backpressure:**
   - When `outbox_events` reaches `system_capacity.max_unpublished_outbox_events`, new run admissions fail fast with HTTP `503 OUTBOX_BACKPRESSURE_EXCEEDED` while idempotent replays of already-accepted runs continue to return `202 Accepted`.
   - Once the broker recovers, `OutboxRelayAndInboxService.relayAndConsumePending(...)` drains the backlog into Kafka and `consumer_inbox` without losing or duplicating a single run.
2. **Stale Worker Fencing & Orphan Recovery:**
   - When `dispatcher-A` (`attempt_token = 1`) stalls past `lease_expires_at` before emitting tokens, `reconcileExpiredLeases()` returns the run to `QUEUED` / `PENDING` and `dispatcher-B` claims it with `attempt_token = 2`.
   - Late heartbeats, delta writes, and terminal completion calls from `dispatcher-A` (`attempt_token = 1`) match `0` rows under `WHERE id = :runId AND attempt_token = :attemptToken AND status = 'RUNNING'` and are discarded.
   - When a worker stalls after emitting partial tokens, `reconcileExpiredLeases()` transitions the run to `FAILED` (`WORKER_LEASE_EXPIRED`), writes the partial assistant message (`status = 'PARTIAL_FAILED'`), and settles consumed tokens with `status = 'INCOMPLETE_SETTLED'`.
3. **50-Subscriber Shared SSE Multiplexer & `resync.required`:**
   - 50 concurrent SSE subscribers stream events simultaneously through a single scheduled batched PostgreSQL poller while Hikari `maximum-pool-size = 10` remains healthy.
   - Reconnecting with a pruned cursor (`Last-Event-ID < min_available_sequence - 1`) or overflowing a slow subscriber's bounded queue emits `event: resync.required` and directs the client to `GET /api/v1/runs/{runId}`.
4. **Monotonic Model Promotion & Rollback:**
   - Promoting `scratch-tiny` from `scratch-tiny-r1` (`release_generation = 1`) to `scratch-tiny-r2` (`release_generation = 2`) leaves pre-existing runs pinned to `scratch-tiny-r1`, routes new runs to `scratch-tiny-r2`, rejects stale `release_generation = 1` events, and rolls back cleanly to `scratch-tiny-r1` via monotonic `release_generation = 3`.
