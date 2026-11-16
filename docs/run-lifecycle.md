# Generation Run Lifecycle & Durable Execution Specification (Day061–Day070)

## 1. Authoritative State Machine

Every generation run (`generation_runs`) transitions monotonically through the following states:

```
QUEUED ──► RUNNING ──► COMPLETED
  │          │
  │          ├───────► CANCELLED
  │          │
  │          └───────► FAILED
  │
  ├──────────────────► CANCELLED
  └──────────────────► FAILED
```

- **Terminal states (`COMPLETED`, `CANCELLED`, `FAILED`) are immutable.** Once a run enters a terminal state, no worker, retry, or manual DLQ replay may transition it back to `QUEUED` or `RUNNING`.
- **Single active run per conversation:** Enforced by the partial unique index `uq_active_run_per_conversation ON generation_runs(conversation_id) WHERE status IN ('QUEUED', 'RUNNING')`.

---

## 2. Canonical Admission & Outbox Backpressure (Day062)

When `POST /api/v1/conversations/{conversationId}/runs` is called:

1. **Idempotent Replay First:** `idempotency_records` is checked before Redis rate limiting or capacity/quota locks so retrying an already-accepted request with the same `Idempotency-Key` and canonical request hash returns the original `202 Accepted` run metadata even during quota exhaustion or queue saturation.
2. **Canonical Lock Acquisition Order:**
   1. `system_capacity` (`WHERE id = 1 FOR UPDATE`)
   2. `usage_buckets` (`WHERE workspace_id = ? AND user_id = ? AND utc_date = ? FOR UPDATE`)
   3. `conversations` (`WHERE id = ? AND owner_user_id = ? AND deleted_at IS NULL FOR UPDATE`)
3. **Outbox High-Water-Mark Backpressure:**
   - Before accepting a new run, the transaction checks the count of unpublished rows in `outbox_events` (`WHERE published_at IS NULL`).
   - If `unpublished_count >= system_capacity.max_unpublished_outbox_events` (default `256`, configurable), the admission is rejected with `503 Service Unavailable` (`OUTBOX_BACKPRESSURE_EXCEEDED`), protecting PostgreSQL from unbounded outbox growth during a prolonged Kafka outage while preserving all previously accepted runs.
4. **Single-Transaction Write Set:**
   - `generation_runs` (`status = 'QUEUED'`, `attempt_token = 0`)
   - `messages` (`role = 'USER'`, `status = 'COMMITTED'`)
   - `usage_reservations` (`status = 'RESERVED'`, pinned to the admission UTC date's `bucket_id`)
   - `usage_buckets` (`reserved_tokens += total_reserved_tokens`)
   - `system_capacity` (`active_or_waiting_runs += 1`)
   - `idempotency_records` (`operation = 'SUBMIT_RUN'`)
   - `run_events` (`sequence = 1`, `event_type = 'run.queued'`)
   - `outbox_events` (`topic = 'generation.requested.v1'`, ID-only envelope)

---

## 3. Outbox Relay, Kafka Delivery & Inbox Deduplication (Day063 & Day067)

1. **Polling Outbox Relay (`OutboxRelayAndInboxService`):**
   - Claims unpublished rows from `outbox_events` using `FOR UPDATE SKIP LOCKED` with a short relay lease (`relay_lease_until`).
   - Validates the strict ID-only Kafka contract via `KafkaContractBridge` (`<= 4096` bytes, no raw prompts/chunks/weights).
   - Publishes to Kafka (`generation.requested.v1` or `model.released.v1`) keyed by `aggregate_key` and sets `published_at = clock_timestamp()` only after broker acknowledgement.
2. **Idempotent Consumer Inbox (`consumer_inbox`):**
   - On receiving a Kafka record, the consumer executes a single PostgreSQL transaction:
     - `INSERT INTO consumer_inbox (id, consumer_name, event_id, payload_hash) VALUES (?, ?, ?, ?) ON CONFLICT (consumer_name, event_id) DO NOTHING`
     - `INSERT INTO work_items (id, work_kind, target_id, state) VALUES (?, 'GENERATION_RUN', ?, 'PENDING') ON CONFLICT (work_kind, target_id) DO NOTHING`
   - Duplicate Kafka deliveries with the same `event_id` become no-ops and never create duplicate `work_items`.
3. **Bounded Retries, Dead-Letter Quarantine (`dlq_events`), and Manual Replay (Day067):**
   - Invalid envelopes or events failing after finite retries (`maxRetries = 3`) are published to `.dlq.v1` (`generation.requested.dlq.v1` / `model.released.dlq.v1`) and recorded in `dlq_events` (`status = 'QUARANTINED'`).
   - Operators can inspect quarantined events and trigger manual replay, which preserves the original `event_id` and re-validates target run state so completed/cancelled runs are never restarted.

---

## 4. Leased Execution, Monotonic Fencing Tokens & Orphan Recovery (Day064 & Day065)

1. **Claiming Work (`FOR UPDATE SKIP LOCKED`):**
   - `FencedGenerationDispatcher` claims eligible `work_items` (`state = 'PENDING'` or expired `state = 'LEASED'`) outside the Kafka consumer poll loop.
   - Increments `work_items.fencing_token` and transitions `generation_runs` to `status = 'RUNNING', attempt_token = attempt_token + 1`, returning the authoritative `attemptToken` for this execution attempt.
2. **Fenced Writes & Heartbeats:**
   - Long model inference calls execute **outside** PostgreSQL transactions while holding the single-GPU semaphore (`Semaphore(1, true)`).
   - A background heartbeat renews `work_items.lease_expires_at` ONLY if `work_items.fencing_token = :attemptToken AND state = 'LEASED'`.
   - Every chunk batch persistence (`text.delta`) and terminal transition (`COMPLETED`, `CANCELLED`, `FAILED`) enforces the fencing predicate:
     ```sql
     WHERE id = :runId AND attempt_token = :attemptToken AND status = 'RUNNING'
     ```
   - If a slow/stalled worker wakes up after its lease expired and another dispatcher claimed the run (`attempt_token = N + 1`), the stale worker's writes match `0` rows and are safely discarded without corrupting assistant messages, `run_events`, or quota settlement.
3. **Orphan / Expired-Lease Reconciliation:**
   - If a worker crashes and its `work_items.lease_expires_at` expires:
     - **Zero visible output tokens emitted (`text.delta` count = 0):** The reconciler safely resets `generation_runs.status = 'QUEUED'` and `work_items.state = 'PENDING'` for a fenced retry (up to `maxAttempts = 3`).
     - **Partial output already emitted (`text.delta` count > 0) or attempts exhausted:** The reconciler persists the partial assistant message (`status = 'PARTIAL_FAILED'`), transitions the run to `FAILED` (`error_code = 'WORKER_LEASE_EXPIRED'`), settles consumed tokens (`usage_reservations.status = 'INCOMPLETE_SETTLED'`) while refunding unused reserved tokens, and appends `run.failed`.
4. **Midnight-Safe Quota Settlement (Day065):**
   - Settlement (`SETTLED`, `REFUNDED`, or `INCOMPLETE_SETTLED`) always updates `usage_buckets` via `usage_reservations.bucket_id` (the bucket locked at admission time), never `CURRENT_DATE`, preventing negative or orphaned `reserved_tokens` when a run starts at `23:59:58 UTC` and finishes at `00:00:04 UTC`.

---

## 5. Shared SSE Event Multiplexer & `resync.required` (Day066 & Day067)

- Instead of dedicating a database-polling thread per connected SSE client, `SharedSseEventMultiplexer` runs **one shared scheduled batch poller** (`100 ms` interval) across all active `runId` subscriptions on the replica.
- Each subscriber has a bounded outbound queue (`maxBufferEvents = 64`).
- **Client Disconnect vs. Run Cancellation:** Closing the SSE HTTP stream removes the subscriber from `SharedSseEventMultiplexer` without cancelling the underlying `generation_runs` execution. Explicit `POST /api/v1/runs/{runId}/cancel` cancels the run.
- **Expired Cursor / Slow Reader Recovery (`resync.required`):**
  - `run_events` are retained for `24 hours`. If a reconnecting client presents a `Last-Event-ID` / `after` cursor that has been pruned (`cursor < min_available_sequence - 1`), or if a slow reader overflows its 64-event outbound buffer, the multiplexer emits an explicit `resync.required` SSE control event containing `{ "run_id": "...", "reason": "...", "snapshot_url": "/api/v1/runs/{runId}" }` and closes the stream cleanly so the client falls back to `GET /api/v1/runs/{runId}`.
