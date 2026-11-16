package com.scratchllm.platform.generation;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.scratchllm.platform.generation.GenerationProvider.FaultInjectionSpec;
import com.scratchllm.platform.generation.GenerationProvider.FinishReason;
import com.scratchllm.platform.generation.GenerationProvider.GenerationCommand;
import com.scratchllm.platform.generation.GenerationProvider.GenerationDeltaChunk;
import com.scratchllm.platform.generation.GenerationProvider.GenerationOutcome;
import com.scratchllm.platform.generation.GenerationProvider.SamplingConfig;
import com.scratchllm.platform.generation.RunDtos.SamplingOptionsDto;
import java.sql.Timestamp;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

/**
 * Day064 & Day065:
 * - Claims `work_items` with `FOR UPDATE SKIP LOCKED` outside the Kafka poll loop
 * - Increments monotonic `work_items.fencing_token` and `generation_runs.attempt_token`
 * - Renews worker leases via bounded background heartbeats during slow prefill/generation
 * - Enforces `WHERE id = :runId AND attempt_token = :attemptToken AND status = 'RUNNING'`
 *   on every chunk batch write and terminal state transition so stale/fenced workers cannot
 *   corrupt assistant messages, `run_events`, or quota settlement
 * - Reconciles expired/orphaned leases (`RUNNING -> QUEUED` before visible output, or
 *   `PARTIAL_FAILED` + `INCOMPLETE_SETTLED` if partial tokens were already emitted)
 * - Settles or refunds token reservations against the original `usage_reservations.bucket_id`
 *   even when completion crosses UTC midnight
 */
@Service
public class FencedGenerationDispatcher {

    public static final int MAX_EXECUTION_ATTEMPTS = 3;

    private final JdbcTemplate jdbcTemplate;
    private final TransactionTemplate transactionTemplate;
    private final GenerationProvider generationProvider;
    private final ObjectMapper objectMapper;
    private final ScheduledExecutorService heartbeatScheduler = Executors.newScheduledThreadPool(2, r -> {
        Thread t = new Thread(r, "fenced-dispatcher-heartbeat");
        t.setDaemon(true);
        return t;
    });

    public FencedGenerationDispatcher(
        JdbcTemplate jdbcTemplate,
        PlatformTransactionManager transactionManager,
        GenerationProvider generationProvider,
        ObjectMapper objectMapper
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.transactionTemplate = new TransactionTemplate(transactionManager);
        this.generationProvider = generationProvider;
        this.objectMapper = objectMapper;
    }

    @jakarta.annotation.PreDestroy
    public void shutdown() {
        heartbeatScheduler.shutdownNow();
    }

    /**
     * Claims the next eligible `GENERATION_RUN` work item using `FOR UPDATE SKIP LOCKED`,
     * increments `attempt_token` and `fencing_token`, and transitions the run to `RUNNING`.
     */
    public Optional<ClaimedWorkItem> claimNextWorkItem(String workerId, Duration leaseDuration) {
        ClaimedWorkItem claimed = transactionTemplate.execute(status -> {
            List<WorkItemRow> rows = jdbcTemplate.query(
                """
                SELECT id, target_id, state, fencing_token, attempt_count
                FROM work_items
                WHERE work_kind = 'GENERATION_RUN'
                  AND (
                    state = 'PENDING'
                    OR (state = 'LEASED' AND lease_expires_at <= clock_timestamp())
                  )
                  AND next_eligible_at <= clock_timestamp()
                ORDER BY next_eligible_at ASC, created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
                """,
                (rs, rowNum) -> new WorkItemRow(
                    UUID.fromString(rs.getString("id")),
                    UUID.fromString(rs.getString("target_id")),
                    rs.getString("state"),
                    rs.getLong("fencing_token"),
                    rs.getInt("attempt_count")
                )
            );
            if (rows.isEmpty()) {
                return null;
            }

            WorkItemRow work = rows.getFirst();
            UUID runId = work.targetId();

            List<RunLockRow> runRows = jdbcTemplate.query(
                """
                SELECT id, conversation_id, status, model_revision, prompt_snapshot,
                       sampling_json::text AS sampling_json, attempt_token, cancel_requested, input_tokens
                FROM generation_runs
                WHERE id = ?
                FOR UPDATE
                """,
                (rs, rowNum) -> new RunLockRow(
                    UUID.fromString(rs.getString("id")),
                    UUID.fromString(rs.getString("conversation_id")),
                    rs.getString("status"),
                    rs.getString("model_revision"),
                    rs.getString("prompt_snapshot"),
                    rs.getString("sampling_json"),
                    rs.getLong("attempt_token"),
                    rs.getBoolean("cancel_requested"),
                    rs.getInt("input_tokens")
                ),
                runId
            );
            if (runRows.isEmpty()) {
                jdbcTemplate.update(
                    "UPDATE work_items SET state = 'FAILED', last_error_code = 'RUN_NOT_FOUND', updated_at = clock_timestamp() WHERE id = ?",
                    work.id()
                );
                return null;
            }

            RunLockRow run = runRows.getFirst();
            if ("COMPLETED".equals(run.status()) || "CANCELLED".equals(run.status()) || "FAILED".equals(run.status())) {
                String terminalWorkState = "COMPLETED".equals(run.status()) ? "COMPLETED"
                    : ("CANCELLED".equals(run.status()) ? "CANCELLED" : "FAILED");
                jdbcTemplate.update(
                    "UPDATE work_items SET state = ?, lease_expires_at = NULL, updated_at = clock_timestamp() WHERE id = ?",
                    terminalWorkState, work.id()
                );
                return null;
            }

            if (run.cancelRequested()) {
                cancelRunInternal(runId);
                jdbcTemplate.update(
                    "UPDATE work_items SET state = 'CANCELLED', lease_expires_at = NULL, updated_at = clock_timestamp() WHERE id = ?",
                    work.id()
                );
                return null;
            }

            // If reclaiming an expired LEASED work item that was RUNNING, check whether visible deltas were already emitted
            if ("RUNNING".equals(run.status())) {
                int emittedDeltas = countEmittedDeltas(runId);
                if (emittedDeltas > 0 || work.attemptCount() >= MAX_EXECUTION_ATTEMPTS) {
                    String partialText = reconstructEmittedDeltaText(runId);
                    failRunWithPartialOutputInternal(
                        run.id(),
                        run.conversationId(),
                        run.attemptToken(),
                        "WORKER_LEASE_EXPIRED",
                        "Worker lease expired after emitting partial output; run terminated safely without duplicate retry.",
                        partialText,
                        run.inputTokens(),
                        emittedDeltas
                    );
                    jdbcTemplate.update(
                        """
                        UPDATE work_items
                        SET state = 'FAILED',
                            last_error_code = 'WORKER_LEASE_EXPIRED',
                            lease_expires_at = NULL,
                            updated_at = clock_timestamp()
                        WHERE id = ?
                        """,
                        work.id()
                    );
                    return null;
                }
            }

            long newAttemptToken = run.attemptToken() + 1L;
            int newAttemptCount = work.attemptCount() + 1;
            Timestamp leaseExpiresAt = Timestamp.from(Instant.now().plusMillis(Math.max(200L, leaseDuration.toMillis())));

            jdbcTemplate.update(
                """
                UPDATE generation_runs
                SET status = 'RUNNING',
                    attempt_token = ?,
                    started_at = COALESCE(started_at, clock_timestamp()),
                    updated_at = clock_timestamp()
                WHERE id = ?
                """,
                newAttemptToken, runId
            );

            jdbcTemplate.update(
                """
                UPDATE work_items
                SET state = 'LEASED',
                    lease_owner = ?,
                    fencing_token = ?,
                    attempt_count = ?,
                    lease_expires_at = ?,
                    last_heartbeat_at = clock_timestamp(),
                    updated_at = clock_timestamp()
                WHERE id = ?
                """,
                workerId, newAttemptToken, newAttemptCount, leaseExpiresAt, work.id()
            );

            appendRunEventInternal(runId, "run.started", Map.of(
                "run_id", runId.toString(),
                "schema_version", 1,
                "model_revision", run.modelRevision(),
                "attempt_token", newAttemptToken,
                "worker_id", workerId
            ));

            return new ClaimedWorkItem(
                work.id(),
                runId,
                run.conversationId(),
                workerId,
                newAttemptToken,
                newAttemptCount,
                run.modelRevision(),
                run.promptSnapshot(),
                run.samplingJson(),
                run.inputTokens()
            );
        });

        return Optional.ofNullable(claimed);
    }

    /**
     * Renews the worker's lease heartbeat ONLY if `fencing_token = attemptToken` and the run
     * is still `RUNNING` under the same `attempt_token`.
     */
    public boolean renewLeaseHeartbeat(UUID runId, long attemptToken, String workerId, Duration leaseDuration) {
        Boolean renewed = transactionTemplate.execute(status -> {
            Integer validRun = jdbcTemplate.queryForObject(
                """
                SELECT count(*)
                FROM generation_runs
                WHERE id = ? AND attempt_token = ? AND status = 'RUNNING' AND cancel_requested = FALSE
                """,
                Integer.class,
                runId, attemptToken
            );
            if (validRun == null || validRun == 0) {
                return false;
            }

            Timestamp newExpiry = Timestamp.from(Instant.now().plusMillis(Math.max(200L, leaseDuration.toMillis())));
            int updated = jdbcTemplate.update(
                """
                UPDATE work_items
                SET lease_expires_at = ?,
                    last_heartbeat_at = clock_timestamp(),
                    updated_at = clock_timestamp()
                WHERE work_kind = 'GENERATION_RUN'
                  AND target_id = ?
                  AND fencing_token = ?
                  AND state = 'LEASED'
                """,
                newExpiry, runId, attemptToken
            );
            return updated == 1;
        });
        return Boolean.TRUE.equals(renewed);
    }

    /**
     * Coalesces and persists a batch of token delta chunks into `run_events` strictly guarded by
     * `WHERE id = :runId AND attempt_token = :attemptToken AND status = 'RUNNING'`.
     * If a stale/fenced worker calls this after losing its lease, `0` rows match and nothing is written.
     */
    public boolean appendFencedDeltaBatch(UUID runId, long attemptToken, List<String> deltas) {
        if (deltas == null || deltas.isEmpty()) {
            return true;
        }
        String coalescedText = String.join("", deltas);
        if (coalescedText.isEmpty()) {
            return true;
        }

        Boolean persisted = transactionTemplate.execute(status -> {
            List<UUID> locked = jdbcTemplate.query(
                """
                SELECT id
                FROM generation_runs
                WHERE id = ? AND attempt_token = ? AND status = 'RUNNING' AND cancel_requested = FALSE
                FOR UPDATE
                """,
                (rs, rowNum) -> UUID.fromString(rs.getString("id")),
                runId, attemptToken
            );
            if (locked.isEmpty()) {
                return false;
            }

            appendRunEventInternal(runId, "text.delta", Map.of(
                "run_id", runId.toString(),
                "schema_version", 1,
                "attempt_token", attemptToken,
                "text", coalescedText
            ));
            return true;
        });
        return Boolean.TRUE.equals(persisted);
    }

    /**
     * Fenced idempotent terminal completion (`RUNNING -> COMPLETED`):
     * - Verifies `WHERE id = :runId AND attempt_token = :attemptToken AND status = 'RUNNING'`
     * - Writes the single `ASSISTANT` message (`uq_assistant_msg_per_run`)
     * - Settles token usage against the reservation's original `usage_reservations.bucket_id`
     *   (midnight-safe across UTC date rollovers!)
     * - Updates `work_items` to `COMPLETED` and appends `run.completed`
     */
    public boolean completeFencedRun(
        UUID runId,
        long attemptToken,
        FinishReason finishReason,
        String fullOutputText,
        int inputTokens,
        int outputTokens
    ) {
        Boolean completed = transactionTemplate.execute(status -> {
            List<UUID> convRows = jdbcTemplate.query(
                """
                SELECT conversation_id
                FROM generation_runs
                WHERE id = ? AND attempt_token = ? AND status = 'RUNNING'
                FOR UPDATE
                """,
                (rs, rowNum) -> UUID.fromString(rs.getString("conversation_id")),
                runId, attemptToken
            );
            if (convRows.isEmpty()) {
                return false;
            }
            UUID conversationId = convRows.getFirst();

            int updatedRun = jdbcTemplate.update(
                """
                UPDATE generation_runs
                SET status = 'COMPLETED',
                    finish_reason = ?,
                    input_tokens = ?,
                    output_tokens = ?,
                    completed_at = clock_timestamp(),
                    updated_at = clock_timestamp()
                WHERE id = ? AND attempt_token = ? AND status = 'RUNNING'
                """,
                finishReason.wireValue(),
                inputTokens,
                outputTokens,
                runId,
                attemptToken
            );
            if (updatedRun == 0) {
                return false;
            }

            Long maxOrder = jdbcTemplate.queryForObject(
                "SELECT COALESCE(MAX(message_order), 0) FROM messages WHERE conversation_id = ?",
                Long.class,
                conversationId
            );
            long assistantOrder = (maxOrder != null ? maxOrder : 0L) + 1L;

            jdbcTemplate.update(
                """
                INSERT INTO messages (id, conversation_id, run_id, role, content, status, message_order)
                VALUES (?, ?, ?, 'ASSISTANT', ?, 'COMMITTED', ?)
                ON CONFLICT (run_id) WHERE role = 'ASSISTANT' DO NOTHING
                """,
                UUID.randomUUID(),
                conversationId,
                runId,
                fullOutputText != null ? fullOutputText : "",
                assistantOrder
            );

            jdbcTemplate.update(
                """
                UPDATE work_items
                SET state = 'COMPLETED',
                    lease_expires_at = NULL,
                    updated_at = clock_timestamp()
                WHERE work_kind = 'GENERATION_RUN' AND target_id = ? AND fencing_token = ?
                """,
                runId, attemptToken
            );

            settleReservationToOriginalBucket(runId, inputTokens + outputTokens, "SETTLED");

            jdbcTemplate.update(
                "UPDATE system_capacity SET active_or_waiting_runs = GREATEST(0, active_or_waiting_runs - 1), updated_at = clock_timestamp() WHERE id = 1"
            );

            appendRunEventInternal(runId, "run.completed", Map.of(
                "run_id", runId.toString(),
                "schema_version", 1,
                "attempt_token", attemptToken,
                "finish_reason", finishReason.wireValue(),
                "input_tokens", inputTokens,
                "output_tokens", outputTokens
            ));
            return true;
        });
        return Boolean.TRUE.equals(completed);
    }

    /**
     * Fenced terminal failure transition (`RUNNING -> FAILED`), supporting both zero-output full refund
     * (`REFUNDED`) and partial-output incomplete settlement (`PARTIAL_FAILED` message + `INCOMPLETE_SETTLED`).
     */
    public boolean failFencedRun(
        UUID runId,
        long attemptToken,
        String errorCode,
        String errorMessage,
        String partialOutputText,
        int inputTokens,
        int emittedOutputTokens
    ) {
        Boolean failed = transactionTemplate.execute(status -> {
            List<UUID> convRows = jdbcTemplate.query(
                """
                SELECT conversation_id
                FROM generation_runs
                WHERE id = ? AND attempt_token = ? AND status = 'RUNNING'
                FOR UPDATE
                """,
                (rs, rowNum) -> UUID.fromString(rs.getString("conversation_id")),
                runId, attemptToken
            );
            if (convRows.isEmpty()) {
                return false;
            }
            UUID conversationId = convRows.getFirst();

            failRunWithPartialOutputInternal(
                runId,
                conversationId,
                attemptToken,
                errorCode != null ? errorCode : "PROVIDER_EXECUTION_ERROR",
                errorMessage != null ? errorMessage : "Generation failed inside provider.",
                partialOutputText,
                inputTokens,
                emittedOutputTokens
            );

            jdbcTemplate.update(
                """
                UPDATE work_items
                SET state = 'FAILED',
                    last_error_code = ?,
                    lease_expires_at = NULL,
                    updated_at = clock_timestamp()
                WHERE work_kind = 'GENERATION_RUN' AND target_id = ? AND fencing_token = ?
                """,
                errorCode != null ? errorCode : "PROVIDER_EXECUTION_ERROR",
                runId,
                attemptToken
            );
            return true;
        });
        return Boolean.TRUE.equals(failed);
    }

    /**
     * Reconciles all expired `LEASED` `work_items` (`lease_expires_at <= clock_timestamp()`):
     * - If zero `text.delta` chunks were emitted and `attempt_count < MAX_EXECUTION_ATTEMPTS`,
     *   transitions `generation_runs` back to `QUEUED` and `work_items` to `PENDING` for fenced retry.
     * - If partial `text.delta` chunks were already emitted (or attempts exhausted), transitions
     *   the run to `FAILED` (`WORKER_LEASE_EXPIRED`), persists the `PARTIAL_FAILED` assistant message,
     *   and settles consumed quota (`INCOMPLETE_SETTLED`) against the original UTC bucket.
     */
    public OrphanReconciliationSummary reconcileExpiredLeases() {
        List<UUID> expiredWorkItemIds = jdbcTemplate.query(
            """
            SELECT id
            FROM work_items
            WHERE work_kind = 'GENERATION_RUN'
              AND state = 'LEASED'
              AND lease_expires_at <= clock_timestamp()
            ORDER BY lease_expires_at ASC
            """,
            (rs, rowNum) -> UUID.fromString(rs.getString("id"))
        );

        int requeuedCount = 0;
        int failedPartialCount = 0;

        for (UUID workItemId : expiredWorkItemIds) {
            String action = transactionTemplate.execute(status -> {
                List<WorkItemRow> lockedWork = jdbcTemplate.query(
                    """
                    SELECT id, target_id, state, fencing_token, attempt_count
                    FROM work_items
                    WHERE id = ? AND state = 'LEASED' AND lease_expires_at <= clock_timestamp()
                    FOR UPDATE SKIP LOCKED
                    """,
                    (rs, rowNum) -> new WorkItemRow(
                        UUID.fromString(rs.getString("id")),
                        UUID.fromString(rs.getString("target_id")),
                        rs.getString("state"),
                        rs.getLong("fencing_token"),
                        rs.getInt("attempt_count")
                    ),
                    workItemId
                );
                if (lockedWork.isEmpty()) {
                    return "NONE";
                }

                WorkItemRow work = lockedWork.getFirst();
                UUID runId = work.targetId();

                List<RunLockRow> runRows = jdbcTemplate.query(
                    """
                    SELECT id, conversation_id, status, model_revision, prompt_snapshot,
                           sampling_json::text AS sampling_json, attempt_token, cancel_requested, input_tokens
                    FROM generation_runs
                    WHERE id = ?
                    FOR UPDATE
                    """,
                    (rs, rowNum) -> new RunLockRow(
                        UUID.fromString(rs.getString("id")),
                        UUID.fromString(rs.getString("conversation_id")),
                        rs.getString("status"),
                        rs.getString("model_revision"),
                        rs.getString("prompt_snapshot"),
                        rs.getString("sampling_json"),
                        rs.getLong("attempt_token"),
                        rs.getBoolean("cancel_requested"),
                        rs.getInt("input_tokens")
                    ),
                    runId
                );
                if (runRows.isEmpty()) {
                    return "NONE";
                }

                RunLockRow run = runRows.getFirst();
                if (!"RUNNING".equals(run.status())) {
                    return "NONE";
                }

                int deltaCount = countEmittedDeltas(runId);
                if (deltaCount == 0 && work.attemptCount() < MAX_EXECUTION_ATTEMPTS) {
                    jdbcTemplate.update(
                        """
                        UPDATE generation_runs
                        SET status = 'QUEUED',
                            updated_at = clock_timestamp()
                        WHERE id = ? AND attempt_token = ? AND status = 'RUNNING'
                        """,
                        runId, run.attemptToken()
                    );
                    jdbcTemplate.update(
                        """
                        UPDATE work_items
                        SET state = 'PENDING',
                            lease_owner = NULL,
                            lease_expires_at = NULL,
                            updated_at = clock_timestamp()
                        WHERE id = ?
                        """,
                        work.id()
                    );
                    return "REQUEUED";
                } else {
                    String partialText = reconstructEmittedDeltaText(runId);
                    failRunWithPartialOutputInternal(
                        runId,
                        run.conversationId(),
                        run.attemptToken(),
                        "WORKER_LEASE_EXPIRED",
                        "Worker lease expired after partial output; terminated safely.",
                        partialText,
                        run.inputTokens(),
                        deltaCount
                    );
                    jdbcTemplate.update(
                        """
                        UPDATE work_items
                        SET state = 'FAILED',
                            last_error_code = 'WORKER_LEASE_EXPIRED',
                            lease_expires_at = NULL,
                            updated_at = clock_timestamp()
                        WHERE id = ?
                        """,
                        work.id()
                    );
                    return "FAILED_PARTIAL";
                }
            });

            if ("REQUEUED".equals(action)) {
                requeuedCount++;
            } else if ("FAILED_PARTIAL".equals(action)) {
                failedPartialCount++;
            }
        }

        return new OrphanReconciliationSummary(requeuedCount, failedPartialCount);
    }

    /**
     * Claims and executes the next pending/expired `work_item` end-to-end with heartbeat renewal,
     * batched delta chunk persistence, and fenced terminal settlement.
     */
    public Optional<ClaimedWorkItem> dispatchAndExecuteNext(String workerId, Duration leaseDuration, int chunkBatchSize) {
        Optional<ClaimedWorkItem> claimedOpt = claimNextWorkItem(workerId, leaseDuration);
        if (claimedOpt.isEmpty()) {
            return Optional.empty();
        }
        ClaimedWorkItem claimed = claimedOpt.get();
        executeClaimedWorkItem(claimed, leaseDuration, Math.max(1, chunkBatchSize));
        return Optional.of(claimed);
    }

    public void executeClaimedWorkItem(ClaimedWorkItem claimed, Duration leaseDuration, int chunkBatchSize) {
        long heartbeatPeriodMs = Math.max(50L, leaseDuration.toMillis() / 3L);
        AtomicBoolean fencedOrCancelled = new AtomicBoolean(false);

        ScheduledFuture<?> heartbeatTask = heartbeatScheduler.scheduleAtFixedRate(
            () -> {
                boolean ok = renewLeaseHeartbeat(claimed.runId(), claimed.attemptToken(), claimed.workerId(), leaseDuration);
                if (!ok) {
                    fencedOrCancelled.set(true);
                }
            },
            heartbeatPeriodMs,
            heartbeatPeriodMs,
            TimeUnit.MILLISECONDS
        );

        try {
            SamplingOptionsDto sampling = parseSampling(claimed.samplingJson());
            long delayMs = (claimed.promptSnapshot().contains("[SLOW_FAKE]") || claimed.promptSnapshot().contains("[SLOW_REAL]"))
                ? 80L
                : 5L;
            FaultInjectionSpec faultSpec = resolveFaultSpec(claimed.promptSnapshot(), delayMs);

            StringBuilder fullOutput = new StringBuilder();
            List<String> pendingDeltaBatch = new ArrayList<>();
            final int[] emittedChunks = {0};

            GenerationCommand command = new GenerationCommand(
                claimed.runId(),
                claimed.attemptToken(),
                claimed.modelRevision(),
                claimed.promptSnapshot(),
                128,
                new SamplingConfig(
                    sampling.max_new_tokens(),
                    sampling.temperature(),
                    sampling.top_k(),
                    sampling.top_p(),
                    sampling.seed() != null ? sampling.seed() : 42L
                ),
                Instant.now().plusSeconds(60),
                faultSpec
            );

            GenerationOutcome outcome = generationProvider.generateStream(
                command,
                (GenerationDeltaChunk chunk) -> {
                    if (fencedOrCancelled.get()) {
                        return;
                    }
                    fullOutput.append(chunk.textDelta());
                    emittedChunks[0]++;
                    pendingDeltaBatch.add(chunk.textDelta());
                    if (pendingDeltaBatch.size() >= chunkBatchSize) {
                        boolean persisted = appendFencedDeltaBatch(
                            claimed.runId(),
                            claimed.attemptToken(),
                            new ArrayList<>(pendingDeltaBatch)
                        );
                        pendingDeltaBatch.clear();
                        if (!persisted) {
                            fencedOrCancelled.set(true);
                        }
                    }
                },
                () -> {
                    if (fencedOrCancelled.get()) {
                        return true;
                    }
                    Boolean invalid = jdbcTemplate.queryForObject(
                        """
                        SELECT cancel_requested OR status <> 'RUNNING' OR attempt_token <> ?
                        FROM generation_runs
                        WHERE id = ?
                        """,
                        Boolean.class,
                        claimed.attemptToken(),
                        claimed.runId()
                    );
                    if (Boolean.TRUE.equals(invalid)) {
                        fencedOrCancelled.set(true);
                        return true;
                    }
                    return false;
                }
            );

            if (!pendingDeltaBatch.isEmpty() && !fencedOrCancelled.get()) {
                boolean persisted = appendFencedDeltaBatch(
                    claimed.runId(),
                    claimed.attemptToken(),
                    new ArrayList<>(pendingDeltaBatch)
                );
                pendingDeltaBatch.clear();
                if (!persisted) {
                    fencedOrCancelled.set(true);
                }
            }

            if (fencedOrCancelled.get()) {
                // Check if explicit cancel was requested while we still hold the current attempt_token
                transactionTemplate.executeWithoutResult(status -> {
                    Boolean cancelReq = jdbcTemplate.queryForObject(
                        "SELECT cancel_requested AND status = 'RUNNING' AND attempt_token = ? FROM generation_runs WHERE id = ?",
                        Boolean.class,
                        claimed.attemptToken(),
                        claimed.runId()
                    );
                    if (Boolean.TRUE.equals(cancelReq)) {
                        cancelRunInternal(claimed.runId());
                        jdbcTemplate.update(
                            "UPDATE work_items SET state = 'CANCELLED', lease_expires_at = NULL, updated_at = clock_timestamp() WHERE id = ?",
                            claimed.workItemId()
                        );
                    }
                });
                return;
            }

            if (outcome.finishReason() == FinishReason.CANCELLED) {
                transactionTemplate.executeWithoutResult(status -> {
                    cancelRunInternal(claimed.runId());
                    jdbcTemplate.update(
                        "UPDATE work_items SET state = 'CANCELLED', lease_expires_at = NULL, updated_at = clock_timestamp() WHERE id = ?",
                        claimed.workItemId()
                    );
                });
                return;
            }

            if (!outcome.isSuccess()) {
                failFencedRun(
                    claimed.runId(),
                    claimed.attemptToken(),
                    outcome.errorCode(),
                    outcome.errorMessage(),
                    fullOutput.toString(),
                    claimed.inputTokens(),
                    emittedChunks[0]
                );
                return;
            }

            completeFencedRun(
                claimed.runId(),
                claimed.attemptToken(),
                outcome.finishReason(),
                fullOutput.toString(),
                outcome.inputTokens(),
                outcome.outputTokens()
            );
        } finally {
            heartbeatTask.cancel(false);
        }
    }

    /**
     * Settles or partially settles a run's token reservation against the reservation's
     * **original `usage_reservations.bucket_id`** (never `CURRENT_DATE`), ensuring
     * runs that start before UTC midnight (`23:59:58Z`) and finish after (`00:00:03Z`)
     * never corrupt or strand quota in either day's bucket.
     */
    public void settleReservationToOriginalBucket(UUID runId, int actualSettledTokens, String targetStatus) {
        jdbcTemplate.query(
            """
            UPDATE usage_reservations
            SET status = ?,
                settled_tokens = ?,
                settled_at = clock_timestamp()
            WHERE run_id = ? AND status = 'RESERVED'
            RETURNING bucket_id, reserved_tokens
            """,
            (rs, rowNum) -> {
                UUID bucketId = UUID.fromString(rs.getString("bucket_id"));
                int reserved = rs.getInt("reserved_tokens");
                int clampedSettled = Math.max(0, Math.min(reserved, actualSettledTokens));
                jdbcTemplate.update(
                    """
                    UPDATE usage_buckets
                    SET reserved_tokens = GREATEST(0, reserved_tokens - ?),
                        settled_tokens = settled_tokens + ?,
                        updated_at = clock_timestamp()
                    WHERE id = ?
                    """,
                    reserved, clampedSettled, bucketId
                );
                return bucketId;
            },
            targetStatus,
            Math.max(0, actualSettledTokens),
            runId
        );
    }

    public void refundReservationToOriginalBucket(UUID runId) {
        jdbcTemplate.query(
            """
            UPDATE usage_reservations
            SET status = 'REFUNDED',
                settled_tokens = 0,
                settled_at = clock_timestamp()
            WHERE run_id = ? AND status = 'RESERVED'
            RETURNING bucket_id, reserved_tokens
            """,
            (rs, rowNum) -> {
                UUID bucketId = UUID.fromString(rs.getString("bucket_id"));
                int reserved = rs.getInt("reserved_tokens");
                jdbcTemplate.update(
                    """
                    UPDATE usage_buckets
                    SET reserved_tokens = GREATEST(0, reserved_tokens - ?),
                        updated_at = clock_timestamp()
                    WHERE id = ?
                    """,
                    reserved, bucketId
                );
                return bucketId;
            },
            runId
        );
    }

    private void failRunWithPartialOutputInternal(
        UUID runId,
        UUID conversationId,
        long attemptToken,
        String errorCode,
        String errorMessage,
        String partialOutputText,
        int inputTokens,
        int emittedOutputTokens
    ) {
        int updated = jdbcTemplate.update(
            """
            UPDATE generation_runs
            SET status = 'FAILED',
                finish_reason = 'error',
                error_code = ?,
                output_tokens = ?,
                completed_at = clock_timestamp(),
                updated_at = clock_timestamp()
            WHERE id = ? AND attempt_token = ? AND status = 'RUNNING'
            """,
            errorCode,
            Math.max(0, emittedOutputTokens),
            runId,
            attemptToken
        );
        if (updated == 0) {
            return;
        }

        if (partialOutputText != null && !partialOutputText.isEmpty()) {
            Long maxOrder = jdbcTemplate.queryForObject(
                "SELECT COALESCE(MAX(message_order), 0) FROM messages WHERE conversation_id = ?",
                Long.class,
                conversationId
            );
            long assistantOrder = (maxOrder != null ? maxOrder : 0L) + 1L;
            jdbcTemplate.update(
                """
                INSERT INTO messages (id, conversation_id, run_id, role, content, status, message_order)
                VALUES (?, ?, ?, 'ASSISTANT', ?, 'PARTIAL_FAILED', ?)
                ON CONFLICT (run_id) WHERE role = 'ASSISTANT' DO NOTHING
                """,
                UUID.randomUUID(),
                conversationId,
                runId,
                partialOutputText,
                assistantOrder
            );
            settleReservationToOriginalBucket(
                runId,
                Math.max(1, inputTokens + emittedOutputTokens),
                "INCOMPLETE_SETTLED"
            );
        } else {
            refundReservationToOriginalBucket(runId);
        }

        jdbcTemplate.update(
            "UPDATE system_capacity SET active_or_waiting_runs = GREATEST(0, active_or_waiting_runs - 1), updated_at = clock_timestamp() WHERE id = 1"
        );

        appendRunEventInternal(runId, "run.failed", Map.of(
            "run_id", runId.toString(),
            "schema_version", 1,
            "attempt_token", attemptToken,
            "error_code", errorCode,
            "message", errorMessage
        ));
    }

    private void cancelRunInternal(UUID runId) {
        int updated = jdbcTemplate.update(
            """
            UPDATE generation_runs
            SET status = 'CANCELLED',
                cancel_requested = TRUE,
                finish_reason = 'cancelled',
                completed_at = clock_timestamp(),
                updated_at = clock_timestamp()
            WHERE id = ? AND status IN ('QUEUED', 'RUNNING')
            """,
            runId
        );
        if (updated > 0) {
            refundReservationToOriginalBucket(runId);
            jdbcTemplate.update(
                "UPDATE system_capacity SET active_or_waiting_runs = GREATEST(0, active_or_waiting_runs - 1), updated_at = clock_timestamp() WHERE id = 1"
            );
            appendRunEventInternal(runId, "run.cancelled", Map.of(
                "run_id", runId.toString(),
                "schema_version", 1,
                "finish_reason", "cancelled"
            ));
        }
    }

    private int countEmittedDeltas(UUID runId) {
        Integer count = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM run_events WHERE run_id = ? AND event_type = 'text.delta'",
            Integer.class,
            runId
        );
        return count != null ? count : 0;
    }

    private String reconstructEmittedDeltaText(UUID runId) {
        List<String> payloads = jdbcTemplate.query(
            """
            SELECT payload_json::text AS payload_json
            FROM run_events
            WHERE run_id = ? AND event_type = 'text.delta'
            ORDER BY sequence ASC
            """,
            (rs, rowNum) -> rs.getString("payload_json"),
            runId
        );
        StringBuilder sb = new StringBuilder();
        for (String p : payloads) {
            try {
                JsonNode node = objectMapper.readTree(p);
                if (node.hasNonNull("text")) {
                    sb.append(node.get("text").asText());
                }
            } catch (Exception ignored) {
            }
        }
        return sb.toString();
    }

    private void appendRunEventInternal(UUID runId, String eventType, Map<String, Object> payload) {
        Long maxSeq = jdbcTemplate.queryForObject(
            "SELECT COALESCE(MAX(sequence), 0) FROM run_events WHERE run_id = ?",
            Long.class,
            runId
        );
        long nextSeq = (maxSeq != null ? maxSeq : 0L) + 1L;
        jdbcTemplate.update(
            """
            INSERT INTO run_events (run_id, sequence, event_type, payload_json)
            VALUES (?, ?, ?, ?::jsonb)
            """,
            runId, nextSeq, eventType, toJson(payload)
        );
    }

    private static FaultInjectionSpec resolveFaultSpec(String prompt, long delayMs) {
        if (prompt.contains("[FAULT:UPSTREAM_503]")) {
            return new FaultInjectionSpec(delayMs, 0, "UPSTREAM_503");
        }
        if (prompt.contains("[FAULT:DISCONNECT]")) {
            return new FaultInjectionSpec(delayMs, 1, "STREAM_DISCONNECTED");
        }
        return new FaultInjectionSpec(delayMs, null, null);
    }

    private SamplingOptionsDto parseSampling(String json) {
        try {
            return objectMapper.readValue(json, SamplingOptionsDto.class);
        } catch (JsonProcessingException ex) {
            return new SamplingOptionsDto(16, 0.8, 40, 0.95, 42L);
        }
    }

    private String toJson(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException ex) {
            throw new IllegalStateException("Failed to serialize JSON", ex);
        }
    }

    public record ClaimedWorkItem(
        UUID workItemId,
        UUID runId,
        UUID conversationId,
        String workerId,
        long attemptToken,
        int attemptCount,
        String modelRevision,
        String promptSnapshot,
        String samplingJson,
        int inputTokens
    ) {}

    public record OrphanReconciliationSummary(
        int requeuedRuns,
        int failedPartialRuns
    ) {}

    private record WorkItemRow(
        UUID id,
        UUID targetId,
        String state,
        long fencingToken,
        int attemptCount
    ) {}

    private record RunLockRow(
        UUID id,
        UUID conversationId,
        String status,
        String modelRevision,
        String promptSnapshot,
        String samplingJson,
        long attemptToken,
        boolean cancelRequested,
        int inputTokens
    ) {}
}
