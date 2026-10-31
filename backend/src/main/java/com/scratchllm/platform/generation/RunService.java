package com.scratchllm.platform.generation;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.scratchllm.platform.generation.GenerationProvider.FaultInjectionSpec;
import com.scratchllm.platform.generation.GenerationProvider.FinishReason;
import com.scratchllm.platform.generation.GenerationProvider.GenerationCommand;
import com.scratchllm.platform.generation.GenerationProvider.GenerationDeltaChunk;
import com.scratchllm.platform.generation.GenerationProvider.GenerationOutcome;
import com.scratchllm.platform.generation.GenerationProvider.SamplingConfig;
import com.scratchllm.platform.generation.RunDtos.RunAcceptedResponse;
import com.scratchllm.platform.generation.RunDtos.RunEventRow;
import com.scratchllm.platform.generation.RunDtos.RunStatusResponse;
import com.scratchllm.platform.generation.RunDtos.SamplingOptionsDto;
import com.scratchllm.platform.generation.RunDtos.SubmitRunRequest;
import com.scratchllm.platform.infrastructure.RedisRateLimiterService;
import com.scratchllm.platform.platform.ApiException;
import jakarta.annotation.PreDestroy;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;
import org.springframework.transaction.support.TransactionTemplate;

/**
 * Day016 & Day018:
 * - Idempotent run creation in PostgreSQL with canonical client request hashing (independent of model alias resolution)
 * - Replay verification BEFORE rate-limit or capacity/quota admission
 * - Canonical deadlock-free lock hierarchy (system_capacity -> usage_buckets -> conversations)
 * - Transactional outbox event creation (generation.requested.v1)
 * - Bounded Phase 2 fake-provider execution, SSE event persistence, and idempotent cancellation
 */
@Service
public class RunService {

    private final JdbcTemplate jdbcTemplate;
    private final TransactionTemplate transactionTemplate;
    private final RedisRateLimiterService rateLimiterService;
    private final GenerationProvider generationProvider;
    private final ObjectMapper objectMapper;
    private final int defaultDailyAllowance;
    private final boolean inlineFakeDispatchEnabled;
    private final ExecutorService fakeExecutor = Executors.newFixedThreadPool(2);

    public RunService(
        JdbcTemplate jdbcTemplate,
        PlatformTransactionManager transactionManager,
        RedisRateLimiterService rateLimiterService,
        GenerationProvider generationProvider,
        ObjectMapper objectMapper,
        @Value("${scratchllm.quota.daily-allowance-tokens:4096}") int defaultDailyAllowance,
        @Value("${scratchllm.execution.inline-fake-dispatch-enabled:true}") boolean inlineFakeDispatchEnabled
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.transactionTemplate = new TransactionTemplate(transactionManager);
        this.rateLimiterService = rateLimiterService;
        this.generationProvider = generationProvider;
        this.objectMapper = objectMapper;
        this.defaultDailyAllowance = defaultDailyAllowance;
        this.inlineFakeDispatchEnabled = inlineFakeDispatchEnabled;
    }

    @PreDestroy
    public void shutdown() {
        fakeExecutor.shutdownNow();
    }

    @Transactional
    public RunAcceptedResponse submitRun(
        UUID userId,
        UUID conversationId,
        String idempotencyKey,
        SubmitRunRequest request
    ) {
        if (idempotencyKey == null || idempotencyKey.trim().length() < 8 || idempotencyKey.trim().length() > 128) {
            throw new ApiException(
                HttpStatus.BAD_REQUEST,
                "INVALID_IDEMPOTENCY_KEY",
                "Idempotency-Key header must be between 8 and 128 characters."
            );
        }
        String cleanKey = idempotencyKey.trim();

        // 1. Verify caller owns the conversation and obtain workspace_id
        List<ConversationMeta> convRows = jdbcTemplate.query(
            """
            SELECT id, workspace_id, owner_user_id
            FROM conversations
            WHERE id = ? AND owner_user_id = ? AND deleted_at IS NULL
            """,
            (rs, rowNum) -> new ConversationMeta(
                UUID.fromString(rs.getString("id")),
                UUID.fromString(rs.getString("workspace_id")),
                UUID.fromString(rs.getString("owner_user_id"))
            ),
            conversationId, userId
        );
        if (convRows.isEmpty()) {
            throw new ApiException(HttpStatus.NOT_FOUND, "CONVERSATION_NOT_FOUND", "Conversation not found.");
        }
        UUID workspaceId = convRows.getFirst().workspaceId();

        // 2. Compute canonical client request hash independently of model_alias -> revision resolution
        String canonicalRequestHash = computeCanonicalRequestHash(conversationId, request);

        // 3. Check idempotent replay BEFORE rate-limiting, capacity checks, or quota admission!
        OptionalReplay existingReplay = checkExistingIdempotencyRecord(
            workspaceId,
            userId,
            cleanKey,
            canonicalRequestHash
        );
        if (existingReplay.found()) {
            return existingReplay.response();
        }

        // 4. Enforce Redis atomic Lua rate limit for new logical run submissions
        rateLimiterService.enforceRunSubmissionLimit(userId.toString());

        // 5. Resolve model_alias to READY revision_tag
        List<ModelRevisionMeta> modelRows = jdbcTemplate.query(
            """
            SELECT revision_tag, context_limit, max_output_tokens
            FROM model_revisions
            WHERE model_alias = ? AND readiness_status = 'READY'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (rs, rowNum) -> new ModelRevisionMeta(
                rs.getString("revision_tag"),
                rs.getInt("context_limit"),
                rs.getInt("max_output_tokens")
            ),
            request.model_alias()
        );
        if (modelRows.isEmpty()) {
            throw new ApiException(
                HttpStatus.BAD_REQUEST,
                "MODEL_NOT_AVAILABLE",
                "Requested model_alias is unknown or not READY: " + request.model_alias()
            );
        }
        ModelRevisionMeta modelMeta = modelRows.getFirst();

        // 6. Acquire PostgreSQL locks in strict canonical order:
        //    (a) system_capacity -> (b) usage_buckets -> (c) conversations
        CapacityRow capacity = jdbcTemplate.queryForObject(
            "SELECT max_waiting_runs, active_or_waiting_runs FROM system_capacity WHERE id = 1 FOR UPDATE",
            (rs, rowNum) -> new CapacityRow(rs.getInt("max_waiting_runs"), rs.getInt("active_or_waiting_runs"))
        );

        LocalDate todayUtc = LocalDate.now(ZoneOffset.UTC);
        jdbcTemplate.update(
            """
            INSERT INTO usage_buckets (id, workspace_id, user_id, utc_date, daily_allowance_tokens, reserved_tokens, settled_tokens)
            VALUES (?, ?, ?, ?, ?, 0, 0)
            ON CONFLICT (workspace_id, user_id, utc_date) DO NOTHING
            """,
            UUID.randomUUID(), workspaceId, userId, todayUtc, defaultDailyAllowance
        );

        UsageBucketRow bucket = jdbcTemplate.queryForObject(
            """
            SELECT id, daily_allowance_tokens, reserved_tokens, settled_tokens
            FROM usage_buckets
            WHERE workspace_id = ? AND user_id = ? AND utc_date = ?
            FOR UPDATE
            """,
            (rs, rowNum) -> new UsageBucketRow(
                UUID.fromString(rs.getString("id")),
                rs.getInt("daily_allowance_tokens"),
                rs.getInt("reserved_tokens"),
                rs.getInt("settled_tokens")
            ),
            workspaceId, userId, todayUtc
        );

        List<UUID> lockedConv = jdbcTemplate.query(
            "SELECT id FROM conversations WHERE id = ? AND owner_user_id = ? AND deleted_at IS NULL FOR UPDATE",
            (rs, rowNum) -> UUID.fromString(rs.getString("id")),
            conversationId, userId
        );
        if (lockedConv.isEmpty()) {
            throw new ApiException(HttpStatus.NOT_FOUND, "CONVERSATION_NOT_FOUND", "Conversation not found.");
        }

        // Re-check idempotency inside the lock in case a concurrent duplicate committed just before us
        OptionalReplay lockedReplay = checkExistingIdempotencyRecord(
            workspaceId,
            userId,
            cleanKey,
            canonicalRequestHash
        );
        if (lockedReplay.found()) {
            return lockedReplay.response();
        }

        // Check active run per conversation
        Integer activeInConv = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM generation_runs WHERE conversation_id = ? AND status IN ('QUEUED', 'RUNNING')",
            Integer.class,
            conversationId
        );
        if (activeInConv != null && activeInConv > 0) {
            throw new ApiException(
                HttpStatus.CONFLICT,
                "ACTIVE_RUN_EXISTS",
                "Conversation already has a queued or running generation run."
            );
        }

        // Check system capacity
        if (capacity.activeOrWaitingRuns() >= capacity.maxWaitingRuns()) {
            throw new ApiException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "SYSTEM_CAPACITY_EXCEEDED",
                "System queue capacity is full; retry later."
            );
        }

        // Compute conservative token reservation
        int estimatedPromptTokens = Math.max(1, (request.prompt().length() + 3) / 4);
        int requestedOutputTokens = request.sampling().max_new_tokens();
        int totalReservedTokens = estimatedPromptTokens + requestedOutputTokens;

        if (bucket.reservedTokens() + bucket.settledTokens() + totalReservedTokens > bucket.dailyAllowanceTokens()) {
            throw new ApiException(
                HttpStatus.TOO_MANY_REQUESTS,
                "QUOTA_EXHAUSTED",
                "Daily token allowance is insufficient for this request."
            );
        }

        // 7. Atomically insert USER message, generation_run, usage_reservation, idempotency_record, run.queued event, and outbox_event
        Long maxOrder = jdbcTemplate.queryForObject(
            "SELECT COALESCE(MAX(message_order), 0) FROM messages WHERE conversation_id = ?",
            Long.class,
            conversationId
        );
        long userMsgOrder = (maxOrder != null ? maxOrder : 0L) + 1L;

        UUID runId = UUID.randomUUID();
        UUID userMessageId = UUID.randomUUID();
        OffsetDateTime now = OffsetDateTime.now(ZoneOffset.UTC);
        OffsetDateTime queueDeadline = now.plusSeconds(60);
        OffsetDateTime execDeadline = now.plusSeconds(120);
        String promptHash = sha256Hex(request.prompt());
        String samplingJson = toJson(request.sampling());

        try {
            jdbcTemplate.update(
                """
                INSERT INTO generation_runs (
                    id, workspace_id, conversation_id, user_id, status, model_revision,
                    prompt_snapshot, prompt_snapshot_hash, sampling_json,
                    queue_deadline_at, execution_deadline_at, input_tokens, output_tokens
                ) VALUES (?, ?, ?, ?, 'QUEUED', ?, ?, ?, ?::jsonb, ?, ?, ?, 0)
                """,
                runId, workspaceId, conversationId, userId, modelMeta.revisionTag(),
                request.prompt(), promptHash, samplingJson,
                Timestamp.from(queueDeadline.toInstant()),
                Timestamp.from(execDeadline.toInstant()),
                estimatedPromptTokens
            );
        } catch (DataIntegrityViolationException ex) {
            throw new ApiException(
                HttpStatus.CONFLICT,
                "ACTIVE_RUN_EXISTS",
                "Conversation already has an active run."
            );
        }

        jdbcTemplate.update(
            """
            INSERT INTO messages (id, conversation_id, run_id, role, content, status, message_order)
            VALUES (?, ?, ?, 'USER', ?, 'COMMITTED', ?)
            """,
            userMessageId, conversationId, runId, request.prompt(), userMsgOrder
        );

        jdbcTemplate.update(
            "UPDATE conversations SET updated_at = clock_timestamp() WHERE id = ?",
            conversationId
        );

        jdbcTemplate.update(
            """
            INSERT INTO usage_reservations (id, run_id, bucket_id, reserved_tokens, status)
            VALUES (?, ?, ?, ?, 'RESERVED')
            """,
            UUID.randomUUID(), runId, bucket.id(), totalReservedTokens
        );

        jdbcTemplate.update(
            "UPDATE usage_buckets SET reserved_tokens = reserved_tokens + ?, updated_at = clock_timestamp() WHERE id = ?",
            totalReservedTokens, bucket.id()
        );

        jdbcTemplate.update(
            "UPDATE system_capacity SET active_or_waiting_runs = active_or_waiting_runs + 1, updated_at = clock_timestamp() WHERE id = 1"
        );

        jdbcTemplate.update(
            """
            INSERT INTO idempotency_records (
                id, workspace_id, user_id, operation, idempotency_key, request_hash, run_id, expires_at
            ) VALUES (?, ?, ?, 'SUBMIT_RUN', ?, ?, ?, ?)
            """,
            UUID.randomUUID(),
            workspaceId,
            userId,
            cleanKey,
            canonicalRequestHash,
            runId,
            Timestamp.from(now.plusHours(24).toInstant())
        );

        String queuedEventPayload = toJson(Map.of(
            "run_id", runId.toString(),
            "schema_version", 1,
            "model_revision", modelMeta.revisionTag()
        ));
        jdbcTemplate.update(
            """
            INSERT INTO run_events (run_id, sequence, event_type, payload_json)
            VALUES (?, 1, 'run.queued', ?::jsonb)
            """,
            runId, queuedEventPayload
        );

        UUID outboxEventId = UUID.randomUUID();
        String outboxPayload = toJson(Map.of(
            "event_id", outboxEventId.toString(),
            "event_type", "generation.requested.v1",
            "schema_version", 1,
            "occurred_at", now.toString(),
            "workspace_id", workspaceId.toString(),
            "conversation_id", conversationId.toString(),
            "run_id", runId.toString(),
            "user_id", userId.toString(),
            "model_revision", modelMeta.revisionTag()
        ));
        jdbcTemplate.update(
            """
            INSERT INTO outbox_events (event_id, topic, schema_version, aggregate_key, payload_json)
            VALUES (?, 'generation.requested.v1', 1, ?, ?::jsonb)
            """,
            outboxEventId, conversationId.toString(), outboxPayload
        );

        RunAcceptedResponse response = buildAcceptedResponse(runId, conversationId, "QUEUED", modelMeta.revisionTag());

        if (inlineFakeDispatchEnabled && TransactionSynchronizationManager.isSynchronizationActive()) {
            TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
                @Override
                public void afterCommit() {
                    fakeExecutor.submit(() -> executeFakeRun(runId));
                }
            });
        }

        return response;
    }

    @Transactional(readOnly = true)
    public RunStatusResponse getRunStatus(UUID userId, UUID runId) {
        List<RunStatusResponse> rows = jdbcTemplate.query(
            """
            SELECT r.id, r.conversation_id, r.status, r.model_revision, r.finish_reason,
                   r.input_tokens, r.output_tokens, r.error_code, r.created_at, r.completed_at,
                   COALESCE(ur.reserved_tokens, 0) AS reserved_tokens,
                   COALESCE(m.content, '') AS output_text
            FROM generation_runs r
            LEFT JOIN usage_reservations ur ON ur.run_id = r.id
            LEFT JOIN messages m ON m.run_id = r.id AND m.role = 'ASSISTANT'
            WHERE r.id = ? AND r.user_id = ?
            """,
            (rs, rowNum) -> new RunStatusResponse(
                UUID.fromString(rs.getString("id")),
                UUID.fromString(rs.getString("conversation_id")),
                rs.getString("status"),
                rs.getString("model_revision"),
                rs.getString("finish_reason"),
                rs.getString("output_text"),
                rs.getInt("reserved_tokens"),
                rs.getInt("input_tokens"),
                rs.getInt("output_tokens"),
                rs.getString("error_code"),
                rs.getObject("created_at", OffsetDateTime.class),
                rs.getObject("completed_at", OffsetDateTime.class)
            ),
            runId, userId
        );
        if (rows.isEmpty()) {
            throw new ApiException(HttpStatus.NOT_FOUND, "RUN_NOT_FOUND", "Generation run not found.");
        }
        return rows.getFirst();
    }

    @Transactional
    public RunStatusResponse cancelRun(UUID userId, UUID runId) {
        List<RunCancelRow> rows = jdbcTemplate.query(
            """
            SELECT id, conversation_id, status
            FROM generation_runs
            WHERE id = ? AND user_id = ?
            FOR UPDATE
            """,
            (rs, rowNum) -> new RunCancelRow(
                UUID.fromString(rs.getString("id")),
                UUID.fromString(rs.getString("conversation_id")),
                rs.getString("status")
            ),
            runId, userId
        );
        if (rows.isEmpty()) {
            throw new ApiException(HttpStatus.NOT_FOUND, "RUN_NOT_FOUND", "Generation run not found.");
        }
        RunCancelRow run = rows.getFirst();
        if ("QUEUED".equals(run.status()) || "RUNNING".equals(run.status())) {
            transitionToCancelled(runId);
        }
        return getRunStatus(userId, runId);
    }

    @Transactional(readOnly = true)
    public List<RunEventRow> listRunEventsAfter(UUID userId, UUID runId, long afterSequence) {
        Integer owned = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM generation_runs WHERE id = ? AND user_id = ?",
            Integer.class,
            runId, userId
        );
        if (owned == null || owned == 0) {
            throw new ApiException(HttpStatus.NOT_FOUND, "RUN_NOT_FOUND", "Generation run not found.");
        }

        return jdbcTemplate.query(
            """
            SELECT sequence, event_type, payload_json::text AS payload_json
            FROM run_events
            WHERE run_id = ? AND sequence > ?
            ORDER BY sequence ASC
            """,
            (rs, rowNum) -> new RunEventRow(
                rs.getLong("sequence"),
                rs.getString("event_type"),
                rs.getString("payload_json")
            ),
            runId, afterSequence
        );
    }

    /**
     * Executes the deterministic fake provider for a QUEUED run and persists incremental SSE events
     * and terminal state transitions.
     */
    public void executeFakeRun(UUID runId) {
        RunExecutionContext ctx = transactionTemplate.execute(status -> {
            List<RunExecutionContext> rows = jdbcTemplate.query(
                """
                SELECT id, conversation_id, status, model_revision, prompt_snapshot, sampling_json::text AS sampling_json, cancel_requested
                FROM generation_runs
                WHERE id = ?
                FOR UPDATE
                """,
                (rs, rowNum) -> new RunExecutionContext(
                    UUID.fromString(rs.getString("id")),
                    UUID.fromString(rs.getString("conversation_id")),
                    rs.getString("status"),
                    rs.getString("model_revision"),
                    rs.getString("prompt_snapshot"),
                    rs.getString("sampling_json"),
                    rs.getBoolean("cancel_requested")
                ),
                runId
            );
            if (rows.isEmpty()) {
                return null;
            }
            RunExecutionContext r = rows.getFirst();
            if (!"QUEUED".equals(r.status()) || r.cancelRequested()) {
                return null;
            }

            jdbcTemplate.update(
                """
                UPDATE generation_runs
                SET status = 'RUNNING', attempt_token = attempt_token + 1, started_at = clock_timestamp(), updated_at = clock_timestamp()
                WHERE id = ?
                """,
                runId
            );
            appendRunEvent(runId, "run.started", Map.of(
                "run_id", runId.toString(),
                "schema_version", 1,
                "model_revision", r.modelRevision()
            ));
            return r;
        });

        if (ctx == null) {
            return;
        }

        SamplingOptionsDto sampling = parseSampling(ctx.samplingJson());
        long delayMs = ctx.promptSnapshot().contains("[SLOW_FAKE]") ? 80L : 5L;
        FaultInjectionSpec faultSpec = resolveFaultSpec(ctx.promptSnapshot(), delayMs);

        AtomicBoolean cancelled = new AtomicBoolean(false);
        StringBuilder fullOutput = new StringBuilder();

        GenerationCommand command = new GenerationCommand(
            runId,
            1L,
            ctx.modelRevision(),
            ctx.promptSnapshot(),
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
                Boolean isCancelReq = jdbcTemplate.queryForObject(
                    "SELECT cancel_requested OR status = 'CANCELLED' FROM generation_runs WHERE id = ?",
                    Boolean.class,
                    runId
                );
                if (Boolean.TRUE.equals(isCancelReq)) {
                    cancelled.set(true);
                    return;
                }
                fullOutput.append(chunk.textDelta());
                transactionTemplate.executeWithoutResult(status ->
                    appendRunEvent(runId, "text.delta", Map.of(
                        "run_id", runId.toString(),
                        "schema_version", 1,
                        "text", chunk.textDelta()
                    ))
                );
            },
            () -> {
                Boolean isCancelReq = jdbcTemplate.queryForObject(
                    "SELECT cancel_requested OR status = 'CANCELLED' FROM generation_runs WHERE id = ?",
                    Boolean.class,
                    runId
                );
                return Boolean.TRUE.equals(isCancelReq) || cancelled.get();
            }
        );

        transactionTemplate.executeWithoutResult(status -> {
            String currentStatus = jdbcTemplate.queryForObject(
                "SELECT status FROM generation_runs WHERE id = ? FOR UPDATE",
                String.class,
                runId
            );
            if (!"RUNNING".equals(currentStatus)) {
                return;
            }
            if (outcome.finishReason() == FinishReason.CANCELLED || cancelled.get()) {
                transitionToCancelled(runId);
                return;
            }
            if (!outcome.isSuccess()) {
                String errorCode = outcome.errorCode() != null ? outcome.errorCode() : "PROVIDER_EXECUTION_ERROR";
                jdbcTemplate.update(
                    """
                    UPDATE generation_runs
                    SET status = 'FAILED',
                        finish_reason = 'error',
                        error_code = ?,
                        completed_at = clock_timestamp(),
                        updated_at = clock_timestamp()
                    WHERE id = ?
                    """,
                    errorCode, runId
                );
                refundReservation(runId);
                jdbcTemplate.update(
                    "UPDATE system_capacity SET active_or_waiting_runs = GREATEST(0, active_or_waiting_runs - 1), updated_at = clock_timestamp() WHERE id = 1"
                );
                appendRunEvent(runId, "run.failed", Map.of(
                    "run_id", runId.toString(),
                    "schema_version", 1,
                    "error_code", errorCode,
                    "message", outcome.errorMessage() != null ? outcome.errorMessage() : "Generation failed inside provider."
                ));
                return;
            }

            Long maxOrder = jdbcTemplate.queryForObject(
                "SELECT COALESCE(MAX(message_order), 0) FROM messages WHERE conversation_id = ?",
                Long.class,
                ctx.conversationId()
            );
            long assistantOrder = (maxOrder != null ? maxOrder : 0L) + 1L;

            jdbcTemplate.update(
                """
                INSERT INTO messages (id, conversation_id, run_id, role, content, status, message_order)
                VALUES (?, ?, ?, 'ASSISTANT', ?, 'COMMITTED', ?)
                """,
                UUID.randomUUID(), ctx.conversationId(), runId, fullOutput.toString(), assistantOrder
            );

            jdbcTemplate.update(
                """
                UPDATE generation_runs
                SET status = 'COMPLETED',
                    finish_reason = ?,
                    input_tokens = ?,
                    output_tokens = ?,
                    completed_at = clock_timestamp(),
                    updated_at = clock_timestamp()
                WHERE id = ?
                """,
                outcome.finishReason().wireValue(),
                outcome.inputTokens(),
                outcome.outputTokens(),
                runId
            );

            settleReservation(runId, outcome.inputTokens() + outcome.outputTokens());
            jdbcTemplate.update(
                "UPDATE system_capacity SET active_or_waiting_runs = GREATEST(0, active_or_waiting_runs - 1), updated_at = clock_timestamp() WHERE id = 1"
            );

            appendRunEvent(runId, "run.completed", Map.of(
                "run_id", runId.toString(),
                "schema_version", 1,
                "finish_reason", outcome.finishReason().wireValue(),
                "input_tokens", outcome.inputTokens(),
                "output_tokens", outcome.outputTokens()
            ));
        });
    }

    private void transitionToCancelled(UUID runId) {
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
            refundReservation(runId);
            jdbcTemplate.update(
                "UPDATE system_capacity SET active_or_waiting_runs = GREATEST(0, active_or_waiting_runs - 1), updated_at = clock_timestamp() WHERE id = 1"
            );
            appendRunEvent(runId, "run.cancelled", Map.of(
                "run_id", runId.toString(),
                "schema_version", 1,
                "finish_reason", "cancelled"
            ));
        }
    }

    private void settleReservation(UUID runId, int actualTotalTokens) {
        jdbcTemplate.query(
            """
            UPDATE usage_reservations
            SET status = 'SETTLED', settled_tokens = ?, settled_at = clock_timestamp()
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
                        settled_tokens = settled_tokens + ?,
                        updated_at = clock_timestamp()
                    WHERE id = ?
                    """,
                    reserved, actualTotalTokens, bucketId
                );
                return bucketId;
            },
            actualTotalTokens, runId
        );
    }

    private void refundReservation(UUID runId) {
        jdbcTemplate.query(
            """
            UPDATE usage_reservations
            SET status = 'REFUNDED', settled_tokens = 0, settled_at = clock_timestamp()
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

    private void appendRunEvent(UUID runId, String eventType, Map<String, Object> payload) {
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

    private OptionalReplay checkExistingIdempotencyRecord(
        UUID workspaceId,
        UUID userId,
        String idempotencyKey,
        String canonicalRequestHash
    ) {
        List<IdempotencyRow> rows = jdbcTemplate.query(
            """
            SELECT ir.request_hash, ir.run_id, ir.expires_at,
                   gr.conversation_id, gr.status, gr.model_revision
            FROM idempotency_records ir
            JOIN generation_runs gr ON gr.id = ir.run_id
            WHERE ir.workspace_id = ?
              AND ir.user_id = ?
              AND ir.operation = 'SUBMIT_RUN'
              AND ir.idempotency_key = ?
            """,
            (rs, rowNum) -> new IdempotencyRow(
                rs.getString("request_hash"),
                UUID.fromString(rs.getString("run_id")),
                rs.getObject("expires_at", OffsetDateTime.class),
                UUID.fromString(rs.getString("conversation_id")),
                rs.getString("status"),
                rs.getString("model_revision")
            ),
            workspaceId, userId, idempotencyKey
        );

        if (rows.isEmpty()) {
            return new OptionalReplay(false, null);
        }

        IdempotencyRow existing = rows.getFirst();
        if (!existing.requestHash().equals(canonicalRequestHash)) {
            throw new ApiException(
                HttpStatus.CONFLICT,
                "IDEMPOTENCY_KEY_CONFLICT",
                "Idempotency-Key was already used with a different request payload."
            );
        }

        return new OptionalReplay(
            true,
            buildAcceptedResponse(
                existing.runId(),
                existing.conversationId(),
                existing.status(),
                existing.modelRevision()
            )
        );
    }

    /**
     * Hashes the canonical client request independently of model_alias -> revision resolution.
     */
    public static String computeCanonicalRequestHash(UUID conversationId, SubmitRunRequest request) {
        List<String> sortedDocs = new ArrayList<>();
        if (request.document_ids() != null) {
            for (UUID docId : request.document_ids()) {
                if (docId != null) {
                    sortedDocs.add(docId.toString());
                }
            }
            Collections.sort(sortedDocs);
        }
        SamplingOptionsDto s = request.sampling();
        String canonical = String.format(
            Locale.ROOT,
            "conv=%s|alias=%s|prompt=%s|max_new=%d|temp=%.4f|top_k=%d|top_p=%.4f|seed=%s|docs=%s",
            conversationId,
            request.model_alias().trim(),
            request.prompt(),
            s.max_new_tokens(),
            s.temperature(),
            s.top_k(),
            s.top_p(),
            s.seed() != null ? s.seed().toString() : "null",
            String.join(",", sortedDocs)
        );
        return sha256Hex(canonical);
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

    private static String sha256Hex(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(input.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException(ex);
        }
    }

    private static RunAcceptedResponse buildAcceptedResponse(
        UUID runId,
        UUID conversationId,
        String status,
        String modelRevision
    ) {
        return new RunAcceptedResponse(
            runId,
            conversationId,
            status,
            modelRevision,
            "/api/v1/runs/" + runId + "/events",
            "/api/v1/runs/" + runId
        );
    }

    private record ConversationMeta(UUID id, UUID workspaceId, UUID ownerUserId) {}
    private record ModelRevisionMeta(String revisionTag, int contextLimit, int maxOutputTokens) {}
    private record CapacityRow(int maxWaitingRuns, int activeOrWaitingRuns) {}
    private record UsageBucketRow(UUID id, int dailyAllowanceTokens, int reservedTokens, int settledTokens) {}
    private record IdempotencyRow(
        String requestHash,
        UUID runId,
        OffsetDateTime expiresAt,
        UUID conversationId,
        String status,
        String modelRevision
    ) {}
    private record OptionalReplay(boolean found, RunAcceptedResponse response) {}
    private record RunCancelRow(UUID id, UUID conversationId, String status) {}
    private record RunExecutionContext(
        UUID id,
        UUID conversationId,
        String status,
        String modelRevision,
        String promptSnapshot,
        String samplingJson,
        boolean cancelRequested
    ) {}
}
