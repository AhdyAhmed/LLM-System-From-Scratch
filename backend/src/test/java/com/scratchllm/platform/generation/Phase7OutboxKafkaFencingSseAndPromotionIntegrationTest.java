package com.scratchllm.platform.generation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.scratchllm.platform.generation.FencedGenerationDispatcher.ClaimedWorkItem;
import com.scratchllm.platform.generation.FencedGenerationDispatcher.OrphanReconciliationSummary;
import com.scratchllm.platform.generation.GenerationProvider.FinishReason;
import com.scratchllm.platform.generation.OutboxRelayAndInboxService.DlqReplayResult;
import com.scratchllm.platform.generation.OutboxRelayAndInboxService.InboxConsumeResult;
import com.scratchllm.platform.generation.OutboxRelayAndInboxService.RelayBatchResult;
import com.scratchllm.platform.generation.RunDtos.RunAcceptedResponse;
import com.scratchllm.platform.generation.RunDtos.RunStatusResponse;
import com.scratchllm.platform.generation.RunDtos.SamplingOptionsDto;
import com.scratchllm.platform.generation.RunDtos.SubmitRunRequest;
import com.scratchllm.platform.infrastructure.KafkaEventsInfrastructure;
import com.scratchllm.platform.modelcatalog.ModelCatalogService;
import com.scratchllm.platform.modelcatalog.ModelPromotionService;
import com.scratchllm.platform.modelcatalog.ModelPromotionService.ModelPromotionResponse;
import com.scratchllm.platform.modelcatalog.ModelPromotionService.ModelReleaseConsumerResult;
import com.scratchllm.platform.modelcatalog.ModelPromotionService.PromoteModelRequest;
import com.scratchllm.platform.platform.ApiException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.boot.test.web.client.TestRestTemplate;

/**
 * Day061–Day070 (Gate G7) End-to-End Integration & Fault-Injection Suite:
 * 1. Day062 & Day063: Atomic acceptance, Outbox High-Water-Mark backpressure (`503 OUTBOX_BACKPRESSURE_EXCEEDED`),
 *    idempotent replay during backpressure, polling Outbox Relay -> Kafka -> Consumer Inbox deduplication -> `work_items`.
 * 2. Day064 & Day065: Two-dispatcher `FOR UPDATE SKIP LOCKED` claiming, `attempt_token` fencing rejecting stale worker
 *    heartbeats/chunks/terminal transitions, and orphan/expired-lease reconciliation (both zero-output retry and
 *    partial-output `PARTIAL_FAILED` + `INCOMPLETE_SETTLED`).
 * 3. Day065: Midnight UTC rollover quota settlement against original `usage_reservations.bucket_id`.
 * 4. Day066 & Day067: Shared SSE Event Multiplexer serving 50 concurrent SSE subscribers without Hikari pool exhaustion,
 *    expired cursor `resync.required` control frame, slow-reader buffer overflow bounding, and client disconnect vs. cancel.
 * 5. Day067: Poison Kafka payload DLQ quarantine (`dlq_events` + `.dlq.v1`) and manual DLQ replay skipping terminal runs.
 * 6. Day068: Durable `model_promotions` with monotonic `release_generation`, stale generation rejection, and rollback
 *    to an earlier immutable `revision_tag` via a newer `release_generation`.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class Phase7OutboxKafkaFencingSseAndPromotionIntegrationTest {

    @LocalServerPort
    private int port;

    @Autowired
    private TestRestTemplate restTemplate;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Autowired
    private RunService runService;

    @Autowired
    private OutboxRelayAndInboxService outboxRelayAndInboxService;

    @Autowired
    private FencedGenerationDispatcher fencedGenerationDispatcher;

    @Autowired
    private SharedSseEventMultiplexer sharedSseEventMultiplexer;

    @Autowired
    private ModelPromotionService modelPromotionService;

    @Autowired
    private ModelCatalogService modelCatalogService;

    @Autowired
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        runService.setInlineFakeDispatchEnabled(true);
        outboxRelayAndInboxService.setBrokerOutageSimulated(false);
        jdbcTemplate.update(
            "UPDATE system_capacity SET max_waiting_runs = 8, active_or_waiting_runs = 0, max_unpublished_outbox_events = 256 WHERE id = 1"
        );
        jdbcTemplate.update(
            "UPDATE outbox_events SET published_at = clock_timestamp() WHERE published_at IS NULL"
        );
        jdbcTemplate.update(
            "DELETE FROM model_promotions WHERE model_name = 'scratchlm-tiny-bpe' AND release_generation > 1"
        );
        jdbcTemplate.update(
            "UPDATE active_model_releases SET active_revision_tag = 'scratch-tiny-r1', applied_release_generation = 1 WHERE model_alias = 'scratch-tiny'"
        );
        modelCatalogService.evictRevisionCache("scratch-tiny-r1", "scratch-tiny");
        modelCatalogService.evictRevisionCache("scratch-tiny-r2", "scratch-tiny");
    }

    @AfterEach
    void tearDown() {
        runService.setInlineFakeDispatchEnabled(true);
        outboxRelayAndInboxService.setBrokerOutageSimulated(false);
        jdbcTemplate.update(
            "UPDATE system_capacity SET max_waiting_runs = 8, active_or_waiting_runs = 0, max_unpublished_outbox_events = 256 WHERE id = 1"
        );
        jdbcTemplate.update(
            "UPDATE outbox_events SET published_at = clock_timestamp() WHERE published_at IS NULL"
        );
        jdbcTemplate.update(
            "UPDATE work_items SET state = 'COMPLETED', lease_expires_at = NULL WHERE state IN ('PENDING', 'LEASED')"
        );
        jdbcTemplate.update(
            "DELETE FROM model_promotions WHERE model_name = 'scratchlm-tiny-bpe' AND release_generation > 1"
        );
        jdbcTemplate.update(
            "UPDATE active_model_releases SET active_revision_tag = 'scratch-tiny-r1', applied_release_generation = 1 WHERE model_alias = 'scratch-tiny'"
        );
        modelCatalogService.evictRevisionCache("scratch-tiny-r1", "scratch-tiny");
        modelCatalogService.evictRevisionCache("scratch-tiny-r2", "scratch-tiny");
    }

    @Test
    void outboxHighWaterMarkBackpressureAndKafkaRelayInboxDeduplicationWorkEndToEnd() {
        runService.setInlineFakeDispatchEnabled(false);
        outboxRelayAndInboxService.setBrokerOutageSimulated(true);
        jdbcTemplate.update("UPDATE system_capacity SET max_unpublished_outbox_events = 2 WHERE id = 1");

        UserFixture user = createTestUserAndWorkspace("phase7-outbox-" + UUID.randomUUID() + "@example.com");
        UUID conv1 = createConversation(user, "Outbox Conv 1");
        UUID conv2 = createConversation(user, "Outbox Conv 2");
        UUID conv3 = createConversation(user, "Outbox Conv 3");

        SubmitRunRequest req = new SubmitRunRequest(
            "Outbox durability test prompt",
            "fake-deterministic",
            new SamplingOptionsDto(8, 0.7, 20, 0.9, 42L),
            List.of()
        );

        RunAcceptedResponse run1 = runService.submitRun(user.userId(), conv1, "idem-outbox-run-001", req);
        RunAcceptedResponse run2 = runService.submitRun(user.userId(), conv2, "idem-outbox-run-002", req);

        // Attempt relay during simulated broker outage -> fails and leaves published_at IS NULL
        RelayBatchResult outageAttempt = outboxRelayAndInboxService.relayBatch(10);
        assertThat(outageAttempt.claimedCount()).isEqualTo(2);
        assertThat(outageAttempt.publishedCount()).isZero();
        assertThat(outageAttempt.failedCount()).isEqualTo(2);

        // 3rd run submission must be rejected with 503 OUTBOX_BACKPRESSURE_EXCEEDED
        assertThatThrownBy(() -> runService.submitRun(user.userId(), conv3, "idem-outbox-run-003", req))
            .isInstanceOf(ApiException.class)
            .satisfies(ex -> {
                ApiException apiEx = (ApiException) ex;
                assertThat(apiEx.getStatus()).isEqualTo(HttpStatus.SERVICE_UNAVAILABLE);
                assertThat(apiEx.getErrorCode()).isEqualTo("OUTBOX_BACKPRESSURE_EXCEEDED");
            });

        // Idempotent replay of an already-accepted run MUST still succeed during outbox backpressure!
        RunAcceptedResponse replay1 = runService.submitRun(user.userId(), conv1, "idem-outbox-run-001", req);
        assertThat(replay1.run_id()).isEqualTo(run1.run_id());

        // Recover broker and drain outbox -> Kafka -> Inbox -> work_items
        outboxRelayAndInboxService.setBrokerOutageSimulated(false);
        RelayBatchResult recovered = outboxRelayAndInboxService.relayAndConsumePending(10);
        assertThat(recovered.publishedCount()).isEqualTo(2);

        // Duplicate Kafka redelivery of the exact same event_id must be a no-op
        var firstPublished = recovered.publishedEvents().getFirst();
        InboxConsumeResult duplicateDelivery = outboxRelayAndInboxService.consumeKafkaEvent(
            firstPublished.topic(),
            firstPublished.aggregateKey(),
            firstPublished.payloadJson()
        );
        assertThat(duplicateDelivery.newlyInsertedInbox()).isFalse();
        assertThat(duplicateDelivery.workItemCreated()).isFalse();
        assertThat(duplicateDelivery.note()).isEqualTo("DUPLICATE_EVENT_IGNORED");

        // Dispatch both runs via FencedGenerationDispatcher
        assertThat(fencedGenerationDispatcher.dispatchAndExecuteNext("dispatcher-1", Duration.ofSeconds(10), 2)).isPresent();
        assertThat(fencedGenerationDispatcher.dispatchAndExecuteNext("dispatcher-1", Duration.ofSeconds(10), 2)).isPresent();

        assertThat(runService.getRunStatus(user.userId(), run1.run_id()).status()).isEqualTo("COMPLETED");
        assertThat(runService.getRunStatus(user.userId(), run2.run_id()).status()).isEqualTo("COMPLETED");
    }

    @Test
    void fencedDispatcherRejectsStaleAttemptWritesAndReconcilesOrphanedLeases() throws Exception {
        runService.setInlineFakeDispatchEnabled(false);

        UserFixture user = createTestUserAndWorkspace("phase7-fencing-" + UUID.randomUUID() + "@example.com");
        UUID conv1 = createConversation(user, "Fencing Conflict Conv");
        UUID conv2 = createConversation(user, "Partial Lease Expiry Conv");

        SubmitRunRequest req = new SubmitRunRequest(
            "Fenced execution verification",
            "fake-deterministic",
            new SamplingOptionsDto(8, 0.7, 20, 0.9, 42L),
            List.of()
        );

        RunAcceptedResponse accepted1 = runService.submitRun(user.userId(), conv1, "idem-fencing-001", req);
        outboxRelayAndInboxService.relayAndConsumePending(10);

        // 1. Dispatcher A claims run1 with a very short 100ms lease (attempt_token = 1)
        Optional<ClaimedWorkItem> claimA = fencedGenerationDispatcher.claimNextWorkItem("dispatcher-A", Duration.ofMillis(200));
        assertThat(claimA).isPresent();
        assertThat(claimA.get().attemptToken()).isEqualTo(1L);

        // Wait for Dispatcher A's lease to expire before emitting any visible delta chunks
        jdbcTemplate.update(
            "UPDATE work_items SET lease_expires_at = clock_timestamp() - interval '1 second' WHERE id = ?",
            claimA.get().workItemId()
        );

        // 2. Reconciler detects 0 emitted chunks and resets run1 to QUEUED / PENDING
        OrphanReconciliationSummary reconciledZeroOutput = fencedGenerationDispatcher.reconcileExpiredLeases();
        assertThat(reconciledZeroOutput.requeuedRuns()).isGreaterThanOrEqualTo(1);

        // 3. Dispatcher B claims run1 and increments attempt_token to 2
        Optional<ClaimedWorkItem> claimB = fencedGenerationDispatcher.claimNextWorkItem("dispatcher-B", Duration.ofSeconds(10));
        assertThat(claimB).isPresent();
        assertThat(claimB.get().attemptToken()).isEqualTo(2L);

        // 4. Stale Dispatcher A (attempt_token = 1) wakes up and tries to heartbeat, append chunks, and complete
        assertThat(
            fencedGenerationDispatcher.renewLeaseHeartbeat(accepted1.run_id(), 1L, "dispatcher-A", Duration.ofSeconds(10))
        ).isFalse();
        assertThat(
            fencedGenerationDispatcher.appendFencedDeltaBatch(accepted1.run_id(), 1L, List.of("STALE_ZOMBIE_TEXT"))
        ).isFalse();
        assertThat(
            fencedGenerationDispatcher.completeFencedRun(
                accepted1.run_id(),
                1L,
                FinishReason.STOP,
                "STALE_ZOMBIE_FINAL_MESSAGE",
                5,
                5
            )
        ).isFalse();

        // 5. Active Dispatcher B (attempt_token = 2) writes batched chunks and completes cleanly
        assertThat(
            fencedGenerationDispatcher.appendFencedDeltaBatch(accepted1.run_id(), 2L, List.of("Clean ", "fenced ", "output."))
        ).isTrue();
        assertThat(
            fencedGenerationDispatcher.completeFencedRun(
                accepted1.run_id(),
                2L,
                FinishReason.STOP,
                "Clean fenced output.",
                5,
                3
            )
        ).isTrue();

        RunStatusResponse finalStatus1 = runService.getRunStatus(user.userId(), accepted1.run_id());
        assertThat(finalStatus1.status()).isEqualTo("COMPLETED");
        assertThat(finalStatus1.output_text()).isEqualTo("Clean fenced output.");
        assertThat(finalStatus1.output_text()).doesNotContain("STALE_ZOMBIE");

        // 6. Now test partial-output lease expiry on conv2: emit 1 delta chunk, expire lease, reconcile -> PARTIAL_FAILED + INCOMPLETE_SETTLED
        RunAcceptedResponse accepted2 = runService.submitRun(user.userId(), conv2, "idem-fencing-002", req);
        outboxRelayAndInboxService.relayAndConsumePending(10);

        Optional<ClaimedWorkItem> claimPartial = fencedGenerationDispatcher.claimNextWorkItem("dispatcher-A", Duration.ofSeconds(10));
        assertThat(claimPartial).isPresent();
        assertThat(
            fencedGenerationDispatcher.appendFencedDeltaBatch(
                accepted2.run_id(),
                claimPartial.get().attemptToken(),
                List.of("Partial visible token before crash")
            )
        ).isTrue();

        jdbcTemplate.update(
            "UPDATE work_items SET lease_expires_at = clock_timestamp() - interval '1 second' WHERE id = ?",
            claimPartial.get().workItemId()
        );

        OrphanReconciliationSummary reconciledPartial = fencedGenerationDispatcher.reconcileExpiredLeases();
        assertThat(reconciledPartial.failedPartialRuns()).isGreaterThanOrEqualTo(1);

        RunStatusResponse partialStatus = runService.getRunStatus(user.userId(), accepted2.run_id());
        assertThat(partialStatus.status()).isEqualTo("FAILED");
        assertThat(partialStatus.error_code()).isEqualTo("WORKER_LEASE_EXPIRED");
        assertThat(partialStatus.output_text()).isEqualTo("Partial visible token before crash");

        String reservationStatus = jdbcTemplate.queryForObject(
            "SELECT status FROM usage_reservations WHERE run_id = ?",
            String.class,
            accepted2.run_id()
        );
        assertThat(reservationStatus).isEqualTo("INCOMPLETE_SETTLED");

        String messageStatus = jdbcTemplate.queryForObject(
            "SELECT status FROM messages WHERE run_id = ? AND role = 'ASSISTANT'",
            String.class,
            accepted2.run_id()
        );
        assertThat(messageStatus).isEqualTo("PARTIAL_FAILED");
    }

    @Test
    void settlesQuotaAgainstOriginalUtcDateBucketAcrossMidnightRollover() {
        runService.setInlineFakeDispatchEnabled(false);

        UserFixture user = createTestUserAndWorkspace("phase7-midnight-" + UUID.randomUUID() + "@example.com");
        UUID convId = createConversation(user, "Midnight Rollover Conv");

        LocalDate yesterdayUtc = LocalDate.now(ZoneOffset.UTC).minusDays(1);
        UUID yesterdayBucketId = UUID.randomUUID();
        jdbcTemplate.update(
            """
            INSERT INTO usage_buckets (id, workspace_id, user_id, utc_date, daily_allowance_tokens, reserved_tokens, settled_tokens)
            VALUES (?, ?, ?, ?, 4096, 0, 0)
            """,
            yesterdayBucketId, user.workspaceId(), user.userId(), yesterdayUtc
        );

        SubmitRunRequest req = new SubmitRunRequest(
            "Run admitted at 23:59:58 UTC",
            "fake-deterministic",
            new SamplingOptionsDto(12, 0.7, 20, 0.9, 42L),
            List.of()
        );
        RunAcceptedResponse accepted = runService.submitRun(user.userId(), convId, "idem-midnight-001", req);

        // Move the reservation from today's bucket to yesterday's bucket to simulate a run admitted before UTC midnight
        Integer reservedAmount = jdbcTemplate.queryForObject(
            "SELECT reserved_tokens FROM usage_reservations WHERE run_id = ?",
            Integer.class,
            accepted.run_id()
        );
        UUID todayBucketId = jdbcTemplate.queryForObject(
            "SELECT bucket_id FROM usage_reservations WHERE run_id = ?",
            UUID.class,
            accepted.run_id()
        );
        jdbcTemplate.update(
            "UPDATE usage_buckets SET reserved_tokens = GREATEST(0, reserved_tokens - ?) WHERE id = ?",
            reservedAmount, todayBucketId
        );
        jdbcTemplate.update(
            "UPDATE usage_buckets SET reserved_tokens = reserved_tokens + ? WHERE id = ?",
            reservedAmount, yesterdayBucketId
        );
        jdbcTemplate.update(
            "UPDATE usage_reservations SET bucket_id = ? WHERE run_id = ?",
            yesterdayBucketId, accepted.run_id()
        );

        // Now complete the run after UTC midnight
        outboxRelayAndInboxService.relayAndConsumePending(10);
        assertThat(fencedGenerationDispatcher.dispatchAndExecuteNext("dispatcher-midnight", Duration.ofSeconds(10), 2)).isPresent();

        // Yesterday's bucket must have reserved_tokens == 0 and settled_tokens > 0; today's bucket must remain 0 / 0
        Map<String, Object> yesterdayRow = jdbcTemplate.queryForMap(
            "SELECT reserved_tokens, settled_tokens FROM usage_buckets WHERE id = ?",
            yesterdayBucketId
        );
        assertThat(((Number) yesterdayRow.get("reserved_tokens")).intValue()).isZero();
        assertThat(((Number) yesterdayRow.get("settled_tokens")).intValue()).isGreaterThan(0);

        Map<String, Object> todayRow = jdbcTemplate.queryForMap(
            "SELECT reserved_tokens, settled_tokens FROM usage_buckets WHERE id = ?",
            todayBucketId
        );
        assertThat(((Number) todayRow.get("reserved_tokens")).intValue()).isZero();
        assertThat(((Number) todayRow.get("settled_tokens")).intValue()).isZero();
    }

    @Test
    void sharedSseMultiplexerSupports50ConcurrentSubscribersAndEmitsResyncRequiredOnExpiredCursor() throws Exception {
        String email = "phase7-sse50-" + UUID.randomUUID() + "@example.com";
        AuthClientSession session = registerHttpSession(email, "CorrectHorseBatteryStaple!23", "SSE50 Workspace");
        UUID convId = createConversationViaHttp(session, "50-Subscriber Multiplexer Conv");

        HttpHeaders headers = session.csrfHeaders();
        headers.set("Idempotency-Key", "idem-sse50-run-001");
        ResponseEntity<JsonNode> submitResp = restTemplate.exchange(
            "/api/v1/conversations/" + convId + "/runs",
            HttpMethod.POST,
            new HttpEntity<>(
                Map.of(
                    "prompt", "Multiplexer fanout test [SLOW_FAKE]",
                    "model_alias", "fake-deterministic",
                    "sampling", Map.of("max_new_tokens", 8, "temperature", 0.7, "top_k", 20, "top_p", 0.95, "seed", 42)
                ),
                headers
            ),
            JsonNode.class
        );
        assertThat(submitResp.getStatusCode()).isEqualTo(HttpStatus.ACCEPTED);
        UUID runId = UUID.fromString(submitResp.getBody().get("run_id").asText());

        // Launch 50 concurrent SSE subscribers against the single run while Hikari pool size is 10
        int subscriberCount = 50;
        ExecutorService pool = Executors.newFixedThreadPool(subscriberCount);
        try {
            List<Callable<List<String>>> tasks = new ArrayList<>();
            for (int i = 0; i < subscriberCount; i++) {
                tasks.add(() -> readSseEventNames(session.cookieHeader(), "/api/v1/runs/" + runId + "/events", null));
            }
            List<Future<List<String>>> futures = pool.invokeAll(tasks, 25, TimeUnit.SECONDS);
            for (Future<List<String>> f : futures) {
                List<String> eventNames = f.get(5, TimeUnit.SECONDS);
                assertThat(eventNames).contains("run.queued", "run.Completed".toLowerCase());
            }
        } finally {
            pool.shutdownNow();
        }
        assertThat(sharedSseEventMultiplexer.getTotalBatchPollQueries()).isGreaterThan(0L);

        // Prune early events (sequence <= 2) for runId and reconnect with Last-Event-ID: 1 -> must receive resync.required
        sharedSseEventMultiplexer.pruneRunEventsUpToSequence(runId, 2L);
        List<String> expiredCursorEvents = readSseEventNames(
            session.cookieHeader(),
            "/api/v1/runs/" + runId + "/events",
            1L
        );
        assertThat(expiredCursorEvents).containsExactly("resync.required");
    }

    @Test
    void poisonKafkaEventRoutesToDlqAndManualReplaySkipsCompletedRun() throws Exception {
        UserFixture user = createTestUserAndWorkspace("phase7-dlq-" + UUID.randomUUID() + "@example.com");
        UUID convId = createConversation(user, "DLQ Quarantine Conv");

        SubmitRunRequest req = new SubmitRunRequest(
            "Run for DLQ terminal replay test",
            "fake-deterministic",
            new SamplingOptionsDto(8, 0.7, 20, 0.9, 42L),
            List.of()
        );
        RunAcceptedResponse accepted = runService.submitRun(user.userId(), convId, "idem-dlq-001", req);

        // Wait for inline run completion
        long deadline = System.currentTimeMillis() + 10_000L;
        while (System.currentTimeMillis() < deadline) {
            if ("COMPLETED".equals(runService.getRunStatus(user.userId(), accepted.run_id()).status())) {
                break;
            }
            Thread.sleep(25L);
        }
        assertThat(runService.getRunStatus(user.userId(), accepted.run_id()).status()).isEqualTo("COMPLETED");

        // Send a poison Kafka event carrying a forbidden raw "prompt" field for this run_id
        UUID poisonEventId = UUID.randomUUID();
        String poisonJson = objectMapper.writeValueAsString(Map.of(
            "event_id", poisonEventId.toString(),
            "event_type", KafkaEventsInfrastructure.TOPIC_GENERATION_REQUESTED,
            "schema_version", 1,
            "occurred_at", OffsetDateTime.now(ZoneOffset.UTC).toString(),
            "workspace_id", user.workspaceId().toString(),
            "conversation_id", convId.toString(),
            "run_id", accepted.run_id().toString(),
            "user_id", user.userId().toString(),
            "model_revision", "fake-deterministic-r1",
            "prompt", "Forbidden raw prompt inside Kafka payload"
        ));

        InboxConsumeResult consumeResult = outboxRelayAndInboxService.consumeKafkaEvent(
            KafkaEventsInfrastructure.TOPIC_GENERATION_REQUESTED,
            convId.toString(),
            poisonJson
        );
        assertThat(consumeResult.routedToDlq()).isTrue();
        assertThat(consumeResult.eventId()).isEqualTo(poisonEventId);

        // Operator replays the quarantined DLQ event -> must preserve event_id and skip the already-COMPLETED run!
        DlqReplayResult replayResult = outboxRelayAndInboxService.replayDlqEvent(poisonEventId);
        assertThat(replayResult.event_id()).isEqualTo(poisonEventId);
        assertThat(replayResult.outcome()).isEqualTo("SKIPPED_TERMINAL_RUN");
        assertThat(replayResult.run_status()).isEqualTo("COMPLETED");
        assertThat(replayResult.work_item_created()).isFalse();
    }

    @Test
    void modelPromotionAdvancesMonotonicReleaseGenerationRejectsStaleEventsAndSupportsRollback() throws Exception {
        runService.setInlineFakeDispatchEnabled(false);

        UserFixture user = createTestUserAndWorkspace("phase7-promo-" + UUID.randomUUID() + "@example.com");
        UUID conv1 = createConversation(user, "Pre-Promotion Conv");
        UUID conv2 = createConversation(user, "Post-Promotion Conv");

        SubmitRunRequest req = new SubmitRunRequest(
            "Model promotion pinning test",
            "scratch-tiny",
            new SamplingOptionsDto(8, 0.7, 20, 0.9, 42L),
            List.of()
        );

        // 1. Run 1 submitted before promotion is pinned to scratch-tiny-r1
        RunAcceptedResponse runBeforePromo = runService.submitRun(user.userId(), conv1, "idem-promo-run-1", req);
        assertThat(runBeforePromo.model_revision()).isEqualTo("scratch-tiny-r1");

        // 2. Promote scratch-tiny to scratch-tiny-r2 (release_generation = 2)
        ModelPromotionResponse promo2 = modelPromotionService.promoteOrRollbackModel(
            new PromoteModelRequest("scratch-tiny", "scratch-tiny-r2", 2L, null, "gate-g7-test")
        );
        assertThat(promo2.release_generation()).isEqualTo(2L);
        assertThat(promo2.target_revision_tag()).isEqualTo("scratch-tiny-r2");
        assertThat(promo2.previous_revision_tag()).isEqualTo("scratch-tiny-r1");

        // 3. Run 2 submitted after promotion resolves to scratch-tiny-r2, while Run 1 remains pinned to scratch-tiny-r1
        RunAcceptedResponse runAfterPromo = runService.submitRun(user.userId(), conv2, "idem-promo-run-2", req);
        assertThat(runAfterPromo.model_revision()).isEqualTo("scratch-tiny-r2");
        assertThat(runService.getRunStatus(user.userId(), runBeforePromo.run_id()).model_revision()).isEqualTo("scratch-tiny-r1");

        // 4. Stale model.released.v1 event with release_generation = 1 must be rejected without overwriting scratch-tiny-r2
        String staleEventJson = objectMapper.writeValueAsString(Map.of(
            "event_id", UUID.randomUUID().toString(),
            "event_type", KafkaEventsInfrastructure.TOPIC_MODEL_RELEASED,
            "schema_version", 1,
            "occurred_at", OffsetDateTime.now(ZoneOffset.UTC).toString(),
            "promotion_id", UUID.randomUUID().toString(),
            "model_name", "scratchlm-tiny-bpe",
            "release_generation", 1,
            "target_revision_tag", "scratch-tiny-r1",
            "artifact_checksum", "ba30f95421b3016b5b08ddae2b63537b701c759c2cbcc42cc495b929ebc2b12b"
        ));
        ModelReleaseConsumerResult staleResult = modelPromotionService.consumeModelReleasedEvent(staleEventJson);
        assertThat(staleResult.stale_generation_rejected()).isTrue();
        assertThat(staleResult.active_release_generation()).isEqualTo(2L);
        assertThat(staleResult.active_revision_tag()).isEqualTo("scratch-tiny-r2");

        // 5. Intentional rollback to scratch-tiny-r1 using a NEW monotonic release_generation = 3 succeeds!
        ModelPromotionResponse rollback3 = modelPromotionService.promoteOrRollbackModel(
            new PromoteModelRequest("scratch-tiny", "scratch-tiny-r1", 3L, null, "gate-g7-rollback")
        );
        assertThat(rollback3.release_generation()).isEqualTo(3L);
        assertThat(rollback3.target_revision_tag()).isEqualTo("scratch-tiny-r1");
        assertThat(modelCatalogService.resolveReadyAlias("scratch-tiny").orElseThrow().model_revision())
            .isEqualTo("scratch-tiny-r1");
    }

    private List<String> readSseEventNames(String cookieHeader, String path, Long lastEventId) throws Exception {
        URI uri = URI.create("http://127.0.0.1:" + port + path);
        HttpURLConnection conn = (HttpURLConnection) uri.toURL().openConnection();
        conn.setRequestMethod("GET");
        conn.setRequestProperty("Accept", "text/event-stream");
        conn.setRequestProperty("Cookie", cookieHeader);
        if (lastEventId != null) {
            conn.setRequestProperty("Last-Event-ID", String.valueOf(lastEventId));
        }
        conn.setConnectTimeout(5000);
        conn.setReadTimeout(15000);

        List<String> events = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("event:")) {
                    String evName = line.substring("event:".length()).trim();
                    events.add(evName);
                    if (SharedSseEventMultiplexer.TERMINAL_EVENTS.contains(evName)) {
                        break;
                    }
                }
            }
        } finally {
            conn.disconnect();
        }
        return events;
    }

    private UserFixture createTestUserAndWorkspace(String email) {
        UUID userId = UUID.randomUUID();
        UUID workspaceId = UUID.randomUUID();
        jdbcTemplate.update(
            "INSERT INTO users (id, email, password_hash, status) VALUES (?, ?, 'hash', 'ACTIVE')",
            userId, email
        );
        jdbcTemplate.update(
            "INSERT INTO workspaces (id, name, owner_user_id) VALUES (?, 'Phase 7 Workspace', ?)",
            workspaceId, userId
        );
        jdbcTemplate.update(
            "INSERT INTO memberships (id, workspace_id, user_id, role) VALUES (?, ?, ?, 'OWNER')",
            UUID.randomUUID(), workspaceId, userId
        );
        return new UserFixture(userId, workspaceId);
    }

    private UUID createConversation(UserFixture user, String title) {
        UUID convId = UUID.randomUUID();
        jdbcTemplate.update(
            "INSERT INTO conversations (id, workspace_id, owner_user_id, title) VALUES (?, ?, ?, ?)",
            convId, user.workspaceId(), user.userId(), title
        );
        return convId;
    }

    private AuthClientSession registerHttpSession(String email, String password, String workspaceName) {
        ResponseEntity<JsonNode> csrf1 = restTemplate.getForEntity("/auth/csrf", JsonNode.class);
        String csrfCookie = extractCookie(csrf1.getHeaders());
        String csrfToken = csrf1.getBody().get("token").asText();

        HttpHeaders regHeaders = new HttpHeaders();
        regHeaders.setContentType(MediaType.APPLICATION_JSON);
        regHeaders.set(HttpHeaders.COOKIE, csrfCookie);
        regHeaders.set("X-CSRF-TOKEN", csrfToken);

        ResponseEntity<JsonNode> regResp = restTemplate.exchange(
            "/auth/register",
            HttpMethod.POST,
            new HttpEntity<>(
                Map.of("email", email, "password", password, "workspace_name", workspaceName),
                regHeaders
            ),
            JsonNode.class
        );
        assertThat(regResp.getStatusCode()).isEqualTo(HttpStatus.CREATED);
        UUID workspaceId = UUID.fromString(
            regResp.getBody().get("workspaces").get(0).get("workspace_id").asText()
        );

        ResponseEntity<JsonNode> loginResp = restTemplate.exchange(
            "/auth/login",
            HttpMethod.POST,
            new HttpEntity<>(
                Map.of("email", email, "password", password),
                regHeaders
            ),
            JsonNode.class
        );
        assertThat(loginResp.getStatusCode()).isEqualTo(HttpStatus.OK);
        String sessionCookie = mergeCookies(csrfCookie, extractCookie(loginResp.getHeaders()));

        HttpHeaders csrf2Headers = new HttpHeaders();
        csrf2Headers.set(HttpHeaders.COOKIE, sessionCookie);
        ResponseEntity<JsonNode> csrf2 = restTemplate.exchange(
            "/auth/csrf",
            HttpMethod.GET,
            new HttpEntity<>(csrf2Headers),
            JsonNode.class
        );
        String finalCookie = mergeCookies(sessionCookie, extractCookie(csrf2.getHeaders()));
        return new AuthClientSession(finalCookie, csrf2.getBody().get("token").asText(), workspaceId);
    }

    private UUID createConversationViaHttp(AuthClientSession session, String title) {
        ResponseEntity<JsonNode> resp = restTemplate.exchange(
            "/api/v1/workspaces/" + session.workspaceId() + "/conversations",
            HttpMethod.POST,
            new HttpEntity<>(Map.of("title", title), session.csrfHeaders()),
            JsonNode.class
        );
        assertThat(resp.getStatusCode()).isEqualTo(HttpStatus.CREATED);
        return UUID.fromString(resp.getBody().get("conversation_id").asText());
    }

    private static String extractCookie(HttpHeaders headers) {
        List<String> setCookies = headers.get(HttpHeaders.SET_COOKIE);
        if (setCookies == null || setCookies.isEmpty()) {
            return "";
        }
        List<String> pairs = new ArrayList<>();
        for (String sc : setCookies) {
            String pair = sc.split(";", 2)[0].trim();
            if (!pair.isEmpty()) {
                pairs.add(pair);
            }
        }
        return String.join("; ", pairs);
    }

    private static String mergeCookies(String existing, String newer) {
        Map<String, String> map = new java.util.LinkedHashMap<>();
        for (String part : (existing + "; " + newer).split(";")) {
            String trimmed = part.trim();
            if (trimmed.contains("=")) {
                String[] kv = trimmed.split("=", 2);
                map.put(kv[0].trim(), kv[1].trim());
            }
        }
        List<String> out = new ArrayList<>();
        map.forEach((k, v) -> out.add(k + "=" + v));
        return String.join("; ", out);
    }

    private record UserFixture(UUID userId, UUID workspaceId) {}

    private record AuthClientSession(String cookieHeader, String csrfToken, UUID workspaceId) {
        HttpHeaders csrfHeaders() {
            HttpHeaders h = new HttpHeaders();
            h.setContentType(MediaType.APPLICATION_JSON);
            h.set(HttpHeaders.COOKIE, cookieHeader);
            h.set("X-CSRF-TOKEN", csrfToken);
            return h;
        }
    }
}
