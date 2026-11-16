package com.scratchllm.platform.generation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.scratchllm.platform.infrastructure.KafkaEventsInfrastructure;
import com.scratchllm.platform.infrastructure.KafkaEventsInfrastructure.KafkaContractBridge;
import com.scratchllm.platform.modelcatalog.ModelPromotionService;
import com.scratchllm.platform.platform.ApiException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

/**
 * Day063 & Day067:
 * - Polling Outbox Relay claiming unpublished `outbox_events` with `FOR UPDATE SKIP LOCKED`,
 *   publishing ID-only events to Kafka (`generation.requested.v1`, `model.released.v1`),
 *   and marking `published_at` only after broker acknowledgement.
 * - Idempotent Kafka Inbox Consumer inserting `consumer_inbox` (`ON CONFLICT (consumer_name, event_id) DO NOTHING`)
 *   and `work_items` (`ON CONFLICT (work_kind, target_id) DO NOTHING`) atomically in one PostgreSQL transaction.
 * - Bounded retry and Dead-Letter Quarantine (`dlq_events` + `.dlq.v1` Kafka topic) for poison/invalid messages,
 *   plus operator inspection and manual DLQ replay preserving original `event_id` and revalidating terminal state.
 */
@Service
public class OutboxRelayAndInboxService {

    public static final String CONSUMER_NAME_GENERATION_DISPATCHER = "generation-dispatcher-inbox";
    public static final int MAX_CONSUMER_RETRIES = 3;

    private final JdbcTemplate jdbcTemplate;
    private final TransactionTemplate transactionTemplate;
    private final KafkaContractBridge kafkaContractBridge;
    private final KafkaTemplate<String, String> kafkaTemplate;
    private final ModelPromotionService modelPromotionService;
    private final ObjectMapper objectMapper;
    private final AtomicBoolean brokerOutageSimulated = new AtomicBoolean(false);

    public OutboxRelayAndInboxService(
        JdbcTemplate jdbcTemplate,
        PlatformTransactionManager transactionManager,
        KafkaContractBridge kafkaContractBridge,
        KafkaTemplate<String, String> kafkaTemplate,
        ModelPromotionService modelPromotionService,
        ObjectMapper objectMapper
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.transactionTemplate = new TransactionTemplate(transactionManager);
        this.kafkaContractBridge = kafkaContractBridge;
        this.kafkaTemplate = kafkaTemplate;
        this.modelPromotionService = modelPromotionService;
        this.objectMapper = objectMapper;
    }

    public void setBrokerOutageSimulated(boolean simulatedOutage) {
        this.brokerOutageSimulated.set(simulatedOutage);
    }

    public boolean isBrokerOutageSimulated() {
        return this.brokerOutageSimulated.get();
    }

    /**
     * Claims up to `batchSize` unpublished outbox rows using `FOR UPDATE SKIP LOCKED`,
     * publishes each to Kafka, and marks `published_at = clock_timestamp()` upon broker ack.
     */
    public RelayBatchResult relayBatch(int batchSize) {
        List<OutboxEventRow> claimed = transactionTemplate.execute(status -> {
            List<OutboxEventRow> rows = jdbcTemplate.query(
                """
                SELECT event_id, topic, schema_version, aggregate_key, payload_json::text AS payload_json, publish_attempts
                FROM outbox_events
                WHERE published_at IS NULL
                  AND (relay_lease_until IS NULL OR relay_lease_until <= clock_timestamp())
                  AND next_retry_at <= clock_timestamp()
                ORDER BY created_at ASC
                LIMIT ?
                FOR UPDATE SKIP LOCKED
                """,
                (rs, rowNum) -> new OutboxEventRow(
                    UUID.fromString(rs.getString("event_id")),
                    rs.getString("topic"),
                    rs.getInt("schema_version"),
                    rs.getString("aggregate_key"),
                    rs.getString("payload_json"),
                    rs.getInt("publish_attempts")
                ),
                batchSize
            );
            for (OutboxEventRow row : rows) {
                jdbcTemplate.update(
                    """
                    UPDATE outbox_events
                    SET publish_attempts = publish_attempts + 1,
                        relay_lease_until = clock_timestamp() + interval '10 seconds'
                    WHERE event_id = ?
                    """,
                    row.eventId()
                );
            }
            return rows;
        });

        if (claimed == null || claimed.isEmpty()) {
            return new RelayBatchResult(0, 0, 0, List.of());
        }

        int published = 0;
        int failed = 0;
        List<OutboxEventRow> publishedEvents = new ArrayList<>();

        for (OutboxEventRow row : claimed) {
            try {
                if (brokerOutageSimulated.get()) {
                    throw new IllegalStateException("Simulated Kafka broker outage: broker unreachable");
                }
                kafkaContractBridge.publishValidated(row.topic(), row.aggregateKey(), row.payloadJson());
                jdbcTemplate.update(
                    """
                    UPDATE outbox_events
                    SET published_at = clock_timestamp(),
                        relay_lease_until = NULL,
                        last_error = NULL
                    WHERE event_id = ?
                    """,
                    row.eventId()
                );
                published++;
                publishedEvents.add(row);
            } catch (Exception ex) {
                failed++;
                jdbcTemplate.update(
                    """
                    UPDATE outbox_events
                    SET relay_lease_until = NULL,
                        last_error = ?,
                        next_retry_at = clock_timestamp()
                    WHERE event_id = ?
                    """,
                    ex.getMessage(),
                    row.eventId()
                );
            }
        }

        return new RelayBatchResult(claimed.size(), published, failed, publishedEvents);
    }

    /**
     * Convenience pipeline step that relays unpublished outbox rows to Kafka and immediately processes
     * the published events through the idempotent inbox consumer.
     */
    public RelayBatchResult relayAndConsumePending(int batchSize) {
        RelayBatchResult relayResult = relayBatch(batchSize);
        for (OutboxEventRow published : relayResult.publishedEvents()) {
            consumeKafkaEvent(published.topic(), published.aggregateKey(), published.payloadJson());
        }
        return relayResult;
    }

    /**
     * Processes a Kafka event (`generation.requested.v1` or `model.released.v1`) with bounded retries,
     * poison-message DLQ routing, and atomic PostgreSQL `consumer_inbox` + `work_items` persistence.
     */
    public InboxConsumeResult consumeKafkaEvent(String topic, String aggregateKey, String rawJson) {
        UUID parsedEventId = tryExtractEventId(rawJson);

        // 1. Validate strict ID-only Kafka contract; invalid envelopes route directly to DLQ after bounded retries
        try {
            kafkaContractBridge.validateEnvelope(topic, rawJson);
        } catch (IllegalArgumentException validationEx) {
            UUID dlqEventId = parsedEventId != null ? parsedEventId : UUID.randomUUID();
            routeToDlq(
                dlqEventId,
                topic,
                aggregateKey != null ? aggregateKey : "unknown",
                rawJson != null ? rawJson : "{}",
                "CONTRACT_VALIDATION_FAILED",
                validationEx.getMessage(),
                MAX_CONSUMER_RETRIES
            );
            return new InboxConsumeResult(dlqEventId, false, false, true, "CONTRACT_VALIDATION_FAILED");
        }

        try {
            JsonNode root = objectMapper.readTree(rawJson);
            UUID eventId = UUID.fromString(root.get("event_id").asText());

            if (KafkaEventsInfrastructure.TOPIC_MODEL_RELEASED.equals(topic)) {
                var promoResult = modelPromotionService.consumeModelReleasedEvent(rawJson);
                return new InboxConsumeResult(
                    eventId,
                    !promoResult.duplicate_ignored(),
                    promoResult.applied(),
                    false,
                    promoResult.stale_generation_rejected() ? "STALE_RELEASE_GENERATION_REJECTED" : null
                );
            }

            if (KafkaEventsInfrastructure.TOPIC_GENERATION_REQUESTED.equals(topic)) {
                UUID runId = UUID.fromString(root.get("run_id").asText());
                String payloadHash = sha256Hex(rawJson);

                return transactionTemplate.execute(status -> {
                    int insertedInbox = jdbcTemplate.update(
                        """
                        INSERT INTO consumer_inbox (id, consumer_name, event_id, payload_hash)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT (consumer_name, event_id) DO NOTHING
                        """,
                        UUID.randomUUID(),
                        CONSUMER_NAME_GENERATION_DISPATCHER,
                        eventId,
                        payloadHash
                    );

                    if (insertedInbox == 0) {
                        // Duplicate Kafka delivery: idempotent no-op!
                        return new InboxConsumeResult(eventId, false, false, false, "DUPLICATE_EVENT_IGNORED");
                    }

                    // Check run existence and terminal state before creating work_item
                    List<String> runStatuses = jdbcTemplate.query(
                        "SELECT status FROM generation_runs WHERE id = ? FOR UPDATE",
                        (rs, rowNum) -> rs.getString("status"),
                        runId
                    );
                    if (runStatuses.isEmpty()) {
                        // Poison reference to non-existent run: route to DLQ after finite retries
                        routeToDlq(
                            eventId,
                            topic,
                            aggregateKey,
                            rawJson,
                            "TARGET_RUN_NOT_FOUND",
                            "generation_runs row does not exist for run_id=" + runId,
                            MAX_CONSUMER_RETRIES
                        );
                        return new InboxConsumeResult(eventId, true, false, true, "TARGET_RUN_NOT_FOUND");
                    }

                    String currentRunStatus = runStatuses.getFirst();
                    if (!"QUEUED".equals(currentRunStatus)) {
                        // Run already RUNNING or in terminal state (COMPLETED, CANCELLED, FAILED)
                        return new InboxConsumeResult(eventId, true, false, false, "RUN_ALREADY_" + currentRunStatus);
                    }

                    int insertedWork = jdbcTemplate.update(
                        """
                        INSERT INTO work_items (id, work_kind, target_id, state)
                        VALUES (?, 'GENERATION_RUN', ?, 'PENDING')
                        ON CONFLICT (work_kind, target_id) DO NOTHING
                        """,
                        UUID.randomUUID(),
                        runId
                    );

                    return new InboxConsumeResult(eventId, true, insertedWork > 0, false, null);
                });
            }

            throw new IllegalArgumentException("Unsupported Kafka topic: " + topic);
        } catch (Exception ex) {
            UUID dlqEventId = parsedEventId != null ? parsedEventId : UUID.randomUUID();
            routeToDlq(
                dlqEventId,
                topic,
                aggregateKey != null ? aggregateKey : "unknown",
                rawJson,
                "CONSUMER_PROCESSING_FAILURE",
                ex.getMessage(),
                MAX_CONSUMER_RETRIES
            );
            return new InboxConsumeResult(dlqEventId, false, false, true, "CONSUMER_PROCESSING_FAILURE");
        }
    }

    /**
     * Routes a poison or retry-exhausted Kafka event to its `.dlq.v1` Kafka topic and persists a
     * durable `QUARANTINED` row in `dlq_events` for operator inspection and manual replay.
     */
    public void routeToDlq(
        UUID eventId,
        String originalTopic,
        String aggregateKey,
        String rawJson,
        String failureReason,
        String errorDetail,
        int retryCount
    ) {
        String dlqTopic = resolveDlqTopic(originalTopic);
        try {
            kafkaTemplate.send(dlqTopic, aggregateKey, rawJson).get(5, TimeUnit.SECONDS);
        } catch (Exception ignored) {
            // Even if Kafka broker is temporarily down, PostgreSQL dlq_events is durable
        }

        jdbcTemplate.update(
            """
            INSERT INTO dlq_events (
                id, event_id, original_topic, dlq_topic, aggregate_key, payload_json,
                failure_reason, error_detail, retry_count, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'QUARANTINED')
            ON CONFLICT (event_id) DO UPDATE
            SET failure_reason = EXCLUDED.failure_reason,
                error_detail = EXCLUDED.error_detail,
                retry_count = EXCLUDED.retry_count,
                status = 'QUARANTINED'
            """,
            UUID.randomUUID(),
            eventId,
            originalTopic,
            dlqTopic,
            aggregateKey,
            rawJson,
            failureReason,
            errorDetail,
            retryCount
        );
    }

    public List<DlqEventRow> listDlqEvents() {
        return jdbcTemplate.query(
            """
            SELECT id, event_id, original_topic, dlq_topic, aggregate_key, payload_json,
                   failure_reason, error_detail, retry_count, status, created_at, replayed_at
            FROM dlq_events
            ORDER BY created_at DESC
            """,
            (rs, rowNum) -> new DlqEventRow(
                UUID.fromString(rs.getString("id")),
                UUID.fromString(rs.getString("event_id")),
                rs.getString("original_topic"),
                rs.getString("dlq_topic"),
                rs.getString("aggregate_key"),
                rs.getString("payload_json"),
                rs.getString("failure_reason"),
                rs.getString("error_detail"),
                rs.getInt("retry_count"),
                rs.getString("status"),
                rs.getObject("created_at", OffsetDateTime.class),
                rs.getObject("replayed_at", OffsetDateTime.class)
            )
        );
    }

    /**
     * Manual operator DLQ replay (Day067):
     * - Preserves the original `event_id`
     * - Re-validates the target run's current state in `generation_runs` so a completed,
     *   cancelled, or failed run is NEVER restarted by a late DLQ replay!
     */
    public DlqReplayResult replayDlqEvent(UUID eventId) {
        return transactionTemplate.execute(status -> {
            List<DlqEventRow> rows = jdbcTemplate.query(
                """
                SELECT id, event_id, original_topic, dlq_topic, aggregate_key, payload_json,
                       failure_reason, error_detail, retry_count, status, created_at, replayed_at
                FROM dlq_events
                WHERE event_id = ?
                FOR UPDATE
                """,
                (rs, rowNum) -> new DlqEventRow(
                    UUID.fromString(rs.getString("id")),
                    UUID.fromString(rs.getString("event_id")),
                    rs.getString("original_topic"),
                    rs.getString("dlq_topic"),
                    rs.getString("aggregate_key"),
                    rs.getString("payload_json"),
                    rs.getString("failure_reason"),
                    rs.getString("error_detail"),
                    rs.getInt("retry_count"),
                    rs.getString("status"),
                    rs.getObject("created_at", OffsetDateTime.class),
                    rs.getObject("replayed_at", OffsetDateTime.class)
                ),
                eventId
            );

            if (rows.isEmpty()) {
                throw new ApiException(HttpStatus.NOT_FOUND, "DLQ_EVENT_NOT_FOUND", "DLQ event not found: " + eventId);
            }

            DlqEventRow dlqRow = rows.getFirst();
            UUID targetRunId = tryExtractUuidField(dlqRow.payloadJson(), "run_id");
            if (targetRunId == null) {
                jdbcTemplate.update(
                    "UPDATE dlq_events SET status = 'DISCARDED', replayed_at = clock_timestamp() WHERE event_id = ?",
                    eventId
                );
                return new DlqReplayResult(eventId, null, "DISCARDED_INVALID_PAYLOAD", null, false);
            }

            List<String> runStatuses = jdbcTemplate.query(
                "SELECT status FROM generation_runs WHERE id = ? FOR UPDATE",
                (rs, rowNum) -> rs.getString("status"),
                targetRunId
            );
            if (runStatuses.isEmpty()) {
                return new DlqReplayResult(eventId, targetRunId, "TARGET_RUN_STILL_MISSING", null, false);
            }

            String currentStatus = runStatuses.getFirst();
            if ("COMPLETED".equals(currentStatus) || "CANCELLED".equals(currentStatus) || "FAILED".equals(currentStatus)) {
                jdbcTemplate.update(
                    "UPDATE dlq_events SET status = 'REPLAYED', replayed_at = clock_timestamp() WHERE event_id = ?",
                    eventId
                );
                return new DlqReplayResult(
                    eventId,
                    targetRunId,
                    "SKIPPED_TERMINAL_RUN",
                    currentStatus,
                    false
                );
            }

            // Preserve original event_id in consumer_inbox and ensure PENDING work_item exists
            jdbcTemplate.update(
                """
                INSERT INTO consumer_inbox (id, consumer_name, event_id, payload_hash)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (consumer_name, event_id) DO NOTHING
                """,
                UUID.randomUUID(),
                CONSUMER_NAME_GENERATION_DISPATCHER,
                dlqRow.eventId(),
                sha256Hex(dlqRow.payloadJson())
            );

            int createdWork = jdbcTemplate.update(
                """
                INSERT INTO work_items (id, work_kind, target_id, state)
                VALUES (?, 'GENERATION_RUN', ?, 'PENDING')
                ON CONFLICT (work_kind, target_id) DO NOTHING
                """,
                UUID.randomUUID(),
                targetRunId
            );

            jdbcTemplate.update(
                "UPDATE dlq_events SET status = 'REPLAYED', replayed_at = clock_timestamp() WHERE event_id = ?",
                eventId
            );

            return new DlqReplayResult(
                eventId,
                targetRunId,
                "REPLAYED_TO_WORK_ITEM",
                currentStatus,
                createdWork > 0
            );
        });
    }

    private static String resolveDlqTopic(String originalTopic) {
        if (KafkaEventsInfrastructure.TOPIC_MODEL_RELEASED.equals(originalTopic)) {
            return KafkaEventsInfrastructure.TOPIC_MODEL_DLQ;
        }
        if (KafkaEventsInfrastructure.TOPIC_DOCUMENT_INGESTION.equals(originalTopic)) {
            return KafkaEventsInfrastructure.TOPIC_DOCUMENT_DLQ;
        }
        return KafkaEventsInfrastructure.TOPIC_GENERATION_DLQ;
    }

    private UUID tryExtractEventId(String rawJson) {
        return tryExtractUuidField(rawJson, "event_id");
    }

    private UUID tryExtractUuidField(String rawJson, String fieldName) {
        if (rawJson == null || rawJson.isBlank()) {
            return null;
        }
        try {
            JsonNode root = objectMapper.readTree(rawJson);
            if (root.hasNonNull(fieldName)) {
                return UUID.fromString(root.get(fieldName).asText());
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    private static String sha256Hex(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(input.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException(ex);
        }
    }

    public record OutboxEventRow(
        UUID eventId,
        String topic,
        int schemaVersion,
        String aggregateKey,
        String payloadJson,
        int publishAttempts
    ) {}

    public record RelayBatchResult(
        int claimedCount,
        int publishedCount,
        int failedCount,
        List<OutboxEventRow> publishedEvents
    ) {}

    public record InboxConsumeResult(
        UUID eventId,
        boolean newlyInsertedInbox,
        boolean workItemCreated,
        boolean routedToDlq,
        String note
    ) {}

    public record DlqEventRow(
        UUID id,
        UUID eventId,
        String originalTopic,
        String dlqTopic,
        String aggregateKey,
        String payloadJson,
        String failureReason,
        String errorDetail,
        int retryCount,
        String status,
        OffsetDateTime createdAt,
        OffsetDateTime replayedAt
    ) {}

    public record DlqReplayResult(
        UUID event_id,
        UUID run_id,
        String outcome,
        String run_status,
        boolean work_item_created
    ) {}
}
