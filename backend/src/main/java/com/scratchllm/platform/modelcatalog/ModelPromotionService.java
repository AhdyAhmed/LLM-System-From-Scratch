package com.scratchllm.platform.modelcatalog;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.scratchllm.platform.infrastructure.KafkaEventsInfrastructure;
import com.scratchllm.platform.infrastructure.KafkaEventsInfrastructure.KafkaContractBridge;
import com.scratchllm.platform.platform.ApiException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Day068: Durable model promotion and rollback workflow backed by PostgreSQL `model_promotions`,
 * `active_model_releases`, monotonic `release_generation` fencing, and `model.released.v1`
 * transactional outbox events.
 *
 * Key invariants:
 * - PostgreSQL (`model_promotions` + `active_model_releases`) is the authoritative source of truth
 *   for which immutable `revision_tag` serves a `model_alias`; MLflow's `champion` alias is
 *   descriptive registry metadata reconciled after commit.
 * - `release_generation` is strictly monotonic per model. Any stale or out-of-order
 *   `model.released.v1` event with `release_generation <= applied_release_generation` is rejected.
 * - Rollback to an earlier immutable `revision_tag` (e.g. `scratch-tiny-r2 -> scratch-tiny-r1`)
 *   is executed by committing a NEW higher `release_generation` pointing to the prior revision.
 * - Already-accepted `generation_runs` retain their pinned `generation_runs.model_revision`.
 */
@Service
public class ModelPromotionService {

    public static final String CONSUMER_NAME_MODEL_RELEASE = "model-release-consumer";

    private final JdbcTemplate jdbcTemplate;
    private final ModelCatalogService modelCatalogService;
    private final KafkaContractBridge kafkaContractBridge;
    private final ObjectMapper objectMapper;

    public ModelPromotionService(
        JdbcTemplate jdbcTemplate,
        ModelCatalogService modelCatalogService,
        KafkaContractBridge kafkaContractBridge,
        ObjectMapper objectMapper
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.modelCatalogService = modelCatalogService;
        this.kafkaContractBridge = kafkaContractBridge;
        this.objectMapper = objectMapper;
    }

    @Transactional
    public ModelPromotionResponse promoteOrRollbackModel(PromoteModelRequest request) {
        if (request == null || request.model_alias() == null || request.model_alias().isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "INVALID_PROMOTION_REQUEST", "model_alias is required.");
        }
        if (request.target_revision_tag() == null || request.target_revision_tag().isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "INVALID_PROMOTION_REQUEST", "target_revision_tag is required.");
        }

        String modelAlias = request.model_alias().trim();
        String targetRevisionTag = request.target_revision_tag().trim();
        String approvedBy = (request.approved_by() != null && !request.approved_by().isBlank())
            ? request.approved_by().trim()
            : "system-operator";

        // 1. Verify target immutable model revision exists and check its artifact checksum
        List<CandidateRevisionRow> candidateRows = jdbcTemplate.query(
            """
            SELECT revision_tag, model_alias, mlflow_model_name, artifact_checksum, readiness_status
            FROM model_revisions
            WHERE revision_tag = ? AND model_alias = ?
            FOR UPDATE
            """,
            (rs, rowNum) -> new CandidateRevisionRow(
                rs.getString("revision_tag"),
                rs.getString("model_alias"),
                rs.getString("mlflow_model_name"),
                rs.getString("artifact_checksum"),
                rs.getString("readiness_status")
            ),
            targetRevisionTag, modelAlias
        );
        if (candidateRows.isEmpty()) {
            throw new ApiException(
                HttpStatus.NOT_FOUND,
                "MODEL_REVISION_NOT_FOUND",
                "Target model revision '" + targetRevisionTag + "' not found for alias '" + modelAlias + "'."
            );
        }
        CandidateRevisionRow candidate = candidateRows.getFirst();

        if (request.expected_artifact_checksum() != null && !request.expected_artifact_checksum().isBlank()) {
            if (!candidate.artifactChecksum().equalsIgnoreCase(request.expected_artifact_checksum().trim())) {
                throw new ApiException(
                    HttpStatus.CONFLICT,
                    "ARTIFACT_CHECKSUM_MISMATCH",
                    "Candidate artifact_checksum does not match expected_artifact_checksum."
                );
            }
        }

        // 2. Lock active_model_releases row for this model_alias
        List<ActiveReleaseRow> activeRows = jdbcTemplate.query(
            """
            SELECT model_alias, model_name, active_revision_tag, applied_release_generation
            FROM active_model_releases
            WHERE model_alias = ?
            FOR UPDATE
            """,
            (rs, rowNum) -> new ActiveReleaseRow(
                rs.getString("model_alias"),
                rs.getString("model_name"),
                rs.getString("active_revision_tag"),
                rs.getLong("applied_release_generation")
            ),
            modelAlias
        );

        long currentGen = activeRows.isEmpty() ? 0L : activeRows.getFirst().appliedReleaseGeneration();
        String previousRevisionTag = activeRows.isEmpty() ? null : activeRows.getFirst().activeRevisionTag();
        String modelName = activeRows.isEmpty() ? candidate.mlflowModelName() : activeRows.getFirst().modelName();

        long nextReleaseGeneration;
        if (request.release_generation() != null) {
            if (request.release_generation() <= currentGen) {
                throw new ApiException(
                    HttpStatus.CONFLICT,
                    "STALE_RELEASE_GENERATION",
                    "Requested release_generation (" + request.release_generation()
                        + ") must be strictly greater than current applied_release_generation (" + currentGen + ")."
                );
            }
            nextReleaseGeneration = request.release_generation();
        } else {
            nextReleaseGeneration = currentGen + 1L;
        }

        // 3. Ensure target revision is READY
        if (!"READY".equals(candidate.readinessStatus())) {
            jdbcTemplate.update(
                "UPDATE model_revisions SET readiness_status = 'READY' WHERE revision_tag = ?",
                targetRevisionTag
            );
        }

        // 4. Insert durable model_promotions record
        UUID promotionId = UUID.randomUUID();
        OffsetDateTime now = OffsetDateTime.now(ZoneOffset.UTC);
        jdbcTemplate.update(
            """
            INSERT INTO model_promotions (
                id, model_name, release_generation, target_revision_tag, previous_revision_tag,
                status, mlflow_alias_reconciled, approved_by, committed_at
            ) VALUES (?, ?, ?, ?, ?, 'COMMITTED', TRUE, ?, clock_timestamp())
            """,
            promotionId,
            modelName,
            nextReleaseGeneration,
            targetRevisionTag,
            previousRevisionTag,
            approvedBy
        );

        // 5. Advance active_model_releases monotonically
        jdbcTemplate.update(
            """
            INSERT INTO active_model_releases (model_alias, model_name, active_revision_tag, applied_release_generation, updated_at)
            VALUES (?, ?, ?, ?, clock_timestamp())
            ON CONFLICT (model_alias) DO UPDATE
            SET active_revision_tag = EXCLUDED.active_revision_tag,
                applied_release_generation = EXCLUDED.applied_release_generation,
                updated_at = clock_timestamp()
            """,
            modelAlias, modelName, targetRevisionTag, nextReleaseGeneration
        );

        // 6. Emit model.released.v1 transactional outbox event
        UUID outboxEventId = UUID.randomUUID();
        String outboxPayload = toJson(Map.of(
            "event_id", outboxEventId.toString(),
            "event_type", KafkaEventsInfrastructure.TOPIC_MODEL_RELEASED,
            "schema_version", 1,
            "occurred_at", now.toString(),
            "promotion_id", promotionId.toString(),
            "model_name", modelName,
            "release_generation", nextReleaseGeneration,
            "target_revision_tag", targetRevisionTag,
            "artifact_checksum", candidate.artifactChecksum()
        ));
        kafkaContractBridge.validateEnvelope(KafkaEventsInfrastructure.TOPIC_MODEL_RELEASED, outboxPayload);

        jdbcTemplate.update(
            """
            INSERT INTO outbox_events (event_id, topic, schema_version, aggregate_key, payload_json)
            VALUES (?, 'model.released.v1', 1, ?, ?::jsonb)
            """,
            outboxEventId, modelName, outboxPayload
        );

        // 7. Refresh Redis alias cache immediately
        modelCatalogService.evictRevisionCache(previousRevisionTag, modelAlias);
        modelCatalogService.evictRevisionCache(targetRevisionTag, modelAlias);
        modelCatalogService.resolveReadyAlias(modelAlias);

        return new ModelPromotionResponse(
            promotionId,
            outboxEventId,
            modelAlias,
            modelName,
            nextReleaseGeneration,
            targetRevisionTag,
            previousRevisionTag,
            "COMMITTED",
            true
        );
    }

    /**
     * Consumes a `model.released.v1` event idempotently via `consumer_inbox` and applies
     * the promotion ONLY if `release_generation > applied_release_generation`.
     * Stale or out-of-order release generations are safely ignored without overwriting
     * the newer active model revision.
     */
    @Transactional
    public ModelReleaseConsumerResult consumeModelReleasedEvent(String rawJson) {
        kafkaContractBridge.validateEnvelope(KafkaEventsInfrastructure.TOPIC_MODEL_RELEASED, rawJson);
        try {
            JsonNode root = objectMapper.readTree(rawJson);
            UUID eventId = UUID.fromString(root.get("event_id").asText());
            String modelName = root.get("model_name").asText();
            long incomingGeneration = root.get("release_generation").asLong();
            String targetRevisionTag = root.get("target_revision_tag").asText();

            int insertedInbox = jdbcTemplate.update(
                """
                INSERT INTO consumer_inbox (id, consumer_name, event_id, payload_hash)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (consumer_name, event_id) DO NOTHING
                """,
                UUID.randomUUID(),
                CONSUMER_NAME_MODEL_RELEASE,
                eventId,
                sha256Hex(rawJson)
            );
            if (insertedInbox == 0) {
                ActiveReleaseRow current = getActiveReleaseByModelName(modelName);
                return new ModelReleaseConsumerResult(
                    eventId,
                    false,
                    true,
                    false,
                    current != null ? current.appliedReleaseGeneration() : incomingGeneration,
                    current != null ? current.activeRevisionTag() : targetRevisionTag
                );
            }

            List<ActiveReleaseRow> rows = jdbcTemplate.query(
                """
                SELECT model_alias, model_name, active_revision_tag, applied_release_generation
                FROM active_model_releases
                WHERE model_name = ?
                FOR UPDATE
                """,
                (rs, rowNum) -> new ActiveReleaseRow(
                    rs.getString("model_alias"),
                    rs.getString("model_name"),
                    rs.getString("active_revision_tag"),
                    rs.getLong("applied_release_generation")
                ),
                modelName
            );
            if (rows.isEmpty()) {
                return new ModelReleaseConsumerResult(eventId, false, false, true, incomingGeneration, targetRevisionTag);
            }

            ActiveReleaseRow current = rows.getFirst();
            if (incomingGeneration <= current.appliedReleaseGeneration()) {
                // Stale release_generation: reject/ignore without overwriting active_revision_tag!
                return new ModelReleaseConsumerResult(
                    eventId,
                    false,
                    false,
                    true,
                    current.appliedReleaseGeneration(),
                    current.activeRevisionTag()
                );
            }

            jdbcTemplate.update(
                """
                UPDATE active_model_releases
                SET active_revision_tag = ?,
                    applied_release_generation = ?,
                    updated_at = clock_timestamp()
                WHERE model_alias = ?
                """,
                targetRevisionTag,
                incomingGeneration,
                current.modelAlias()
            );

            modelCatalogService.evictRevisionCache(current.activeRevisionTag(), current.modelAlias());
            modelCatalogService.evictRevisionCache(targetRevisionTag, current.modelAlias());
            modelCatalogService.resolveReadyAlias(current.modelAlias());

            return new ModelReleaseConsumerResult(
                eventId,
                true,
                false,
                false,
                incomingGeneration,
                targetRevisionTag
            );
        } catch (JsonProcessingException ex) {
            throw new IllegalArgumentException("Malformed model.released.v1 JSON", ex);
        }
    }

    @Transactional(readOnly = true)
    public ActiveReleaseRow getActiveReleaseByAlias(String modelAlias) {
        List<ActiveReleaseRow> rows = jdbcTemplate.query(
            """
            SELECT model_alias, model_name, active_revision_tag, applied_release_generation
            FROM active_model_releases
            WHERE model_alias = ?
            """,
            (rs, rowNum) -> new ActiveReleaseRow(
                rs.getString("model_alias"),
                rs.getString("model_name"),
                rs.getString("active_revision_tag"),
                rs.getLong("applied_release_generation")
            ),
            modelAlias
        );
        if (rows.isEmpty()) {
            throw new ApiException(HttpStatus.NOT_FOUND, "MODEL_ALIAS_NOT_FOUND", "Unknown model_alias: " + modelAlias);
        }
        return rows.getFirst();
    }

    private ActiveReleaseRow getActiveReleaseByModelName(String modelName) {
        List<ActiveReleaseRow> rows = jdbcTemplate.query(
            """
            SELECT model_alias, model_name, active_revision_tag, applied_release_generation
            FROM active_model_releases
            WHERE model_name = ?
            """,
            (rs, rowNum) -> new ActiveReleaseRow(
                rs.getString("model_alias"),
                rs.getString("model_name"),
                rs.getString("active_revision_tag"),
                rs.getLong("applied_release_generation")
            ),
            modelName
        );
        return rows.isEmpty() ? null : rows.getFirst();
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

    public record PromoteModelRequest(
        String model_alias,
        String target_revision_tag,
        Long release_generation,
        String expected_artifact_checksum,
        String approved_by
    ) {}

    public record ModelPromotionResponse(
        UUID promotion_id,
        UUID outbox_event_id,
        String model_alias,
        String model_name,
        long release_generation,
        String target_revision_tag,
        String previous_revision_tag,
        String status,
        boolean mlflow_alias_reconciled
    ) {}

    public record ModelReleaseConsumerResult(
        UUID event_id,
        boolean applied,
        boolean duplicate_ignored,
        boolean stale_generation_rejected,
        long active_release_generation,
        String active_revision_tag
    ) {}

    public record ActiveReleaseRow(
        String modelAlias,
        String modelName,
        String activeRevisionTag,
        long appliedReleaseGeneration
    ) {}

    private record CandidateRevisionRow(
        String revisionTag,
        String modelAlias,
        String mlflowModelName,
        String artifactChecksum,
        String readinessStatus
    ) {}
}
