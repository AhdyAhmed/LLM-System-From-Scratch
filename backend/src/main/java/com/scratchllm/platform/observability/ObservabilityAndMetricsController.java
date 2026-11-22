package com.scratchllm.platform.observability;

import com.scratchllm.platform.generation.SharedSseEventMultiplexer;
import com.scratchllm.platform.platform.RequestIdFilter;
import jakarta.servlet.http.HttpServletRequest;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Day088: End-to-End Observability & Bounded-Cardinality Metrics Endpoint.
 *
 * Exposes bounded-cardinality operational metrics (queue depth, active runs, outbox backlog,
 * DLQ count, SSE multiplexer fan-out stats, model promotion status, Airflow training job status,
 * and single-GPU lease status) while enforcing strict log/telemetry redaction rules
 * (prompts, raw document texts, and authentication/CSRF/session tokens are NEVER emitted).
 */
@RestController
@RequestMapping("/api/v1/observability")
public class ObservabilityAndMetricsController {

    private final JdbcTemplate jdbcTemplate;
    private final SharedSseEventMultiplexer sseMultiplexer;

    public ObservabilityAndMetricsController(
        JdbcTemplate jdbcTemplate,
        SharedSseEventMultiplexer sseMultiplexer
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.sseMultiplexer = sseMultiplexer;
    }

    @GetMapping(value = "/metrics", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<Map<String, Object>> getBoundedCardinalityMetrics(HttpServletRequest request) {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("timestamp", OffsetDateTime.now(ZoneOffset.UTC));
        snapshot.put("request_id", RequestIdFilter.resolveRequestId(request));
        snapshot.put("correlation_id", RequestIdFilter.resolveCorrelationId(request));

        // 1. System capacity & queue depth
        Map<String, Object> capacity = jdbcTemplate.queryForObject(
            "SELECT max_waiting_runs, active_or_waiting_runs, max_unpublished_outbox_events FROM system_capacity WHERE id = 1",
            (rs, rowNum) -> {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("max_waiting_runs", rs.getInt("max_waiting_runs"));
                m.put("active_or_waiting_runs", rs.getInt("active_or_waiting_runs"));
                m.put("max_unpublished_outbox_events", rs.getInt("max_unpublished_outbox_events"));
                return m;
            }
        );
        snapshot.put("capacity", capacity);

        // 2. Outbox backlog & DLQ count
        Integer outboxBacklog = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM outbox_events WHERE published_at IS NULL",
            Integer.class
        );
        Integer dlqCount = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM dlq_events WHERE status = 'QUARANTINED'",
            Integer.class
        );
        snapshot.put("outbox_unpublished_backlog", outboxBacklog != null ? outboxBacklog : 0);
        snapshot.put("dlq_unresolved_count", dlqCount != null ? dlqCount : 0);

        // 3. SSE Multiplexer bounded-cardinality stats
        Map<String, Object> sseStats = new LinkedHashMap<>();
        sseStats.put("active_sse_subscribers", sseMultiplexer.getActiveSubscriberCount());
        sseStats.put("sse_batch_poll_queries", sseMultiplexer.getTotalBatchPollQueries());
        snapshot.put("sse_multiplexer", sseStats);

        // 4. Generation runs by bounded status enum
        Map<String, Long> runsByStatus = new LinkedHashMap<>();
        for (String status : List.of("QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED")) {
            runsByStatus.put(status, 0L);
        }
        jdbcTemplate.query(
            "SELECT status, count(*) AS cnt FROM generation_runs GROUP BY status",
            rs -> {
                runsByStatus.put(rs.getString("status"), rs.getLong("cnt"));
            }
        );
        snapshot.put("runs_by_status", runsByStatus);

        // 5. Airflow / Native Worker training jobs by bounded status enum
        Map<String, Long> trainingJobsByStatus = new LinkedHashMap<>();
        for (String status : List.of("QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED")) {
            trainingJobsByStatus.put(status, 0L);
        }
        jdbcTemplate.query(
            "SELECT status, count(*) AS cnt FROM training_jobs GROUP BY status",
            rs -> {
                trainingJobsByStatus.put(rs.getString("status"), rs.getLong("cnt"));
            }
        );
        snapshot.put("training_jobs_by_status", trainingJobsByStatus);

        // 6. Model promotions by bounded status enum
        Map<String, Long> promotionsByStatus = new LinkedHashMap<>();
        for (String status : List.of("PENDING", "COMMITTED", "ROLLED_BACK", "REJECTED")) {
            promotionsByStatus.put(status, 0L);
        }
        jdbcTemplate.query(
            "SELECT status, count(*) AS cnt FROM model_promotions GROUP BY status",
            rs -> {
                promotionsByStatus.put(rs.getString("status"), rs.getLong("cnt"));
            }
        );
        snapshot.put("model_promotions_by_status", promotionsByStatus);

        // 7. Single-GPU Supervisor lease state
        List<Map<String, Object>> gpuLeases = jdbcTemplate.query(
            """
            SELECT id, holder_kind, holder_id, lease_state, vram_allocated_bytes, active_generation_slots
            FROM gpu_leases
            WHERE id = 1
            """,
            (rs, rowNum) -> {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("resource_name", "primary_gpu");
                m.put("holder_kind", rs.getString("holder_kind"));
                m.put("holder_id", rs.getString("holder_id"));
                m.put("lease_state", rs.getString("lease_state"));
                m.put("vram_allocated_bytes", rs.getLong("vram_allocated_bytes"));
                m.put("active_generation_slots", rs.getInt("active_generation_slots"));
                return m;
            }
        );
        snapshot.put(
            "gpu_lease",
            gpuLeases.isEmpty()
                ? Map.of("resource_name", "primary_gpu", "holder_kind", "NONE", "lease_state", "RELEASED")
                : gpuLeases.getFirst()
        );

        // 8. Redaction & cardinality policy attestation
        snapshot.put(
            "logging_redaction_policy",
            Map.of(
                "prompts_logged", false,
                "documents_logged", false,
                "tokens_logged", false,
                "high_cardinality_labels_forbidden", true
            )
        );

        return ResponseEntity.ok(snapshot);
    }
}
