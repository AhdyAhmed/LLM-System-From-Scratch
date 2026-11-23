package com.scratchllm.platform.observability;

import com.scratchllm.platform.platform.ApiException;
import jakarta.annotation.PostConstruct;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

/**
 * Day098: Startup Configuration Validation, Dependency Readiness Probe, and Graceful Drain Coordinator.
 *
 * Key invariants:
 * - Invalid runtime limits or blank internal credentials fail fast at startup (`@PostConstruct` /
 *   `validateStartupConfiguration(...)`).
 * - Entering `DRAINING` mode (`initiateGracefulDrain`) immediately stops new run admission
 *   (`503 SERVICE_DRAINING`), preserves all `outbox_events` and `work_items` state in PostgreSQL,
 *   and transitions any idle `INFERENCE` GPU lease to `RELEASED` so active GPU ownership is cleanly resolved.
 */
@Service
public class RuntimeReadinessAndDrainService {

    private final JdbcTemplate jdbcTemplate;
    private final StringRedisTemplate redisTemplate;
    private final int dailyAllowanceTokens;
    private final int sessionTtlSeconds;
    private final int runMaxRequests;
    private final int runWindowSeconds;
    private final String internalToken;
    private final String replicaId;
    private final AtomicBoolean draining = new AtomicBoolean(false);

    public RuntimeReadinessAndDrainService(
        JdbcTemplate jdbcTemplate,
        StringRedisTemplate redisTemplate,
        @Value("${scratchllm.quota.daily-allowance-tokens:4096}") int dailyAllowanceTokens,
        @Value("${scratchllm.auth.session-ttl-seconds:28800}") int sessionTtlSeconds,
        @Value("${scratchllm.ratelimit.run-max-requests:15}") int runMaxRequests,
        @Value("${scratchllm.ratelimit.run-window-seconds:60}") int runWindowSeconds,
        @Value("${scratchllm.inference.internal-token:scratchllm-internal-dev-token-day051}") String internalToken,
        @Value("${scratchllm.replica.id:replica-1}") String replicaId
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.redisTemplate = redisTemplate;
        this.dailyAllowanceTokens = dailyAllowanceTokens;
        this.sessionTtlSeconds = sessionTtlSeconds;
        this.runMaxRequests = runMaxRequests;
        this.runWindowSeconds = runWindowSeconds;
        this.internalToken = internalToken;
        this.replicaId = replicaId;
    }

    @PostConstruct
    public void validateStartupConfigurationOnBoot() {
        validateConfigurationLimits(
            dailyAllowanceTokens,
            sessionTtlSeconds,
            runMaxRequests,
            runWindowSeconds,
            internalToken
        );
    }

    public static void validateConfigurationLimits(
        int dailyAllowanceTokens,
        int sessionTtlSeconds,
        int runMaxRequests,
        int runWindowSeconds,
        String internalToken
    ) {
        if (dailyAllowanceTokens <= 0) {
            throw new IllegalStateException("INVALID_CONFIG_LIMIT: dailyAllowanceTokens must be > 0");
        }
        if (sessionTtlSeconds < 60) {
            throw new IllegalStateException("INVALID_CONFIG_LIMIT: sessionTtlSeconds must be >= 60");
        }
        if (runMaxRequests <= 0 || runWindowSeconds <= 0) {
            throw new IllegalStateException("INVALID_CONFIG_LIMIT: rate-limit runMaxRequests and runWindowSeconds must be > 0");
        }
        if (internalToken == null || internalToken.trim().length() < 16) {
            throw new IllegalStateException("INVALID_CONFIG_SECRET: internalToken must be at least 16 characters");
        }
    }

    public boolean isDraining() {
        return draining.get();
    }

    public void assertAcceptingAdmissions() {
        if (draining.get()) {
            throw new ApiException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "SERVICE_DRAINING",
                "API replica '" + replicaId + "' is draining; new run admissions are paused while in-flight work completes."
            );
        }
    }

    public Map<String, Object> initiateGracefulDrain(String reason) {
        draining.set(true);

        // Resolve idle INFERENCE GPU lease ownership on drain without disturbing active TRAINING leases
        // or deleting outbox_events / work_items
        int releasedGpuRows = jdbcTemplate.update(
            """
            UPDATE gpu_leases
            SET holder_kind = 'NONE',
                holder_id = 'none',
                native_pid = 0,
                process_create_time_utc = 0.0,
                lease_state = 'RELEASED',
                vram_allocated_bytes = 0,
                active_generation_slots = 0,
                updated_at = now()
            WHERE id = 1
              AND holder_kind = 'INFERENCE'
              AND active_generation_slots = 0
            """
        );

        Integer preservedOutbox = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM outbox_events WHERE published_at IS NULL",
            Integer.class
        );
        Integer preservedWorkItems = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM work_items WHERE state IN ('PENDING', 'LEASED')",
            Integer.class
        );

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("replica_id", replicaId);
        result.put("draining", true);
        result.put("reason", (reason != null && !reason.isBlank()) ? reason : "operator_initiated_drain");
        result.put("idle_gpu_leases_released", releasedGpuRows);
        result.put("preserved_unpublished_outbox_events", preservedOutbox != null ? preservedOutbox : 0);
        result.put("preserved_active_work_items", preservedWorkItems != null ? preservedWorkItems : 0);
        result.put("timestamp", OffsetDateTime.now(ZoneOffset.UTC));
        return result;
    }

    public Map<String, Object> resumeAdmissions() {
        draining.set(false);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("replica_id", replicaId);
        result.put("draining", false);
        result.put("status", "READY");
        result.put("timestamp", OffsetDateTime.now(ZoneOffset.UTC));
        return result;
    }

    public Map<String, Object> checkReadiness() {
        boolean postgresUp = false;
        boolean redisUp = false;

        try {
            Integer one = jdbcTemplate.queryForObject("SELECT 1", Integer.class);
            postgresUp = (one != null && one == 1);
        } catch (Exception ignored) {
            postgresUp = false;
        }

        try {
            String pong = redisTemplate.getConnectionFactory() != null
                ? redisTemplate.getConnectionFactory().getConnection().ping()
                : null;
            redisUp = (pong != null && !pong.isBlank());
        } catch (Exception ignored) {
            redisUp = false;
        }

        boolean readyForTraffic = postgresUp && redisUp && !draining.get();
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("replica_id", replicaId);
        report.put("status", readyForTraffic ? "READY" : (draining.get() ? "DRAINING" : "DEGRADED"));
        report.put("accepting_admissions", readyForTraffic);
        report.put("draining", draining.get());
        report.put("dependencies", Map.of(
            "postgresql", postgresUp ? "UP" : "DOWN",
            "redis", redisUp ? "UP" : "DOWN"
        ));
        report.put("timestamp", OffsetDateTime.now(ZoneOffset.UTC));
        return report;
    }
}
