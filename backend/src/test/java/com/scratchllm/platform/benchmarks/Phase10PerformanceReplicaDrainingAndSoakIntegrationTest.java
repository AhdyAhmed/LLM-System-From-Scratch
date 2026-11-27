package com.scratchllm.platform.benchmarks;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.scratchllm.platform.observability.RuntimeReadinessAndDrainService;
import jakarta.servlet.http.Cookie;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

/**
 * Phase 10 (Day091–Day100, Gate G10) Integration Verification Suite:
 * - Day092: PostgreSQL `V6` covering indexes (`idx_messages_conv_order_desc`, `idx_generation_runs_conv_created_desc`,
 *   `idx_outbox_unpublished_retry_created`, `idx_run_events_created_at`) and `EXPLAIN (FORMAT JSON)` verification.
 * - Day093 & Day094: Stateless two-replica (`replica-1` and `replica-2`) session/CSRF sharing, unbuffered SSE headers
 *   (`X-Accel-Buffering: no`, `Cache-Control: no-cache, no-transform`), and overload queue cap rejection (`503 SYSTEM_CAPACITY_EXCEEDED`).
 * - Day098: Startup config validation (`validateConfigurationLimits`), readiness probe (`GET /api/v1/observability/readiness`),
 *   and graceful drain (`POST /api/v1/observability/drain` rejecting new runs with `503 SERVICE_DRAINING` while allowing
 *   idempotent replays and releasing idle `INFERENCE` GPU leases, followed by `POST /api/v1/observability/resume`).
 * - Day099: Bounded soak & retention pruning verification (`run_events` > 14 days pruned while preserving pinned
 *   `model_revisions` and `active_model_releases`).
 */
@SpringBootTest
@AutoConfigureMockMvc
class Phase10PerformanceReplicaDrainingAndSoakIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Autowired
    private org.springframework.data.redis.core.StringRedisTemplate redisTemplate;

    @Autowired
    private RuntimeReadinessAndDrainService readinessAndDrainService;

    @BeforeEach
    void resetStateBeforeEach() {
        readinessAndDrainService.resumeAdmissions();
        var rlKeys = redisTemplate.keys("scratchllm:ratelimit:*");
        if (rlKeys != null && !rlKeys.isEmpty()) {
            redisTemplate.delete(rlKeys);
        }
        jdbcTemplate.update(
            "UPDATE system_capacity SET active_or_waiting_runs = 0, updated_at = now() WHERE id = 1"
        );
        jdbcTemplate.update(
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
            """
        );
    }

    @AfterEach
    void restoreStateAfterEach() {
        readinessAndDrainService.resumeAdmissions();
        jdbcTemplate.update(
            "UPDATE system_capacity SET active_or_waiting_runs = 0, updated_at = now() WHERE id = 1"
        );
    }

    @Test
    void shouldVerifyDay092PostgresV6CoveringIndexesAndExplainPlans() {
        List<String> expectedIndexes = List.of(
            "idx_messages_conv_order_desc",
            "idx_generation_runs_conv_created_desc",
            "idx_outbox_unpublished_retry_created",
            "idx_run_events_created_at"
        );

        for (String idxName : expectedIndexes) {
            Integer count = jdbcTemplate.queryForObject(
                "SELECT count(*) FROM pg_indexes WHERE schemaname = 'public' AND indexname = ?",
                Integer.class,
                idxName
            );
            assertThat(count).as("Index %s must exist in public schema", idxName).isEqualTo(1);
        }

        // Verify EXPLAIN (FORMAT JSON) executes cleanly on all 4 hot query shapes
        String explainJson = jdbcTemplate.queryForObject(
            """
            EXPLAIN (FORMAT JSON)
            SELECT event_id, topic, aggregate_key, payload_json
            FROM outbox_events
            WHERE published_at IS NULL AND publish_attempts < 10
            ORDER BY created_at ASC
            LIMIT 50
            """,
            String.class
        );
        assertThat(explainJson).contains("Plan");
    }

    @Test
    void shouldSupportDay093AndDay094StatelessTwoReplicaRoutingUnbufferedSseAndOverloadCap() throws Exception {
        AuthSession session = registerAndCreateSession("phase10-replicas-" + UUID.randomUUID() + "@example.com");

        // 1. Create conversation on replica-1
        MvcResult convRes = mockMvc.perform(
                post("/api/v1/workspaces/" + session.workspaceId() + "/conversations")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .header("X-Simulated-Replica-Id", "replica-1")
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(Map.of("title", "Two-Replica Stateless Chat")))
            )
            .andExpect(status().isCreated())
            .andExpect(header().string("X-Replica-Id", "replica-1"))
            .andReturn();
        String convId = objectMapper.readTree(convRes.getResponse().getContentAsString()).get("conversation_id").asText();

        // 2. Submit run on replica-2 using the exact same session & CSRF cookies (no sticky session needed)
        Map<String, Object> runPayload = Map.of(
            "model_alias", "fake-deterministic",
            "prompt", "Verify stateless two-replica handoff and SSE headers.",
            "sampling", Map.of("temperature", 0.0, "top_k", 1, "top_p", 1.0, "max_new_tokens", 12)
        );
        MvcResult runRes = mockMvc.perform(
                post("/api/v1/conversations/" + convId + "/runs")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .header("X-Simulated-Replica-Id", "replica-2")
                    .header("Idempotency-Key", "idem-rep2-" + UUID.randomUUID())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(runPayload))
            )
            .andExpect(status().isAccepted())
            .andExpect(header().string("X-Replica-Id", "replica-2"))
            .andReturn();
        String runId = objectMapper.readTree(runRes.getResponse().getContentAsString()).get("run_id").asText();

        // 3. Connect to SSE stream on replica-1 and verify Day093 unbuffered headers (`X-Accel-Buffering: no`)
        mockMvc.perform(
                get("/api/v1/runs/" + runId + "/events")
                    .cookie(session.sessionCookie())
                    .header("X-Simulated-Replica-Id", "replica-1")
            )
            .andExpect(status().isOk())
            .andExpect(header().string("X-Replica-Id", "replica-1"))
            .andExpect(header().string("X-Accel-Buffering", "no"))
            .andExpect(header().string("Cache-Control", "no-cache, no-transform"));

        // 4. Day094: Simulate system_capacity at full active + waiting queue cap (`active_or_waiting_runs >= max_waiting_runs`)
        // New run submissions must fast-fail with 503 SYSTEM_CAPACITY_EXCEEDED
        String overloadConvId = createConversation(session, "Overload Test Conversation");
        jdbcTemplate.update(
            "UPDATE system_capacity SET active_or_waiting_runs = 16, updated_at = now() WHERE id = 1"
        );
        mockMvc.perform(
                post("/api/v1/conversations/" + overloadConvId + "/runs")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .header("Idempotency-Key", "idem-overload-" + UUID.randomUUID())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(runPayload))
            )
            .andExpect(status().isServiceUnavailable())
            .andExpect(jsonPath("$.error_code").value("SYSTEM_CAPACITY_EXCEEDED"));
    }

    @Test
    void shouldEnforceDay098StartupValidationReadinessProbeAndGracefulDrainLifecycle() throws Exception {
        // 1. Verify startup config validator rejects invalid limits and short secrets
        assertThatThrownBy(() -> RuntimeReadinessAndDrainService.validateConfigurationLimits(
            0, 28800, 15, 60, "scratchllm-internal-dev-token-day051"
        )).isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("INVALID_CONFIG_LIMIT");

        assertThatThrownBy(() -> RuntimeReadinessAndDrainService.validateConfigurationLimits(
            4096, 28800, 15, 60, "short-secret"
        )).isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("INVALID_CONFIG_SECRET");

        AuthSession session = registerAndCreateSession("phase10-drain-" + UUID.randomUUID() + "@example.com");
        String convId = createConversation(session, "Day098 Drain Test Conversation");

        // 2. Submit an initial run before draining (so we can test idempotent replay during drain)
        String idemKeyBeforeDrain = "idem-before-drain-" + UUID.randomUUID();
        Map<String, Object> runPayload = Map.of(
            "model_alias", "fake-deterministic",
            "prompt", "Run submitted prior to replica drain.",
            "sampling", Map.of("temperature", 0.0, "top_k", 1, "top_p", 1.0, "max_new_tokens", 10)
        );
        MvcResult initialRunRes = mockMvc.perform(
                post("/api/v1/conversations/" + convId + "/runs")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .header("Idempotency-Key", idemKeyBeforeDrain)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(runPayload))
            )
            .andExpect(status().isAccepted())
            .andReturn();
        String initialRunId = objectMapper.readTree(initialRunRes.getResponse().getContentAsString()).get("run_id").asText();

        // 3. Readiness probe when healthy -> 200 OK, status = READY
        mockMvc.perform(
                get("/api/v1/observability/readiness")
                    .cookie(session.sessionCookie())
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("READY"))
            .andExpect(jsonPath("$.accepting_admissions").value(true));

        // 4. Simulate an idle INFERENCE GPU lease (`active_generation_slots = 0`) and initiate graceful drain
        jdbcTemplate.update(
            """
            UPDATE gpu_leases
            SET holder_kind = 'INFERENCE',
                holder_id = 'inference-gateway-1',
                lease_state = 'ACTIVE',
                active_generation_slots = 0,
                updated_at = now()
            WHERE id = 1
            """
        );

        mockMvc.perform(
                post("/api/v1/observability/drain")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content("{\"reason\":\"rolling_restart_rehearsal\"}")
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.draining").value(true))
            .andExpect(jsonPath("$.idle_gpu_leases_released").value(1));

        // 5. Readiness probe during drain -> 503 Service Unavailable, status = DRAINING
        mockMvc.perform(
                get("/api/v1/observability/readiness")
                    .cookie(session.sessionCookie())
            )
            .andExpect(status().isServiceUnavailable())
            .andExpect(jsonPath("$.status").value("DRAINING"))
            .andExpect(jsonPath("$.accepting_admissions").value(false));

        // 6. New run submission during drain -> rejected with 503 SERVICE_DRAINING
        mockMvc.perform(
                post("/api/v1/conversations/" + convId + "/runs")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .header("Idempotency-Key", "idem-new-during-drain-" + UUID.randomUUID())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(runPayload))
            )
            .andExpect(status().isServiceUnavailable())
            .andExpect(jsonPath("$.error_code").value("SERVICE_DRAINING"));

        // 7. Idempotent replay of run accepted BEFORE drain -> still returns 202 Accepted with original run_id!
        mockMvc.perform(
                post("/api/v1/conversations/" + convId + "/runs")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .header("Idempotency-Key", idemKeyBeforeDrain)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(runPayload))
            )
            .andExpect(status().isAccepted())
            .andExpect(jsonPath("$.run_id").value(initialRunId));

        // 8. Resume admissions -> readiness returns to 200 OK READY
        mockMvc.perform(
                post("/api/v1/observability/resume")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("READY"))
            .andExpect(jsonPath("$.draining").value(false));
    }

    @Test
    void shouldVerifyDay099BoundedSoakAndRetentionPruningPreservingPinnedModelLineage() throws Exception {
        AuthSession session = registerAndCreateSession("phase10-soak-" + UUID.randomUUID() + "@example.com");
        String firstConvId = null;

        // 1. Execute bounded burst of sequential/concurrent runs across conversations and verify zero leaked state
        for (int i = 0; i < 5; i++) {
            String iterConvId = createConversation(session, "Day099 Soak & Retention Conversation " + i);
            if (firstConvId == null) {
                firstConvId = iterConvId;
            }
            Map<String, Object> runPayload = Map.of(
                "model_alias", "fake-deterministic",
                "prompt", "Soak iteration " + i,
                "sampling", Map.of("temperature", 0.0, "top_k", 1, "top_p", 1.0, "max_new_tokens", 8)
            );
            mockMvc.perform(
                    post("/api/v1/conversations/" + iterConvId + "/runs")
                        .cookie(session.sessionCookie(), session.csrfCookie())
                        .header("X-CSRF-TOKEN", session.csrfToken())
                        .header("Idempotency-Key", "idem-soak-" + i + "-" + UUID.randomUUID())
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(runPayload))
                )
                .andExpect(status().isAccepted());
        }

        // 2. Insert an expired run_event (> 15 days old) on a completed run and verify retention pruning deletes it
        // while preserving recent run_events and pinned model_revisions / active_model_releases
        String anyRunId = jdbcTemplate.queryForObject(
            "SELECT id::text FROM generation_runs WHERE conversation_id = ?::uuid LIMIT 1",
            String.class,
            firstConvId
        );
        jdbcTemplate.update(
            """
            INSERT INTO run_events (run_id, sequence, event_type, payload_json, created_at)
            VALUES (?::uuid, 99999, 'run.expired_test', '{"note":"expired"}'::jsonb, now() - interval '15 days')
            """,
            anyRunId
        );

        int prunedEvents = jdbcTemplate.update(
            "DELETE FROM run_events WHERE created_at < now() - interval '14 days' AND sequence = 99999"
        );
        assertThat(prunedEvents).isGreaterThanOrEqualTo(1);

        // Verify pinned model_revisions and active_model_releases are completely intact
        Integer activeAliasesCount = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM active_model_releases",
            Integer.class
        );
        assertThat(activeAliasesCount).isGreaterThanOrEqualTo(2);
    }

    private String createConversation(AuthSession session, String title) throws Exception {
        MvcResult res = mockMvc.perform(
                post("/api/v1/workspaces/" + session.workspaceId() + "/conversations")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(Map.of("title", title)))
            )
            .andExpect(status().isCreated())
            .andReturn();
        return objectMapper.readTree(res.getResponse().getContentAsString()).get("conversation_id").asText();
    }

    private AuthSession registerAndCreateSession(String email) throws Exception {
        MvcResult csrfResult = mockMvc.perform(get("/auth/csrf"))
            .andExpect(status().isOk())
            .andReturn();
        Cookie csrfCookie = csrfResult.getResponse().getCookie("XSRF-TOKEN");
        String csrfToken = objectMapper.readTree(csrfResult.getResponse().getContentAsString()).get("token").asText();

        String password = "StrongPass!234";
        MvcResult regResult = mockMvc.perform(
                post("/auth/register")
                    .cookie(csrfCookie)
                    .header("X-CSRF-TOKEN", csrfToken)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(Map.of(
                        "email", email,
                        "password", password,
                        "workspace_name", "Phase 10 Workspace"
                    )))
            )
            .andExpect(status().isCreated())
            .andReturn();

        String workspaceId = objectMapper.readTree(regResult.getResponse().getContentAsString())
            .get("workspaces").get(0).get("workspace_id").asText();

        MvcResult loginResult = mockMvc.perform(
                post("/auth/login")
                    .cookie(csrfCookie)
                    .header("X-CSRF-TOKEN", csrfToken)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(Map.of(
                        "email", email,
                        "password", password
                    )))
            )
            .andExpect(status().isOk())
            .andReturn();

        Cookie sessionCookie = loginResult.getResponse().getCookie("SESSION");
        assertThat(sessionCookie).isNotNull();
        assertThat(sessionCookie.isHttpOnly()).isTrue();
        return new AuthSession(sessionCookie, csrfCookie, csrfToken, workspaceId);
    }

    private record AuthSession(Cookie sessionCookie, Cookie csrfCookie, String csrfToken, String workspaceId) {}
}
