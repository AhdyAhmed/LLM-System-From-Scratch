package com.scratchllm.platform.ops;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.Cookie;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;
import java.util.UUID;
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
 * Phase 11 (Day101–Day110, Gate G11) Integration Verification Suite:
 * - Day102: Reverse-proxy TLS/HTTPS forwarded-header & cookie/SSE hardening (`GET /api/v1/ops/proxy-rehearsal`).
 * - Day103 & Day104: Multi-database & artifact backup inventory (`POST /api/v1/ops/backups`) and isolated
 *   restore drill (`POST /api/v1/ops/backups/{backupId}/restore-drill`) verifying session invalidation,
 *   model cache rebuild, deletion tombstone re-application, `consumer_inbox` replay deduplication, and
 *   `gpu_leases` (`id = 1`) reset so two GPU owners never resume simultaneously.
 * - Day105: Expand/contract migration (`V7`) & monotonic `release_generation` model rollback preserving
 *   pinned `generation_runs.model_revision` on in-flight runs and ignoring stale `model.released.v1` events.
 * - Day106: Retention & deduplication horizon enforcement (`336h > 168h`) and rejection of expired manual
 *   replays (`409 REPLAY_BEYOND_DEDUPE_HORIZON`).
 * - Day107: Single-service dependency outage and disk-pressure drills (`POST /api/v1/ops/drills/execute`).
 */
@SpringBootTest
@AutoConfigureMockMvc
class Phase11DeploymentBackupRestoreRollbackAndOutageIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Autowired
    private org.springframework.data.redis.core.StringRedisTemplate redisTemplate;

    @BeforeEach
    void resetGpuLeasesAndRateLimits() {
        clearRateLimitKeys();
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

    @org.junit.jupiter.api.AfterEach
    void clearRateLimitKeys() {
        var rlKeys = redisTemplate.keys("scratchllm:ratelimit:*");
        if (rlKeys != null && !rlKeys.isEmpty()) {
            redisTemplate.delete(rlKeys);
        }
    }

    @Test
    void shouldVerifyDay102ReverseProxyTlsForwardedHeadersAndSecurityHeaders() throws Exception {
        AuthSession session = registerAndCreateSession("phase11-proxy-" + UUID.randomUUID() + "@example.com");

        mockMvc.perform(
                get("/api/v1/ops/proxy-rehearsal")
                    .cookie(session.sessionCookie())
                    .header("X-Forwarded-Proto", "https")
                    .header("X-Forwarded-For", "127.0.0.1")
                    .header("X-Forwarded-Port", "8443")
            )
            .andExpect(status().isOk())
            .andExpect(header().string("X-Content-Type-Options", "nosniff"))
            .andExpect(header().string("X-Frame-Options", "DENY"))
            .andExpect(header().string("Referrer-Policy", "no-referrer"))
            .andExpect(jsonPath("$.deployment_profile").value("local-private-demo"))
            .andExpect(jsonPath("$.forwarded_proto").value("https"))
            .andExpect(jsonPath("$.forwarded_port").value("8443"))
            .andExpect(jsonPath("$.tls_terminated_at_proxy").value(true))
            .andExpect(jsonPath("$.sse_buffering_disabled").value(true));
    }

    @Test
    void shouldExecuteDay103BackupSnapshotAndDay104IsolatedRestoreDrillSafely() throws Exception {
        AuthSession session = registerAndCreateSession("phase11-backup-" + UUID.randomUUID() + "@example.com");

        // 1. Create a conversation and soft-delete it so we can verify tombstone preservation across backup/restore
        String deletedConvId = createConversation(session, "Conversation To Tombstone Before Backup");
        jdbcTemplate.update(
            "UPDATE conversations SET deleted_at = clock_timestamp() WHERE id = ?::uuid",
            deletedConvId
        );

        // Simulate an active GPU lease prior to restore drill to verify restore resets GPU ownership cleanly
        jdbcTemplate.update(
            """
            UPDATE gpu_leases
            SET holder_kind = 'INFERENCE',
                holder_id = 'stale-restored-holder',
                lease_state = 'ACTIVE',
                updated_at = now()
            WHERE id = 1
            """
        );

        // 2. Day103: Create multi-database & artifact backup inventory
        String backupLabel = "day103-it-" + UUID.randomUUID();
        MvcResult backupRes = mockMvc.perform(
                post("/api/v1/ops/backups")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(Map.of("backup_label", backupLabel)))
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.backup_label").value(backupLabel))
            .andExpect(jsonPath("$.status").value("COMPLETED"))
            .andExpect(jsonPath("$.inventory.disposable_excluded_stores.redis_sessions").value("INVALIDATE_ON_RESTORE"))
            .andReturn();

        JsonNode backupJson = objectMapper.readTree(backupRes.getResponse().getContentAsString());
        String backupId = backupJson.get("backup_id").asText();
        assertThat(backupJson.get("manifest_sha256").asText()).hasSize(64);

        // 3. Day104: Execute isolated restore drill (note: this invalidates Redis sessions as required by Day104!)
        MvcResult drillRes = mockMvc.perform(
                post("/api/v1/ops/backups/" + backupId + "/restore-drill")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.backup_id").value(backupId))
            .andExpect(jsonPath("$.status").value("PASSED"))
            .andExpect(jsonPath("$.gpu_lease_reset_cleanly").value(true))
            .andExpect(jsonPath("$.deduplicated_outbox_replays").value(1))
            .andReturn();

        JsonNode drillJson = objectMapper.readTree(drillRes.getResponse().getContentAsString());
        assertThat(drillJson.get("rebuilt_ready_models_count").asInt()).isGreaterThanOrEqualTo(2);
        assertThat(drillJson.get("reapplied_tombstones_count").asInt()).isGreaterThanOrEqualTo(1);

        // Verify singleton gpu_leases (id = 1) was reset to RELEASED so two GPU owners never resume
        String leaseStateAfterRestore = jdbcTemplate.queryForObject(
            "SELECT lease_state FROM gpu_leases WHERE id = 1",
            String.class
        );
        assertThat(leaseStateAfterRestore).isEqualTo("RELEASED");
    }

    @Test
    void shouldExecuteDay105MonotonicModelRollbackPreservingPinnedRunRevisionAndRejectingStaleEvents() throws Exception {
        AuthSession session = registerAndCreateSession("phase11-rollback-" + UUID.randomUUID() + "@example.com");

        // Ensure two distinct immutable revisions exist for `fake-deterministic` so we can promote -> submit run -> rollback
        String revV1 = "fake-deterministic-r1";
        String revV2 = "fake-deterministic-r2-canary";
        jdbcTemplate.update(
            """
            INSERT INTO model_revisions (
                id, model_alias, revision_tag, mlflow_model_name, mlflow_model_version, mlflow_run_id,
                artifact_checksum, tokenizer_hash, context_limit, max_output_tokens,
                is_scratch_model, readiness_status, capability_note
            ) VALUES (
                ?, 'fake-deterministic', ?, 'fake-deterministic', '2', 'run-v2-canary',
                'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
                128, 64, FALSE, 'READY', 'Phase 11 canary revision for rollback drill'
            )
            ON CONFLICT (revision_tag) DO UPDATE SET readiness_status = 'READY'
            """,
            UUID.nameUUIDFromBytes(revV2.getBytes( java.nio.charset.StandardCharsets.UTF_8)),
            revV2
        );

        // 1. Promote `fake-deterministic` to `fake-deterministic-r2-canary`
        MvcResult promoteRes = mockMvc.perform(
                post("/api/v1/models/promotions")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(Map.of(
                        "model_alias", "fake-deterministic",
                        "target_revision_tag", revV2,
                        "approved_by", "phase11-operator"
                    )))
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.target_revision_tag").value(revV2))
            .andReturn();
        long promotedGen = objectMapper.readTree(promoteRes.getResponse().getContentAsString())
            .get("release_generation").asLong();

        // 2. Submit a generation run while `fake-deterministic` points to `fake-deterministic-r2-canary`
        String convId = createConversation(session, "Rollback Pinned Revision Conversation");
        MvcResult runBeforeRollback = mockMvc.perform(
                post("/api/v1/conversations/" + convId + "/runs")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .header("Idempotency-Key", "idem-rb-" + UUID.randomUUID())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(Map.of(
                        "model_alias", "fake-deterministic",
                        "prompt", "Verify pinned revision before rollback",
                        "sampling", Map.of("temperature", 0.0, "top_k", 1, "top_p", 1.0, "max_new_tokens", 8)
                    )))
            )
            .andExpect(status().isAccepted())
            .andExpect(jsonPath("$.model_revision").value(revV2))
            .andReturn();
        String pinnedRunId = objectMapper.readTree(runBeforeRollback.getResponse().getContentAsString())
            .get("run_id").asText();

        // 3. Roll back `fake-deterministic` to prior immutable revision `fake-deterministic-r1` using a NEW higher release_generation
        MvcResult rollbackRes = mockMvc.perform(
                post("/api/v1/models/promotions")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(Map.of(
                        "model_alias", "fake-deterministic",
                        "target_revision_tag", revV1,
                        "approved_by", "phase11-rollback-operator"
                    )))
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.target_revision_tag").value(revV1))
            .andExpect(jsonPath("$.previous_revision_tag").value(revV2))
            .andReturn();
        long rollbackGen = objectMapper.readTree(rollbackRes.getResponse().getContentAsString())
            .get("release_generation").asLong();
        assertThat(rollbackGen).isEqualTo(promotedGen + 1L);

        // 4. Attempting a promotion with a stale release_generation <= rollbackGen is rejected (409 STALE_RELEASE_GENERATION)
        mockMvc.perform(
                post("/api/v1/models/promotions")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(Map.of(
                        "model_alias", "fake-deterministic",
                        "target_revision_tag", revV2,
                        "release_generation", promotedGen
                    )))
            )
            .andExpect(status().isConflict())
            .andExpect(jsonPath("$.error_code").value("STALE_RELEASE_GENERATION"));

        // 5. Verify existing run submitted before rollback STILL retains its pinned `fake-deterministic-r2-canary` revision!
        String storedRunRevision = jdbcTemplate.queryForObject(
            "SELECT model_revision FROM generation_runs WHERE id = ?::uuid",
            String.class,
            pinnedRunId
        );
        assertThat(storedRunRevision).isEqualTo(revV2);
    }

    @Test
    void shouldEnforceDay106RetentionAndDedupeHorizonAndExecuteDay107OutageDrills() throws Exception {
        AuthSession session = registerAndCreateSession("phase11-ops-" + UUID.randomUUID() + "@example.com");

        // 1. Day106: Normal retention run verifies consumer_inbox dedupe (336h) > outbox replay horizon (168h)
        mockMvc.perform(
                post("/api/v1/ops/retention/run")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content("{\"requested_replay_age_hours\":72}")
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.consumer_inbox_dedupe_horizon_hours").value(336))
            .andExpect(jsonPath("$.outbox_replay_horizon_hours").value(168))
            .andExpect(jsonPath("$.dedupe_exceeds_replay_horizon").value(true))
            .andExpect(jsonPath("$.status").value("ENFORCED"));

        // 2. Day106: Manual replay request older than 168h is rejected with 409 REPLAY_BEYOND_DEDUPE_HORIZON
        mockMvc.perform(
                post("/api/v1/ops/retention/run")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content("{\"requested_replay_age_hours\":240}")
            )
            .andExpect(status().isConflict())
            .andExpect(jsonPath("$.error_code").value("REPLAY_BEYOND_DEDUPE_HORIZON"));

        // 3. Day107: Execute all 5 single-service dependency outage and disk-pressure drills
        List<String> drills = List.of(
            "DISK_PRESSURE",
            "REDIS_CACHE_FALLBACK",
            "KAFKA_OUTBOX_BACKPRESSURE",
            "MLFLOW_PROMOTION_BLOCKED",
            "AIRFLOW_WORKER_RECOVERY"
        );
        for (String drillKind : drills) {
            mockMvc.perform(
                    post("/api/v1/ops/drills/execute")
                        .cookie(session.sessionCookie(), session.csrfCookie())
                        .header("X-CSRF-TOKEN", session.csrfToken())
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(Map.of("drill_kind", drillKind)))
                )
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.drill_kind").value(drillKind))
                .andExpect(jsonPath("$.passed").value(true));
        }

        Integer recordedDrillsCount = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM ops_outage_drills WHERE passed = TRUE",
            Integer.class
        );
        assertThat(recordedDrillsCount).isGreaterThanOrEqualTo(5);
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
                        "workspace_name", "Phase 11 Workspace"
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
