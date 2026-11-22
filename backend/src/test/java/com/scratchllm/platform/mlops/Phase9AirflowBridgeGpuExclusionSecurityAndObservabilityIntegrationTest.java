package com.scratchllm.platform.mlops;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.header;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.Cookie;
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
 * Phase 9 (Day081–Day090, Gate G9) Integration Verification Suite:
 * - Day082 & Day085: Authenticated `/internal/training-jobs` bridge (`X-Internal-Service-Token`),
 *   stable `submission_key` (`dag_id:dag_run_id:task_id:map_index` ignoring `try_number`),
 *   `409 CONFIG_HASH_CONFLICT`, `400 UNSAFE_INPUT_REJECTED`, and retry recovery.
 * - Day083: Single-GPU mutual exclusion in `gpu_leases` (`primary_gpu`) blocking `scratch-tiny`
 *   inference with `503 GPU_BUSY_WITH_TRAINING` while allowing `fake-deterministic` runs, plus
 *   OS-verified `(native_pid, process_create_time_utc)` stale lease reclamation.
 * - Day087 & Day088: Security regression suite (`400 TOKEN_IN_URL_FORBIDDEN`, `403 CSRF_INVALID`)
 *   and `GET /api/v1/observability/metrics` bounded-cardinality telemetry + `X-Correlation-Id` propagation.
 */
@SpringBootTest
@AutoConfigureMockMvc
class Phase9AirflowBridgeGpuExclusionSecurityAndObservabilityIntegrationTest {

    private static final String INTERNAL_TOKEN = "scratchllm-internal-dev-token-day051";

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @BeforeEach
    void resetGpuLeasesAndTrainingJobs() {
        jdbcTemplate.update("DELETE FROM training_jobs");
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

    @Test
    void shouldEnforceInternalTokenStableSubmissionKeyAndConfigHashConflictOnBridge() throws Exception {
        Map<String, Object> jobPayload = Map.of(
            "dag_id", "scratchlm_offline_training_v1",
            "dag_run_id", "manual__2026-10-17T10-00-00",
            "task_id", "submit_and_poll_training",
            "map_index", -1,
            "try_number", 1,
            "config_id", "scratch-tiny-pretrain-v1",
            "config_hash", "a1b2c3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "dataset_manifest_sha256", "d1d2d3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "tokenizer_sha256", "f1f2f3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc"
        );

        // 1. Missing X-Internal-Service-Token -> 401 INTERNAL_AUTH_FAILED
        mockMvc.perform(
                post("/internal/training-jobs")
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(jobPayload))
            )
            .andExpect(status().isUnauthorized())
            .andExpect(jsonPath("$.error_code").value("INTERNAL_AUTH_FAILED"));

        // 2. Valid submission (try_number=1) succeeds and records stable submission_key
        MvcResult createRes = mockMvc.perform(
                post("/internal/training-jobs")
                    .header("X-Internal-Service-Token", INTERNAL_TOKEN)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(jobPayload))
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("COMPLETED"))
            .andExpect(jsonPath("$.reattached").value(false))
            .andExpect(jsonPath("$.attempt_count").value(1))
            .andExpect(
                jsonPath("$.submission_key")
                    .value("scratchlm_offline_training_v1:manual__2026-10-17T10-00-00:submit_and_poll_training:-1")
            )
            .andReturn();

        String jobId = objectMapper.readTree(createRes.getResponse().getContentAsString()).get("job_id").asText();

        // 3. Retry with try_number=2 and identical config_hash reattaches to the same job_id!
        Map<String, Object> retryPayload = Map.of(
            "dag_id", "scratchlm_offline_training_v1",
            "dag_run_id", "manual__2026-10-17T10-00-00",
            "task_id", "submit_and_poll_training",
            "map_index", -1,
            "try_number", 2,
            "config_id", "scratch-tiny-pretrain-v1",
            "config_hash", "a1b2c3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "dataset_manifest_sha256", "d1d2d3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "tokenizer_sha256", "f1f2f3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc"
        );
        mockMvc.perform(
                post("/internal/training-jobs")
                    .header("X-Internal-Service-Token", INTERNAL_TOKEN)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(retryPayload))
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.job_id").value(jobId))
            .andExpect(jsonPath("$.reattached").value(true))
            .andExpect(jsonPath("$.attempt_count").value(1));

        // 4. Conflicting config_hash on same submission_key -> 409 CONFIG_HASH_CONFLICT
        Map<String, Object> conflictPayload = Map.of(
            "dag_id", "scratchlm_offline_training_v1",
            "dag_run_id", "manual__2026-10-17T10-00-00",
            "task_id", "submit_and_poll_training",
            "map_index", -1,
            "try_number", 3,
            "config_id", "scratch-tiny-pretrain-v1",
            "config_hash", "ffffffffffff0000111122223333444455556666777788889999aaaabbbbcccc",
            "dataset_manifest_sha256", "d1d2d3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "tokenizer_sha256", "f1f2f3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc"
        );
        mockMvc.perform(
                post("/internal/training-jobs")
                    .header("X-Internal-Service-Token", INTERNAL_TOKEN)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(conflictPayload))
            )
            .andExpect(status().isConflict())
            .andExpect(jsonPath("$.error_code").value("CONFIG_HASH_CONFLICT"));

        // 5. Unsafe config_id or command injection / path traversal -> 400 UNSAFE_INPUT_REJECTED
        Map<String, Object> unsafePayload = Map.of(
            "dag_id", "scratchlm_offline_training_v1",
            "dag_run_id", "manual__2026-10-17T11-00-00",
            "task_id", "submit_and_poll_training",
            "map_index", -1,
            "try_number", 1,
            "config_id", "scratch-tiny-pretrain-v1",
            "config_hash", "a1b2c3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "dataset_manifest_sha256", "d1d2d3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "tokenizer_sha256", "f1f2f3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "resume_from_checkpoint", "../../Windows/System32/cmd.exe"
        );
        mockMvc.perform(
                post("/internal/training-jobs")
                    .header("X-Internal-Service-Token", INTERNAL_TOKEN)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(unsafePayload))
            )
            .andExpect(status().isBadRequest())
            .andExpect(jsonPath("$.error_code").value("UNSAFE_INPUT_REJECTED"));
    }

    @Test
    void shouldEnforceGpuMutualExclusionBetweenTrainingAndScratchModelWhileAllowingFakeDeterministic() throws Exception {
        AuthSession session = registerAndCreateSession("phase9-gpu-" + UUID.randomUUID() + "@example.com");

        // Ensure scratch-tiny-r1 model revision is READY in catalog for admission testing
        jdbcTemplate.update(
            "UPDATE model_revisions SET readiness_status = 'READY' WHERE revision_tag = 'scratch-tiny-r1'"
        );

        // 1. Start a RUNNING training job that actively holds the primary_gpu lease
        Map<String, Object> holdGpuPayload = Map.of(
            "dag_id", "scratchlm_offline_training_v1",
            "dag_run_id", "manual__2026-10-17T12-00-00",
            "task_id", "submit_and_poll_training",
            "map_index", -1,
            "try_number", 1,
            "config_id", "scratch-tiny-pretrain-v1",
            "config_hash", "b1b2c3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "dataset_manifest_sha256", "d1d2d3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "tokenizer_sha256", "f1f2f3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "hold_gpu_lease_running", true
        );

        MvcResult trainRes = mockMvc.perform(
                post("/internal/training-jobs")
                    .header("X-Internal-Service-Token", INTERNAL_TOKEN)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(holdGpuPayload))
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("RUNNING"))
            .andReturn();
        String activeJobId = objectMapper.readTree(trainRes.getResponse().getContentAsString()).get("job_id").asText();

        // 2. Submitting a second concurrent training job while job 1 is alive -> 409 GPU_EXCLUSION_CONFLICT
        Map<String, Object> secondTrainPayload = Map.of(
            "dag_id", "scratchlm_offline_training_v1",
            "dag_run_id", "manual__2026-10-17T12-05-00",
            "task_id", "submit_and_poll_training",
            "map_index", -1,
            "try_number", 1,
            "config_id", "scratch-tiny-pretrain-v1",
            "config_hash", "b1b2c3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "dataset_manifest_sha256", "d1d2d3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc",
            "tokenizer_sha256", "f1f2f3d4e5f60000111122223333444455556666777788889999aaaabbbbcccc"
        );
        mockMvc.perform(
                post("/internal/training-jobs")
                    .header("X-Internal-Service-Token", INTERNAL_TOKEN)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(secondTrainPayload))
            )
            .andExpect(status().isConflict())
            .andExpect(jsonPath("$.error_code").value("GPU_EXCLUSION_CONFLICT"));

        // 3. Submitting an online generation run for `scratch-tiny` (GPU model) -> 503 GPU_BUSY_WITH_TRAINING
        String convId = createConversation(session, "GPU Exclusion Test Conversation");
        Map<String, Object> scratchRunPayload = Map.of(
            "model_alias", "scratch-tiny",
            "prompt", "Explain how rotary or learned positional embeddings work.",
            "sampling", Map.of("temperature", 0.0, "top_k", 1, "top_p", 1.0, "max_new_tokens", 16)
        );
        mockMvc.perform(
                post("/api/v1/conversations/" + convId + "/runs")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .header("Idempotency-Key", "idem-gpu-busy-" + UUID.randomUUID())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(scratchRunPayload))
            )
            .andExpect(status().isServiceUnavailable())
            .andExpect(jsonPath("$.error_code").value("GPU_BUSY_WITH_TRAINING"));

        // 4. Submitting an online generation run for `fake-deterministic` (CPU model) succeeds uninterrupted!
        Map<String, Object> fakeRunPayload = Map.of(
            "model_alias", "fake-deterministic",
            "prompt", "Hello while training holds the GPU!",
            "sampling", Map.of("temperature", 0.0, "top_k", 1, "top_p", 1.0, "max_new_tokens", 16)
        );
        mockMvc.perform(
                post("/api/v1/conversations/" + convId + "/runs")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .header("Idempotency-Key", "idem-cpu-ok-" + UUID.randomUUID())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(fakeRunPayload))
            )
            .andExpect(status().isAccepted())
            .andExpect(jsonPath("$.status").value("QUEUED"));

        // 5. Cancel the active training job -> releases GPU lease immediately
        mockMvc.perform(
                post("/internal/training-jobs/" + activeJobId + "/cancel")
                    .header("X-Internal-Service-Token", INTERNAL_TOKEN)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content("{\"reason\":\"Preempted for interactive test\"}")
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("CANCELLED"));

        // 6. Simulate a stale GPU lease with a non-existent PID (e.g., 99999999) -> next training job reclaims it!
        jdbcTemplate.update(
            """
            UPDATE gpu_leases
            SET holder_kind = 'TRAINING',
                holder_id = ?,
                native_pid = 99999999,
                process_create_time_utc = 1700000000.0,
                lease_state = 'ACTIVE',
                updated_at = now()
            WHERE id = 1
            """,
            UUID.randomUUID().toString()
        );

        mockMvc.perform(
                post("/internal/training-jobs")
                    .header("X-Internal-Service-Token", INTERNAL_TOKEN)
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(secondTrainPayload))
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("COMPLETED"));
    }

    @Test
    void shouldEnforceDay087SecurityRegressionsAndDay088ObservabilityMetrics() throws Exception {
        AuthSession session = registerAndCreateSession("phase9-sec-obs-" + UUID.randomUUID() + "@example.com");

        // 1. Day087: Passing token in URL query parameters is strictly forbidden -> 400 TOKEN_IN_URL_FORBIDDEN
        mockMvc.perform(
                get("/api/v1/observability/metrics?token=secret-leak-in-url")
                    .cookie(session.sessionCookie())
            )
            .andExpect(status().isBadRequest())
            .andExpect(jsonPath("$.error_code").value("TOKEN_IN_URL_FORBIDDEN"));

        // 2. Day087: State-changing request without X-CSRF-TOKEN -> 403 CSRF_INVALID
        mockMvc.perform(
                post("/api/v1/workspaces/" + session.workspaceId() + "/conversations")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content("{\"title\":\"Missing CSRF\"}")
            )
            .andExpect(status().isForbidden())
            .andExpect(jsonPath("$.error_code").value("CSRF_INVALID"));

        // 3. Day088: End-to-end correlation ID & traceparent propagation + bounded-cardinality metrics
        String customCorrelationId = "corr-phase9-day088-test";
        String customTraceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

        MvcResult metricsRes = mockMvc.perform(
                get("/api/v1/observability/metrics")
                    .cookie(session.sessionCookie())
                    .header("X-Correlation-Id", customCorrelationId)
                    .header("traceparent", customTraceparent)
            )
            .andExpect(status().isOk())
            .andExpect(header().string("X-Correlation-Id", customCorrelationId))
            .andExpect(header().string("traceparent", customTraceparent))
            .andExpect(jsonPath("$.correlation_id").value(customCorrelationId))
            .andExpect(jsonPath("$.capacity.max_waiting_runs").value(8))
            .andExpect(jsonPath("$.logging_redaction_policy.prompts_logged").value(false))
            .andExpect(jsonPath("$.logging_redaction_policy.documents_logged").value(false))
            .andExpect(jsonPath("$.logging_redaction_policy.tokens_logged").value(false))
            .andExpect(jsonPath("$.gpu_lease.resource_name").value("primary_gpu"))
            .andReturn();

        JsonNode metricsJson = objectMapper.readTree(metricsRes.getResponse().getContentAsString());
        assertThat(metricsJson.has("runs_by_status")).isTrue();
        assertThat(metricsJson.has("training_jobs_by_status")).isTrue();
        assertThat(metricsJson.has("model_promotions_by_status")).isTrue();
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
                        "workspace_name", "Phase 9 Workspace"
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
