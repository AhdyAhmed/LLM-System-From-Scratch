package com.scratchllm.platform.ops;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.Cookie;
import java.util.Map;
import java.util.UUID;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

/**
 * Phase 12 (Day111–Day120, Gate G12) Integration Verification Suite:
 * - Day116 & Day118: End-to-end release manifest endpoint (`GET /api/v1/release/manifest`) verifying
 *   `release_version = v1.0.0`, `gate = G12`, `flyway_schema_version = V7`, active ready models,
 *   SSE `Last-Event-ID` reconnect ergonomics, and honest capability disclaimer.
 * - Day116 & Day120: Full end-to-end release candidate smoke flow (CSRF -> register -> login ->
 *   create conversation -> submit idempotent run -> stream SSE events -> reconnect with `Last-Event-ID`).
 */
@SpringBootTest
@AutoConfigureMockMvc
class Phase12ReleaseManifestDemoAndGateG12IntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private StringRedisTemplate redisTemplate;

    @BeforeEach
    void clearRateLimitKeysBefore() {
        clearRateLimits();
    }

    @AfterEach
    void clearRateLimitKeysAfter() {
        clearRateLimits();
    }

    private void clearRateLimits() {
        var rlKeys = redisTemplate.keys("scratchllm:ratelimit:*");
        if (rlKeys != null && !rlKeys.isEmpty()) {
            redisTemplate.delete(rlKeys);
        }
    }

    @Test
    void shouldExposeDay118AndDay120ReleaseManifestAndExecuteEndToEndReleaseCandidateDemo() throws Exception {
        AuthSession session = registerAndCreateSession("phase12-rc-" + UUID.randomUUID() + "@example.com");

        // 1. Verify GET /api/v1/release/manifest exposes v1.0.0 Gate G12 metadata and CLI/SSE ergonomics
        MvcResult manifestRes = mockMvc.perform(
                get("/api/v1/release/manifest")
                    .cookie(session.sessionCookie())
            )
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.release_version").value("v1.0.0"))
            .andExpect(jsonPath("$.gate").value("G12"))
            .andExpect(jsonPath("$.deployment_profile").value("local-private-demo"))
            .andExpect(jsonPath("$.flyway_schema_version").value("V7"))
            .andExpect(jsonPath("$.cli_and_sse_ergonomics.reconnect_header").value("Last-Event-ID"))
            .andExpect(jsonPath("$.cli_and_sse_ergonomics.idempotency_header").value("Idempotency-Key"))
            .andReturn();

        JsonNode manifestJson = objectMapper.readTree(manifestRes.getResponse().getContentAsString());
        assertThat(manifestJson.get("active_ready_models").size()).isGreaterThanOrEqualTo(2);
        assertThat(manifestJson.get("retention_policies_configured").asInt()).isGreaterThanOrEqualTo(5);
        assertThat(manifestJson.get("honest_capability_disclaimer").asText()).contains("ScratchGPT");

        // 2. Execute end-to-end release candidate demo flow: create conversation -> submit run -> SSE reconnect
        MvcResult convRes = mockMvc.perform(
                post("/api/v1/workspaces/" + session.workspaceId() + "/conversations")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(Map.of("title", "Gate G12 Release Candidate Demo")))
            )
            .andExpect(status().isCreated())
            .andReturn();
        String convId = objectMapper.readTree(convRes.getResponse().getContentAsString()).get("conversation_id").asText();

        MvcResult runRes = mockMvc.perform(
                post("/api/v1/conversations/" + convId + "/runs")
                    .cookie(session.sessionCookie(), session.csrfCookie())
                    .header("X-CSRF-TOKEN", session.csrfToken())
                    .header("Idempotency-Key", "idem-g12-rc-" + UUID.randomUUID())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(Map.of(
                        "model_alias", "fake-deterministic",
                        "prompt", "Gate G12 final release verification prompt.",
                        "sampling", Map.of("temperature", 0.0, "top_k", 1, "top_p", 1.0, "max_new_tokens", 10)
                    )))
            )
            .andExpect(status().isAccepted())
            .andReturn();
        String runId = objectMapper.readTree(runRes.getResponse().getContentAsString()).get("run_id").asText();

        // Stream SSE events with Last-Event-ID: 1 reconnect header
        mockMvc.perform(
                get("/api/v1/runs/" + runId + "/events")
                    .cookie(session.sessionCookie())
                    .header("Last-Event-ID", "1")
            )
            .andExpect(status().isOk());
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
                        "workspace_name", "Phase 12 Workspace"
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
        return new AuthSession(sessionCookie, csrfCookie, csrfToken, workspaceId);
    }

    private record AuthSession(Cookie sessionCookie, Cookie csrfCookie, String csrfToken, String workspaceId) {}
}
