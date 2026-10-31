package com.scratchllm.platform.generation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.request;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.scratchllm.platform.infrastructure.RedisRateLimiterService;
import com.scratchllm.platform.infrastructure.RedisRateLimiterService.RateLimitResult;
import jakarta.servlet.http.Cookie;
import java.util.Map;
import java.util.UUID;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

/**
 * Day016, Day017, Day018 & Day020 integration tests:
 * - Idempotent run submission, replay while active/quota-exhausted, and conflict detection on changed body
 * - Atomic Redis Lua rate limiter enforcement
 * - Fake-provider SSE streaming, assistant message persistence, quota settlement, and run cancellation
 */
@SpringBootTest
@AutoConfigureMockMvc
class Phase2RunIdempotencyRateLimitAndSseIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Autowired
    private RedisRateLimiterService rateLimiterService;

    @Test
    void enforcesAtomicRedisLuaRateLimitWithBoundedTtl() {
        String customKey = "test-actor-" + UUID.randomUUID();
        RateLimitResult r1 = rateLimiterService.evaluateCustomLimit(customKey, 30, 2);
        RateLimitResult r2 = rateLimiterService.evaluateCustomLimit(customKey, 30, 2);
        RateLimitResult r3 = rateLimiterService.evaluateCustomLimit(customKey, 30, 2);

        assertThat(r1.allowed()).isTrue();
        assertThat(r1.currentCount()).isEqualTo(1L);
        assertThat(r2.allowed()).isTrue();
        assertThat(r2.currentCount()).isEqualTo(2L);
        assertThat(r3.allowed()).isFalse();
        assertThat(r3.currentCount()).isEqualTo(3L);
        assertThat(r3.ttlSeconds()).isBetween(1L, 30L);
    }

    @Test
    void submitsIdempotentRunStreamsSseEventsPersistsHistoryAndSupportsReplayAndCancellation() throws Exception {
        AuthClient user = registerLoginAndCreateConversation("runner-" + UUID.randomUUID() + "@example.com");

        String idemKey = "idem-run-" + UUID.randomUUID();
        Map<String, Object> runPayload = Map.of(
            "prompt", "Once upon a time, the little robot counted tokens.",
            "model_alias", "fake-deterministic",
            "sampling", Map.of(
                "max_new_tokens", 16,
                "temperature", 0.7,
                "top_k", 40,
                "top_p", 0.95,
                "seed", 42
            )
        );

        // 1. Submit initial run -> 202 Accepted
        MvcResult firstSubmit = mockMvc.perform(post("/api/v1/conversations/" + user.conversationId() + "/runs")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .header("X-CSRF-TOKEN", user.csrfToken())
                .header("Idempotency-Key", idemKey)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(runPayload)))
            .andExpect(status().isAccepted())
            .andExpect(jsonPath("$.model_revision").value("fake-deterministic-r1"))
            .andReturn();

        String runId = objectMapper.readTree(firstSubmit.getResponse().getContentAsString())
            .get("run_id").asText();

        // 2. Duplicate submission with the exact same Idempotency-Key and payload returns the exact same run_id
        //    EVEN IF the user's daily quota is now set to 0 (proving replay checks precede admission/quota checks)
        jdbcTemplate.update(
            "UPDATE usage_buckets SET daily_allowance_tokens = 0 WHERE user_id = ?",
            user.userId()
        );

        mockMvc.perform(post("/api/v1/conversations/" + user.conversationId() + "/runs")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .header("X-CSRF-TOKEN", user.csrfToken())
                .header("Idempotency-Key", idemKey)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(runPayload)))
            .andExpect(status().isAccepted())
            .andExpect(jsonPath("$.run_id").value(runId));

        // Restore allowance
        jdbcTemplate.update(
            "UPDATE usage_buckets SET daily_allowance_tokens = 4096 WHERE user_id = ?",
            user.userId()
        );

        // 3. Reusing the same Idempotency-Key with a modified prompt fails with 409 IDEMPOTENCY_KEY_CONFLICT
        Map<String, Object> modifiedPayload = Map.of(
            "prompt", "Changed prompt body with same idempotency key!",
            "model_alias", "fake-deterministic",
            "sampling", Map.of(
                "max_new_tokens", 16,
                "temperature", 0.7,
                "top_k", 40,
                "top_p", 0.95,
                "seed", 42
            )
        );
        mockMvc.perform(post("/api/v1/conversations/" + user.conversationId() + "/runs")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .header("X-CSRF-TOKEN", user.csrfToken())
                .header("Idempotency-Key", idemKey)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(modifiedPayload)))
            .andExpect(status().isConflict())
            .andExpect(jsonPath("$.error_code").value("IDEMPOTENCY_KEY_CONFLICT"));

        // 4. Stream SSE events from GET /api/v1/runs/{runId}/events until terminal event
        MvcResult sseAsync = mockMvc.perform(get("/api/v1/runs/" + runId + "/events")
                .cookie(user.sessionCookie())
                .accept(MediaType.TEXT_EVENT_STREAM))
            .andExpect(request().asyncStarted())
            .andReturn();

        MvcResult sseCompleted = mockMvc.perform(asyncDispatch(sseAsync))
            .andExpect(status().isOk())
            .andReturn();

        String sseBody = sseCompleted.getResponse().getContentAsString();
        assertThat(sseBody)
            .contains("event:run.queued")
            .contains("event:run.started")
            .contains("event:text.delta")
            .contains("event:run.completed");

        // 5. Verify run status is COMPLETED and assistant message is committed in message history
        mockMvc.perform(get("/api/v1/runs/" + runId).cookie(user.sessionCookie()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("COMPLETED"))
            .andExpect(jsonPath("$.finish_reason").value("stop"))
            .andExpect(jsonPath("$.output_text").isNotEmpty());

        mockMvc.perform(get("/api/v1/conversations/" + user.conversationId() + "/messages")
                .cookie(user.sessionCookie()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.items.length()").value(2))
            .andExpect(jsonPath("$.items[0].role").value("USER"))
            .andExpect(jsonPath("$.items[1].role").value("ASSISTANT"));

        // 6. Submit a second slow run and cancel it immediately via POST /api/v1/runs/{runId}/cancel
        Map<String, Object> slowPayload = Map.of(
            "prompt", "[SLOW_FAKE] Long story that we will cancel mid-flight.",
            "model_alias", "fake-deterministic",
            "sampling", Map.of(
                "max_new_tokens", 32,
                "temperature", 0.7,
                "top_k", 40,
                "top_p", 0.95,
                "seed", 99
            )
        );
        MvcResult slowSubmit = mockMvc.perform(post("/api/v1/conversations/" + user.conversationId() + "/runs")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .header("X-CSRF-TOKEN", user.csrfToken())
                .header("Idempotency-Key", "idem-cancel-" + UUID.randomUUID())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(slowPayload)))
            .andExpect(status().isAccepted())
            .andReturn();

        String slowRunId = objectMapper.readTree(slowSubmit.getResponse().getContentAsString())
            .get("run_id").asText();

        mockMvc.perform(post("/api/v1/runs/" + slowRunId + "/cancel")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .header("X-CSRF-TOKEN", user.csrfToken()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("CANCELLED"))
            .andExpect(jsonPath("$.finish_reason").value("cancelled"));
    }

    private AuthClient registerLoginAndCreateConversation(String email) throws Exception {
        MvcResult csrfRes = mockMvc.perform(get("/auth/csrf")).andExpect(status().isOk()).andReturn();
        String csrfToken = objectMapper.readTree(csrfRes.getResponse().getContentAsString()).get("token").asText();
        Cookie xsrfCookie = csrfRes.getResponse().getCookie("XSRF-TOKEN");

        MvcResult regRes = mockMvc.perform(post("/auth/register")
                .cookie(xsrfCookie)
                .header("X-CSRF-TOKEN", csrfToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "email", email,
                    "password", "StrongPass#2026!",
                    "workspace_name", "Run Test Workspace"
                ))))
            .andExpect(status().isCreated())
            .andReturn();

        JsonNode profile = objectMapper.readTree(regRes.getResponse().getContentAsString());
        UUID userId = UUID.fromString(profile.get("user_id").asText());
        UUID workspaceId = UUID.fromString(profile.get("workspaces").get(0).get("workspace_id").asText());

        MvcResult loginRes = mockMvc.perform(post("/auth/login")
                .cookie(xsrfCookie)
                .header("X-CSRF-TOKEN", csrfToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "email", email,
                    "password", "StrongPass#2026!"
                ))))
            .andExpect(status().isOk())
            .andReturn();

        Cookie sessionCookie = loginRes.getResponse().getCookie("SESSION");

        MvcResult convRes = mockMvc.perform(post("/api/v1/workspaces/" + workspaceId + "/conversations")
                .cookie(xsrfCookie, sessionCookie)
                .header("X-CSRF-TOKEN", csrfToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("title", "Run Test Chat"))))
            .andExpect(status().isCreated())
            .andReturn();

        UUID convId = UUID.fromString(
            objectMapper.readTree(convRes.getResponse().getContentAsString()).get("conversation_id").asText()
        );

        return new AuthClient(userId, workspaceId, convId, csrfToken, xsrfCookie, sessionCookie);
    }

    private record AuthClient(
        UUID userId,
        UUID workspaceId,
        UUID conversationId,
        String csrfToken,
        Cookie xsrfCookie,
        Cookie sessionCookie
    ) {}
}
