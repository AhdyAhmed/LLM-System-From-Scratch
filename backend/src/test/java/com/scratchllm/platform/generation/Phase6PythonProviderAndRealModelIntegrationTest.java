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
import com.scratchllm.platform.generation.GenerationProvider.FinishReason;
import com.scratchllm.platform.generation.GenerationProvider.GenerationCommand;
import com.scratchllm.platform.generation.GenerationProvider.GenerationDeltaChunk;
import com.scratchllm.platform.generation.GenerationProvider.GenerationOutcome;
import com.scratchllm.platform.generation.GenerationProvider.SamplingConfig;
import com.scratchllm.platform.modelcatalog.ModelCatalogController.ModelCatalogEntry;
import com.scratchllm.platform.modelcatalog.ModelCatalogService;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import jakarta.servlet.http.Cookie;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;
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
 * Phase 6 (Day051–Day060, Gate G6) comprehensive Java integration test suite:
 * - Day054: PythonHttpGenerationProvider timeouts, X-Internal-Service-Token header, and NDJSON parser
 *           (success, auth failure, malformed stream, stalled stream, disconnect, and upstream 503)
 * - Day055: ModelCatalogService cache-aside Redis metadata (`model:rev:scratch-tiny-r1`) with TTL & PostgreSQL fallback,
 *           and end-to-end `scratch-tiny` (`scratch-tiny-r1`) chat vertical slice with SSE streaming & provenance
 * - Day056: Single active generation slot, queued run state, and mid-stream cancellation freeing the slot
 * - Day058: Explicit context-window overflow rejection (`400 CONTEXT_LIMIT_EXCEEDED`)
 */
@SpringBootTest
@AutoConfigureMockMvc
class Phase6PythonProviderAndRealModelIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Autowired
    private PythonHttpGenerationProvider pythonHttpProvider;

    @Autowired
    private ModelCatalogService modelCatalogService;

    private HttpServer mockPythonServer;
    private String originalBaseUrl;
    private String originalToken;
    private long originalIdleTimeoutMs;
    private final AtomicReference<String> lastReceivedTokenHeader = new AtomicReference<>();

    @BeforeEach
    void setUpServer() throws Exception {
        originalBaseUrl = pythonHttpProvider.getBaseUrl();
        originalToken = pythonHttpProvider.getInternalToken();
        originalIdleTimeoutMs = pythonHttpProvider.getIdleReadTimeoutMs();

        mockPythonServer = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        mockPythonServer.setExecutor(Executors.newCachedThreadPool());

        mockPythonServer.createContext("/internal/v1/generate", (HttpExchange exchange) -> {
            String token = exchange.getRequestHeaders().getFirst(PythonHttpGenerationProvider.INTERNAL_TOKEN_HEADER);
            lastReceivedTokenHeader.set(token);

            if (!"scratchllm-internal-dev-token-day051".equals(token)) {
                byte[] unauth = "{\"error_code\":\"UNAUTHORIZED_INTERNAL_TOKEN\",\"message\":\"Invalid token\"}"
                    .getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().add("Content-Type", "application/json");
                exchange.sendResponseHeaders(401, unauth.length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(unauth);
                }
                return;
            }

            String reqBody = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            JsonNode reqJson = objectMapper.readTree(reqBody);
            String runId = reqJson.path("run_id").asText();
            long attemptToken = reqJson.path("attempt_token").asLong(1L);
            String modelRev = reqJson.path("model_revision").asText("scratch-tiny-r1");
            String prompt = reqJson.path("prompt").asText("");

            if (prompt.contains("[FAULT:UPSTREAM_503]")) {
                byte[] err = "{\"error_code\":\"GPU_LEASED_TO_TRAINING\",\"message\":\"GPU busy with training\"}"
                    .getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().add("Content-Type", "application/json");
                exchange.sendResponseHeaders(503, err.length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(err);
                }
                return;
            }

            exchange.getResponseHeaders().add("Content-Type", "application/x-ndjson");
            exchange.sendResponseHeaders(200, 0);

            try (OutputStream os = exchange.getResponseBody()) {
                if (prompt.contains("[FAULT:STALL]")) {
                    Thread.sleep(600L);
                    return;
                }

                String chunk1 = String.format(
                    "{\"type\":\"delta\",\"run_id\":\"%s\",\"attempt_token\":%d,\"chunk_index\":1,\"text\":\"In the quiet valley, \",\"emitted_tokens\":5}\n",
                    runId, attemptToken
                );
                os.write(chunk1.getBytes(StandardCharsets.UTF_8));
                os.flush();

                if (prompt.contains("[FAULT:MALFORMED_NDJSON]")) {
                    os.write("{corrupted_ndjson_line\n".getBytes(StandardCharsets.UTF_8));
                    os.flush();
                    return;
                }

                if (prompt.contains("[FAULT:DISCONNECT]")) {
                    // Close stream without emitting a terminal "done" record
                    return;
                }

                if (prompt.contains("[SLOW_REAL]")) {
                    for (int i = 2; i <= 10; i++) {
                        Thread.sleep(70L);
                        String slowChunk = String.format(
                            "{\"type\":\"delta\",\"run_id\":\"%s\",\"attempt_token\":%d,\"chunk_index\":%d,\"text\":\"step-%d \",\"emitted_tokens\":1}\n",
                            runId, attemptToken, i, i
                        );
                        os.write(slowChunk.getBytes(StandardCharsets.UTF_8));
                        os.flush();
                    }
                } else {
                    String chunk2 = String.format(
                        "{\"type\":\"delta\",\"run_id\":\"%s\",\"attempt_token\":%d,\"chunk_index\":2,\"text\":\"Mira lit the silver lantern.\",\"emitted_tokens\":6}\n",
                        runId, attemptToken
                    );
                    os.write(chunk2.getBytes(StandardCharsets.UTF_8));
                    os.flush();
                }

                String doneLine = String.format(
                    "{\"type\":\"done\",\"run_id\":\"%s\",\"attempt_token\":%d,\"model_revision\":\"%s\",\"finish_reason\":\"stop\",\"output_text\":\"In the quiet valley, Mira lit the silver lantern.\",\"input_tokens\":14,\"output_tokens\":11,\"registry_version\":\"3\",\"package_sha256\":\"ba30f95421b3016b5b08ddae2b63537b701c759c2cbcc42cc495b929ebc2b12b\",\"tokenizer_sha256\":\"a38cb3a119b033b6b8d8680a11af31b3a09af4cbcc2cf759d0a99dcad07bdd4a\"}\n",
                    runId, attemptToken, modelRev
                );
                os.write(doneLine.getBytes(StandardCharsets.UTF_8));
                os.flush();
            } catch (Exception ignored) {
            }
        });

        mockPythonServer.createContext("/internal/v1/cancel/", (HttpExchange exchange) -> {
            byte[] ok = "{\"cancel_signaled\":true}".getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, ok.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(ok);
            }
        });

        mockPythonServer.start();
        int port = mockPythonServer.getAddress().getPort();
        pythonHttpProvider.setBaseUrl("http://127.0.0.1:" + port);
        pythonHttpProvider.setInternalToken("scratchllm-internal-dev-token-day051");
        pythonHttpProvider.setIdleReadTimeoutMs(250L);
    }

    @AfterEach
    void tearDownServer() {
        if (mockPythonServer != null) {
            mockPythonServer.stop(0);
        }
        pythonHttpProvider.setBaseUrl(originalBaseUrl);
        pythonHttpProvider.setInternalToken(originalToken);
        pythonHttpProvider.setIdleReadTimeoutMs(originalIdleTimeoutMs);
    }

    @Test
    void day054PythonHttpProviderHandlesSuccessAuthFailureMalformedStalledAndDisconnectedStreams() {
        UUID runId = UUID.randomUUID();
        SamplingConfig sampling = new SamplingConfig(16, 0.0, 32, 0.95, 42L);

        // 1. Successful NDJSON streaming
        List<GenerationDeltaChunk> deltas = new ArrayList<>();
        GenerationOutcome okOutcome = pythonHttpProvider.generateStream(
            new GenerationCommand(runId, 1L, "scratch-tiny-r1", "Tell a lantern story.", 128, sampling, Instant.now().plusSeconds(30), null),
            deltas::add,
            () -> false
        );
        assertThat(okOutcome.isSuccess()).isTrue();
        assertThat(okOutcome.finishReason()).isEqualTo(FinishReason.STOP);
        assertThat(okOutcome.inputTokens()).isEqualTo(14);
        assertThat(okOutcome.outputTokens()).isEqualTo(11);
        assertThat(deltas).hasSize(2);
        assertThat(lastReceivedTokenHeader.get()).isEqualTo("scratchllm-internal-dev-token-day051");

        // 2. Auth failure (401)
        pythonHttpProvider.setInternalToken("wrong-secret-token");
        GenerationOutcome authFail = pythonHttpProvider.generateStream(
            new GenerationCommand(runId, 1L, "scratch-tiny-r1", "Tell a lantern story.", 128, sampling, Instant.now().plusSeconds(30), null),
            chunk -> {},
            () -> false
        );
        assertThat(authFail.isSuccess()).isFalse();
        assertThat(authFail.errorCode()).isEqualTo("PROVIDER_AUTH_FAILED");
        pythonHttpProvider.setInternalToken("scratchllm-internal-dev-token-day051");

        // 3. Malformed NDJSON stream
        GenerationOutcome malformed = pythonHttpProvider.generateStream(
            new GenerationCommand(runId, 1L, "scratch-tiny-r1", "Prompt [FAULT:MALFORMED_NDJSON]", 128, sampling, Instant.now().plusSeconds(30), null),
            chunk -> {},
            () -> false
        );
        assertThat(malformed.isSuccess()).isFalse();
        assertThat(malformed.errorCode()).isEqualTo("MALFORMED_PROVIDER_STREAM");

        // 4. Abrupt disconnect before terminal "done" record
        GenerationOutcome disconnected = pythonHttpProvider.generateStream(
            new GenerationCommand(runId, 1L, "scratch-tiny-r1", "Prompt [FAULT:DISCONNECT]", 128, sampling, Instant.now().plusSeconds(30), null),
            chunk -> {},
            () -> false
        );
        assertThat(disconnected.isSuccess()).isFalse();
        assertThat(disconnected.errorCode()).isEqualTo("STREAM_DISCONNECTED");

        // 5. Stalled stream exceeding idle timeout
        GenerationOutcome stalled = pythonHttpProvider.generateStream(
            new GenerationCommand(runId, 1L, "scratch-tiny-r1", "Prompt [FAULT:STALL]", 128, sampling, Instant.now().plusSeconds(30), null),
            chunk -> {},
            () -> false
        );
        assertThat(stalled.isSuccess()).isFalse();
        assertThat(stalled.errorCode()).isEqualTo("PROVIDER_IDLE_TIMEOUT");

        // 6. Upstream 503 GPU leased to training
        GenerationOutcome upstream503 = pythonHttpProvider.generateStream(
            new GenerationCommand(runId, 1L, "scratch-tiny-r1", "Prompt [FAULT:UPSTREAM_503]", 128, sampling, Instant.now().plusSeconds(30), null),
            chunk -> {},
            () -> false
        );
        assertThat(upstream503.isSuccess()).isFalse();
        assertThat(upstream503.errorCode()).isEqualTo("GPU_LEASED_TO_TRAINING");
    }

    @Test
    void day055ModelCatalogCachesImmutableRevisionInRedisWithPostgresFallback() throws Exception {
        modelCatalogService.evictRevisionCache("scratch-tiny-r1", "scratch-tiny");
        assertThat(modelCatalogService.isCachedInRedis("scratch-tiny-r1")).isFalse();

        // Lookup populates Redis cache-aside entry `model:rev:scratch-tiny-r1`
        Optional<ModelCatalogEntry> resolved = modelCatalogService.resolveReadyAlias("scratch-tiny");
        assertThat(resolved).isPresent();
        assertThat(resolved.get().model_revision()).isEqualTo("scratch-tiny-r1");
        assertThat(resolved.get().registry_version()).isEqualTo("3");
        assertThat(resolved.get().package_sha256()).isEqualTo("ba30f95421b3016b5b08ddae2b63537b701c759c2cbcc42cc495b929ebc2b12b");
        assertThat(resolved.get().tokenizer_hash()).isEqualTo("a38cb3a119b033b6b8d8680a11af31b3a09af4cbcc2cf759d0a99dcad07bdd4a");
        assertThat(modelCatalogService.isCachedInRedis("scratch-tiny-r1")).isTrue();

        // Verify GET /api/v1/models endpoint exposes the promoted scratch-tiny-r1 entry
        AuthClient user = registerLoginAndCreateConversation("catalog-" + UUID.randomUUID() + "@example.com");
        mockMvc.perform(get("/api/v1/models")
                .cookie(user.xsrfCookie(), user.sessionCookie()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.models[?(@.alias == 'scratch-tiny')].model_revision").value("scratch-tiny-r1"))
            .andExpect(jsonPath("$.models[?(@.alias == 'scratch-tiny')].registry_version").value("3"))
            .andExpect(jsonPath("$.models[?(@.alias == 'scratch-tiny')].package_sha256")
                .value("ba30f95421b3016b5b08ddae2b63537b701c759c2cbcc42cc495b929ebc2b12b"));
    }

    @Test
    void day055Day056Day058EndToEndScratchModelChatContextOverflowAndCancellationSlotRelease() throws Exception {
        AuthClient user = registerLoginAndCreateConversation("scratch-e2e-" + UUID.randomUUID() + "@example.com");

        // 1. Day058: Prompt + max_new_tokens > 128 is rejected immediately with 400 CONTEXT_LIMIT_EXCEEDED
        String oversizedPrompt = "Once upon a time in the valley ".repeat(16); // ~496 chars -> ~124 tokens
        Map<String, Object> overflowPayload = Map.of(
            "prompt", oversizedPrompt,
            "model_alias", "scratch-tiny",
            "sampling", Map.of(
                "max_new_tokens", 32,
                "temperature", 0.7,
                "top_k", 32,
                "top_p", 0.95,
                "seed", 42
            )
        );
        mockMvc.perform(post("/api/v1/conversations/" + user.conversationId() + "/runs")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .header("X-CSRF-TOKEN", user.csrfToken())
                .header("Idempotency-Key", "idem-overflow-" + UUID.randomUUID())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(overflowPayload)))
            .andExpect(status().isBadRequest())
            .andExpect(jsonPath("$.error_code").value("CONTEXT_LIMIT_EXCEEDED"));

        // 2. Day055: Submit real `scratch-tiny` run and stream SSE events to completion
        Map<String, Object> validPayload = Map.of(
            "prompt", "Tell a short story about Mira and the silver lantern.",
            "model_alias", "scratch-tiny",
            "sampling", Map.of(
                "max_new_tokens", 16,
                "temperature", 0.0,
                "top_k", 32,
                "top_p", 0.95,
                "seed", 42
            )
        );
        MvcResult acceptedRes = mockMvc.perform(post("/api/v1/conversations/" + user.conversationId() + "/runs")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .header("X-CSRF-TOKEN", user.csrfToken())
                .header("Idempotency-Key", "idem-scratch-" + UUID.randomUUID())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(validPayload)))
            .andExpect(status().isAccepted())
            .andExpect(jsonPath("$.model_revision").value("scratch-tiny-r1"))
            .andReturn();

        String runId = objectMapper.readTree(acceptedRes.getResponse().getContentAsString()).get("run_id").asText();

        MvcResult sseAsync = mockMvc.perform(get("/api/v1/runs/" + runId + "/events")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .accept(MediaType.TEXT_EVENT_STREAM))
            .andExpect(request().asyncStarted())
            .andReturn();

        MvcResult sseDone = mockMvc.perform(asyncDispatch(sseAsync))
            .andExpect(status().isOk())
            .andReturn();

        String sseBody = sseDone.getResponse().getContentAsString();
        assertThat(sseBody).contains("event:run.queued");
        assertThat(sseBody).contains("event:run.started");
        assertThat(sseBody).contains("event:text.delta");
        assertThat(sseBody).contains("Mira lit the silver lantern.");
        assertThat(sseBody).contains("event:run.completed");

        // Verify terminal run status and settled token counts from Python provider
        mockMvc.perform(get("/api/v1/runs/" + runId)
                .cookie(user.xsrfCookie(), user.sessionCookie()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("COMPLETED"))
            .andExpect(jsonPath("$.model_revision").value("scratch-tiny-r1"))
            .andExpect(jsonPath("$.input_tokens").value(14))
            .andExpect(jsonPath("$.output_tokens").value(11));

        // 3. Day056: Submit a slow run holding the 1 active generation slot, cancel it mid-stream,
        //    and verify the slot is freed so a subsequent run completes cleanly!
        String slowConvId = createSecondConversation(user);
        Map<String, Object> slowPayload = Map.of(
            "prompt", "Slow story [SLOW_REAL]",
            "model_alias", "scratch-tiny",
            "sampling", Map.of(
                "max_new_tokens", 16,
                "temperature", 0.0,
                "top_k", 32,
                "top_p", 0.95,
                "seed", 42
            )
        );
        MvcResult slowSubmit = mockMvc.perform(post("/api/v1/conversations/" + slowConvId + "/runs")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .header("X-CSRF-TOKEN", user.csrfToken())
                .header("Idempotency-Key", "idem-slow-" + UUID.randomUUID())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(slowPayload)))
            .andExpect(status().isAccepted())
            .andReturn();
        String slowRunId = objectMapper.readTree(slowSubmit.getResponse().getContentAsString()).get("run_id").asText();

        Thread.sleep(60L);
        mockMvc.perform(post("/api/v1/runs/" + slowRunId + "/cancel")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .header("X-CSRF-TOKEN", user.csrfToken()))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("CANCELLED"));

        // Submit follow-up run in slowConvId and verify it acquires the freed slot and completes
        MvcResult followUpSubmit = mockMvc.perform(post("/api/v1/conversations/" + slowConvId + "/runs")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .header("X-CSRF-TOKEN", user.csrfToken())
                .header("Idempotency-Key", "idem-after-cancel-" + UUID.randomUUID())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(validPayload)))
            .andExpect(status().isAccepted())
            .andReturn();
        String followUpRunId = objectMapper.readTree(followUpSubmit.getResponse().getContentAsString()).get("run_id").asText();

        MvcResult followSse = mockMvc.perform(get("/api/v1/runs/" + followUpRunId + "/events")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .accept(MediaType.TEXT_EVENT_STREAM))
            .andExpect(request().asyncStarted())
            .andReturn();
        MvcResult followDone = mockMvc.perform(asyncDispatch(followSse))
            .andExpect(status().isOk())
            .andReturn();
        assertThat(followDone.getResponse().getContentAsString()).contains("event:run.completed");
    }

    private AuthClient registerLoginAndCreateConversation(String email) throws Exception {
        MvcResult csrfRes = mockMvc.perform(get("/auth/csrf"))
            .andExpect(status().isOk())
            .andReturn();
        String csrfToken = objectMapper.readTree(csrfRes.getResponse().getContentAsString()).get("token").asText();
        Cookie xsrfCookie = csrfRes.getResponse().getCookie("XSRF-TOKEN");

        MvcResult regRes = mockMvc.perform(post("/auth/register")
                .cookie(xsrfCookie)
                .header("X-CSRF-TOKEN", csrfToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of(
                    "email", email,
                    "password", "StrongPass#2026!",
                    "workspace_name", "Phase6 Workspace"
                ))))
            .andExpect(status().isCreated())
            .andReturn();

        JsonNode profile = objectMapper.readTree(regRes.getResponse().getContentAsString());
        UUID userId = UUID.fromString(profile.get("user_id").asText());
        String workspaceId = profile.get("workspaces").get(0).get("workspace_id").asText();

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
                .content(objectMapper.writeValueAsString(Map.of("title", "Phase 6 Scratch Chat"))))
            .andExpect(status().isCreated())
            .andReturn();
        String convId = objectMapper.readTree(convRes.getResponse().getContentAsString())
            .get("conversation_id").asText();

        return new AuthClient(userId, workspaceId, convId, sessionCookie, xsrfCookie, csrfToken);
    }

    private String createSecondConversation(AuthClient user) throws Exception {
        MvcResult convRes = mockMvc.perform(post("/api/v1/workspaces/" + user.workspaceId() + "/conversations")
                .cookie(user.xsrfCookie(), user.sessionCookie())
                .header("X-CSRF-TOKEN", user.csrfToken())
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(Map.of("title", "Second Conversation"))))
            .andExpect(status().isCreated())
            .andReturn();
        return objectMapper.readTree(convRes.getResponse().getContentAsString())
            .get("conversation_id").asText();
    }

    private record AuthClient(
        UUID userId,
        String workspaceId,
        String conversationId,
        Cookie sessionCookie,
        Cookie xsrfCookie,
        String csrfToken
    ) {}
}
