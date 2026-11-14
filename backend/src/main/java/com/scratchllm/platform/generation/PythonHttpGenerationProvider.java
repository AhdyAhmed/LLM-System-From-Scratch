package com.scratchllm.platform.generation;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

/**
 * Day054: Java HTTP model-provider adapter (`PythonHttpGenerationProvider`) communicating with
 * the internal Python FastAPI scratch-model worker (`/internal/v1/generate` and `/internal/v1/cancel/{run_id}`).
 * Enforces explicit connection, idle-stream, and total timeouts, `X-Internal-Service-Token` authentication,
 * payload bounds, and resilient NDJSON stream parsing (handling success, malformed NDJSON, stalled stream,
 * abrupt disconnect before terminal record, context overflow, and auth failure).
 */
@Component
public class PythonHttpGenerationProvider implements GenerationProvider {

    public static final String PROVIDER_ID = "python-http-scratch";
    public static final String INTERNAL_TOKEN_HEADER = "X-Internal-Service-Token";

    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;
    private final ExecutorService lineReadExecutor = Executors.newCachedThreadPool();

    private volatile String baseUrl;
    private volatile String internalToken;
    private volatile long connectTimeoutMs;
    private volatile long idleReadTimeoutMs;
    private volatile long totalTimeoutMs;
    private volatile int maxLineBytes;

    public PythonHttpGenerationProvider(
        ObjectMapper objectMapper,
        @Value("${scratchllm.inference.base-url:http://127.0.0.1:8001}") String baseUrl,
        @Value("${scratchllm.inference.internal-token:scratchllm-internal-dev-token-day051}") String internalToken,
        @Value("${scratchllm.inference.connect-timeout-ms:3000}") long connectTimeoutMs,
        @Value("${scratchllm.inference.idle-read-timeout-ms:5000}") long idleReadTimeoutMs,
        @Value("${scratchllm.inference.total-timeout-ms:30000}") long totalTimeoutMs,
        @Value("${scratchllm.inference.max-line-bytes:65536}") int maxLineBytes
    ) {
        this.objectMapper = objectMapper;
        this.baseUrl = trimTrailingSlash(baseUrl);
        this.internalToken = internalToken;
        this.connectTimeoutMs = Math.max(250L, connectTimeoutMs);
        this.idleReadTimeoutMs = Math.max(100L, idleReadTimeoutMs);
        this.totalTimeoutMs = Math.max(500L, totalTimeoutMs);
        this.maxLineBytes = Math.max(1024, maxLineBytes);
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofMillis(this.connectTimeoutMs))
            .build();
    }

    @PreDestroy
    public void shutdown() {
        lineReadExecutor.shutdownNow();
    }

    public void setBaseUrl(String baseUrl) {
        this.baseUrl = trimTrailingSlash(baseUrl);
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    public void setInternalToken(String internalToken) {
        this.internalToken = internalToken;
    }

    public String getInternalToken() {
        return internalToken;
    }

    public void setIdleReadTimeoutMs(long idleReadTimeoutMs) {
        this.idleReadTimeoutMs = Math.max(50L, idleReadTimeoutMs);
    }

    public long getIdleReadTimeoutMs() {
        return idleReadTimeoutMs;
    }

    @Override
    public String providerId() {
        return PROVIDER_ID;
    }

    @Override
    public boolean supportsModelRevision(String modelRevision) {
        if (modelRevision == null) {
            return false;
        }
        return modelRevision.startsWith("scratch-tiny") || modelRevision.startsWith("scratchlm-");
    }

    @Override
    public GenerationOutcome generateStream(
        GenerationCommand command,
        ChunkConsumer chunkConsumer,
        CancellationSignal cancellationSignal
    ) {
        int fallbackInputTokens = Math.max(1, (command.prompt().length() + 3) / 4);
        StringBuilder accumulated = new StringBuilder();
        int emittedOutputTokens = 0;

        try {
            String requestJson = buildRequestJson(command);
            long remainingDeadlineMs = Math.max(
                200L,
                Math.min(
                    totalTimeoutMs,
                    Duration.between(Instant.now(), command.deadline()).toMillis()
                )
            );

            HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/internal/v1/generate"))
                .timeout(Duration.ofMillis(remainingDeadlineMs))
                .header("Content-Type", "application/json")
                .header("Accept", "application/x-ndjson")
                .header(INTERNAL_TOKEN_HEADER, internalToken != null ? internalToken : "")
                .POST(HttpRequest.BodyPublishers.ofString(requestJson, StandardCharsets.UTF_8))
                .build();

            HttpResponse<InputStream> response = httpClient.send(
                httpRequest,
                HttpResponse.BodyHandlers.ofInputStream()
            );

            int status = response.statusCode();
            if (status == 401 || status == 403) {
                closeQuietly(response.body());
                return errorOutcome(
                    command,
                    accumulated.toString(),
                    fallbackInputTokens,
                    emittedOutputTokens,
                    "PROVIDER_AUTH_FAILED",
                    "Python inference service rejected X-Internal-Service-Token (" + status + ")."
                );
            }

            if (status != 200) {
                String errBody = readBoundedString(response.body(), 8192);
                String upstreamCode = extractErrorCode(errBody, status == 400 ? "CONTEXT_LIMIT_EXCEEDED" : "UPSTREAM_" + status);
                String upstreamMsg = extractErrorMessage(errBody, "Upstream Python provider returned HTTP " + status);
                return errorOutcome(
                    command,
                    accumulated.toString(),
                    fallbackInputTokens,
                    emittedOutputTokens,
                    upstreamCode,
                    upstreamMsg
                );
            }

            try (InputStream bodyStream = response.body();
                 BufferedReader reader = new BufferedReader(new InputStreamReader(bodyStream, StandardCharsets.UTF_8))) {

                while (true) {
                    if (cancellationSignal.isCancelled()) {
                        sendCancelSignalQuietly(command.runId().toString());
                        return new GenerationOutcome(
                            command.runId(),
                            command.attemptToken(),
                            command.modelRevision(),
                            FinishReason.CANCELLED,
                            accumulated.toString(),
                            fallbackInputTokens,
                            emittedOutputTokens,
                            "RUN_CANCELLED",
                            "Generation cancelled by caller signal."
                        );
                    }

                    if (Instant.now().isAfter(command.deadline())) {
                        sendCancelSignalQuietly(command.runId().toString());
                        return errorOutcome(
                            command,
                            accumulated.toString(),
                            fallbackInputTokens,
                            emittedOutputTokens,
                            "EXECUTION_DEADLINE_EXCEEDED",
                            "Generation exceeded its configured execution deadline."
                        );
                    }

                    String line = readLineWithIdleTimeout(reader, bodyStream, idleReadTimeoutMs);
                    if (line == null) {
                        // Stream reached EOF before a terminal "done" record was received
                        return errorOutcome(
                            command,
                            accumulated.toString(),
                            fallbackInputTokens,
                            emittedOutputTokens,
                            "STREAM_DISCONNECTED",
                            "Python NDJSON stream closed before emitting a terminal 'done' record."
                        );
                    }

                    String trimmed = line.trim();
                    if (trimmed.isEmpty()) {
                        continue;
                    }
                    if (trimmed.length() > maxLineBytes) {
                        return errorOutcome(
                            command,
                            accumulated.toString(),
                            fallbackInputTokens,
                            emittedOutputTokens,
                            "MALFORMED_PROVIDER_STREAM",
                            "NDJSON line exceeded maxLineBytes (" + maxLineBytes + ")."
                        );
                    }

                    final JsonNode node;
                    try {
                        node = objectMapper.readTree(trimmed);
                    } catch (Exception parseEx) {
                        return errorOutcome(
                            command,
                            accumulated.toString(),
                            fallbackInputTokens,
                            emittedOutputTokens,
                            "MALFORMED_PROVIDER_STREAM",
                            "Malformed JSON in provider NDJSON stream: " + parseEx.getMessage()
                        );
                    }

                    if (!node.isObject() || !node.hasNonNull("type")) {
                        return errorOutcome(
                            command,
                            accumulated.toString(),
                            fallbackInputTokens,
                            emittedOutputTokens,
                            "MALFORMED_PROVIDER_STREAM",
                            "NDJSON record missing required 'type' field."
                        );
                    }

                    String recordType = node.get("type").asText();
                    if ("delta".equals(recordType)) {
                        int chunkIdx = node.path("chunk_index").asInt(1);
                        String textDelta = node.path("text").asText("");
                        int emitted = node.path("emitted_tokens").asInt(1);
                        accumulated.append(textDelta);
                        emittedOutputTokens += Math.max(0, emitted);

                        chunkConsumer.onChunk(new GenerationDeltaChunk(
                            command.runId(),
                            command.attemptToken(),
                            chunkIdx,
                            textDelta,
                            emitted
                        ));
                    } else if ("done".equals(recordType)) {
                        String reasonStr = node.path("finish_reason").asText("stop");
                        FinishReason finishReason = mapFinishReason(reasonStr);
                        int inTok = node.path("input_tokens").asInt(fallbackInputTokens);
                        int outTok = node.path("output_tokens").asInt(emittedOutputTokens);
                        String errCode = node.hasNonNull("error_code") ? node.get("error_code").asText() : null;
                        String errMsg = node.hasNonNull("error_message") ? node.get("error_message").asText() : null;
                        String finalText = node.hasNonNull("output_text")
                            ? node.get("output_text").asText()
                            : accumulated.toString();

                        return new GenerationOutcome(
                            command.runId(),
                            command.attemptToken(),
                            command.modelRevision(),
                            finishReason,
                            finalText,
                            inTok,
                            outTok,
                            errCode,
                            errMsg
                        );
                    } else {
                        return errorOutcome(
                            command,
                            accumulated.toString(),
                            fallbackInputTokens,
                            emittedOutputTokens,
                            "MALFORMED_PROVIDER_STREAM",
                            "Unknown NDJSON record type: " + recordType
                        );
                    }
                }
            }
        } catch (IdleStreamTimeoutException idleEx) {
            sendCancelSignalQuietly(command.runId().toString());
            return errorOutcome(
                command,
                accumulated.toString(),
                fallbackInputTokens,
                emittedOutputTokens,
                "PROVIDER_IDLE_TIMEOUT",
                idleEx.getMessage()
            );
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
            sendCancelSignalQuietly(command.runId().toString());
            return new GenerationOutcome(
                command.runId(),
                command.attemptToken(),
                command.modelRevision(),
                FinishReason.CANCELLED,
                accumulated.toString(),
                fallbackInputTokens,
                emittedOutputTokens,
                "THREAD_INTERRUPTED",
                "Generation interrupted."
            );
        } catch (Exception ex) {
            return errorOutcome(
                command,
                accumulated.toString(),
                fallbackInputTokens,
                emittedOutputTokens,
                "PROVIDER_CONNECTION_ERROR",
                "Failed communicating with Python inference provider: " + ex.getMessage()
            );
        }
    }

    private String readLineWithIdleTimeout(
        BufferedReader reader,
        InputStream rawStream,
        long timeoutMs
    ) throws Exception {
        Future<String> future = lineReadExecutor.submit(reader::readLine);
        try {
            return future.get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException te) {
            future.cancel(true);
            closeQuietly(rawStream);
            throw new IdleStreamTimeoutException(
                "Python provider NDJSON stream stalled for more than " + timeoutMs + " ms."
            );
        } catch (ExecutionException ee) {
            Throwable cause = ee.getCause();
            if (cause instanceof Exception ex) {
                throw ex;
            }
            throw new RuntimeException(cause);
        }
    }

    public void sendCancelSignalQuietly(String runId) {
        try {
            HttpRequest cancelReq = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/internal/v1/cancel/" + runId))
                .timeout(Duration.ofMillis(1500))
                .header(INTERNAL_TOKEN_HEADER, internalToken != null ? internalToken : "")
                .POST(HttpRequest.BodyPublishers.noBody())
                .build();
            httpClient.sendAsync(cancelReq, HttpResponse.BodyHandlers.discarding());
        } catch (Exception ignored) {
            // Best-effort cancellation notification to Python worker
        }
    }

    private String buildRequestJson(GenerationCommand cmd) throws Exception {
        Map<String, Object> sampling = new LinkedHashMap<>();
        sampling.put("max_new_tokens", cmd.sampling().maxNewTokens());
        sampling.put("temperature", cmd.sampling().temperature());
        sampling.put("top_k", cmd.sampling().topK());
        sampling.put("top_p", cmd.sampling().topP());
        if (cmd.sampling().seed() != null) {
            sampling.put("seed", cmd.sampling().seed());
        }

        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("schema_version", 1);
        payload.put("run_id", cmd.runId().toString());
        payload.put("attempt_token", cmd.attemptToken());
        payload.put("model_revision", cmd.modelRevision());
        payload.put("prompt", cmd.prompt());
        payload.put("context_limit", cmd.contextLimit());
        payload.put("sampling", sampling);
        payload.put("deadline_epoch_ms", cmd.deadline().toEpochMilli());

        FaultInjectionSpec fault = cmd.faultInjection();
        if (fault != null && (fault.chunkDelayMs() > 0 || fault.failAfterChunks() != null || fault.errorCode() != null)) {
            Map<String, Object> faultMap = new LinkedHashMap<>();
            faultMap.put("chunk_delay_ms", fault.chunkDelayMs());
            if (fault.failAfterChunks() != null) {
                faultMap.put("fail_after_chunks", fault.failAfterChunks());
            }
            if (fault.errorCode() != null) {
                faultMap.put("error_code", fault.errorCode());
            }
            payload.put("fault_injection", faultMap);
        }
        return objectMapper.writeValueAsString(payload);
    }

    private String extractErrorCode(String body, String fallback) {
        if (body == null || body.isBlank()) {
            return fallback;
        }
        try {
            JsonNode node = objectMapper.readTree(body);
            if (node.hasNonNull("error_code")) {
                return node.get("error_code").asText();
            }
            if (node.hasNonNull("detail") && node.get("detail").hasNonNull("error_code")) {
                return node.get("detail").get("error_code").asText();
            }
        } catch (Exception ignored) {
        }
        return fallback;
    }

    private String extractErrorMessage(String body, String fallback) {
        if (body == null || body.isBlank()) {
            return fallback;
        }
        try {
            JsonNode node = objectMapper.readTree(body);
            if (node.hasNonNull("message")) {
                return node.get("message").asText();
            }
            if (node.hasNonNull("detail") && node.get("detail").hasNonNull("message")) {
                return node.get("detail").get("message").asText();
            }
        } catch (Exception ignored) {
        }
        return fallback;
    }

    private static FinishReason mapFinishReason(String raw) {
        if (raw == null) {
            return FinishReason.STOP;
        }
        return switch (raw.trim().toLowerCase()) {
            case "stop" -> FinishReason.STOP;
            case "length" -> FinishReason.LENGTH;
            case "cancelled" -> FinishReason.CANCELLED;
            default -> FinishReason.ERROR;
        };
    }

    private static GenerationOutcome errorOutcome(
        GenerationCommand cmd,
        String partialText,
        int inputTokens,
        int outputTokens,
        String errorCode,
        String errorMessage
    ) {
        return new GenerationOutcome(
            cmd.runId(),
            cmd.attemptToken(),
            cmd.modelRevision(),
            FinishReason.ERROR,
            partialText,
            inputTokens,
            outputTokens,
            errorCode,
            errorMessage
        );
    }

    private static String readBoundedString(InputStream in, int maxBytes) {
        if (in == null) {
            return "";
        }
        try (InputStream stream = in) {
            byte[] bytes = stream.readNBytes(maxBytes);
            return new String(bytes, StandardCharsets.UTF_8);
        } catch (Exception ex) {
            return "";
        }
    }

    private static void closeQuietly(InputStream in) {
        if (in != null) {
            try {
                in.close();
            } catch (Exception ignored) {
            }
        }
    }

    private static String trimTrailingSlash(String url) {
        if (url == null) {
            return "http://127.0.0.1:8001";
        }
        String clean = url.trim();
        while (clean.endsWith("/")) {
            clean = clean.substring(0, clean.length() - 1);
        }
        return clean;
    }

    private static final class IdleStreamTimeoutException extends Exception {
        IdleStreamTimeoutException(String message) {
            super(message);
        }
    }
}
