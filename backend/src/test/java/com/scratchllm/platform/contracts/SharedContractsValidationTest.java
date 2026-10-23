package com.scratchllm.platform.contracts;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Day005 contract validation test ensuring Java and Python share identical JSON/YAML contract fixtures.
 */
class SharedContractsValidationTest {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final Path contractsDir = Path.of("..", "contracts");

    @Test
    void openApiAndJsonSchemasExistAndContainRequiredEndpoints() throws Exception {
        Path openApiPath = contractsDir.resolve("openapi.yaml");
        assertThat(openApiPath).exists();
        String openApiText = Files.readString(openApiPath);
        assertThat(openApiText).contains(
            "/auth/csrf:",
            "/auth/login:",
            "/api/v1/workspaces/{workspaceId}/conversations:",
            "/api/v1/conversations/{conversationId}/runs:",
            "/api/v1/runs/{runId}/events:",
            "/api/v1/runs/{runId}/cancel:",
            "/api/v1/models:",
            "/api/v1/workspaces/{workspaceId}/search:"
        );

        JsonNode inferenceSchema = objectMapper.readTree(
            Files.readString(contractsDir.resolve("internal-inference.schema.json"))
        );
        assertThat(inferenceSchema.get("title").asText()).isEqualTo("InternalJavaToPythonInferenceContract");

        JsonNode sseSchema = objectMapper.readTree(
            Files.readString(contractsDir.resolve("sse-events.schema.json"))
        );
        assertThat(sseSchema.get("title").asText()).isEqualTo("DurableSseEventEnvelope");
    }

    @Test
    void sharedJsonFixturesParseAndConformToContractInvariants() throws Exception {
        Path fixturesDir = contractsDir.resolve("fixtures");

        JsonNode runRequest = objectMapper.readTree(
            Files.readString(fixturesDir.resolve("run-submission-request.json"))
        );
        assertThat(runRequest.get("model_alias").asText()).isEqualTo("scratch-tiny");
        assertThat(runRequest.get("sampling").get("max_new_tokens").asInt()).isBetween(1, 64);

        JsonNode internalRequest = objectMapper.readTree(
            Files.readString(fixturesDir.resolve("internal-generate-request.json"))
        );
        assertThat(internalRequest.get("schema_version").asInt()).isEqualTo(1);
        assertThat(internalRequest.get("attempt_token").asLong()).isGreaterThanOrEqualTo(1L);
        assertThat(internalRequest.get("context_limit").asInt()).isEqualTo(128);

        List<String> sseLines = Files.readAllLines(fixturesDir.resolve("sse-stream-sample.jsonl"));
        assertThat(sseLines).hasSize(5);
        long prevId = 0;
        for (String line : sseLines) {
            JsonNode evt = objectMapper.readTree(line);
            long id = evt.get("id").asLong();
            assertThat(id).isEqualTo(prevId + 1);
            prevId = id;
            assertThat(evt.get("data").get("schema_version").asInt()).isEqualTo(1);
        }
    }
}
