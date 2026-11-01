package com.scratchllm.platform.infrastructure;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.apache.kafka.clients.admin.NewTopic;
import org.apache.kafka.common.config.TopicConfig;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.TopicBuilder;
import org.springframework.kafka.core.KafkaTemplate;

/**
 * Day019: Versioned Kafka topics, retry/DLQ conventions, partition/retention strategy,
 * and strict ID-only contract validation for generation.requested.v1,
 * document.ingestion.requested.v1, and model.released.v1.
 */
@Configuration
public class KafkaEventsInfrastructure {

    public static final String TOPIC_GENERATION_REQUESTED = "generation.requested.v1";
    public static final String TOPIC_GENERATION_RETRY = "generation.requested.retry.v1";
    public static final String TOPIC_GENERATION_DLQ = "generation.requested.dlq.v1";

    public static final String TOPIC_DOCUMENT_INGESTION = "document.ingestion.requested.v1";
    public static final String TOPIC_DOCUMENT_RETRY = "document.ingestion.requested.retry.v1";
    public static final String TOPIC_DOCUMENT_DLQ = "document.ingestion.requested.dlq.v1";

    public static final String TOPIC_MODEL_RELEASED = "model.released.v1";
    public static final String TOPIC_MODEL_RETRY = "model.released.retry.v1";
    public static final String TOPIC_MODEL_DLQ = "model.released.dlq.v1";

    public static final int MAX_KAFKA_PAYLOAD_BYTES = 4096;

    private static final Set<String> FORBIDDEN_HEAVY_FIELDS = Set.of(
        "prompt",
        "prompt_snapshot",
        "text",
        "chunk_text",
        "document_body",
        "embedding",
        "weights",
        "content"
    );

    private static final Map<String, List<String>> REQUIRED_FIELDS_BY_TOPIC = Map.of(
        TOPIC_GENERATION_REQUESTED, List.of(
            "event_id", "event_type", "schema_version", "occurred_at",
            "workspace_id", "conversation_id", "run_id", "user_id", "model_revision"
        ),
        TOPIC_DOCUMENT_INGESTION, List.of(
            "event_id", "event_type", "schema_version", "occurred_at",
            "workspace_id", "document_id", "document_revision", "uploader_user_id", "content_checksum"
        ),
        TOPIC_MODEL_RELEASED, List.of(
            "event_id", "event_type", "schema_version", "occurred_at",
            "promotion_id", "model_name", "release_generation", "target_revision_tag", "artifact_checksum"
        )
    );

    @Bean
    public NewTopic generationRequestedTopic() {
        return buildBoundedTopic(TOPIC_GENERATION_REQUESTED, 3);
    }

    @Bean
    public NewTopic generationRetryTopic() {
        return buildBoundedTopic(TOPIC_GENERATION_RETRY, 3);
    }

    @Bean
    public NewTopic generationDlqTopic() {
        return buildBoundedTopic(TOPIC_GENERATION_DLQ, 1);
    }

    @Bean
    public NewTopic documentIngestionTopic() {
        return buildBoundedTopic(TOPIC_DOCUMENT_INGESTION, 3);
    }

    @Bean
    public NewTopic documentRetryTopic() {
        return buildBoundedTopic(TOPIC_DOCUMENT_RETRY, 3);
    }

    @Bean
    public NewTopic documentDlqTopic() {
        return buildBoundedTopic(TOPIC_DOCUMENT_DLQ, 1);
    }

    @Bean
    public NewTopic modelReleasedTopic() {
        return buildBoundedTopic(TOPIC_MODEL_RELEASED, 1);
    }

    @Bean
    public NewTopic modelRetryTopic() {
        return buildBoundedTopic(TOPIC_MODEL_RETRY, 1);
    }

    @Bean
    public NewTopic modelDlqTopic() {
        return buildBoundedTopic(TOPIC_MODEL_DLQ, 1);
    }

    @Bean
    public KafkaContractBridge kafkaContractBridge(
        KafkaTemplate<String, String> kafkaTemplate,
        ObjectMapper objectMapper
    ) {
        return new KafkaContractBridge(kafkaTemplate, objectMapper);
    }

    private static NewTopic buildBoundedTopic(String name, int partitions) {
        return TopicBuilder.name(name)
            .partitions(partitions)
            .replicas(1)
            .config(TopicConfig.RETENTION_MS_CONFIG, String.valueOf(24 * 60 * 60 * 1000L))
            .config(TopicConfig.MAX_MESSAGE_BYTES_CONFIG, "16384")
            .build();
    }

    public static class KafkaContractBridge {

        private final KafkaTemplate<String, String> kafkaTemplate;
        private final ObjectMapper objectMapper;

        public KafkaContractBridge(
            KafkaTemplate<String, String> kafkaTemplate,
            ObjectMapper objectMapper
        ) {
            this.kafkaTemplate = kafkaTemplate;
            this.objectMapper = objectMapper;
        }

        public void validateEnvelope(String expectedTopic, String rawJson) {
            if (rawJson == null || rawJson.isBlank()) {
                throw new IllegalArgumentException("Kafka event payload must not be empty");
            }
            byte[] utf8Bytes = rawJson.getBytes(StandardCharsets.UTF_8);
            if (utf8Bytes.length > MAX_KAFKA_PAYLOAD_BYTES) {
                throw new IllegalArgumentException(
                    "Kafka event payload exceeds MAX_KAFKA_PAYLOAD_BYTES (" + MAX_KAFKA_PAYLOAD_BYTES + " bytes): " + utf8Bytes.length
                );
            }

            List<String> requiredFields = REQUIRED_FIELDS_BY_TOPIC.get(expectedTopic);
            if (requiredFields == null) {
                throw new IllegalArgumentException("Unsupported Kafka topic contract: " + expectedTopic);
            }

            try {
                JsonNode root = objectMapper.readTree(rawJson);
                if (!root.isObject()) {
                    throw new IllegalArgumentException("Kafka event payload must be a JSON object");
                }
                for (String forbidden : FORBIDDEN_HEAVY_FIELDS) {
                    if (root.has(forbidden)) {
                        throw new IllegalArgumentException(
                            "Forbidden bulk field '" + forbidden + "' in Kafka event; Kafka carries IDs only"
                        );
                    }
                }
                for (String req : requiredFields) {
                    if (!root.hasNonNull(req)) {
                        throw new IllegalArgumentException(
                            "Missing required field '" + req + "' for topic " + expectedTopic
                        );
                    }
                }
                String eventType = root.get("event_type").asText();
                if (!expectedTopic.equals(eventType)) {
                    throw new IllegalArgumentException(
                        "event_type '" + eventType + "' does not match topic '" + expectedTopic + "'"
                    );
                }
                int schemaVersion = root.get("schema_version").asInt(-1);
                if (schemaVersion != 1) {
                    throw new IllegalArgumentException("Unsupported schema_version: " + schemaVersion);
                }
                // Validate UUID format on event_id
                UUID.fromString(root.get("event_id").asText());
            } catch (IllegalArgumentException ex) {
                throw ex;
            } catch (Exception ex) {
                throw new IllegalArgumentException("Malformed JSON in Kafka event payload", ex);
            }
        }

        public void publishValidated(String topic, String aggregateKey, String rawJson) {
            validateEnvelope(topic, rawJson);
            try {
                kafkaTemplate.send(topic, aggregateKey, rawJson).get(10, TimeUnit.SECONDS);
            } catch (Exception ex) {
                throw new IllegalStateException("Failed to publish event to Kafka topic " + topic, ex);
            }
        }
    }
}
