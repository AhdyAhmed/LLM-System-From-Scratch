package com.scratchllm.platform.infrastructure;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.scratchllm.platform.infrastructure.KafkaEventsInfrastructure.KafkaContractBridge;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;

/**
 * Day019 integration test against the live single-broker Kafka KRaft container (127.0.0.1:9094):
 * - Publishes and consumes golden fixtures for generation.requested.v1, document.ingestion.requested.v1,
 *   and model.released.v1
 * - Verifies oversized payloads and forbidden bulk fields (prompt, chunk_text, weights) are rejected
 */
@SpringBootTest
class Phase2KafkaContractsIntegrationTest {

    @Autowired
    private KafkaContractBridge kafkaContractBridge;

    @Value("${spring.kafka.bootstrap-servers}")
    private String bootstrapServers;

    @Test
    void publishesAndConsumesAllThreeVersionedKafkaContractFixtures() throws Exception {
        Path kafkaFixturesDir = Path.of("..", "contracts", "fixtures", "kafka").normalize();

        Map<String, String> topicToFixture = Map.of(
            KafkaEventsInfrastructure.TOPIC_GENERATION_REQUESTED,
            Files.readString(kafkaFixturesDir.resolve("generation-requested-v1.json")),
            KafkaEventsInfrastructure.TOPIC_DOCUMENT_INGESTION,
            Files.readString(kafkaFixturesDir.resolve("document-ingestion-requested-v1.json")),
            KafkaEventsInfrastructure.TOPIC_MODEL_RELEASED,
            Files.readString(kafkaFixturesDir.resolve("model-released-v1.json"))
        );

        String testGroupId = "phase2-contract-verifier-" + UUID.randomUUID();

        try (KafkaConsumer<String, String> consumer = new KafkaConsumer<>(Map.of(
            ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers,
            ConsumerConfig.GROUP_ID_CONFIG, testGroupId,
            ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest",
            ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false",
            ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName(),
            ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName()
        ))) {
            consumer.subscribe(List.copyOf(topicToFixture.keySet()));

            for (Map.Entry<String, String> entry : topicToFixture.entrySet()) {
                kafkaContractBridge.publishValidated(entry.getKey(), "agg-key-1", entry.getValue());
            }

            int receivedCount = 0;
            long deadline = System.currentTimeMillis() + 15_000L;
            while (receivedCount < 3 && System.currentTimeMillis() < deadline) {
                ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(250));
                for (ConsumerRecord<String, String> rec : records) {
                    kafkaContractBridge.validateEnvelope(rec.topic(), rec.value());
                    receivedCount++;
                }
            }
            assertThat(receivedCount).isGreaterThanOrEqualTo(3);
        }
    }

    @Test
    void rejectsForbiddenBulkFieldsAndOversizedKafkaPayloads() {
        String withForbiddenPrompt = """
            {
              "event_id": "a1111111-2222-4333-8444-555555555501",
              "event_type": "generation.requested.v1",
              "schema_version": 1,
              "occurred_at": "2026-10-16T05:00:00Z",
              "workspace_id": "b1111111-2222-4333-8444-555555555501",
              "conversation_id": "c1111111-2222-4333-8444-555555555501",
              "run_id": "d1111111-2222-4333-8444-555555555501",
              "user_id": "e1111111-2222-4333-8444-555555555501",
              "model_revision": "fake-deterministic-r1",
              "prompt": "Kafka must never carry raw prompts or token chunks"
            }
            """;

        assertThatThrownBy(() -> kafkaContractBridge.validateEnvelope(
            KafkaEventsInfrastructure.TOPIC_GENERATION_REQUESTED,
            withForbiddenPrompt
        ))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("Forbidden bulk field 'prompt'");

        String oversized = "{" + "\"padding\":\"" + "x".repeat(5000) + "\"}";
        assertThatThrownBy(() -> kafkaContractBridge.validateEnvelope(
            KafkaEventsInfrastructure.TOPIC_GENERATION_REQUESTED,
            oversized
        ))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("MAX_KAFKA_PAYLOAD_BYTES");
    }
}
