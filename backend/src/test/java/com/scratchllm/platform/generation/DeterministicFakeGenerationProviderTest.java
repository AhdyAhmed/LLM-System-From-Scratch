package com.scratchllm.platform.generation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.scratchllm.platform.generation.GenerationProvider.FaultInjectionSpec;
import com.scratchllm.platform.generation.GenerationProvider.FinishReason;
import com.scratchllm.platform.generation.GenerationProvider.GenerationCommand;
import com.scratchllm.platform.generation.GenerationProvider.GenerationDeltaChunk;
import com.scratchllm.platform.generation.GenerationProvider.GenerationOutcome;
import com.scratchllm.platform.generation.GenerationProvider.SamplingConfig;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;

class DeterministicFakeGenerationProviderTest {

    private final DeterministicFakeGenerationProvider provider = new DeterministicFakeGenerationProvider();

    @Test
    void streamsDeterministicChunksAndCompletesWithoutGpu() {
        UUID runId = UUID.fromString("11111111-2222-4333-8444-555555555555");
        GenerationCommand command = new GenerationCommand(
            runId,
            1L,
            DeterministicFakeGenerationProvider.REVISION_TAG,
            "Once upon a time, a tiny robot learned to count.",
            128,
            new SamplingConfig(32, 0.8, 40, 0.95, 42L),
            Instant.now().plusSeconds(60),
            FaultInjectionSpec.none()
        );

        List<GenerationDeltaChunk> chunks = new ArrayList<>();
        GenerationOutcome outcome = provider.generateStream(command, chunks::add, () -> false);

        assertThat(outcome.isSuccess()).isTrue();
        assertThat(outcome.finishReason()).isEqualTo(FinishReason.STOP);
        assertThat(chunks).hasSize(4);
        assertThat(outcome.fullText()).isEqualTo(
            "Once upon a time, a tiny deterministic model streamed four verifiable chunks without touching the GPU."
        );
        assertThat(outcome.outputTokens()).isEqualTo(16);
        assertThat(outcome.errorCode()).isNull();
    }

    @Test
    void stopsAndReportsInjectedErrorMidStream() {
        UUID runId = UUID.randomUUID();
        GenerationCommand command = new GenerationCommand(
            runId,
            2L,
            DeterministicFakeGenerationProvider.REVISION_TAG,
            "Trigger a mid-stream failure after two chunks.",
            128,
            new SamplingConfig(32, 0.8, 40, 0.95, 42L),
            Instant.now().plusSeconds(60),
            new FaultInjectionSpec(0L, 2, "SIMULATED_MODEL_OOM")
        );

        List<GenerationDeltaChunk> chunks = new ArrayList<>();
        GenerationOutcome outcome = provider.generateStream(command, chunks::add, () -> false);

        assertThat(outcome.isSuccess()).isFalse();
        assertThat(outcome.finishReason()).isEqualTo(FinishReason.ERROR);
        assertThat(outcome.errorCode()).isEqualTo("SIMULATED_MODEL_OOM");
        assertThat(chunks).hasSize(2);
        assertThat(outcome.outputTokens()).isEqualTo(8);
    }

    @Test
    void honoursCancellationSignalBetweenChunks() {
        UUID runId = UUID.randomUUID();
        GenerationCommand command = new GenerationCommand(
            runId,
            3L,
            DeterministicFakeGenerationProvider.REVISION_TAG,
            "Cancel after the first chunk.",
            128,
            new SamplingConfig(32, 0.8, 40, 0.95, 42L),
            Instant.now().plusSeconds(60),
            FaultInjectionSpec.none()
        );

        List<GenerationDeltaChunk> chunks = new ArrayList<>();
        AtomicBoolean cancelled = new AtomicBoolean(false);

        GenerationOutcome outcome = provider.generateStream(
            command,
            chunk -> {
                chunks.add(chunk);
                cancelled.set(true);
            },
            cancelled::get
        );

        assertThat(outcome.isSuccess()).isFalse();
        assertThat(outcome.finishReason()).isEqualTo(FinishReason.CANCELLED);
        assertThat(outcome.errorCode()).isEqualTo("RUN_CANCELLED");
        assertThat(chunks).hasSize(1);
    }

    @Test
    void rejectsOutOfBoundsSamplingOptions() {
        assertThatThrownBy(() -> new SamplingConfig(128, 0.8, 40, 0.95, 42L))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("maxNewTokens");
    }
}
