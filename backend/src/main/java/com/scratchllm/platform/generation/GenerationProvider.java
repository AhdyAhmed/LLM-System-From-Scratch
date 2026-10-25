package com.scratchllm.platform.generation;

import java.time.Instant;
import java.util.Objects;
import java.util.UUID;

/**
 * Day005 / Day007 internal provider SPI between the Java dispatcher and model execution backends
 * (deterministic fake provider in tests/offline dev, or Python FastAPI scratch-model worker).
 */
public interface GenerationProvider {

    String providerId();

    boolean supportsModelRevision(String modelRevision);

    GenerationOutcome generateStream(
        GenerationCommand command,
        ChunkConsumer chunkConsumer,
        CancellationSignal cancellationSignal
    );

    @FunctionalInterface
    interface ChunkConsumer {
        void onChunk(GenerationDeltaChunk chunk);
    }

    @FunctionalInterface
    interface CancellationSignal {
        boolean isCancelled();
    }

    record SamplingConfig(
        int maxNewTokens,
        double temperature,
        int topK,
        double topP,
        Long seed
    ) {
        public SamplingConfig {
            if (maxNewTokens < 1 || maxNewTokens > 64) {
                throw new IllegalArgumentException("maxNewTokens must be in [1, 64]");
            }
            if (temperature < 0.0 || temperature > 2.0) {
                throw new IllegalArgumentException("temperature must be in [0.0, 2.0]");
            }
            if (topK < 1 || topK > 256) {
                throw new IllegalArgumentException("topK must be in [1, 256]");
            }
            if (topP < 0.05 || topP > 1.0) {
                throw new IllegalArgumentException("topP must be in [0.05, 1.0]");
            }
        }
    }

    record FaultInjectionSpec(
        long chunkDelayMs,
        Integer failAfterChunks,
        String errorCode
    ) {
        public static FaultInjectionSpec none() {
            return new FaultInjectionSpec(0L, null, null);
        }
    }

    record GenerationCommand(
        UUID runId,
        long attemptToken,
        String modelRevision,
        String prompt,
        int contextLimit,
        SamplingConfig sampling,
        Instant deadline,
        FaultInjectionSpec faultInjection
    ) {
        public GenerationCommand {
            Objects.requireNonNull(runId, "runId must not be null");
            if (attemptToken < 1) {
                throw new IllegalArgumentException("attemptToken must be >= 1");
            }
            Objects.requireNonNull(modelRevision, "modelRevision must not be null");
            Objects.requireNonNull(prompt, "prompt must not be null");
            if (prompt.isBlank() || prompt.length() > 16384) {
                throw new IllegalArgumentException("prompt must be non-blank and <= 16384 chars");
            }
            Objects.requireNonNull(sampling, "sampling must not be null");
            Objects.requireNonNull(deadline, "deadline must not be null");
            if (faultInjection == null) {
                faultInjection = FaultInjectionSpec.none();
            }
        }
    }

    record GenerationDeltaChunk(
        UUID runId,
        long attemptToken,
        int chunkIndex,
        String textDelta,
        int emittedTokens
    ) {}

    enum FinishReason {
        STOP("stop"),
        LENGTH("length"),
        CANCELLED("cancelled"),
        ERROR("error");

        private final String wireValue;

        FinishReason(String wireValue) {
            this.wireValue = wireValue;
        }

        public String wireValue() {
            return wireValue;
        }
    }

    record GenerationOutcome(
        UUID runId,
        long attemptToken,
        String modelRevision,
        FinishReason finishReason,
        String fullText,
        int inputTokens,
        int outputTokens,
        String errorCode,
        String errorMessage
    ) {
        public boolean isSuccess() {
            return finishReason == FinishReason.STOP || finishReason == FinishReason.LENGTH;
        }
    }
}
