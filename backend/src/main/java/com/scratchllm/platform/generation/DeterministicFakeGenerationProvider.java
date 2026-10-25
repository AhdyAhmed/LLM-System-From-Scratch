package com.scratchllm.platform.generation;

import java.time.Instant;
import java.util.List;
import org.springframework.stereotype.Component;

/**
 * Day007 deterministic fake generation provider.
 * Enables full backend orchestration, SSE replay, quota settlement, and fault-injection testing
 * without requiring an active GPU or while a long training job owns the Quadro M2000M.
 */
@Component
public class DeterministicFakeGenerationProvider implements GenerationProvider {

    public static final String PROVIDER_ID = "fake-deterministic";
    public static final String REVISION_TAG = "fake-deterministic-r1";

    private static final List<String> DEFAULT_CHUNKS = List.of(
        "Once upon a time",
        ", a tiny deterministic model",
        " streamed four verifiable chunks",
        " without touching the GPU."
    );

    @Override
    public String providerId() {
        return PROVIDER_ID;
    }

    @Override
    public boolean supportsModelRevision(String modelRevision) {
        return REVISION_TAG.equals(modelRevision) || PROVIDER_ID.equals(modelRevision);
    }

    @Override
    public GenerationOutcome generateStream(
        GenerationCommand command,
        ChunkConsumer chunkConsumer,
        CancellationSignal cancellationSignal
    ) {
        int inputTokens = Math.max(1, (command.prompt().length() + 3) / 4);
        int maxTokens = command.sampling().maxNewTokens();
        FaultInjectionSpec fault = command.faultInjection();

        StringBuilder accumulated = new StringBuilder();
        int outputTokens = 0;

        for (int idx = 0; idx < DEFAULT_CHUNKS.size(); idx++) {
            if (cancellationSignal.isCancelled()) {
                return new GenerationOutcome(
                    command.runId(),
                    command.attemptToken(),
                    command.modelRevision(),
                    FinishReason.CANCELLED,
                    accumulated.toString(),
                    inputTokens,
                    outputTokens,
                    "RUN_CANCELLED",
                    "Generation cancelled by caller signal."
                );
            }

            if (Instant.now().isAfter(command.deadline())) {
                return new GenerationOutcome(
                    command.runId(),
                    command.attemptToken(),
                    command.modelRevision(),
                    FinishReason.ERROR,
                    accumulated.toString(),
                    inputTokens,
                    outputTokens,
                    "EXECUTION_DEADLINE_EXCEEDED",
                    "Generation exceeded its configured execution deadline."
                );
            }

            if (fault.failAfterChunks() != null && idx >= fault.failAfterChunks()) {
                String code = (fault.errorCode() != null && !fault.errorCode().isBlank())
                    ? fault.errorCode()
                    : "INJECTED_PROVIDER_FAILURE";
                return new GenerationOutcome(
                    command.runId(),
                    command.attemptToken(),
                    command.modelRevision(),
                    FinishReason.ERROR,
                    accumulated.toString(),
                    inputTokens,
                    outputTokens,
                    code,
                    "Deterministic fault injected after " + idx + " chunks."
                );
            }

            if (fault.chunkDelayMs() > 0) {
                try {
                    Thread.sleep(fault.chunkDelayMs());
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return new GenerationOutcome(
                        command.runId(),
                        command.attemptToken(),
                        command.modelRevision(),
                        FinishReason.CANCELLED,
                        accumulated.toString(),
                        inputTokens,
                        outputTokens,
                        "THREAD_INTERRUPTED",
                        "Provider sleep interrupted."
                    );
                }
            }

            String piece = DEFAULT_CHUNKS.get(idx);
            int pieceTokens = 4;
            if (outputTokens + pieceTokens > maxTokens) {
                int remaining = Math.max(0, maxTokens - outputTokens);
                if (remaining > 0) {
                    accumulated.append(piece);
                    outputTokens += remaining;
                    chunkConsumer.onChunk(new GenerationDeltaChunk(
                        command.runId(),
                        command.attemptToken(),
                        idx + 1,
                        piece,
                        remaining
                    ));
                }
                return new GenerationOutcome(
                    command.runId(),
                    command.attemptToken(),
                    command.modelRevision(),
                    FinishReason.LENGTH,
                    accumulated.toString(),
                    inputTokens,
                    outputTokens,
                    null,
                    null
                );
            }

            accumulated.append(piece);
            outputTokens += pieceTokens;
            chunkConsumer.onChunk(new GenerationDeltaChunk(
                command.runId(),
                command.attemptToken(),
                idx + 1,
                piece,
                pieceTokens
            ));
        }

        return new GenerationOutcome(
            command.runId(),
            command.attemptToken(),
            command.modelRevision(),
            FinishReason.STOP,
            accumulated.toString(),
            inputTokens,
            outputTokens,
            null,
            null
        );
    }
}
