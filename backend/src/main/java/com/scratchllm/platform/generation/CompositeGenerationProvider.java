package com.scratchllm.platform.generation;

import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

/**
 * Day054 & Day055: Primary routing GenerationProvider that delegates:
 *   - `fake-deterministic*` revisions to `DeterministicFakeGenerationProvider` (for fast unit/offline tests), and
 *   - `scratch-tiny*` / real scratch-model revisions to `PythonHttpGenerationProvider`.
 */
@Component
@Primary
public class CompositeGenerationProvider implements GenerationProvider {

    private final DeterministicFakeGenerationProvider fakeProvider;
    private final PythonHttpGenerationProvider pythonHttpProvider;

    public CompositeGenerationProvider(
        DeterministicFakeGenerationProvider fakeProvider,
        PythonHttpGenerationProvider pythonHttpProvider
    ) {
        this.fakeProvider = fakeProvider;
        this.pythonHttpProvider = pythonHttpProvider;
    }

    @Override
    public String providerId() {
        return "composite-routing-provider";
    }

    @Override
    public boolean supportsModelRevision(String modelRevision) {
        return fakeProvider.supportsModelRevision(modelRevision)
            || pythonHttpProvider.supportsModelRevision(modelRevision);
    }

    @Override
    public GenerationOutcome generateStream(
        GenerationCommand command,
        ChunkConsumer chunkConsumer,
        CancellationSignal cancellationSignal
    ) {
        if (fakeProvider.supportsModelRevision(command.modelRevision())) {
            return fakeProvider.generateStream(command, chunkConsumer, cancellationSignal);
        }
        return pythonHttpProvider.generateStream(command, chunkConsumer, cancellationSignal);
    }
}
