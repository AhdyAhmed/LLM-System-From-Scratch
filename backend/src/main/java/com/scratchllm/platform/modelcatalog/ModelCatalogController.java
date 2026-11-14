package com.scratchllm.platform.modelcatalog;

import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

/**
 * Day055: Exposes GET /api/v1/models and GET /api/v1/models/{revisionTag} returning immutable
 * model catalog entries backed by cache-aside Redis metadata (`model:rev:{revision}`) and PostgreSQL.
 */
@RestController
public class ModelCatalogController {

    private final ModelCatalogService modelCatalogService;

    public ModelCatalogController(ModelCatalogService modelCatalogService) {
        this.modelCatalogService = modelCatalogService;
    }

    @GetMapping("/api/v1/models")
    public ResponseEntity<ModelCatalogResponse> listModels() {
        return ResponseEntity.ok(new ModelCatalogResponse(modelCatalogService.listReadyModels()));
    }

    @GetMapping("/api/v1/models/{revisionTag}")
    public ResponseEntity<ModelCatalogEntry> getModelByRevision(@PathVariable String revisionTag) {
        return modelCatalogService.getByRevisionTag(revisionTag)
            .map(ResponseEntity::ok)
            .orElseGet(() -> ResponseEntity.notFound().build());
    }

    public record ModelCatalogEntry(
        String alias,
        String model_revision,
        String registry_version,
        String package_sha256,
        String tokenizer_hash,
        int context_limit,
        int max_output_tokens,
        boolean is_scratch_model,
        String capability_note
    ) {}

    public record ModelCatalogResponse(List<ModelCatalogEntry> models) {}
}
