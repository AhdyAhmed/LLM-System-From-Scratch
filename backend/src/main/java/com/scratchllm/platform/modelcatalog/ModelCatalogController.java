package com.scratchllm.platform.modelcatalog;

import com.scratchllm.platform.modelcatalog.ModelPromotionService.ActiveReleaseRow;
import com.scratchllm.platform.modelcatalog.ModelPromotionService.ModelPromotionResponse;
import com.scratchllm.platform.modelcatalog.ModelPromotionService.PromoteModelRequest;
import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

/**
 * Day055 & Day068: Exposes GET /api/v1/models, GET /api/v1/models/{revisionTag},
 * GET /api/v1/models/aliases/{modelAlias}/active, and POST /api/v1/models/promotions
 * backed by cache-aside Redis metadata (`model:rev:{revision}`), PostgreSQL `model_promotions`,
 * and monotonic `release_generation` tracking.
 */
@RestController
public class ModelCatalogController {

    private final ModelCatalogService modelCatalogService;
    private final ModelPromotionService modelPromotionService;

    public ModelCatalogController(
        ModelCatalogService modelCatalogService,
        ModelPromotionService modelPromotionService
    ) {
        this.modelCatalogService = modelCatalogService;
        this.modelPromotionService = modelPromotionService;
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

    @GetMapping("/api/v1/models/aliases/{modelAlias}/active")
    public ResponseEntity<ActiveReleaseRow> getActiveAliasRelease(@PathVariable String modelAlias) {
        return ResponseEntity.ok(modelPromotionService.getActiveReleaseByAlias(modelAlias));
    }

    @PostMapping("/api/v1/models/promotions")
    public ResponseEntity<ModelPromotionResponse> promoteModel(@RequestBody PromoteModelRequest request) {
        return ResponseEntity.ok(modelPromotionService.promoteOrRollbackModel(request));
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
