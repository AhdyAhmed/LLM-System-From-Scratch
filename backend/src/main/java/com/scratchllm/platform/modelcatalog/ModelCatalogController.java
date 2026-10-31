package com.scratchllm.platform.modelcatalog;

import java.util.List;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Exposes GET /api/v1/models returning immutable model catalog entries.
 */
@RestController
public class ModelCatalogController {

    private final JdbcTemplate jdbcTemplate;

    public ModelCatalogController(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    @GetMapping("/api/v1/models")
    public ResponseEntity<ModelCatalogResponse> listModels() {
        List<ModelCatalogEntry> models = jdbcTemplate.query(
            """
            SELECT model_alias, revision_tag, mlflow_model_version, tokenizer_hash,
                   context_limit, max_output_tokens, is_scratch_model, capability_note
            FROM model_revisions
            WHERE readiness_status = 'READY'
            ORDER BY created_at ASC
            """,
            (rs, rowNum) -> new ModelCatalogEntry(
                rs.getString("model_alias"),
                rs.getString("revision_tag"),
                rs.getString("mlflow_model_version"),
                rs.getString("tokenizer_hash"),
                rs.getInt("context_limit"),
                rs.getInt("max_output_tokens"),
                rs.getBoolean("is_scratch_model"),
                rs.getString("capability_note")
            )
        );
        return ResponseEntity.ok(new ModelCatalogResponse(models));
    }

    public record ModelCatalogEntry(
        String alias,
        String model_revision,
        String registry_version,
        String tokenizer_hash,
        int context_limit,
        int max_output_tokens,
        boolean is_scratch_model,
        String capability_note
    ) {}

    public record ModelCatalogResponse(List<ModelCatalogEntry> models) {}
}
