package com.scratchllm.platform.ops;

import com.scratchllm.platform.modelcatalog.ModelCatalogController.ModelCatalogEntry;
import com.scratchllm.platform.modelcatalog.ModelCatalogService;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Day118 & Day119: Release Candidate Manifest & CLI/API Ergonomics Endpoint (`GET /api/v1/release/manifest`).
 * Exposes the frozen release version (`v1.0.0`), active Flyway migration level (`V7`), active model catalog
 * revisions with SHA-256 checksums, SSE reconnect instructions (`Last-Event-ID`), and candid model capability
 * disclaimers for headless CLI and Postman consumers.
 */
@RestController
public class ReleaseManifestController {

    private final JdbcTemplate jdbcTemplate;
    private final ModelCatalogService modelCatalogService;

    public ReleaseManifestController(
        JdbcTemplate jdbcTemplate,
        ModelCatalogService modelCatalogService
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.modelCatalogService = modelCatalogService;
    }

    @GetMapping("/api/v1/release/manifest")
    public ResponseEntity<Map<String, Object>> getReleaseManifest() {
        List<ModelCatalogEntry> readyModels = modelCatalogService.listReadyModels();
        Integer backupsVerified = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM ops_backup_manifests",
            Integer.class
        );
        Integer retentionPoliciesCount = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM ops_retention_policies",
            Integer.class
        );

        Map<String, Object> sseErgonomics = new LinkedHashMap<>();
        sseErgonomics.put("stream_endpoint_template", "/api/v1/runs/{runId}/events");
        sseErgonomics.put("reconnect_header", "Last-Event-ID");
        sseErgonomics.put("unbuffered_proxy_header", "X-Accel-Buffering: no");
        sseErgonomics.put("idempotency_header", "Idempotency-Key");
        sseErgonomics.put("csrf_header", "X-CSRF-TOKEN");

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("release_version", "v1.0.0");
        body.put("gate", "G12");
        body.put("deployment_profile", "local-private-demo");
        body.put("flyway_schema_version", "V7");
        body.put("active_ready_models", readyModels);
        body.put("retention_policies_configured", retentionPoliciesCount != null ? retentionPoliciesCount : 0);
        body.put("backup_snapshots_recorded", backupsVerified != null ? backupsVerified : 0);
        body.put("cli_and_sse_ergonomics", sseErgonomics);
        body.put("honest_capability_disclaimer",
            "ScratchGPT (~0.85M-1.25M parameters) is trained from random initialization on a narrow TinyStories/SFT "
            + "corpus to demonstrate end-to-end LLM architecture and distributed systems engineering; it is not a "
            + "general-purpose assistant."
        );
        body.put("timestamp", OffsetDateTime.now(ZoneOffset.UTC));
        return ResponseEntity.ok(body);
    }
}
