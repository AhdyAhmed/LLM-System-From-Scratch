-- =============================================================================
-- ScratchLLM Platform - Phase 11 (Day103-Day107, Gate G11)
-- Expand/Contract Operational Schema: Backup Manifests, Restore Drills,
-- Retention/Deduplication Horizon Policies, and Single-Service Outage Drills
-- =============================================================================

CREATE TABLE IF NOT EXISTS ops_backup_manifests (
    id UUID PRIMARY KEY,
    backup_label VARCHAR(120) NOT NULL UNIQUE,
    databases_included JSONB NOT NULL,
    artifact_inventory_json JSONB NOT NULL,
    manifest_sha256 VARCHAR(64) NOT NULL CHECK (char_length(manifest_sha256) = 64),
    authoritative_items_count INT NOT NULL CHECK (authoritative_items_count >= 0),
    disposable_state_policy VARCHAR(255) NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('COMPLETED', 'VERIFIED', 'EXPIRED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS ops_restore_drills (
    id UUID PRIMARY KEY,
    backup_id UUID NOT NULL REFERENCES ops_backup_manifests(id),
    rto_duration_ms BIGINT NOT NULL CHECK (rto_duration_ms >= 0),
    rpo_seconds BIGINT NOT NULL CHECK (rpo_seconds >= 0),
    sessions_invalidated_count INT NOT NULL CHECK (sessions_invalidated_count >= 0),
    reconciled_inflight_runs INT NOT NULL CHECK (reconciled_inflight_runs >= 0),
    reapplied_tombstones_count INT NOT NULL CHECK (reapplied_tombstones_count >= 0),
    deduplicated_outbox_replays INT NOT NULL CHECK (deduplicated_outbox_replays >= 0),
    gpu_lease_reset_cleanly BOOLEAN NOT NULL DEFAULT TRUE,
    status VARCHAR(32) NOT NULL CHECK (status IN ('PASSED', 'FAILED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS ops_retention_policies (
    resource_type VARCHAR(64) PRIMARY KEY,
    retention_hours INT NOT NULL CHECK (retention_hours >= 1),
    dedupe_horizon_hours INT NOT NULL CHECK (dedupe_horizon_hours >= retention_hours),
    preserve_pinned_lineage BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

INSERT INTO ops_retention_policies (resource_type, retention_hours, dedupe_horizon_hours, preserve_pinned_lineage)
VALUES
    ('consumer_inbox', 336, 336, TRUE),
    ('outbox_events', 168, 336, TRUE),
    ('run_events', 336, 336, TRUE),
    ('orphan_documents', 24, 336, TRUE),
    ('model_revisions', 87600, 87600, TRUE)
ON CONFLICT (resource_type) DO NOTHING;

CREATE TABLE IF NOT EXISTS ops_outage_drills (
    id UUID PRIMARY KEY,
    drill_kind VARCHAR(64) NOT NULL,
    target_service VARCHAR(64) NOT NULL,
    expected_behavior VARCHAR(255) NOT NULL,
    observed_status_code INT NOT NULL,
    observed_error_code VARCHAR(64) NOT NULL,
    passed BOOLEAN NOT NULL DEFAULT TRUE,
    executed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
