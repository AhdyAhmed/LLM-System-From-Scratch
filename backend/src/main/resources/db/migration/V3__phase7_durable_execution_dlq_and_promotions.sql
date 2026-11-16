-- Day061, Day062, Day067, Day068: Phase 7 Durable Execution, Outbox Backpressure,
-- Dead-Letter Quarantine (dlq_events), and Monotonic Model Release Generation Tracking.

-- 1. Outbox High-Water-Mark Backpressure on system_capacity (Day062)
ALTER TABLE system_capacity
    ADD COLUMN IF NOT EXISTS max_unpublished_outbox_events INT NOT NULL DEFAULT 256
    CHECK (max_unpublished_outbox_events >= 1);

UPDATE system_capacity
SET max_unpublished_outbox_events = 256,
    updated_at = clock_timestamp()
WHERE id = 1;

-- 2. Outbox Relay Retry / Error Tracking (Day063)
ALTER TABLE outbox_events
    ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    ADD COLUMN IF NOT EXISTS last_error TEXT;

-- 3. Work Items Heartbeat & Last Error Tracking (Day064)
ALTER TABLE work_items
    ADD COLUMN IF NOT EXISTS last_heartbeat_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_error_code VARCHAR(64);

-- 4. Durable Dead-Letter Quarantine Table for Poison/Invalid/Exhausted Kafka Events (Day067)
CREATE TABLE IF NOT EXISTS dlq_events (
    id UUID PRIMARY KEY,
    event_id UUID NOT NULL UNIQUE,
    original_topic VARCHAR(120) NOT NULL,
    dlq_topic VARCHAR(120) NOT NULL,
    aggregate_key VARCHAR(120) NOT NULL,
    payload_json TEXT NOT NULL,
    failure_reason VARCHAR(120) NOT NULL,
    error_detail TEXT,
    retry_count INT NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    status VARCHAR(32) NOT NULL CHECK (status IN ('QUARANTINED', 'REPLAYED', 'DISCARDED')),
    replayed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX IF NOT EXISTS idx_dlq_events_status_created
    ON dlq_events (status, created_at DESC);

-- 5. Monotonic Active Model Release Generation Table (Day068)
CREATE TABLE IF NOT EXISTS active_model_releases (
    model_alias VARCHAR(64) PRIMARY KEY,
    model_name VARCHAR(120) NOT NULL,
    active_revision_tag VARCHAR(120) NOT NULL REFERENCES model_revisions(revision_tag),
    applied_release_generation BIGINT NOT NULL CHECK (applied_release_generation >= 1),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

INSERT INTO active_model_releases (
    model_alias, model_name, active_revision_tag, applied_release_generation
) VALUES
    ('fake-deterministic', 'fake-deterministic', 'fake-deterministic-r1', 1),
    ('scratch-tiny', 'scratchlm-tiny-bpe', 'scratch-tiny-r1', 1)
ON CONFLICT (model_alias) DO NOTHING;

-- Also register an earlier/alternative ready revision for promotion & rollback testing (Day068)
INSERT INTO model_revisions (
    id, model_alias, revision_tag, mlflow_model_name, mlflow_model_version, mlflow_run_id,
    artifact_checksum, tokenizer_hash, context_limit, max_output_tokens, is_scratch_model,
    readiness_status, capability_note
) VALUES (
    '00000000-0000-4000-8000-000000000004',
    'scratch-tiny',
    'scratch-tiny-r2',
    'scratchlm-tiny-bpe',
    '4',
    '0323942735134499b66f1ac177908bbe',
    'ba30f95421b3016b5b08ddae2b63537b701c759c2cbcc42cc495b929ebc2b12b',
    'a38cb3a119b033b6b8d8680a11af31b3a09af4cbcc2cf759d0a99dcad07bdd4a',
    128,
    64,
    TRUE,
    'READY',
    'Candidate scratch-tiny-r2 revision for monotonic release_generation promotion and rollback verification.'
) ON CONFLICT (revision_tag) DO NOTHING;
