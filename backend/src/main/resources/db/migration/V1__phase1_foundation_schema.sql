-- Day006: Phase 1 Foundation Schema for ScratchLLM Platform
-- Authoritative PostgreSQL schema with partial unique indexes, composite workspace constraints,
-- outbox/inbox/work_items durability tables, and pgvector document chunks.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Identity and Workspace Ownership
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(254) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('ACTIVE', 'SUSPENDED', 'DELETED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE workspaces (
    id UUID PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    owner_user_id UUID NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE memberships (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(32) NOT NULL CHECK (role IN ('OWNER', 'MEMBER')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_membership_workspace_user UNIQUE (workspace_id, user_id)
);

-- 2. Creator-Private Conversations inside Workspaces
CREATE TABLE conversations (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    owner_user_id UUID NOT NULL REFERENCES users(id),
    title VARCHAR(160) NOT NULL,
    version BIGINT NOT NULL DEFAULT 0,
    deleted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT fk_conversation_membership
        FOREIGN KEY (workspace_id, owner_user_id)
        REFERENCES memberships (workspace_id, user_id)
);

CREATE INDEX idx_conversations_workspace_owner_updated
    ON conversations (workspace_id, owner_user_id, updated_at DESC, id DESC)
    WHERE deleted_at IS NULL;

-- 3. Immutable Model Revisions & Promotions
CREATE TABLE model_revisions (
    id UUID PRIMARY KEY,
    model_alias VARCHAR(64) NOT NULL,
    revision_tag VARCHAR(120) NOT NULL UNIQUE,
    mlflow_model_name VARCHAR(120) NOT NULL,
    mlflow_model_version VARCHAR(64) NOT NULL,
    mlflow_run_id VARCHAR(64) NOT NULL,
    artifact_checksum VARCHAR(64) NOT NULL,
    tokenizer_hash VARCHAR(64) NOT NULL,
    context_limit INT NOT NULL CHECK (context_limit > 0),
    max_output_tokens INT NOT NULL CHECK (max_output_tokens > 0 AND max_output_tokens <= context_limit),
    is_scratch_model BOOLEAN NOT NULL,
    readiness_status VARCHAR(32) NOT NULL CHECK (readiness_status IN ('READY', 'DRAINING', 'UNHEALTHY', 'RETIRED')),
    capability_note TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE model_promotions (
    id UUID PRIMARY KEY,
    model_name VARCHAR(120) NOT NULL,
    release_generation BIGINT NOT NULL CHECK (release_generation >= 1),
    target_revision_tag VARCHAR(120) NOT NULL REFERENCES model_revisions(revision_tag),
    previous_revision_tag VARCHAR(120) REFERENCES model_revisions(revision_tag),
    status VARCHAR(32) NOT NULL CHECK (status IN ('PENDING', 'DRAINING', 'VERIFYING', 'COMMITTED', 'ROLLED_BACK', 'FAILED')),
    mlflow_alias_reconciled BOOLEAN NOT NULL DEFAULT FALSE,
    approved_by VARCHAR(120) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    committed_at TIMESTAMPTZ,
    CONSTRAINT uq_model_release_generation UNIQUE (model_name, release_generation)
);

-- 4. System Capacity Singleton and Daily Usage Quota Buckets (Canonical Lock Hierarchy)
CREATE TABLE system_capacity (
    id INT PRIMARY KEY CHECK (id = 1),
    max_waiting_runs INT NOT NULL CHECK (max_waiting_runs >= 1),
    active_or_waiting_runs INT NOT NULL DEFAULT 0 CHECK (active_or_waiting_runs >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE usage_buckets (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    user_id UUID NOT NULL REFERENCES users(id),
    utc_date DATE NOT NULL,
    daily_allowance_tokens INT NOT NULL CHECK (daily_allowance_tokens >= 0),
    reserved_tokens INT NOT NULL DEFAULT 0 CHECK (reserved_tokens >= 0),
    settled_tokens INT NOT NULL DEFAULT 0 CHECK (settled_tokens >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_usage_bucket_daily UNIQUE (workspace_id, user_id, utc_date)
);

-- 5. Generation Runs, Messages, Reservations, Idempotency, and Durable SSE Run Events
CREATE TABLE generation_runs (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    user_id UUID NOT NULL REFERENCES users(id),
    status VARCHAR(32) NOT NULL CHECK (status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'CANCELLED', 'FAILED')),
    model_revision VARCHAR(120) NOT NULL REFERENCES model_revisions(revision_tag),
    prompt_snapshot TEXT NOT NULL,
    prompt_snapshot_hash VARCHAR(64) NOT NULL,
    sampling_json JSONB NOT NULL,
    queue_deadline_at TIMESTAMPTZ NOT NULL,
    execution_deadline_at TIMESTAMPTZ NOT NULL,
    attempt_token BIGINT NOT NULL DEFAULT 0 CHECK (attempt_token >= 0),
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    finish_reason VARCHAR(32) CHECK (finish_reason IS NULL OR finish_reason IN ('stop', 'length', 'cancelled', 'error')),
    error_code VARCHAR(64),
    input_tokens INT NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
    output_tokens INT NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- Enforce at most ONE active (QUEUED or RUNNING) generation run per conversation
CREATE UNIQUE INDEX uq_active_run_per_conversation
    ON generation_runs (conversation_id)
    WHERE status IN ('QUEUED', 'RUNNING');

CREATE INDEX idx_generation_runs_user_status
    ON generation_runs (user_id, status);

CREATE TABLE messages (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    run_id UUID REFERENCES generation_runs(id),
    role VARCHAR(32) NOT NULL CHECK (role IN ('SYSTEM', 'USER', 'ASSISTANT')),
    content TEXT NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('COMMITTED', 'PARTIAL_FAILED', 'CANCELLED')),
    message_order BIGINT NOT NULL CHECK (message_order >= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_conversation_message_order UNIQUE (conversation_id, message_order)
);

-- Enforce at most ONE assistant output message per generation run
CREATE UNIQUE INDEX uq_assistant_msg_per_run
    ON messages (run_id)
    WHERE role = 'ASSISTANT';

CREATE TABLE usage_reservations (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL UNIQUE REFERENCES generation_runs(id),
    bucket_id UUID NOT NULL REFERENCES usage_buckets(id),
    reserved_tokens INT NOT NULL CHECK (reserved_tokens >= 0),
    settled_tokens INT CHECK (settled_tokens IS NULL OR settled_tokens >= 0),
    status VARCHAR(32) NOT NULL CHECK (status IN ('RESERVED', 'SETTLED', 'REFUNDED', 'INCOMPLETE_SETTLED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    settled_at TIMESTAMPTZ
);

CREATE TABLE idempotency_records (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    user_id UUID NOT NULL REFERENCES users(id),
    operation VARCHAR(64) NOT NULL,
    idempotency_key VARCHAR(128) NOT NULL,
    request_hash VARCHAR(64) NOT NULL,
    run_id UUID REFERENCES generation_runs(id),
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_idempotency_scope_key UNIQUE (workspace_id, user_id, operation, idempotency_key)
);

CREATE TABLE run_events (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES generation_runs(id) ON DELETE CASCADE,
    sequence BIGINT NOT NULL CHECK (sequence >= 1),
    event_type VARCHAR(32) NOT NULL,
    payload_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_run_event_sequence UNIQUE (run_id, sequence)
);

CREATE INDEX idx_run_events_replay
    ON run_events (run_id, sequence ASC);

-- 6. Transactional Outbox, Consumer Inbox, and Leased Execution Work Items
CREATE TABLE outbox_events (
    event_id UUID PRIMARY KEY,
    topic VARCHAR(120) NOT NULL,
    schema_version INT NOT NULL DEFAULT 1,
    aggregate_key VARCHAR(120) NOT NULL,
    payload_json JSONB NOT NULL,
    publish_attempts INT NOT NULL DEFAULT 0 CHECK (publish_attempts >= 0),
    relay_lease_until TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX idx_outbox_unpublished
    ON outbox_events (created_at ASC)
    WHERE published_at IS NULL;

CREATE TABLE consumer_inbox (
    id UUID PRIMARY KEY,
    consumer_name VARCHAR(120) NOT NULL,
    event_id UUID NOT NULL,
    payload_hash VARCHAR(64) NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_consumer_inbox_event UNIQUE (consumer_name, event_id)
);

CREATE TABLE work_items (
    id UUID PRIMARY KEY,
    work_kind VARCHAR(64) NOT NULL CHECK (work_kind IN ('GENERATION_RUN', 'DOCUMENT_INGESTION')),
    target_id UUID NOT NULL,
    state VARCHAR(32) NOT NULL CHECK (state IN ('PENDING', 'LEASED', 'COMPLETED', 'FAILED', 'CANCELLED')),
    lease_owner VARCHAR(120),
    lease_expires_at TIMESTAMPTZ,
    fencing_token BIGINT NOT NULL DEFAULT 0 CHECK (fencing_token >= 0),
    attempt_count INT NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    next_eligible_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_work_item_target UNIQUE (work_kind, target_id)
);

CREATE INDEX idx_work_items_claimable
    ON work_items (work_kind, next_eligible_at ASC)
    WHERE state IN ('PENDING', 'LEASED');

-- 7. Workspace-Shared Documents, Chunks (pgvector + FTS), and Retrieval Snapshots
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    uploader_user_id UUID NOT NULL REFERENCES users(id),
    storage_key VARCHAR(255) NOT NULL UNIQUE,
    content_checksum VARCHAR(64) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    mime_type VARCHAR(120) NOT NULL,
    byte_size BIGINT NOT NULL CHECK (byte_size >= 0 AND byte_size <= 2097152),
    ingestion_status VARCHAR(32) NOT NULL CHECK (ingestion_status IN ('PENDING', 'PROCESSING', 'READY', 'FAILED', 'DELETED')),
    revision INT NOT NULL DEFAULT 1 CHECK (revision >= 1),
    deleted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    document_revision INT NOT NULL,
    chunk_order INT NOT NULL CHECK (chunk_order >= 0),
    source_location VARCHAR(120) NOT NULL,
    chunk_text TEXT NOT NULL,
    lexical_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    embedding_revision VARCHAR(64) NOT NULL,
    embedding vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_document_chunk_order UNIQUE (document_id, document_revision, chunk_order)
);

CREATE INDEX idx_document_chunks_workspace
    ON document_chunks (workspace_id, document_id);

CREATE INDEX idx_document_chunks_lexical
    ON document_chunks USING GIN (lexical_tsv);

CREATE TABLE retrieval_snapshots (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL UNIQUE REFERENCES generation_runs(id) ON DELETE CASCADE,
    snapshot_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- 8. Native Training Jobs, GPU Leases, Feedback, and Audit Events
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY,
    submission_key VARCHAR(255) NOT NULL UNIQUE,
    config_id VARCHAR(120) NOT NULL,
    config_hash VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL CHECK (status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
    native_pid BIGINT,
    process_create_time_utc TIMESTAMPTZ,
    checkpoint_uri VARCHAR(512),
    mlflow_run_id VARCHAR(64),
    heartbeat_at TIMESTAMPTZ,
    deadline_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE feedback (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    user_id UUID NOT NULL REFERENCES users(id),
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    rating VARCHAR(16) NOT NULL CHECK (rating IN ('UP', 'DOWN')),
    reason VARCHAR(500),
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT uq_user_message_feedback UNIQUE (user_id, message_id)
);

CREATE TABLE audit_events (
    id UUID PRIMARY KEY,
    workspace_id UUID,
    actor_user_id UUID,
    action VARCHAR(64) NOT NULL,
    resource_type VARCHAR(64) NOT NULL,
    resource_id UUID,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- Seed singleton capacity row and baseline model revisions
INSERT INTO system_capacity (id, max_waiting_runs, active_or_waiting_runs)
VALUES (1, 8, 0);

INSERT INTO model_revisions (
    id, model_alias, revision_tag, mlflow_model_name, mlflow_model_version, mlflow_run_id,
    artifact_checksum, tokenizer_hash, context_limit, max_output_tokens, is_scratch_model,
    readiness_status, capability_note
) VALUES
(
    '00000000-0000-4000-8000-000000000001',
    'fake-deterministic',
    'fake-deterministic-r1',
    'fake-deterministic',
    '1',
    'run-fake-0001',
    '0000000000000000000000000000000000000000000000000000000000000001',
    '0000000000000000000000000000000000000000000000000000000000000001',
    128,
    64,
    FALSE,
    'READY',
    'Deterministic development fake provider for backend orchestration, SSE replay, and fault-injection testing.'
),
(
    '00000000-0000-4000-8000-000000000002',
    'scratch-tiny',
    'scratch-tiny-r0',
    'scratch-tiny-lm',
    '0',
    'run-bootstrap-0000',
    '0000000000000000000000000000000000000000000000000000000000000002',
    '0000000000000000000000000000000000000000000000000000000000000002',
    128,
    64,
    TRUE,
    'READY',
    '0.85M-parameter decoder-only transformer trained from random weights on narrow storytelling text; educational capability only.'
);
