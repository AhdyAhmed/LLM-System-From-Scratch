-- Day082, Day083, Day085 & Day088: Phase 9 Native Training Jobs Lineage,
-- OS-Verified GPU Supervisor (`pid` + `process_create_time_utc`), and Observability Indexes.

ALTER TABLE training_jobs
    ADD COLUMN IF NOT EXISTS dag_id VARCHAR(120) NOT NULL DEFAULT 'manual_cli',
    ADD COLUMN IF NOT EXISTS dag_run_id VARCHAR(120) NOT NULL DEFAULT 'manual_run',
    ADD COLUMN IF NOT EXISTS task_id VARCHAR(120) NOT NULL DEFAULT 'train_task',
    ADD COLUMN IF NOT EXISTS map_index INT NOT NULL DEFAULT -1,
    ADD COLUMN IF NOT EXISTS dataset_manifest_sha256 VARCHAR(64) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS tokenizer_sha256 VARCHAR(64) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS checkpoint_sha256 VARCHAR(64),
    ADD COLUMN IF NOT EXISTS package_sha256 VARCHAR(64),
    ADD COLUMN IF NOT EXISTS artifact_logical_uri VARCHAR(512),
    ADD COLUMN IF NOT EXISTS mlflow_model_version VARCHAR(64),
    ADD COLUMN IF NOT EXISTS validation_gate_passed BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS attempt_count INT NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    ADD COLUMN IF NOT EXISTS fencing_token BIGINT NOT NULL DEFAULT 0 CHECK (fencing_token >= 0),
    ADD COLUMN IF NOT EXISTS error_code VARCHAR(64),
    ADD COLUMN IF NOT EXISTS error_message VARCHAR(500),
    ADD COLUMN IF NOT EXISTS metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_training_jobs_status_deadline
    ON training_jobs (status, deadline_at ASC);

CREATE INDEX IF NOT EXISTS idx_training_jobs_mlflow_run
    ON training_jobs (mlflow_run_id)
    WHERE mlflow_run_id IS NOT NULL;
