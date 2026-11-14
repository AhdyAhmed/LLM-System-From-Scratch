-- Day052, Day055, Day056: Phase 6 Real Scratch Model Promotion (`scratch-tiny-r1`)
-- and Exclusive Single-GPU Lease Singleton Table (`gpu_leases`).

-- 1. Retire bootstrap placeholder `scratch-tiny-r0` and register the verified Gate G5 candidate `scratch-tiny-r1`
UPDATE model_revisions
SET readiness_status = 'RETIRED'
WHERE revision_tag = 'scratch-tiny-r0';

INSERT INTO model_revisions (
    id, model_alias, revision_tag, mlflow_model_name, mlflow_model_version, mlflow_run_id,
    artifact_checksum, tokenizer_hash, context_limit, max_output_tokens, is_scratch_model,
    readiness_status, capability_note
) VALUES (
    '00000000-0000-4000-8000-000000000003',
    'scratch-tiny',
    'scratch-tiny-r1',
    'scratchlm-tiny-bpe',
    '3',
    '0323942735134499b66f1ac177908bbe',
    'ba30f95421b3016b5b08ddae2b63537b701c759c2cbcc42cc495b929ebc2b12b',
    'a38cb3a119b033b6b8d8680a11af31b3a09af4cbcc2cf759d0a99dcad07bdd4a',
    128,
    64,
    TRUE,
    'READY',
    '0.85M-parameter decoder-only ScratchGPT (V=512 byte-level BPE, C=128, L=4, h=4, d=128) pretrained and narrow-SFT fine-tuned from scratch; Gate G5 promoted package.'
) ON CONFLICT (revision_tag) DO NOTHING;

INSERT INTO model_promotions (
    id, model_name, release_generation, target_revision_tag, previous_revision_tag,
    status, mlflow_alias_reconciled, approved_by, committed_at
) VALUES (
    '00000000-0000-4000-8000-000000000060',
    'scratchlm-tiny-bpe',
    1,
    'scratch-tiny-r1',
    'scratch-tiny-r0',
    'COMMITTED',
    TRUE,
    'gate-g5-promotion',
    clock_timestamp()
) ON CONFLICT (model_name, release_generation) DO NOTHING;

-- 2. Single-GPU Exclusive Lease Singleton Table (Day056)
CREATE TABLE IF NOT EXISTS gpu_leases (
    id INT PRIMARY KEY CHECK (id = 1),
    holder_kind VARCHAR(32) NOT NULL CHECK (holder_kind IN ('INFERENCE', 'TRAINING', 'NONE')),
    holder_id VARCHAR(120) NOT NULL,
    native_pid BIGINT NOT NULL,
    process_create_time_utc DOUBLE PRECISION NOT NULL,
    lease_state VARCHAR(32) NOT NULL CHECK (lease_state IN ('ACTIVE', 'DRAINING', 'RELEASED')),
    vram_allocated_bytes BIGINT NOT NULL DEFAULT 0 CHECK (vram_allocated_bytes >= 0),
    active_generation_slots INT NOT NULL DEFAULT 0 CHECK (active_generation_slots >= 0 AND active_generation_slots <= 1),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

INSERT INTO gpu_leases (
    id, holder_kind, holder_id, native_pid, process_create_time_utc, lease_state, vram_allocated_bytes, active_generation_slots
) VALUES (
    1, 'INFERENCE', 'inference-worker-main', 1, 0.0, 'ACTIVE', 0, 0
) ON CONFLICT (id) DO NOTHING;
