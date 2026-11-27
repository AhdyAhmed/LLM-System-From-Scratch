# Backup, Isolated Restore Drill, Model Rollback & Retention Guide (`Day103`–`Day107`)

## 1. Authoritative vs. Disposable State Inventory (`Day103`)

Implemented in [`BackupRestoreAndOperationsService.java`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/backend/src/main/java/com/scratchllm/platform/ops/BackupRestoreAndOperationsService.java), [`ml/src/scratchlm/ops/backup_restore.py`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/ops/backup_restore.py), and [`scripts/backup_platform.ps1`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/scripts/backup_platform.ps1).

| Store / Artifact | Classification | Backup & Restore Policy |
| :--- | :--- | :--- |
| PostgreSQL (`llm_app`, `mlflow_db`, `airflow_db`) | **Authoritative** | Point-in-time logical snapshot + `ops_backup_manifests` SHA-256 inventory |
| Pinned Model Packages (`scratchlm_candidate_pkg`, `frozen_g5_candidate.json`) | **Authoritative** | Backed up with per-file SHA-256 hashes; verified before serving activation |
| Uploaded Document Sources & `pgvector` Chunks | **Authoritative** | Backed up with SHA-256 content hashes; soft-delete tombstones preserved |
| Airflow DAG & Training Configs (`pipelines/dags/`, `ml/configs/`) | **Authoritative** | Versioned in Git + checksummed in backup bundle |
| Redis Sessions (`spring:session:*`) | **Disposable** | **Deliberately invalidated on restore** so all users re-authenticate cleanly |
| Redis Rate-Limit Counters (`rl:*`) | **Disposable** | Reinitialized to conservative sliding/token-bucket windows on restore |
| Redis Model Metadata Cache (`model:rev:*`, `model:alias:*`) | **Disposable** | Rebuilt automatically from PostgreSQL `model_revisions` + `active_model_releases` |
| Kafka Broker Logs (`generation.requested.v1`, `document.ingestion.requested.v1`, `model.released.v1`) | **Disposable / Reconstructible** | Unpublished/unresolved `outbox_events` re-driven from PostgreSQL; duplicates suppressed by `consumer_inbox` (`uq_consumer_inbox_event`) |
| Secrets & `.env` Credentials | **External** | Stored outside backup archives; validated at boot by `RuntimeReadinessAndDrainService` |

---

## 2. Isolated Restore Drill Evidence (`Day104`)

Command:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\restore_drill.ps1 -BackupLabel day104-restore-drill
```
Endpoint: `POST /api/v1/ops/backups/{backupId}/restore-drill`

### Measured Restore Drill Results
- **Target RPO / RTO**: RPO $\le 24\text{ hours}$ ($0\text{ s}$ for transactional outbox snapshot), RTO $< 60\text{ minutes}$.
- **Measured Drill RTO**: `< 50 ms` for local artifact bundle & metadata verification (`status = PASSED`).
- **Safety Invariants Verified**:
  1. Original production databases and model artifact directories remain untouched during isolated restore verification.
  2. All `spring:session:*` keys in Redis are invalidated (`sessions_invalidated = true`) and `model:rev:*` cache entries are rebuilt from PostgreSQL.
  3. Deletion tombstones (`conversations.deleted_at IS NOT NULL`, `documents.status = 'DELETED'`) are re-applied so deleted data cannot reappear after restore.
  4. Replaying an outbox event into `consumer_inbox` (`ON CONFLICT (consumer_name, event_id) DO NOTHING`) is idempotently deduplicated (`deduplicated_outbox_replays = 1`).
  5. Singleton `gpu_leases` (`id = 1`) is reset to `holder_kind = 'NONE', lease_state = 'RELEASED'` so restore never resumes two concurrent GPU owners.

---

## 3. Expand/Contract Migration & Monotonic `release_generation` Model Rollback (`Day105`)

- **Expand/Contract Migration**: [`V7__phase11_ops_backup_audit_and_retention_policies.sql`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/backend/src/main/resources/db/migration/V7__phase11_ops_backup_audit_and_retention_policies.sql) uses purely additive `CREATE TABLE IF NOT EXISTS` and `ON CONFLICT DO NOTHING` statements so the previous application binary remains compatible with the expanded schema.
- **Monotonic Model Rollback**: Implemented in [`ModelPromotionService.java`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/backend/src/main/java/com/scratchllm/platform/modelcatalog/ModelPromotionService.java):
  - Rolling back `scratch-tiny` from `scratch-tiny-r1` (`release_generation = N`) to `scratch-mini-r1` commits a **newer** `release_generation = N + 1` selecting `scratch-mini-r1`, emits `model.released.v1` with `release_generation = N + 1`, and refreshes Redis cache.
  - Any delayed/stale `model.released.v1` event with `release_generation <= applied_release_generation` is ignored (`staleGenerationIgnored = true`).
  - Existing `generation_runs` keep their immutable `generation_runs.model_revision` pinned at submission time.

---

## 4. Retention & Deduplication Horizon Policy (`Day106`)

Configured in `ops_retention_policies` (`V7`) and enforced by `POST /api/v1/ops/retention/run`:
- `consumer_inbox` deduplication retention (`336 hours` / `14 days`) **strictly exceeds** the allowed `outbox_events` / Kafka replay horizon (`168 hours` / `7 days`).
- Manual replay requests older than `168 hours` fail closed with `409 REPLAY_BEYOND_DEDUPE_HORIZON`.
- Pinned `model_revisions`, `active_model_releases`, and referenced MLflow artifacts are permanently protected (`preserve_pinned_lineage = true`).

---

## 5. Single-Service Dependency Outage & Disk-Pressure Drills (`Day107`)

Executed via `POST /api/v1/ops/drills/execute` (`ops_outage_drills`):
- `DISK_PRESSURE`: Rejects new uploads and training jobs with `507 DISK_PRESSURE_THRESHOLD_EXCEEDED` while keeping read/status endpoints healthy.
- `REDIS_CACHE_FALLBACK`: Evicts `model:rev:*` keys and verifies transparent PostgreSQL fallback.
- `KAFKA_OUTBOX_BACKPRESSURE`: Buffers events in `outbox_events` and rejects overflow at high-water mark with `503 OUTBOX_BACKPRESSURE_EXCEEDED`.
- `MLFLOW_PROMOTION_BLOCKED`: Continues active inference from local verified package cache while blocking unverified promotions with `503 MLFLOW_UNAVAILABLE_PROMOTION_BLOCKED`.
- `AIRFLOW_WORKER_RECOVERY`: Reconciles persisted `training_jobs` and stale `gpu_leases` by `(native_pid, process_create_time_utc)`.
