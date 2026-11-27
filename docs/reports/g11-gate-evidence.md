# Gate G11 Evidence Report — Deployment Rehearsal, Backups, Rollback, and Operations (`Day101`–`Day110`)

## 1. Executive Summary

Gate **G11** verifies that the ScratchLLM Platform has a documented, reproducible local/private deployment profile, TLS/SSE reverse-proxy configuration, checksummed multi-database backup and isolated restore drills, expand/contract schema migration (`V7`), monotonic `release_generation` model rollback, retention/deduplication horizon enforcement, single-service outage runbooks, and a complete-stack tiny-workload rehearsal (`Day108`).

| Day | Deliverable | Artifact / Evidence | Status |
| :--- | :--- | :--- | :--- |
| `Day101` | Release Deployment Profile & Port/Resource Footprint | [`docs/deployment.md`](../deployment.md) | **PASSED** |
| `Day102` | Reverse-Proxy TLS, Cookie, SSE & Forwarded-Header Rehearsal | [`infra/nginx/nginx-local-tls-rehearsal.conf`](../../infra/nginx/nginx-local-tls-rehearsal.conf), `GET /api/v1/ops/proxy-rehearsal` | **PASSED** |
| `Day103` | Multi-DB & Artifact Backup Inventory with SHA-256 Checksums | [`scripts/backup_platform.ps1`](../../scripts/backup_platform.ps1), `POST /api/v1/ops/backups` | **PASSED** |
| `Day104` | Isolated Restore Drill, Session Invalidation & Inbox Dedupe | [`docs/restore-drill.md`](../restore-drill.md), [`scripts/restore_drill.ps1`](../../scripts/restore_drill.ps1) | **PASSED** |
| `Day105` | Expand/Contract Migration (`V7`) & Monotonic Model Rollback | [`V7__phase11_ops_backup_audit_and_retention_policies.sql`](../../backend/src/main/resources/db/migration/V7__phase11_ops_backup_audit_and_retention_policies.sql), [`ModelPromotionService.java`](../../backend/src/main/java/com/scratchllm/platform/modelcatalog/ModelPromotionService.java) | **PASSED** |
| `Day106` | Retention & Deduplication Horizon Enforcement (`336h > 168h`) | `POST /api/v1/ops/retention/run` (`409 REPLAY_BEYOND_DEDUPE_HORIZON`) | **PASSED** |
| `Day107` | Single-Service Dependency Outage & Disk-Pressure Drills | `POST /api/v1/ops/drills/execute` (`ops_outage_drills`) | **PASSED** |
| `Day108` | Complete-Stack Tiny-Workload Rehearsal + GPU Conflict | [`benchmarks/results/day108_full_stack_rehearsal.json`](../../benchmarks/results/day108_full_stack_rehearsal.json) | **PASSED** |
| `Day109` | Release Checklist & Legacy GPU / Runtime Maintenance Audit | [`docs/release-checklist.md`](../release-checklist.md) | **PASSED** |
| `Day110` | Gate G11 Automated Verification & Freeze | `scripts/check_all.ps1` (`0 errors, 0 failures`) | **PASSED** |

---

## 2. Key Operational Verification Highlights

1. **Authoritative vs. Disposable Recovery (`Day103`–`Day104`)**:
   - Backups record SHA-256 hashes of `llm_app`, `mlflow_db`, `airflow_db`, DAG definitions, and `scratchlm_candidate_pkg`.
   - Restore drills verify checksums in an isolated directory, invalidate all `spring:session:*` keys in Redis, rebuild `model:rev:*` metadata cache from PostgreSQL, re-apply deletion tombstones, deduplicate replayed outbox events via `consumer_inbox`, and reset `gpu_leases` (`id = 1`) to `RELEASED`.
2. **Monotonic Model Rollback (`Day105`)**:
   - Rolling back to a prior immutable `revision_tag` increments `release_generation` (`N -> N + 1`), emits `model.released.v1`, rejects stale `release_generation <= N` events, and preserves the pinned `model_revision` on already-accepted `generation_runs`.
3. **Complete-Stack Tiny-Workload Rehearsal (`Day108`)**:
   - Exercised API + Redis + Kafka + online `fake-deterministic` chat + offline Airflow/MLflow pipeline + deliberate GPU training/serving conflict (`503 GPU_BUSY_WITH_TRAINING` on `scratch-tiny` while `fake-deterministic` succeeded) with `0` failures and `7.8 GB` peak RAM (`< 16 GB` workstation budget).
