# MLOps & Airflow Operations Runbook (Day086)

## 1. Purpose & Scope

This runbook documents the operational procedures for the **ScratchLLM Offline Training & MLOps Pipeline** orchestrated by Apache Airflow (`scratchlm_offline_training_v1`) and executed via the authenticated **Native Training Worker & Single-GPU Supervisor** (`POST /internal/training-jobs`).

---

## 2. Architecture & Execution Topology

- **Control Plane / Orchestrator**: Apache Airflow DAG [`scratchlm_offline_training_v1`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/pipelines/dags/scratchlm_offline_training_dag.py)
  - Manually triggered (`schedule=None`, `catchup=False`)
  - Concurrency bounds: `max_active_runs=1`, `max_active_tasks=1`, pool `single_gpu_training_pool`
  - **XCom-Light Policy**: Airflow XCom stores only compact JSON metadata (< 4 KiB) containing logical IDs, SHA-256 digests, and logical artifact URIs. Raw datasets, tokenized shards, and model weights never traverse the Airflow metadata database.
- **Native Worker & Exclusive GPU Supervisor**:
  - Python worker ([`native_worker.py`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/pipelines/native_worker.py)) and Spring Boot PostgreSQL supervisor bridge ([`TrainingJobAndGpuSupervisorService.java`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/backend/src/main/java/com/scratchllm/platform/mlops/TrainingJobAndGpuSupervisorService.java))
  - Authenticated exclusively via `X-Internal-Service-Token`
  - Coordinates single-GPU ownership in PostgreSQL `gpu_leases` (`resource_name = 'primary_gpu'`)

---

## 3. Standard Operating Procedures

### 3.1 Triggering an Offline Training Run

1. Verify that no critical interactive GPU benchmark is currently running:
   ```bash
   curl -s http://127.0.0.1:8080/api/v1/observability/metrics | jq '.gpu_lease'
   ```
2. Trigger the DAG manually with an allowlisted `config_id` (`scratch-tiny-pretrain-v1`, `scratch-tiny-sft-v1`, or `scratch-tiny-eval-v1`):
   ```bash
   airflow dags trigger scratchlm_offline_training_v1 \
     --conf '{"config_id": "scratch-tiny-pretrain-v1"}'
   ```
3. Monitor task progression across the 5 bounded stages:
   - `validate_dataset` -> `prepare_tokenized_shards` -> `submit_and_poll_training_job` -> `evaluate_candidate` -> `register_candidate_if_passed`

### 3.2 Handling Worker Crashes & Task Retries (Stable Submission Key Recovery)

- Every training submission derives a deterministic `submission_key`:
  ```text
  {dag_id}:{dag_run_id}:{task_id}:{map_index}
  ```
- Notice that `try_number` is **intentionally excluded** from `submission_key`.
- **If the Airflow task crashes while the native worker is still running**:
  - On Airflow retry (`try_number = 2`), `POST /internal/training-jobs` matches the existing `submission_key` in `training_jobs`, verifies `config_hash` equality, and **reattaches** (`reattached = true`) without spawning a second GPU process.
- **If the native worker process crashed**:
  - The GPU supervisor verifies `(native_pid, process_create_time_utc)` against the OS process table (`ProcessHandle` / `psutil.Process.create_time()`).
  - Seeing that the prior process is no longer alive (or that its OS start timestamp differs due to PID reuse), the supervisor reclaims the `primary_gpu` lease, increments `fencing_token`, and resumes training from `last_checkpoint_path` verified via `torch.load(..., weights_only=True)` + SHA-256 checksum.

### 3.3 Canceling a Hung or Preempted Training Job

To cancel an active training job and immediately release the `primary_gpu` lease back to online inference:

```bash
curl -X POST "http://127.0.0.1:8080/internal/training-jobs/${JOB_ID}/cancel" \
  -H "X-Internal-Service-Token: ${INTERNAL_SERVICE_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"reason": "Operator manual preemption for interactive GPU serving"}'
```

### 3.4 Promoting or Rolling Back a Registered Candidate Model

Airflow **never** automatically promotes a trained model to `champion` or live serving. Even when `validation_gate_passed = true`, the DAG only registers the artifact as a `candidate` version.

To promote a verified candidate revision (`scratch-tiny-r2`) to live serving:

1. Inspect evaluation report and Gate G7/G9 metrics.
2. Execute the controlled promotion endpoint (`POST /api/v1/models/promotions`):
   - Commits a new row in `model_promotions` with monotonic `release_generation = N + 1`.
   - Updates `active_model_releases` and emits `model.released.v1` via `outbox_events`.
3. To roll back to `scratch-tiny-r1`, submit a new promotion targeting `scratch-tiny-r1` (which advances `release_generation = N + 2`). In-flight runs remain pinned to their accepted `generation_runs.model_revision`.
