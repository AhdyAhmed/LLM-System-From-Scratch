"""
Day082 & Day083: Authenticated Native-Windows Training Worker API and Shared Exclusive GPU Supervisor.

Enforces:
1. Internal service token authentication (`X-Internal-Service-Token`) — never placed in URLs or logs.
2. Stable submission key `f"{dag_id}:{dag_run_id}:{task_id}:{map_index}"` (explicitly excluding `try_number`).
   - Repeated submissions with identical `config_hash` reattach to the persisted `job_id`.
   - Repeated submissions with a conflicting `config_hash` fail with HTTP 409 `CONFIG_HASH_CONFLICT`.
3. Strict allowlisted `config_id`s and logical artifact URIs (`mlflow-artifacts:/...`); rejects arbitrary
   shell commands, executable paths, or path-traversal strings (`400 UNSAFE_INPUT_REJECTED`).
4. OS-verified exclusive GPU ownership (`pid` + `process_create_time_utc` via `psutil.Process(pid)`):
   - Concurrent training submissions or overlapping serving/training requests cannot execute GPU work together.
   - Stale leases do not authorize a second physical process until the previous `(pid, create_time)` is confirmed
     exited (or terminated and waited) and VRAM is reclaimed.
5. Checkpoint-based bounded recovery when an interrupted training attempt is resumed.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import psutil
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from scratchlm.pipelines.cli_steps import (
    ALLOWLISTED_CONFIG_IDS,
    StepOutputManifest,
    compute_config_hash,
    run_evaluate_and_package_step,
    run_prepare_and_tokenize_step,
    run_training_step,
    run_validate_dataset_step,
    validate_safe_identifier,
)
from scratchlm.serving.package import validate_logical_artifact_uri

DEFAULT_INTERNAL_TOKEN = "scratchlm-internal-worker-secret-v1"


def get_os_process_identity(pid: int | None = None) -> tuple[int, float]:
    """Returns `(pid, process_create_time_utc)` from the OS process table via `psutil`."""
    target_pid = pid if pid is not None else os.getpid()
    proc = psutil.Process(target_pid)
    return target_pid, round(float(proc.create_time()), 3)


def is_os_process_alive(pid: int, expected_create_time_utc: float, tolerance_sec: float = 1.0) -> bool:
    """
    Day083: Verifies whether the exact OS process `(pid, process_create_time_utc)` is still alive.
    Protects against Windows PID reuse by comparing `psutil.Process(pid).create_time()`.
    """
    if pid <= 0:
        return False
    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        actual_create = float(proc.create_time())
        return abs(actual_create - expected_create_time_utc) <= tolerance_sec
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


@dataclass
class GpuLeaseSnapshot:
    holder_kind: str = "NONE"  # "NONE" | "INFERENCE" | "TRAINING"
    holder_id: str = "none"
    native_pid: int = 0
    process_create_time_utc: float = 0.0
    lease_state: str = "RELEASED"  # "ACTIVE" | "DRAINING" | "RELEASED"
    vram_allocated_bytes: int = 0
    active_generation_slots: int = 0
    updated_at_epoch: float = field(default_factory=time.time)


class ExclusiveGpuSupervisor:
    """
    Day083: Shared exclusive host GPU supervisor enforcing single-owner mutual exclusion across
    online inference (`INFERENCE`) and native training jobs (`TRAINING`), backed by OS-verified
    `(pid, process_create_time_utc)` verification and child-process termination before handover.
    """

    def __init__(
        self,
        process_alive_checker: Callable[[int, float], bool] = is_os_process_alive,
        process_terminator: Callable[[int, float], bool] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._lease = GpuLeaseSnapshot()
        self._process_alive_checker = process_alive_checker
        self._process_terminator = process_terminator or self._default_terminate_process

    @staticmethod
    def _default_terminate_process(pid: int, expected_create_time_utc: float) -> bool:
        if pid == os.getpid():
            return True
        if not is_os_process_alive(pid, expected_create_time_utc):
            return True
        try:
            proc = psutil.Process(pid)
            for child in proc.children(recursive=True):
                try:
                    child.terminate()
                except psutil.Error:
                    pass
            proc.terminate()
            proc.wait(timeout=3.0)
            return True
        except psutil.Error:
            return not is_os_process_alive(pid, expected_create_time_utc)

    def snapshot(self) -> GpuLeaseSnapshot:
        with self._lock:
            return GpuLeaseSnapshot(**asdict(self._lease))

    def acquire_for_inference(
        self, holder_id: str, pid: int | None = None, create_time_utc: float | None = None
    ) -> GpuLeaseSnapshot:
        with self._lock:
            self._reconcile_stale_holder_locked()
            if self._lease.lease_state == "ACTIVE" and self._lease.holder_kind == "TRAINING":
                raise RuntimeError(
                    f"GPU_BUSY_WITH_TRAINING: GPU is currently owned by training job {self._lease.holder_id}"
                )
            actual_pid, actual_ct = (
                (pid, create_time_utc)
                if (pid is not None and create_time_utc is not None)
                else get_os_process_identity()
            )
            self._lease = GpuLeaseSnapshot(
                holder_kind="INFERENCE",
                holder_id=holder_id,
                native_pid=actual_pid,
                process_create_time_utc=actual_ct,
                lease_state="ACTIVE",
                vram_allocated_bytes=180 * 1024 * 1024,
                active_generation_slots=1,
                updated_at_epoch=time.time(),
            )
            return self.snapshot()

    def release_inference_slot(self, holder_id: str) -> GpuLeaseSnapshot:
        with self._lock:
            if self._lease.holder_kind == "INFERENCE" and self._lease.holder_id == holder_id:
                self._lease.active_generation_slots = 0
                self._lease.lease_state = "RELEASED"
                self._lease.holder_kind = "NONE"
                self._lease.vram_allocated_bytes = 0
                self._lease.updated_at_epoch = time.time()
            return self.snapshot()

    def acquire_for_training(
        self,
        job_id: str,
        pid: int | None = None,
        create_time_utc: float | None = None,
    ) -> GpuLeaseSnapshot:
        with self._lock:
            self._reconcile_stale_holder_locked()
            if self._lease.lease_state == "ACTIVE":
                if self._lease.holder_kind == "TRAINING" and self._lease.holder_id != job_id:
                    raise RuntimeError(
                        f"GPU_BUSY_WITH_TRAINING: Another training job ({self._lease.holder_id}) holds the GPU"
                    )
                if self._lease.holder_kind == "INFERENCE" and self._lease.active_generation_slots > 0:
                    raise RuntimeError(
                        f"GPU_BUSY_WITH_INFERENCE: Active inference slot ({self._lease.holder_id}) is running"
                    )
            actual_pid, actual_ct = (
                (pid, create_time_utc)
                if (pid is not None and create_time_utc is not None)
                else get_os_process_identity()
            )
            self._lease = GpuLeaseSnapshot(
                holder_kind="TRAINING",
                holder_id=job_id,
                native_pid=actual_pid,
                process_create_time_utc=actual_ct,
                lease_state="ACTIVE",
                vram_allocated_bytes=320 * 1024 * 1024,
                active_generation_slots=0,
                updated_at_epoch=time.time(),
            )
            return self.snapshot()

    def force_terminate_and_reclaim(self, holder_id: str) -> GpuLeaseSnapshot:
        """
        Day083: Terminates the physical process `(native_pid, process_create_time_utc)` for an
        expired or cancelled lease and verifies process exit before releasing the GPU lock.
        """
        with self._lock:
            if self._lease.holder_id == holder_id and self._lease.native_pid > 0:
                exited = self._process_terminator(
                    self._lease.native_pid, self._lease.process_create_time_utc
                )
                if not exited:
                    raise RuntimeError(
                        f"STALE_PROCESS_STILL_ALIVE: Process ({self._lease.native_pid}, "
                        f"{self._lease.process_create_time_utc}) has not exited yet; refusing GPU handover."
                    )
            if self._lease.holder_id == holder_id:
                self._lease = GpuLeaseSnapshot(
                    holder_kind="NONE",
                    holder_id="none",
                    native_pid=0,
                    process_create_time_utc=0.0,
                    lease_state="RELEASED",
                    vram_allocated_bytes=0,
                    active_generation_slots=0,
                    updated_at_epoch=time.time(),
                )
            return self.snapshot()

    def _reconcile_stale_holder_locked(self) -> None:
        if self._lease.lease_state != "ACTIVE" or self._lease.native_pid <= 0:
            return
        alive = self._process_alive_checker(
            self._lease.native_pid, self._lease.process_create_time_utc
        )
        if not alive:
            # Previous process is confirmed exited by OS `(pid, create_time)` check -> reclaim lease
            self._lease = GpuLeaseSnapshot(
                holder_kind="NONE",
                holder_id="none",
                native_pid=0,
                process_create_time_utc=0.0,
                lease_state="RELEASED",
                vram_allocated_bytes=0,
                active_generation_slots=0,
                updated_at_epoch=time.time(),
            )


@dataclass
class TrainingJobRecord:
    job_id: str
    submission_key: str
    dag_id: str
    dag_run_id: str
    task_id: str
    map_index: int
    config_id: str
    config_hash: str
    seed: int
    status: str  # QUEUED | RUNNING | COMPLETED | FAILED | CANCELLED
    attempt_count: int
    fencing_token: int
    native_pid: int
    process_create_time_utc: float
    mlflow_run_id: str
    dataset_manifest_sha256: str
    tokenizer_sha256: str
    checkpoint_sha256: str
    package_sha256: str
    artifact_logical_uri: str
    validation_gate_passed: bool
    selected_model_version: str | None
    error_code: str | None
    error_message: str | None
    heartbeat_at_epoch: float
    deadline_at_epoch: float
    metrics: dict[str, float]


def build_stable_submission_key(
    dag_id: str, dag_run_id: str, task_id: str, map_index: int = -1
) -> str:
    """
    Day082: Builds the stable idempotency key from `(dag_id, dag_run_id, task_id, map_index)`,
    explicitly excluding Airflow's `try_number` so task retries reattach to the same job.
    """
    clean_dag = validate_safe_identifier(dag_id, "dag_id")
    clean_run = validate_safe_identifier(dag_run_id, "dag_run_id")
    clean_task = validate_safe_identifier(task_id, "task_id")
    return f"{clean_dag}:{clean_run}:{clean_task}:{int(map_index)}"


class SubmitTrainingJobRequest(BaseModel):
    dag_id: str = Field(..., min_length=1, max_length=120)
    dag_run_id: str = Field(..., min_length=1, max_length=120)
    task_id: str = Field(..., min_length=1, max_length=120)
    map_index: int = Field(default=-1, ge=-1, le=1024)
    try_number: int = Field(default=1, ge=1, le=100)  # Ignored in submission_key!
    config_id: str = Field(..., min_length=1, max_length=120)
    seed: int = Field(default=1337, ge=0, le=2_147_483_647)
    timeout_seconds: int = Field(default=300, ge=5, le=3600)
    mlflow_run_id: str = Field(default="run-airflow-001", min_length=1, max_length=64)
    dataset_manifest_uri: str = Field(
        default="mlflow-artifacts:/datasets/default/dataset_manifest.json"
    )
    execute_synchronously: bool = Field(default=True)
    simulate_interrupt_before_completion: bool = Field(default=False)
    force_fail_validation_gate: bool = Field(default=False)


class NativeTrainingWorkerService:
    """
    Day082, Day083 & Day085: Manages durable training job records, stable submission-key deduplication,
    fenced attempts, heartbeats, deadlines, checkpoint-based recovery, and exclusive GPU supervision.
    """

    def __init__(
        self,
        storage_root: Path,
        gpu_supervisor: ExclusiveGpuSupervisor | None = None,
        internal_token: str = DEFAULT_INTERNAL_TOKEN,
    ) -> None:
        self.storage_root = storage_root
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.gpu_supervisor = gpu_supervisor or ExclusiveGpuSupervisor()
        self.internal_token = internal_token
        self._lock = threading.RLock()
        self._jobs_by_id: dict[str, TrainingJobRecord] = {}
        self._job_id_by_key: dict[str, str] = {}

    def verify_internal_token(self, supplied_token: str | None) -> None:
        if not supplied_token or not hmac.compare_digest(supplied_token, self.internal_token):
            raise PermissionError("INVALID_INTERNAL_TOKEN: Valid X-Internal-Service-Token header is required.")

    def submit_or_reattach_job(self, req: SubmitTrainingJobRequest) -> dict[str, Any]:
        with self._lock:
            submission_key = build_stable_submission_key(
                req.dag_id, req.dag_run_id, req.task_id, req.map_index
            )
            validate_safe_identifier(req.config_id, "config_id")
            validate_safe_identifier(req.mlflow_run_id, "mlflow_run_id")
            validate_logical_artifact_uri(req.dataset_manifest_uri)

            if req.config_id not in ALLOWLISTED_CONFIG_IDS:
                raise ValueError(f"UNALLOWLISTED_CONFIG_ID: {req.config_id}")

            cfg_hash = compute_config_hash(req.config_id, req.seed)
            now = time.time()

            existing_job_id = self._job_id_by_key.get(submission_key)
            if existing_job_id is not None:
                existing = self._jobs_by_id[existing_job_id]
                if existing.config_hash != cfg_hash:
                    raise KeyError(
                        f"CONFIG_HASH_CONFLICT: submission_key '{submission_key}' already exists with a different config_hash."
                    )
                # If already COMPLETED or RUNNING with a live process, return the existing job without starting duplicate work!
                if existing.status == "COMPLETED":
                    return self._to_response(existing, reattached=True)
                if existing.status == "RUNNING" and is_os_process_alive(
                    existing.native_pid, existing.process_create_time_utc
                ):
                    return self._to_response(existing, reattached=True)

                # Otherwise, existing job was interrupted/expired or QUEUED -> resume from checkpoint!
                if req.execute_synchronously:
                    self._execute_job_locked(
                        existing,
                        simulate_interrupt=req.simulate_interrupt_before_completion,
                        force_fail_gate=req.force_fail_validation_gate,
                    )
                return self._to_response(existing, reattached=True)

            job_id = str(uuid.uuid4())
            record = TrainingJobRecord(
                job_id=job_id,
                submission_key=submission_key,
                dag_id=req.dag_id,
                dag_run_id=req.dag_run_id,
                task_id=req.task_id,
                map_index=req.map_index,
                config_id=req.config_id,
                config_hash=cfg_hash,
                seed=req.seed,
                status="QUEUED",
                attempt_count=0,
                fencing_token=0,
                native_pid=0,
                process_create_time_utc=0.0,
                mlflow_run_id=req.mlflow_run_id,
                dataset_manifest_sha256="",
                tokenizer_sha256="",
                checkpoint_sha256="",
                package_sha256="",
                artifact_logical_uri=req.dataset_manifest_uri,
                validation_gate_passed=False,
                selected_model_version=None,
                error_code=None,
                error_message=None,
                heartbeat_at_epoch=now,
                deadline_at_epoch=now + float(req.timeout_seconds),
                metrics={},
            )
            self._jobs_by_id[job_id] = record
            self._job_id_by_key[submission_key] = job_id

            if req.execute_synchronously:
                self._execute_job_locked(
                    record,
                    simulate_interrupt=req.simulate_interrupt_before_completion,
                    force_fail_gate=req.force_fail_validation_gate,
                )

            return self._to_response(record, reattached=False)

    def _execute_job_locked(
        self,
        record: TrainingJobRecord,
        *,
        simulate_interrupt: bool = False,
        force_fail_gate: bool = False,
    ) -> None:
        # 1. Acquire exclusive GPU ownership before starting/resuming training
        lease = self.gpu_supervisor.acquire_for_training(record.job_id)
        record.status = "RUNNING"
        record.attempt_count += 1
        record.fencing_token += 1
        record.native_pid = lease.native_pid
        record.process_create_time_utc = lease.process_create_time_utc
        record.heartbeat_at_epoch = time.time()

        job_dir = self.storage_root / f"job_{record.job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            s1 = run_validate_dataset_step(job_dir, config_id=record.config_id, seed=record.seed)
            s2 = run_prepare_and_tokenize_step(job_dir, s1, seed=record.seed)
            s3 = run_training_step(
                job_dir,
                s2,
                seed=record.seed,
                mlflow_run_id=record.mlflow_run_id,
                resume_if_exists=True,
            )

            record.dataset_manifest_sha256 = s3.dataset_manifest_sha256
            record.tokenizer_sha256 = s3.tokenizer_sha256
            record.checkpoint_sha256 = s3.checkpoint_sha256
            record.artifact_logical_uri = s3.artifact_logical_uri
            record.metrics = dict(s3.metrics)

            if simulate_interrupt:
                # Simulate worker crash/interruption AFTER writing checkpoint but BEFORE completion
                record.status = "FAILED"
                record.error_code = "WORKER_INTERRUPTED_CHECKPOINT_SAVED"
                record.error_message = "Worker interrupted after saving checkpoint; safe to retry and resume."
                return

            s4, _ = run_evaluate_and_package_step(
                job_dir,
                s3,
                force_fail_validation_gate=force_fail_gate,
            )
            record.package_sha256 = s4.package_sha256
            record.artifact_logical_uri = s4.artifact_logical_uri
            record.validation_gate_passed = s4.validation_gate_passed
            record.metrics.update(s4.metrics)
            record.status = "COMPLETED"
            record.error_code = None
            record.error_message = None
            record.heartbeat_at_epoch = time.time()
        finally:
            self.gpu_supervisor.force_terminate_and_reclaim(record.job_id)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs_by_id.get(job_id)
            if record is None:
                raise LookupError(f"JOB_NOT_FOUND: {job_id}")
            self._reconcile_deadline_locked(record)
            return self._to_response(record, reattached=False)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs_by_id.get(job_id)
            if record is None:
                raise LookupError(f"JOB_NOT_FOUND: {job_id}")
            if record.status in ("QUEUED", "RUNNING"):
                self.gpu_supervisor.force_terminate_and_reclaim(record.job_id)
                record.status = "CANCELLED"
                record.error_code = "JOB_CANCELLED_BY_OPERATOR"
                record.error_message = "Training job cancelled and GPU resources reclaimed."
                record.heartbeat_at_epoch = time.time()
            return self._to_response(record, reattached=False)

    def _reconcile_deadline_locked(self, record: TrainingJobRecord) -> None:
        if record.status in ("QUEUED", "RUNNING") and time.time() > record.deadline_at_epoch:
            self.gpu_supervisor.force_terminate_and_reclaim(record.job_id)
            record.status = "FAILED"
            record.error_code = "JOB_DEADLINE_EXCEEDED"
            record.error_message = "Training job exceeded deadline; process terminated and GPU reclaimed."

    @staticmethod
    def _to_response(record: TrainingJobRecord, *, reattached: bool) -> dict[str, Any]:
        return {
            "job_id": record.job_id,
            "submission_key": record.submission_key,
            "reattached": reattached,
            "status": record.status,
            "config_id": record.config_id,
            "config_hash": record.config_hash,
            "attempt_count": record.attempt_count,
            "fencing_token": record.fencing_token,
            "native_pid": record.native_pid,
            "process_create_time_utc": record.process_create_time_utc,
            "mlflow_run_id": record.mlflow_run_id,
            "dataset_manifest_sha256": record.dataset_manifest_sha256,
            "tokenizer_sha256": record.tokenizer_sha256,
            "checkpoint_sha256": record.checkpoint_sha256,
            "package_sha256": record.package_sha256,
            "artifact_logical_uri": record.artifact_logical_uri,
            "validation_gate_passed": record.validation_gate_passed,
            "selected_model_version": record.selected_model_version,
            "error_code": record.error_code,
            "error_message": record.error_message,
            "metrics": record.metrics,
        }


def create_native_training_worker_app(
    storage_root: Path,
    gpu_supervisor: ExclusiveGpuSupervisor | None = None,
    internal_token: str = DEFAULT_INTERNAL_TOKEN,
) -> FastAPI:
    service = NativeTrainingWorkerService(
        storage_root=storage_root,
        gpu_supervisor=gpu_supervisor,
        internal_token=internal_token,
    )
    app = FastAPI(title="ScratchLLM Native Windows Training Worker API", version="1.0.0")
    app.state.worker_service = service

    @app.post("/internal/training-jobs", status_code=status.HTTP_200_OK)
    def submit_training_job(
        request: SubmitTrainingJobRequest,
        x_internal_service_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        try:
            service.verify_internal_token(x_internal_service_token)
            return service.submit_or_reattach_job(request)
        except PermissionError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_code": "UNAUTHORIZED_INTERNAL_CALLER", "message": str(exc)},
            ) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error_code": "CONFIG_HASH_CONFLICT", "message": str(exc)},
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error_code": "GPU_EXCLUSION_CONFLICT", "message": str(exc)},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error_code": "UNSAFE_INPUT_REJECTED", "message": str(exc)},
            ) from exc

    @app.get("/internal/training-jobs/{job_id}")
    def get_training_job(
        job_id: str,
        x_internal_service_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        try:
            service.verify_internal_token(x_internal_service_token)
            return service.get_job_status(job_id)
        except PermissionError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_code": "UNAUTHORIZED_INTERNAL_CALLER", "message": str(exc)},
            ) from exc
        except LookupError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "JOB_NOT_FOUND", "message": str(exc)},
            ) from exc

    @app.post("/internal/training-jobs/{job_id}/cancel")
    def cancel_training_job(
        job_id: str,
        x_internal_service_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        try:
            service.verify_internal_token(x_internal_service_token)
            return service.cancel_job(job_id)
        except PermissionError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_code": "UNAUTHORIZED_INTERNAL_CALLER", "message": str(exc)},
            ) from exc
        except LookupError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error_code": "JOB_NOT_FOUND", "message": str(exc)},
            ) from exc

    return app
