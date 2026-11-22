"""
Phase 9 (Day081–Day090, Gate G9) Automated Verification Suite:
- Day081: Pipeline step CLI packaging (`validate_dataset`, `prepare_and_tokenize`, `train_model`,
  `evaluate_and_package`, `conditional_register_candidate`) with typed `StepOutputManifest` and
  `torch.load(..., weights_only=True)` + SHA-256 checkpoint verification.
- Day082: Authenticated Native Training Worker FastAPI bridge (`X-Internal-Service-Token`),
  stable submission key `dag_id:dag_run_id:task_id:map_index` (ignoring `try_number`),
  `409 CONFIG_HASH_CONFLICT`, and `400 UNSAFE_INPUT_REJECTED` on shell/path injection.
- Day083: `ExclusiveGpuSupervisor` single-GPU mutual exclusion between `INFERENCE` and `TRAINING`,
  OS process identity `(pid, process_create_time_utc)` verification via `psutil`, and stale process reclamation.
- Day084 & Day085: `scratchlm_offline_training_v1` Airflow DAG execution, XCom-light (< 4 KiB) invariants,
  finite retry reattachment/checkpoint recovery after worker interrupt, failed validation gate blocking
  candidate registration, and never automatically deploying/promoting `champion`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.dags.scratchlm_offline_training_dag import (
    DAG_ID,
    DAG_SPECIFICATION,
    TRAINING_POOL_NAME,
    execute_offline_dag_run,
)
from scratchlm.pipelines.cli_steps import (
    run_conditional_register_candidate_step,
    run_evaluate_and_package_step,
    run_prepare_and_tokenize_step,
    run_training_step,
    run_validate_dataset_step,
)
from scratchlm.pipelines.native_worker import (
    DEFAULT_INTERNAL_TOKEN,
    ExclusiveGpuSupervisor,
    NativeTrainingWorkerService,
     build_stable_submission_key,
    create_native_training_worker_app,
    get_os_process_identity,
    is_os_process_alive,
)


def test_day081_cli_steps_manifest_pipeline_and_weights_only_security(tmp_path: Path) -> None:
    work_dir = tmp_path / "cli_step_run"

    # Step 1: Validate dataset
    s1 = run_validate_dataset_step(work_dir=work_dir, config_id="tiny-cpu-smoke", seed=1337)
    assert s1.status == "COMPLETED"
    assert len(s1.dataset_manifest_sha256) == 64
    assert s1.artifact_logical_uri.startswith("mlflow-artifacts:/datasets/")

    # Step 2: Prepare & tokenize
    s2 = run_prepare_and_tokenize_step(work_dir=work_dir, prev_manifest=s1, seed=1337)
    assert s2.status == "COMPLETED"
    assert len(s2.tokenizer_sha256) == 64
    assert s2.artifact_logical_uri.startswith("mlflow-artifacts:/tokenizers/")

    # Step 3: Train model & verify checkpoint resume
    s3 = run_training_step(
        work_dir=work_dir,
        prev_manifest=s2,
        seed=1337,
        mlflow_run_id="run-cli-001",
        resume_if_exists=True,
    )
    assert s3.status == "COMPLETED"
    assert len(s3.checkpoint_sha256) == 64
    assert s3.metrics["resumed_from_checkpoint"] == 0.0

    # Re-running Step 3 with resume_if_exists=True verifies SHA-256 and loads via weights_only=True
    s3_resumed = run_training_step(
        work_dir=work_dir,
        prev_manifest=s2,
        seed=1337,
        mlflow_run_id="run-cli-001",
        resume_if_exists=True,
    )
    assert s3_resumed.metrics["resumed_from_checkpoint"] == 1.0
    assert s3_resumed.checkpoint_sha256 == s3.checkpoint_sha256

    # Step 4: Evaluate & package
    s4, pkg_manifest = run_evaluate_and_package_step(
        work_dir=work_dir,
        prev_manifest=s3_resumed,
        model_name="scratchlm-tiny-bpe",
        force_fail_validation_gate=False,
    )
    assert s4.validation_gate_passed is True
    assert len(s4.package_sha256) == 64

    # Step 5: Conditional candidate registration
    s5 = run_conditional_register_candidate_step(
        work_dir=work_dir,
        eval_manifest=s4,
        pkg_manifest=pkg_manifest,
        tracking_uri="http://127.0.0.1:59999",  # offline spool fallback
    )
    assert s5.status == "REGISTERED_CANDIDATE"
    assert s5.selected_model_version is not None


def test_day082_native_worker_auth_stable_key_conflict_and_injection_rejection(
    tmp_path: Path,
) -> None:
    app = create_native_training_worker_app(
        storage_root=tmp_path / "worker_store",
        internal_token=DEFAULT_INTERNAL_TOKEN,
    )
    client = TestClient(app)

    payload = {
        "dag_id": "scratchlm_offline_training_v1",
        "dag_run_id": "manual__2026-10-17T07-00-00",
        "task_id": "submit_and_poll_training",
        "map_index": -1,
        "try_number": 1,
        "config_id": "tiny-cpu-smoke",
        "seed": 1337,
        "mlflow_run_id": "run-worker-test-1",
    }

    # 1. Missing or invalid internal token -> 401 UNAUTHORIZED_INTERNAL_CALLER
    unauth = client.post("/internal/training-jobs", json=payload)
    assert unauth.status_code == 401
    assert unauth.json()["detail"]["error_code"] == "UNAUTHORIZED_INTERNAL_CALLER"

    headers = {"X-Internal-Service-Token": DEFAULT_INTERNAL_TOKEN}

    # 2. First submission (try_number=1) executes and completes
    resp1 = client.post("/internal/training-jobs", json=payload, headers=headers)
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["status"] == "COMPLETED"
    assert body1["reattached"] is False
    assert body1["attempt_count"] == 1
    assert (
        body1["submission_key"]
        == "scratchlm_offline_training_v1:manual__2026-10-17T07-00-00:submit_and_poll_training:-1"
    )

    # 3. Retry submission with try_number=2 and identical config reattaches to same job_id without re-running!
    retry_payload = dict(payload, try_number=2)
    resp2 = client.post("/internal/training-jobs", json=retry_payload, headers=headers)
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["job_id"] == body1["job_id"]
    assert body2["reattached"] is True
    assert body2["attempt_count"] == 1

    # 4. Conflicting config_hash (different seed) on same submission_key -> 409 CONFIG_HASH_CONFLICT
    conflict_payload = dict(payload, try_number=3, seed=9999)
    resp_conflict = client.post("/internal/training-jobs", json=conflict_payload, headers=headers)
    assert resp_conflict.status_code == 409
    assert resp_conflict.json()["detail"]["error_code"] == "CONFIG_HASH_CONFLICT"

    # 5. Unsafe config_id or command injection / path traversal -> 400 UNSAFE_INPUT_REJECTED
    unsafe_payload = dict(
        payload,
        dag_run_id="manual__injection_attempt",
        config_id="tiny-cpu-smoke; powershell -Command calc.exe",
    )
    resp_unsafe = client.post("/internal/training-jobs", json=unsafe_payload, headers=headers)
    assert resp_unsafe.status_code == 400
    assert resp_unsafe.json()["detail"]["error_code"] == "UNSAFE_INPUT_REJECTED"


def test_day083_exclusive_gpu_supervisor_pid_create_time_and_stale_reclamation() -> None:
    my_pid, my_ct = get_os_process_identity()
    assert is_os_process_alive(my_pid, my_ct) is True
    # Simulated PID reuse (wrong create_time_utc by +60s) must return False!
    assert is_os_process_alive(my_pid, my_ct + 60.0) is False

    alive_state = {"pid_alive": True}
    supervisor = ExclusiveGpuSupervisor(
        process_alive_checker=lambda pid, ct: alive_state["pid_alive"],
        process_terminator=lambda pid, ct: True,
    )

    # 1. Acquire GPU for training job-A
    lease_a = supervisor.acquire_for_training("job-A", pid=42420, create_time_utc=1700000000.0)
    assert lease_a.holder_kind == "TRAINING"
    assert lease_a.holder_id == "job-A"
    assert lease_a.lease_state == "ACTIVE"

    # 2. Concurrent training job-B or online inference is rejected while job-A is alive
    with pytest.raises(RuntimeError, match="GPU_BUSY_WITH_TRAINING"):
        supervisor.acquire_for_training("job-B", pid=42421, create_time_utc=1700000005.0)

    with pytest.raises(RuntimeError, match="GPU_BUSY_WITH_TRAINING"):
        supervisor.acquire_for_inference("inf-run-1")

    # 3. Simulate job-A process exit -> supervisor automatically reclaims stale lease and grants to inference
    alive_state["pid_alive"] = False
    inf_lease = supervisor.acquire_for_inference("inf-run-1", pid=my_pid, create_time_utc=my_ct)
    assert inf_lease.holder_kind == "INFERENCE"
    assert inf_lease.holder_id == "inf-run-1"

    # 4. While inference slot is active, training cannot preempt it
    alive_state["pid_alive"] = True
    with pytest.raises(RuntimeError, match="GPU_BUSY_WITH_INFERENCE"):
        supervisor.acquire_for_training("job-C")

    # 5. Releasing inference slot allows training job-C to acquire GPU
    supervisor.release_inference_slot("inf-run-1")
    lease_c = supervisor.acquire_for_training("job-C", pid=my_pid, create_time_utc=my_ct)
    assert lease_c.holder_kind == "TRAINING"
    assert lease_c.holder_id == "job-C"


def test_day084_and_day085_airflow_dag_xcom_light_retry_resume_and_failed_gate(
    tmp_path: Path,
) -> None:
    # Verify DAG specification bounds
    assert DAG_SPECIFICATION["dag_id"] == DAG_ID
    assert DAG_SPECIFICATION["schedule"] is None
    assert DAG_SPECIFICATION["max_active_runs"] == 1
    assert DAG_SPECIFICATION["max_active_tasks"] == 1
    assert DAG_SPECIFICATION["default_args"]["pool"] == TRAINING_POOL_NAME

    worker_service = NativeTrainingWorkerService(storage_root=tmp_path / "dag_worker_store")

    # 1. Execute DAG with simulated worker crash on try_number=1 after saving checkpoint;
    #    try_number=2 reattaches via stable submission_key, resumes from checkpoint, and registers candidate
    res_recovered = execute_offline_dag_run(
        work_dir=tmp_path / "dag_run_1",
        dag_run_id="manual__2026-10-17T08-00-00",
        config_id="tiny-cpu-smoke",
        seed=1337,
        worker_service=worker_service,
        tracking_uri="http://127.0.0.1:59999",
        force_fail_validation_gate=False,
        simulate_worker_interrupt_on_try_1=True,
    )
    assert res_recovered["attempt_count"] == 2
    assert res_recovered["validation_gate_passed"] is True
    assert res_recovered["registration_status"] == "REGISTERED_CANDIDATE"
    assert res_recovered["selected_model_version"] is not None
    assert res_recovered["serving_deployed"] is False

    # Verify all XCom entries are < 2 KiB and use logical artifact URIs
    for task_id, xcom_payload in res_recovered["xcom"].items():
        encoded = json.dumps(xcom_payload).encode("utf-8")
        assert len(encoded) < 2048, f"XCom for {task_id} too large"
        assert xcom_payload["artifact_logical_uri"].startswith("mlflow-artifacts:/")

    # 2. Execute DAG where validation gate fails -> candidate registration is blocked!
    res_failed_gate = execute_offline_dag_run(
        work_dir=tmp_path / "dag_run_2",
        dag_run_id="manual__2026-10-17T09-00-00",
        config_id="tiny-cpu-smoke",
        seed=1337,
        worker_service=worker_service,
        tracking_uri="http://127.0.0.1:59999",
        force_fail_validation_gate=True,
        simulate_worker_interrupt_on_try_1=False,
    )
    assert res_failed_gate["validation_gate_passed"] is False
    assert res_failed_gate["registration_status"] == "SKIPPED_FAILED_GATE"
    assert res_failed_gate["selected_model_version"] is None
    assert res_failed_gate["serving_deployed"] is False
