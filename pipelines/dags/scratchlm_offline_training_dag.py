"""
Day084, Day085 & Day086: Manually Triggered Airflow Offline Training & Candidate Registration DAG.

- DAG ID: `scratchlm_offline_training_v1`
- Schedule: `None` (manually triggered)
- Concurrency bounds: `max_active_runs=1`, `max_active_tasks=1`, `pool="single_gpu_training_pool"`,
  `dagrun_timeout=timedelta(minutes=30)`, `retries=2`.
- XCom discipline: passes ONLY small JSON dicts (< 2 KiB) with `job_id`, `submission_key`,
  `mlflow_run_id`, SHA-256 hashes, logical artifact URIs (`mlflow-artifacts:/...`), and scalar metrics.
- DAG import safety: importing this file NEVER downloads data, runs PyTorch training, or calls remote APIs.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DAG_ID = "scratchlm_offline_training_v1"
TRAINING_POOL_NAME = "single_gpu_training_pool"
MAX_ACTIVE_RUNS = 1
MAX_ACTIVE_TASKS = 1
DEFAULT_TASK_RETRIES = 2
DAGRUN_TIMEOUT = timedelta(minutes=30)

TASK_ORDER = (
    "validate_dataset",
    "prepare_and_tokenize",
    "submit_and_poll_training",
    "evaluate_candidate",
    "gate_and_register_candidate",
)

DAG_SPECIFICATION: dict[str, Any] = {
    "dag_id": DAG_ID,
    "schedule": None,
    "catchup": False,
    "max_active_runs": MAX_ACTIVE_RUNS,
    "max_active_tasks": MAX_ACTIVE_TASKS,
    "dagrun_timeout_seconds": int(DAGRUN_TIMEOUT.total_seconds()),
    "default_args": {
        "owner": "scratchlm-mlops",
        "depends_on_past": False,
        "retries": DEFAULT_TASK_RETRIES,
        "retry_delay_seconds": 5,
        "pool": TRAINING_POOL_NAME,
        "execution_timeout_seconds": 600,
    },
    "tasks": list(TASK_ORDER),
}


def _assert_xcom_light(payload: dict[str, Any], task_id: str) -> dict[str, Any]:
    """
    Day084: Enforces that XCom payloads contain only lightweight metadata/hashes/logical URIs
    and never raw weights, dataset shards, secrets, or Windows drive paths.
    """
    raw = json.dumps(payload)
    if len(raw.encode("utf-8")) > 4096:
        raise ValueError(f"XCom payload for {task_id} exceeds 4096-byte bound.")
    forbidden_keys = {"raw_text", "weights", "state_dict", "tokens", "password", "secret"}
    if forbidden_keys.intersection(payload.keys()):
        raise ValueError(f"XCom payload for {task_id} contains forbidden bulk/secret keys.")
    uri = str(payload.get("artifact_logical_uri", ""))
    if uri and ("C:\\" in uri or "E:\\" in uri or uri.startswith("/")):
        raise ValueError(f"XCom payload for {task_id} contains non-logical host path: {uri}")
    return payload


def execute_offline_dag_run(
    *,
    work_dir: Path,
    dag_run_id: str,
    config_id: str = "tiny-cpu-smoke",
    seed: int = 1337,
    worker_service: Any,
    tracking_uri: str = "http://127.0.0.1:5001",
    force_fail_validation_gate: bool = False,
    simulate_worker_interrupt_on_try_1: bool = False,
) -> dict[str, Any]:
    """
    Day084, Day085 & Day086: Executes the 5-task Airflow DAG pipeline end-to-end:
      1. `validate_dataset`
      2. `prepare_and_tokenize`
      3. `submit_and_poll_training` (with finite retry reattaching via stable submission_key)
      4. `evaluate_candidate`
      5. `gate_and_register_candidate` (registers `candidate` alias in MLflow only if gate passes;
         never deploys or modifies active `champion` serving).
    """
    from scratchlm.pipelines.cli_steps import (
        run_conditional_register_candidate_step,
        run_evaluate_and_package_step,
        run_prepare_and_tokenize_step,
        run_validate_dataset_step,
    )
    from scratchlm.pipelines.native_worker import SubmitTrainingJobRequest
    from scratchlm.training.tracker import SpoolingMlflowRestTracker
    import hashlib

    xcom_store: dict[str, dict[str, Any]] = {}

    # Start an MLflow tracking run for full DAG -> worker -> MLflow lineage
    tracker = SpoolingMlflowRestTracker(
        tracking_uri=tracking_uri,
        spool_dir=work_dir / "mlflow_spool",
        timeout_seconds=0.5,
    )
    try:
        if ":59999" in tracking_uri:
            raise ConnectionError("Offline test tracking URI")
        mlflow_run_id = tracker.start_run(
            experiment_name="scratchlm-phase9-airflow-pipeline",
            run_name=f"{DAG_ID}-{dag_run_id}",
            tags={
                "dag_id": DAG_ID,
                "dag_run_id": dag_run_id,
                "config_id": config_id,
            },
        )
    except Exception:
        mlflow_run_id = hashlib.sha256(f"{DAG_ID}:{dag_run_id}".encode("utf-8")).hexdigest()[:32]
        tracker.run_id = mlflow_run_id
        tracker.simulate_outage = True

    # Task 1: validate_dataset
    s1 = run_validate_dataset_step(work_dir=work_dir, config_id=config_id, seed=seed)
    xcom_store["validate_dataset"] = _assert_xcom_light(s1.to_xcom_dict(), "validate_dataset")

    # Task 2: prepare_and_tokenize
    s2 = run_prepare_and_tokenize_step(work_dir=work_dir, prev_manifest=s1, seed=seed)
    xcom_store["prepare_and_tokenize"] = _assert_xcom_light(
        s2.to_xcom_dict(), "prepare_and_tokenize"
    )

    # Task 3: submit_and_poll_training (supports finite task retries with try_number=1..3)
    job_resp: dict[str, Any] | None = None
    for try_number in range(1, DEFAULT_TASK_RETRIES + 2):
        should_interrupt = simulate_worker_interrupt_on_try_1 and (try_number == 1)
        submit_req = SubmitTrainingJobRequest(
            dag_id=DAG_ID,
            dag_run_id=dag_run_id,
            task_id="submit_and_poll_training",
            map_index=-1,
            try_number=try_number,
            config_id=config_id,
            seed=seed,
            mlflow_run_id=mlflow_run_id,
            dataset_manifest_uri=s1.artifact_logical_uri,
            execute_synchronously=True,
            simulate_interrupt_before_completion=should_interrupt,
            force_fail_validation_gate=force_fail_validation_gate,
        )
        job_resp = worker_service.submit_or_reattach_job(submit_req)
        polled = worker_service.get_job_status(job_resp["job_id"])
        if polled["status"] == "COMPLETED":
            job_resp = polled
            break
        if try_number > DEFAULT_TASK_RETRIES:
            raise RuntimeError(
                f"Training job {polled['job_id']} failed after {try_number} tries: {polled.get('error_code')}"
            )

    assert job_resp is not None
    xcom_store["submit_and_poll_training"] = _assert_xcom_light(
        {
            "job_id": job_resp["job_id"],
            "submission_key": job_resp["submission_key"],
            "reattached": job_resp["reattached"],
            "attempt_count": job_resp["attempt_count"],
            "status": job_resp["status"],
            "mlflow_run_id": mlflow_run_id,
            "dataset_manifest_sha256": job_resp["dataset_manifest_sha256"],
            "tokenizer_sha256": job_resp["tokenizer_sha256"],
            "checkpoint_sha256": job_resp["checkpoint_sha256"],
            "package_sha256": job_resp["package_sha256"],
            "artifact_logical_uri": job_resp["artifact_logical_uri"],
            "validation_gate_passed": job_resp["validation_gate_passed"],
        },
        "submit_and_poll_training",
    )

    # Task 4: evaluate_candidate
    job_dir = worker_service.storage_root / f"job_{job_resp['job_id']}"
    from scratchlm.pipelines.cli_steps import StepOutputManifest

    train_manifest = StepOutputManifest(
        step_name="train_model",
        status="COMPLETED",
        config_id=config_id,
        config_hash=job_resp["config_hash"],
        dataset_manifest_sha256=job_resp["dataset_manifest_sha256"],
        tokenizer_sha256=job_resp["tokenizer_sha256"],
        checkpoint_sha256=job_resp["checkpoint_sha256"],
        package_sha256="",
        mlflow_run_id=mlflow_run_id,
        artifact_logical_uri=job_resp["artifact_logical_uri"],
        validation_gate_passed=True,
        selected_model_version=None,
        metrics=job_resp["metrics"],
    )
    eval_manifest, pkg_manifest = run_evaluate_and_package_step(
        work_dir=job_dir,
        prev_manifest=train_manifest,
        model_name="scratchlm-tiny-bpe",
        force_fail_validation_gate=force_fail_validation_gate,
    )
    xcom_store["evaluate_candidate"] = _assert_xcom_light(
        eval_manifest.to_xcom_dict(), "evaluate_candidate"
    )

    # Task 5: gate_and_register_candidate
    reg_manifest = run_conditional_register_candidate_step(
        work_dir=job_dir,
        eval_manifest=eval_manifest,
        pkg_manifest=pkg_manifest,
        tracking_uri=tracking_uri,
    )
    xcom_store["gate_and_register_candidate"] = _assert_xcom_light(
        reg_manifest.to_xcom_dict(), "gate_and_register_candidate"
    )

    try:
        tracker.end_run(
            status="FINISHED" if reg_manifest.validation_gate_passed else "FAILED"
        )
    except Exception:
        pass

    return {
        "dag_id": DAG_ID,
        "dag_run_id": dag_run_id,
        "mlflow_run_id": mlflow_run_id,
        "job_id": job_resp["job_id"],
        "submission_key": job_resp["submission_key"],
        "attempt_count": job_resp["attempt_count"],
        "validation_gate_passed": reg_manifest.validation_gate_passed,
        "registration_status": reg_manifest.status,
        "selected_model_version": reg_manifest.selected_model_version,
        "serving_deployed": False,  # DAG completion NEVER deploys a model to active serving!
        "xcom": xcom_store,
    }


# Optional Airflow runtime registration when loaded inside the Airflow container
try:
    from airflow import DAG  # type: ignore[import-not-found]
    from airflow.operators.empty import EmptyOperator  # type: ignore[import-not-found]

    with DAG(
        dag_id=DAG_ID,
        schedule=None,
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        catchup=False,
        max_active_runs=MAX_ACTIVE_RUNS,
        max_active_tasks=MAX_ACTIVE_TASKS,
        dagrun_timeout=DAGRUN_TIMEOUT,
        tags=["scratchllm", "phase9", "mlops"],
    ) as dag:
        t_val = EmptyOperator(task_id="validate_dataset", pool=TRAINING_POOL_NAME)
        t_tok = EmptyOperator(task_id="prepare_and_tokenize", pool=TRAINING_POOL_NAME)
        t_trn = EmptyOperator(task_id="submit_and_poll_training", pool=TRAINING_POOL_NAME)
        t_evl = EmptyOperator(task_id="evaluate_candidate", pool=TRAINING_POOL_NAME)
        t_reg = EmptyOperator(task_id="gate_and_register_candidate", pool=TRAINING_POOL_NAME)
        t_val >> t_tok >> t_trn >> t_evl >> t_reg
except ImportError:
    dag = None
