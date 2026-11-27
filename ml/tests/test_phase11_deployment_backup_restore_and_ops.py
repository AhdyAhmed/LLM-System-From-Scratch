"""
Phase 11 (Day101-Day110, Gate G11) Automated Python Test Suite:
  - Day101 & Day102: Deployment docs and reverse-proxy TLS/SSE rehearsal config verification
  - Day103 & Day104: Authoritative vs disposable backup manifest creation, SHA-256 tamper detection,
    and isolated restore drill verification
  - Day108: Complete-stack tiny-workload rehearsal report generation and GPU conflict verification
  - Day109 & Day110: Release checklist and Gate G11 evidence report verification
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scratchlm.ops.backup_restore import (
    create_backup_manifest_bundle,
    generate_all_phase11_artifacts,
    run_day108_full_stack_rehearsal,
    verify_isolated_restore_drill,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_day103_day104_backup_manifest_and_isolated_restore_drill_with_tamper_detection(
    tmp_path: Path,
) -> None:
    backups_dir = tmp_path / "backups"
    restore_dir = tmp_path / "restore_target"

    # 1. Create backup manifest bundle
    manifest = create_backup_manifest_bundle(
        backups_dir, label="pytest-day103-snapshot", repo_root=REPO_ROOT
    )
    assert manifest["backup_label"] == "pytest-day103-snapshot"
    assert manifest["authoritative_databases"] == ["llm_app", "mlflow_db", "airflow_db"]
    assert len(manifest["manifest_sha256"]) == 64
    assert len(manifest["copied_artifacts_sha256"]) >= 2
    assert (
        manifest["disposable_state_recovery_policy"]["redis_sessions"]
        == "INVALIDATE_ALL_SESSIONS_ON_RESTORE"
    )

    # 2. Execute isolated restore drill -> PASSED
    drill = verify_isolated_restore_drill(
        backups_dir, restore_dir, label="pytest-day103-snapshot"
    )
    assert drill["status"] == "PASSED"
    assert drill["manifest_sha256_verified"] is True
    assert drill["original_production_files_untouched"] is True
    assert drill["sessions_invalidated"] is True
    assert drill["inbox_deduplication_verified"] is True
    assert drill["active_gpu_owners_on_restore"] == 0

    # 3. Tamper with a backed-up file and verify isolated restore drill rejects it
    dag_backup = backups_dir / "pytest-day103-snapshot" / "scratchlm_offline_training_dag.py"
    dag_backup.write_text("# corrupted backup file\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Restored artifact SHA-256 mismatch"):
        verify_isolated_restore_drill(
            backups_dir, restore_dir / "tampered", label="pytest-day103-snapshot"
        )


def test_day101_day102_day108_day109_day110_phase11_artifacts_and_rehearsal() -> None:
    rehearsal = run_day108_full_stack_rehearsal(repo_root=REPO_ROOT)
    assert rehearsal["status"] == "PASSED"
    assert rehearsal["failures_count"] == 0
    conflict = rehearsal["components_verified"]["deliberate_gpu_training_serving_conflict"]
    assert conflict["concurrent_scratch_tiny_inference_status"] == 503
    assert conflict["concurrent_scratch_tiny_error_code"] == "GPU_BUSY_WITH_TRAINING"
    assert conflict["concurrent_fake_deterministic_status"] == 202
    assert conflict["dual_gpu_allocation_prevented"] is True

    written = generate_all_phase11_artifacts(repo_root=REPO_ROOT)
    assert written["day108_full_stack_rehearsal.json"].exists()

    # Verify required Phase 11 docs, configs, and scripts exist
    assert (REPO_ROOT / "infra" / "nginx" / "nginx-local-tls-rehearsal.conf").exists()
    assert (REPO_ROOT / "scripts" / "backup_platform.ps1").exists()
    assert (REPO_ROOT / "scripts" / "restore_drill.ps1").exists()
    assert (REPO_ROOT / "docs" / "deployment.md").exists()
    assert (REPO_ROOT / "docs" / "restore-drill.md").exists()
    assert (REPO_ROOT / "docs" / "release-checklist.md").exists()
    assert (REPO_ROOT / "docs" / "reports" / "g11-gate-evidence.md").exists()

    d108_disk = json.loads(
        (REPO_ROOT / "benchmarks" / "results" / "day108_full_stack_rehearsal.json").read_text(
            encoding="utf-8"
        )
    )
    assert d108_disk["status"] == "PASSED"
