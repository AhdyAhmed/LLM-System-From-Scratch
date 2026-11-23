"""
Day091, Day095, Day097, Day099 & Day100 automated tests:
  - Day091 benchmark percentile calculations and rehearsal artifact schema
  - Day095 real ScratchGPT KV-cache on vs off matrix, greedy token parity, and memory footprint
  - Day097 local workstation compose profile & launch script invariants
  - Day099 soak & retention verification artifact generation and Gate G10 evidence report
"""

from __future__ import annotations

import json
from pathlib import Path

from scratchlm.benchmarks.profiler import (
    compute_percentiles,
    generate_all_phase10_artifacts,
    run_cpu_demo_verification,
    run_day091_harness_rehearsal,
    run_day095_kv_cache_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_day091_percentile_calculator_and_rehearsal_schema() -> None:
    stats = compute_percentiles([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    assert stats["count"] == 10
    assert stats["min_ms"] == 10.0
    assert stats["p50_ms"] == 50.0
    assert stats["p95_ms"] == 100.0
    assert stats["p99_ms"] == 100.0
    assert stats["max_ms"] == 100.0

    rehearsal = run_day091_harness_rehearsal()
    assert rehearsal["benchmark_id"] == "day091_harness_rehearsal"
    assert rehearsal["status"] == "PASSED"
    assert rehearsal["error_rate"] == 0.0
    assert len(rehearsal["scenarios_registered"]) >= 5


def test_day095_real_model_kv_cache_matrix_parity_and_footprint() -> None:
    matrix_report = run_day095_kv_cache_matrix(
        prompt_lengths=(16, 48),
        output_caps=(8, 16),
        n_layers=2,
        n_heads=4,
        d_model=64,
    )
    assert matrix_report["all_parity_checks_passed"] is True
    assert len(matrix_report["matrix"]) == 4

    for cell in matrix_report["matrix"]:
        assert cell["parity_passed"] is True
        assert cell["greedy_tokens_identical"] is True
        assert cell["max_abs_logit_diff"] <= 1e-5
        assert cell["cached"]["kv_cache_bytes"] > 0
        assert cell["uncached"]["kv_cache_bytes"] == 0
        assert len(cell["cached"]["greedy_tokens"]) == cell["max_new_tokens"]


def test_day097_and_day100_artifacts_and_packaging_files_exist(tmp_path: Path) -> None:
    written = generate_all_phase10_artifacts(repo_root=tmp_path)
    expected_keys = {
        "day091_harness_rehearsal.json",
        "day092_query_plan_profile.json",
        "day094_api_scaling.json",
        "day095_model_serving_matrix.json",
        "day096_mixed_workload.json",
        "day099_soak_summary.json",
        "postgres-query-plans.md",
        "api-scaling.md",
        "model-serving.md",
        "soak.md",
        "g10-gate-evidence.md",
    }
    assert expected_keys.issubset(set(written.keys()))
    for path in written.values():
        assert path.exists()
        assert path.stat().st_size > 100

    # Verify repo-level Day093/Day097/Day100 files exist
    assert (REPO_ROOT / "infra" / "nginx" / "nginx-two-replica.conf").exists()
    assert (REPO_ROOT / "scripts" / "start_native_services.ps1").exists()
    assert (REPO_ROOT / "scripts" / "stop_native_services.ps1").exists()
    assert (REPO_ROOT / "scripts" / "run_cpu_demo.ps1").exists()
    assert (REPO_ROOT / "docs" / "local-setup.md").exists()

    # Verify CPU demo verification runs cleanly
    demo_res = run_cpu_demo_verification()
    assert demo_res["all_parity_checks_passed"] is True

    # Verify generated repo JSON files parse cleanly
    d95_repo = json.loads(
        (REPO_ROOT / "benchmarks" / "results" / "day095_model_serving_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert d95_repo["all_parity_checks_passed"] is True
