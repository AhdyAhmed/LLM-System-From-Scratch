"""
Phase 12 (Day111-Day120, Gate G12) Automated Python Test Suite:
  - Day114: Sealed held-out final test evaluation (`evaluate_sealed_final_test_once`) comparing
    trained release package vs untrained random-init and unigram/bigram baselines with identical tokenization
  - Day111-Day116: Verification of README, ADRs, Transformer walkthrough, Model/Data/Tokenizer cards,
    consolidated engineering report, demo script, and Postman v2.1 collection
  - Day117-Day120: Clean-checkout rehearsal (`verify_clean_checkout_rehearsal`) and release candidate
    manifest (`build_release_candidate_manifest`) with SHA-256 artifact checksums.
"""

from __future__ import annotations

import json
from pathlib import Path

from scratchlm.release.final_release import (
    build_release_candidate_manifest,
    evaluate_sealed_final_test_once,
    generate_all_phase12_artifacts,
    verify_clean_checkout_rehearsal,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_day114_sealed_final_test_evaluation_and_baselines() -> None:
    report = evaluate_sealed_final_test_once(repo_root=REPO_ROOT)
    assert report["evaluation_id"] == "day114_sealed_final_test_evaluation"
    assert report["split"] == "held_out_test_final"

    metrics = report["metrics"]
    assert metrics["evaluated_tokens"] > 10
    assert metrics["trained_release_cross_entropy"] > 0.0
    assert metrics["trained_release_perplexity"] < metrics["untrained_random_perplexity"]
    assert metrics["trained_release_bits_per_byte"] > 0.0
    assert len(report["fixed_seed_samples"]) == 2
    assert (
        report["fixed_seed_samples"][1]["category"]
        == "candid_out_of_domain_failure_example"
    )


def test_day117_day119_day120_clean_checkout_rehearsal_and_release_manifest() -> None:
    all_res = generate_all_phase12_artifacts(repo_root=REPO_ROOT)
    rehearsal = all_res["clean_checkout_rehearsal"]
    rc_manifest = all_res["release_candidate_manifest"]

    assert rehearsal["status"] == "PASSED"
    assert rehearsal["missing_files"] == []
    assert rehearsal["undocumented_global_assumptions"] == 0
    assert rehearsal["model_package_reload_verified"] is True
    assert rehearsal["flyway_migrations_verified"] == [
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
    ]

    assert rc_manifest["release_version"] == "v1.0.0"
    assert rc_manifest["gate"] == "G12"
    assert rc_manifest["status"] == "RELEASE_READY"
    assert len(rc_manifest["artifact_checksums_sha256"]) >= 6

    # Validate Postman collection JSON structure
    postman_path = REPO_ROOT / "docs" / "postman" / "scratchllm-release-collection.json"
    postman_json = json.loads(postman_path.read_text(encoding="utf-8"))
    assert "ScratchLLM Platform v1.0.0" in postman_json["info"]["name"]
    assert len(postman_json["item"]) >= 5
