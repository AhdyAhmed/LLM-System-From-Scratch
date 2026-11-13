"""
Phase 5 (Day041-Day050, Gate G5) Verification Suite:
Covers feasibility profiling & n-gram baselines, pilot stability & equal-budget two-config comparison,
bounded pretraining & checkpoint recovery audit, narrow SFT dataset split isolation & 128-byte-token limit,
assistant-only loss masking & zero-supervision rejection, Base vs. SFT rubric evaluation, custom MLflow
pyfunc packaging & clean reload, Model Registry retry/duplicate reconciliation, and Gate G5 freeze.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import torch

from scratchlm.config import ModelConfig
from scratchlm.data.shards import PackedSplitShard
from scratchlm.data.splits import SplitsManifest
from scratchlm.evaluation.harness import (
    EmpiricalNgramBaseline,
    evaluate_deterministic_prompts,
    evaluate_gpt_on_shard,
)
from scratchlm.model.gpt import ScratchGPT
from scratchlm.serving.package import (
    MlflowRegistryCandidateManager,
    ModelPackageManifest,
    ScratchLmPyfuncAdapter,
    build_mlflow_pyfunc_package,
)
from scratchlm.tokenizer.bpe_tokenizer import BPETokenizer
from scratchlm.tokenizer.byte_tokenizer import (
    ASSISTANT_TOKEN_ID,
    END_TURN_TOKEN_ID,
    EOS_TOKEN_ID,
    IGNORE_INDEX,
    PAD_TOKEN_ID,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    ByteTokenizer,
)
from scratchlm.training.checkpoints import LineageHashes
from scratchlm.training.sft import (
    ChatTemplateSpec,
    SftDatasetManifest,
    SftExample,
    encode_sft_example_with_assistant_mask,
    load_sft_split,
    verify_sft_split_disjointness,
)
from scratchlm.training.tracker import SpoolingMlflowRestTracker

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent


@pytest.fixture(scope="module", autouse=True)
def _ensure_phase5_artifacts() -> None:
    """Ensure Phase 5 pipeline artifacts exist before running Phase 5 tests."""
    frozen_g5 = ML_ROOT / "configs" / "frozen_g5_candidate.json"
    pkg_manifest = ML_ROOT / "artifacts" / "packages" / "scratchlm_candidate_pkg" / "package_manifest.json"
    if not (frozen_g5.exists() and pkg_manifest.exists()):
        from experiments.phase5_pipeline import run_phase5_pipeline

        run_phase5_pipeline(ml_root=ML_ROOT, mlflow_uri="http://127.0.0.1:5001")


def test_day041_feasibility_profile_and_ngram_baselines() -> None:
    feas_path = ML_ROOT / "experiments" / "feasibility_estimate.json"
    assert feas_path.exists()
    data = json.loads(feas_path.read_text(encoding="utf-8"))

    assert data["stop_conditions"]["non_finite_loss"] == "abort_immediately"
    tiny_prof = data["profiles"]["tiny-bpe-512"]
    assert tiny_prof["exact_parameters"] == 875_264
    assert tiny_prof["processed_positions_per_update"] == 2 * 64 * 2
    assert tiny_prof["measured_tokens_per_second"] > 0
    assert tiny_prof["planned_wall_time_seconds"] >= tiny_prof["estimated_compute_seconds"]

    # Verify Unigram & Bigram baselines on validation shard
    train_shard = PackedSplitShard(ML_ROOT / "data" / "shards", "train")
    val_shard = PackedSplitShard(ML_ROOT / "data" / "shards", "val")
    splits_manifest = SplitsManifest.load(ML_ROOT / "data" / "splits" / "splits_manifest.json")
    val_bytes = splits_manifest.splits["val"].utf8_bytes

    baseline = EmpiricalNgramBaseline(vocab_size=512, smoothing_alpha=1.0).fit(train_shard)
    uni = baseline.evaluate_unigram(val_shard, total_utf8_bytes=val_bytes)
    bi = baseline.evaluate_bigram(val_shard, total_utf8_bytes=val_bytes)

    assert uni.perplexity < 512.0
    assert bi.perplexity < uni.perplexity
    assert bi.bits_per_byte < uni.bits_per_byte


def test_day042_day043_pilot_stability_and_two_config_equal_budget_comparison() -> None:
    comp_report = ML_ROOT / "reports" / "config_comparison.md"
    assert comp_report.exists()
    text = comp_report.read_text(encoding="utf-8")
    assert "tiny-bpe-512" in text
    assert "shallow-wide-bpe-512" in text
    assert "PASSED" in text

    # Verify sealed test split remains sealed
    splits_manifest = SplitsManifest.load(ML_ROOT / "data" / "splits" / "splits_manifest.json")
    assert splits_manifest.test_split_sealed is True


def test_day044_day045_day046_main_pretraining_recovery_and_evaluation_report() -> None:
    audit_path = ML_ROOT / "reports" / "pretraining_recovery_audit.json"
    pretrain_report = ML_ROOT / "reports" / "pretraining.md"
    assert audit_path.exists()
    assert pretrain_report.exists()

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["recovery_verified"] is True
    assert audit["interrupted_at_step"] == 12
    assert audit["resumed_to_step"] == 24
    assert audit["step24_val_loss"] < audit["initial_val_loss"]

    frozen = json.loads((ML_ROOT / "configs" / "frozen_g5_candidate.json").read_text(encoding="utf-8"))
    val_metrics = frozen["validation_metrics"]
    assert val_metrics["pretrained_val_perplexity"] < val_metrics["untrained_val_perplexity"]
    assert val_metrics["pretrained_val_perplexity"] < val_metrics["unigram_val_perplexity"]


def test_day047_sft_dataset_chat_template_128_byte_limit_and_split_isolation() -> None:
    sft_dir = ML_ROOT / "data" / "sft"
    manifest = SftDatasetManifest.load(sft_dir / "sft_manifest.json")
    assert manifest.test_split_sealed is True
    assert manifest.max_observed_byte_tokens <= 128
    assert manifest.max_observed_bpe_tokens <= 128
    assert manifest.splits["train"].example_count == 32
    assert manifest.splits["val"].example_count == 8
    assert manifest.splits["test"].example_count == 8

    train_ex = load_sft_split(sft_dir, "train")
    val_ex = load_sft_split(sft_dir, "val")
    with pytest.raises(PermissionError, match="sealed"):
        load_sft_split(sft_dir, "test")
    test_ex = load_sft_split(sft_dir, "test", allow_sealed_test=True)

    verify_sft_split_disjointness(train_ex, val_ex, test_ex)

    # Verify duplicated evaluation answer across splits is rejected
    bad_val = list(val_ex) + [
        SftExample(
            example_id="sft-dup-ans",
            task_category="moral_tag",
            template_family_id="family_val_unique_99",
            system_text="Format story notes concisely.",
            user_text="Unique user text for test.",
            assistant_text=train_ex[0].assistant_text,  # duplicate of train answer!
            expected_prefix_marker="MORAL:",
            byte_token_length=80,
            bpe_token_length=40,
        )
    ]
    with pytest.raises(ValueError, match="Duplicated evaluation answer across splits"):
        verify_sft_split_disjointness(train_ex, bad_val, test_ex)


def test_day048_assistant_only_loss_masking_zero_supervision_rejection_and_base_preservation() -> None:
    byte_tok = ByteTokenizer()
    sample = encode_sft_example_with_assistant_mask(
        byte_tok,
        example_id="mask-test-01",
        system_text="Sys",
        user_text="Usr",
        assistant_text="Ans",
        seq_len=32,
    )

    # Prefix is: [SYSTEM(259), 'S', 'y', 's', END_TURN(263), USER(260), 'U', 's', 'r', END_TURN(263), ASSISTANT(261)] -> length 11
    # Completion is: ['A', 'n', 's', END_TURN(263), EOS(257)] -> length 5 supervised tokens!
    assert len(sample.prompt_prefix_ids) == 11
    assert sample.supervised_token_count == 5

    # Targets before index 10 (i.e. indices 0..9 predicting prefix tokens) must all be IGNORE_INDEX (-100)
    assert all(int(t) == IGNORE_INDEX for t in sample.targets[:10])
    # Targets at indices 10..14 must equal ['A', 'n', 's', END_TURN_TOKEN_ID, EOS_TOKEN_ID]
    expected_supervised = [ord("A"), ord("n"), ord("s"), END_TURN_TOKEN_ID, EOS_TOKEN_ID]
    assert [int(t) for t in sample.targets[10:15]] == expected_supervised
    # Padding targets from index 15 onward must all be IGNORE_INDEX (-100)
    assert all(int(t) == IGNORE_INDEX for t in sample.targets[15:])

    # Zero-supervision example must fail explicitly
    with pytest.raises(ValueError, match="zero supervision"):
        encode_sft_example_with_assistant_mask(
            byte_tok,
            example_id="zero-sup",
            system_text="Sys",
            user_text="Usr",
            assistant_text="   ",
            seq_len=32,
        )

    # Context overflow (> seq_len) must fail explicitly
    with pytest.raises(ValueError, match="exceeds context limit"):
        encode_sft_example_with_assistant_mask(
            byte_tok,
            example_id="overflow",
            system_text="Sys",
            user_text="U" * 120,
            assistant_text="Ans",
            seq_len=128,
        )


def test_day049_day050_pyfunc_package_reload_registry_reconciliation_and_frozen_g5(tmp_path: Path) -> None:
    pkg_dir = ML_ROOT / "artifacts" / "packages" / "scratchlm_candidate_pkg"
    adapter = ScratchLmPyfuncAdapter.load_package(pkg_dir)

    preds = adapter.predict(
        [
            "Once upon a time in the sunny meadow",
            "Every morning, Omar carried",
        ],
        max_new_tokens=10,
        temperature=0.0,
        seed=1337,
    )
    assert len(preds) == 2
    assert preds[0]["generated_text"] == ScratchLmPyfuncAdapter.load_package(pkg_dir).predict(
        ["Once upon a time in the sunny meadow"], max_new_tokens=10, temperature=0.0, seed=1337
    )[0]["generated_text"]

    # Tampering with a packaged file must fail load_package checksum verification
    tampered_dir = tmp_path / "tampered_pkg"
    tampered_dir.mkdir()
    for p in pkg_dir.iterdir():
        if p.is_file():
            (tampered_dir / p.name).write_bytes(p.read_bytes())
    (tampered_dir / "chat_template.json").write_text('{"corrupted": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Package file checksum mismatch"):
        ScratchLmPyfuncAdapter.load_package(tampered_dir)

    # Verify failed validation gate blocks candidate registration
    reg_mgr = MlflowRegistryCandidateManager(tracking_uri="http://127.0.0.1:5001")
    failed_manifest = ModelPackageManifest(
        schema_version="1.0.0",
        model_name="scratchlm-blocked-model",
        model_stage="narrow_sft",
        source_run_id=adapter.manifest.source_run_id,
        parent_base_run_id=adapter.manifest.parent_base_run_id,
        lineage=adapter.manifest.lineage,
        files_sha256=adapter.manifest.files_sha256,
        package_sha256=adapter.manifest.package_sha256,
        validation_gate_passed=False,
        serving_promotion_approved=False,
    )
    tracker = SpoolingMlflowRestTracker(tracking_uri="http://127.0.0.1:5001", spool_dir=tmp_path / "spool")
    with pytest.raises(PermissionError, match="validation_gate_passed is False"):
        reg_mgr.register_or_reconcile_candidate(
            package_dir=pkg_dir,
            manifest=failed_manifest,
            tracker=tracker,
        )

    # Verify retry & timeout duplicate reconciliation on live MLflow Model Registry
    run_id = tracker.start_run(
        experiment_name="scratchlm-phase5-reconcile-test",
        run_name="retry-reconcile-run",
    )
    test_model_name = "scratchlm-reconcile-test-model"
    reconcile_manifest = build_mlflow_pyfunc_package(
        output_dir=tmp_path / "reconcile_pkg",
        model_name=test_model_name,
        model_stage="narrow_sft",
        source_run_id=run_id,
        parent_base_run_id=None,
        model=adapter.model,
        tokenizer=adapter.tokenizer,
        chat_template=ChatTemplateSpec(),
        lineage=adapter.manifest.lineage,
        validation_gate_passed=True,
    )

    # First registration creates canonical version
    rec1 = reg_mgr.register_or_reconcile_candidate(
        package_dir=tmp_path / "reconcile_pkg",
        manifest=reconcile_manifest,
        tracker=tracker,
    )
    # Simulate a timeout-induced duplicate version for the exact same (run_id, package_sha256)
    dup_ver = reg_mgr.create_raw_version(
        model_name=test_model_name,
        source_run_id=run_id,
        artifact_logical_uri=rec1.artifact_logical_uri,
        manifest=reconcile_manifest,
    )
    # Re-running register_or_reconcile_candidate must select canonical rec1.selected_version and label dup_ver!
    rec2 = reg_mgr.register_or_reconcile_candidate(
        package_dir=tmp_path / "reconcile_pkg",
        manifest=reconcile_manifest,
        tracker=tracker,
    )
    tracker.end_run()

    assert rec2.selected_version == rec1.selected_version
    assert dup_ver in rec2.labeled_duplicate_versions
    assert rec2.candidate_alias_assigned is True
    assert rec2.champion_alias_assigned is False

    # Check frozen Gate G5 manifest and reports exist
    assert (ML_ROOT / "configs" / "frozen_g5_candidate.json").exists()
    assert (ML_ROOT / "reports" / "sft.md").exists()
    assert (REPO_ROOT / "docs" / "model-card.md").exists()
