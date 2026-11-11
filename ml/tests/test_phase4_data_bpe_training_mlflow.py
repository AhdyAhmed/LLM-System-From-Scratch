"""
Phase 4 (Day031-Day040, Gate G4) Verification Suite:
Covers corpus manifest & budgets, streaming normalization & deduplication, frozen split disjointness
& sealed test guards, deterministic byte-level BPE with lexicographical tie-breaking, BPE compression
& streaming UTF-8 decode, memory-mapped EOS-packed shards with source tracing, FP32 AdamW training
with exact gradient accumulation, atomic checkpointing & deterministic resume, and MLflow tracking
over `mlflow-artifacts:/` with bounded local spooling during tracking-service outages.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import torch

from scratchlm.config import ModelConfig
from scratchlm.data.ingestion import (
    CleanedCorpusRecord,
    RawCorpusRecord,
    generate_curated_story_stream,
    ingest_bounded_corpus,
    normalize_document_text,
)
from scratchlm.data.manifest import CorpusManifest, StorageBudget
from scratchlm.data.shards import (
    DeterministicShardIterator,
    PackedBatch,
    PackedSplitShard,
    build_packed_split_shard,
)
from scratchlm.data.splits import (
    SplitsManifest,
    assert_train_only_texts,
    assign_and_freeze_splits,
    load_split_records,
    verify_split_disjointness,
)
from scratchlm.model.gpt import ScratchGPT
from scratchlm.tokenizer.bpe_tokenizer import (
    BPETokenizer,
    count_adjacent_pairs,
    select_best_pair,
)
from scratchlm.tokenizer.byte_tokenizer import EOS_TOKEN_ID, IGNORE_INDEX, PAD_TOKEN_ID
from scratchlm.training.checkpoints import (
    AtomicCheckpointManager,
    CheckpointIntegrityError,
    LineageHashes,
)
from scratchlm.training.loop import (
    TrainingConfig,
    build_adamw_optimizer,
    compute_cosine_lr_with_warmup,
    execute_accumulated_optimizer_step,
    run_reproducible_training,
    seed_everything,
)
from scratchlm.training.tracker import (
    InMemoryExperimentTracker,
    SpoolingMlflowRestTracker,
    TrackingSpoolOverflowError,
    validate_logical_artifact_uri,
)

ML_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _ensure_phase4_data_and_shards() -> None:
    """Ensure deterministic shards exist on disk even in a clean git clone where *.bin is gitignored."""
    tok_path = ML_ROOT / "artifacts" / "tokenizer" / "bpe_tiny_v1.json"
    train_bin = ML_ROOT / "data" / "shards" / "train.bin"
    val_bin = ML_ROOT / "data" / "shards" / "val.bin"
    if not (tok_path.exists() and train_bin.exists() and val_bin.exists()):
        from experiments.phase4_pipeline import run_phase4_pipeline

        run_phase4_pipeline(ml_root=ML_ROOT, mlflow_uri=None)


def test_day031_corpus_manifest_and_storage_budget(tmp_path: Path) -> None:
    manifest_path = ML_ROOT / "data" / "manifest.json"
    assert manifest_path.exists(), "ml/data/manifest.json must exist"
    manifest = CorpusManifest.load(manifest_path)
    assert manifest.dataset_id == "scratch-curated-tinystories-v1"
    assert manifest.license_spdx == "CDLA-Sharing-1.0"
    assert manifest.exclude_private_chat_and_uploads is True
    assert manifest.accepted_record_count == 120
    assert manifest.duplicate_record_count == 2
    assert manifest.rejected_record_count == 3

    # Verify StorageBudget rejects > 2 GiB caps
    bad_budget = StorageBudget(
        max_raw_bytes=3 * 1024 * 1024 * 1024,
        max_processed_bytes=1024,
        max_records=10,
        max_cache_bytes=1024,
        min_record_chars=10,
        max_record_chars=100,
    )
    with pytest.raises(ValueError, match="max_raw_bytes"):
        bad_budget.validate()


def test_day032_streaming_ingestion_normalization_and_deduplication() -> None:
    raw_dirty = "  Hello\x00\x07   world!\r\n\r\n\r\nSecond   line.\t "
    normalized = normalize_document_text(raw_dirty)
    assert normalized == "Hello world!\n\nSecond line."

    budget = StorageBudget(
        max_raw_bytes=10 * 1024 * 1024,
        max_processed_bytes=10 * 1024 * 1024,
        max_records=5000,
        max_cache_bytes=10 * 1024 * 1024,
        min_record_chars=40,
        max_record_chars=2048,
    )
    accepted, report = ingest_bounded_corpus(generate_curated_story_stream(), budget)
    assert len(accepted) == 120
    assert report.duplicate_records == 2
    assert report.rejected_records == 3
    assert report.stopped_by_cap is False

    # Verify privacy: rejected/duplicate audit entries contain doc_ids, never raw private text
    report_json = str(report.to_dict())
    assert "private chat message appeared" not in report_json
    assert "story-reject-private-0001" in report.rejected_doc_ids_by_reason["private_or_disallowed_source"]

    # Verify dry-run stops deterministically at max_records cap
    capped_budget = StorageBudget(
        max_raw_bytes=10 * 1024 * 1024,
        max_processed_bytes=10 * 1024 * 1024,
        max_records=15,
        max_cache_bytes=10 * 1024 * 1024,
        min_record_chars=40,
        max_record_chars=2048,
    )
    capped_accepted, capped_report = ingest_bounded_corpus(generate_curated_story_stream(), capped_budget)
    assert len(capped_accepted) == 15
    assert capped_report.stopped_by_cap is True
    assert capped_report.stop_reason == "max_records_reached"


def test_day033_frozen_splits_disjointness_and_sealed_test_guard() -> None:
    splits_dir = ML_ROOT / "data" / "splits"
    manifest = SplitsManifest.load(splits_dir / "splits_manifest.json")
    assert manifest.test_split_sealed is True
    assert manifest.splits["train"].record_count == 96
    assert manifest.splits["val"].record_count == 12
    assert manifest.splits["test"].record_count == 12

    train_recs = load_split_records(splits_dir, "train")
    val_recs = load_split_records(splits_dir, "val")

    # Loading sealed test split without explicit override must fail
    with pytest.raises(PermissionError, match="sealed"):
        load_split_records(splits_dir, "test")

    test_recs = load_split_records(splits_dir, "test", allow_sealed_test=True)
    verify_split_disjointness(train_recs, val_recs, test_recs)

    # Injecting a near-duplicate across train and val must fail verify_split_disjointness
    leaked_val = list(val_recs) + [
        CleanedCorpusRecord(
            doc_id="leaked-doc-9999",
            text=train_recs[0].text.upper() + "!!!",
            utf8_bytes=len(train_recs[0].text),
            exact_sha256="0" * 64,
            canonical_dedup_hash="0" * 64,
            author_tag="test",
        )
    ]
    with pytest.raises(ValueError, match="normalized near-duplicate"):
        verify_split_disjointness(train_recs, leaked_val, test_recs)

    # Passing a val or test record into assert_train_only_texts must raise PermissionError
    with pytest.raises(PermissionError, match="Data leakage blocked"):
        assert_train_only_texts([train_recs[0], val_recs[0]], manifest)
    with pytest.raises(PermissionError, match="Data leakage blocked"):
        assert_train_only_texts([test_recs[0].text], manifest)


def test_day034_deterministic_bpe_merges_and_tie_breaking() -> None:
    # Tie-breaking check: pairs (10, 20), (5, 99), and (5, 12) all have frequency 4 -> must pick (5, 12)
    counts = {(10, 20): 4, (5, 99): 4, (5, 12): 4, (1, 2): 3}
    chosen = select_best_pair(counts, min_frequency=2)
    assert chosen == ((5, 12), 4)

    splits_dir = ML_ROOT / "data" / "splits"
    manifest = SplitsManifest.load(splits_dir / "splits_manifest.json")
    train_recs = load_split_records(splits_dir, "train")
    val_recs = load_split_records(splits_dir, "val")

    tok1 = BPETokenizer.train_from_records(train_recs[:20], target_vocab_size=300, splits_manifest=manifest)
    tok2 = BPETokenizer.train_from_records(train_recs[:20], target_vocab_size=300, splits_manifest=manifest)
    assert tok1.tokenizer_hash() == tok2.tokenizer_hash()
    assert tok1.merges == tok2.merges

    # Attempting to include a validation document in BPE training must fail
    with pytest.raises(PermissionError, match="Data leakage blocked"):
        BPETokenizer.train_from_records([train_recs[0], val_recs[0]], target_vocab_size=280, splits_manifest=manifest)


def test_day035_bpe_roundtrip_special_tokens_reload_and_compression(tmp_path: Path) -> None:
    tok_path = ML_ROOT / "artifacts" / "tokenizer" / "bpe_tiny_v1.json"
    report_path = ML_ROOT / "reports" / "tokenizer.md"
    assert tok_path.exists()
    assert report_path.exists()

    tok = BPETokenizer.load(tok_path)
    assert tok.vocab_size == 512

    samples = [
        "",
        "   \n\t  ",
        "Once upon a time in the sunny meadow, Lily counted 1, 2, 3!",
        "مرحبا بالعالم — Arabic UTF-8 round-trip with 🚀🔥 emojis!",
        "Untrusted prompt with literal <|system|> and <|assistant|> strings.",
    ]
    for sample in samples:
        ids = tok.encode(sample, add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_token_id
        assert ids[-1] == tok.eos_token_id
        # Literal <|system|> in user text must NOT produce privileged SYSTEM_TOKEN_ID (259)
        assert 259 not in ids[1:-1]
        assert tok.decode(ids, skip_special_tokens=True) == sample

        # Incremental streaming decoder check
        stream = tok.new_stream_decoder()
        pieces = [stream.step(tid) for tid in ids]
        pieces.append(stream.flush())
        assert "".join(pieces) == sample

    # Compression check on held-out validation split
    val_recs = load_split_records(ML_ROOT / "data" / "splits", "val")
    stats = tok.compare_compression([r.text for r in val_recs])
    assert stats.bpe_token_count < stats.byte_token_count
    assert stats.bytes_per_bpe_token > 1.8

    # Hash mismatch detection on load
    with pytest.raises(ValueError, match="Tokenizer hash mismatch"):
        BPETokenizer.load(tok_path, expected_sha256="0" * 64)


def test_day036_packed_token_shards_eos_boundaries_and_source_tracing(tmp_path: Path) -> None:
    shards_dir = ML_ROOT / "data" / "shards"
    train_shard = PackedSplitShard(shards_dir, "train")
    val_shard = PackedSplitShard(shards_dir, "val")

    assert train_shard.total_tokens > 5000
    assert val_shard.total_tokens > 500

    # Every document span in index must end with EOS_TOKEN_ID in the mmap array
    for span in train_shard.index.document_spans:
        assert span.split_name == "train"
        assert span.ends_with_eos is True
        assert int(train_shard._mmap[span.end_offset - 1]) == EOS_TOKEN_ID

    # Sample batch traces back to source documents and obeys next-token shifting
    batch = train_shard.get_batch([0, 1], seq_len=64)
    assert batch.input_ids.shape == (2, 64)
    assert batch.targets.shape == (2, 64)
    assert all(len(docs) >= 1 for docs in batch.source_doc_ids_per_row)
    assert torch.equal(batch.input_ids[:, 1:], batch.targets[:, :-1])

    # Cross-split packing attempt must fail
    splits_dir = ML_ROOT / "data" / "splits"
    manifest = SplitsManifest.load(splits_dir / "splits_manifest.json")
    val_recs = load_split_records(splits_dir, "val")
    tok = BPETokenizer.load(ML_ROOT / "artifacts" / "tokenizer" / "bpe_tiny_v1.json")
    with pytest.raises(PermissionError, match="Cross-split packing violation"):
        build_packed_split_shard(
            val_recs,
            split_name="train",
            dataset_id=manifest.dataset_id,
            split_file_sha256=manifest.splits["train"].file_sha256,
            tokenizer=tok,
            output_dir=tmp_path,
            allowed_doc_ids=set(manifest.splits["train"].doc_ids),
        )


def test_day037_fp32_training_loop_adamw_accumulation_and_pluggable_tracker() -> None:
    cfg = ModelConfig(
        profile_name="tiny-test",
        vocab_size=512,
        max_seq_len=64,
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_ff=256,
        dropout=0.0,
        use_bias=True,
        tie_word_embeddings=True,
        layer_norm_eps=1e-5,
    )
    train_cfg = TrainingConfig(
        seed=42,
        seq_len=32,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        max_optimizer_steps=6,
        max_lr=3e-3,
        min_lr=3e-4,
        warmup_steps=2,
        weight_decay=0.01,
        grad_clip_norm=10.0,
    )

    # Verify cosine schedule warmup and decay
    assert abs(compute_cosine_lr_with_warmup(1, train_cfg) - 1.5e-3) < 1e-9
    assert abs(compute_cosine_lr_with_warmup(2, train_cfg) - 3.0e-3) < 1e-9
    assert abs(compute_cosine_lr_with_warmup(6, train_cfg) - 3.0e-4) < 1e-9

    # Verify gradient accumulation equivalence: 2 microbatches of size 2 == 1 full batch of size 4
    train_shard = PackedSplitShard(ML_ROOT / "data" / "shards", "train")
    mb1 = train_shard.get_batch([0, 1], seq_len=32)
    mb2 = train_shard.get_batch([2, 3], seq_len=32)
    full_batch = train_shard.get_batch([0, 1, 2, 3], seq_len=32)

    model_accum = ScratchGPT(cfg, init_seed=99)
    model_single = ScratchGPT(cfg, init_seed=99)
    opt_accum = build_adamw_optimizer(model_accum, train_cfg)
    opt_single = build_adamw_optimizer(model_single, train_cfg)

    loss_accum, gn_accum, valid_accum = execute_accumulated_optimizer_step(
        model_accum, opt_accum, [mb1, mb2], learning_rate=1e-3, grad_clip_norm=100.0
    )
    loss_single, gn_single, valid_single = execute_accumulated_optimizer_step(
        model_single, opt_single, [full_batch], learning_rate=1e-3, grad_clip_norm=100.0
    )

    assert valid_accum == valid_single
    assert abs(loss_accum - loss_single) < 1e-6
    assert abs(gn_accum - gn_single) < 1e-5
    for p1, p2 in zip(model_accum.parameters(), model_single.parameters()):
        assert p1.grad is not None and p2.grad is not None
        assert torch.allclose(p1.grad, p2.grad, atol=1e-6)
        assert torch.allclose(p1, p2, atol=1e-4)

    # Verify masked tokens (IGNORE_INDEX = -100) do not affect loss
    masked_targets = full_batch.targets.clone()
    masked_targets[:, 16:] = IGNORE_INDEX
    _, loss_half_a = model_single(full_batch.input_ids, targets=masked_targets)
    altered_inputs = full_batch.input_ids.clone()
    altered_inputs[:, 17:] = PAD_TOKEN_ID
    _, loss_half_b = model_single(altered_inputs, targets=masked_targets)
    assert loss_half_a is not None and loss_half_b is not None
    assert torch.allclose(loss_half_a, loss_half_b, atol=1e-6)


def test_day038_atomic_checkpoint_resume_and_corruption_rejection(tmp_path: Path) -> None:
    cfg = ModelConfig(
        profile_name="tiny-resume-test",
        vocab_size=512,
        max_seq_len=64,
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_ff=256,
        dropout=0.0,
        use_bias=True,
        tie_word_embeddings=True,
        layer_norm_eps=1e-5,
    )
    train_cfg = TrainingConfig(
        seed=2026,
        seq_len=32,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        max_optimizer_steps=6,
        max_lr=3e-3,
        min_lr=3e-4,
        warmup_steps=2,
        eval_interval_steps=3,
        eval_batches=2,
        checkpoint_interval_steps=3,
    )
    lineage = LineageHashes(
        dataset_manifest_sha256="a" * 64,
        splits_manifest_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        config_sha256=train_cfg.config_sha256(cfg.__dict__),
    )
    train_shard = PackedSplitShard(ML_ROOT / "data" / "shards", "train")
    val_shard = PackedSplitShard(ML_ROOT / "data" / "shards", "val")

    # 1. Run uninterrupted for 6 steps
    ckpt_uninterrupted = tmp_path / "uninterrupted"
    model_full = ScratchGPT(cfg, init_seed=77)
    summary_full = run_reproducible_training(
        model=model_full,
        train_shard=train_shard,
        val_shard=val_shard,
        config=train_cfg,
        lineage=lineage,
        checkpoint_dir=ckpt_uninterrupted,
    )

    # 2. Run 3 steps, stop, then resume in a fresh model instance for steps 4..6
    ckpt_resumed = tmp_path / "resumed"
    model_part1 = ScratchGPT(cfg, init_seed=77)
    run_reproducible_training(
        model=model_part1,
        train_shard=train_shard,
        val_shard=val_shard,
        config=train_cfg,
        lineage=lineage,
        checkpoint_dir=ckpt_resumed,
        stop_after_step=3,
    )

    model_part2 = ScratchGPT(cfg, init_seed=1)  # Different initial weights, overwritten by resume
    summary_resumed = run_reproducible_training(
        model=model_part2,
        train_shard=train_shard,
        val_shard=val_shard,
        config=train_cfg,
        lineage=lineage,
        checkpoint_dir=ckpt_resumed,
        resume_from_checkpoint=True,
    )

    assert abs(summary_full.final_train_loss - summary_resumed.final_train_loss) < 1e-6
    assert abs(summary_full.final_val_loss - summary_resumed.final_val_loss) < 1e-6
    for p_full, p_res in zip(model_full.parameters(), model_part2.parameters()):
        assert torch.allclose(p_full, p_res, atol=1e-6)

    # Verify bounded retention: only last.pt, best.pt, and checkpoint_manifest.json exist
    saved_files = {p.name for p in ckpt_resumed.iterdir()}
    assert saved_files == {"last.pt", "best.pt", "checkpoint_manifest.json"}

    # Verify incompatible lineage hash fails load_and_verify
    mgr = AtomicCheckpointManager(ckpt_resumed)
    bad_lineage = LineageHashes(
        dataset_manifest_sha256="f" * 64,
        splits_manifest_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        config_sha256=lineage.config_sha256,
    )
    with pytest.raises(CheckpointIntegrityError, match="Dataset manifest hash mismatch"):
        mgr.load_and_verify("last", expected_lineage=bad_lineage, model=model_part2)

    # Verify corrupted checkpoint file fails checksum verification
    last_pt = ckpt_resumed / "last.pt"
    raw = bytearray(last_pt.read_bytes())
    raw[-1] ^= 0xFF
    last_pt.write_bytes(bytes(raw))
    with pytest.raises(CheckpointIntegrityError, match="Corrupted checkpoint"):
        mgr.load_and_verify("last", expected_lineage=lineage, model=model_part2)


def test_day039_day040_mlflow_lineage_logical_uris_and_outage_spool_recovery(tmp_path: Path) -> None:
    # 1. Logical URI validation rules
    validate_logical_artifact_uri("mlflow-artifacts:/1/run123/artifacts/checkpoints/best.pt")
    with pytest.raises(ValueError, match="must start with 'mlflow-artifacts:/'"):
        validate_logical_artifact_uri("E:\\llm-data\\artifacts\\best.pt")
    with pytest.raises(ValueError, match="must start with 'mlflow-artifacts:/'"):
        validate_logical_artifact_uri("/opt/mlflow/artifacts/best.pt")

    # 2. Outage spool overflow protection
    tiny_spool_dir = tmp_path / "tiny_spool"
    tracker_overflow = SpoolingMlflowRestTracker(
        tracking_uri="http://127.0.0.1:59999",  # unreachable port
        spool_dir=tiny_spool_dir,
        max_spool_bytes=250,
        timeout_seconds=0.2,
    )
    tracker_overflow.run_id = "run-overflow-test"
    tracker_overflow.simulate_outage = True
    tracker_overflow.log_metrics({"loss": 4.5}, step=1)
    assert tracker_overflow.pending_spool_count() == 1
    with pytest.raises(TrackingSpoolOverflowError):
        for s in range(2, 10):
            tracker_overflow.log_metrics({"loss": 4.0 / s}, step=s)

    # 3. Live MLflow server integration: run creation, outage spooling, flush reconciliation, and artifact round-trip
    mlflow_url = "http://127.0.0.1:5001"
    try:
        resp = httpx.get(f"{mlflow_url}/health", timeout=2.0)
        mlflow_online = resp.status_code == 200
    except httpx.HTTPError:
        mlflow_online = False

    assert mlflow_online, "MLflow tracking server must be online at http://127.0.0.1:5001 for Phase 4 Gate G4"

    spool_dir = tmp_path / "live_spool"
    tracker = SpoolingMlflowRestTracker(tracking_uri=mlflow_url, spool_dir=spool_dir)
    run_id = tracker.start_run(
        experiment_name="scratchlm-phase4-pytest",
        run_name="g4-outage-recovery-test",
        tags={"gate": "G4", "tokenizer_sha256": "verified"},
    )
    assert tracker.run_artifact_uri is not None
    validate_logical_artifact_uri(tracker.run_artifact_uri)

    # Simulate brief tracking outage during training steps 1..2
    tracker.simulate_outage = True
    tracker.log_params({"seq_len": 64, "vocab_size": 512})
    tracker.log_metrics({"train_loss": 5.25, "val_loss": 5.10}, step=1)
    sample_artifact = tmp_path / "lineage_probe.json"
    sample_artifact.write_text('{"probe": "cross-os-mlflow-artifacts"}\n', encoding="utf-8")
    logical_uri = tracker.log_artifact(sample_artifact, artifact_subpath="lineage")
    validate_logical_artifact_uri(logical_uri)
    assert tracker.pending_spool_count() == 3

    # Restore tracking connectivity and end run (which flushes the spool)
    tracker.simulate_outage = False
    flushed = tracker.flush_spool()
    assert flushed == 3
    assert tracker.pending_spool_count() == 0
    tracker.end_run(status="FINISHED")

    # Verify run state and download artifact via mlflow-artifacts:/ URI proxy
    run_data = tracker.get_run(run_id)
    assert run_data["info"]["status"] == "FINISHED"
    downloaded_bytes = tracker.download_artifact_bytes(logical_uri)
    assert downloaded_bytes == sample_artifact.read_bytes()
