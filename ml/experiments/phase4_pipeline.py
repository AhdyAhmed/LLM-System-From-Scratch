"""
Day031-Day040 End-to-End Pipeline:
1. Bounded corpus ingestion, normalization, deduplication, and `ml/data/manifest.json`.
2. Document-level split freezing (`ml/data/splits/`) and leakage verification.
3. Train-only deterministic BPE tokenizer (`ml/artifacts/tokenizer/bpe_tiny_v1.json`) and `ml/reports/tokenizer.md`.
4. Memory-mapped EOS-packed token shards (`ml/data/shards/`).
5. Reproducible FP32 training, atomic checkpointing, resume verification, and MLflow (`mlflow-artifacts:/`) lineage logging.
"""

from __future__ import annotations

import json
from pathlib import Path

from scratchlm.config import ModelConfig
from scratchlm.data.ingestion import (
    build_default_corpus_manifest,
    generate_curated_story_stream,
    ingest_bounded_corpus,
)
from scratchlm.data.manifest import StorageBudget
from scratchlm.data.shards import PackedSplitShard, build_packed_split_shard
from scratchlm.data.splits import assign_and_freeze_splits, load_split_records
from scratchlm.model.gpt import ScratchGPT
from scratchlm.tokenizer.bpe_tokenizer import BPETokenizer
from scratchlm.training.checkpoints import LineageHashes
from scratchlm.training.loop import TrainingConfig, run_reproducible_training
from scratchlm.training.tracker import (
    InMemoryExperimentTracker,
    SpoolingMlflowRestTracker,
    validate_logical_artifact_uri,
)


def write_tokenizer_report(
    report_path: Path,
    tokenizer: BPETokenizer,
    stats_train: object,
    stats_val: object,
    tokenizer_sha256: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    top_merges = tokenizer.merges[:10]
    merge_rows = []
    for m in top_merges:
        decoded_repr = bytes.fromhex(m.merged_bytes_hex).decode("utf-8", errors="replace").replace("\n", "\\n")
        merge_rows.append(
            f"| {m.rank} | `({m.token_id_a}, {m.token_id_b})` | `{m.new_token_id}` | {m.training_frequency} | `{m.merged_bytes_hex}` | `{decoded_repr}` |"
        )
    merge_table = "\n".join(merge_rows)

    content = f"""# Day035: Educational Byte-Level BPE Tokenizer Report

## 1. Tokenizer Configuration & Provenance
- **Tokenizer Type:** Educational Byte-Level BPE (`byte_level_bpe`)
- **Artifact Path:** [`ml/artifacts/tokenizer/bpe_tiny_v1.json`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/artifacts/tokenizer/bpe_tiny_v1.json)
- **Tokenizer SHA-256:** `{tokenizer_sha256}`
- **Training Dataset ID:** `{tokenizer.training_dataset_id}`
- **Training Split SHA-256:** `{tokenizer.training_split_sha256}`
- **Base Byte Vocabulary:** `256` (`0..255`)
- **Reserved Special Tokens:** `8` (`256..263`: `<|pad|>`, `<|eos|>`, `<|bos|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|context|>`, `<|end_turn|>`)
- **Learned BPE Merges:** `{len(tokenizer.merges)}`
- **Total Vocabulary Size:** `{tokenizer.vocab_size}`
- **Deterministic Tie-Breaking Rule:** Lexicographically smallest `(token_id_a, token_id_b)` when adjacent pair frequencies tie.

## 2. Compression Comparison: `BPETokenizer` vs. `ByteTokenizer`

| Split Sample | Documents | Total UTF-8 Bytes | Byte Tokens (`V=264`) | BPE Tokens (`V={tokenizer.vocab_size}`) | Bytes / BPE Token | Token Count Ratio (`BPE / Byte`) | Encode Time (ms) | Decode Time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` (`96` docs) | {stats_train.sample_documents} | {stats_train.total_utf8_bytes} | {stats_train.byte_token_count} | {stats_train.bpe_token_count} | **{stats_train.bytes_per_bpe_token:.4f}** | **{stats_train.token_reduction_ratio:.4f}** | {stats_train.encode_time_ms:.3f} | {stats_train.decode_time_ms:.3f} |
| `val` (`12` docs, held-out) | {stats_val.sample_documents} | {stats_val.total_utf8_bytes} | {stats_val.byte_token_count} | {stats_val.bpe_token_count} | **{stats_val.bytes_per_bpe_token:.4f}** | **{stats_val.token_reduction_ratio:.4f}** | {stats_val.encode_time_ms:.3f} | {stats_val.decode_time_ms:.3f} |

## 3. First 10 Learned Merges (Rank 0..9)

| Rank | Constituent Pair `(a, b)` | New Token ID | Training Frequency | Merged Hex | Decoded Subword |
| ---: | --- | ---: | ---: | --- | --- |
{merge_table}

## 4. Known Limitations of the Educational Implementation
1. **Algorithmic Complexity:** Pair counting iterates over pre-tokenized unique word chunks in pure Python (`O(M * U)` where `M` is the number of merges and `U` is unique chunk length sum). Suitable for small bounded corpora (`<= 50 MiB`), not multi-gigabyte corpora without a C++/Rust priority-queue index.
2. **Domain Specialization:** Merges learned on English narrative stories compress English story vocabulary to `{stats_val.bytes_per_bpe_token:.2f}` bytes/token while falling back losslessly to raw UTF-8 byte tokens (`1.00` byte/token) for unseen scripts such as Arabic or emoji.
3. **Held-Out Evaluation Hygiene:** Validation and sealed test splits are strictly excluded via [`assert_train_only_texts`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/data/splits.py#L192-L225).
"""
    report_path.write_text(content, encoding="utf-8")


def run_phase4_pipeline(
    ml_root: Path | None = None,
    mlflow_uri: str | None = "http://127.0.0.1:5001",
) -> dict:
    root = ml_root or Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    splits_dir = data_dir / "splits"
    shards_dir = data_dir / "shards"
    artifacts_dir = root / "artifacts"
    tokenizer_path = artifacts_dir / "tokenizer" / "bpe_tiny_v1.json"
    ckpt_dir = artifacts_dir / "checkpoints" / "phase4_run"
    spool_dir = artifacts_dir / "spool"
    reports_dir = root / "reports"

    # --- Day031 & Day032: Bounded Ingestion & Corpus Manifest ---
    budget = StorageBudget(
        max_raw_bytes=10 * 1024 * 1024,
        max_processed_bytes=10 * 1024 * 1024,
        max_records=5000,
        max_cache_bytes=10 * 1024 * 1024,
        min_record_chars=40,
        max_record_chars=2048,
    )
    accepted, audit_report = ingest_bounded_corpus(generate_curated_story_stream(), budget)
    corpus_manifest = build_default_corpus_manifest(accepted, audit_report, budget)
    dataset_manifest_sha = corpus_manifest.save(data_dir / "manifest.json")
    (data_dir / "ingestion_audit.json").write_text(
        json.dumps(audit_report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # --- Day033: Freeze Train / Val / Test Splits ---
    _, splits_manifest = assign_and_freeze_splits(
        accepted,
        dataset_id=corpus_manifest.dataset_id,
        corpus_sha256=corpus_manifest.corpus_sha256,
        output_dir=splits_dir,
        split_seed=20261016,
    )
    splits_manifest_sha = splits_manifest.manifest_sha256()

    train_records = load_split_records(splits_dir, "train")
    val_records = load_split_records(splits_dir, "val")

    # --- Day034 & Day035: Train Deterministic Byte-Level BPE Tokenizer on Train Only ---
    bpe_tokenizer = BPETokenizer.train_from_records(
        train_records,
        target_vocab_size=512,
        min_frequency=2,
        splits_manifest=splits_manifest,
    )
    tokenizer_sha = bpe_tokenizer.save(tokenizer_path)

    stats_train = bpe_tokenizer.compare_compression([r.text for r in train_records])
    stats_val = bpe_tokenizer.compare_compression([r.text for r in val_records])
    write_tokenizer_report(
        reports_dir / "tokenizer.md",
        bpe_tokenizer,
        stats_train,
        stats_val,
        tokenizer_sha,
    )

    # --- Day036: Build Memory-Mapped EOS-Packed Token Shards ---
    build_packed_split_shard(
        train_records,
        split_name="train",
        dataset_id=corpus_manifest.dataset_id,
        split_file_sha256=splits_manifest.splits["train"].file_sha256,
        tokenizer=bpe_tokenizer,
        output_dir=shards_dir,
        allowed_doc_ids=set(splits_manifest.splits["train"].doc_ids),
    )
    build_packed_split_shard(
        val_records,
        split_name="val",
        dataset_id=corpus_manifest.dataset_id,
        split_file_sha256=splits_manifest.splits["val"].file_sha256,
        tokenizer=bpe_tokenizer,
        output_dir=shards_dir,
        allowed_doc_ids=set(splits_manifest.splits["val"].doc_ids),
    )

    train_shard = PackedSplitShard(shards_dir, "train")
    val_shard = PackedSplitShard(shards_dir, "val")

    # --- Day037-Day040: Reproducible FP32 Training, Checkpointing & MLflow Tracking ---
    gpt_cfg = ModelConfig(
        profile_name="tiny-bpe-512",
        vocab_size=bpe_tokenizer.vocab_size,
        max_seq_len=128,
        n_layers=4,
        n_heads=4,
        d_model=128,
        d_ff=512,
        dropout=0.0,
        use_bias=True,
        tie_word_embeddings=True,
        layer_norm_eps=1e-5,
    )
    train_cfg = TrainingConfig(
        seed=1337,
        seq_len=64,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        max_optimizer_steps=12,
        max_lr=3e-3,
        min_lr=3e-4,
        warmup_steps=2,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        eval_interval_steps=4,
        eval_batches=2,
        checkpoint_interval_steps=4,
        device="cpu",
    )

    lineage = LineageHashes(
        dataset_manifest_sha256=dataset_manifest_sha,
        splits_manifest_sha256=splits_manifest_sha,
        tokenizer_sha256=tokenizer_sha,
        config_sha256=train_cfg.config_sha256(gpt_cfg.__dict__),
        git_commit="phase4-g4-verified",
    )

    if mlflow_uri is not None:
        tracker = SpoolingMlflowRestTracker(tracking_uri=mlflow_uri, spool_dir=spool_dir)
    else:
        tracker = InMemoryExperimentTracker()

    model = ScratchGPT(gpt_cfg)
    summary = run_reproducible_training(
        model=model,
        train_shard=train_shard,
        val_shard=val_shard,
        config=train_cfg,
        lineage=lineage,
        checkpoint_dir=ckpt_dir,
        tracker=tracker,
        experiment_name="scratchlm-phase4-gate-g4",
        run_name="phase4-tiny-bpe-fp32",
    )

    for uri in summary.logged_artifact_uris:
        validate_logical_artifact_uri(uri)

    return {
        "dataset_manifest_sha256": dataset_manifest_sha,
        "splits_manifest_sha256": splits_manifest_sha,
        "tokenizer_sha256": tokenizer_sha,
        "vocab_size": bpe_tokenizer.vocab_size,
        "bpe_bytes_per_token_val": stats_val.bytes_per_bpe_token,
        "train_shard_tokens": train_shard.total_tokens,
        "val_shard_tokens": val_shard.total_tokens,
        "run_id": summary.run_id,
        "initial_train_loss": summary.initial_train_loss,
        "final_train_loss": summary.final_train_loss,
        "initial_val_loss": summary.initial_val_loss,
        "final_val_loss": summary.final_val_loss,
        "logged_artifact_uris": summary.logged_artifact_uris,
    }


if __name__ == "__main__":
    result = run_phase4_pipeline()
    print(json.dumps(result, indent=2))
