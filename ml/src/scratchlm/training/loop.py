"""
Day037-Day040: FP32 reproducible training loop with AdamW weight-decay groups,
exact valid-token-weighted gradient accumulation, linear warmup + cosine decay,
gradient clipping, finite-value checks, validation evaluation, atomic checkpointing,
and pluggable ExperimentTracker integration.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from scratchlm.data.shards import DeterministicShardIterator, PackedBatch, PackedSplitShard
from scratchlm.model.gpt import ScratchGPT
from scratchlm.training.checkpoints import AtomicCheckpointManager, CheckpointMetadata, LineageHashes
from scratchlm.training.tracker import ExperimentTracker, InMemoryExperimentTracker


class NonFiniteLossError(RuntimeError):
    """Raised when loss or gradient norm becomes NaN or Inf during training."""


@dataclass(frozen=True)
class TrainingConfig:
    """Deterministic configuration for an FP32 pretraining run."""

    seed: int = 1337
    seq_len: int = 64
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    max_optimizer_steps: int = 10
    max_lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 2
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0
    eval_interval_steps: int = 5
    eval_batches: int = 2
    checkpoint_interval_steps: int = 5
    device: str = "cpu"

    def validate(self) -> None:
        if self.seq_len <= 0 or self.micro_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("seq_len, micro_batch_size, and gradient_accumulation_steps must be > 0")
        if self.max_optimizer_steps <= 0:
            raise ValueError("max_optimizer_steps must be > 0")
        if self.max_lr <= 0 or self.min_lr < 0 or self.min_lr > self.max_lr:
            raise ValueError("Invalid learning rate bounds")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be > 0")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    def config_sha256(self, model_config_dict: dict[str, Any]) -> str:
        combined = {
            "training_config": asdict(self),
            "model_config": model_config_dict,
        }
        raw = json.dumps(combined, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class StepMetrics:
    """Telemetry recorded at each completed optimizer step."""

    optimizer_step: int
    micro_step: int
    train_loss: float
    val_loss: float | None
    learning_rate: float
    grad_norm: float
    step_valid_tokens: int
    cumulative_valid_tokens: int
    step_time_ms: float
    tokens_per_sec: float
    rss_memory_mib: float


@dataclass(frozen=True)
class TrainingRunSummary:
    """Summary returned when `run_reproducible_training` finishes."""

    run_id: str
    initial_train_loss: float
    final_train_loss: float
    initial_val_loss: float
    final_val_loss: float
    completed_optimizer_steps: int
    cumulative_valid_tokens: int
    step_history: list[StepMetrics]
    last_checkpoint: CheckpointMetadata | None
    best_checkpoint: CheckpointMetadata | None
    logged_artifact_uris: list[str]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_adamw_optimizer(model: ScratchGPT, config: TrainingConfig) -> torch.optim.AdamW:
    """
    Create AdamW with explicit parameter groups:
    - 2D weight matrices (linear projections & embeddings) receive `config.weight_decay`.
    - 1D parameters (biases, LayerNorm gamma/beta) receive `weight_decay = 0.0`.
    Deduplicates shared/tied parameters (`token_embedding.weight` == `lm_head.weight`).
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    seen_ptrs: set[int] = set()

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        optim_groups,
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        eps=1e-8,
    )


def compute_cosine_lr_with_warmup(step: int, config: TrainingConfig) -> float:
    """
    Compute learning rate at 1-indexed `step` using linear warmup followed by cosine decay.
    """
    if step <= 0:
        raise ValueError("step must be >= 1")
    if config.warmup_steps > 0 and step <= config.warmup_steps:
        return config.max_lr * (step / config.warmup_steps)
    if step >= config.max_optimizer_steps:
        return config.min_lr
    decay_steps = max(1, config.max_optimizer_steps - config.warmup_steps)
    progress = (step - config.warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + cosine * (config.max_lr - config.min_lr)


@torch.no_grad()
def evaluate_validation_loss(
    model: ScratchGPT,
    val_shard: PackedSplitShard,
    *,
    seq_len: int,
    batch_size: int,
    max_batches: int,
    device: str = "cpu",
) -> float:
    """
    Deterministic validation loss weighted strictly by valid non-padded target tokens.
    Preserves the caller's model training/eval mode.
    """
    was_training = model.training
    model.eval()
    try:
        val_iter = DeterministicShardIterator(
            val_shard,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=99991,
            shuffle=False,
            device=device,
        )
        total_nll = 0.0
        total_valid = 0
        for _ in range(max_batches):
            batch = val_iter.next_batch()
            if batch.total_valid_targets == 0:
                continue
            _, loss = model(batch.input_ids, targets=batch.targets)
            if loss is None or not torch.isfinite(loss):
                raise NonFiniteLossError("Non-finite validation loss encountered")
            total_nll += float(loss.item()) * batch.total_valid_targets
            total_valid += batch.total_valid_targets
        if total_valid == 0:
            raise ValueError("Validation evaluation found 0 valid target tokens")
        return total_nll / total_valid
    finally:
        model.train(was_training)


def execute_accumulated_optimizer_step(
    model: ScratchGPT,
    optimizer: torch.optim.Optimizer,
    micro_batches: list[PackedBatch],
    *,
    learning_rate: float,
    grad_clip_norm: float,
) -> tuple[float, float, int]:
    r"""
    Execute one optimizer update across `micro_batches` with exact valid-token weighting.
    Because `masked_cross_entropy_loss` returns the mean over valid tokens in each microbatch,
    scaling microbatch $k$'s loss by $N_k / \sum_j N_j$ guarantees that accumulated gradients
    identically match a single concatenated batch of all microbatches.
    """
    total_valid = sum(b.total_valid_targets for b in micro_batches)
    if total_valid <= 0:
        raise ValueError("Cannot execute optimizer step with 0 total valid targets across microbatches")

    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    optimizer.zero_grad(set_to_none=True)
    weighted_loss_sum = 0.0

    for mb in micro_batches:
        if mb.total_valid_targets == 0:
            continue
        weight = mb.total_valid_targets / float(total_valid)
        _, mb_loss = model(mb.input_ids, targets=mb.targets)
        if mb_loss is None or not torch.isfinite(mb_loss):
            raise NonFiniteLossError(f"Non-finite microbatch loss: {mb_loss}")
        scaled_loss = mb_loss * weight
        scaled_loss.backward()
        weighted_loss_sum += float(mb_loss.item()) * weight

    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    grad_norm = float(grad_norm_tensor.item())
    if not math.isfinite(grad_norm):
        raise NonFiniteLossError(f"Non-finite gradient norm detected: {grad_norm}")

    optimizer.step()
    return weighted_loss_sum, grad_norm, total_valid


def run_reproducible_training(
    *,
    model: ScratchGPT,
    train_shard: PackedSplitShard,
    val_shard: PackedSplitShard,
    config: TrainingConfig,
    lineage: LineageHashes,
    checkpoint_dir: Path | None = None,
    tracker: ExperimentTracker | None = None,
    experiment_name: str = "scratchlm-phase4-pretraining",
    run_name: str = "tiny-bpe-pretrain",
    resume_from_checkpoint: bool = False,
    stop_after_step: int | None = None,
) -> TrainingRunSummary:
    """
    Execute or resume a bounded, reproducible FP32 training run (Day037-Day040).
    """
    config.validate()
    active_tracker: ExperimentTracker = tracker if tracker is not None else InMemoryExperimentTracker()
    model.to(config.device)
    model.train()

    optimizer = build_adamw_optimizer(model, config)
    train_iter = DeterministicShardIterator(
        train_shard,
        seq_len=config.seq_len,
        batch_size=config.micro_batch_size,
        seed=config.seed,
        shuffle=True,
        device=config.device,
    )

    ckpt_mgr = AtomicCheckpointManager(checkpoint_dir) if checkpoint_dir is not None else None
    start_step = 0
    micro_step = 0
    cumulative_valid_tokens = 0
    best_ckpt: CheckpointMetadata | None = None
    last_ckpt: CheckpointMetadata | None = None

    if resume_from_checkpoint:
        if ckpt_mgr is None:
            raise ValueError("checkpoint_dir is required when resume_from_checkpoint=True")
        restored_meta = ckpt_mgr.load_and_verify(
            "last",
            expected_lineage=lineage,
            model=model,
            optimizer=optimizer,
            train_iterator=train_iter,
            restore_rng=True,
        )
        start_step = restored_meta.optimizer_step
        micro_step = restored_meta.micro_step
        cumulative_valid_tokens = restored_meta.processed_valid_tokens
        last_ckpt = restored_meta
    else:
        seed_everything(config.seed)

    run_id = active_tracker.start_run(
        experiment_name=experiment_name,
        run_name=run_name,
        tags={
            "dataset_manifest_sha256": lineage.dataset_manifest_sha256,
            "splits_manifest_sha256": lineage.splits_manifest_sha256,
            "tokenizer_sha256": lineage.tokenizer_sha256,
            "config_sha256": lineage.config_sha256,
            "git_commit": lineage.git_commit,
            "resumed": str(resume_from_checkpoint).lower(),
        },
    )

    active_tracker.log_params(
        {
            **asdict(config),
            **{f"model_{k}": v for k, v in asdict(model.config).items()},
            **lineage.to_dict(),
        }
    )

    initial_val_loss = evaluate_validation_loss(
        model,
        val_shard,
        seq_len=config.seq_len,
        batch_size=config.micro_batch_size,
        max_batches=config.eval_batches,
        device=config.device,
    )
    if start_step == 0:
        active_tracker.log_metrics({"val_loss": initial_val_loss}, step=0)

    step_history: list[StepMetrics] = []
    proc = psutil.Process()
    target_final_step = stop_after_step if stop_after_step is not None else config.max_optimizer_steps

    for step in range(start_step + 1, target_final_step + 1):
        t0 = time.perf_counter()
        lr = compute_cosine_lr_with_warmup(step, config)
        micro_batches = [train_iter.next_batch() for _ in range(config.gradient_accumulation_steps)]
        micro_step += config.gradient_accumulation_steps

        train_loss, grad_norm, step_valid = execute_accumulated_optimizer_step(
            model,
            optimizer,
            micro_batches,
            learning_rate=lr,
            grad_clip_norm=config.grad_clip_norm,
        )
        cumulative_valid_tokens += step_valid
        elapsed_s = max(1e-6, time.perf_counter() - t0)
        step_time_ms = round(elapsed_s * 1000.0, 3)
        tokens_per_sec = round(step_valid / elapsed_s, 2)
        rss_mib = round(proc.memory_info().rss / (1024 * 1024), 2)

        val_loss: float | None = None
        if step % config.eval_interval_steps == 0 or step == target_final_step:
            val_loss = evaluate_validation_loss(
                model,
                val_shard,
                seq_len=config.seq_len,
                batch_size=config.micro_batch_size,
                max_batches=config.eval_batches,
                device=config.device,
            )

        metrics_payload: dict[str, float] = {
            "train_loss": train_loss,
            "learning_rate": lr,
            "grad_norm": grad_norm,
            "step_valid_tokens": float(step_valid),
            "cumulative_valid_tokens": float(cumulative_valid_tokens),
            "step_time_ms": step_time_ms,
            "tokens_per_sec": tokens_per_sec,
            "rss_memory_mib": rss_mib,
        }
        if val_loss is not None:
            metrics_payload["val_loss"] = val_loss
        active_tracker.log_metrics(metrics_payload, step=step)

        step_record = StepMetrics(
            optimizer_step=step,
            micro_step=micro_step,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=lr,
            grad_norm=grad_norm,
            step_valid_tokens=step_valid,
            cumulative_valid_tokens=cumulative_valid_tokens,
            step_time_ms=step_time_ms,
            tokens_per_sec=tokens_per_sec,
            rss_memory_mib=rss_mib,
        )
        step_history.append(step_record)

        if ckpt_mgr is not None and (step % config.checkpoint_interval_steps == 0 or step == target_final_step):
            sched_state = {"step": step, "lr": lr}
            last_ckpt = ckpt_mgr.save_checkpoint(
                slot_name="last",
                model=model,
                optimizer=optimizer,
                scheduler_state=sched_state,
                train_iterator=train_iter,
                optimizer_step=step,
                micro_step=micro_step,
                processed_valid_tokens=cumulative_valid_tokens,
                train_loss=train_loss,
                val_loss=val_loss,
                lineage=lineage,
            )
            if val_loss is not None and (ckpt_mgr.best_val_loss is None or val_loss <= ckpt_mgr.best_val_loss):
                best_ckpt = ckpt_mgr.save_checkpoint(
                    slot_name="best",
                    model=model,
                    optimizer=optimizer,
                    scheduler_state=sched_state,
                    train_iterator=train_iter,
                    optimizer_step=step,
                    micro_step=micro_step,
                    processed_valid_tokens=cumulative_valid_tokens,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lineage=lineage,
                )

    logged_uris: list[str] = []
    if ckpt_mgr is not None and ckpt_mgr.manifest_path.exists():
        logged_uris.append(active_tracker.log_artifact(ckpt_mgr.manifest_path, artifact_subpath="checkpoints"))
        best_pt = ckpt_mgr.checkpoint_dir / "best.pt"
        if best_pt.exists():
            logged_uris.append(active_tracker.log_artifact(best_pt, artifact_subpath="checkpoints"))

    active_tracker.end_run(status="FINISHED")

    initial_train = step_history[0].train_loss if step_history else 0.0
    final_train = step_history[-1].train_loss if step_history else 0.0
    final_val = (
        step_history[-1].val_loss
        if (step_history and step_history[-1].val_loss is not None)
        else initial_val_loss
    )

    return TrainingRunSummary(
        run_id=run_id,
        initial_train_loss=initial_train,
        final_train_loss=final_train,
        initial_val_loss=initial_val_loss,
        final_val_loss=final_val,
        completed_optimizer_steps=target_final_step,
        cumulative_valid_tokens=cumulative_valid_tokens,
        step_history=step_history,
        last_checkpoint=last_ckpt,
        best_checkpoint=best_ckpt,
        logged_artifact_uris=logged_uris,
    )
