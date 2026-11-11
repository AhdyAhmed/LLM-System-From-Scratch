"""
Day038: Atomic checkpoint persistence, bounded retention (best.pt + last.pt),
full RNG/iterator state recovery, and lineage/checksum verification.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scratchlm.config import ModelConfig
from scratchlm.data.shards import DeterministicShardIterator
from scratchlm.model.gpt import ScratchGPT


class CheckpointIntegrityError(RuntimeError):
    """Raised when a checkpoint file fails checksum, schema, or lineage hash verification."""


@dataclass(frozen=True)
class LineageHashes:
    """Immutable cryptographic lineage hashes bound into every checkpoint."""

    dataset_manifest_sha256: str
    splits_manifest_sha256: str
    tokenizer_sha256: str
    config_sha256: str
    git_commit: str = "local-phase4-head"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointMetadata:
    """Immutable manifest entry for a saved checkpoint file."""

    schema_version: str
    slot_name: str  # "last" or "best"
    filename: str
    file_sha256: str
    file_bytes: int
    optimizer_step: int
    micro_step: int
    processed_valid_tokens: int
    train_loss: float
    val_loss: float | None
    lineage: LineageHashes
    model_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_rng_states() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch (CPU + optional CUDA) RNG states."""
    state: dict[str, Any] = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_states(rng_state: dict[str, Any]) -> None:
    """Restore Python, NumPy, and PyTorch RNG states for deterministic continuation."""
    random.setstate(rng_state["python_rng"])
    np.random.set_state(rng_state["numpy_rng"])
    cpu_rng = rng_state["torch_cpu_rng"]
    if not isinstance(cpu_rng, torch.ByteTensor):
        cpu_rng = torch.tensor(cpu_rng, dtype=torch.uint8)
    torch.set_rng_state(cpu_rng)
    if torch.cuda.is_available() and "torch_cuda_rng_all" in rng_state:
        cuda_states = [
            s if isinstance(s, torch.ByteTensor) else torch.tensor(s, dtype=torch.uint8)
            for s in rng_state["torch_cuda_rng_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


class AtomicCheckpointManager:
    """
    Manages atomic checkpoint writes (`last.pt`, `best.pt`) and `checkpoint_manifest.json`
    with bounded disk retention (`max_total_bytes`, default <= 2 GiB).
    """

    ALLOWED_SLOTS = {"last", "best"}

    def __init__(
        self,
        checkpoint_dir: Path,
        max_total_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if max_total_bytes <= 0 or max_total_bytes > 2 * 1024 * 1024 * 1024:
            raise ValueError("max_total_bytes must be in (0, 2 GiB]")
        self.checkpoint_dir = checkpoint_dir
        self.max_total_bytes = max_total_bytes
        self.manifest_path = checkpoint_dir / "checkpoint_manifest.json"
        self.best_val_loss: float | None = None

    def _cleanup_stale_files(self) -> None:
        if not self.checkpoint_dir.exists():
            return
        allowed_names = {"last.pt", "best.pt", "checkpoint_manifest.json"}
        for p in self.checkpoint_dir.iterdir():
            if p.is_file() and p.name not in allowed_names:
                p.unlink(missing_ok=True)

    def save_checkpoint(
        self,
        *,
        slot_name: str,
        model: ScratchGPT,
        optimizer: torch.optim.Optimizer,
        scheduler_state: dict[str, Any],
        train_iterator: DeterministicShardIterator,
        optimizer_step: int,
        micro_step: int,
        processed_valid_tokens: int,
        train_loss: float,
        val_loss: float | None,
        lineage: LineageHashes,
    ) -> CheckpointMetadata:
        if slot_name not in self.ALLOWED_SLOTS:
            raise ValueError(f"Invalid slot_name '{slot_name}', must be one of {sorted(self.ALLOWED_SLOTS)}")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_files()

        target_path = self.checkpoint_dir / f"{slot_name}.pt"
        tmp_path = self.checkpoint_dir / f"{slot_name}.pt.tmp"

        payload = {
            "schema_version": "1.0.0",
            "slot_name": slot_name,
            "optimizer_step": int(optimizer_step),
            "micro_step": int(micro_step),
            "processed_valid_tokens": int(processed_valid_tokens),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "lineage": lineage.to_dict(),
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": dict(scheduler_state),
            "train_iterator_state": train_iterator.state_dict(),
            "rng_states": capture_rng_states(),
        }

        torch.save(payload, tmp_path)
        raw_bytes = tmp_path.read_bytes()
        file_sha = hashlib.sha256(raw_bytes).hexdigest()
        file_size = len(raw_bytes)

        # Check storage budget before finalizing replacement
        other_bytes = sum(
            p.stat().st_size
            for p in self.checkpoint_dir.iterdir()
            if p.is_file() and p not in {target_path, tmp_path}
        )
        if other_bytes + file_size > self.max_total_bytes:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checkpoint storage budget exceeded: {other_bytes + file_size} > {self.max_total_bytes}"
            )

        tmp_path.replace(target_path)
        self._cleanup_stale_files()

        meta = CheckpointMetadata(
            schema_version="1.0.0",
            slot_name=slot_name,
            filename=target_path.name,
            file_sha256=file_sha,
            file_bytes=file_size,
            optimizer_step=int(optimizer_step),
            micro_step=int(micro_step),
            processed_valid_tokens=int(processed_valid_tokens),
            train_loss=float(train_loss),
            val_loss=float(val_loss) if val_loss is not None else None,
            lineage=lineage,
            model_config=asdict(model.config),
        )
        self._update_manifest(meta)
        if slot_name == "best" and val_loss is not None:
            self.best_val_loss = float(val_loss)
        return meta

    def _update_manifest(self, meta: CheckpointMetadata) -> None:
        existing: dict[str, Any] = {"schema_version": "1.0.0", "slots": {}}
        if self.manifest_path.exists():
            try:
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {"schema_version": "1.0.0", "slots": {}}
        existing["slots"][meta.slot_name] = meta.to_dict()
        content = json.dumps(existing, indent=2, sort_keys=True) + "\n"
        tmp_manifest = self.manifest_path.with_suffix(".json.tmp")
        tmp_manifest.write_text(content, encoding="utf-8")
        tmp_manifest.replace(self.manifest_path)

    def load_and_verify(
        self,
        slot_name: str,
        *,
        expected_lineage: LineageHashes,
        model: ScratchGPT,
        optimizer: torch.optim.Optimizer | None = None,
        train_iterator: DeterministicShardIterator | None = None,
        restore_rng: bool = True,
    ) -> CheckpointMetadata:
        if slot_name not in self.ALLOWED_SLOTS:
            raise ValueError(f"Unknown slot_name '{slot_name}'")
        if not self.manifest_path.exists():
            raise CheckpointIntegrityError(f"Missing checkpoint_manifest.json in {self.checkpoint_dir}")

        manifest_data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        slots = manifest_data.get("slots", {})
        if slot_name not in slots:
            raise CheckpointIntegrityError(f"Slot '{slot_name}' not found in checkpoint_manifest.json")

        slot_meta_raw = slots[slot_name]
        ckpt_path = self.checkpoint_dir / slot_meta_raw["filename"]
        if not ckpt_path.exists():
            raise CheckpointIntegrityError(f"Checkpoint file {ckpt_path} is missing")

        raw_bytes = ckpt_path.read_bytes()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha != slot_meta_raw["file_sha256"]:
            raise CheckpointIntegrityError(
                f"Corrupted checkpoint {ckpt_path}: expected SHA-256 {slot_meta_raw['file_sha256']}, got {actual_sha}"
            )

        try:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise CheckpointIntegrityError(f"Failed to deserialize checkpoint {ckpt_path}: {exc}") from exc

        saved_lineage = LineageHashes(**payload["lineage"])
        if saved_lineage.dataset_manifest_sha256 != expected_lineage.dataset_manifest_sha256:
            raise CheckpointIntegrityError("Dataset manifest hash mismatch on checkpoint load")
        if saved_lineage.splits_manifest_sha256 != expected_lineage.splits_manifest_sha256:
            raise CheckpointIntegrityError("Splits manifest hash mismatch on checkpoint load")
        if saved_lineage.tokenizer_sha256 != expected_lineage.tokenizer_sha256:
            raise CheckpointIntegrityError("Tokenizer hash mismatch on checkpoint load")
        if saved_lineage.config_sha256 != expected_lineage.config_sha256:
            raise CheckpointIntegrityError("Model/Training config hash mismatch on checkpoint load")

        saved_gpt_cfg = ModelConfig(**payload["model_config"])
        if saved_gpt_cfg != model.config:
            raise CheckpointIntegrityError("Model architecture configuration does not match checkpoint")

        model.load_state_dict(payload["model_state_dict"], strict=True)
        # Re-establish weight tying invariant after load_state_dict
        if model.config.tie_word_embeddings:
            model.lm_head.weight = model.embeddings.token_embedding.weight

        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if train_iterator is not None:
            train_iterator.load_state_dict(payload["train_iterator_state"])
        if restore_rng and "rng_states" in payload:
            restore_rng_states(payload["rng_states"])

        return CheckpointMetadata(
            schema_version=payload["schema_version"],
            slot_name=payload["slot_name"],
            filename=ckpt_path.name,
            file_sha256=actual_sha,
            file_bytes=len(raw_bytes),
            optimizer_step=int(payload["optimizer_step"]),
            micro_step=int(payload["micro_step"]),
            processed_valid_tokens=int(payload["processed_valid_tokens"]),
            train_loss=float(payload["train_loss"]),
            val_loss=float(payload["val_loss"]) if payload["val_loss"] is not None else None,
            lineage=saved_lineage,
            model_config=payload["model_config"],
        )
