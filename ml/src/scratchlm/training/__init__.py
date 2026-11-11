"""Training loop, AdamW setup, cosine scheduler, gradient clipping, atomic checkpointing, and MLflow tracking."""

from scratchlm.training.checkpoints import (
    AtomicCheckpointManager,
    CheckpointIntegrityError,
    CheckpointMetadata,
    LineageHashes,
    capture_rng_states,
    restore_rng_states,
)
from scratchlm.training.loop import (
    NonFiniteLossError,
    StepMetrics,
    TrainingConfig,
    TrainingRunSummary,
    build_adamw_optimizer,
    compute_cosine_lr_with_warmup,
    evaluate_validation_loss,
    execute_accumulated_optimizer_step,
    run_reproducible_training,
    seed_everything,
)
from scratchlm.training.tracker import (
    ExperimentTracker,
    InMemoryExperimentTracker,
    SpooledTrackingEvent,
    SpoolingMlflowRestTracker,
    TrackingSpoolOverflowError,
    validate_logical_artifact_uri,
)

__all__ = [
    "AtomicCheckpointManager",
    "CheckpointIntegrityError",
    "CheckpointMetadata",
    "ExperimentTracker",
    "InMemoryExperimentTracker",
    "LineageHashes",
    "NonFiniteLossError",
    "SpooledTrackingEvent",
    "SpoolingMlflowRestTracker",
    "StepMetrics",
    "TrackingSpoolOverflowError",
    "TrainingConfig",
    "TrainingRunSummary",
    "build_adamw_optimizer",
    "capture_rng_states",
    "compute_cosine_lr_with_warmup",
    "evaluate_validation_loss",
    "execute_accumulated_optimizer_step",
    "restore_rng_states",
    "run_reproducible_training",
    "seed_everything",
    "validate_logical_artifact_uri",
]
