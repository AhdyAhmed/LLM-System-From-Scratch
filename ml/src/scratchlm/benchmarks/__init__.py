"""Phase 10 (Day091-Day100) performance benchmark harness, KV-cache profiler, and soak reporter."""

from scratchlm.benchmarks.profiler import (
    KVCacheMatrixCell,
    compute_percentiles,
    generate_all_phase10_artifacts,
    run_cpu_demo_verification,
    run_day091_harness_rehearsal,
    run_day095_kv_cache_matrix,
)

__all__ = [
    "KVCacheMatrixCell",
    "compute_percentiles",
    "generate_all_phase10_artifacts",
    "run_cpu_demo_verification",
    "run_day091_harness_rehearsal",
    "run_day095_kv_cache_matrix",
]
