#!/usr/bin/env python3
"""
Day091-Day100 Top-level Benchmark Harness CLI (`benchmarks/harness.py`).
Delegates to `scratchlm.benchmarks.profiler` to generate reproducible JSON metrics
in `benchmarks/results/` and Markdown engineering reports in `benchmarks/` and `docs/reports/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_SRC = REPO_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from scratchlm.benchmarks.profiler import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
