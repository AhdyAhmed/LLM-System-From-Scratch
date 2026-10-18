#!/usr/bin/env python3
"""Day002 non-destructive environment and hardware probe.

Verifies Python interpreter, package versions, disk headroom, CPU tensor forward/backward
math, and (when available) CUDA sm_50 FP32 forward/backward smoke execution.
Writes structured JSON evidence to docs/reports/day002-environment-probe.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _run_cmd(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return out.strip().splitlines()[0] if out.strip() else "unknown"
    except Exception as exc:  # pragma: no cover - defensive fallback
        return f"unavailable ({exc})"


def _run_torch_smoke(device_name: str) -> dict[str, Any]:
    import torch

    torch.manual_seed(42)
    start = time.perf_counter()
    device = torch.device(device_name)

    x = torch.randn(32, 32, dtype=torch.float32, device=device, requires_grad=True)
    w = torch.randn(32, 32, dtype=torch.float32, device=device, requires_grad=True)
    y = (x @ w).tanh().mean()
    y.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    loss_val = float(y.detach().cpu().item())
    grad_finite = bool(
        torch.isfinite(x.grad).all().item() and torch.isfinite(w.grad).all().item()
    )
    return {
        "device": device_name,
        "status": "PASS" if (grad_finite and abs(loss_val) < 10.0) else "FAIL",
        "dtype": "float32",
        "shape": [32, 32],
        "loss": round(loss_val, 6),
        "gradients_finite": grad_finite,
        "elapsed_ms": elapsed_ms,
    }


def collect_probe() -> dict[str, Any]:
    import torch

    disk_summary: dict[str, Any] = {}
    for drive in ("C:\\", "E:\\"):
        if os.path.exists(drive):
            usage = shutil.disk_usage(drive)
            disk_summary[drive.rstrip("\\")] = {
                "total_gib": round(usage.total / (1024**3), 2),
                "used_gib": round(usage.used / (1024**3), 2),
                "free_gib": round(usage.free / (1024**3), 2),
            }

    cpu_smoke = _run_torch_smoke("cpu")
    cuda_available = bool(torch.cuda.is_available())
    gpu_info: dict[str, Any] = {
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "compiled_arch_list": torch.cuda.get_arch_list() if cuda_available else [],
    }

    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu_info.update(
            {
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_vram_mib": round(props.total_memory / (1024**2), 1),
                "sm_50_present": "sm_50" in torch.cuda.get_arch_list(),
                "gpu_smoke": _run_torch_smoke("cuda:0"),
            }
        )
    else:
        gpu_info["gpu_smoke"] = {
            "device": "cuda:0",
            "status": "SKIPPED_CPU_FALLBACK_ACTIVE",
        }

    return {
        "probe_version": 1,
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
        },
        "runtimes": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "java_version": _run_cmd(["java", "-version"]),
            "docker_version": _run_cmd(["docker", "--version"]),
            "torch_version": torch.__version__,
        },
        "storage": disk_summary,
        "cpu_smoke": cpu_smoke,
        "gpu": gpu_info,
        "fallback_policy": (
            "CPU tiny-model profile remains a first-class supported execution path "
            "whenever GPU VRAM is occupied or unavailable."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Day002 environment probe.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/day002-environment-probe.json"),
        help="Path to write JSON probe report.",
    )
    args = parser.parse_args()

    report = collect_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"[probe] CPU smoke: {report['cpu_smoke']['status']} ({report['cpu_smoke']['elapsed_ms']} ms)")
    gpu_smoke = report["gpu"]["gpu_smoke"]
    print(f"[probe] GPU smoke: {gpu_smoke['status']} on {report['gpu'].get('device_name', 'CPU-only')}")
    print(f"[probe] Wrote report to {args.output}")

    if report["cpu_smoke"]["status"] != "PASS":
        return 1
    if report["gpu"]["cuda_available"] and gpu_smoke["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
