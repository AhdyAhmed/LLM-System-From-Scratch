"""
Day059: Real-model inference benchmark script measuring:
  - Cold-start vs warm TTFT (time-to-first-token, ms)
  - Inter-token latency (mean, p50, p95 ms/token)
  - End-to-end throughput (tokens/sec) across fixed prompt lengths and output token lengths
  - KV-cached (`ScratchKVCache`) vs uncached full-prefix recomputation speedup and exact logit parity
  - Memory footprint (process RSS, model parameter bytes, KV-cache bytes, CUDA allocated bytes)
  - Mid-stream cancellation latency (ms to free the active GPU generation slot)
Writes `ml/reports/real_model_benchmark.json` and `benchmarks/real-model-baseline.md`.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from fastapi.testclient import TestClient

from scratchlm.inference.kv_cache import ScratchKVCache, verify_kv_cache_parity
from scratchlm.inference.sampling import sample_next_token
from scratchlm.serving.app import (
    DEFAULT_INTERNAL_SERVICE_TOKEN,
    ScratchInferenceEngine,
    create_inference_app,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@torch.no_grad()
def _profile_decode_mode(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    use_kv_cache: bool,
    device: torch.device,
) -> dict[str, float]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    kv_cache = (
        ScratchKVCache(n_layers=model.config.n_layers, max_seq_len=model.config.max_seq_len)  # type: ignore[attr-defined]
        if use_kv_cache
        else None
    )
    seq = list(prompt_ids)
    step_latencies_ms: list[float] = []
    ttft_ms = 0.0

    for step_idx in range(max_new_tokens):
        s_start = time.perf_counter()
        if kv_cache is not None:
            step_tokens = seq if step_idx == 0 else [seq[-1]]
            inp = torch.tensor([step_tokens], dtype=torch.long, device=device)
            logits, _ = model(inp, kv_cache=kv_cache)
        else:
            inp = torch.tensor([seq], dtype=torch.long, device=device)
            logits, _ = model(inp)

        next_id = sample_next_token(logits[0, -1, :], temperature=0.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        s_end = time.perf_counter()

        elapsed_ms = (s_end - s_start) * 1000.0
        if step_idx == 0:
            ttft_ms = elapsed_ms
        else:
            step_latencies_ms.append(elapsed_ms)
        seq.append(next_id)

    total_ms = (time.perf_counter() - t0) * 1000.0
    sorted_itl = sorted(step_latencies_ms) if step_latencies_ms else [ttft_ms]
    p50_itl = sorted_itl[len(sorted_itl) // 2]
    p95_idx = min(len(sorted_itl) - 1, int(len(sorted_itl) * 0.95))
    p95_itl = sorted_itl[p95_idx]
    mean_itl = statistics.mean(sorted_itl)
    tokens_per_sec = (max_new_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0
    kv_bytes = kv_cache.total_bytes() if kv_cache is not None else 0

    return {
        "ttft_ms": round(ttft_ms, 3),
        "itl_mean_ms": round(mean_itl, 3),
        "itl_p50_ms": round(p50_itl, 3),
        "itl_p95_ms": round(p95_itl, 3),
        "total_ms": round(total_ms, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "kv_cache_bytes": kv_bytes,
    }


def run_real_model_benchmark() -> dict[str, Any]:
    repo_root = _repo_root()

    # 1. Measure cold-start model package load + first token latency
    cold_t0 = time.perf_counter()
    engine = ScratchInferenceEngine()
    assert engine.runtime is not None
    cold_load_ms = (time.perf_counter() - cold_t0) * 1000.0

    model = engine.runtime.adapter.model
    tokenizer = engine.runtime.adapter.tokenizer
    device = next(model.parameters()).device

    # Cold first-token forward pass
    sample_prompt_ids = tokenizer.encode("Once upon a time in a quiet valley", add_bos=True, add_eos=False)
    cold_step_t0 = time.perf_counter()
    with torch.no_grad():
        inp0 = torch.tensor([sample_prompt_ids], dtype=torch.long, device=device)
        _ = model(inp0)
        if device.type == "cuda":
            torch.cuda.synchronize()
    cold_ttft_ms = (time.perf_counter() - cold_step_t0) * 1000.0

    # 2. Verify KV-cache mathematical parity on the real G5 model
    parity_report = verify_kv_cache_parity(model, sample_prompt_ids, decode_steps=24, atol=1e-5)
    assert parity_report.passed, f"KV cache parity failed: {parity_report}"

    # 3. Profile across fixed prompt lengths (16, 48, 80 tokens) and output lengths (16, 32 tokens)
    base_ids = tokenizer.encode(
        "Once upon a time in a quiet green valley, a curious child named Mira found a glowing silver lantern beside the old stone bridge and asked how it worked.",
        add_bos=True,
        add_eos=False,
    )
    while len(base_ids) < 96:
        base_ids.extend(base_ids[1:])

    profiles: list[dict[str, Any]] = []
    for prompt_len in (16, 48, 80):
        for out_len in (16, 32):
            if prompt_len + out_len > model.config.max_seq_len:
                continue
            p_ids = base_ids[:prompt_len]
            cached_stats = _profile_decode_mode(
                model, p_ids, out_len, use_kv_cache=True, device=device
            )
            uncached_stats = _profile_decode_mode(
                model, p_ids, out_len, use_kv_cache=False, device=device
            )
            speedup = (
                round(uncached_stats["total_ms"] / cached_stats["total_ms"], 2)
                if cached_stats["total_ms"] > 0
                else 1.0
            )
            profiles.append(
                {
                    "prompt_tokens": prompt_len,
                    "output_tokens": out_len,
                    "cached": cached_stats,
                    "uncached": uncached_stats,
                    "kv_cache_speedup_x": speedup,
                }
            )

    # 4. Measure mid-stream cancellation latency through the FastAPI service
    app = create_inference_app(engine)
    client = TestClient(app)
    headers = {"X-Internal-Service-Token": DEFAULT_INTERNAL_SERVICE_TOKEN}
    cancel_run_id = "00000000-0000-4000-8000-000000000599"

    cancel_latency_ms = 0.0
    stream_done_event = threading.Event()

    def _run_slow_stream() -> None:
        with client.stream(
            "POST",
            "/internal/v1/generate",
            headers=headers,
            json={
                "schema_version": 1,
                "run_id": cancel_run_id,
                "attempt_token": 1,
                "model_revision": "scratch-tiny-r1",
                "prompt": "Once upon a time [SLOW_REAL]",
                "context_limit": 128,
                "sampling": {
                    "max_new_tokens": 16,
                    "temperature": 0.0,
                    "top_k": 32,
                    "top_p": 0.95,
                    "seed": 42,
                },
                "deadline_epoch_ms": int((time.time() + 60) * 1000),
                "fault_injection": {"chunk_delay_ms": 40},
            },
        ) as resp:
            for _line in resp.iter_lines():
                pass
        stream_done_event.set()

    worker = threading.Thread(target=_run_slow_stream, daemon=True)
    worker.start()
    time.sleep(0.06)
    c_t0 = time.perf_counter()
    c_resp = client.post(f"/internal/v1/cancel/{cancel_run_id}", headers=headers)
    assert c_resp.status_code == 200
    stream_done_event.wait(timeout=2.0)
    cancel_latency_ms = round((time.perf_counter() - c_t0) * 1000.0, 2)

    # 5. Memory footprint
    proc = psutil.Process(os.getpid())
    rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    cuda_alloc_bytes = (
        int(torch.cuda.memory_allocated())
        if device.type == "cuda" and torch.cuda.is_available()
        else 0
    )

    report = {
        "day": "Day059",
        "model_revision": engine.runtime.model_revision,
        "selected_model_name": engine.runtime.selected_model_name,
        "registry_version": engine.runtime.registry_version,
        "package_sha256": engine.runtime.package_sha256,
        "tokenizer_sha256": engine.runtime.tokenizer_sha256,
        "device": str(device),
        "cold_package_load_ms": round(cold_load_ms, 2),
        "cold_ttft_ms": round(cold_ttft_ms, 3),
        "warm_ttft_ms": profiles[0]["cached"]["ttft_ms"],
        "kv_cache_parity": {
            "prompt_len": parity_report.prompt_len,
            "decode_steps": parity_report.decode_steps,
            "max_abs_logit_diff": parity_report.max_abs_logit_diff,
            "tolerance": parity_report.tolerance,
            "passed": parity_report.passed,
            "greedy_tokens_match": parity_report.greedy_tokens_match,
        },
        "memory": {
            "process_rss_mb": rss_mb,
            "model_parameters_bytes": param_bytes,
            "cuda_allocated_bytes": cuda_alloc_bytes,
            "max_kv_cache_bytes_at_112_tokens": profiles[-1]["cached"]["kv_cache_bytes"],
        },
        "cancellation_slot_release_latency_ms": cancel_latency_ms,
        "profiles": profiles,
    }

    json_out = repo_root / "ml" / "reports" / "real_model_benchmark.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Real-Model Inference Baseline (`ScratchGPT` — Day059)",
        "",
        "## 1. Model & Runtime Provenance",
        f"- **Model Revision**: `{report['model_revision']}` (`{report['selected_model_name']}` v`{report['registry_version']}`)",
        f"- **Package SHA-256**: `{report['package_sha256']}`",
        f"- **Tokenizer SHA-256**: `{report['tokenizer_sha256']}`",
        f"- **Execution Device**: `{report['device']}`",
        f"- **Cold Package Load Latency**: `{report['cold_package_load_ms']} ms`",
        f"- **Cold TTFT**: `{report['cold_ttft_ms']} ms` | **Warm TTFT (16-token prompt)**: `{report['warm_ttft_ms']} ms`",
        f"- **Mid-Stream Cancellation Slot Release Latency**: `{report['cancellation_slot_release_latency_ms']} ms`",
        "",
        "## 2. KV-Cache Mathematical Parity (Day057)",
        f"- **Prompt Tokens**: `{parity_report.prompt_len}` | **Decode Steps**: `{parity_report.decode_steps}`",
        f"- **Max Absolute Logit Difference (Cached vs Uncached)**: `{parity_report.max_abs_logit_diff:.2e}` (tolerance `<= {parity_report.tolerance:.1e}`)",
        f"- **Greedy Token Trajectory Match**: `{parity_report.greedy_tokens_match}` (`PASS`)",
        "",
        "## 3. Memory Footprint",
        f"- **Process RSS**: `{rss_mb} MB`",
        f"- **Model Parameter Footprint (FP32)**: `{param_bytes:,} bytes` (`{param_bytes / (1024 * 1024):.2f} MB`)",
        f"- **CUDA Allocated Memory**: `{cuda_alloc_bytes:,} bytes`",
        f"- **KV-Cache Tensor Footprint (112 sequence tokens)**: `{report['memory']['max_kv_cache_bytes_at_112_tokens']:,} bytes`",
        "",
        "## 4. Latency & Throughput Matrix (KV-Cached vs Uncached)",
        "",
        "| Prompt Tokens | Output Tokens | Warm TTFT (ms) | KV-Cached ITL p50 (ms) | KV-Cached ITL p95 (ms) | KV-Cached Throughput (tok/s) | Uncached Throughput (tok/s) | KV Speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in profiles:
        c = p["cached"]
        u = p["uncached"]
        md_lines.append(
            f"| {p['prompt_tokens']} | {p['output_tokens']} | {c['ttft_ms']} | {c['itl_p50_ms']} | "
            f"{c['itl_p95_ms']} | {c['tokens_per_sec']} | {u['tokens_per_sec']} | {p['kv_cache_speedup_x']}x |"
        )
    md_lines.append("")

    md_out = repo_root / "benchmarks" / "real-model-baseline.md"
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    summary = run_real_model_benchmark()
    print(
        json.dumps(
            {
                "model_revision": summary["model_revision"],
                "device": summary["device"],
                "cold_ttft_ms": summary["cold_ttft_ms"],
                "warm_ttft_ms": summary["warm_ttft_ms"],
                "kv_parity_max_diff": summary["kv_cache_parity"]["max_abs_logit_diff"],
                "cancel_latency_ms": summary["cancellation_slot_release_latency_ms"],
            },
            indent=2,
        )
    )
