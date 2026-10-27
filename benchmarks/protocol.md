# Benchmark Protocol

## 1. Hardware & Runtime Baseline

All benchmark runs record the exact hardware fingerprint via `scripts/probe_environment.py`:
- **CPU:** Intel Core i7-6820HQ (4C/8T)
- **RAM:** 16 GB DDR4
- **GPU:** NVIDIA Quadro M2000M (4096 MiB VRAM, Maxwell `sm_50`, FP32 execution)
- **OS:** Windows 10 (`10.0.19045`)
- **Java:** OpenJDK `21.0.12`
- **Python / PyTorch:** Python `3.12.0`, PyTorch `2.6.0+cu118`

## 2. Measurement Rules

1. **Warm-up:** Exclude the first `3` warm-up requests after model load or JVM start.
2. **Sample Size:** Run `N = 50` sequential iterations for single-stream latency benchmarks and `N = 100` across concurrency levels `c in {1, 2, 4, 8}` for control-plane stress tests.
3. **Metrics Captured:**
   - **TTFT (Time-To-First-Token):** Elapsed milliseconds from `POST /api/v1/conversations/{id}/runs` receipt to first `token.delta` SSE emission (`p50`, `p95`, `p99`).
   - **ITL (Inter-Token Latency):** Mean and `p95` milliseconds between consecutive `token.delta` events.
   - **Throughput:** Generated tokens per second (`tokens/sec`).
   - **Memory Ceiling:** Peak JVM heap RSS (`MB`) and peak CUDA allocated/reserved memory (`MiB`).
4. **KV-Cache Verification Requirement (Phase 6):**
   - Measure autoregressive generation at `prompt_tokens = 128`, `max_new_tokens = 128` with `use_cache=False` vs `use_cache=True` and record the speedup ratio and memory delta.
