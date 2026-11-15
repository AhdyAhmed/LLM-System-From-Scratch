# Real-Model Inference Baseline (`ScratchGPT` — Day059)

## 1. Model & Runtime Provenance
- **Model Revision**: `scratch-tiny-r1` (`scratchlm-tiny-bpe` v`3`)
- **Package SHA-256**: `ba30f95421b3016b5b08ddae2b63537b701c759c2cbcc42cc495b929ebc2b12b`
- **Tokenizer SHA-256**: `a38cb3a119b033b6b8d8680a11af31b3a09af4cbcc2cf759d0a99dcad07bdd4a`
- **Execution Device**: `cuda:0`
- **Cold Package Load Latency**: `2170.4 ms`
- **Cold TTFT**: `241.371 ms` | **Warm TTFT (16-token prompt)**: `10.109 ms`
- **Mid-Stream Cancellation Slot Release Latency**: `63.19 ms`

## 2. KV-Cache Mathematical Parity (Day057)
- **Prompt Tokens**: `17` | **Decode Steps**: `24`
- **Max Absolute Logit Difference (Cached vs Uncached)**: `1.91e-06` (tolerance `<= 1.0e-05`)
- **Greedy Token Trajectory Match**: `True` (`PASS`)

## 3. Memory Footprint
- **Process RSS**: `580.32 MB`
- **Model Parameter Footprint (FP32)**: `3,501,056 bytes` (`3.34 MB`)
- **CUDA Allocated Memory**: `24,076,800 bytes`
- **KV-Cache Tensor Footprint (112 sequence tokens)**: `454,656 bytes`

## 4. Latency & Throughput Matrix (KV-Cached vs Uncached)

| Prompt Tokens | Output Tokens | Warm TTFT (ms) | KV-Cached ITL p50 (ms) | KV-Cached ITL p95 (ms) | KV-Cached Throughput (tok/s) | Uncached Throughput (tok/s) | KV Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 10.109 | 9.227 | 13.358 | 108.97 | 119.77 | 0.91x |
| 16 | 32 | 9.092 | 9.282 | 11.824 | 102.94 | 100.89 | 1.02x |
| 48 | 16 | 8.823 | 9.03 | 43.918 | 74.78 | 76.7 | 0.97x |
| 48 | 32 | 9.167 | 8.934 | 14.876 | 103.73 | 79.69 | 1.3x |
| 80 | 16 | 12.136 | 7.803 | 12.425 | 113.16 | 99.3 | 1.14x |
| 80 | 32 | 11.545 | 9.613 | 16.83 | 91.3 | 98.82 | 0.92x |
