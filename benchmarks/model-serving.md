# Real-Model Serving & KV-Cache Benchmark Matrix (`Day095`)

Model Profile: `scratch-mini-bench` (`875264` trainable parameters, `max_seq_len = 128`)

| Prompt Tokens | Max New Tokens | Cached TTFT | Cached Total Decode | Uncached Total Decode | Cached `tok/s` | Uncached `tok/s` | Speedup Ratio | KV Cache Footprint | Exact Greedy Parity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `16` | `16` | `33.62 ms` | `211.5148 ms` | `288.6116 ms` | `75.64` | `55.44` | **1.364x** | `126976 B` | `True` |
| `16` | `32` | `39.2392 ms` | `542.6051 ms` | `1294.4741 ms` | `58.97` | `24.72` | **2.386x** | `192512 B` | `True` |
| `64` | `16` | `24.765 ms` | `281.931 ms` | `371.8656 ms` | `56.75` | `43.03` | **1.319x** | `323584 B` | `True` |
| `64` | `32` | `21.1861 ms` | `549.4552 ms` | `932.6813 ms` | `58.24` | `34.31` | **1.697x** | `389120 B` | `True` |
| `96` | `16` | `23.3144 ms` | `211.4639 ms` | `294.2496 ms` | `75.66` | `54.38` | **1.391x** | `454656 B` | `True` |
| `96` | `32` | `15.6871 ms` | `370.6499 ms` | `602.7451 ms` | `86.33` | `53.09` | **1.626x** | `520192 B` | `True` |
