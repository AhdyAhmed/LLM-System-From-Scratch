# Real-Model Serving & KV-Cache Benchmark Matrix (`Day095`)

Model Profile: `scratch-mini-bench` (`875264` trainable parameters, `max_seq_len = 128`)

| Prompt Tokens | Max New Tokens | Cached TTFT | Cached Total Decode | Uncached Total Decode | Cached `tok/s` | Uncached `tok/s` | Speedup Ratio | KV Cache Footprint | Exact Greedy Parity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `16` | `16` | `20.3007 ms` | `183.5512 ms` | `183.112 ms` | `87.17` | `87.38` | **0.998x** | `126976 B` | `True` |
| `16` | `32` | `9.328 ms` | `316.9811 ms` | `368.7232 ms` | `100.95` | `86.79` | **1.163x** | `192512 B` | `True` |
| `64` | `16` | `7.4567 ms` | `183.4248 ms` | `269.823 ms` | `87.23` | `59.3` | **1.471x** | `323584 B` | `True` |
| `64` | `32` | `14.3271 ms` | `327.3148 ms` | `630.3515 ms` | `97.77` | `50.77` | **1.926x** | `389120 B` | `True` |
| `96` | `16` | `13.2654 ms` | `157.3128 ms` | `308.8986 ms` | `101.71` | `51.8` | **1.964x** | `454656 B` | `True` |
| `96` | `32` | `13.9053 ms` | `336.456 ms` | `642.5669 ms` | `95.11` | `49.8` | **1.91x** | `520192 B` | `True` |
