# Real-Model Serving & KV-Cache Benchmark Matrix (`Day095`)

Model Profile: `scratch-mini-bench` (`875264` trainable parameters, `max_seq_len = 128`)

| Prompt Tokens | Max New Tokens | Cached TTFT | Cached Total Decode | Uncached Total Decode | Cached `tok/s` | Uncached `tok/s` | Speedup Ratio | KV Cache Footprint | Exact Greedy Parity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `16` | `16` | `5.0648 ms` | `104.4242 ms` | `156.8186 ms` | `153.22` | `102.03` | **1.502x** | `126976 B` | `True` |
| `16` | `32` | `5.7024 ms` | `216.8435 ms` | `275.5169 ms` | `147.57` | `116.15` | **1.271x** | `192512 B` | `True` |
| `64` | `16` | `25.3816 ms` | `134.665 ms` | `154.2546 ms` | `118.81` | `103.72` | **1.145x** | `323584 B` | `True` |
| `64` | `32` | `11.1137 ms` | `369.4662 ms` | `465.8956 ms` | `86.61` | `68.68` | **1.261x** | `389120 B` | `True` |
| `96` | `16` | `8.9874 ms` | `104.1703 ms` | `175.9843 ms` | `153.59` | `90.92` | **1.689x** | `454656 B` | `True` |
| `96` | `32` | `8.3042 ms` | `227.1215 ms` | `385.9906 ms` | `140.89` | `82.9` | **1.699x** | `520192 B` | `True` |
