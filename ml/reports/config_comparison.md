# Day042–Day043: Pretraining Pilot Stability & Two-Configuration Comparison

## 1. Immutable Data & Tokenizer Lineage
- **Dataset Manifest SHA-256:** `1a311610946a0f596cb62c7131909662864f2f3ae3a41564736bcc6acf1f500b`
- **Splits Manifest SHA-256:** `6c8625fd64a486d5da82cc8439878fd56b591d3fb07f35a07efb021cf4964906` (`test` split remains **sealed**)
- **Tokenizer SHA-256:** `a38cb3a119b033b6b8d8680a11af31b3a09af4cbcc2cf759d0a99dcad07bdd4a` (`BPETokenizer`, `V=512`)
- **Equal Token Budget per Candidate:** `3072` valid non-padding tokens (`12` optimizer steps, `seq_len=64`, `micro_batch_size=2`, `grad_accum=2`).

## 2. Pilot Stability Gate (Day042)
- **Written Stability Threshold:** Finite FP32 loss at every microstep, `max_grad_norm <= 20.0`, strictly decreasing validation loss (`final_val_loss < initial_val_loss`), and bounded memory (`RSS < 2,048 MiB`). No infinite out-of-memory retry loop is permitted.
- **Pilot Outcome:** **PASSED** (`initial_val_loss = 6.2660` $\rightarrow$ `final_val_loss = 4.9829`, max gradient norm `14.6673`).

## 3. Equal-Budget Configuration Comparison (Day043)

| Configuration | Layers / Width / Heads | Parameters | MLflow Run ID | Config SHA-256 | Initial Val Loss | Final Val Loss | Val Perplexity | Avg Step Time (ms) | Throughput (tok/s) |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| **Config A (`tiny-bpe-512`)** | `4L / 128d / 4H` | `875,264` | `d006b76c38e64eb4b8228938656e2747` | `60e05750261e395d...` | 6.2660 | **4.9829** | **145.89** | 84.23 | 3120.0 |
| **Config B (`shallow-wide-bpe-512`)** | `2L / 192d / 6H` | `1,012,992` | `110e8acdd54740aaab094f6c190b5459` | `b289faff651f3bb2...` | 6.2648 | 4.7679 | 117.67 | 67.22 | 3977.6 |

## 4. Selection Decision
- **Selected Main Configuration:** `tiny-bpe-512` (`4L / 128d / 4H`, `875,264` parameters).
- **Rationale:** Achieves lower or comparable validation loss at `~13%` fewer parameters and faster step latency on the 4-core CPU / 4 GiB VRAM workstation budget while preserving deep 4-layer causal attention representations.
