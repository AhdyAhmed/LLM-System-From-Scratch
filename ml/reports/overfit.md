# Day029 Toy-Batch Overfit & Causality Report (`ScratchGPT` `tiny` Profile)

- **Experiment Script:** [`ml/experiments/overfit_tiny_batch.py`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/experiments/overfit_tiny_batch.py)
- **Model Profile:** `tiny` (`4` layers, `d_model = 128`, `n_heads = 4`, `d_ff = 512`, `vocab_size = 264`, `max_seq_len = 128`, `843,520` parameters)
- **Random Seed:** `1337` (deterministic CPU FP32 execution, `dropout = 0.0`)
- **Optimizer:** `AdamW(lr=3e-3, weight_decay=0.0)` with gradient norm clipping (`max_norm = 1.0`)

---

## 1. Baseline vs. `ScratchGPT` Falling-Loss Curve

- **Theoretical Uniform Initialization Cross-Entropy ($\ln(264)$):** `5.5759` nats
- **Measured Step-1 Cross-Entropy (`ScratchGPT` with $1/\sqrt{2L}$ residual scaling):** `5.5702` nats
- **Empirical Unigram / Byte-Frequency Baseline ($H(p) = -\sum_v p(v)\log p(v)$):** `2.9906` nats

| Step | Cross-Entropy Loss (nats) | Perplexity ($\exp(\mathcal{L})$) | Comparison vs. Unigram Baseline (`2.9906`) |
| ---: | ---: | ---: | :--- |
| `1` | `5.5702` | `262.49` | Above unigram baseline (random uniform init) |
| `20` | `1.0226` | `2.78` | `-1.9680` nats below unigram baseline |
| `40` | `0.0459` | `1.05` | `-2.9447` nats below unigram baseline |
| `60` | `0.0176` | `1.02` | `-2.9730` nats below unigram baseline |
| `80` | `0.0156` | `1.02` | `-2.9750` nats below unigram baseline |
| `100` | `0.0151` | `1.02` | `-2.9755` nats below unigram baseline |
| `120` | `0.0149` | `1.02` | `-2.9757` nats below unigram baseline |
| `140` | `0.0147` | `1.01` | `-2.9759` nats below unigram baseline |

---

## 2. Full-Model Causality (Future-Token Invariance) Verification

Before and after training, tokens at positions $t > 12$ were mutated (`(token + 37) % 256`) in evaluation mode (`model.eval()`).
- **Maximum absolute logit difference across all positions $0 \le t \le 12$ and all $V = 264$ vocabulary entries:**
  $$\max_{b,\, t \le 12,\, v} \bigl|\text{logits}_{\text{orig}}[b, t, v] - \text{logits}_{\text{mutated}}[b, t, v]\bigr| = \mathbf{0.00 \times 10^0}$$

---

## 3. Greedy Continuation & EOS Stopping Verification

- **Prompt Prefix:** `"<|bos|>Once upon a time"`
- **Greedy Decoded Output (`temperature = 0.0`):** `", the little robot counted 1, 2, 3."`
- **Finish Reason:** `"stop"` (terminated cleanly upon generating `EOS_TOKEN_ID = 257`).
