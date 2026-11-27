# Custom Transformer Architecture Walkthrough (`Day113`)

This walkthrough traces our from-scratch decoder-only GPT implementation in [`ml/src/scratchlm/`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/) directly to code symbols, tensor shapes, mathematical formulas, and automated correctness tests.

> [!NOTE]
> All forward-pass modules, causal masks, LayerNorm statistics, next-token cross-entropy, and incremental KV-cache operations are implemented from scratch using explicit PyTorch tensor operations (`torch.Tensor`). Backward gradient propagation uses PyTorch's built-in autograd engine rather than a custom autograd engine.

---

## 1. Implementation Symbol Map

| Component | Source File & Symbol | Tensor Shape Contract |
| :--- | :--- | :--- |
| Model Configuration & Param Count | [`ModelConfig`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/config.py#L11-L70) | $N_{\text{params}} = Vd + Cd + L(12d^2 + 13d) + 2d$ (with weight tying & bias) |
| Byte-Level BPE Tokenizer | [`ByteLevelBPETokenizer`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/tokenizer/bpe_tokenizer.py) | `str -> list[int]` ($V = 512$, 256 byte base + special tokens + learned merges) |
| Layer Normalization | [`ScratchLayerNorm`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/model/normalization.py) | `[B, T, d_model] -> [B, T, d_model]` ($\mu, \sigma^2$ over last dim, $\epsilon = 10^{-5}$) |
| Token & Learned Positional Embeddings | [`TokenAndPositionEmbedding`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/model/blocks.py) | `input_ids [B, T_q]` + `position_offset P` $\to$ `[B, T_q, d_model]` |
| Multi-Head Causal Self-Attention | [`MultiHeadCausalSelfAttention`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/model/attention.py) | $Q: [B, h, T_q, d_h]$, $K, V: [B, h, P + T_q, d_h] \to [B, T_q, d_{\text{model}}]$ |
| Pre-Norm Transformer Block | [`PreNormTransformerBlock`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/model/blocks.py) | $x \leftarrow x + \text{Attn}(\text{LN}_1(x))$, $x \leftarrow x + \text{MLP}(\text{LN}_2(x))$ |
| Decoder-Only GPT & Weight Tying | [`ScratchGPT`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/model/gpt.py#L53-L151) | `logits [B, T_q, V]` where $W_{\text{lm\_head}} \equiv W_{\text{tok\_emb}}$ |
| Masked Next-Token Cross-Entropy | [`masked_next_token_cross_entropy`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/model/gpt.py#L20-L50) | `logits [B, T, V]`, `targets [B, T]` (`IGNORE_INDEX = -100` masked out) |
| Incremental Per-Layer KV Cache | [`ScratchKVCache`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/inference/kv_cache.py#L35-L118) | Stores `k, v` of shape `[B, h, T_cached, d_h]` per layer ($P + T_q \le C$) |

---

## 2. Core Mathematical Formulation

### 2.1 Custom LayerNorm (`ScratchLayerNorm`)
For hidden state $x \in \mathbb{R}^{B \times T \times d}$:
$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2, \quad \text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

### 2.2 Causal Masking with KV-Cache Offset $P$ (`MultiHeadCausalSelfAttention`)
Given cached prefix length $P \ge 0$ and current query length $T_q \ge 1$ (so total key length is $T_k = P + T_q$):
$$S = \frac{QK^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times h \times T_q \times T_k}$$
Query position $q \in \{0, \dots, T_q - 1\}$ corresponds to absolute sequence index $P + q$ and may only attend to key position $k \in \{0, \dots, T_k - 1\}$ when $k \le P + q$. All entries with $k > P + q$ are masked to $-\infty$ prior to softmax:
$$M_{q, k} = \begin{cases} 0 & \text{if } k \le P + q \\ -\infty & \text{if } k > P + q \end{cases}, \quad A = \text{softmax}(S + M)$$
Thus, future token leakage $\frac{\partial \text{logits}_{t}}{\partial x_{t'}}$ for $t' > t$ is identically **$0.0$**.

### 2.3 Initialization & Tied Embeddings (`ScratchGPT`)
- All linear and embedding weights are initialized from $\mathcal{N}(0, 0.02^2)$.
- Residual output projections (`out_proj` in attention and `fc2` in the GELU MLP) are scaled by $\frac{1}{\sqrt{2 \cdot n_{\text{layers}}}}$ to bound residual stream variance across depth $L$.
- Output projection shares the exact storage tensor of the token embedding matrix: `self.lm_head.weight = self.embeddings.token_embedding.weight`.

---

## 3. Worked Tiny Numerical Example

Consider batch size $B = 1$, sequence length $T = 3$ tokens (`[12, 45, 78]`), $d_{\text{model}} = 64$, $n_{\text{heads}} = 4$ ($d_h = 16$), $n_{\text{layers}} = 2$, $V = 512$:

1. **Prefill Step ($P = 0, T_q = 3$)**:
   - `input_ids` shape `[1, 3]` enters `TokenAndPositionEmbedding` at positions `[0, 1, 2]`, producing $x^{(0)} \in \mathbb{R}^{1 \times 3 \times 64}$.
   - In Layer 0, `q, k, v` each have shape `[1, 4, 3, 16]`. `ScratchKVCache.append(0, k, v)` stores `[1, 4, 3, 16]` (`past_len` becomes `3`).
   - Attention score matrix $S \in \mathbb{R}^{1 \times 4 \times 3 \times 3}$ applies upper-triangular mask above the main diagonal ($k > q$), and produces `[1, 3, 64]`.
   - Final logits have shape `[1, 3, 512]`. Next token $y_3 = \arg\max \text{logits}[0, 2, :]$ is selected.
2. **Incremental Decode Step 1 ($P = 3, T_q = 1$)**:
   - Only `[ [y_3] ]` of shape `[1, 1]` is passed into `model(single_token_inp, kv_cache=kv_cache)`.
   - `TokenAndPositionEmbedding` uses `position_offset = kv_cache.past_len = 3` (lookup position `[3]`).
   - New `k_new, v_new` of shape `[1, 4, 1, 16]` are concatenated with cached `[1, 4, 3, 16]` along `dim=2` to form `full_k, full_v` of shape `[1, 4, 4, 16]`.
   - Attention scores $S \in \mathbb{R}^{1 \times 4 \times 1 \times 4}$ attend to all 4 valid keys ($k \le 3 + 0$), producing identical logits to full-prefix `[1, 4]` recomputation within FP32 precision ($\le 1.9 \times 10^{-6}$).

---

## 4. Automated Correctness Test Suite References

- [`ml/tests/test_phase3_transformer_core.py`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/tests/test_phase3_transformer_core.py): Verifies exact parameter counts, zero future-token gradient leakage, `ScratchLayerNorm` parity against `torch.nn.functional.layer_norm`, masked cross-entropy `IGNORE_INDEX` invariance, and deterministic 64-token toy sequence memorization (`loss < 0.05`).
- [`ml/tests/test_phase6_inference_kv_cache_and_service.py`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/tests/test_phase6_inference_kv_cache_and_service.py): Verifies [`verify_kv_cache_parity`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/inference/kv_cache.py#L133-L210) across multi-step autoregressive decoding.
