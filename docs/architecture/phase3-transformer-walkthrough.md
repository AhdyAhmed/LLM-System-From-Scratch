# Phase 3 Architecture Walkthrough: Forward & Backward Path of `ScratchGPT`

## 1. End-to-End Tensor Shapes & Forward Path

Given a batch of token IDs $X \in \{0, \dots, V-1\}^{B \times T}$ with $T \le C$ (`max_seq_len`):

1. **Token & Learned Positional Embeddings (`TokenAndPositionEmbedding`):**
   - Token lookup: $E_{\text{tok}} = W_E[X] \in \mathbb{R}^{B \times T \times d}$ where $W_E \in \mathbb{R}^{V \times d}$.
   - Position lookup: for positions $p = (0, 1, \dots, T-1)$, $E_{\text{pos}} = W_P[p] \in \mathbb{R}^{1 \times T \times d}$ where $W_P \in \mathbb{R}^{C \times d}$.
   - Initial residual stream: $H^{(0)} = \text{Dropout}(E_{\text{tok}} + E_{\text{pos}}) \in \mathbb{R}^{B \times T \times d}$.

2. **Pre-Norm Decoder Blocks ($l = 1, \dots, L$, `PreNormTransformerBlock`):**
   - **Explicit LayerNorm (`ScratchLayerNorm`):**
     $$\mu = \frac{1}{d}\sum_{k=1}^d h_k, \quad \sigma^2 = \frac{1}{d}\sum_{k=1}^d (h_k - \mu)^2, \quad \text{LN}(h) = \gamma \odot \frac{h - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
   - **Causal Multi-Head Self-Attention (`CausalMultiHeadSelfAttention`):**
     - Linear projections: $Q = \text{LN}_1(H^{(l-1)}) W_Q + b_Q$, $K = \text{LN}_1(H^{(l-1)}) W_K + b_K$, $V = \text{LN}_1(H^{(l-1)}) W_V + b_V \in \mathbb{R}^{B \times T \times d}$.
     - Head split (`split_heads`): reshape $[B, T, h, d_{\text{head}}] \to [B, h, T, d_{\text{head}}]$ with $d_{\text{head}} = d / h$.
     - Scaled dot-product & causal mask:
       $$S = \frac{Q K^\top}{\sqrt{d_{\text{head}}}} + M_{\text{causal}} \in \mathbb{R}^{B \times h \times T \times T}, \quad M_{\text{causal}}[i, j] = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$
     - Numerically stable softmax (`stable_softmax`):
       $$A = \text{softmax}(S - \max_j S) \in \mathbb{R}^{B \times h \times T \times T}$$
     - Head merge & output projection: $O = \text{Merge}(A V) W_O + b_O \in \mathbb{R}^{B \times T \times d}$.
     - First residual addition: $\tilde{H}^{(l)} = H^{(l-1)} + \text{Dropout}(O)$.
   - **Feed-Forward Network (`ScratchMLP`):**
     $$\text{MLP}(z) = \text{GELU}(z W_{\text{up}} + b_{\text{up}}) W_{\text{down}} + b_{\text{down}}, \quad W_{\text{up}} \in \mathbb{R}^{d \times 4d}, \; W_{\text{down}} \in \mathbb{R}^{4d \times d}$$
     - Second residual addition: $H^{(l)} = \tilde{H}^{(l)} + \text{Dropout}(\text{MLP}(\text{LN}_2(\tilde{H}^{(l)})))$.

3. **Final LayerNorm & Tied Language Model Head (`ScratchGPT`):**
   - $H_{\text{final}} = \text{LN}_f(H^{(L)}) \in \mathbb{R}^{B \times T \times d}$.
   - Tied projection ($W_{\text{lm\_head}} \equiv W_E$):
     $$\text{Logits} = H_{\text{final}} W_E^\top \in \mathbb{R}^{B \times T \times V}$$

## 2. Backward Path, Residual Variance Scaling & Tied-Weight Gradients

1. **Masked Next-Token Cross-Entropy Gradient:**
   - For valid supervised positions $\mathcal{V} = \{(b, t) : Y_{b,t} \ne -100\}$ with $N_{\text{valid}} = |\mathcal{V}|$:
     $$\mathcal{L} = -\frac{1}{N_{\text{valid}}} \sum_{(b,t) \in \mathcal{V}} \log P_{b,t,Y_{b,t}}, \quad \frac{\partial \mathcal{L}}{\partial \text{Logits}_{b,t,v}} = \begin{cases} \frac{1}{N_{\text{valid}}}(P_{b,t,v} - \mathbf{1}_{[v = Y_{b,t}]}) & (b,t) \in \mathcal{V} \\ 0 & (b,t) \notin \mathcal{V} \end{cases}$$
   - Padded positions (`PAD_TOKEN_ID = 256`, mapped to `IGNORE_INDEX = -100`) receive identically zero gradient.
2. **Unimpeded Pre-Norm Residual Highway:**
   - Because each block applies LayerNorm *inside* the sub-layer branch ($H^{(l)} = H^{(l-1)} + f_l(\text{LN}(H^{(l-1)}))$), the Jacobian $\frac{\partial H^{(L)}}{\partial H^{(0)}}$ contains the identity matrix $I + \sum_{l=1}^L \dots$, allowing gradients to flow directly from $\text{LN}_f$ back to $E_{\text{tok}}$ and $E_{\text{pos}}$ without vanishing at step 0.
   - Initializing $W_O$ and $W_{\text{down}}$ with $\sigma_{\text{res}} = \frac{0.02}{\sqrt{2L}}$ keeps the variance of the $2L$ residual additions $\mathcal{O}(1)$ at initialization.
3. **Weight-Tying Gradient Accumulation:**
   - Because `self.lm_head.weight` and `self.embeddings.token_embedding.weight` reference the exact same `nn.Parameter` tensor, PyTorch's autograd engine accumulates both the output projection gradient $H_{\text{final}}^\top \frac{\partial \mathcal{L}}{\partial \text{Logits}}$ and the input token lookup gradient into `W_E.grad` in a single backward pass.
