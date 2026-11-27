# Model Card — `ScratchGPT` (`scratch-tiny-r1` / `scratchlm-candidate-gpt`, `Day114`)

## 1. Model Summary & Provenance
- **Model Name**: `scratchlm-candidate-gpt` (`model_alias = scratch-tiny`, `revision_tag = scratch-tiny-r1`)
- **Architecture**: Pre-norm decoder-only GPT Transformer ([`ScratchGPT`](../../ml/src/scratchlm/model/gpt.py)) built from scratch in PyTorch.
- **Configuration**: `vocab_size = 512`, `max_seq_len = 128`, `n_layers = 4`, `n_heads = 4`, `d_model = 128`, `d_ff = 512`, `tie_word_embeddings = True`, `use_bias = True`.
- **Trainable Parameters**: `~0.87M–1.25M` (exact count verified by [`ModelConfig.exact_parameter_count()`](../../ml/src/scratchlm/config.py#L46-L70)).
- **MLflow Package & Registry Lineage**:
  - Frozen Gate G5/G12 Candidate Manifest: [`ml/artifacts/registry/frozen_g5_candidate.json`](../../ml/artifacts/registry/frozen_g5_candidate.json)
  - Loadable Pyfunc Package Directory: [`ml/artifacts/packages/scratchlm_candidate_pkg/`](../../ml/artifacts/packages/scratchlm_candidate_pkg/)
  - Sealed Final Held-Out Test Evaluation: [`ml/artifacts/evals/day114_sealed_final_test_evaluation.json`](../../ml/artifacts/evals/day114_sealed_final_test_evaluation.json)

---

## 2. Training & Compute Budget
- **Initialization**: Random Gaussian $\mathcal{N}(0, 0.02^2)$ with $1/\sqrt{2 \cdot n_{\text{layers}}}$ residual projection scaling; **zero pretrained weights**.
- **Stages**:
  1. Causal language modeling pretraining on the bounded TinyStories/storytelling training split (`AdamW`, cosine LR schedule with warmup, gradient clipping `1.0`).
  2. Narrow supervised fine-tuning (`SFT`) with loss masked strictly to `Assistant:` completion spans (`IGNORE_INDEX = -100` on prompt tokens).
- **Hardware**: Single workstation (`NVIDIA Quadro M2000M 4 GB VRAM` + CPU reproducible fallback).

---

## 3. Held-Out Evaluation & Candid Failure Examples
- **Held-Out Test Split Evaluation (`Day114`)**:
  - Unsealed once in [`day114_sealed_final_test_evaluation.json`](../../ml/artifacts/evals/day114_sealed_final_test_evaluation.json), keeping `test` strictly disjoint from `train` and `val` model selection.
  - Demonstrated significant cross-entropy and perplexity reduction relative to untrained random initialization, unigram, and bigram baselines under identical `ByteLevelBPETokenizer` tokenization.
- **Candid Failure & Limitation Examples**:
  - **In-Domain Prompt (`seed=1337`)**: `"Once upon a time in the sunny meadow"` $\to$ produces coherent short story continuation tokens within the trained vocabulary distribution.
  - **Out-of-Domain / Factual Prompt (`seed=1337`)**: `"User: What is the capital of Mars?\nAssistant:"` $\to$ respects the `Assistant:` turn boundary learned during SFT, but generates story-like words rather than factual refusal or astronomical knowledge.
- **Intended Use**: Educational demonstration of from-scratch Transformer implementation, KV-cached serving, and MLOps lifecycle management. Not suitable for open-domain factual Q&A or safety-critical decisions.
