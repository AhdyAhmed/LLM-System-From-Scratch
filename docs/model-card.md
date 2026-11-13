# Model Card: `scratchlm-tiny-bpe` (Phase 5 Gate G5 Candidate)

## 1. Model Details
- **Architecture:** Decoder-only Pre-Norm GPT Transformer ([`ScratchGPT`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/model/gpt.py#L53-L147)) implemented from scratch in PyTorch without `nn.MultiheadAttention` or `nn.Transformer*`.
- **Configuration (`tiny-bpe-512`):** `4` layers, `d_model = 128`, `n_heads = 4` (`d_head = 32`), `d_ff = 512`, `max_seq_len = 128`, tied input/output embeddings, `875,264` trainable parameters.
- **Tokenizer:** Deterministic Byte-Level BPE ([`BPETokenizer`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/tokenizer/bpe_tokenizer.py#L121-L332), `vocab_size = 512`, SHA-256 `a38cb3a119b033b6b8d8680a11af31b3a09af4cbcc2cf759d0a99dcad07bdd4a`).
- **MLflow Registered Model & Version:** `scratchlm-tiny-bpe` version `3` (`candidate` alias).
- **Immutable Package SHA-256:** `ba30f95421b3016b5b08ddae2b63537b701c759c2cbcc42cc495b929ebc2b12b`

## 2. Training Data & Lineage
- **Pretraining Corpus:** `scratch-curated-tinystories-v1` (`CDLA-Sharing-1.0`, manifest SHA-256 `71c1e64092950da3b70306c0cac71e627dcbb0df8594ed00f5e51eb5d697dc06`).
- **Narrow SFT Dataset:** `scratch-narrow-story-sft-v1` (`32` train, `8` validation, `8` sealed test examples across disjoint task/template families).
- **Parent Base Pretraining Run ID:** `0a783f1243ed4cb7a5e3eb58cad2f79a`
- **SFT Run ID:** `0323942735134499b66f1ac177908bbe`

## 3. Measured Performance & Limitations
- **Validation Perplexity (`val.bin`):** Base model reduces validation perplexity from `512.0` (random) and `222.5` (Unigram baseline) down to `133.85`.
- **Narrow SFT Format Compliance:** Improves format marker adherence (`MORAL:`, `ITEM=`) on held-out validation families from `0.0%` to `0.0%`.
- **Explicit Limitations:**
  1. Context window is strictly bounded to `128` tokens.
  2. Trained on a small synthetic storytelling corpus; does not possess general factual knowledge, code generation ability, or multi-turn reasoning.
  3. Candidate registration (`candidate` alias) does not authorize active production traffic (`serving_promotion_approved = false` until Phase 6/7 release controller verification).
