# Tokenizer Card — `ByteLevelBPETokenizer` (`Day114`)

## 1. Tokenizer Architecture & Implementation
- **Implementation**: [`ml/src/scratchlm/tokenizer/bpe_tokenizer.py`](../../ml/src/scratchlm/tokenizer/bpe_tokenizer.py) (`ByteLevelBPETokenizer`).
- **Base Alphabet**: 256 raw UTF-8 byte tokens (`0..255`), guaranteeing **zero `<unk>` (unknown) tokens** and exact lossless round-trip `decode(encode(text)) == text` for arbitrary UTF-8 strings.
- **Special Tokens**:
  - `<|pad|>` (`256`)
  - `<|bos|>` (`257`)
  - `<|eos|>` (`258`, `EOS_TOKEN_ID`)
  - `<|user|>` (`259`)
  - `<|assistant|>` (`260`)
  - `<|system|>` (`261`)
- **Learned BPE Merges**: Deterministic pair-frequency merge table trained strictly on the `train` split up to `vocab_size = 512`.

---

## 2. Artifact Checksum & Perplexity Comparability Notice
- **Packaged Tokenizer Artifact**: [`ml/artifacts/packages/scratchlm_candidate_pkg/tokenizer.json`](../../ml/artifacts/packages/scratchlm_candidate_pkg/tokenizer.json)
- **SHA-256 Verification**: Enforced at model package load time (`ScratchLmPyfuncAdapter.load_package`) and during `POST /api/v1/models/promotions` (`tokenizer_hash`).
- **Perplexity Comparability Rule**: Token-level perplexity (`exp(cross_entropy)`) depends on the segmentation granularity (tokens per UTF-8 byte). All perplexity comparisons in [`day114_sealed_final_test_evaluation.json`](../../ml/artifacts/evals/day114_sealed_final_test_evaluation.json) use the **exact same `ByteLevelBPETokenizer`** and also report tokenization-normalized **bits-per-byte (`BPB`)**.
