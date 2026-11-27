# Data Card — Pretraining, Narrow SFT & RAG Evaluation Splits (`Day114`)

## 1. Dataset Sources & Licensing Provenance
- **Pretraining Storytelling Corpus**:
  - Derived from a bounded, deterministic local subset inspired by the TinyStories domain ([TinyStories dataset card](https://huggingface.co/datasets/roneneldan/TinyStories), listed under `cdla-sharing-1.0`) combined with self-authored domain fixtures in [`ml/src/scratchlm/data/`](../../ml/src/scratchlm/data/).
  - Raw bulk datasets are excluded from Git via `.gitignore`; deterministic split manifests (`manifest.json`) record exact document counts, byte sizes, and SHA-256 hashes.
- **Narrow Supervised Fine-Tuning (`SFT`) Dataset**:
  - Self-authored `User:` / `Assistant:` instruction-following pairs (`CC0-1.0` / MIT project license) formatted via [`chat_template.json`](../../ml/artifacts/packages/scratchlm_candidate_pkg/chat_template.json) with target loss masking (`IGNORE_INDEX = -100` on non-assistant tokens).
- **RAG Retrieval Evaluation Corpus (`Day072`)**:
  - Self-authored workspace documents and 30+ labeled answerable, unanswerable, and prompt-injection test queries ([`ml/src/scratchlm/rag/`](../../ml/src/scratchlm/rag/)).

---

## 2. Disjoint Split Hygiene & Contamination Controls
- **Deterministic Partitioning**: Documents are normalized (NFC Unicode, canonical LF newlines), deduplicated by SHA-256 content hash, and partitioned into strictly disjoint `train`, `val`, and `test` splits (`0` overlapping document hashes across splits).
- **Tokenizer Isolation**: [`ByteLevelBPETokenizer`](../../ml/src/scratchlm/tokenizer/bpe_tokenizer.py) is trained **exclusively** on the `train` split.
- **Sealed Final Test Protocol (`Day114`)**: All hyperparameter and checkpoint promotion decisions (`Gate G4`, `Gate G5`) used only `val` loss/perplexity. The `test` split remained sealed until [`evaluate_sealed_final_test_once`](../../ml/src/scratchlm/release/final_release.py) in Phase 12.
