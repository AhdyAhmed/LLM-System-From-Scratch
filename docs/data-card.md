# Data Card: Curated TinyStories-Style Pretraining Corpus (`scratch-curated-tinystories-v1`)

## 1. Dataset Overview & Provenance (Day031)
- **Dataset ID:** `scratch-curated-tinystories-v1`
- **Source Inspiration:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) simple narrative storytelling structure (`CDLA-Sharing-1.0`).
- **Implementation Strategy:** Self-authored, deterministic 120-document narrative grid with explicit duplicate/rejection test cases (`curated-2026-10-16-r1`), generated and verified locally via [`ingest_bounded_corpus`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/data/ingestion.py#L84-L186) without requiring multi-gigabyte external downloads on a storage-constrained host (`C:` ~12.4 GiB, `E:` ~17.5 GiB).
- **License:** `CDLA-Sharing-1.0` (Community Data License Agreement – Sharing, Version 1.0).
- **Manifest Path:** [`ml/data/manifest.json`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/data/manifest.json)

## 2. Storage Budgets & Privacy Rules (Day031–Day032)
1. **Storage Caps (`StorageBudget`):**
   - `max_raw_bytes`: `10,485,760` (10 MiB for the local fixture; hard-capped at `<= 2 GiB` by [`StorageBudget.validate`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/data/manifest.py#L22-L33))
   - `max_processed_bytes`: `10,485,760` (10 MiB)
   - `max_records`: `5,000`
   - `min_record_chars`: `40`
   - `max_record_chars`: `2,048`
2. **Strict Exclusion of Private Data:**
   - Any record tagged `private_chat` or `user_upload` is immediately rejected by [`ingest_bounded_corpus`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/data/ingestion.py#L118-L120).
   - Ingestion audit reports record only `doc_id` and rejection/duplication category—never raw user text.

## 3. Text Normalization & Deduplication (Day032)
Every raw record passes through [`normalize_document_text`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/data/ingestion.py#L54-L74):
1. Unicode `NFC` normalization.
2. Line-ending canonicalization (`\r\n` and `\r` $\rightarrow$ `\n`).
3. Stripping of non-printable control characters (`unicodedata.category(ch).startswith("C")`, preserving `\n` and `\t`).
4. Horizontal whitespace collapsing and consecutive newline capping (`<= 2`).
5. **Two-tier Deduplication:**
   - **Exact Hash:** SHA-256 of normalized UTF-8 text.
   - **Canonical Near-Duplicate Hash ([`compute_canonical_dedup_hash`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/data/ingestion.py#L77-L82)):** `NFKC` + `.casefold()` + removal of all punctuation/whitespace so punctuation or casing variants cannot leak across train/val/test splits.

## 4. Document-Level Split Freezing & Held-Out Test Seal (Day033)
- **Split Ratios:** `80%` train (`96` documents), `10%` validation (`12` documents), `10%` test (`12` documents) with deterministic seed `20261016`.
- **Leakage Verification:** [`verify_split_disjointness`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/data/splits.py#L64-L103) asserts zero overlap in `doc_id`, `exact_sha256`, and `canonical_dedup_hash` across all three splits.
- **Sealed Test Split:** [`load_split_records`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/data/splits.py#L168-L189) raises `PermissionError` if `split_name == "test"` unless `allow_sealed_test=True` is explicitly provided for final gate evaluation, and [`assert_train_only_texts`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/data/splits.py#L192-L225) blocks any validation or test document from entering BPE tokenizer training.
