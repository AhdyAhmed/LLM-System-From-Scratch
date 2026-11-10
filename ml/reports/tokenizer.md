# Day035: Educational Byte-Level BPE Tokenizer Report

## 1. Tokenizer Configuration & Provenance
- **Tokenizer Type:** Educational Byte-Level BPE (`byte_level_bpe`)
- **Artifact Path:** [`ml/artifacts/tokenizer/bpe_tiny_v1.json`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/artifacts/tokenizer/bpe_tiny_v1.json)
- **Tokenizer SHA-256:** `a38cb3a119b033b6b8d8680a11af31b3a09af4cbcc2cf759d0a99dcad07bdd4a`
- **Training Dataset ID:** `scratch-curated-tinystories-v1`
- **Training Split SHA-256:** `ba26d0a6bcd9fac379072de8dae125a11f080311cee8dc402aaeee08c764d831`
- **Base Byte Vocabulary:** `256` (`0..255`)
- **Reserved Special Tokens:** `8` (`256..263`: `<|pad|>`, `<|eos|>`, `<|bos|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|context|>`, `<|end_turn|>`)
- **Learned BPE Merges:** `248`
- **Total Vocabulary Size:** `512`
- **Deterministic Tie-Breaking Rule:** Lexicographically smallest `(token_id_a, token_id_b)` when adjacent pair frequencies tie.

## 2. Compression Comparison: `BPETokenizer` vs. `ByteTokenizer`

| Split Sample | Documents | Total UTF-8 Bytes | Byte Tokens (`V=264`) | BPE Tokens (`V=512`) | Bytes / BPE Token | Token Count Ratio (`BPE / Byte`) | Encode Time (ms) | Decode Time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` (`96` docs) | 96 | 34959 | 34959 | 10889 | **3.2105** | **0.3115** | 61.495 | 2.932 |
| `val` (`12` docs, held-out) | 12 | 4396 | 4396 | 1395 | **3.1513** | **0.3173** | 9.118 | 1.109 |

## 3. First 10 Learned Merges (Rank 0..9)

| Rank | Constituent Pair `(a, b)` | New Token ID | Training Frequency | Merged Hex | Decoded Subword |
| ---: | --- | ---: | ---: | --- | --- |
| 0 | `(32, 116)` | `264` | 930 | `2074` | ` t` |
| 1 | `(101, 114)` | `265` | 823 | `6572` | `er` |
| 2 | `(264, 104)` | `266` | 677 | `207468` | ` th` |
| 3 | `(32, 97)` | `267` | 609 | `2061` | ` a` |
| 4 | `(101, 100)` | `268` | 557 | `6564` | `ed` |
| 5 | `(105, 110)` | `269` | 530 | `696e` | `in` |
| 6 | `(32, 99)` | `270` | 467 | `2063` | ` c` |
| 7 | `(266, 101)` | `271` | 452 | `20746865` | ` the` |
| 8 | `(111, 110)` | `272` | 445 | `6f6e` | `on` |
| 9 | `(97, 114)` | `273` | 442 | `6172` | `ar` |

## 4. Known Limitations of the Educational Implementation
1. **Algorithmic Complexity:** Pair counting iterates over pre-tokenized unique word chunks in pure Python (`O(M * U)` where `M` is the number of merges and `U` is unique chunk length sum). Suitable for small bounded corpora (`<= 50 MiB`), not multi-gigabyte corpora without a C++/Rust priority-queue index.
2. **Domain Specialization:** Merges learned on English narrative stories compress English story vocabulary to `3.15` bytes/token while falling back losslessly to raw UTF-8 byte tokens (`1.00` byte/token) for unseen scripts such as Arabic or emoji.
3. **Held-Out Evaluation Hygiene:** Validation and sealed test splits are strictly excluded via [`assert_train_only_texts`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/ml/src/scratchlm/data/splits.py#L192-L225).
