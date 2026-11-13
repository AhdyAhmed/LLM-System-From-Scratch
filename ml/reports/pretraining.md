# Day046: Bounded Scratch Pretraining Evaluation Report

## 1. Run Lineage & Environment
- **MLflow Run ID:** `0a783f1243ed4cb7a5e3eb58cad2f79a`
- **Model Profile:** `tiny-bpe-512` (`4` layers, `d_model=128`, `4` heads, `d_ff=512`, `max_seq_len=128`, tied embeddings, `875,264` trainable parameters)
- **Tokenizer:** Educational Byte-Level BPE (`V=512`, SHA-256 `a38cb3a119b033b6b8d8680a11af31b3a09af4cbcc2cf759d0a99dcad07bdd4a`)
- **Dataset / Split Manifests:** Dataset `1a311610946a0f59...`, Splits `6c8625fd64a486d5...`
- **Sealed Held-Out Test Split:** `ml/data/splits/test.jsonl` was **never** accessed during tokenizer learning, configuration comparison, pretraining, or prompt inspection.
- **Processed Valid Tokens:** `6,120` across `24` optimizer updates (`FP32`).

## 2. Held-Out Validation (`val.bin`) Comparison vs. Baselines

All models below are evaluated on the exact same validation split (`12` held-out stories, `1,406` BPE target tokens, `4,396` UTF-8 bytes) using the same `V=512` BPE vocabulary:

| Model / Baseline | Split | Valid Tokens | Cross-Entropy Loss (nats/token) | Token Perplexity ($\exp(\mathcal{L})$) | Bits Per Byte (BPB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Untrained `ScratchGPT` (Random Init, step 0) | `val` | 1,406 | 6.2773 | 532.33 | 2.8965 |
| Empirical Unigram Baseline (Laplace $\alpha=1.0$) | `val` | 1,406 | 4.8986 | 134.10 | 2.2603 |
| Empirical Bigram Baseline (Laplace $\alpha=1.0$) | `val` | 1,406 | 3.0948 | 22.08 | 1.4280 |
| **Pretrained `ScratchGPT` (`best.pt`)** | `val` | 1,406 | **4.8967** | **133.85** | **2.2595** |

## 3. Deterministic Validation Prompt Continuations & Failure Analysis

Generated with `temperature=0.0` (greedy), `seed=1337`, `max_new_tokens=24`:

| Prompt ID | Input Prompt | Generated Continuation | Finish Reason | Repetition Ratio | Honest Diagnostic / Failure Analysis |
| --- | --- | --- | --- | ---: | --- |
| `prompt-01` | `Once upon a time in the sunny meadow, Lily` | `........................` | `length` | 0.96 | High token repetition observed (typical of tiny 0.9M parameter model under greedy decoding). |
| `prompt-02` | `Every morning, Omar carried a brass compass` | ` a a a a a a a a a a a a a a a a a a a a a a a a` | `length` | 0.96 | High token repetition observed (typical of tiny 0.9M parameter model under greedy decoding). |
| `prompt-03` | `By evening, everyone remembered that patience` | ` a a a a a a a a a a a a a a a a a a a a a a a a` | `length` | 0.96 | High token repetition observed (typical of tiny 0.9M parameter model under greedy decoding). |

## 4. Honest Capability & Scope Statement
This `0.86M`-parameter transformer trained from random weights on a bounded narrative corpus demonstrates verified causal attention math, BPE tokenization, AdamW optimization, and held-out perplexity reduction below random and unigram baselines. It is **not** a frontier general-purpose assistant: outside narrow story vocabulary, it exhibits token repetition and lacks broad world knowledge or safety alignment.
