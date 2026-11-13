# Day049: Narrow Supervised Fine-Tuning (SFT) Evaluation Report

## 1. SFT Lineage & Base Checkpoint Preservation (Day048)
- **SFT MLflow Run ID:** `0323942735134499b66f1ac177908bbe` (Child of Base Run `0a783f1243ed4cb7a5e3eb58cad2f79a`)
- **Frozen Base Checkpoint SHA-256:** `6380ea7d6097d5ce6264afc6d220498de849981dd4f192cb811b8a9a6894e3d7` (Verified unmodified after SFT)
- **SFT Checkpoint SHA-256:** `846f90fad9f218941c0ff26b4f253a41f39ac5308204c9dd7403a647bd45e3b9`
- **SFT Dataset Manifest SHA-256:** `71c1e64092950da3b70306c0cac71e627dcbb0df8594ed00f5e51eb5d697dc06` (`32` train, `8` val, `8` sealed test examples across disjoint task/template families; all `<= 128` byte tokens)
- **Supervision Policy:** Assistant-only loss mask (`<|system|>`, `<|user|>`, and `<|pad|>` masked with `IGNORE_INDEX = -100`; assistant content + `<|end_turn|>` + `<|eos|>` supervised).
- **SFT Optimization Convergence:** `initial_sft_train_loss = 6.4371` $\rightarrow$ `final_sft_train_loss = 3.9105`; `initial_sft_val_loss = 6.5495` $\rightarrow$ `final_sft_val_loss = 4.4703`.

## 2. Base vs. SFT Validation Comparison

| Metric | Base Model (`tiny-bpe-512`) | Narrow SFT Model (`sft-tiny-bpe-512`) |
| --- | ---: | ---: |
| Held-Out SFT Format Marker Compliance (`8` unseen family examples) | `0.0%` | **`0.0%`** |
| Average Generated Token Repetition Ratio | `0.8750` | `0.9312` |
| General Story LM Validation Cross-Entropy (`val.bin`) | `4.8967` | `6.6850` |
| General Story LM Validation Perplexity (`val.bin`) | `133.85` | `800.35` |

## 3. Held-Out Validation Prompt Outputs (`val.jsonl`)

| Example ID | User Prompt | Reference Answer | Base Output (Compliant?) | SFT Output (Compliant?) |
| --- | --- | --- | --- | --- |
| `sft-0033` | `Write moral: Sammy folded the map before rain fell.` | `MORAL: early prep saves dry maps.` | `. a a a a a a a a a a a a a a a` (`False`) | `EEEEEEEEEE` (`False`) |
| `sft-0034` | `Write moral: Hana tied the red ribbon neatly.` | `MORAL: neat knots hold strong.` | `. a a a a a a a a a a a a a a a` (`False`) | `EEEEEEEEEEEEEE` (`False`) |
| `sft-0035` | `Write moral: Leo listened to the quiet brook.` | `MORAL: quiet ears hear the stream.` | `turnuz turn turn turnuz turn turn turn turn turn turn turn turn turn turn` (`False`) | `EEEEEEEEEEEEEEEE` (`False`) |
| `sft-0036` | `Write moral: Dina counted seven silver coins.` | `MORAL: careful counts prevent loss.` | `. a a a a a a a a a a a a a a a` (`False`) | `EEEEEEEEEEEEEEE` (`False`) |
| `sft-0037` | `Extract item: Sammy packed a wool cloak.` | `ITEM=wool cloak` | `a a a a a a a a a a a a a a a a` (`False`) | `EEEEEEEEEEEEEEEE` (`False`) |
| `sft-0038` | `Extract item: Hana polished a silver spoon.` | `ITEM=silver spoon` | `. a a a a a a a a a a a a a a a` (`False`) | `EEEEEEEEEEEEEEEE` (`False`) |
| `sft-0039` | `Extract item: Leo carved a pine whistle.` | `ITEM=pine whistle` | `uz turn turnuzuz turn turn turn turn turn turn turn turn turn turn turn` (`False`) | `EEEEEEEEEEEEEEEE` (`False`) |
| `sft-0040` | `Extract item: Dina locked a cedar box.` | `ITEM=cedar box` | `turn. a a a a a a a a a a a a a a` (`False`) | `EEEEEEEEEEEEEEEE` (`False`) |

## 4. Packaging & MLflow Registry Reconciliation (Day049–Day050)
- **Registered Model Name:** `scratchlm-tiny-bpe`
- **Selected Registry Version:** `v3` (Alias: `candidate`; Active Serving `champion` alias: `False`)
- **Package SHA-256:** `ba30f95421b3016b5b08ddae2b63537b701c759c2cbcc42cc495b929ebc2b12b`
- **Artifact Logical URI:** `mlflow-artifacts:/scratchlm-phase5-sft/0323942735134499b66f1ac177908bbe/artifacts/model_package`
- **Labeled Timeout Duplicates Reconciled:** `[]`
