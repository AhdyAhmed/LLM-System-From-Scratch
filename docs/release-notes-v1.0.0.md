# Release Notes — ScratchLLM Platform `v1.0.0` (`Day119`–`Day120`)

## 1. Release Highlights

`ScratchLLM Platform v1.0.0` completes the 120-day roadmap (`Gate G1` through `Gate G12`), delivering a headless, reproducible LLM platform built from scratch on a single workstation:

1. **Custom PyTorch Transformer & BPE Tokenizer**:
   - [`ByteLevelBPETokenizer`](../ml/src/scratchlm/tokenizer/bpe_tokenizer.py) (`vocab_size = 512`) and [`ScratchGPT`](../ml/src/scratchlm/model/gpt.py) (`max_seq_len = 128`, pre-norm causal self-attention, tied embeddings, and [`ScratchKVCache`](../ml/src/scratchlm/inference/kv_cache.py)).
   - Packaged as an immutable, SHA-256-verified MLflow pyfunc artifact ([`scratchlm_candidate_pkg`](../ml/artifacts/packages/scratchlm_candidate_pkg/)) evaluated once on the sealed held-out test split ([`day114_sealed_final_test_evaluation.json`](../ml/artifacts/evals/day114_sealed_final_test_evaluation.json)).
2. **Production-Disciplined Java 21 / Spring Boot Modular Monolith (`Flyway V1–V7`)**:
   - Stateless two-replica horizontal scaling behind Nginx (`X-Accel-Buffering: no`) sharing Redis sessions, atomic Lua rate limits, and `model:rev:*` metadata cache.
   - Durable PostgreSQL `run_events` SSE streaming (`Last-Event-ID` replay) with a shared batched multiplexer.
   - Kafka transactional outbox (`outbox_events`), idempotent `consumer_inbox`, DLQ routing, and fenced `work_items`.
   - Workspace-isolated hybrid `pgvector` + FTS retrieval-augmented generation with server-side citation verification.
   - Airflow-to-native-host-worker training bridge with singleton `gpu_leases` (`id = 1`) mutual exclusion (`503 GPU_BUSY_WITH_TRAINING`).

---

## 2. Verification Summary (`Gate G12`)

- **Secret Scan**: `PASS` (`0` leaked credentials or keys)
- **Docker Compose Profiles**: `PASS` (`db`, `platform`, `tracking`, `pipelines`, `full`)
- **Python Test Suite (`pytest`)**: `70 passed, 0 failed`
- **Java Integration Suite (`Maven Surefire`)**: `43 passed, 0 failed, 0 errors`
- **Clean-Checkout Rehearsal**: `PASSED` ([`docs/reports/day117-clean-checkout-rehearsal.json`](reports/day117-clean-checkout-rehearsal.json))
- **Release Candidate Manifest**: [`ml/artifacts/release_candidate_manifest.json`](../ml/artifacts/release_candidate_manifest.json)
