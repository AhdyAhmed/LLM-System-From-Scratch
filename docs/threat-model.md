# Threat Model & Security Baseline

## 1. Trust Boundaries

1. **Public HTTP/SSE Boundary (`Client <-> Java Control Plane`):**
   - Untrusted external callers.
   - All requests pass through `X-Correlation-Id` tracing, Spring Security filter chains, rate limiting, and RFC 9457 Problem Details error sanitization.
2. **Internal Loopback Inference Boundary (`Java Control Plane <-> Python FastAPI Worker`):**
   - Bound strictly to `127.0.0.1:8091`.
   - Never exposed to external networks or public container port bindings.
3. **Persistence & Messaging Boundary (`Java / Python <-> PostgreSQL 16, Redis, Kafka`):**
   - Separate database catalogs (`llm_app`, `mlflow_db`, `airflow_db`) with dedicated least-privilege database credentials.

## 2. STRIDE Threat Matrix

| Threat Category | Attack Vector | Mitigation Enforced in Design & Code |
| :--- | :--- | :--- |
| **Spoofing** | Forged JWT or replayed refresh token | Short-lived access JWTs (15 min) + hashed refresh token family rotation (`refresh_tokens.token_hash`) with automatic family revocation on reuse |
| **Tampering** | Cross-user conversation or document access | Every SQL query in `conversation` and `knowledge` scopes by `user_id` AND resource `id`; missing or unauthorized resources return uniform `404` |
| **Repudiation** | Unaudited model promotion or run cancellation | Append-only `audit_events` and `outbox_events` written in the same ACID transaction as domain state changes |
| **Information Disclosure** | Stack traces or SQL details leaked in 500 errors | RFC 9457 Problem Details handler strips internal exception messages and logs full stack traces only with `correlation_id` |
| **Denial of Service** | Concurrent run flooding or VRAM exhaustion on 4 GB GPU | Partial unique index `uq_active_run_per_conversation`, Redis token-bucket rate limiter, and Python `asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)` returning immediate `503 ENGINE_OVERLOADED` |
| **Elevation of Privilege** | Prompt injection via retrieved RAG chunks | Retrieved chunks are wrapped in explicit untrusted `<context doc_id="...">` delimiters and never override system instructions |
