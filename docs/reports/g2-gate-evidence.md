# Gate G2 Verification Evidence Report (Phase 2: Day011–Day020)

- **Gate:** G2 — Accounts, Redis, Kafka Foundations, and API Clients
- **Status:** PASSED (`0` errors, `0` failures)
- **Verification Date:** 2026-10-16

## 1. Day-by-Day Deliverables Checklist

| Day | Deliverable | Artifact / Implementation Path | Verification Status |
| :--- | :--- | :--- | :--- |
| **Day011** | Account Creation, BCrypt Hashing & Uniform Auth Responses | `IdentityService.java`, `IdentityDtos.java`, `ApiErrorExceptionHandler.java` | Verified (`$2a$12$` hash, `409 ACCOUNT_ALREADY_EXISTS`, `401 INVALID_CREDENTIALS`) |
| **Day012** | Redis Session Store, Session Fixation Rotation, Logout & CSRF | `RedisSessionStore.java`, `AuthController.java`, `SecurityFoundationConfig.java` | Verified (`HttpOnly`, `SameSite=Lax`, session rotation, `403 CSRF_INVALID`, `503 REDIS_UNAVAILABLE`) |
| **Day013** | Creator-Private Conversation CRUD & Membership+Owner Scoping | `ConversationService.java`, `ConversationController.java` | Verified (User B gets `404` on User A's conversation even when both share a workspace) |
| **Day014** | Keyset-Paginated Conversations & Immutable Message Ordering | `ConversationService.java`, `ConversationDtos.java`, `Phase2ConversationAndMessagePaginationIntegrationTest.java` | Verified (stable Base64URL cursors, zero JPA entity lazy-loading) |
| **Day015** | Postman Collection, `curl.exe` Cookie-Jar Walkthrough & Python `httpx` Client | `contracts/postman/scratchllm-phase2.postman_collection.json`, `scripts/demo_curl_walkthrough.ps1`, `ml/src/scratchlm/client.py` | Verified |
| **Day016** | Idempotent Run Creation, Canonical Request Hashing & Lock Hierarchy | `RunService.java`, `RunDtos.java` | Verified (replay precedes quota/rate checks; `system_capacity -> usage_buckets -> conversations` lock order) |
| **Day017** | Redis Atomic Lua Rate Limiter (`auth` and `run` namespaces) | `RedisRateLimiterService.java` | Verified (`EVAL` single Lua script with bounded `EXPIRE` TTL) |
| **Day018** | Fake-Provider SSE Streaming, Incremental UTF-8 Client & Cancellation | `RunController.java`, `RunService.java`, `ml/src/scratchlm/client.py` | Verified (`run.queued`, `run.started`, `text.delta`, `run.completed`, `run.cancelled`) |
| **Day019** | Single-Broker Kafka KRaft Topics, Schemas & Producer/Consumer Fixtures | `contracts/kafka-events.schema.json`, `contracts/fixtures/kafka/*`, `KafkaEventsInfrastructure.java` | Verified (`generation.requested.v1`, `document.ingestion.requested.v1`, `model.released.v1`) |
| **Day020** | Gate G2 Integration Verification & Evidence Report | `Phase2AuthAndRedisSessionIntegrationTest.java`, `Phase2RunIdempotencyRateLimitAndSseIntegrationTest.java`, `Phase2KafkaContractsIntegrationTest.java`, `ml/tests/test_phase2_client_and_contracts.py` | Verified |

## 2. Automated Test Suite Summary

- **Python ML, Client & Contract Suite (`pytest` in `ml/`):**
  - `19 passed` (`test_phase1_foundation.py` + `test_phase2_client_and_contracts.py`: Kafka schema validation, bulk-field rejection, Postman URL hygiene, split multi-byte UTF-8 SSE decoding for Arabic/emoji, and `ScratchLlmClient` CSRF + `Last-Event-ID` reconnect).
- **Java 21 Spring Boot, PostgreSQL 16, Redis 7.4 & Kafka KRaft 3.9 Integration Suite (`.\mvnw.cmd test` in `backend/`):**
  - `18 tests run, 0 failures, 0 errors`:
    - `ModuleBoundariesTest` (`2` tests)
    - `SharedContractsValidationTest` (`2` tests)
    - `DeterministicFakeGenerationProviderTest` (`4` tests)
    - `PostgresFoundationSchemaIntegrationTest` (`3` tests)
    - `Phase2AuthAndRedisSessionIntegrationTest` (`2` tests)
    - `Phase2ConversationAndMessagePaginationIntegrationTest` (`1` comprehensive multi-user isolation + pagination test)
    - `Phase2RunIdempotencyRateLimitAndSseIntegrationTest` (`2` tests)
    - `Phase2KafkaContractsIntegrationTest` (`2` live Kafka KRaft publish/consume + validation tests)
