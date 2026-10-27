# Test Strategy

## 1. Testing Philosophy & Pyramid

This repository enforces a **zero-mock database** and **deterministic ML contract** testing philosophy:

| Layer | Tools & Runtime | Scope | Speed Budget |
| :--- | :--- | :--- | :--- |
| **Static & Contract Tests** | `jsonschema` (Python), Jackson + OpenAPI validator (Java) | `contracts/openapi.yaml`, `internal-inference.schema.json`, `sse-events.schema.json`, golden fixtures | `< 2 s` |
| **Architecture & Module Boundary Tests** | JUnit 5 reflection (`ModuleBoundariesTest`) | Enforce strict package acyclicity and `GenerationProvider` SPI isolation | `< 1 s` |
| **Python ML Unit & Property Tests** | `pytest` on `tiny.json` (`843,520` params) | Tokenizer round-trip, causal masking, finite-difference gradient check, KV-cache equivalence | `< 10 s` |
| **Java Integration Tests** | Spring Boot Test + real PostgreSQL 16 (`pgvector/pgvector:pg16` on `127.0.0.1:54322`) | Flyway migrations, partial unique indexes (`uq_active_run_per_conversation`, `uq_assistant_msg_per_run`), transactional rollback savepoints | `< 15 s` |
| **End-to-End & Fault Injection** | Deterministic fake provider (`DeterministicFakeGenerationProvider`) & internal worker | Timeout, upstream `503`, disconnect cancellation, outbox replay, idempotency | `< 20 s` |

## 2. Deterministic Seed Policy

- Every randomized operation in `ml/` accepts an explicit `seed` (default `1337`).
- `DeterministicFakeGenerationProvider` in `backend/` computes a `SHA-256` digest of `(modelCode, seed, prompt)` so identical inputs emit identical token chunks across JVM restarts and OS platforms.

## 3. PostgreSQL Savepoint Discipline in Integration Tests

PostgreSQL marks the current transaction as `aborted` (`SQLState 25P02`) whenever a constraint violation (`UNIQUE`, `CHECK`, `FOREIGN KEY`) occurs. Tests that intentionally assert `DataIntegrityViolationException` inside `@Transactional` methods wrap the failing statement in a native SQL `SAVEPOINT sp_constraint_test` / `ROLLBACK TO SAVEPOINT sp_constraint_test` (`executeInSavepoint`) so subsequent assertions in the same test method can continue cleanly.
