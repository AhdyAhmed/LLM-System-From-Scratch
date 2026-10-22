# Local Infrastructure Setup and Reset Guide

Status: Verified Phase 1 Baseline (Day004 — Gate G1)
Companion documents: [environment.md](environment.md) | [schema.md](schema.md) | [design.md](../design.md)

## 1. Staged Compose Profiles

To keep memory, CPU, and disk usage bounded on the 4-core `i7-6820HQ`, `infra/compose.yml` separates services into explicit profiles:

- **`db` profile (Phase 1 & 3–5 foundation):** Starts only `postgres` (`pgvector/pgvector:pg16`) on `127.0.0.1:54322` and initializes three isolated databases (`llm_app`, `mlflow_db`, `airflow_db`) with separate credentials (`llm_app_user`, `mlflow_user`, `airflow_user`) and `pgvector`/`pgcrypto` extensions.
- **`platform` profile (Phase 2 & 6–8):** Starts `postgres`, `redis` (`127.0.0.1:63799`, `noeviction`), and single-broker KRaft `kafka` (`127.0.0.1:9094`).
- **`tracking` profile (Phase 4–6 & 9–12):** Starts `postgres` and `mlflow` (`127.0.0.1:5001`) with `--serve-artifacts` (`mlflow-artifacts:/`).
- **`pipelines` profile (Phase 9–12):** Starts `postgres` and `airflow` (`127.0.0.1:8088`, `LocalExecutor`, parallelism `2`) with `host.docker.internal` mapped to the native Windows host gateway.

## 2. Commands for Phase 1 (`db` Profile)

### Start PostgreSQL only (Phase 1 requirement)
```powershell
docker compose -f infra/compose.yml --profile db up -d
docker compose -f infra/compose.yml ps
```

### Verify Isolated Databases and `pgvector` Extension
```powershell
docker exec scratchllm-postgres psql -U llm_app_user -d llm_app -c "SELECT current_database(), current_user, extname FROM pg_extension WHERE extname IN ('vector', 'pgcrypto');"
```

### Run Backend Flyway Migrations and Integration Tests
```powershell
cd backend
.\mvnw.cmd test
```

### Safe Development Stop and Reset
- Stop running containers while preserving volumes:
  ```powershell
  docker compose -f infra/compose.yml --profile db --profile platform --profile tracking --profile pipelines stop
  ```
- Full clean reset (drops local development database/cache volumes):
  ```powershell
  docker compose -f infra/compose.yml --profile db --profile platform --profile tracking --profile pipelines down -v
  ```

## 3. Port and Security Policy

- Every container port in `infra/compose.yml` binds explicitly to `127.0.0.1` (`127.0.0.1:54322`, `127.0.0.1:63799`, `127.0.0.1:9094`, `127.0.0.1:5001`, `127.0.0.1:8088`) so local development services are never exposed on public LAN/Wi-Fi interfaces.
- Application JPA (`llm_app_user`) has no permissions on `mlflow_db` or `airflow_db`.
