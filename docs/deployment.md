# Release Deployment Profile & Reverse-Proxy Rehearsal (`Day101`–`Day102`)

## 1. Selected Release Deployment Profile (`Day101`)

The ScratchLLM Platform uses the **`local-private-demo`** deployment profile by default:
- All data stores and operator consoles bind strictly to `127.0.0.1` (`localhost`) and are **never** exposed to public network interfaces.
- Zero paid cloud infrastructure is required; the entire system runs on a single developer workstation (`8 CPU threads`, `16 GB RAM`, `1x NVIDIA Quadro M2000M 4 GB VRAM` or CPU fallback, `80 GB SSD`).
- No public URL is claimed; local HTTP/HTTPS reverse-proxy termination is rehearsed via [`infra/nginx/nginx-local-tls-rehearsal.conf`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/infra/nginx/nginx-local-tls-rehearsal.conf) and [`infra/nginx/nginx-two-replica.conf`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/infra/nginx/nginx-two-replica.conf).

---

## 2. Port & Network Isolation Inventory

| Service | Host Bind Address & Port | Container Port | Exposure Tier | Authentication / Access Control |
| :--- | :--- | :--- | :--- | :--- |
| Reverse Proxy (TLS Rehearsal) | `127.0.0.1:8443` (`HTTPS`) / `127.0.0.1:8090` (`HTTP`) | N/A | Local Entrypoint | Session cookie (`HttpOnly`, `SameSite=Lax`, `Secure`) + CSRF (`X-CSRF-TOKEN`) |
| Spring Boot API (`replica-1`) | `127.0.0.1:8080` | N/A (Native) | Private App | Session + CSRF (`X-Internal-Service-Token` for `/internal/**`) |
| Spring Boot API (`replica-2`) | `127.0.0.1:8082` | N/A (Native) | Private App | Stateless round-robin peer sharing Redis & PostgreSQL |
| FastAPI Inference Gateway | `127.0.0.1:8091` | N/A (Native) | Private Internal | `X-Internal-Service-Token` required (`401` on missing/invalid) |
| PostgreSQL (`pgvector:pg16`) | `127.0.0.1:54322` | `5432` | Private Data | Separate DB users (`llm_app_user`, `mlflow_user`, `airflow_user`) |
| Redis (`7.4-alpine`) | `127.0.0.1:63799` | `6379` | Private State | Localhost-bound (`384 MB` `maxmemory`, `noeviction`) |
| Apache Kafka (`3.9.2` KRaft) | `127.0.0.1:9094` | `9092` / `9094` | Private Broker | Localhost-bound (`24h` / `512 MB` retention) |
| MLflow Tracking & Registry | `127.0.0.1:5001` | `5000` | Private MLOps | `--allowed-hosts localhost:*,127.0.0.1:*,mlflow:5000,host.docker.internal:*` |
| Apache Airflow (`3.2.2`) | `127.0.0.1:8088` | `8080` | Private MLOps | Localhost-bound (`LocalExecutor`, `max_active_runs_per_dag=1`) |

---

## 3. Reverse-Proxy TLS, Cookie, SSE & Forwarded-Header Verification (`Day102`)

Configuration: [`infra/nginx/nginx-local-tls-rehearsal.conf`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/infra/nginx/nginx-local-tls-rehearsal.conf) and endpoint `GET /api/v1/ops/proxy-rehearsal` in [`OperationsAndDrillsController.java`](file:///e:/backend-java/3-projects/7-LLM/central/LLM-System-From-Scratch/backend/src/main/java/com/scratchllm/platform/ops/OperationsAndDrillsController.java).

1. **Forwarded Headers**: Proxy sets `X-Forwarded-Proto https`, `X-Forwarded-For`, and `X-Forwarded-Port 8443`.
2. **Cookie Hardening**: `SESSION` cookie is `HttpOnly; SameSite=Lax` (and `Secure` under TLS proxy termination); `XSRF-TOKEN` cookie is paired with mandatory `X-CSRF-TOKEN` header on all state-changing requests.
3. **Unbuffered SSE Streaming**: `location ~ ^/api/v1/runs/[^/]+/events$` disables `proxy_buffering off;` and `proxy_cache off;`, sets `proxy_read_timeout 300s;`, and emits `X-Accel-Buffering: no` and `Cache-Control: no-cache, no-transform`.
4. **Security Response Headers**: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy: no-referrer`.
