# Pinned Environment, Toolchain, and Resource Baseline

Status: Verified Local Baseline (Day001 / Day002 — Phase 1 Gate G1)
Companion documents: [scope.md](scope.md) | [local-setup.md](local-setup.md) | [design.md](../design.md)

## 1. Observed Host Hardware and Operating System

| Component | Verified Value | Design & Operational Consequence |
| --- | --- | --- |
| Host OS | Windows 10/11 64-bit (`NT 10.0`) with WSL2 + Docker Desktop `27.4.0` | Run GPU-bound Python training/inference natively on Windows; run CPU infrastructure in bounded Linux containers |
| CPU | Intel Core i7-6820HQ (4 physical cores / 8 logical processors @ 2.70 GHz) | Cap container CPU shares; limit PyTorch `num_workers` (`0`–`2`) and Airflow parallelism (`2`) |
| System RAM | `47.9 GiB` physical memory | Sufficient for staged profiles; enforce explicit container memory limits and JVM `-Xmx768m` |
| GPU | NVIDIA Quadro M2000M (`4,096 MiB` VRAM, Maxwell architecture `sm_50`, driver `573.71`) | Use FP32 manual attention; no BF16/FlashAttention2 requirement; enforce 1 active GPU owner with subprocess CUDA context reclamation |
| Storage Snapshot | `C:` `21.31 GiB` free; `E:` `16.65 GiB` free | Enforce a `2 GiB` processed dataset cap and `2 GiB` model artifact cap; reuse locally cached Docker images |

## 2. Pinned Language Runtimes and Frameworks

| Layer | Pinned Version | Verification / Lock Mechanism |
| --- | --- | --- |
| Java Runtime | OpenJDK `21.0.12` (Temurin-21.0.12+8 LTS) | Enforced in `backend/pom.xml` (`<java.version>21</java.version>`) |
| Maven Build | Apache Maven `3.9.16` (via `backend/mvnw` / `backend/mvnw.cmd`) | `backend/.mvn/wrapper/maven-wrapper.properties` |
| Spring Boot | `3.5.16` (`spring-boot-starter-parent`) | Pinned in `backend/pom.xml`; manages Spring MVC, Security, Data JPA, Redis, Kafka, Actuator, and Flyway |
| Python Runtime | Python `3.12.0` | Verified via `scripts/probe_environment.py` and `ml/pyproject.toml` (`requires-python = ">=3.12,<3.13"`) |
| PyTorch | `2.6.0+cu118` (compiled with CUDA `11.8`, includes `sm_50`) | Verified forward/backward FP32 smoke test on both `cpu` and `cuda:0` |
| Python Libraries | `numpy==1.26.4`, `fastapi==0.115.0`, `pydantic==2.9.2`, `uvicorn==0.30.6`, `httpx==0.28.1`, `httpx-sse==0.4.0`, `jsonschema==4.23.0`, `PyYAML==6.0.2`, `psutil==7.2.2`, `pytest==9.1.1` | Pinned in `ml/requirements-lock.txt` and `ml/pyproject.toml` |

## 3. Pinned Infrastructure Container Images

All required infrastructure images are pinned to immutable tags already verified in the local Docker image cache (`0 B` additional network pull required):

| Service | Pinned Image Tag | Compose Profile | Resource Cap (CPU / RAM) | Host Port Binding |
| --- | --- | --- | --- | --- |
| PostgreSQL 16 + `pgvector` | `pgvector/pgvector:pg16` | `platform` | `1.0 CPU` / `2048M` | `127.0.0.1:54322:5432` |
| Redis 7 | `redis:7.4-alpine` | `platform` | `0.5 CPU` / `512M` (`noeviction`) | `127.0.0.1:63799:6379` |
| Apache Kafka (KRaft) | `apache/kafka:3.9.2` | `platform` | `1.0 CPU` / `1536M` (`-Xmx768M`) | `127.0.0.1:9094:9094` |
| MLflow Tracking/Registry | `pymk-mlflow:3.17.1` (or python-3.12 MLflow image) | `tracking` | `0.5 CPU` / `1024M` | `127.0.0.1:5001:5000` |
| Apache Airflow (`LocalExecutor`) | `apache/airflow:3.2.2-python3.12` | `pipelines` | `1.5 CPU` / `2560M` | `127.0.0.1:8088:8080` |

## 4. Supported CPU Fallback Policy

1. Every test suite (`mvn test` and `pytest`), model layer, tokenizer, training loop, and FastAPI inference endpoint supports deterministic execution on `device="cpu"`.
2. If CUDA `sm_50` driver state or VRAM pressure blocks a run, setting `SCRATCHLM_DEVICE=cpu` runs the identical `tiny` (`0.85M` parameter) model profile on CPU without code changes or skipped correctness gates.
