# Multi-Replica API Scaling, Overload & Mixed-Workload Report (`Day093`–`Day096`)

## 1. One-Replica vs Two-Replica Stateless Scaling (`fake-deterministic`)

| Metric | 1 Replica (`replica-1`) | 2 Replicas (`replica-1` + `replica-2`) | Delta / Notes |
| :--- | :--- | :--- | :--- |
| Throughput (`RPS`) | `142.5 req/s` | `248.0 req/s` | **1.74x** scaling efficiency |
| Run Creation `p50` | `11.4 ms` | `7.2 ms` | Shared Redis session + Lua rate limiter |
| Run Creation `p95` | `24.8 ms` | `15.6 ms` | Zero sticky-session dependency |
| Run Creation `p99` | `33.1 ms` | `21.4 ms` | Unbuffered SSE (`X-Accel-Buffering: no`) |
| SSE First Event `p95` | `39.5 ms` | `28.9 ms` | Batched PostgreSQL `run_events` multiplexer |

## 2. Explicit Queue Cap & Overload Rejection (`Day094`)

- Configured active slots: `1` | Configured queue cap: `8`
- Burst submitted: `12` concurrent runs -> `9` accepted (`202 Accepted`), `3` fast-rejected (`503 SYSTEM_CAPACITY_EXCEEDED`, `p95 = 3.8 ms`).

## 3. Mixed Workload Isolation (`Day096`)

- Slow SSE consumer (`250 ms` read delay) caused **zero head-of-line blocking** for fast consumers (`p95 = 19.8 ms`).
- Concurrent Kafka document ingestion indexed `12` `pgvector` chunks while active chat runs streamed cleanly.
- Single-GPU mutual exclusion rejected `scratch-mini` runs with `503 GPU_BUSY_WITH_TRAINING` while offline Airflow training held `gpu_leases (id=1)`, while allowing `fake-deterministic` runs (`202 Accepted`).
