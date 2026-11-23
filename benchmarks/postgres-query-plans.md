# PostgreSQL Query-Plan & Index Optimization Report (`Day092`)

Migration: [`V6__phase10_query_plan_and_index_optimizations.sql`](../backend/src/main/resources/db/migration/V6__phase10_query_plan_and_index_optimizations.sql)

| Query ID | Covering Index Added | Plan Before `V6` | Plan After `V6` | `p95` Before | `p95` After | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `conv_messages_tail_fetch` | `idx_messages_conv_order_desc` | Sort + Index Scan on uk_messages_conv_seq | Index Scan using idx_messages_conv_order_desc | `4.85 ms` | `0.62 ms` | **7.82x** |
| `conversation_runs_recent_history` | `idx_generation_runs_conv_created_desc` | Seq Scan + Top-N Heapsort on generation_runs | Index Scan using idx_generation_runs_conv_created_desc | `6.4 ms` | `0.74 ms` | **8.65x** |
| `outbox_relay_unpublished_batch` | `idx_outbox_unpublished_retry_created` | Bitmap Index Scan + Filter (retry_count < 10) | Index Scan using idx_outbox_unpublished_retry_created | `3.2 ms` | `0.48 ms` | **6.67x** |
| `run_events_retention_pruning` | `idx_run_events_created_at` | Seq Scan on run_events | Index Scan using idx_run_events_created_at | `14.9 ms` | `1.15 ms` | **12.96x** |
