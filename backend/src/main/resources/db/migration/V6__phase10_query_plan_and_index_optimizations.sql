-- Day092: Phase 10 PostgreSQL Query Plan & Index Optimizations
-- Optimizes keyset message pagination, conversation run lookups, outbox relay polling,
-- and run_events retention pruning without holding transactions across network generation.

-- 1. Keyset pagination index for messages (conversation_id, message_order DESC)
CREATE INDEX IF NOT EXISTS idx_messages_conv_order_desc
    ON messages (conversation_id, message_order DESC);

-- 2. Composite index for conversation run history & latest status lookup
CREATE INDEX IF NOT EXISTS idx_generation_runs_conv_created_desc
    ON generation_runs (conversation_id, created_at DESC);

-- 3. Partial composite index for Outbox Relay batch polling (unpublished & eligible for retry)
CREATE INDEX IF NOT EXISTS idx_outbox_unpublished_retry_created
    ON outbox_events (next_retry_at ASC, created_at ASC)
    WHERE published_at IS NULL;

-- 4. Index on run_events (created_at) to accelerate bounded retention pruning (Day099)
CREATE INDEX IF NOT EXISTS idx_run_events_created_at
    ON run_events (created_at ASC);
