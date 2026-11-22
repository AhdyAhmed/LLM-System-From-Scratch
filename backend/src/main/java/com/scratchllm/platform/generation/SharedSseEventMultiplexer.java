package com.scratchllm.platform.generation;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.scratchllm.platform.generation.RunDtos.RunEventRow;
import jakarta.annotation.PreDestroy;
import java.io.IOException;
import java.sql.Array;
import java.sql.Connection;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * Day066 & Day067: Shared SSE Event Multiplexer.
 *
 * Replaces per-connection database polling loops with:
 * - A single scheduled batched PostgreSQL poller per replica (`pollActiveRunsBatch`)
 * - Bounded per-subscriber outbound queues (`maxQueueCapacity = 64`)
 * - Automatic `resync.required` control events when a client reconnects with an expired/pruned
 *   cursor (`initialCursor < minAvailableSequence - 1`) or when a slow reader overflows its
 *   bounded outbound event queue
 * - Strict separation between SSE client disconnect (detaches subscriber only) and explicit
 *   run cancellation (`POST /api/v1/runs/{runId}/cancel`)
 */
@Service
public class SharedSseEventMultiplexer {

    public static final Set<String> TERMINAL_EVENTS = Set.of(
        "run.completed",
        "run.cancelled",
        "run.failed",
        "resync.required"
    );

    public static final int DEFAULT_SUBSCRIBER_QUEUE_CAPACITY = 64;

    private final JdbcTemplate jdbcTemplate;
    private final RunService runService;
    private final ObjectMapper objectMapper;
    private final ConcurrentHashMap<UUID, CopyOnWriteArrayList<SseSubscriber>> subscribersByRun = new ConcurrentHashMap<>();
    private final ScheduledExecutorService pollerScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "sse-multiplexer-poller");
        t.setDaemon(true);
        return t;
    });
    private final ExecutorService emitterFlushExecutor = Executors.newCachedThreadPool(r -> {
        Thread t = new Thread(r, "sse-multiplexer-flush");
        t.setDaemon(true);
        return t;
    });
    private final AtomicLong totalBatchPollQueries = new AtomicLong(0);
    private volatile int defaultSubscriberQueueCapacity;

    public SharedSseEventMultiplexer(
        JdbcTemplate jdbcTemplate,
        RunService runService,
        ObjectMapper objectMapper,
        @Value("${scratchllm.sse.poll-interval-ms:25}") long pollIntervalMs,
        @Value("${scratchllm.sse.subscriber-queue-capacity:64}") int subscriberQueueCapacity
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.runService = runService;
        this.objectMapper = objectMapper;
        this.defaultSubscriberQueueCapacity = Math.max(2, subscriberQueueCapacity);
        long safeInterval = Math.max(15L, pollIntervalMs);
        this.pollerScheduler.scheduleWithFixedDelay(this::pollActiveRunsBatchSafely, safeInterval, safeInterval, TimeUnit.MILLISECONDS);
    }

    @PreDestroy
    public void shutdown() {
        pollerScheduler.shutdownNow();
        emitterFlushExecutor.shutdownNow();
    }

    public void setDefaultSubscriberQueueCapacity(int capacity) {
        this.defaultSubscriberQueueCapacity = Math.max(1, capacity);
    }

    public int getDefaultSubscriberQueueCapacity() {
        return this.defaultSubscriberQueueCapacity;
    }

    public long getTotalBatchPollQueries() {
        return totalBatchPollQueries.get();
    }

    public int getActiveSubscriberCount() {
        int count = 0;
        for (CopyOnWriteArrayList<SseSubscriber> list : subscribersByRun.values()) {
            count += list.size();
        }
        return count;
    }

    /**
     * Subscribes an authenticated caller to SSE events for `runId` after `initialCursor`.
     * Ownership is verified synchronously before creating the `SseEmitter` so unauthorized
     * callers receive HTTP 404 immediately.
     */
    public SseEmitter subscribe(UUID userId, UUID runId, long initialCursor) {
        return subscribeWithCustomCapacity(userId, runId, initialCursor, defaultSubscriberQueueCapacity, 0L);
    }

    /**
     * Allows reliability tests to subscribe with a custom outbound queue capacity or simulated
     * per-event consumer write delay (to test slow-reader overflow bounding).
     */
    public SseEmitter subscribeWithCustomCapacity(
        UUID userId,
        UUID runId,
        long initialCursor,
        int queueCapacity,
        long simulatedWriteDelayMs
    ) {
        // 1. Synchronously verify ownership (throws 404 RUN_NOT_FOUND if not owned)
        runService.verifyRunOwnership(userId, runId);

        SseEmitter emitter = new SseEmitter(30_000L);

        // 2. Check cursor expiry if initialCursor > 0 (or if events were pruned past 24h retention)
        if (initialCursor > 0 && isCursorExpired(runId, initialCursor)) {
            pollerScheduler.schedule(
                () -> emitterFlushExecutor.submit(() -> sendResyncRequiredDirect(emitter, runId, initialCursor, "CURSOR_EXPIRED")),
                10L,
                TimeUnit.MILLISECONDS
            );
            return emitter;
        }

        SseSubscriber subscriber = new SseSubscriber(
            UUID.randomUUID(),
            runId,
            emitter,
            initialCursor,
            Math.max(1, queueCapacity),
            Math.max(0L, simulatedWriteDelayMs)
        );

        // Register lifecycle callbacks: client disconnect ONLY detaches subscriber, NEVER cancels the run
        emitter.onCompletion(() -> unsubscribe(subscriber));
        emitter.onTimeout(() -> {
            unsubscribe(subscriber);
            emitter.complete();
        });
        emitter.onError(ex -> unsubscribe(subscriber));

        // 3. Fetch and enqueue initial catch-up events strictly in sequence order BEFORE exposing subscriber
        //    to pollActiveRunsBatch so concurrent poller ticks cannot advance lastQueuedSequence ahead of sequence 1.
        List<RunEventRow> catchUpEvents = runService.listRunEventsAfter(userId, runId, initialCursor);

        for (RunEventRow ev : catchUpEvents) {
            if (ev.sequence() <= subscriber.getLastQueuedSequence()) {
                continue;
            }
            if (!subscriber.offerEvent(ev)) {
                subscriber.markOverflowed();
                break;
            }
        }

        if (!subscriber.isOverflowed()) {
            subscribersByRun.compute(runId, (k, list) -> {
                CopyOnWriteArrayList<SseSubscriber> target = (list != null) ? list : new CopyOnWriteArrayList<>();
                target.add(subscriber);
                return target;
            });
        }

        pollerScheduler.schedule(() -> scheduleSubscriberDrain(subscriber), 10L, TimeUnit.MILLISECONDS);
        return emitter;
    }

    /**
     * Prunes `run_events` older than `retentionAge` (or sequences `<= upToSequence` for testing cursor expiry).
     */
    public int pruneRunEventsOlderThan(Duration retentionAge) {
        long seconds = Math.max(1L, retentionAge.toSeconds());
        return jdbcTemplate.update(
            "DELETE FROM run_events WHERE created_at < clock_timestamp() - (? * interval '1 second')",
            seconds
        );
    }

    public int pruneRunEventsUpToSequence(UUID runId, long maxSequenceToDelete) {
        return jdbcTemplate.update(
            "DELETE FROM run_events WHERE run_id = ? AND sequence <= ?",
            runId, maxSequenceToDelete
        );
    }

    private boolean isCursorExpired(UUID runId, long initialCursor) {
        List<MinMaxSeqRow> rows = jdbcTemplate.query(
            "SELECT MIN(sequence) AS min_seq, MAX(sequence) AS max_seq FROM run_events WHERE run_id = ?",
            (rs, rowNum) -> {
                long min = rs.getLong("min_seq");
                boolean minNull = rs.wasNull();
                long max = rs.getLong("max_seq");
                boolean maxNull = rs.wasNull();
                return new MinMaxSeqRow(minNull ? null : min, maxNull ? null : max);
            },
            runId
        );
        if (rows.isEmpty() || rows.getFirst().minSeq() == null) {
            // All events for this run were pruned while client requested initialCursor > 0
            return true;
        }
        long minAvailableSequence = rows.getFirst().minSeq();
        // If minAvailableSequence is e.g. 5 and client asks for events after cursor 2, sequences 3 and 4 are missing!
        return initialCursor < (minAvailableSequence - 1L);
    }

    private void pollActiveRunsBatchSafely() {
        try {
            pollActiveRunsBatch();
        } catch (Exception ignored) {
        }
    }

    /**
     * Executes ONE batched PostgreSQL query across all active subscribed `runId`s and multiplexes
     * newly persisted `run_events` rows into each subscriber's bounded queue.
     */
    public void pollActiveRunsBatch() {
        if (subscribersByRun.isEmpty()) {
            return;
        }

        Map<UUID, Long> minCursorByRun = new HashMap<>();
        for (Map.Entry<UUID, CopyOnWriteArrayList<SseSubscriber>> entry : subscribersByRun.entrySet()) {
            CopyOnWriteArrayList<SseSubscriber> subs = entry.getValue();
            if (subs.isEmpty()) {
                subscribersByRun.remove(entry.getKey(), subs);
                continue;
            }
            long minSeq = Long.MAX_VALUE;
            for (SseSubscriber sub : subs) {
                if (!sub.isClosed()) {
                    minSeq = Math.min(minSeq, sub.getLastQueuedSequence());
                }
            }
            if (minSeq != Long.MAX_VALUE) {
                minCursorByRun.put(entry.getKey(), minSeq);
            }
        }

        if (minCursorByRun.isEmpty()) {
            return;
        }

        long globalMinSeq = minCursorByRun.values().stream().mapToLong(Long::longValue).min().orElse(0L);
        UUID[] activeRunIds = minCursorByRun.keySet().toArray(new UUID[0]);

        totalBatchPollQueries.incrementAndGet();

        List<RunScopedEventRow> batchedEvents = jdbcTemplate.execute(
            (Connection conn) -> {
                Array sqlArray = conn.createArrayOf("uuid", activeRunIds);
                try (var ps = conn.prepareStatement(
                    """
                    SELECT run_id, sequence, event_type, payload_json::text AS payload_json
                    FROM run_events
                    WHERE run_id = ANY (?) AND sequence > ?
                    ORDER BY run_id, sequence ASC
                    """
                )) {
                    ps.setArray(1, sqlArray);
                    ps.setLong(2, globalMinSeq);
                    try (var rs = ps.executeQuery()) {
                        List<RunScopedEventRow> list = new ArrayList<>();
                        while (rs.next()) {
                            list.add(new RunScopedEventRow(
                                UUID.fromString(rs.getString("run_id")),
                                rs.getLong("sequence"),
                                rs.getString("event_type"),
                                rs.getString("payload_json")
                            ));
                        }
                        return list;
                    }
                } finally {
                    sqlArray.free();
                }
            }
        );

        if (batchedEvents == null || batchedEvents.isEmpty()) {
            return;
        }

        for (RunScopedEventRow row : batchedEvents) {
            CopyOnWriteArrayList<SseSubscriber> subs = subscribersByRun.get(row.runId());
            if (subs == null || subs.isEmpty()) {
                continue;
            }
            RunEventRow ev = new RunEventRow(row.sequence(), row.eventType(), row.payloadJson());
            for (SseSubscriber sub : subs) {
                if (sub.isClosed() || row.sequence() <= sub.getLastQueuedSequence()) {
                    continue;
                }
                if (!sub.offerEvent(ev)) {
                    // Slow reader overflowed its bounded queue -> detach and emit resync.required
                    sub.markOverflowed();
                    unsubscribe(sub);
                }
                scheduleSubscriberDrain(sub);
            }
        }
    }

    private void scheduleSubscriberDrain(SseSubscriber subscriber) {
        if (!subscriber.tryAcquireDrainLock()) {
            return;
        }
        emitterFlushExecutor.submit(() -> {
            try {
                while (!subscriber.isClosed()) {
                    if (subscriber.isOverflowed()) {
                        sendResyncRequiredDirect(
                            subscriber.emitter(),
                            subscriber.runId(),
                            subscriber.getLastSentSequence(),
                            "SLOW_READER_BUFFER_OVERFLOW"
                        );
                        subscriber.close();
                        unsubscribe(subscriber);
                        return;
                    }

                    RunEventRow next = subscriber.pollEvent();
                    if (next == null) {
                        break;
                    }

                    if (subscriber.simulatedWriteDelayMs() > 0L) {
                        Thread.sleep(subscriber.simulatedWriteDelayMs());
                    }

                    sendSseEvent(subscriber.emitter(), next);
                    subscriber.setLastSentSequence(next.sequence());

                    if (TERMINAL_EVENTS.contains(next.eventType())) {
                        subscriber.close();
                        unsubscribe(subscriber);
                        subscriber.emitter().complete();
                        return;
                    }
                }
            } catch (Exception ex) {
                subscriber.close();
                unsubscribe(subscriber);
                try {
                    subscriber.emitter().complete();
                } catch (Exception ignored) {
                }
            } finally {
                subscriber.releaseDrainLock();
                if (!subscriber.isClosed() && (!subscriber.isQueueEmpty() || subscriber.isOverflowed())) {
                    scheduleSubscriberDrain(subscriber);
                }
            }
        });
    }

    private void unsubscribe(SseSubscriber subscriber) {
        subscriber.close();
        subscribersByRun.computeIfPresent(subscriber.runId(), (k, list) -> {
            list.remove(subscriber);
            return list.isEmpty() ? null : list;
        });
    }

    private void sendResyncRequiredDirect(SseEmitter emitter, UUID runId, long lastEventId, String reason) {
        try {
            String payload = toJson(Map.of(
                "run_id", runId.toString(),
                "schema_version", 1,
                "reason", reason,
                "last_event_id", lastEventId,
                "snapshot_url", "/api/v1/runs/" + runId
            ));
            emitter.send(
                SseEmitter.event()
                    .name("resync.required")
                    .data(payload, MediaType.APPLICATION_JSON)
            );
            emitter.complete();
        } catch (Exception ex) {
            emitter.completeWithError(ex);
        }
    }

    private static void sendSseEvent(SseEmitter emitter, RunEventRow ev) throws IOException {
        emitter.send(
            SseEmitter.event()
                .id(String.valueOf(ev.sequence()))
                .name(ev.eventType())
                .data(ev.payloadJson(), MediaType.APPLICATION_JSON)
        );
    }

    private String toJson(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException ex) {
            throw new IllegalStateException("Failed to serialize JSON", ex);
        }
    }

    private record MinMaxSeqRow(Long minSeq, Long maxSeq) {}
    private record RunScopedEventRow(UUID runId, long sequence, String eventType, String payloadJson) {}

    private static final class SseSubscriber {
        private final UUID subscriberId;
        private final UUID runId;
        private final SseEmitter emitter;
        private final ArrayBlockingQueue<RunEventRow> outboundQueue;
        private final long simulatedWriteDelayMs;
        private final AtomicLong lastQueuedSequence;
        private final AtomicLong lastSentSequence;
        private final AtomicBoolean draining = new AtomicBoolean(false);
        private final AtomicBoolean closed = new AtomicBoolean(false);
        private final AtomicBoolean overflowed = new AtomicBoolean(false);

        SseSubscriber(
            UUID subscriberId,
            UUID runId,
            SseEmitter emitter,
            long initialCursor,
            int queueCapacity,
            long simulatedWriteDelayMs
        ) {
            this.subscriberId = subscriberId;
            this.runId = runId;
            this.emitter = emitter;
            this.outboundQueue = new ArrayBlockingQueue<>(queueCapacity);
            this.simulatedWriteDelayMs = simulatedWriteDelayMs;
            this.lastQueuedSequence = new AtomicLong(initialCursor);
            this.lastSentSequence = new AtomicLong(initialCursor);
        }

        UUID runId() {
            return runId;
        }

        SseEmitter emitter() {
            return emitter;
        }

        long simulatedWriteDelayMs() {
            return simulatedWriteDelayMs;
        }

        boolean offerEvent(RunEventRow ev) {
            if (closed.get() || overflowed.get()) {
                return false;
            }
            boolean added = outboundQueue.offer(ev);
            if (added) {
                lastQueuedSequence.set(ev.sequence());
            }
            return added;
        }

        RunEventRow pollEvent() {
            return outboundQueue.poll();
        }

        boolean isQueueEmpty() {
            return outboundQueue.isEmpty();
        }

        long getLastQueuedSequence() {
            return lastQueuedSequence.get();
        }

        long getLastSentSequence() {
            return lastSentSequence.get();
        }

        void setLastSentSequence(long seq) {
            lastSentSequence.set(seq);
        }

        boolean tryAcquireDrainLock() {
            return draining.compareAndSet(false, true);
        }

        void releaseDrainLock() {
            draining.set(false);
        }

        boolean isClosed() {
            return closed.get();
        }

        void close() {
            closed.set(true);
        }

        boolean isOverflowed() {
            return overflowed.get();
        }

        void markOverflowed() {
            overflowed.set(true);
        }
    }
}
