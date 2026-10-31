package com.scratchllm.platform.generation;

import com.scratchllm.platform.generation.RunDtos.RunAcceptedResponse;
import com.scratchllm.platform.generation.RunDtos.RunEventRow;
import com.scratchllm.platform.generation.RunDtos.RunStatusResponse;
import com.scratchllm.platform.generation.RunDtos.SubmitRunRequest;
import com.scratchllm.platform.identity.AuthenticatedPrincipal;
import jakarta.annotation.PreDestroy;
import jakarta.validation.Valid;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * Day016 & Day018 REST and SSE endpoints for generation runs:
 * - POST /api/v1/conversations/{conversationId}/runs
 * - GET /api/v1/runs/{runId}
 * - GET /api/v1/runs/{runId}/events
 * - POST /api/v1/runs/{runId}/cancel
 */
@RestController
public class RunController {

    private static final Set<String> TERMINAL_EVENTS = Set.of(
        "run.completed",
        "run.cancelled",
        "run.failed"
    );

    private final RunService runService;
    private final ExecutorService sseExecutor = Executors.newCachedThreadPool();

    public RunController(RunService runService) {
        this.runService = runService;
    }

    @PreDestroy
    public void shutdown() {
        sseExecutor.shutdownNow();
    }

    @PostMapping("/api/v1/conversations/{conversationId}/runs")
    public ResponseEntity<RunAcceptedResponse> submitRun(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID conversationId,
        @RequestHeader(name = "Idempotency-Key") String idempotencyKey,
        @Valid @RequestBody SubmitRunRequest request
    ) {
        RunAcceptedResponse accepted = runService.submitRun(
            principal.userId(),
            conversationId,
            idempotencyKey,
            request
        );
        return ResponseEntity.status(HttpStatus.ACCEPTED).body(accepted);
    }

    @GetMapping("/api/v1/runs/{runId}")
    public ResponseEntity<RunStatusResponse> getRunStatus(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID runId
    ) {
        return ResponseEntity.ok(runService.getRunStatus(principal.userId(), runId));
    }

    @PostMapping("/api/v1/runs/{runId}/cancel")
    public ResponseEntity<RunStatusResponse> cancelRun(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID runId
    ) {
        return ResponseEntity.ok(runService.cancelRun(principal.userId(), runId));
    }

    @GetMapping(value = "/api/v1/runs/{runId}/events", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamRunEvents(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID runId,
        @RequestHeader(name = "Last-Event-ID", required = false) Long lastEventId,
        @RequestParam(name = "after", required = false) Long afterCursor
    ) {
        long initialCursor = 0L;
        if (lastEventId != null && lastEventId >= 0) {
            initialCursor = lastEventId;
        } else if (afterCursor != null && afterCursor >= 0) {
            initialCursor = afterCursor;
        }

        // Verify ownership synchronously before returning SseEmitter so 404 is returned immediately if unauthorized
        List<RunEventRow> initialBatch = runService.listRunEventsAfter(principal.userId(), runId, initialCursor);

        SseEmitter emitter = new SseEmitter(30_000L);
        final long startSeq = initialCursor;

        sseExecutor.submit(() -> {
            long cursor = startSeq;
            try {
                for (RunEventRow ev : initialBatch) {
                    sendSseEvent(emitter, ev);
                    cursor = ev.sequence();
                    if (TERMINAL_EVENTS.contains(ev.eventType())) {
                        emitter.complete();
                        return;
                    }
                }

                long deadlineMillis = System.currentTimeMillis() + 15_000L;
                while (System.currentTimeMillis() < deadlineMillis) {
                    List<RunEventRow> nextEvents = runService.listRunEventsAfter(principal.userId(), runId, cursor);
                    for (RunEventRow ev : nextEvents) {
                        sendSseEvent(emitter, ev);
                        cursor = ev.sequence();
                        if (TERMINAL_EVENTS.contains(ev.eventType())) {
                            emitter.complete();
                            return;
                        }
                    }
                    Thread.sleep(25L);
                }
                emitter.complete();
            } catch (Exception ex) {
                emitter.completeWithError(ex);
            }
        });

        return emitter;
    }

    private static void sendSseEvent(SseEmitter emitter, RunEventRow ev) throws IOException {
        emitter.send(
            SseEmitter.event()
                .id(String.valueOf(ev.sequence()))
                .name(ev.eventType())
                .data(ev.payloadJson(), MediaType.APPLICATION_JSON)
        );
    }
}
