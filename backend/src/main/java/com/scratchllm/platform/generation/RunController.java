package com.scratchllm.platform.generation;

import com.scratchllm.platform.generation.OutboxRelayAndInboxService.DlqEventRow;
import com.scratchllm.platform.generation.OutboxRelayAndInboxService.DlqReplayResult;
import com.scratchllm.platform.generation.RunDtos.RunAcceptedResponse;
import com.scratchllm.platform.generation.RunDtos.RunStatusResponse;
import com.scratchllm.platform.generation.RunDtos.SubmitRunRequest;
import com.scratchllm.platform.identity.AuthenticatedPrincipal;
import jakarta.validation.Valid;
import java.util.List;
import java.util.Map;
import java.util.UUID;
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
 * Day016, Day018, Day066 & Day067 REST and SSE endpoints for generation runs and DLQ operations:
 * - POST /api/v1/conversations/{conversationId}/runs
 * - GET /api/v1/runs/{runId}
 * - GET /api/v1/runs/{runId}/events (multiplexed via SharedSseEventMultiplexer)
 * - POST /api/v1/runs/{runId}/cancel
 * - GET /api/v1/admin/dlq
 * - POST /api/v1/admin/dlq/{eventId}/replay
 */
@RestController
public class RunController {

    private final RunService runService;
    private final SharedSseEventMultiplexer sharedSseEventMultiplexer;
    private final OutboxRelayAndInboxService outboxRelayAndInboxService;

    public RunController(
        RunService runService,
        SharedSseEventMultiplexer sharedSseEventMultiplexer,
        OutboxRelayAndInboxService outboxRelayAndInboxService
    ) {
        this.runService = runService;
        this.sharedSseEventMultiplexer = sharedSseEventMultiplexer;
        this.outboxRelayAndInboxService = outboxRelayAndInboxService;
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
        @RequestParam(name = "after", required = false) Long afterCursor,
        @RequestParam(name = "max_queue_capacity", required = false) Integer maxQueueCapacity,
        @RequestParam(name = "simulated_write_delay_ms", required = false) Long simulatedWriteDelayMs,
        jakarta.servlet.http.HttpServletResponse response
    ) {
        // Day093: Disable reverse-proxy response buffering for SSE streaming across replicas
        response.setHeader("X-Accel-Buffering", "no");
        response.setHeader("Cache-Control", "no-cache, no-transform");

        long initialCursor = 0L;
        if (lastEventId != null && lastEventId >= 0) {
            initialCursor = lastEventId;
        } else if (afterCursor != null && afterCursor >= 0) {
            initialCursor = afterCursor;
        }

        if (maxQueueCapacity != null || simulatedWriteDelayMs != null) {
            return sharedSseEventMultiplexer.subscribeWithCustomCapacity(
                principal.userId(),
                runId,
                initialCursor,
                maxQueueCapacity != null ? maxQueueCapacity : sharedSseEventMultiplexer.getDefaultSubscriberQueueCapacity(),
                simulatedWriteDelayMs != null ? simulatedWriteDelayMs : 0L
            );
        }

        return sharedSseEventMultiplexer.subscribe(principal.userId(), runId, initialCursor);
    }

    @GetMapping("/api/v1/admin/dlq")
    public ResponseEntity<Map<String, List<DlqEventRow>>> listDlqEvents() {
        return ResponseEntity.ok(Map.of("items", outboxRelayAndInboxService.listDlqEvents()));
    }

    @PostMapping("/api/v1/admin/dlq/{eventId}/replay")
    public ResponseEntity<DlqReplayResult> replayDlqEvent(@PathVariable UUID eventId) {
        return ResponseEntity.ok(outboxRelayAndInboxService.replayDlqEvent(eventId));
    }
}
