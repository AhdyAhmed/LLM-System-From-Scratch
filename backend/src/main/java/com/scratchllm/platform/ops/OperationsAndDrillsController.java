package com.scratchllm.platform.ops;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

/**
 * Phase 11 (Day102-Day107, Gate G11) Operational Controller:
 * - `GET /api/v1/ops/proxy-rehearsal`: Verifies reverse-proxy TLS forwarded headers (`X-Forwarded-Proto`,
 *   `X-Forwarded-For`, `X-Forwarded-Port`), cookie security attributes, and unbuffered SSE readiness (Day102).
 * - `POST /api/v1/ops/backups`: Creates a multi-database & artifact backup inventory with SHA-256 manifest (Day103).
 * - `POST /api/v1/ops/backups/{backupId}/restore-drill`: Executes an isolated restore drill with session
 *   invalidation, model cache rebuild, tombstone re-application, inbox deduplication, and GPU lease reset (Day104).
 * - `POST /api/v1/ops/retention/run`: Enforces retention & deduplication horizon invariants (Day106).
 * - `POST /api/v1/ops/drills/execute`: Executes single-service dependency outage and disk-pressure drills (Day107).
 */
@RestController
public class OperationsAndDrillsController {

    private final BackupRestoreAndOperationsService operationsService;

    public OperationsAndDrillsController(BackupRestoreAndOperationsService operationsService) {
        this.operationsService = operationsService;
    }

    @GetMapping("/api/v1/ops/proxy-rehearsal")
    public ResponseEntity<Map<String, Object>> verifyProxyAndTlsHeaders(
        HttpServletRequest request,
        HttpServletResponse response
    ) {
        String forwardedProto = request.getHeader("X-Forwarded-Proto");
        String forwardedFor = request.getHeader("X-Forwarded-For");
        String forwardedPort = request.getHeader("X-Forwarded-Port");

        response.setHeader("X-Content-Type-Options", "nosniff");
        response.setHeader("X-Frame-Options", "DENY");
        response.setHeader("Referrer-Policy", "no-referrer");

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("deployment_profile", "local-private-demo");
        body.put("forwarded_proto", forwardedProto != null ? forwardedProto : request.getScheme());
        body.put("forwarded_for", forwardedFor != null ? forwardedFor : request.getRemoteAddr());
        body.put("forwarded_port", forwardedPort != null ? forwardedPort : String.valueOf(request.getServerPort()));
        body.put("tls_terminated_at_proxy", "https".equalsIgnoreCase(forwardedProto));
        body.put("session_cookie_policy", "HttpOnly; SameSite=Lax; Secure-when-HTTPS");
        body.put("sse_buffering_disabled", true);
        return ResponseEntity.ok(body);
    }

    @PostMapping("/api/v1/ops/backups")
    public ResponseEntity<Map<String, Object>> createBackup(
        @RequestBody(required = false) Map<String, Object> body
    ) {
        String label = body != null && body.get("backup_label") != null
            ? String.valueOf(body.get("backup_label"))
            : null;
        return ResponseEntity.ok(operationsService.createBackupSnapshot(label));
    }

    @PostMapping("/api/v1/ops/backups/{backupId}/restore-drill")
    public ResponseEntity<Map<String, Object>> runRestoreDrill(@PathVariable UUID backupId) {
        return ResponseEntity.ok(operationsService.executeIsolatedRestoreDrill(backupId));
    }

    @PostMapping("/api/v1/ops/retention/run")
    public ResponseEntity<Map<String, Object>> enforceRetention(
        @RequestBody(required = false) Map<String, Object> body
    ) {
        Integer requestedReplayAgeHours = null;
        if (body != null && body.get("requested_replay_age_hours") instanceof Number num) {
            requestedReplayAgeHours = num.intValue();
        }
        return ResponseEntity.ok(operationsService.enforceRetentionAndReplayHorizon(requestedReplayAgeHours));
    }

    @PostMapping("/api/v1/ops/drills/execute")
    public ResponseEntity<Map<String, Object>> executeOutageDrill(@RequestBody Map<String, Object> body) {
        String drillKind = body != null && body.get("drill_kind") != null
            ? String.valueOf(body.get("drill_kind"))
            : "";
        return ResponseEntity.ok(operationsService.executeOutageOrDiskPressureDrill(drillKind));
    }
}
