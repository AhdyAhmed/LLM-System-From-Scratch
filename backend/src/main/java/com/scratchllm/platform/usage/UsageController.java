package com.scratchllm.platform.usage;

import com.scratchllm.platform.identity.AuthenticatedPrincipal;
import com.scratchllm.platform.platform.ApiException;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.util.List;
import java.util.UUID;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * Exposes GET /api/v1/usage for caller-scoped daily token allowance, reservations, and settled usage.
 */
@RestController
public class UsageController {

    private final JdbcTemplate jdbcTemplate;
    private final int defaultDailyAllowance;

    public UsageController(
        JdbcTemplate jdbcTemplate,
        @Value("${scratchllm.quota.daily-allowance-tokens:4096}") int defaultDailyAllowance
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.defaultDailyAllowance = defaultDailyAllowance;
    }

    @GetMapping("/api/v1/usage")
    public ResponseEntity<UsageSummaryResponse> getUsage(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @RequestParam(required = false) UUID workspaceId
    ) {
        UUID resolvedWorkspaceId = workspaceId != null ? workspaceId : resolvePrimaryWorkspaceId(principal.userId());
        LocalDate todayUtc = LocalDate.now(ZoneOffset.UTC);

        List<UsageSummaryResponse> rows = jdbcTemplate.query(
            """
            SELECT daily_allowance_tokens, reserved_tokens, settled_tokens
            FROM usage_buckets
            WHERE workspace_id = ? AND user_id = ? AND utc_date = ?
            """,
            (rs, rowNum) -> {
                int limit = rs.getInt("daily_allowance_tokens");
                int reserved = rs.getInt("reserved_tokens");
                int settled = rs.getInt("settled_tokens");
                int remaining = Math.max(0, limit - reserved - settled);
                return new UsageSummaryResponse(todayUtc, limit, reserved, settled, remaining);
            },
            resolvedWorkspaceId, principal.userId(), todayUtc
        );

        if (rows.isEmpty()) {
            return ResponseEntity.ok(new UsageSummaryResponse(
                todayUtc,
                defaultDailyAllowance,
                0,
                0,
                defaultDailyAllowance
            ));
        }
        return ResponseEntity.ok(rows.getFirst());
    }

    private UUID resolvePrimaryWorkspaceId(UUID userId) {
        List<UUID> workspaces = jdbcTemplate.query(
            "SELECT workspace_id FROM memberships WHERE user_id = ? ORDER BY created_at ASC LIMIT 1",
            (rs, rowNum) -> UUID.fromString(rs.getString("workspace_id")),
            userId
        );
        if (workspaces.isEmpty()) {
            throw new ApiException(HttpStatus.NOT_FOUND, "WORKSPACE_NOT_FOUND", "No workspace membership found.");
        }
        return workspaces.getFirst();
    }

    public record UsageSummaryResponse(
        LocalDate utc_date,
        int daily_limit_tokens,
        int reserved_tokens,
        int settled_tokens,
        int remaining_tokens
    ) {}
}
