package com.scratchllm.platform.conversation;

import com.scratchllm.platform.conversation.ConversationDtos.ConversationPageResponse;
import com.scratchllm.platform.conversation.ConversationDtos.ConversationSummaryResponse;
import com.scratchllm.platform.conversation.ConversationDtos.CreateConversationRequest;
import com.scratchllm.platform.conversation.ConversationDtos.MessageDto;
import com.scratchllm.platform.conversation.ConversationDtos.MessagePageResponse;
import com.scratchllm.platform.conversation.ConversationDtos.RenameConversationRequest;
import com.scratchllm.platform.platform.ApiException;
import java.nio.charset.StandardCharsets;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.Base64;
import java.util.List;
import java.util.UUID;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Day013 & Day014: Creator-private conversation CRUD, membership + owner scoping,
 * optimistic locking, tombstone deletion, and keyset pagination for conversations and messages.
 */
@Service
public class ConversationService {

    private final JdbcTemplate jdbcTemplate;

    public ConversationService(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    @Transactional
    public ConversationSummaryResponse createConversation(
        UUID userId,
        UUID workspaceId,
        CreateConversationRequest request
    ) {
        verifyWorkspaceMembership(userId, workspaceId);

        UUID conversationId = UUID.randomUUID();
        String trimmedTitle = request.title().trim();

        return jdbcTemplate.queryForObject(
            """
            INSERT INTO conversations (id, workspace_id, owner_user_id, title, version)
            VALUES (?, ?, ?, ?, 0)
            RETURNING id, workspace_id, owner_user_id, title, version, updated_at
            """,
            (rs, rowNum) -> mapConversationSummary(rs),
            conversationId, workspaceId, userId, trimmedTitle
        );
    }

    @Transactional(readOnly = true)
    public ConversationSummaryResponse getConversation(UUID userId, UUID conversationId) {
        List<ConversationSummaryResponse> rows = jdbcTemplate.query(
            """
            SELECT id, workspace_id, owner_user_id, title, version, updated_at
            FROM conversations
            WHERE id = ? AND owner_user_id = ? AND deleted_at IS NULL
            """,
            (rs, rowNum) -> mapConversationSummary(rs),
            conversationId, userId
        );
        if (rows.isEmpty()) {
            throw conversationNotFound();
        }
        return rows.getFirst();
    }

    @Transactional(readOnly = true)
    public ConversationPageResponse listConversations(
        UUID userId,
        UUID workspaceId,
        String cursor,
        int limit
    ) {
        if (limit < 1 || limit > 50) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "INVALID_LIMIT", "Limit must be between 1 and 50.");
        }
        verifyWorkspaceMembership(userId, workspaceId);

        int fetchSize = limit + 1;
        List<ConversationSummaryResponse> fetched;

        if (cursor == null || cursor.isBlank()) {
            fetched = jdbcTemplate.query(
                """
                SELECT id, workspace_id, owner_user_id, title, version, updated_at
                FROM conversations
                WHERE workspace_id = ? AND owner_user_id = ? AND deleted_at IS NULL
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (rs, rowNum) -> mapConversationSummary(rs),
                workspaceId, userId, fetchSize
            );
        } else {
            ConversationCursor decoded = decodeConversationCursor(cursor);
            Timestamp cursorTs = Timestamp.from(decoded.updatedAt().toInstant());
            fetched = jdbcTemplate.query(
                """
                SELECT id, workspace_id, owner_user_id, title, version, updated_at
                FROM conversations
                WHERE workspace_id = ? AND owner_user_id = ? AND deleted_at IS NULL
                  AND (updated_at < ? OR (updated_at = ? AND id < ?))
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (rs, rowNum) -> mapConversationSummary(rs),
                workspaceId, userId, cursorTs, cursorTs, decoded.id(), fetchSize
            );
        }

        boolean hasMore = fetched.size() > limit;
        List<ConversationSummaryResponse> items = hasMore ? fetched.subList(0, limit) : fetched;
        String nextCursor = null;
        if (hasMore && !items.isEmpty()) {
            ConversationSummaryResponse last = items.getLast();
            nextCursor = encodeConversationCursor(last.updated_at(), last.conversation_id());
        }
        return new ConversationPageResponse(items, nextCursor, hasMore);
    }

    @Transactional
    public ConversationSummaryResponse renameConversation(
        UUID userId,
        UUID conversationId,
        RenameConversationRequest request
    ) {
        List<Long> existingVersion = jdbcTemplate.query(
            "SELECT version FROM conversations WHERE id = ? AND owner_user_id = ? AND deleted_at IS NULL",
            (rs, rowNum) -> rs.getLong("version"),
            conversationId, userId
        );
        if (existingVersion.isEmpty()) {
            throw conversationNotFound();
        }

        List<ConversationSummaryResponse> updated = jdbcTemplate.query(
            """
            UPDATE conversations
            SET title = ?, version = version + 1, updated_at = clock_timestamp()
            WHERE id = ? AND owner_user_id = ? AND deleted_at IS NULL AND version = ?
            RETURNING id, workspace_id, owner_user_id, title, version, updated_at
            """,
            (rs, rowNum) -> mapConversationSummary(rs),
            request.title().trim(), conversationId, userId, request.version()
        );
        if (updated.isEmpty()) {
            throw new ApiException(
                HttpStatus.CONFLICT,
                "VERSION_CONFLICT",
                "Conversation version does not match current version."
            );
        }
        return updated.getFirst();
    }

    @Transactional
    public void deleteConversation(UUID userId, UUID conversationId) {
        int updated = jdbcTemplate.update(
            """
            UPDATE conversations
            SET deleted_at = clock_timestamp(), version = version + 1, updated_at = clock_timestamp()
            WHERE id = ? AND owner_user_id = ? AND deleted_at IS NULL
            """,
            conversationId, userId
        );
        if (updated == 0) {
            throw conversationNotFound();
        }

        // Cancel any active run on this conversation and refund its reservation
        List<UUID> activeRuns = jdbcTemplate.query(
            """
            UPDATE generation_runs
            SET status = 'CANCELLED',
                cancel_requested = TRUE,
                finish_reason = 'cancelled',
                completed_at = clock_timestamp(),
                updated_at = clock_timestamp()
            WHERE conversation_id = ? AND status IN ('QUEUED', 'RUNNING')
            RETURNING id
            """,
            (rs, rowNum) -> UUID.fromString(rs.getString("id")),
            conversationId
        );

        for (UUID runId : activeRuns) {
            jdbcTemplate.update(
                "UPDATE system_capacity SET active_or_waiting_runs = GREATEST(0, active_or_waiting_runs - 1), updated_at = clock_timestamp() WHERE id = 1"
            );
            jdbcTemplate.query(
                """
                UPDATE usage_reservations
                SET status = 'REFUNDED', settled_tokens = 0, settled_at = clock_timestamp()
                WHERE run_id = ? AND status = 'RESERVED'
                RETURNING bucket_id, reserved_tokens
                """,
                (rs, rowNum) -> {
                    UUID bucketId = UUID.fromString(rs.getString("bucket_id"));
                    int reserved = rs.getInt("reserved_tokens");
                    jdbcTemplate.update(
                        "UPDATE usage_buckets SET reserved_tokens = GREATEST(0, reserved_tokens - ?), updated_at = clock_timestamp() WHERE id = ?",
                        reserved, bucketId
                    );
                    return bucketId;
                },
                runId
            );
        }
    }

    @Transactional(readOnly = true)
    public MessagePageResponse listMessages(
        UUID userId,
        UUID conversationId,
        String cursor,
        int limit
    ) {
        if (limit < 1 || limit > 100) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "INVALID_LIMIT", "Limit must be between 1 and 100.");
        }
        // Enforce creator-private ownership
        getConversation(userId, conversationId);

        long afterOrder = 0L;
        if (cursor != null && !cursor.isBlank()) {
            afterOrder = decodeMessageCursor(cursor);
        }

        int fetchSize = limit + 1;
        List<MessageDto> fetched = jdbcTemplate.query(
            """
            SELECT id, conversation_id, run_id, role, content, status, message_order, created_at
            FROM messages
            WHERE conversation_id = ? AND message_order > ?
            ORDER BY message_order ASC
            LIMIT ?
            """,
            (rs, rowNum) -> {
                String rawRunId = rs.getString("run_id");
                return new MessageDto(
                    UUID.fromString(rs.getString("id")),
                    UUID.fromString(rs.getString("conversation_id")),
                    rawRunId != null ? UUID.fromString(rawRunId) : null,
                    rs.getString("role"),
                    rs.getString("content"),
                    rs.getString("status"),
                    rs.getLong("message_order"),
                    rs.getObject("created_at", OffsetDateTime.class)
                );
            },
            conversationId, afterOrder, fetchSize
        );

        boolean hasMore = fetched.size() > limit;
        List<MessageDto> items = hasMore ? fetched.subList(0, limit) : fetched;
        String nextCursor = null;
        if (hasMore && !items.isEmpty()) {
            nextCursor = encodeMessageCursor(items.getLast().message_order());
        }
        return new MessagePageResponse(items, nextCursor, hasMore);
    }

    public void verifyWorkspaceMembership(UUID userId, UUID workspaceId) {
        Integer count = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM memberships WHERE workspace_id = ? AND user_id = ?",
            Integer.class,
            workspaceId, userId
        );
        if (count == null || count == 0) {
            throw new ApiException(
                HttpStatus.NOT_FOUND,
                "WORKSPACE_NOT_FOUND",
                "Workspace not found or caller is not a member."
            );
        }
    }

    private static ConversationSummaryResponse mapConversationSummary(ResultSet rs) throws SQLException {
        return new ConversationSummaryResponse(
            UUID.fromString(rs.getString("id")),
            UUID.fromString(rs.getString("workspace_id")),
            UUID.fromString(rs.getString("owner_user_id")),
            rs.getString("title"),
            rs.getLong("version"),
            rs.getObject("updated_at", OffsetDateTime.class)
        );
    }

    private static String encodeConversationCursor(OffsetDateTime updatedAt, UUID id) {
        long epochMicros = ChronoUnit.MICROS.between(Instant.EPOCH, updatedAt.toInstant());
        String raw = epochMicros + ":" + id;
        return Base64.getUrlEncoder().withoutPadding().encodeToString(raw.getBytes(StandardCharsets.UTF_8));
    }

    private static ConversationCursor decodeConversationCursor(String cursor) {
        try {
            String raw = new String(Base64.getUrlDecoder().decode(cursor), StandardCharsets.UTF_8);
            String[] parts = raw.split(":", 2);
            if (parts.length != 2) {
                throw new IllegalArgumentException("Malformed cursor");
            }
            long epochMicros = Long.parseLong(parts[0]);
            Instant instant = Instant.EPOCH.plus(epochMicros, ChronoUnit.MICROS);
            UUID id = UUID.fromString(parts[1]);
            return new ConversationCursor(OffsetDateTime.ofInstant(instant, ZoneOffset.UTC), id);
        } catch (Exception ex) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "INVALID_CURSOR", "Malformed pagination cursor.");
        }
    }

    private static String encodeMessageCursor(long messageOrder) {
        String raw = "msg:" + messageOrder;
        return Base64.getUrlEncoder().withoutPadding().encodeToString(raw.getBytes(StandardCharsets.UTF_8));
    }

    private static long decodeMessageCursor(String cursor) {
        try {
            String raw = new String(Base64.getUrlDecoder().decode(cursor), StandardCharsets.UTF_8);
            if (!raw.startsWith("msg:")) {
                throw new IllegalArgumentException("Invalid message cursor prefix");
            }
            long order = Long.parseLong(raw.substring(4));
            if (order < 0) {
                throw new IllegalArgumentException("Negative message order");
            }
            return order;
        } catch (Exception ex) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "INVALID_CURSOR", "Malformed message pagination cursor.");
        }
    }

    private static ApiException conversationNotFound() {
        return new ApiException(
            HttpStatus.NOT_FOUND,
            "CONVERSATION_NOT_FOUND",
            "Conversation not found."
        );
    }

    private record ConversationCursor(OffsetDateTime updatedAt, UUID id) {}
}
