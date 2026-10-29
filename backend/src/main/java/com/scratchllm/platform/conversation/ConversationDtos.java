package com.scratchllm.platform.conversation;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;

/**
 * Pure DTO records for creator-private conversation CRUD and keyset-paginated message history.
 */
public final class ConversationDtos {

    private ConversationDtos() {}

    public record CreateConversationRequest(
        @NotBlank(message = "Conversation title is required")
        @Size(min = 1, max = 160, message = "Title must be between 1 and 160 characters")
        String title
    ) {}

    public record RenameConversationRequest(
        @NotBlank(message = "Conversation title is required")
        @Size(min = 1, max = 160, message = "Title must be between 1 and 160 characters")
        String title,

        @NotNull(message = "Optimistic lock version is required")
        @Min(value = 0, message = "Version must be non-negative")
        Long version
    ) {}

    public record ConversationSummaryResponse(
        UUID conversation_id,
        UUID workspace_id,
        UUID owner_user_id,
        String title,
        long version,
        OffsetDateTime updated_at
    ) {}

    public record ConversationPageResponse(
        List<ConversationSummaryResponse> items,
        String next_cursor,
        boolean has_more
    ) {}

    public record MessageDto(
        UUID message_id,
        UUID conversation_id,
        UUID run_id,
        String role,
        String content,
        String status,
        long message_order,
        OffsetDateTime created_at
    ) {}

    public record MessagePageResponse(
        List<MessageDto> items,
        String next_cursor,
        boolean has_more
    ) {}
}
