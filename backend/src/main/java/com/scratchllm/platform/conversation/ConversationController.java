package com.scratchllm.platform.conversation;

import com.scratchllm.platform.conversation.ConversationDtos.ConversationPageResponse;
import com.scratchllm.platform.conversation.ConversationDtos.ConversationSummaryResponse;
import com.scratchllm.platform.conversation.ConversationDtos.CreateConversationRequest;
import com.scratchllm.platform.conversation.ConversationDtos.MessagePageResponse;
import com.scratchllm.platform.conversation.ConversationDtos.RenameConversationRequest;
import com.scratchllm.platform.identity.AuthenticatedPrincipal;
import jakarta.validation.Valid;
import java.util.UUID;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * Day013 & Day014 REST endpoints for creator-private conversations and keyset-paginated message history.
 */
@RestController
public class ConversationController {

    private final ConversationService conversationService;

    public ConversationController(ConversationService conversationService) {
        this.conversationService = conversationService;
    }

    @PostMapping("/api/v1/workspaces/{workspaceId}/conversations")
    public ResponseEntity<ConversationSummaryResponse> createConversation(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID workspaceId,
        @Valid @RequestBody CreateConversationRequest request
    ) {
        ConversationSummaryResponse created = conversationService.createConversation(
            principal.userId(),
            workspaceId,
            request
        );
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }

    @GetMapping("/api/v1/workspaces/{workspaceId}/conversations")
    public ResponseEntity<ConversationPageResponse> listConversations(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID workspaceId,
        @RequestParam(required = false) String cursor,
        @RequestParam(defaultValue = "20") int limit
    ) {
        return ResponseEntity.ok(
            conversationService.listConversations(principal.userId(), workspaceId, cursor, limit)
        );
    }

    @GetMapping("/api/v1/conversations/{conversationId}")
    public ResponseEntity<ConversationSummaryResponse> getConversation(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID conversationId
    ) {
        return ResponseEntity.ok(
            conversationService.getConversation(principal.userId(), conversationId)
        );
    }

    @PatchMapping("/api/v1/conversations/{conversationId}")
    public ResponseEntity<ConversationSummaryResponse> renameConversation(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID conversationId,
        @Valid @RequestBody RenameConversationRequest request
    ) {
        return ResponseEntity.ok(
            conversationService.renameConversation(principal.userId(), conversationId, request)
        );
    }

    @DeleteMapping("/api/v1/conversations/{conversationId}")
    public ResponseEntity<Void> deleteConversation(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID conversationId
    ) {
        conversationService.deleteConversation(principal.userId(), conversationId);
        return ResponseEntity.noContent().build();
    }

    @GetMapping("/api/v1/conversations/{conversationId}/messages")
    public ResponseEntity<MessagePageResponse> listMessages(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID conversationId,
        @RequestParam(required = false) String cursor,
        @RequestParam(defaultValue = "50") int limit
    ) {
        return ResponseEntity.ok(
            conversationService.listMessages(principal.userId(), conversationId, cursor, limit)
        );
    }
}
