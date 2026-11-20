package com.scratchllm.platform.documents;

import com.scratchllm.platform.documents.DocumentIngestionAndRetrievalService.CitationValidationRequest;
import com.scratchllm.platform.documents.DocumentIngestionAndRetrievalService.CitationValidationResponse;
import com.scratchllm.platform.documents.DocumentIngestionAndRetrievalService.DocumentDeleteResponse;
import com.scratchllm.platform.documents.DocumentIngestionAndRetrievalService.DocumentMetadataDto;
import com.scratchllm.platform.documents.DocumentIngestionAndRetrievalService.RetrievalSearchRequest;
import com.scratchllm.platform.documents.DocumentIngestionAndRetrievalService.RetrievalSearchResponse;
import com.scratchllm.platform.documents.DocumentIngestionAndRetrievalService.RunCitationsResponse;
import com.scratchllm.platform.documents.DocumentIngestionAndRetrievalService.UploadDocumentRequest;
import com.scratchllm.platform.identity.AuthenticatedPrincipal;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

/**
 * Day072, Day075, Day076 & Day077 REST endpoints for workspace-shared document uploads,
 * uploader/owner deletion, hybrid retrieval search, and server-validated run citations.
 */
@RestController
public class DocumentController {

    private final DocumentIngestionAndRetrievalService documentService;

    public DocumentController(DocumentIngestionAndRetrievalService documentService) {
        this.documentService = documentService;
    }

    @PostMapping("/api/v1/workspaces/{workspaceId}/documents")
    public ResponseEntity<DocumentMetadataDto> uploadDocument(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID workspaceId,
        @RequestBody UploadDocumentRequest request
    ) {
        DocumentMetadataDto created = documentService.uploadDocument(principal.userId(), workspaceId, request);
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }

    @GetMapping("/api/v1/workspaces/{workspaceId}/documents")
    public ResponseEntity<Map<String, List<DocumentMetadataDto>>> listWorkspaceDocuments(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID workspaceId
    ) {
        return ResponseEntity.ok(Map.of(
            "items",
            documentService.listWorkspaceDocuments(principal.userId(), workspaceId)
        ));
    }

    @GetMapping("/api/v1/documents/{documentId}")
    public ResponseEntity<DocumentMetadataDto> getDocument(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID documentId
    ) {
        return ResponseEntity.ok(documentService.getDocument(principal.userId(), documentId));
    }

    @DeleteMapping("/api/v1/documents/{documentId}")
    public ResponseEntity<DocumentDeleteResponse> deleteDocument(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID documentId
    ) {
        return ResponseEntity.ok(documentService.deleteDocument(principal.userId(), documentId));
    }

    @PostMapping({
        "/api/v1/workspaces/{workspaceId}/retrieval/search",
        "/api/v1/workspaces/{workspaceId}/documents/search"
    })
    public ResponseEntity<RetrievalSearchResponse> searchWorkspace(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID workspaceId,
        @RequestBody RetrievalSearchRequest request
    ) {
        return ResponseEntity.ok(documentService.searchWorkspaceChunks(principal.userId(), workspaceId, request));
    }

    @GetMapping("/api/v1/runs/{runId}/citations")
    public ResponseEntity<RunCitationsResponse> getRunCitations(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID runId
    ) {
        return ResponseEntity.ok(documentService.getRunCitations(principal.userId(), runId));
    }

    @PostMapping("/api/v1/runs/{runId}/citations/validate")
    public ResponseEntity<CitationValidationResponse> validateRunCitations(
        @AuthenticationPrincipal AuthenticatedPrincipal principal,
        @PathVariable UUID runId,
        @RequestBody CitationValidationRequest request
    ) {
        return ResponseEntity.ok(documentService.validateRunCitations(
            principal.userId(),
            runId,
            request != null ? request.citation_ids() : List.of()
        ));
    }
}
