package com.scratchllm.platform.documents;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.scratchllm.platform.documents.DeterministicCpuEmbedder.DocumentChunkSpec;
import com.scratchllm.platform.generation.OutboxRelayAndInboxService;
import com.scratchllm.platform.infrastructure.KafkaEventsInfrastructure;
import com.scratchllm.platform.platform.ApiException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Stream;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionTemplate;

/**
 * Day072–Day079: Workspace-isolated document upload, Kafka outbox/inbox ingestion,
 * deterministic chunking & 384-D CPU `pgvector` indexing, lexical/vector/hybrid search
 * with pre-ranking membership filters, budgeted RAG context packing, tombstone protection,
 * and server-validated citations.
 */
@Service
public class DocumentIngestionAndRetrievalService {

    public static final int MAX_DOCUMENT_BYTES = 65_536;
    public static final Set<String> ALLOWED_MIME_TYPES = Set.of("text/plain", "text/markdown");

    private final JdbcTemplate jdbcTemplate;
    private final TransactionTemplate transactionTemplate;
    private final OutboxRelayAndInboxService outboxRelayAndInboxService;
    private final ObjectMapper objectMapper;
    private final Path uploadStorageRoot;
    private volatile boolean autoIngestOnUpload = true;

    public DocumentIngestionAndRetrievalService(
        JdbcTemplate jdbcTemplate,
        PlatformTransactionManager transactionManager,
        OutboxRelayAndInboxService outboxRelayAndInboxService,
        ObjectMapper objectMapper
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.transactionTemplate = new TransactionTemplate(transactionManager);
        this.outboxRelayAndInboxService = outboxRelayAndInboxService;
        this.objectMapper = objectMapper;
        this.uploadStorageRoot = Path.of("target", "uploads").toAbsolutePath().normalize();
        try {
            Files.createDirectories(this.uploadStorageRoot);
        } catch (IOException ex) {
            throw new IllegalStateException("Failed to initialize upload storage directory", ex);
        }
    }

    public void setAutoIngestOnUpload(boolean autoIngestOnUpload) {
        this.autoIngestOnUpload = autoIngestOnUpload;
    }

    public boolean isAutoIngestOnUpload() {
        return this.autoIngestOnUpload;
    }

    /**
     * Day072 & Day073: Validates workspace membership, filename safety (rejecting path traversal),
     * MIME type (`text/plain`, `text/markdown`), and bounded UTF-8 byte size, then writes to an
     * opaque UUID storage key and emits a `document.ingestion.requested.v1` outbox event.
     */
    public DocumentMetadataDto uploadDocument(UUID userId, UUID workspaceId, UploadDocumentRequest request) {
        verifyWorkspaceMembership(userId, workspaceId);
        validateUploadRequest(request);

        String cleanFilename = request.filename().trim();
        String cleanMime = request.mime_type().trim().toLowerCase(Locale.ROOT);
        String normalizedContent = DeterministicCpuEmbedder.normalizeDocumentText(request.content());
        byte[] utf8Bytes = normalizedContent.getBytes(StandardCharsets.UTF_8);

        if (utf8Bytes.length == 0) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "EMPTY_DOCUMENT", "Document content must not be empty.");
        }
        if (utf8Bytes.length > MAX_DOCUMENT_BYTES) {
            throw new ApiException(
                HttpStatus.PAYLOAD_TOO_LARGE,
                "DOCUMENT_TOO_LARGE",
                "Document exceeds maximum allowed UTF-8 size of " + MAX_DOCUMENT_BYTES + " bytes."
            );
        }

        UUID documentId = request.document_id() != null ? request.document_id() : UUID.randomUUID();
        String contentChecksum = sha256Hex(normalizedContent);

        DocumentMetadataDto created = transactionTemplate.execute(status -> {
            List<ExistingDocRow> existingRows = jdbcTemplate.query(
                "SELECT id, workspace_id, revision, ingestion_status FROM documents WHERE id = ? FOR UPDATE",
                (rs, rowNum) -> new ExistingDocRow(
                    UUID.fromString(rs.getString("id")),
                    UUID.fromString(rs.getString("workspace_id")),
                    rs.getInt("revision"),
                    rs.getString("ingestion_status")
                ),
                documentId
            );

            int nextRevision = 1;
            if (!existingRows.isEmpty()) {
                ExistingDocRow existing = existingRows.getFirst();
                if (!existing.workspaceId().equals(workspaceId)) {
                    throw new ApiException(HttpStatus.FORBIDDEN, "WORKSPACE_ACCESS_DENIED", "Document belongs to another workspace.");
                }
                if ("DELETED".equals(existing.ingestionStatus())) {
                    throw new ApiException(HttpStatus.CONFLICT, "DOCUMENT_DELETED", "Deleted document cannot be overwritten or reactivated.");
                }
                nextRevision = existing.revision() + 1;
            }

            // Opaque storage key: never uses the user-supplied filename in a filesystem path
            String storageKey = "doc-" + documentId + "-rev-" + nextRevision + ".utf8";
            Path targetFile = uploadStorageRoot.resolve(storageKey).normalize();
            if (!targetFile.startsWith(uploadStorageRoot)) {
                throw new ApiException(HttpStatus.BAD_REQUEST, "INVALID_STORAGE_PATH", "Invalid storage path.");
            }
            try {
                Files.write(targetFile, utf8Bytes);
            } catch (IOException ex) {
                throw new IllegalStateException("Failed to persist uploaded document bytes", ex);
            }

            if (existingRows.isEmpty()) {
                jdbcTemplate.update(
                    """
                    INSERT INTO documents (
                        id, workspace_id, uploader_user_id, storage_key, content_checksum,
                        filename, mime_type, byte_size, ingestion_status, revision,
                        raw_content, chunker_version, embedding_model_revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?)
                    """,
                    documentId,
                    workspaceId,
                    userId,
                    storageKey,
                    contentChecksum,
                    cleanFilename,
                    cleanMime,
                    utf8Bytes.length,
                    nextRevision,
                    normalizedContent,
                    DeterministicCpuEmbedder.CHUNKER_VERSION,
                    DeterministicCpuEmbedder.EMBEDDING_MODEL_REVISION
                );
            } else {
                jdbcTemplate.update(
                    """
                    UPDATE documents
                    SET storage_key = ?,
                        content_checksum = ?,
                        filename = ?,
                        mime_type = ?,
                        byte_size = ?,
                        ingestion_status = 'PENDING',
                        revision = ?,
                        raw_content = ?,
                        updated_at = clock_timestamp()
                    WHERE id = ?
                    """,
                    storageKey,
                    contentChecksum,
                    cleanFilename,
                    cleanMime,
                    utf8Bytes.length,
                    nextRevision,
                    normalizedContent,
                    documentId
                );
            }

            // Emit ID-only document.ingestion.requested.v1 transactional outbox event (Day073)
            UUID outboxEventId = UUID.randomUUID();
            OffsetDateTime now = OffsetDateTime.now(ZoneOffset.UTC);
            String outboxPayload = toJson(Map.of(
                "event_id", outboxEventId.toString(),
                "event_type", KafkaEventsInfrastructure.TOPIC_DOCUMENT_INGESTION,
                "schema_version", 1,
                "occurred_at", now.toString(),
                "workspace_id", workspaceId.toString(),
                "document_id", documentId.toString(),
                "document_revision", nextRevision,
                "uploader_user_id", userId.toString(),
                "content_checksum", contentChecksum
            ));

            jdbcTemplate.update(
                """
                INSERT INTO outbox_events (event_id, topic, schema_version, aggregate_key, payload_json)
                VALUES (?, 'document.ingestion.requested.v1', 1, ?, ?::jsonb)
                """,
                outboxEventId, documentId.toString(), outboxPayload
            );

            return getDocumentInternal(documentId);
        });

        if (autoIngestOnUpload && created != null) {
            outboxRelayAndInboxService.relayAndConsumePending(10);
            dispatchAndExecuteDocumentIngestion(documentId, "auto-ingestion-worker");
            return getDocumentInternal(documentId);
        }

        return created;
    }

    /**
     * Day073, Day074, Day075 & Day079:
     * Claims a `DOCUMENT_INGESTION` work item (`FOR UPDATE SKIP LOCKED`), checks for deletion
     * tombstones (`deleted_at IS NOT NULL OR ingestion_status = 'DELETED'`), deterministically
     * chunks and embeds (`pretrained-cpu-hash384-v1`) the document into `document_chunks`,
     * removes superseded revision chunks, and activates `ingestion_status = 'READY'`.
     */
    public boolean dispatchAndExecuteDocumentIngestion(UUID documentId, String workerId) {
        Boolean ingested = transactionTemplate.execute(status -> {
            // Claim or create work_item for this document
            jdbcTemplate.update(
                """
                INSERT INTO work_items (id, work_kind, target_id, state)
                VALUES (?, 'DOCUMENT_INGESTION', ?, 'PENDING')
                ON CONFLICT (work_kind, target_id) DO NOTHING
                """,
                UUID.randomUUID(), documentId
            );

            List<UUID> claimedWork = jdbcTemplate.query(
                """
                SELECT id FROM work_items
                WHERE work_kind = 'DOCUMENT_INGESTION' AND target_id = ?
                FOR UPDATE SKIP LOCKED
                """,
                (rs, rowNum) -> UUID.fromString(rs.getString("id")),
                documentId
            );
            if (claimedWork.isEmpty()) {
                return false;
            }
            UUID workItemId = claimedWork.getFirst();

            jdbcTemplate.update(
                """
                UPDATE work_items
                SET state = 'LEASED',
                    lease_owner = ?,
                    fencing_token = fencing_token + 1,
                    attempt_count = attempt_count + 1,
                    lease_expires_at = clock_timestamp() + interval '30 seconds',
                    last_heartbeat_at = clock_timestamp(),
                    updated_at = clock_timestamp()
                WHERE id = ?
                """,
                workerId, workItemId
            );

            List<DocIngestionLockRow> docRows = jdbcTemplate.query(
                """
                SELECT id, workspace_id, storage_key, raw_content, ingestion_status, revision, deleted_at
                FROM documents
                WHERE id = ?
                FOR UPDATE
                """,
                (rs, rowNum) -> new DocIngestionLockRow(
                    UUID.fromString(rs.getString("id")),
                    UUID.fromString(rs.getString("workspace_id")),
                    rs.getString("storage_key"),
                    rs.getString("raw_content"),
                    rs.getString("ingestion_status"),
                    rs.getInt("revision"),
                    rs.getObject("deleted_at", OffsetDateTime.class)
                ),
                documentId
            );

            if (docRows.isEmpty()) {
                jdbcTemplate.update(
                    "UPDATE work_items SET state = 'FAILED', last_error_code = 'DOCUMENT_NOT_FOUND', updated_at = clock_timestamp() WHERE id = ?",
                    workItemId
                );
                return false;
            }

            DocIngestionLockRow doc = docRows.getFirst();

            // Day079 Tombstone Guard: Never reactivate or index a document that has been deleted!
            if (doc.deletedAt() != null || "DELETED".equals(doc.ingestionStatus())) {
                jdbcTemplate.update("DELETE FROM document_chunks WHERE document_id = ?", documentId);
                jdbcTemplate.update(
                    "UPDATE work_items SET state = 'CANCELLED', lease_expires_at = NULL, updated_at = clock_timestamp() WHERE id = ?",
                    workItemId
                );
                return false;
            }

            jdbcTemplate.update(
                "UPDATE documents SET ingestion_status = 'PROCESSING', updated_at = clock_timestamp() WHERE id = ?",
                documentId
            );

            String textToChunk = doc.rawContent();
            if (textToChunk == null || textToChunk.isBlank()) {
                Path fileOnDisk = uploadStorageRoot.resolve(doc.storageKey()).normalize();
                if (Files.exists(fileOnDisk)) {
                    try {
                        textToChunk = Files.readString(fileOnDisk, StandardCharsets.UTF_8);
                    } catch (IOException ignored) {
                    }
                }
            }

            List<DocumentChunkSpec> specs = DeterministicCpuEmbedder.chunkDocumentText(
                textToChunk != null ? textToChunk : "",
                doc.revision(),
                260,
                40
            );

            for (DocumentChunkSpec spec : specs) {
                jdbcTemplate.update(
                    """
                    INSERT INTO document_chunks (
                        id, document_id, workspace_id, document_revision, chunk_order,
                        source_location, chunk_text, embedding_revision, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?::vector)
                    ON CONFLICT (document_id, document_revision, chunk_order) DO UPDATE
                    SET source_location = EXCLUDED.source_location,
                        chunk_text = EXCLUDED.chunk_text,
                        embedding_revision = EXCLUDED.embedding_revision,
                        embedding = EXCLUDED.embedding
                    """,
                    UUID.randomUUID(),
                    documentId,
                    doc.workspaceId(),
                    doc.revision(),
                    spec.chunkOrder(),
                    spec.sourceLocation(),
                    spec.chunkText(),
                    spec.embeddingRevision(),
                    spec.pgVectorLiteral()
                );
            }

            // Remove chunks from any older document_revision or excess chunk_order
            jdbcTemplate.update(
                "DELETE FROM document_chunks WHERE document_id = ? AND (document_revision <> ? OR chunk_order >= ?)",
                documentId, doc.revision(), specs.size()
            );

            // Final tombstone check before marking READY
            Boolean stillAlive = jdbcTemplate.queryForObject(
                "SELECT deleted_at IS NULL AND ingestion_status <> 'DELETED' FROM documents WHERE id = ?",
                Boolean.class,
                documentId
            );
            if (!Boolean.TRUE.equals(stillAlive)) {
                jdbcTemplate.update("DELETE FROM document_chunks WHERE document_id = ?", documentId);
                jdbcTemplate.update(
                    "UPDATE work_items SET state = 'CANCELLED', lease_expires_at = NULL, updated_at = clock_timestamp() WHERE id = ?",
                    workItemId
                );
                return false;
            }

            jdbcTemplate.update(
                "UPDATE documents SET ingestion_status = 'READY', updated_at = clock_timestamp() WHERE id = ?",
                documentId
            );
            jdbcTemplate.update(
                "UPDATE work_items SET state = 'COMPLETED', lease_expires_at = NULL, updated_at = clock_timestamp() WHERE id = ?",
                workItemId
            );
            return true;
        });

        return Boolean.TRUE.equals(ingested);
    }

    @Transactional(readOnly = true)
    public List<DocumentMetadataDto> listWorkspaceDocuments(UUID userId, UUID workspaceId) {
        verifyWorkspaceMembership(userId, workspaceId);
        return jdbcTemplate.query(
            """
            SELECT d.id, d.workspace_id, d.uploader_user_id, d.storage_key, d.content_checksum,
                   d.filename, d.mime_type, d.byte_size, d.ingestion_status, d.revision,
                   d.chunker_version, d.embedding_model_revision, d.created_at, d.updated_at,
                   (SELECT count(*) FROM document_chunks dc WHERE dc.document_id = d.id AND dc.document_revision = d.revision) AS chunk_count
            FROM documents d
            WHERE d.workspace_id = ? AND d.deleted_at IS NULL AND d.ingestion_status <> 'DELETED'
            ORDER BY d.created_at DESC
            """,
            (rs, rowNum) -> new DocumentMetadataDto(
                UUID.fromString(rs.getString("id")),
                UUID.fromString(rs.getString("workspace_id")),
                UUID.fromString(rs.getString("uploader_user_id")),
                rs.getString("filename"),
                rs.getString("mime_type"),
                rs.getLong("byte_size"),
                rs.getString("content_checksum"),
                rs.getString("ingestion_status"),
                rs.getInt("revision"),
                rs.getInt("chunk_count"),
                rs.getString("chunker_version"),
                rs.getString("embedding_model_revision"),
                rs.getObject("created_at", OffsetDateTime.class),
                rs.getObject("updated_at", OffsetDateTime.class)
            ),
            workspaceId
        );
    }

    @Transactional(readOnly = true)
    public DocumentMetadataDto getDocument(UUID userId, UUID documentId) {
        DocumentMetadataDto doc = getDocumentInternal(documentId);
        if (doc == null || "DELETED".equals(doc.ingestion_status())) {
            throw new ApiException(HttpStatus.NOT_FOUND, "DOCUMENT_NOT_FOUND", "Document not found.");
        }
        verifyWorkspaceMembership(userId, doc.workspace_id());
        return doc;
    }

    /**
     * Day072 & Day079: Deletes a document ONLY if the caller is the original `uploader_user_id`
     * OR a workspace `OWNER`. Regular workspace members who did not upload the document are
     * rejected with `403 DOCUMENT_DELETE_FORBIDDEN`.
     * Immediately sets a deletion tombstone (`ingestion_status = 'DELETED', deleted_at = clock_timestamp()`),
     * deletes indexed chunks, cancels pending ingestion work items, and removes the opaque file.
     */
    @Transactional
    public DocumentDeleteResponse deleteDocument(UUID userId, UUID documentId) {
        List<DocDeleteLockRow> rows = jdbcTemplate.query(
            """
            SELECT id, workspace_id, uploader_user_id, storage_key, ingestion_status, revision
            FROM documents
            WHERE id = ?
            FOR UPDATE
            """,
            (rs, rowNum) -> new DocDeleteLockRow(
                UUID.fromString(rs.getString("id")),
                UUID.fromString(rs.getString("workspace_id")),
                UUID.fromString(rs.getString("uploader_user_id")),
                rs.getString("storage_key"),
                rs.getString("ingestion_status"),
                rs.getInt("revision")
            ),
            documentId
        );
        if (rows.isEmpty()) {
            throw new ApiException(HttpStatus.NOT_FOUND, "DOCUMENT_NOT_FOUND", "Document not found.");
        }
        DocDeleteLockRow doc = rows.getFirst();

        String callerRole = getWorkspaceMemberRole(userId, doc.workspaceId())
            .orElseThrow(() -> new ApiException(HttpStatus.NOT_FOUND, "DOCUMENT_NOT_FOUND", "Document not found."));

        boolean isUploader = doc.uploaderUserId().equals(userId);
        boolean isWorkspaceOwner = "OWNER".equals(callerRole);
        if (!isUploader && !isWorkspaceOwner) {
            throw new ApiException(
                HttpStatus.FORBIDDEN,
                "DOCUMENT_DELETE_FORBIDDEN",
                "Only the document uploader or workspace owner may delete this document."
            );
        }

        int deletedChunks = jdbcTemplate.update(
            "DELETE FROM document_chunks WHERE document_id = ?",
            documentId
        );

        jdbcTemplate.update(
            """
            UPDATE documents
            SET ingestion_status = 'DELETED',
                deleted_at = clock_timestamp(),
                raw_content = '',
                updated_at = clock_timestamp()
            WHERE id = ?
            """,
            documentId
        );

        jdbcTemplate.update(
            """
            UPDATE work_items
            SET state = 'CANCELLED',
                lease_expires_at = NULL,
                updated_at = clock_timestamp()
            WHERE work_kind = 'DOCUMENT_INGESTION' AND target_id = ?
            """,
            documentId
        );

        deleteStorageFileQuietly(doc.storageKey());

        return new DocumentDeleteResponse(documentId, doc.workspaceId(), "DELETED", deletedChunks);
    }

    /**
     * Reconciles and removes any orphaned upload files in `target/uploads/` whose storage key
     * does not match a non-deleted document in PostgreSQL (Day072).
     */
    public int reconcileOrphanedUploadFiles() {
        Set<String> activeStorageKeys = new HashSet<>(jdbcTemplate.query(
            "SELECT storage_key FROM documents WHERE deleted_at IS NULL AND ingestion_status <> 'DELETED'",
            (rs, rowNum) -> rs.getString("storage_key")
        ));
        int removed = 0;
        try (Stream<Path> files = Files.list(uploadStorageRoot)) {
            for (Path p : files.toList()) {
                String name = p.getFileName().toString();
                if (!activeStorageKeys.contains(name)) {
                    Files.deleteIfExists(p);
                    removed++;
                }
            }
        } catch (IOException ignored) {
        }
        return removed;
    }

    /**
     * Day075 & Day076: Workspace-filtered retrieval returning ranked snippets, source metadata,
     * and server-generated `citation_id`s.
     *
     * CRITICAL SECURITY INVARIANT:
     * Workspace membership (`JOIN memberships m ON m.workspace_id = d.workspace_id AND m.user_id = ?`)
     * and `d.workspace_id = ? AND d.ingestion_status = 'READY' AND d.deleted_at IS NULL` are enforced
     * INSIDE the PostgreSQL query BEFORE ranking or limiting.
     */
    @Transactional(readOnly = true)
    public RetrievalSearchResponse searchWorkspaceChunks(
        UUID userId,
        UUID workspaceId,
        RetrievalSearchRequest request
    ) {
        verifyWorkspaceMembership(userId, workspaceId);

        if (request == null || request.query() == null || request.query().isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "INVALID_RETRIEVAL_QUERY", "Search query is required.");
        }

        String query = request.query().trim();
        String mode = (request.mode() != null && !request.mode().isBlank())
            ? request.mode().trim().toLowerCase(Locale.ROOT)
            : "hybrid";
        int topK = request.top_k() != null ? Math.max(1, Math.min(20, request.top_k())) : 3;
        double minScore = request.min_score() != null ? request.min_score() : 0.18;

        float[] queryVec = DeterministicCpuEmbedder.embedText384(query);
        String pgVectorLiteral = DeterministicCpuEmbedder.toPgVectorLiteral(queryVec);
        Set<String> queryTerms = new HashSet<>(DeterministicCpuEmbedder.tokenizeRetrievalTerms(query));

        // Execute pre-filtered SQL query inside PostgreSQL joining documents + memberships + pgvector + FTS
        List<RawCandidateChunk> candidates = jdbcTemplate.query(
            """
            SELECT dc.id AS chunk_id,
                   dc.document_id,
                   d.filename,
                   dc.document_revision,
                   dc.chunk_order,
                   dc.source_location,
                   dc.chunk_text,
                   dc.embedding_revision,
                   GREATEST(0.0, 1.0 - (dc.embedding <=> ?::vector)) AS vector_cosine,
                   ts_rank_cd(dc.lexical_tsv, websearch_to_tsquery('english', ?)) AS fts_rank
            FROM document_chunks dc
            JOIN documents d
              ON d.id = dc.document_id
             AND d.revision = dc.document_revision
            JOIN memberships m
              ON m.workspace_id = d.workspace_id
             AND m.user_id = ?
            WHERE d.workspace_id = ?
              AND d.ingestion_status = 'READY'
              AND d.deleted_at IS NULL
            """,
            (rs, rowNum) -> new RawCandidateChunk(
                UUID.fromString(rs.getString("chunk_id")),
                UUID.fromString(rs.getString("document_id")),
                rs.getString("filename"),
                rs.getInt("document_revision"),
                rs.getInt("chunk_order"),
                rs.getString("source_location"),
                rs.getString("chunk_text"),
                rs.getString("embedding_revision"),
                rs.getDouble("vector_cosine"),
                rs.getDouble("fts_rank")
            ),
            pgVectorLiteral,
            query,
            userId,
            workspaceId
        );

        // Optional document_ids filter within the authorized workspace
        if (request.document_ids() != null && !request.document_ids().isEmpty()) {
            Set<UUID> allowedDocs = new HashSet<>(request.document_ids());
            candidates = candidates.stream()
                .filter(c -> allowedDocs.contains(c.documentId()))
                .toList();
        }

        if (candidates.isEmpty()) {
            return new RetrievalSearchResponse(
                workspaceId,
                mode,
                DeterministicCpuEmbedder.EMBEDDING_MODEL_REVISION,
                List.of()
            );
        }

        // Compute normalized lexical score (PostgreSQL FTS + stemmed term overlap) and vector cosine score
        Map<UUID, Double> lexScores = new HashMap<>();
        Map<UUID, Double> vecScores = new HashMap<>();

        for (RawCandidateChunk c : candidates) {
            Set<String> chunkTerms = new HashSet<>(DeterministicCpuEmbedder.tokenizeRetrievalTerms(c.chunkText()));
            int overlap = 0;
            for (String qt : queryTerms) {
                if (chunkTerms.contains(qt)) {
                    overlap++;
                }
            }
            double termOverlapRatio = overlap / Math.max(1.0, (double) queryTerms.size());
            double combinedLex = Math.min(1.0, 0.75 * termOverlapRatio + Math.min(0.25, c.ftsRank()));
            lexScores.put(c.chunkId(), combinedLex);
            vecScores.put(c.chunkId(), c.vectorCosine());
        }

        List<RawCandidateChunk> lexSorted = new ArrayList<>(candidates);
        lexSorted.sort(Comparator.comparingDouble((RawCandidateChunk c) -> -lexScores.get(c.chunkId()))
            .thenComparing(RawCandidateChunk::filename)
            .thenComparingInt(RawCandidateChunk::chunkOrder));

        List<RawCandidateChunk> vecSorted = new ArrayList<>(candidates);
        vecSorted.sort(Comparator.comparingDouble((RawCandidateChunk c) -> -vecScores.get(c.chunkId()))
            .thenComparing(RawCandidateChunk::filename)
            .thenComparingInt(RawCandidateChunk::chunkOrder));

        Map<UUID, Integer> lexRank = new HashMap<>();
        Map<UUID, Integer> vecRank = new HashMap<>();
        for (int i = 0; i < lexSorted.size(); i++) {
            lexRank.put(lexSorted.get(i).chunkId(), i + 1);
        }
        for (int i = 0; i < vecSorted.size(); i++) {
            vecRank.put(vecSorted.get(i).chunkId(), i + 1);
        }

        List<RetrievedChunkDto> results = new ArrayList<>();
        if ("lexical".equals(mode)) {
            for (RawCandidateChunk c : lexSorted) {
                double s = lexScores.get(c.chunkId());
                if (s >= minScore) {
                    results.add(toRetrievedChunkDto(c, s));
                }
                if (results.size() >= topK) {
                    break;
                }
            }
        } else if ("vector".equals(mode)) {
            for (RawCandidateChunk c : vecSorted) {
                double s = vecScores.get(c.chunkId());
                if (s >= minScore) {
                    results.add(toRetrievedChunkDto(c, s));
                }
                if (results.size() >= topK) {
                    break;
                }
            }
        } else {
            // Hybrid mode: RRF + weighted lexical/vector blend
            List<ScoredCandidate> hybridScored = new ArrayList<>();
            for (RawCandidateChunk c : candidates) {
                double lScore = lexScores.get(c.chunkId());
                double vScore = vecScores.get(c.chunkId());
                double rrf = (1.0 / (60.0 + lexRank.get(c.chunkId()))) + (1.0 / (60.0 + vecRank.get(c.chunkId())));
                double baseBlend = 0.55 * lScore + 0.45 * vScore;
                if (baseBlend >= minScore) {
                    hybridScored.add(new ScoredCandidate(c, baseBlend + 5.0 * rrf));
                }
            }
            hybridScored.sort(Comparator.comparingDouble((ScoredCandidate sc) -> -sc.score())
                .thenComparing(sc -> sc.chunk().filename())
                .thenComparingInt(sc -> sc.chunk().chunkOrder()));

            for (ScoredCandidate sc : hybridScored) {
                results.add(toRetrievedChunkDto(sc.chunk(), sc.score()));
                if (results.size() >= topK) {
                    break;
                }
            }
        }

        return new RetrievalSearchResponse(
            workspaceId,
            mode,
            DeterministicCpuEmbedder.EMBEDDING_MODEL_REVISION,
            results
        );
    }

    /**
     * Day077: Assembles retrieved evidence within the `ScratchGPT` context budget (`contextLimit = 128`),
     * deterministically truncating snippets while preserving exact `citation_id` and `chunk_id` metadata,
     * persists the snapshot in `retrieval_snapshots`, and returns the packed evidence + citations.
     */
    @Transactional
    public RagContextSnapshot assembleRagContextAndPersistSnapshot(
        UUID userId,
        UUID workspaceId,
        UUID runId,
        String prompt,
        List<UUID> requestedDocumentIds,
        int modelContextLimit,
        int estimatedPromptTokens,
        int reservedOutputTokens
    ) {
        if (requestedDocumentIds != null && !requestedDocumentIds.isEmpty()) {
            for (UUID docId : requestedDocumentIds) {
                if (docId == null) {
                    continue;
                }
                DocumentMetadataDto doc = getDocumentInternal(docId);
                if (doc == null || "DELETED".equals(doc.ingestion_status())) {
                    throw new ApiException(HttpStatus.NOT_FOUND, "DOCUMENT_NOT_FOUND", "Document not found: " + docId);
                }
                if (!doc.workspace_id().equals(workspaceId)) {
                    throw new ApiException(
                        HttpStatus.FORBIDDEN,
                        "WORKSPACE_ACCESS_DENIED",
                        "Cannot attach cross-workspace document to generation run: " + docId
                    );
                }
            }
        }

        RetrievalSearchResponse searchResp = searchWorkspaceChunks(
            userId,
            workspaceId,
            new RetrievalSearchRequest(prompt, "hybrid", 3, 0.15, requestedDocumentIds)
        );

        int availableEvidenceTokens = Math.max(0, modelContextLimit - estimatedPromptTokens - reservedOutputTokens - 6);
        int maxEvidenceChars = Math.max(32, availableEvidenceTokens * 4);

        List<RetrievedChunkDto> packedCitations = new ArrayList<>();
        StringBuilder packedEvidenceText = new StringBuilder();
        int usedChars = 0;

        for (RetrievedChunkDto hit : searchResp.results()) {
            if (usedChars >= maxEvidenceChars) {
                break;
            }
            int remainingChars = maxEvidenceChars - usedChars;
            String snippet = hit.snippet();
            boolean truncated = false;
            if (snippet.length() > remainingChars) {
                snippet = snippet.substring(0, Math.max(16, remainingChars)).trim() + "...";
                truncated = true;
            }

            RetrievedChunkDto boundedHit = new RetrievedChunkDto(
                hit.citation_id(),
                hit.chunk_id(),
                hit.document_id(),
                hit.filename(),
                hit.document_revision(),
                hit.chunk_order(),
                hit.source_location(),
                snippet,
                hit.score(),
                truncated,
                "AVAILABLE"
            );
            packedCitations.add(boundedHit);
            packedEvidenceText
                .append("<untrusted_document_evidence citation_id=\"")
                .append(boundedHit.citation_id())
                .append("\" source=\"")
                .append(boundedHit.filename())
                .append("#")
                .append(boundedHit.source_location())
                .append("\">\n")
                .append(boundedHit.snippet())
                .append("\n</untrusted_document_evidence>\n");
            usedChars += snippet.length();
        }

        RagContextSnapshot snapshot = new RagContextSnapshot(
            runId,
            workspaceId,
            DeterministicCpuEmbedder.EMBEDDING_MODEL_REVISION,
            availableEvidenceTokens,
            packedEvidenceText.toString().trim(),
            packedCitations
        );

        jdbcTemplate.update(
            """
            INSERT INTO retrieval_snapshots (id, run_id, snapshot_json)
            VALUES (?, ?, ?::jsonb)
            ON CONFLICT (run_id) DO UPDATE
            SET snapshot_json = EXCLUDED.snapshot_json
            """,
            UUID.randomUUID(),
            runId,
            toJson(snapshot)
        );

        return snapshot;
    }

    /**
     * Day076, Day077 & Day079: Returns the run's persisted retrieval citations, dynamically checking
     * whether each cited document is still active (`AVAILABLE`) or has been deleted (`UNAVAILABLE_DELETED`).
     */
    @Transactional(readOnly = true)
    public RunCitationsResponse getRunCitations(UUID userId, UUID runId) {
        Integer owned = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM generation_runs WHERE id = ? AND user_id = ?",
            Integer.class,
            runId, userId
        );
        if (owned == null || owned == 0) {
            throw new ApiException(HttpStatus.NOT_FOUND, "RUN_NOT_FOUND", "Generation run not found.");
        }

        List<String> snapshotRows = jdbcTemplate.query(
            "SELECT snapshot_json::text AS snapshot_json FROM retrieval_snapshots WHERE run_id = ?",
            (rs, rowNum) -> rs.getString("snapshot_json"),
            runId
        );
        if (snapshotRows.isEmpty()) {
            return new RunCitationsResponse(runId, List.of());
        }

        try {
            JsonNode root = objectMapper.readTree(snapshotRows.getFirst());
            List<RetrievedChunkDto> storedCitations = objectMapper.convertValue(
                root.get("citations"),
                new TypeReference<List<RetrievedChunkDto>>() {}
            );

            List<RetrievedChunkDto> reconciled = new ArrayList<>(storedCitations.size());
            for (RetrievedChunkDto c : storedCitations) {
                Boolean alive = jdbcTemplate.queryForObject(
                    """
                    SELECT count(*) > 0
                    FROM documents d
                    JOIN document_chunks dc ON dc.document_id = d.id AND dc.id = ?
                    WHERE d.id = ? AND d.deleted_at IS NULL AND d.ingestion_status = 'READY'
                    """,
                    Boolean.class,
                    c.chunk_id(),
                    c.document_id()
                );
                String liveStatus = Boolean.TRUE.equals(alive) ? "AVAILABLE" : "UNAVAILABLE_DELETED";
                String liveSnippet = Boolean.TRUE.equals(alive)
                    ? c.snippet()
                    : "[SOURCE UNAVAILABLE: Document was deleted or superseded]";
                reconciled.add(new RetrievedChunkDto(
                    c.citation_id(),
                    c.chunk_id(),
                    c.document_id(),
                    c.filename(),
                    c.document_revision(),
                    c.chunk_order(),
                    c.source_location(),
                    liveSnippet,
                    c.score(),
                    c.truncated(),
                    liveStatus
                ));
            }
            return new RunCitationsResponse(runId, reconciled);
        } catch (Exception ex) {
            throw new IllegalStateException("Failed to deserialize retrieval_snapshots", ex);
        }
    }

    /**
     * Day076 & Day079: Server-side validation of candidate citation IDs against the run's
     * persisted `retrieval_snapshots` and live workspace document state.
     * Never trusts model-invented references or cross-workspace chunk IDs.
     */
    @Transactional(readOnly = true)
    public CitationValidationResponse validateRunCitations(UUID userId, UUID runId, List<String> candidateCitationIds) {
        RunCitationsResponse runCitations = getRunCitations(userId, runId);
        Map<String, RetrievedChunkDto> byCitationId = new HashMap<>();
        for (RetrievedChunkDto c : runCitations.citations()) {
            byCitationId.put(c.citation_id(), c);
        }

        List<CitationValidationItem> items = new ArrayList<>();
        if (candidateCitationIds != null) {
            for (String cid : candidateCitationIds) {
                RetrievedChunkDto match = byCitationId.get(cid);
                if (match == null) {
                    items.add(new CitationValidationItem(
                        cid,
                        false,
                        "REJECTED_UNVERIFIED_CITATION",
                        null,
                        null,
                        null
                    ));
                } else if ("UNAVAILABLE_DELETED".equals(match.availability_status())) {
                    items.add(new CitationValidationItem(
                        cid,
                        false,
                        "UNAVAILABLE_DELETED",
                        match.document_id(),
                        match.chunk_id(),
                        match.snippet()
                    ));
                } else {
                    items.add(new CitationValidationItem(
                        cid,
                        true,
                        "VALIDATED",
                        match.document_id(),
                        match.chunk_id(),
                        match.snippet()
                    ));
                }
            }
        }
        return new CitationValidationResponse(runId, items);
    }

    public static String buildCitationId(UUID chunkId) {
        return "cit-" + chunkId.toString().replace("-", "").substring(0, 10);
    }

    private RetrievedChunkDto toRetrievedChunkDto(RawCandidateChunk c, double score) {
        double rounded = Math.round(score * 10_000.0) / 10_000.0;
        return new RetrievedChunkDto(
            buildCitationId(c.chunkId()),
            c.chunkId(),
            c.documentId(),
            c.filename(),
            c.documentRevision(),
            c.chunkOrder(),
            c.sourceLocation(),
            c.chunkText(),
            rounded,
            false,
            "AVAILABLE"
        );
    }

    private void validateUploadRequest(UploadDocumentRequest request) {
        if (request == null || request.filename() == null || request.filename().isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "INVALID_FILENAME", "Document filename is required.");
        }
        String fn = request.filename().trim();
        if (fn.length() > 255
            || fn.contains("..")
            || fn.contains("/")
            || fn.contains("\\")
            || fn.contains("\0")
            || fn.contains(":")) {
            throw new ApiException(
                HttpStatus.BAD_REQUEST,
                "INVALID_FILENAME",
                "Filename must not contain path separators, '..', drive prefixes, or null bytes."
            );
        }
        if (request.mime_type() == null || !ALLOWED_MIME_TYPES.contains(request.mime_type().trim().toLowerCase(Locale.ROOT))) {
            throw new ApiException(
                HttpStatus.BAD_REQUEST,
                "UNSUPPORTED_MEDIA_TYPE",
                "Only text/plain and text/markdown uploads are supported."
            );
        }
        if (request.content() == null || request.content().isBlank()) {
            throw new ApiException(HttpStatus.BAD_REQUEST, "EMPTY_DOCUMENT", "Document content is required.");
        }
    }

    private void verifyWorkspaceMembership(UUID userId, UUID workspaceId) {
        if (getWorkspaceMemberRole(userId, workspaceId).isEmpty()) {
            throw new ApiException(
                HttpStatus.FORBIDDEN,
                "WORKSPACE_ACCESS_DENIED",
                "User is not a member of workspace " + workspaceId
            );
        }
    }

    private Optional<String> getWorkspaceMemberRole(UUID userId, UUID workspaceId) {
        List<String> roles = jdbcTemplate.query(
            "SELECT role FROM memberships WHERE workspace_id = ? AND user_id = ?",
            (rs, rowNum) -> rs.getString("role"),
            workspaceId, userId
        );
        return roles.isEmpty() ? Optional.empty() : Optional.of(roles.getFirst());
    }

    private DocumentMetadataDto getDocumentInternal(UUID documentId) {
        List<DocumentMetadataDto> rows = jdbcTemplate.query(
            """
            SELECT d.id, d.workspace_id, d.uploader_user_id, d.filename, d.mime_type,
                   d.byte_size, d.content_checksum, d.ingestion_status, d.revision,
                   d.chunker_version, d.embedding_model_revision, d.created_at, d.updated_at,
                   (SELECT count(*) FROM document_chunks dc WHERE dc.document_id = d.id AND dc.document_revision = d.revision) AS chunk_count
            FROM documents d
            WHERE d.id = ?
            """,
            (rs, rowNum) -> new DocumentMetadataDto(
                UUID.fromString(rs.getString("id")),
                UUID.fromString(rs.getString("workspace_id")),
                UUID.fromString(rs.getString("uploader_user_id")),
                rs.getString("filename"),
                rs.getString("mime_type"),
                rs.getLong("byte_size"),
                rs.getString("content_checksum"),
                rs.getString("ingestion_status"),
                rs.getInt("revision"),
                rs.getInt("chunk_count"),
                rs.getString("chunker_version"),
                rs.getString("embedding_model_revision"),
                rs.getObject("created_at", OffsetDateTime.class),
                rs.getObject("updated_at", OffsetDateTime.class)
            ),
            documentId
        );
        return rows.isEmpty() ? null : rows.getFirst();
    }

    private void deleteStorageFileQuietly(String storageKey) {
        if (storageKey == null || storageKey.isBlank()) {
            return;
        }
        try {
            Path p = uploadStorageRoot.resolve(storageKey).normalize();
            if (p.startsWith(uploadStorageRoot)) {
                Files.deleteIfExists(p);
            }
        } catch (IOException ignored) {
        }
    }

    private String toJson(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException ex) {
            throw new IllegalStateException("Failed to serialize JSON", ex);
        }
    }

    private static String sha256Hex(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(digest.digest(input.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException(ex);
        }
    }

    public record UploadDocumentRequest(
        UUID document_id,
        String filename,
        String mime_type,
        String content
    ) {}

    public record DocumentMetadataDto(
        UUID document_id,
        UUID workspace_id,
        UUID uploader_user_id,
        String filename,
        String mime_type,
        long byte_size,
        String content_checksum,
        String ingestion_status,
        int revision,
        int chunk_count,
        String chunker_version,
        String embedding_model_revision,
        OffsetDateTime created_at,
        OffsetDateTime updated_at
    ) {}

    public record DocumentDeleteResponse(
        UUID document_id,
        UUID workspace_id,
        String status,
        int removed_chunks
    ) {}

    public record RetrievalSearchRequest(
        String query,
        String mode,
        Integer top_k,
        Double min_score,
        List<UUID> document_ids
    ) {}

    public record RetrievedChunkDto(
        String citation_id,
        UUID chunk_id,
        UUID document_id,
        String filename,
        int document_revision,
        int chunk_order,
        String source_location,
        String snippet,
        double score,
        boolean truncated,
        String availability_status
    ) {}

    public record RetrievalSearchResponse(
        UUID workspace_id,
        String mode,
        String embedding_model_revision,
        List<RetrievedChunkDto> results
    ) {}

    public record RagContextSnapshot(
        UUID run_id,
        UUID workspace_id,
        String embedding_model_revision,
        int budget_tokens,
        String formatted_evidence_block,
        List<RetrievedChunkDto> citations
    ) {}

    public record RunCitationsResponse(
        UUID run_id,
        List<RetrievedChunkDto> citations
    ) {}

    public record CitationValidationRequest(
        List<String> citation_ids
    ) {}

    public record CitationValidationItem(
        String citation_id,
        boolean valid,
        String status,
        UUID document_id,
        UUID chunk_id,
        String snippet
    ) {}

    public record CitationValidationResponse(
        UUID run_id,
        List<CitationValidationItem> results
    ) {}

    private record ExistingDocRow(UUID id, UUID workspaceId, int revision, String ingestionStatus) {}
    private record DocIngestionLockRow(
        UUID id,
        UUID workspaceId,
        String storageKey,
        String rawContent,
        String ingestionStatus,
        int revision,
        OffsetDateTime deletedAt
    ) {}
    private record DocDeleteLockRow(
        UUID id,
        UUID workspaceId,
        UUID uploaderUserId,
        String storageKey,
        String ingestionStatus,
        int revision
    ) {}
    private record RawCandidateChunk(
        UUID chunkId,
        UUID documentId,
        String filename,
        int documentRevision,
        int chunkOrder,
        String sourceLocation,
        String chunkText,
        String embeddingRevision,
        double vectorCosine,
        double ftsRank
    ) {}
    private record ScoredCandidate(RawCandidateChunk chunk, double score) {}
}
