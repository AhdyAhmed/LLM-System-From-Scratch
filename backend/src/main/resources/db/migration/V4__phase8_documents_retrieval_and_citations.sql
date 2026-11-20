-- Day072, Day073, Day074, Day075 & Day077: Phase 8 Document Ingestion, pgvector Hybrid Retrieval,
-- Tombstone Protection, and Server-Validated Retrieval Snapshots.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS raw_content TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS chunker_version VARCHAR(64) NOT NULL DEFAULT 'paragraph-window-v1',
    ADD COLUMN IF NOT EXISTS embedding_model_revision VARCHAR(120) NOT NULL DEFAULT 'pretrained-cpu-hash384-v1',
    ADD COLUMN IF NOT EXISTS error_code VARCHAR(64);

CREATE INDEX IF NOT EXISTS idx_documents_workspace_status
    ON documents (workspace_id, ingestion_status)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_rev
    ON document_chunks (document_id, document_revision, chunk_order ASC);
