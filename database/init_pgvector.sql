CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,

    chunk_position INTEGER UNIQUE NOT NULL,
    chunk_id TEXT UNIQUE NOT NULL,

    content TEXT NOT NULL,

    source TEXT NOT NULL,
    document_format TEXT,
    page_number INTEGER,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    embedding VECTOR(768) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_source
ON document_chunks(source);

CREATE INDEX IF NOT EXISTS idx_document_chunks_format
ON document_chunks(document_format);

CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata
ON document_chunks USING GIN(metadata);