CREATE SCHEMA IF NOT EXISTS main_agent;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS main_agent.document_chunks (
  chunk_id VARCHAR(120) PRIMARY KEY,
  document_id VARCHAR(120) NOT NULL,
  text TEXT NOT NULL,
  embedding vector(1536) NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_main_agent_document_chunks_document_id
  ON main_agent.document_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_main_agent_document_chunks_metadata_gin
  ON main_agent.document_chunks USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_main_agent_document_chunks_embedding
  ON main_agent.document_chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
