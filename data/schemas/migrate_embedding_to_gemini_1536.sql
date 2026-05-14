BEGIN;

-- Existing embeddings were generated with a different model/dimension.
-- Clear rows first, then set vector dimensions to 1536 for pgvector ANN indexes.
TRUNCATE TABLE main_agent.document_chunks;
DROP INDEX IF EXISTS main_agent.idx_main_agent_document_chunks_embedding;
ALTER TABLE main_agent.document_chunks
  ALTER COLUMN embedding TYPE vector(1536);
CREATE INDEX IF NOT EXISTS idx_main_agent_document_chunks_embedding
  ON main_agent.document_chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

TRUNCATE TABLE library.guide_doc_chunks;
DROP INDEX IF EXISTS library.idx_guide_doc_chunks_embedding;
ALTER TABLE library.guide_doc_chunks
  ALTER COLUMN embedding TYPE vector(1536);
CREATE INDEX IF NOT EXISTS idx_guide_doc_chunks_embedding
  ON library.guide_doc_chunks
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

COMMIT;
