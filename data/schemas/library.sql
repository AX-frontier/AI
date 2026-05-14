CREATE SCHEMA IF NOT EXISTS library;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS library.books (
  id BIGSERIAL PRIMARY KEY,
  excel_no INT,
  bib_no VARCHAR(50) NOT NULL,
  reg_no VARCHAR(50) NOT NULL,
  title VARCHAR(500) NOT NULL,
  author TEXT,
  publisher VARCHAR(300),
  publish_year INT,
  holding_location VARCHAR(100),
  holding_call_no VARCHAR(100),
  material_type VARCHAR(50),
  location_symbol VARCHAR(50),
  stack_location VARCHAR(100),
  stack_shelf VARCHAR(100),
  isbn TEXT,
  title_normalized TEXT,
  author_normalized TEXT,
  publisher_normalized TEXT,
  holding_call_no_normalized TEXT,
  stack_location_normalized TEXT,
  stack_shelf_normalized TEXT,
  search_text_normalized TEXT,
  raw_data JSONB NOT NULL DEFAULT '{}'::jsonb,
  import_document_id BIGINT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT uq_books_reg_no UNIQUE (reg_no),
  CONSTRAINT uq_books_bib_reg UNIQUE (bib_no, reg_no)
);

CREATE TABLE IF NOT EXISTS library.guide_docs (
  id BIGSERIAL PRIMARY KEY,
  source_url TEXT,
  title VARCHAR(500) NOT NULL,
  content TEXT NOT NULL,
  content_hash TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE library.guide_docs
  ADD COLUMN IF NOT EXISTS content_hash TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_guide_docs_source_url
  ON library.guide_docs (source_url);

CREATE INDEX IF NOT EXISTS idx_guide_docs_content_hash
  ON library.guide_docs (content_hash);

CREATE TABLE IF NOT EXISTS library.guide_doc_chunks (
  id BIGSERIAL PRIMARY KEY,
  guide_doc_id BIGINT NOT NULL REFERENCES library.guide_docs(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  embedding vector(1536) NOT NULL,
  content_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT uq_guide_doc_chunks_doc_chunk UNIQUE (guide_doc_id, chunk_index),
  CONSTRAINT uq_guide_doc_chunks_content_hash UNIQUE (guide_doc_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_guide_doc_chunks_embedding
  ON library.guide_doc_chunks
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_books_title_trgm
  ON library.books
  USING gin (title gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_books_author_trgm
  ON library.books
  USING gin (author gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_books_publisher_trgm
  ON library.books
  USING gin (publisher gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_books_holding_call_no_trgm
  ON library.books
  USING gin (holding_call_no gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_books_stack_location_trgm
  ON library.books
  USING gin (stack_location gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_books_stack_shelf_trgm
  ON library.books
  USING gin (stack_shelf gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_books_title_normalized_trgm
  ON library.books
  USING gin (title_normalized gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_books_author_normalized_trgm
  ON library.books
  USING gin (author_normalized gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_books_publisher_normalized_trgm
  ON library.books
  USING gin (publisher_normalized gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_books_holding_call_no_normalized_trgm
  ON library.books
  USING gin (holding_call_no_normalized gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_books_stack_shelf_normalized_trgm
  ON library.books
  USING gin (stack_shelf_normalized gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_books_search_text_normalized_trgm
  ON library.books
  USING gin (search_text_normalized gin_trgm_ops);
