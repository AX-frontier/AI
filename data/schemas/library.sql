CREATE SCHEMA IF NOT EXISTS library;

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
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
