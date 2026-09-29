-- ContractLens schema. Applied idempotently at startup by PgVectorStore.ensure_schema().
-- {dim} is substituted with the configured embedding dimension.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS contractlens_meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id          text PRIMARY KEY,
    title       text NOT NULL,
    doc_type    text NOT NULL DEFAULT 'other',
    source      text NOT NULL DEFAULT '',
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    chunk_count integer NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id          text PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     integer NOT NULL,
    section     text NOT NULL DEFAULT '',
    page        integer,
    text        text NOT NULL,
    char_start  integer NOT NULL DEFAULT 0,
    char_end    integer NOT NULL DEFAULT 0,
    embedding   vector({dim}) NOT NULL,
    -- Full-text column for the keyword side of hybrid retrieval. Section headings are
    -- included so a query like "limitation of liability" hits the clause directly.
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(section, '') || ' ' || text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_document_idx  ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx       ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
