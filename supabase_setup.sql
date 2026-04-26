-- =============================================================================
-- GitLab Handbook Assistant — Supabase Setup Script
-- =============================================================================
-- This script sets up the pgvector extension, creates the chunks table,
-- adds an HNSW index for fast similarity search, and creates the
-- match_chunks function for cosine similarity retrieval.
-- =============================================================================

-- Step 1: Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Create the gitlab_chunks table
CREATE TABLE IF NOT EXISTS gitlab_chunks (
    id              BIGSERIAL PRIMARY KEY,
    chunk_text      TEXT NOT NULL,
    embedding       VECTOR(768) NOT NULL,   -- text-embedding-004 outputs 768 dimensions
    source_url      TEXT NOT NULL,
    section_title   TEXT NOT NULL DEFAULT '',
    page_title      TEXT NOT NULL DEFAULT '',
    source_type     TEXT NOT NULL DEFAULT 'handbook',  -- "handbook" or "direction"
    chunk_hash      TEXT UNIQUE,            -- SHA-256 hash for deduplication
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Step 3: Create an HNSW index on the embedding column for fast similarity search
-- Using cosine distance operator (<=>) for HNSW
CREATE INDEX IF NOT EXISTS idx_gitlab_chunks_embedding
ON gitlab_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Step 4: Create index on source_type for filtered queries
CREATE INDEX IF NOT EXISTS idx_gitlab_chunks_source_type
ON gitlab_chunks (source_type);

-- Step 5: Create index on chunk_hash for fast deduplication lookups
CREATE INDEX IF NOT EXISTS idx_gitlab_chunks_chunk_hash
ON gitlab_chunks (chunk_hash);

-- Step 6: Create the match_chunks function
-- Takes a query embedding and returns the top k chunks by cosine similarity
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding VECTOR(768),
    match_count     INT DEFAULT 5,
    filter_type     TEXT DEFAULT NULL
)
RETURNS TABLE (
    id              BIGINT,
    chunk_text      TEXT,
    source_url      TEXT,
    section_title   TEXT,
    page_title      TEXT,
    source_type     TEXT,
    similarity      FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        gc.id,
        gc.chunk_text,
        gc.source_url,
        gc.section_title,
        gc.page_title,
        gc.source_type,
        1 - (gc.embedding <=> query_embedding) AS similarity
    FROM gitlab_chunks gc
    WHERE
        CASE
            WHEN filter_type IS NOT NULL THEN gc.source_type = filter_type
            ELSE TRUE
        END
    ORDER BY gc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Step 7: Create a boosted match function for section-aware routing
-- Returns chunks with a boost for matching source_type
CREATE OR REPLACE FUNCTION match_chunks_boosted(
    query_embedding     VECTOR(768),
    match_count         INT DEFAULT 5,
    boost_type          TEXT DEFAULT NULL,
    boost_factor        FLOAT DEFAULT 0.05
)
RETURNS TABLE (
    id              BIGINT,
    chunk_text      TEXT,
    source_url      TEXT,
    section_title   TEXT,
    page_title      TEXT,
    source_type     TEXT,
    similarity      FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        gc.id,
        gc.chunk_text,
        gc.source_url,
        gc.section_title,
        gc.page_title,
        gc.source_type,
        (1 - (gc.embedding <=> query_embedding))
            + CASE
                WHEN boost_type IS NOT NULL AND gc.source_type = boost_type
                THEN boost_factor
                ELSE 0.0
              END AS similarity
    FROM gitlab_chunks gc
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;
