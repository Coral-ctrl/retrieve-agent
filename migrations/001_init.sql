-- Phase 2 schema: durable storage for distilled topics, with vector search
-- for de-duplication across sources.
--
-- Run by `python -m retrieve init-db`, which wraps this file in a
-- CREATE EXTENSION + CREATE TABLE IF NOT EXISTS so it's safe to run more
-- than once (e.g. against a fresh Neon branch).

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per file the agent has distilled.
CREATE TABLE IF NOT EXISTS sources (
    id          SERIAL PRIMARY KEY,
    file_name   TEXT NOT NULL,
    course      TEXT,
    model       TEXT NOT NULL,
    agent_note  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per distinct concept. Concepts are global, not per-source: the
-- same idea taught in two different weeks' slides becomes one row here,
-- linked to both sources through topic_sources. See db.py's
-- find_nearest_topic for how a new topic is matched against existing ones
-- before a row is inserted.
CREATE TABLE IF NOT EXISTS topics (
    id           SERIAL PRIMARY KEY,
    title        TEXT NOT NULL,
    summary      TEXT NOT NULL,
    key_points   JSONB NOT NULL,
    tags         TEXT[] NOT NULL,
    difficulty   TEXT NOT NULL,
    -- Dimension matches Voyage's voyage-3-lite model (see embeddings.py).
    -- Changing embedding models means changing this column and re-embedding.
    embedding    VECTOR(512) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Which sources (and which pages within them) a topic came from. A topic
-- gains a new row here, instead of a whole new topics row, whenever it's
-- matched as a near-duplicate of one already saved.
CREATE TABLE IF NOT EXISTS topic_sources (
    topic_id      INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    source_id     INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    source_pages  INTEGER[] NOT NULL,
    PRIMARY KEY (topic_id, source_id)
);

-- HNSW index for fast approximate nearest-neighbour search on cosine
-- distance. Built with default parameters; fine at the scale one person's
-- notes reach.
CREATE INDEX IF NOT EXISTS topics_embedding_hnsw_idx
    ON topics USING hnsw (embedding vector_cosine_ops);
