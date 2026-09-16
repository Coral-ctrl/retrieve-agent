-- Phase 3: ratings, and embeddings become optional.

-- A topic can now be stored without a vector. This is what lets the
-- database work with only DATABASE_URL set: de-duplication needs
-- embeddings, but storing and reviewing topics doesn't. Topics with a NULL
-- embedding are simply never candidates in find_nearest_topic.
ALTER TABLE topics ALTER COLUMN embedding DROP NOT NULL;

-- One row per time you rate a topic. Append-only on purpose: the history
-- (not just the latest rating) is what Phase 4's review policy will use to
-- decide what to resurface and when.
CREATE TABLE IF NOT EXISTS reviews (
    id           SERIAL PRIMARY KEY,
    topic_id     INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    rating       TEXT NOT NULL CHECK (rating IN ('solid', 'shaky', 'forgotten')),
    reviewed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS reviews_topic_id_reviewed_at_idx
    ON reviews (topic_id, reviewed_at DESC);
