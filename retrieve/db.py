"""Durable storage for distilled topics: Postgres with pgvector.

Phase 1 stored plain files because the review loop didn't exist yet to need
anything queryable. Phase 2 added this alongside the file output (storage.py
still writes the Markdown vault) so notes stay readable in Obsidian while
also living somewhere vector search can reach. Phase 3 adds ratings, which
is the part files were never going to do well.

The schema is described in migrations/: sources, topics, a topic_sources
join table linking the two (the same concept often appears in more than one
source file), and reviews. Repository is the only class the rest of the app
talks to; it hides the SQL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from retrieve.schema import Difficulty, Rating, StoredTopic, TopicSource

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def connect(database_url: str) -> psycopg.Connection:
    """Open a connection with the pgvector type adapter registered, so
    Python lists can be passed and read back as VECTOR columns directly.

    The extension is enabled here, before the adapter is registered, because
    register_vector looks the `vector` type up in the catalog and fails on a
    database that doesn't have it yet -- which is exactly the state a fresh
    Neon project is in when `init-db` first connects. CREATE EXTENSION IF NOT
    EXISTS is a no-op once it's there, so this is safe on every connect.
    """
    conn = psycopg.connect(database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    """Apply every migration in order. Each one is idempotent (CREATE ...
    IF NOT EXISTS, ALTER ... DROP NOT NULL), so running this against an
    already-initialised database is a no-op."""
    for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        conn.execute(migration_path.read_text(encoding="utf-8"))


@dataclass
class NearestTopic:
    id: int
    title: str
    distance: float  # cosine distance: 0 = identical, 2 = opposite


class Repository:
    """One connection's worth of reads and writes against the schema above."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def transaction(self):
        """All-or-nothing scope for a group of writes. sync.py wraps a whole
        digest in one, so a failure part-way leaves no half-saved source.
        Works whether the connection is in autocommit mode (the normal case)
        or already inside a transaction (the tests' rollback fixture)."""
        return self._conn.transaction()

    # --- sources ----------------------------------------------------------

    def get_or_create_source(
        self, file_name: str, course: str | None, model: str, agent_note: str | None
    ) -> int:
        """Re-running summarize on the same file (or importing a digest that
        was already imported) reuses the existing source row rather than
        creating a second one, so topic_sources links stay consolidated."""
        existing = self._conn.execute(
            "SELECT id FROM sources WHERE file_name = %s AND course IS NOT DISTINCT FROM %s",
            (file_name, course),
        ).fetchone()
        if existing:
            return existing[0]
        row = self._conn.execute(
            """
            INSERT INTO sources (file_name, course, model, agent_note)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (file_name, course, model, agent_note),
        ).fetchone()
        return row[0]

    def source_exists(self, file_name: str, course: str | None) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sources WHERE file_name = %s AND course IS NOT DISTINCT FROM %s",
            (file_name, course),
        ).fetchone()
        return row is not None

    # --- topics -----------------------------------------------------------

    def find_nearest_topic(self, embedding: list[float]) -> NearestTopic | None:
        """The closest existing embedded topic by cosine distance, or None if
        nothing has an embedding yet. Topics stored without a vector (no
        Voyage key at the time) are never candidates here; the HNSW index in
        the migration is what keeps this cheap once there are many."""
        row = self._conn.execute(
            """
            SELECT id, title, embedding <=> %s AS distance
            FROM topics
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s
            LIMIT 1
            """,
            (Vector(embedding), Vector(embedding)),
        ).fetchone()
        if row is None:
            return None
        return NearestTopic(id=row[0], title=row[1], distance=float(row[2]))

    def find_topic_by_title(self, title: str) -> NearestTopic | None:
        """Exact (case-insensitive) title match. The fallback de-duplication
        used when there's no embedder, so at least re-running the same file
        doesn't double every topic."""
        row = self._conn.execute(
            "SELECT id, title FROM topics WHERE lower(title) = lower(%s) LIMIT 1", (title,)
        ).fetchone()
        if row is None:
            return None
        return NearestTopic(id=row[0], title=row[1], distance=0.0)

    def insert_topic(self, topic: Any, embedding: list[float] | None) -> int:
        row = self._conn.execute(
            """
            INSERT INTO topics (title, summary, key_points, tags, difficulty, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                topic.title,
                topic.summary,
                json.dumps(topic.key_points),
                topic.tags,
                topic.difficulty.value,
                Vector(embedding) if embedding is not None else None,
            ),
        ).fetchone()
        return row[0]

    def link_source(self, topic_id: int, source_id: int, source_pages: list[int]) -> None:
        """Record that `topic_id` appears in `source_id`, at these pages.
        Safe to call more than once for the same pair: a re-run merges the
        page lists instead of erroring on the primary key."""
        self._conn.execute(
            """
            INSERT INTO topic_sources (topic_id, source_id, source_pages)
            VALUES (%s, %s, %s)
            ON CONFLICT (topic_id, source_id) DO UPDATE
            SET source_pages = (
                SELECT ARRAY(
                    SELECT DISTINCT unnest(topic_sources.source_pages || EXCLUDED.source_pages)
                    ORDER BY 1
                )
            )
            """,
            (topic_id, source_id, source_pages),
        )

    def merge_tags(self, topic_id: int, tags: list[str]) -> None:
        """Union new tags into an existing topic instead of dropping them,
        since a duplicate found from a different course often carries tags
        (e.g. the module name) the first save didn't have."""
        self._conn.execute(
            """
            UPDATE topics
            SET tags = (SELECT ARRAY(SELECT DISTINCT unnest(tags || %s) ORDER BY 1)),
                updated_at = now()
            WHERE id = %s
            """,
            (tags, topic_id),
        )

    def source_count_for_topic(self, topic_id: int) -> int:
        row = self._conn.execute(
            "SELECT count(*) FROM topic_sources WHERE topic_id = %s", (topic_id,)
        ).fetchone()
        return row[0]

    # --- reviews ----------------------------------------------------------

    def record_review(self, topic_id: int, rating: Rating) -> None:
        self._conn.execute(
            "INSERT INTO reviews (topic_id, rating) VALUES (%s, %s)", (topic_id, rating.value)
        )

    def list_topics(self, course: str | None = None) -> list[StoredTopic]:
        """Every topic with its sources and a summary of its review history.
        Ordering is deliberately not done here: review.py decides what comes
        up next, so that policy can be unit-tested and, in Phase 4,
        replaced without touching SQL.

        One person's notes are a few hundred topics at most, so loading them
        all and sorting in Python is simpler than paging and fast enough.
        """
        rows = self._conn.execute(
            """
            SELECT
                t.id, t.title, t.summary, t.key_points, t.tags, t.difficulty,
                COALESCE(
                    json_agg(json_build_object(
                        'file_name', s.file_name, 'course', s.course, 'pages', ts.source_pages
                    ) ORDER BY s.created_at) FILTER (WHERE s.id IS NOT NULL),
                    '[]'
                ) AS sources,
                (SELECT count(*) FROM reviews r WHERE r.topic_id = t.id) AS review_count,
                -- id breaks ties: now() is fixed for a whole transaction, so two
                -- reviews in one transaction share a timestamp.
                (SELECT r.rating FROM reviews r WHERE r.topic_id = t.id
                 ORDER BY r.reviewed_at DESC, r.id DESC LIMIT 1) AS last_rating,
                (SELECT max(r.reviewed_at) FROM reviews r WHERE r.topic_id = t.id) AS last_reviewed_at
            FROM topics t
            LEFT JOIN topic_sources ts ON ts.topic_id = t.id
            LEFT JOIN sources s ON s.id = ts.source_id
            WHERE %s::text IS NULL
               OR EXISTS (
                    SELECT 1 FROM topic_sources ts2 JOIN sources s2 ON s2.id = ts2.source_id
                    WHERE ts2.topic_id = t.id AND lower(s2.course) = lower(%s)
               )
            GROUP BY t.id
            ORDER BY t.id
            """,
            (course, course),
        ).fetchall()

        return [
            StoredTopic(
                id=row[0],
                title=row[1],
                summary=row[2],
                key_points=row[3],
                tags=list(row[4]),
                difficulty=Difficulty(row[5]),
                sources=[TopicSource(**source) for source in row[6]],
                review_count=row[7],
                last_rating=Rating(row[8]) if row[8] else None,
                last_reviewed_at=row[9],
            )
            for row in rows
        ]
