"""Runs Repository against a real Postgres + pgvector database.

test_sync.py proves the de-duplication *decision* against a fake repository.
This proves db.py's SQL actually does what Repository promises: cosine
distance ordering, the topic_sources merge-on-conflict, and tag unioning,
against the real thing rather than a Python re-implementation of it.

Needs a reachable Postgres with the pgvector extension available (not
necessarily pre-created -- init_schema creates it). Points at
RETRIEVE_TEST_DATABASE_URL, or localhost with the default local dev
credentials if that's unset. Skips cleanly if nothing is reachable, the
same way the rest of the suite runs without an API key.
"""

from __future__ import annotations

import os

import pytest

psycopg = pytest.importorskip("psycopg")

from retrieve.db import Repository, connect, init_schema  # noqa: E402
from retrieve.schema import Difficulty, Rating, Topic  # noqa: E402

TEST_DATABASE_URL = os.getenv(
    "RETRIEVE_TEST_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/retrieve_test"
)


@pytest.fixture
def repo():
    try:
        conn = connect(TEST_DATABASE_URL)
    except psycopg.OperationalError as error:
        pytest.skip(f"No reachable test database at {TEST_DATABASE_URL}: {error}")

    init_schema(conn)
    # Each test runs inside a transaction that's rolled back afterwards, so
    # tests don't need to clean up after themselves or interfere with each
    # other. autocommit is on for init_schema (DDL) but off inside the test.
    conn.autocommit = False
    yield Repository(conn)
    conn.rollback()
    conn.close()


def _topic(title: str = "TCP three-way handshake") -> Topic:
    return Topic(
        title=title,
        summary="TCP opens a connection with SYN, SYN-ACK, ACK before data flows.",
        key_points=["SYN", "SYN-ACK", "ACK"],
        tags=["networking"],
        difficulty=Difficulty.CORE,
        source_pages=[2],
    )


def _vector(seed: float, dimensions: int = 512) -> list[float]:
    # A cheap stand-in for a real embedding: same seed -> identical vector
    # (distance 0), different seed -> a different direction.
    return [seed] + [0.0] * (dimensions - 1)


def test_insert_and_find_nearest_topic_round_trips_through_postgres(repo) -> None:
    source_id = repo.get_or_create_source("week-1.pdf", "Networking", "claude-sonnet-5", None)
    topic_id = repo.insert_topic(_topic(), _vector(1.0))
    repo.link_source(topic_id, source_id, [2])

    nearest = repo.find_nearest_topic(_vector(1.0))

    assert nearest.id == topic_id
    assert nearest.title == "TCP three-way handshake"
    assert nearest.distance == pytest.approx(0.0, abs=1e-6)


def test_find_nearest_topic_ranks_by_cosine_distance(repo) -> None:
    source_id = repo.get_or_create_source("week-1.pdf", "Networking", "claude-sonnet-5", None)
    close_id = repo.insert_topic(_topic("TCP three-way handshake"), _vector(1.0))
    far_id = repo.insert_topic(_topic("UDP basics"), [0.0, 1.0] + [0.0] * 510)
    repo.link_source(close_id, source_id, [2])
    repo.link_source(far_id, source_id, [5])

    nearest = repo.find_nearest_topic(_vector(1.0))

    assert nearest.id == close_id


def test_link_source_merges_page_lists_on_repeated_calls(repo) -> None:
    source_id = repo.get_or_create_source("week-1.pdf", "Networking", "claude-sonnet-5", None)
    topic_id = repo.insert_topic(_topic(), _vector(1.0))

    repo.link_source(topic_id, source_id, [2])
    repo.link_source(topic_id, source_id, [4, 3])  # a later call, overlapping + new pages

    row = repo._conn.execute(
        "SELECT source_pages FROM topic_sources WHERE topic_id = %s AND source_id = %s",
        (topic_id, source_id),
    ).fetchone()
    assert row[0] == [2, 3, 4]


def test_merge_tags_unions_without_duplicating(repo) -> None:
    topic_id = repo.insert_topic(_topic(), _vector(1.0))

    repo.merge_tags(topic_id, ["networking", "interview-prep"])

    row = repo._conn.execute("SELECT tags FROM topics WHERE id = %s", (topic_id,)).fetchone()
    assert sorted(row[0]) == ["interview-prep", "networking"]


def test_source_count_for_topic_counts_distinct_linked_sources(repo) -> None:
    topic_id = repo.insert_topic(_topic(), _vector(1.0))
    source_one = repo.get_or_create_source("week-1.pdf", "Networking", "claude-sonnet-5", None)
    source_two = repo.get_or_create_source("week-5-recap.pdf", "Networking", "claude-sonnet-5", None)
    repo.link_source(topic_id, source_one, [2])
    repo.link_source(topic_id, source_two, [9])

    assert repo.source_count_for_topic(topic_id) == 2


# --- Phase 3 -----------------------------------------------------------------

def test_get_or_create_source_returns_the_existing_row_for_the_same_file_and_course(repo) -> None:
    first = repo.get_or_create_source("week-1.pdf", "Networking", "claude-sonnet-5", None)
    again = repo.get_or_create_source("week-1.pdf", "Networking", "claude-sonnet-5", "re-run")
    no_course = repo.get_or_create_source("week-1.pdf", None, "claude-sonnet-5", None)

    assert again == first
    assert no_course != first  # NULL course is compared properly, not treated as unknown
    assert repo.source_exists("week-1.pdf", "Networking")
    assert not repo.source_exists("week-2.pdf", "Networking")


def test_topics_can_be_stored_without_an_embedding_and_are_skipped_by_vector_search(repo) -> None:
    unembedded_id = repo.insert_topic(_topic("Stored with no Voyage key"), None)
    embedded_id = repo.insert_topic(_topic("Stored with a vector"), _vector(1.0))

    nearest = repo.find_nearest_topic([0.0, 1.0] + [0.0] * 510)  # far from the embedded one

    assert nearest.id == embedded_id  # the NULL-embedding row is never a candidate
    assert repo.find_topic_by_title("stored with NO voyage key").id == unembedded_id
    assert repo.find_topic_by_title("nothing like this") is None


def test_list_topics_aggregates_sources_and_review_history(repo) -> None:
    source_one = repo.get_or_create_source("week-1.pdf", "Networking", "claude-sonnet-5", None)
    source_two = repo.get_or_create_source("week-5.pdf", "Networking", "claude-sonnet-5", None)
    topic_id = repo.insert_topic(_topic(), _vector(1.0))
    repo.link_source(topic_id, source_one, [2])
    repo.link_source(topic_id, source_two, [9, 10])
    repo.record_review(topic_id, Rating.FORGOTTEN)
    repo.record_review(topic_id, Rating.SHAKY)  # the later one is the "last" rating

    [stored] = repo.list_topics()

    assert stored.id == topic_id
    assert stored.title == "TCP three-way handshake"
    assert stored.key_points == ["SYN", "SYN-ACK", "ACK"]
    assert stored.difficulty == Difficulty.CORE
    assert [(s.file_name, s.pages) for s in stored.sources] == [("week-1.pdf", [2]), ("week-5.pdf", [9, 10])]
    assert stored.courses == ["Networking"]
    assert stored.review_count == 2
    assert stored.last_rating == Rating.SHAKY
    assert stored.last_reviewed_at is not None


def test_list_topics_includes_never_reviewed_and_unlinked_topics(repo) -> None:
    topic_id = repo.insert_topic(_topic(), None)

    [stored] = repo.list_topics()

    assert stored.id == topic_id
    assert stored.sources == []
    assert stored.review_count == 0
    assert stored.last_rating is None


def test_list_topics_can_be_filtered_by_course_case_insensitively(repo) -> None:
    networking = repo.get_or_create_source("net.pdf", "Networking", "claude-sonnet-5", None)
    databases = repo.get_or_create_source("db.pdf", "Databases", "claude-sonnet-5", None)
    net_topic = repo.insert_topic(_topic("TCP"), None)
    db_topic = repo.insert_topic(_topic("Indexes"), None)
    repo.link_source(net_topic, networking, [1])
    repo.link_source(db_topic, databases, [1])

    assert [t.title for t in repo.list_topics(course="networking")] == ["TCP"]
    assert sorted(t.title for t in repo.list_topics()) == ["Indexes", "TCP"]
    assert repo.list_topics(course="History") == []
