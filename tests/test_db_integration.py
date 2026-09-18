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
    # Each test runs inside an outer transaction that is always rolled back,
    # so tests don't need to clean up after themselves or interfere with
    # each other. Because the fixture owns the OUTER transaction, any
    # Repository.transaction() a test (or sync.py) opens becomes a savepoint
    # inside it rather than a real commit.
    with conn.transaction(force_rollback=True):
        yield Repository(conn)
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


def test_init_db_works_on_a_fresh_database_without_the_extension() -> None:
    """The shared retrieve_test database already has pgvector enabled, which
    hid a real bug: connect() registered the vector adapter before anything
    had created the type, so `init-db` failed on a brand-new Neon project.
    This creates a genuinely empty database and runs the full path on it."""
    fresh_name = "retrieve_test_fresh"
    maintenance_url = TEST_DATABASE_URL.rsplit("/", 1)[0] + "/postgres"
    try:
        admin = psycopg.connect(maintenance_url, autocommit=True)
    except psycopg.OperationalError as error:
        pytest.skip(f"No reachable test server at {maintenance_url}: {error}")

    admin.execute(f"DROP DATABASE IF EXISTS {fresh_name}")
    try:
        admin.execute(f"CREATE DATABASE {fresh_name}")
    except psycopg.errors.InsufficientPrivilege:
        pytest.skip("Test role can't create databases")

    try:
        fresh_url = TEST_DATABASE_URL.rsplit("/", 1)[0] + f"/{fresh_name}"
        conn = connect(fresh_url)  # this is the call that used to fail
        init_schema(conn)
        assert Repository(conn).list_topics() == []
        conn.close()
    finally:
        admin.execute(f"DROP DATABASE IF EXISTS {fresh_name}")
        admin.close()


def test_transaction_rolls_back_everything_on_failure(repo) -> None:
    """A database error part-way through a sync must not leave a source row
    with only some of its topics. Only real Postgres can prove the rollback."""
    with pytest.raises(RuntimeError, match="boom"):
        with repo.transaction():
            source_id = repo.get_or_create_source("week-1.pdf", "Networking", "claude-sonnet-5", None)
            topic_id = repo.insert_topic(_topic(), _vector(1.0))
            repo.link_source(topic_id, source_id, [2])
            raise RuntimeError("boom")

    assert not repo.source_exists("week-1.pdf", "Networking")
    assert repo.list_topics() == []


def test_sync_digest_against_real_postgres_is_atomic_and_idempotent(repo) -> None:
    """End to end through sync.py: a fake embedder, the real Repository.
    Re-running the same digest merges every topic back into the rows it
    created the first time -- which is also the recovery path after a
    failed sync: just run summarize again."""
    from retrieve.schema import StudyDigest
    from retrieve.sync import sync_digest
    from tests.fakes import FakeEmbeddingClient
    from retrieve.embeddings import topic_embedding_text

    topics = [_topic("TCP three-way handshake"), _topic("TCP vs UDP")]
    embedder = FakeEmbeddingClient({
        topic_embedding_text(topics[0]): _vector(1.0),
        topic_embedding_text(topics[1]): [0.0, 1.0] + [0.0] * 510,
    })
    digest = StudyDigest(source_file="week-1.pdf", course="Networking",
                         model="claude-sonnet-5", topics=topics)

    first = sync_digest(digest, repo, embedder)
    second = sync_digest(digest, repo, embedder)

    assert [r.status for r in first] == ["new", "new"]
    assert [r.status for r in second] == ["merged", "merged"]
    assert len(repo.list_topics()) == 2
    assert all(r.source_count == 1 for r in second)  # same source, not a second row


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
    # The full history comes back oldest-first, which is what the scheduler
    # replays; the summary properties are derived from it.
    assert [e.rating for e in stored.reviews] == [Rating.FORGOTTEN, Rating.SHAKY]
    assert stored.reviews[0].reviewed_at <= stored.reviews[1].reviewed_at
    assert stored.review_count == 2
    assert stored.last_rating == Rating.SHAKY
    assert stored.last_reviewed_at == stored.reviews[1].reviewed_at


def test_list_topics_includes_never_reviewed_and_unlinked_topics(repo) -> None:
    topic_id = repo.insert_topic(_topic(), None)

    [stored] = repo.list_topics()

    assert stored.id == topic_id
    assert stored.sources == []
    assert stored.reviews == []
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


# --- Phase 5: search ---------------------------------------------------------

def test_find_similar_topics_returns_nearest_first_with_distances(repo) -> None:
    source_id = repo.get_or_create_source("week-1.pdf", "Networking", "claude-sonnet-5", None)
    close = repo.insert_topic(_topic("TCP three-way handshake"), _vector(1.0))
    far = repo.insert_topic(_topic("UDP basics"), [0.0, 1.0] + [0.0] * 510)
    unembedded = repo.insert_topic(_topic("Stored with no key"), None)
    for tid in (close, far, unembedded):
        repo.link_source(tid, source_id, [1])

    # A query vector between the two, but closer to `close`.
    results = repo.find_similar_topics([0.9, 0.1] + [0.0] * 510, limit=5)

    assert [t.title for t, _ in results] == ["TCP three-way handshake", "UDP basics"]
    assert results[0][1] < results[1][1]
    assert results[0][0].sources[0].file_name == "week-1.pdf"  # full StoredTopic, not just an id


def test_find_similar_topics_respects_limit_and_course(repo) -> None:
    net = repo.get_or_create_source("net.pdf", "Networking", "claude-sonnet-5", None)
    dbs = repo.get_or_create_source("db.pdf", "Databases", "claude-sonnet-5", None)
    tcp = repo.insert_topic(_topic("TCP"), _vector(1.0))
    idx = repo.insert_topic(_topic("Indexes"), _vector(0.99))
    repo.link_source(tcp, net, [1])
    repo.link_source(idx, dbs, [1])

    assert len(repo.find_similar_topics(_vector(1.0), limit=1)) == 1
    assert [t.title for t, _ in repo.find_similar_topics(_vector(1.0), limit=5, course="databases")] == ["Indexes"]


def test_search_topics_by_text_matches_plain_english_and_ranks_titles_first(repo) -> None:
    source_id = repo.get_or_create_source("week-1.pdf", "Networking", "claude-sonnet-5", None)
    in_title = repo.insert_topic(Topic(
        title="Congestion control", summary="How TCP avoids overwhelming the network.",
        key_points=["slow start"], tags=["tcp"], difficulty=Difficulty.CORE, source_pages=[1]), None)
    in_summary = repo.insert_topic(Topic(
        title="Flow control", summary="Distinct from congestion control; protects the receiver.",
        key_points=["receiver window"], tags=["tcp"], difficulty=Difficulty.CORE, source_pages=[2]), None)
    unrelated = repo.insert_topic(_topic("UDP basics"), None)
    for tid in (in_title, in_summary, unrelated):
        repo.link_source(tid, source_id, [1])

    results = repo.search_topics_by_text("how does congestion control work", limit=5)

    # Stemming means 'congestion' matches, stop words are ignored, and the
    # title match outranks the summary-only match.
    assert [t.title for t in results] == ["Congestion control", "Flow control"]
    assert repo.search_topics_by_text("quantum entanglement", limit=5) == []
