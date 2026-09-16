"""Tests for the de-duplication decision in sync.py, against a fake
repository and a fake embedder so they run offline with no database or
API key. tests/test_db_integration.py separately proves the real Repository
implements the same contract against real Postgres + pgvector.
"""

from retrieve.schema import Difficulty, StudyDigest, Topic
from retrieve.sync import format_sync_report, sync_digest
from tests.fakes import FakeEmbeddingClient, FakeRepository


def _topic(title: str, pages: list[int] = [1]) -> Topic:
    return Topic(
        title=title,
        summary="A summary that is definitely long enough to pass validation.",
        key_points=["a key point"],
        tags=["networking"],
        difficulty=Difficulty.CORE,
        source_pages=pages,
    )


def _digest(topics: list[Topic], source_file: str = "week-1.pdf") -> StudyDigest:
    return StudyDigest(source_file=source_file, course="Networking",
                       model="claude-sonnet-5", topics=topics)


def test_a_topic_with_no_close_match_is_inserted_as_new() -> None:
    topic = _topic("TCP three-way handshake")
    embedder = FakeEmbeddingClient({_embed_text(topic): [1.0, 0.0]})
    repository = FakeRepository()

    results = sync_digest(_digest([topic]), repository, embedder, dedup_threshold=0.1)

    assert results[0].status == "new"
    assert len(repository.topics) == 1
    assert repository.topic_sources[(1, 1)] == [1]


def test_a_near_duplicate_topic_is_merged_not_inserted() -> None:
    first = _topic("TCP three-way handshake", pages=[2])
    second = _topic("The TCP handshake", pages=[9])
    # Identical vectors -> distance 0, well under any reasonable threshold.
    embedder = FakeEmbeddingClient({
        _embed_text(first): [1.0, 0.0],
        _embed_text(second): [1.0, 0.0],
    })
    repository = FakeRepository()

    sync_digest(_digest([first], "week-1.pdf"), repository, embedder, dedup_threshold=0.1)
    results = sync_digest(_digest([second], "week-5-recap.pdf"), repository, embedder, dedup_threshold=0.1)

    assert results[0].status == "merged"
    assert results[0].matched_title == "TCP three-way handshake"
    assert results[0].distance == 0.0
    assert len(repository.topics) == 1  # no second row created
    assert repository.topic_sources[(1, 2)] == [9]  # linked to the second source's pages
    assert results[0].source_count == 2


def test_a_related_but_distinct_topic_above_threshold_is_kept_separate() -> None:
    tcp = _topic("TCP three-way handshake", pages=[2])
    udp = _topic("UDP vs TCP trade-offs", pages=[6])
    embedder = FakeEmbeddingClient({
        _embed_text(tcp): [1.0, 0.0],
        _embed_text(udp): [0.0, 1.0],  # orthogonal -> cosine distance 1.0
    })
    repository = FakeRepository()

    sync_digest(_digest([tcp]), repository, embedder, dedup_threshold=0.1)
    results = sync_digest(_digest([udp]), repository, embedder, dedup_threshold=0.1)

    assert results[0].status == "new"
    assert len(repository.topics) == 2


def test_merging_unions_tags_from_both_saves() -> None:
    first = Topic(title="TCP handshake", summary="x" * 25, key_points=["a"],
                  tags=["networking"], difficulty=Difficulty.CORE, source_pages=[1])
    second = Topic(title="TCP handshake recap", summary="y" * 25, key_points=["a"],
                   tags=["interview-prep"], difficulty=Difficulty.CORE, source_pages=[1])
    embedder = FakeEmbeddingClient({
        _embed_text(first): [1.0, 0.0],
        _embed_text(second): [1.0, 0.0],
    })
    repository = FakeRepository()

    sync_digest(_digest([first]), repository, embedder, dedup_threshold=0.1)
    sync_digest(_digest([second]), repository, embedder, dedup_threshold=0.1)

    assert repository.topics[1]["tags"] == ["interview-prep", "networking"]


def test_without_an_embedder_topics_are_stored_unembedded_and_matched_by_title() -> None:
    first = _topic("TCP three-way handshake", pages=[2])
    reworded = _topic("The TCP handshake", pages=[9])  # would match by meaning, not by title
    same_title = _topic("tcp three-way handshake", pages=[11])  # case differs only
    repository = FakeRepository()

    sync_digest(_digest([first], "week-1.pdf"), repository, embedder=None)
    results = sync_digest(_digest([reworded, same_title], "week-5.pdf"), repository, embedder=None)

    assert repository.topics[1]["embedding"] is None
    assert results[0].status == "new"  # no embeddings -> rewording isn't caught
    assert results[1].status == "merged"  # but an identical title is
    assert results[1].matched_title == "TCP three-way handshake"


def test_re_running_the_same_source_reuses_its_source_row() -> None:
    topic = _topic("TCP three-way handshake")
    embedder = FakeEmbeddingClient({_embed_text(topic): [1.0, 0.0]})
    repository = FakeRepository()

    sync_digest(_digest([topic], "week-1.pdf"), repository, embedder)
    sync_digest(_digest([topic], "week-1.pdf"), repository, embedder)

    assert len(repository.sources) == 1
    assert len(repository.topics) == 1


def test_format_sync_report_summarises_new_and_merged_counts() -> None:
    report = format_sync_report([
        _result("new"),
        _result("merged", matched_title="TCP three-way handshake", distance=0.04, source_count=2),
    ])
    assert report.startswith("1 new, 1 merged with existing topics")
    assert "merged into 'TCP three-way handshake'" in report
    assert "distance 0.040" in report


def _embed_text(topic: Topic) -> str:
    from retrieve.embeddings import topic_embedding_text
    return topic_embedding_text(topic)


def _result(status: str, **kwargs):
    from retrieve.sync import TopicSyncResult
    return TopicSyncResult(topic=_topic("A topic"), status=status, **kwargs)
