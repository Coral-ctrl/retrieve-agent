"""Sync a distilled digest into the database, de-duplicating as it goes.

This is the piece that makes "de-duplication across sources" concrete: for
every topic the agent saved, embed it, look for the nearest existing topic
by meaning (not by title string), and either merge into that match or
insert a new row. Kept separate from db.py so the decision logic (what
counts as "the same topic") can be unit-tested against a fake repository
and a fake embedder, with no real database or API key needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from retrieve.embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingClient, embed_topic
from retrieve.schema import StudyDigest, Topic

# Cosine distance below which two topics are treated as the same concept.
# 0 = identical text; in practice two slightly-reworded descriptions of the
# same idea land well under 0.1, while genuinely different topics in the
# same course are usually above 0.2. Override with RETRIEVE_DEDUP_THRESHOLD
# if your material runs closer together than that.
DEFAULT_DEDUP_THRESHOLD = 0.10


@dataclass
class TopicSyncResult:
    topic: Topic
    status: str  # "new" or "merged"
    matched_title: str | None = None
    distance: float | None = None
    source_count: int | None = None


def sync_digest(
    digest: StudyDigest,
    repository: object,
    embedder: EmbeddingClient | None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> list[TopicSyncResult]:
    """Write every topic in `digest` to `repository`. Returns one result per
    topic, in the same order, so the caller can report what happened.

    With no `embedder` (no Voyage key configured), topics are stored without
    a vector and matching falls back to exact title, which is enough to stop
    a re-run of the same file from doubling everything, but won't catch the
    same concept worded differently across sources.
    """
    source_id = repository.get_or_create_source(
        digest.source_file, digest.course, digest.model, digest.agent_note
    )

    results: list[TopicSyncResult] = []
    for topic in digest.topics:
        if embedder is not None:
            embedding = embed_topic(topic, embedder, model=embedding_model)
            nearest = repository.find_nearest_topic(embedding)
        else:
            embedding = None
            nearest = repository.find_topic_by_title(topic.title)

        if nearest is not None and nearest.distance <= dedup_threshold:
            repository.link_source(nearest.id, source_id, topic.source_pages)
            repository.merge_tags(nearest.id, topic.tags)
            results.append(TopicSyncResult(
                topic=topic,
                status="merged",
                matched_title=nearest.title,
                distance=nearest.distance,
                source_count=repository.source_count_for_topic(nearest.id),
            ))
        else:
            topic_id = repository.insert_topic(topic, embedding)
            repository.link_source(topic_id, source_id, topic.source_pages)
            results.append(TopicSyncResult(topic=topic, status="new"))

    return results


def format_sync_report(results: list[TopicSyncResult]) -> str:
    new_count = sum(1 for r in results if r.status == "new")
    merged_count = len(results) - new_count

    lines = [f"{new_count} new, {merged_count} merged with existing topics"]
    for result in results:
        if result.status == "merged":
            lines.append(
                f"  - '{result.topic.title}' merged into '{result.matched_title}' "
                f"(distance {result.distance:.3f}, now in {result.source_count} source(s))"
            )
    return "\n".join(lines)
