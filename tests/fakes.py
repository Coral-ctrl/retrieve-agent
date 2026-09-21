"""A fake Claude client and response builders for tests.

The fake client replays a scripted list of responses, so the agent loop can
be tested exactly, offline, and for free.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(tool_id: str, name: str, tool_input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def fake_response(*blocks: SimpleNamespace, stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
    )


class FakeMessagesClient:
    """Stands in for anthropic.Anthropic().messages."""

    def __init__(self, scripted_responses: list[SimpleNamespace]) -> None:
        self.scripted_responses = list(scripted_responses)
        self.received_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        # Snapshot the message list, since the agent keeps appending to it.
        self.received_calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self.scripted_responses:
            raise AssertionError("Agent made more model calls than the test scripted.")
        return self.scripted_responses.pop(0)


class FakeEmbeddingClient:
    """Stands in for voyageai.Client(). Returns hand-assigned vectors keyed
    by exact input text, so a test can control distance between topics
    without needing real embeddings or a network call."""

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self.vectors_by_text = vectors_by_text
        self.received_calls: list[dict[str, Any]] = []

    def embed(self, texts: list[str], model: str, input_type: str, output_dimension: int = 512) -> SimpleNamespace:
        self.received_calls.append({
            "texts": list(texts), "model": model, "input_type": input_type,
            "output_dimension": output_dimension,
        })
        try:
            embeddings = [self.vectors_by_text[text] for text in texts]
        except KeyError as missing_text:
            raise AssertionError(
                f"FakeEmbeddingClient wasn't given a vector for: {missing_text}"
            ) from None
        return SimpleNamespace(embeddings=embeddings)


class FakeRepository:
    """An in-memory stand-in for retrieve.db.Repository, so sync.py's
    dedup decisions can be tested without a real Postgres connection."""

    def __init__(self) -> None:
        self.sources: dict[int, dict[str, Any]] = {}
        self.topics: dict[int, dict[str, Any]] = {}
        self.topic_sources: dict[tuple[int, int], list[int]] = {}
        self.reviews: list[tuple[int, Any]] = []
        self._next_source_id = 1
        self._next_topic_id = 1

    def transaction(self):
        from contextlib import nullcontext
        return nullcontext()

    def get_or_create_source(self, file_name, course, model, agent_note) -> int:
        for source_id, source in self.sources.items():
            if source["file_name"] == file_name and source["course"] == course:
                return source_id
        source_id = self._next_source_id
        self._next_source_id += 1
        self.sources[source_id] = {
            "file_name": file_name, "course": course, "model": model, "agent_note": agent_note,
        }
        return source_id

    def source_exists(self, file_name, course) -> bool:
        return any(s["file_name"] == file_name and s["course"] == course
                   for s in self.sources.values())

    def find_nearest_topic(self, embedding: list[float]):
        from retrieve.db import NearestTopic

        embedded = {tid: t for tid, t in self.topics.items() if t["embedding"] is not None}
        if not embedded:
            return None
        topic_id, distance = min(
            ((tid, _cosine_distance(t["embedding"], embedding)) for tid, t in embedded.items()),
            key=lambda pair: pair[1],
        )
        return NearestTopic(id=topic_id, title=self.topics[topic_id]["title"], distance=distance)

    def find_topic_by_title(self, title: str):
        from retrieve.db import NearestTopic

        for topic_id, topic in self.topics.items():
            if topic["title"].lower() == title.lower():
                return NearestTopic(id=topic_id, title=topic["title"], distance=0.0)
        return None

    def insert_topic(self, topic, embedding: list[float] | None) -> int:
        topic_id = self._next_topic_id
        self._next_topic_id += 1
        self.topics[topic_id] = {"title": topic.title, "tags": list(topic.tags), "embedding": embedding}
        return topic_id

    def record_review(self, topic_id: int, rating) -> None:
        self.reviews.append((topic_id, rating))

    def link_source(self, topic_id: int, source_id: int, source_pages: list[int]) -> None:
        existing = self.topic_sources.get((topic_id, source_id), [])
        self.topic_sources[(topic_id, source_id)] = sorted(set(existing) | set(source_pages))

    def merge_tags(self, topic_id: int, tags: list[str]) -> None:
        self.topics[topic_id]["tags"] = sorted(set(self.topics[topic_id]["tags"]) | set(tags))

    def source_count_for_topic(self, topic_id: int) -> int:
        return sum(1 for (tid, _sid) in self.topic_sources if tid == topic_id)

    def list_sources(self):
        from retrieve.db import SourceInfo

        counts = {}
        for (topic_id, source_id) in self.topic_sources:
            counts[source_id] = counts.get(source_id, 0) + 1
        return [
            SourceInfo(id=sid, file_name=s["file_name"], course=s["course"],
                      topic_count=counts.get(sid, 0))
            for sid, s in sorted(self.sources.items())
        ]

    def delete_source(self, file_name: str, course):
        from retrieve.db import DeleteSourceResult

        source_id = next(
            (sid for sid, s in self.sources.items()
             if s["file_name"] == file_name and s["course"] == course),
            None,
        )
        if source_id is None:
            return None

        topic_ids_here = [tid for (tid, sid) in self.topic_sources if sid == source_id]
        self.sources.pop(source_id)
        self.topic_sources = {
            (tid, sid): pages for (tid, sid), pages in self.topic_sources.items()
            if sid != source_id
        }

        still_linked = {tid for (tid, _sid) in self.topic_sources}
        deleted_titles = []
        for topic_id in topic_ids_here:
            if topic_id not in still_linked:
                deleted_titles.append(self.topics.pop(topic_id)["title"])

        return DeleteSourceResult(
            source_id=source_id,
            topics_unlinked=len(topic_ids_here) - len(deleted_titles),
            topics_deleted=deleted_titles,
        )


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)
