"""Turns a topic into a vector, so near-duplicate concepts can be found by
meaning rather than by matching titles.

Anthropic doesn't serve an embeddings endpoint, so this uses Voyage AI
(anthropic's recommended embeddings partner) behind a small protocol, the
same pattern agent.py uses for the Claude client: production code depends on
an interface, not a concrete SDK, so tests can supply a fake and run offline.
"""

from __future__ import annotations

from typing import Protocol

from retrieve.schema import Topic

# voyage-3-lite is cheap and its 512 dimensions keep the pgvector index
# small. The topics.embedding column in migrations/001_init.sql is sized
# for this model; switching models means changing both and re-embedding.
DEFAULT_EMBEDDING_MODEL = "voyage-3-lite"
EMBEDDING_DIMENSIONS = 512


class EmbeddingClient(Protocol):
    """The one method we need from voyageai.Client()."""

    def embed(self, texts: list[str], model: str, input_type: str) -> object: ...


def topic_embedding_text(topic: Topic) -> str:
    """The text embedded for a topic: title and summary carry the concept's
    meaning; key_points and tags add exact terms without repeating them
    verbatim, which is closer to how a learner would describe the topic."""
    return "\n".join([
        topic.title,
        topic.summary,
        "Key points: " + "; ".join(topic.key_points),
        "Tags: " + ", ".join(topic.tags),
    ])


def embed_topic(topic: Topic, client: EmbeddingClient, model: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    """Embed one topic. input_type='document' matches Voyage's asymmetric
    mode for the side being stored and searched, as opposed to 'query' for
    the side doing the searching (this project only ever searches with
    another topic's embedding, so every embedding here is a 'document')."""
    result = client.embed([topic_embedding_text(topic)], model=model, input_type="document")
    return list(result.embeddings[0])
