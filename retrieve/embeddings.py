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

# voyage-4-lite is Voyage's current small model: same price per token as the
# older voyage-3-lite, but eligible for Voyage's 200M-free-tokens-a-month
# tier (voyage-3-lite isn't -- Voyage limits free tokens to current-generation
# models), which covers this project's usage many times over. It defaults to
# 1024 dimensions but supports 256/512/1024/2048 via output_dimension, so we
# ask for 512 explicitly to keep the pgvector index small -- the
# topics.embedding column in migrations/001_init.sql is sized for that.
# Changing EMBEDDING_DIMENSIONS means changing the column and re-embedding.
DEFAULT_EMBEDDING_MODEL = "voyage-4-lite"
EMBEDDING_DIMENSIONS = 512


class EmbeddingClient(Protocol):
    """The one method we need from voyageai.Client()."""

    def embed(self, texts: list[str], model: str, input_type: str, output_dimension: int) -> object: ...


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


def embed_topic(
    topic: Topic,
    client: EmbeddingClient,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Embed one topic. input_type='document' matches Voyage's asymmetric
    mode for the side being stored and searched, as opposed to 'query' for
    the side doing the searching (this project only ever searches with
    another topic's embedding, so every embedding here is a 'document')."""
    result = client.embed(
        [topic_embedding_text(topic)], model=model, input_type="document", output_dimension=dimensions
    )
    return list(result.embeddings[0])
