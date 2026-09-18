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
# older voyage-3-lite, but eligible for Voyage's one-time 200M free tokens
# per account (voyage-3-lite isn't -- Voyage limits free tokens to
# current-generation models). A topic is a few hundred tokens, so that
# allowance is on the order of a million topics; this project won't dent it.
#
# It defaults to 1024 dimensions but supports 256/512/1024/2048 via
# output_dimension, so we ask for 512 explicitly to keep the pgvector index
# small -- the topics.embedding column in migrations/001_init.sql is sized
# for that. Changing EMBEDDING_DIMENSIONS means changing the column and
# re-embedding.
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


def embed_topics(
    topics: list[Topic],
    client: EmbeddingClient,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> list[list[float]]:
    """Embed every topic in ONE request, returning vectors in the same order.

    One request per digest rather than one per topic matters more than it
    looks: a Voyage account with no payment method is limited to 3 requests
    a minute, so embedding a six-topic file one topic at a time fails
    halfway through. Batched, a whole file is a single request.

    input_type='document' matches Voyage's asymmetric mode for the side
    being stored and searched, as opposed to 'query' for the side doing the
    searching (this project only ever searches with another topic's
    embedding, so every embedding here is a 'document').
    """
    if not topics:
        return []
    result = client.embed(
        [topic_embedding_text(topic) for topic in topics],
        model=model,
        input_type="document",
        output_dimension=dimensions,
    )
    return [list(vector) for vector in result.embeddings]


def embed_topic(
    topic: Topic,
    client: EmbeddingClient,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Embed a single topic. Prefer embed_topics for more than one."""
    return embed_topics([topic], client, model=model, dimensions=dimensions)[0]


def embed_query(
    question: str,
    client: EmbeddingClient,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
) -> list[float]:
    """Embed a search question. input_type='query' is the other half of
    Voyage's asymmetric mode: notes are stored as 'document', questions are
    embedded as 'query', and the model is trained so a short question lands
    near the longer passage that answers it."""
    result = client.embed([question], model=model, input_type="query", output_dimension=dimensions)
    return list(result.embeddings[0])
