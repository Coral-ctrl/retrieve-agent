from retrieve.embeddings import embed_topic, topic_embedding_text
from retrieve.schema import Difficulty, Topic
from tests.fakes import FakeEmbeddingClient

TOPIC = Topic(
    title="TCP three-way handshake",
    summary="TCP opens a connection with SYN, SYN-ACK, ACK before data flows.",
    key_points=["SYN", "SYN-ACK", "ACK"],
    tags=["networking", "tcp"],
    difficulty=Difficulty.CORE,
    source_pages=[2],
)


def test_topic_embedding_text_includes_every_field() -> None:
    text = topic_embedding_text(TOPIC)
    assert "TCP three-way handshake" in text
    assert "SYN, SYN-ACK, ACK before data flows" in text
    assert "SYN; SYN-ACK; ACK" in text
    assert "networking, tcp" in text


def test_embed_topic_sends_the_embedding_text_and_returns_the_vector() -> None:
    text = topic_embedding_text(TOPIC)
    client = FakeEmbeddingClient({text: [0.1, 0.2, 0.3]})

    vector = embed_topic(TOPIC, client, model="voyage-3-lite")

    assert vector == [0.1, 0.2, 0.3]
    assert client.received_calls == [{
        "texts": [text], "model": "voyage-3-lite", "input_type": "document",
    }]
