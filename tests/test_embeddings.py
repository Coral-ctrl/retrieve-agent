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

    vector = embed_topic(TOPIC, client, model="voyage-4-lite", dimensions=256)

    assert vector == [0.1, 0.2, 0.3]
    assert client.received_calls == [{
        "texts": [text], "model": "voyage-4-lite", "input_type": "document", "output_dimension": 256,
    }]


def test_embed_topics_sends_every_topic_in_one_request_in_order() -> None:
    from retrieve.embeddings import embed_topics

    other = TOPIC.model_copy(update={"title": "UDP basics"})
    client = FakeEmbeddingClient({
        topic_embedding_text(TOPIC): [1.0, 0.0],
        topic_embedding_text(other): [0.0, 1.0],
    })

    vectors = embed_topics([TOPIC, other], client)

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert len(client.received_calls) == 1  # one request, not one per topic
    assert client.received_calls[0]["texts"] == [topic_embedding_text(TOPIC), topic_embedding_text(other)]


def test_embed_topics_with_nothing_to_embed_makes_no_request() -> None:
    from retrieve.embeddings import embed_topics

    client = FakeEmbeddingClient({})
    assert embed_topics([], client) == []
    assert client.received_calls == []


def test_embed_topic_defaults_to_voyage_4_lite_at_512_dimensions() -> None:
    text = topic_embedding_text(TOPIC)
    client = FakeEmbeddingClient({text: [0.0]})

    embed_topic(TOPIC, client)

    assert client.received_calls[0]["model"] == "voyage-4-lite"
    assert client.received_calls[0]["output_dimension"] == 512
