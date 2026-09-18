"""Tests for `ask`: retrieval routing, the prompt we build, and how the
answer is read back -- all against fakes, so no Voyage or Claude call.
tests/test_db_integration.py covers the real vector and full-text queries.
"""

from datetime import datetime, timezone

from retrieve.ask import (
    RetrievedTopic,
    answer_question,
    format_answer,
    format_notes_for_prompt,
    retrieve_topics,
)
from retrieve.run_log import RunLogger
from retrieve.schema import Difficulty, StoredTopic, TopicSource
from tests.fakes import FakeEmbeddingClient, FakeMessagesClient, fake_response, text_block


def _stored(topic_id: int, title: str, summary: str = "TCP opens with SYN, SYN-ACK, ACK.") -> StoredTopic:
    return StoredTopic(
        id=topic_id, title=title, summary=summary, key_points=["SYN", "ACK"], tags=["tcp"],
        difficulty=Difficulty.CORE,
        sources=[TopicSource(file_name="week-1.pdf", course="Networking", pages=[2, 3])],
    )


class StubStore:
    def __init__(self, vector_hits=(), text_hits=()):
        self.vector_hits = list(vector_hits)
        self.text_hits = list(text_hits)
        self.calls: list[tuple] = []

    def find_similar_topics(self, embedding, limit, course=None):
        self.calls.append(("vector", embedding, limit, course))
        return self.vector_hits[:limit]

    def search_topics_by_text(self, query, limit, course=None):
        self.calls.append(("text", query, limit, course))
        return self.text_hits[:limit]


# --- retrieval routing ------------------------------------------------------

def test_with_an_embedder_the_question_is_embedded_as_a_query_and_vector_searched() -> None:
    handshake = _stored(1, "TCP three-way handshake")
    embedder = FakeEmbeddingClient({"how does tcp connect?": [1.0, 0.0]})
    store = StubStore(vector_hits=[(handshake, 0.21)])

    retrieved, used_vector = retrieve_topics("how does tcp connect?", store, embedder,
                                             top_k=3, course="Networking")

    assert used_vector
    assert embedder.received_calls[0]["input_type"] == "query"  # not 'document'
    assert store.calls == [("vector", [1.0, 0.0], 3, "Networking")]
    assert [(r.topic.title, r.distance) for r in retrieved] == [("TCP three-way handshake", 0.21)]


def test_without_an_embedder_it_falls_back_to_text_search() -> None:
    store = StubStore(text_hits=[_stored(1, "TCP three-way handshake")])

    retrieved, used_vector = retrieve_topics("tcp handshake", store, embedder=None, top_k=5)

    assert not used_vector
    assert store.calls == [("text", "tcp handshake", 5, None)]
    assert retrieved[0].distance is None


# --- the prompt and the answer ----------------------------------------------

def test_notes_are_numbered_with_sources_and_key_points_in_the_prompt() -> None:
    text = format_notes_for_prompt([
        RetrievedTopic(_stored(1, "TCP three-way handshake"), 0.2),
        RetrievedTopic(_stored(2, "TCP vs UDP", "UDP has no handshake."), 0.4),
    ])

    assert text.startswith("[1] TCP three-way handshake  (week-1.pdf p.2,3)\nTCP opens with SYN, SYN-ACK, ACK.\n- SYN\n- ACK")
    assert "\n\n[2] TCP vs UDP  (week-1.pdf p.2,3)\nUDP has no handshake." in text


def test_answer_question_grounds_the_call_and_reads_back_citations(tmp_path) -> None:
    retrieved = [
        RetrievedTopic(_stored(1, "TCP three-way handshake"), 0.2),
        RetrievedTopic(_stored(2, "TCP vs UDP"), 0.5),
    ]
    client = FakeMessagesClient([
        fake_response(text_block("TCP connects with SYN, SYN-ACK, ACK [1]."), stop_reason="end_turn"),
    ])
    logger = RunLogger(tmp_path, "ask")

    answer = answer_question("how does tcp connect?", retrieved, client, "claude-haiku-4-5", logger)

    request = client.received_calls[0]
    assert "answer only from the notes" in request["system"].lower()
    assert request["messages"][0]["content"].startswith("Notes:\n\n[1] TCP three-way handshake")
    assert request["messages"][0]["content"].endswith("Question: how does tcp connect?")
    assert answer.text == "TCP connects with SYN, SYN-ACK, ACK [1]."
    assert answer.unused_notes == ["TCP vs UDP"]  # retrieved but not cited
    assert answer.input_tokens == 1000 and answer.output_tokens == 200
    assert logger.tool_call_count == 0 and logger.total_input_tokens == 1000


def test_answer_question_with_nothing_retrieved_makes_no_model_call(tmp_path) -> None:
    client = FakeMessagesClient([])  # any call would fail the test

    answer = answer_question("anything", [], client, "claude-haiku-4-5", RunLogger(tmp_path, "ask"))

    assert answer.text == "No notes found to answer from."
    assert client.received_calls == []


def test_format_answer_shows_matches_answer_sources_and_cost() -> None:
    from retrieve.ask import Answer

    answer = Answer(
        question="q", text="TCP connects with SYN, SYN-ACK, ACK [1].",
        retrieved=[RetrievedTopic(_stored(1, "TCP three-way handshake"), 0.2134)],
        used_vector_search=True, input_tokens=800, output_tokens=60,
    )

    text = format_answer(answer, cost_usd=0.0011)

    assert "Found 1 relevant note(s) by meaning:" in text
    assert "[1] TCP three-way handshake  distance 0.21" in text
    assert "TCP connects with SYN, SYN-ACK, ACK [1]." in text
    assert "Sources:\n  [1] week-1.pdf p.2, 3" in text
    assert "800 input / 60 output tokens, ~US$0.0011" in text


def test_format_answer_says_when_search_was_by_keyword() -> None:
    from retrieve.ask import Answer

    answer = Answer(question="q", text="...", used_vector_search=False,
                    retrieved=[RetrievedTopic(_stored(1, "TCP three-way handshake"), None)])

    assert "by keyword (no VOYAGE_API_KEY)" in format_answer(answer, None)
