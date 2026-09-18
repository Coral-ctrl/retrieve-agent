"""The web app, exercised through FastAPI's test client against a fake
repository and a scripted Claude client -- no server, browser, database or
API key. These check that each page wires the right inputs to the already-
tested functions and renders what they return; the study logic itself is
covered in test_schedule.py, test_review.py, test_stats.py and test_ask.py.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from retrieve.schema import Difficulty, Rating, ReviewEvent, StoredTopic, TopicSource
from retrieve.web.app import WebDeps, create_app
from tests.fakes import FakeMessagesClient, fake_response, text_block, tool_use_block

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)


def _topic(topic_id: int, title: str, *ratings: Rating, days_ago: int = 1,
           course: str = "Networking") -> StoredTopic:
    last = NOW - timedelta(days=days_ago)
    return StoredTopic(
        id=topic_id, title=title, summary=f"Summary of {title}.", key_points=["point one", "point two"],
        tags=["tcp"], difficulty=Difficulty.CORE,
        sources=[TopicSource(file_name="week-1.pdf", course=course, pages=[2])],
        reviews=[ReviewEvent(rating=r, reviewed_at=last - timedelta(days=len(ratings) - 1 - i))
                 for i, r in enumerate(ratings)],
    )


class WebFakeRepository:
    """Just enough of Repository for the pages, keyed by course, plus the
    sync.sync_digest surface used by the New material page."""

    def __init__(self, topics: list[StoredTopic]) -> None:
        self.topics = {t.id: t for t in topics}
        self.recorded: list[tuple[int, Rating]] = []
        self._next_id = max(self.topics, default=0) + 1
        self._synced: list = []  # digests handed to sync_digest, for assertions

    def list_topics(self, course=None):
        return [t for t in self.topics.values()
                if course is None or course.lower() in [c.lower() for c in t.courses]]

    def get_topics_by_ids(self, ids):
        return [self.topics[i] for i in ids if i in self.topics]

    def record_review(self, topic_id, rating):
        self.recorded.append((topic_id, rating))
        topic = self.topics[topic_id]
        topic.reviews.append(ReviewEvent(rating=rating, reviewed_at=NOW))

    def find_similar_topics(self, embedding, limit, course=None):
        raise AssertionError("no embedder in these tests")

    def search_topics_by_text(self, query, limit, course=None):
        return [t for t in self.list_topics(course) if any(w in t.title.lower() for w in query.lower().split())][:limit]

    # --- sync.sync_digest surface (New material page) ------------------------

    def transaction(self):
        from contextlib import nullcontext
        return nullcontext()

    def get_or_create_source(self, file_name, course, model, agent_note):
        return 1

    def find_nearest_topic(self, embedding):
        return None  # every test topic here has no embedding to match against

    def find_topic_by_title(self, title):
        for topic in self.topics.values():
            if topic.title.lower() == title.lower():
                from retrieve.db import NearestTopic
                return NearestTopic(id=topic.id, title=topic.title, distance=0.0)
        return None

    def insert_topic(self, topic, embedding):
        topic_id = self._next_id
        self._next_id += 1
        self.topics[topic_id] = StoredTopic(
            id=topic_id, title=topic.title, summary=topic.summary, key_points=topic.key_points,
            tags=topic.tags, difficulty=topic.difficulty,
            sources=[TopicSource(file_name="upload", course=None, pages=topic.source_pages)],
        )
        self._synced.append(topic)
        return topic_id

    def link_source(self, topic_id, source_id, source_pages):
        pass

    def merge_tags(self, topic_id, tags):
        pass

    def source_count_for_topic(self, topic_id):
        return 1


def make_client(topics, messages_client=None, tmp_path=None, database_available=True):
    repo = WebFakeRepository(topics)

    @contextmanager
    def open_repository():
        yield repo

    app = create_app(WebDeps(
        open_repository=open_repository, embedder=None, messages_client=messages_client,
        model="claude-haiku-4-5", embedding_model="voyage-4-lite",
        logs_dir=tmp_path, knowledge_dir=tmp_path, max_turns=10, dedup_threshold=0.10,
        database_available=database_available, now=lambda: NOW,
    ))
    return TestClient(app), repo


# --- dashboard ----------------------------------------------------------------

def test_dashboard_shows_counts_queue_and_all_topics(tmp_path) -> None:
    client, _ = make_client([
        _topic(1, "TCP three-way handshake"),  # new
        _topic(2, "Flow control", Rating.SOLID, days_ago=3),  # 1-day interval -> overdue
        _topic(3, "TCP vs UDP", Rating.SOLID, Rating.SOLID, days_ago=1),  # due in 2 days
    ], tmp_path=tmp_path)

    page = client.get("/").text

    assert "3 topics" in page
    assert "to review now" in page and ">2<" in page  # 1 due + 1 new
    assert "1 due · 1 new" in page
    assert "Flow control" in page and "TCP three-way handshake" in page
    assert "overdue by 2 days (last: solid)" in page
    assert "due in 2 days (last: solid)" in page
    assert ">new<" in page


def test_dashboard_can_filter_by_course(tmp_path) -> None:
    client, _ = make_client([
        _topic(1, "TCP", course="Networking"),
        _topic(2, "Indexes", course="Databases"),
    ], tmp_path=tmp_path)

    page = client.get("/?course=Databases").text

    assert "Indexes" in page and "TCP three" not in page
    assert "<h1>Databases</h1>" in page


def test_dashboard_with_no_topics_explains_what_to_run(tmp_path) -> None:
    client, _ = make_client([], tmp_path=tmp_path)
    assert "retrieve summarize" in client.get("/").text


# --- review -----------------------------------------------------------------

def test_review_shows_the_most_overdue_topic_title_without_the_answer(tmp_path) -> None:
    client, _ = make_client([
        _topic(1, "Less overdue", Rating.SOLID, days_ago=2),
        _topic(2, "Most overdue", Rating.SOLID, days_ago=5),
    ], tmp_path=tmp_path)

    page = client.get("/review").text

    assert 'class="topic-title">Most overdue<' in page
    assert "Summary of Most overdue." not in page  # recall first: no answer on the page
    assert "2 to review" in page
    assert "Reveal" in page


def test_reveal_returns_summary_key_points_and_rating_buttons_with_previews(tmp_path) -> None:
    client, _ = make_client([_topic(7, "Flow control", Rating.SOLID, Rating.SOLID)], tmp_path=tmp_path)

    fragment = client.get("/review/7/reveal").text

    assert "Summary of Flow control." in fragment
    assert "<li>point one</li>" in fragment
    # Preview of what each rating would do: two solids -> 8d, shaky repeats 3d, forgotten 1d.
    assert "back in 8d" in fragment and "back in 3d" in fragment and "back in 1d" in fragment
    assert client.get("/review/999/reveal").status_code == 404


def test_rating_records_the_review_and_redirects_to_the_next_topic(tmp_path) -> None:
    client, repo = make_client([
        _topic(1, "First", Rating.SOLID, days_ago=5),
        _topic(2, "Second", Rating.SOLID, days_ago=3),
    ], tmp_path=tmp_path)

    response = client.post("/review/1/rate", data={"rating": "solid", "course": ""}, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/review?course=&just=1&rated=solid&days=3"
    assert repo.recorded == [(1, Rating.SOLID)]

    next_page = client.get(response.headers["location"]).text
    assert "Rated <strong>solid</strong>" in next_page and "back in 3 days" in next_page
    assert 'class="topic-title">Second<' in next_page  # First is no longer due


def test_rating_accepts_terminal_shortcut_letters_and_rejects_junk(tmp_path) -> None:
    client, repo = make_client([_topic(1, "First")], tmp_path=tmp_path)

    assert client.post("/review/1/rate", data={"rating": "f"}, follow_redirects=False).status_code == 303
    assert repo.recorded == [(1, Rating.FORGOTTEN)]
    assert client.post("/review/1/rate", data={"rating": "excellent"}).status_code == 400
    assert client.post("/review/999/rate", data={"rating": "solid"}).status_code == 404


def test_review_when_nothing_is_due_offers_all_and_all_includes_upcoming(tmp_path) -> None:
    client, _ = make_client([_topic(1, "Ahead", Rating.SOLID, Rating.SOLID, days_ago=1)], tmp_path=tmp_path)

    caught_up = client.get("/review").text
    assert "All caught up" in caught_up and "1 topic scheduled for later" in caught_up

    ahead = client.get("/review?all=true").text
    assert 'class="topic-title">Ahead<' in ahead and "including upcoming" in ahead


def test_review_with_no_topics_at_all(tmp_path) -> None:
    client, _ = make_client([], tmp_path=tmp_path)
    assert "Nothing to review yet" in client.get("/review").text


# --- ask --------------------------------------------------------------------

def test_ask_answers_from_notes_and_shows_sources(tmp_path) -> None:
    messages = FakeMessagesClient([
        fake_response(text_block("Slow start backs off on loss [1]."), stop_reason="end_turn"),
    ])
    client, _ = make_client([_topic(1, "Congestion control")], messages_client=messages, tmp_path=tmp_path)

    page = client.post("/ask", data={"question": "how does congestion control work", "course": ""}).text

    assert "Slow start backs off on loss [1]." in page
    assert "Congestion control" in page and "week-1.pdf p.2" in page
    assert "by keyword (no VOYAGE_API_KEY)" in page
    assert "~US$" in page
    assert messages.received_calls[0]["messages"][0]["content"].startswith("Notes:\n\n[1] Congestion control")


def test_ask_page_without_an_api_key_is_read_only(tmp_path) -> None:
    client, _ = make_client([_topic(1, "Congestion control")], messages_client=None, tmp_path=tmp_path)

    page = client.get("/ask").text
    assert "ANTHROPIC_API_KEY isn't set" in page and "disabled" in page

    posted = client.post("/ask", data={"question": "anything", "course": ""}).text
    assert "ANTHROPIC_API_KEY isn't set" in posted


TOPIC_INPUT = {
    "title": "Congestion window",
    "summary": "TCP shrinks its congestion window on packet loss to ease off the network.",
    "key_points": ["grows on success", "halves on loss"],
    "tags": ["tcp"],
    "difficulty": "core",
    "source_pages": [1],
}


def _upload_file(name: str = "notes.txt", text: str = "TCP shrinks its window on loss."):
    return {"file": (name, text.encode("utf-8"), "text/plain")}


def test_new_page_is_disabled_without_an_api_key(tmp_path) -> None:
    client, _ = make_client([], messages_client=None, tmp_path=tmp_path)

    page = client.get("/new").text
    assert "ANTHROPIC_API_KEY isn't set" in page and "disabled" in page

    posted = client.post("/new", data={"course": ""}, files=_upload_file()).text
    assert "ANTHROPIC_API_KEY isn't set" in posted


def test_new_material_is_summarized_and_synced(tmp_path) -> None:
    messages = FakeMessagesClient([
        fake_response(tool_use_block("t0", "read_pages", {"start_page": 1, "end_page": 1})),
        fake_response(tool_use_block("t1", "save_topic", TOPIC_INPUT)),
        fake_response(tool_use_block("t2", "finish", {"note": "Done."})),
    ])
    client, repo = make_client([], messages_client=messages, tmp_path=tmp_path)

    page = client.post("/new", data={"course": "Networking"}, files=_upload_file()).text

    assert "Congestion window" in page
    assert "1 new, 0 merged" in page
    assert "~US$" in page
    assert any(t.title == "Congestion window" for t in repo.topics.values())


def test_new_material_with_an_unsupported_file_shows_the_error(tmp_path) -> None:
    client, _ = make_client([], messages_client=FakeMessagesClient([]), tmp_path=tmp_path)

    page = client.post("/new", data={"course": ""}, files=_upload_file(name="slides.key")).text

    assert "Can&#39;t read" in page or "Can't read" in page


def test_new_material_when_the_agent_saves_nothing_shows_the_error(tmp_path) -> None:
    # Never reads the page or saves a topic, so every finish attempt is
    # rejected until make_client's max_turns (10) runs out.
    messages = FakeMessagesClient([fake_response(tool_use_block(f"t{i}", "finish", {"note": "done"}))
                                    for i in range(10)])
    client, _ = make_client([], messages_client=messages, tmp_path=tmp_path)

    page = client.post("/new", data={"course": ""}, files=_upload_file()).text

    assert "didn&#39;t save any topics" in page or "didn't save any topics" in page


def test_new_material_without_a_database_skips_sync(tmp_path) -> None:
    messages = FakeMessagesClient([
        fake_response(tool_use_block("t0", "read_pages", {"start_page": 1, "end_page": 1})),
        fake_response(tool_use_block("t1", "save_topic", TOPIC_INPUT)),
        fake_response(tool_use_block("t2", "finish", {"note": "Done."})),
    ])
    client, _ = make_client([], messages_client=messages, tmp_path=tmp_path, database_available=False)

    page = client.post("/new", data={"course": ""}, files=_upload_file()).text

    assert "Congestion window" in page
    assert "saved to files only" in page or "not synced" in page


def test_ask_with_no_matching_notes_makes_no_model_call(tmp_path) -> None:
    messages = FakeMessagesClient([])  # any call would raise
    client, _ = make_client([_topic(1, "Congestion control")], messages_client=messages, tmp_path=tmp_path)

    page = client.post("/ask", data={"question": "quantum entanglement", "course": ""}).text

    assert "No notes found to answer from." in page
    assert messages.received_calls == []
