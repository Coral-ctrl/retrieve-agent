"""Tests for the review queue order and the review session.

The session talks to a scripted console instead of the real terminal, the
same way the agent-loop tests script the Claude client, so every keypress
path (rate, skip, quit, typo) is checked exactly and offline.
"""

from datetime import datetime, timedelta, timezone

from retrieve.review import ReviewSession, order_for_review
from retrieve.schema import Difficulty, Rating, StoredTopic, TopicSource
from tests.fakes import FakeRepository

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def _stored(topic_id: int, title: str, last_rating: Rating | None = None,
            reviewed_days_ago: int | None = None) -> StoredTopic:
    return StoredTopic(
        id=topic_id,
        title=title,
        summary="TCP opens a connection with SYN, SYN-ACK, ACK before data flows.",
        key_points=["SYN", "SYN-ACK", "ACK"],
        tags=["networking"],
        difficulty=Difficulty.CORE,
        sources=[TopicSource(file_name="week-1.pdf", course="Networking", pages=[2])],
        review_count=0 if last_rating is None else 1,
        last_rating=last_rating,
        last_reviewed_at=NOW - timedelta(days=reviewed_days_ago) if reviewed_days_ago is not None else None,
    )


class ScriptedConsole:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.output: list[str] = []

    def write(self, text: str) -> None:
        self.output.append(text)

    def read(self, prompt: str) -> str:
        self.output.append(prompt)
        if not self.answers:
            raise AssertionError("Session asked for more input than the test scripted.")
        return self.answers.pop(0)

    @property
    def transcript(self) -> str:
        return "\n".join(self.output)


# --- queue order ----------------------------------------------------------

def test_never_reviewed_topics_come_first_then_forgotten_then_shaky_then_solid() -> None:
    topics = [
        _stored(1, "Solid one", Rating.SOLID, reviewed_days_ago=1),
        _stored(2, "Shaky one", Rating.SHAKY, reviewed_days_ago=1),
        _stored(3, "Forgotten one", Rating.FORGOTTEN, reviewed_days_ago=1),
        _stored(4, "New one"),
    ]

    ordered = [t.title for t in order_for_review(topics)]

    assert ordered == ["New one", "Forgotten one", "Shaky one", "Solid one"]


def test_within_a_rating_the_least_recently_reviewed_comes_first() -> None:
    topics = [
        _stored(1, "Seen yesterday", Rating.SHAKY, reviewed_days_ago=1),
        _stored(2, "Seen last month", Rating.SHAKY, reviewed_days_ago=30),
    ]

    assert [t.title for t in order_for_review(topics)] == ["Seen last month", "Seen yesterday"]


# --- session --------------------------------------------------------------

def test_title_is_shown_before_the_summary_is_revealed() -> None:
    console = ScriptedConsole(["", "s"])
    session = ReviewSession([_stored(1, "TCP three-way handshake")], FakeRepository(), console)

    session.run()

    transcript = console.transcript
    title_position = transcript.index("[1/1] TCP three-way handshake")
    reveal_prompt_position = transcript.index("Explain it to yourself, then press Enter")
    summary_position = transcript.index("TCP opens a connection")
    assert title_position < reveal_prompt_position < summary_position


def test_ratings_are_recorded_and_summarised() -> None:
    repository = FakeRepository()
    console = ScriptedConsole(["", "s", "", "f", "", "h"])
    topics = [_stored(1, "One"), _stored(2, "Two"), _stored(3, "Three")]

    summary = ReviewSession(topics, repository, console).run()

    assert repository.reviews == [(1, Rating.SOLID), (2, Rating.FORGOTTEN), (3, Rating.SHAKY)]
    assert summary.reviewed == 3
    assert "Rated 3 topic(s): 1 solid, 1 shaky, 1 forgotten." in console.transcript


def test_skip_records_nothing_and_moves_on() -> None:
    repository = FakeRepository()
    console = ScriptedConsole(["", "k", "", "s"])

    summary = ReviewSession([_stored(1, "One"), _stored(2, "Two")], repository, console).run()

    assert repository.reviews == [(2, Rating.SOLID)]
    assert summary.skipped == 1
    assert "1 skipped" in console.transcript


def test_quit_stops_at_either_prompt() -> None:
    repository = FakeRepository()

    # Quit before revealing the first topic.
    ReviewSession([_stored(1, "One"), _stored(2, "Two")], repository, ScriptedConsole(["q"])).run()
    assert repository.reviews == []

    # Quit at the rating prompt: the reveal happened, but nothing is recorded.
    console = ScriptedConsole(["", "q"])
    ReviewSession([_stored(1, "One"), _stored(2, "Two")], repository, console).run()
    assert repository.reviews == []
    assert "No topics rated." in console.transcript


def test_an_unknown_key_is_re_asked_instead_of_recorded() -> None:
    repository = FakeRepository()
    console = ScriptedConsole(["", "x", "S"])  # typo, then a capital letter

    ReviewSession([_stored(1, "One")], repository, console).run()

    assert "Please type s, h, f, k or q." in console.transcript
    assert repository.reviews == [(1, Rating.SOLID)]


def test_empty_queue_explains_itself() -> None:
    console = ScriptedConsole([])

    ReviewSession([], FakeRepository(), console).run()

    assert "Nothing to review yet" in console.transcript


def test_context_line_shows_course_source_and_last_rating() -> None:
    console = ScriptedConsole(["", "s"])
    topic = _stored(1, "One", Rating.SHAKY, reviewed_days_ago=3)

    ReviewSession([topic], FakeRepository(), console).run()

    assert "core | Networking | from week-1.pdf | last time: shaky" in console.transcript
