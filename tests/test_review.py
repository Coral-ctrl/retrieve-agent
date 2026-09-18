"""Tests for the review queue order and the review session.

The session talks to a scripted console instead of the real terminal, the
same way the agent-loop tests script the Claude client, so every keypress
path (rate, skip, quit, typo) is checked exactly and offline.
"""

from datetime import datetime, timedelta, timezone

from retrieve.review import ReviewSession, order_for_review
from retrieve.schema import Difficulty, Rating, ReviewEvent, StoredTopic, TopicSource
from tests.fakes import FakeRepository

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def _stored(topic_id: int, title: str, *ratings: Rating,
            last_review_days_ago: int = 1) -> StoredTopic:
    """A topic with the given rating history, one review per day, the last of
    them `last_review_days_ago` days before NOW."""
    last_at = NOW - timedelta(days=last_review_days_ago)
    reviews = [
        ReviewEvent(rating=r, reviewed_at=last_at - timedelta(days=len(ratings) - 1 - i))
        for i, r in enumerate(ratings)
    ]
    return StoredTopic(
        id=topic_id,
        title=title,
        summary="TCP opens a connection with SYN, SYN-ACK, ACK before data flows.",
        key_points=["SYN", "SYN-ACK", "ACK"],
        tags=["networking"],
        difficulty=Difficulty.CORE,
        sources=[TopicSource(file_name="week-1.pdf", course="Networking", pages=[2])],
        reviews=reviews,
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
#
# Intervals from schedule.py: one solid -> due 1 day after; two solids -> 3
# days after; forgotten -> 1 day after. `last_review_days_ago` positions that
# due date relative to NOW.

def test_only_due_and_new_topics_are_queued_by_default() -> None:
    topics = [
        _stored(1, "Due: forgotten yesterday", Rating.FORGOTTEN, last_review_days_ago=1),
        _stored(2, "Not due: two solids, reviewed yesterday", Rating.SOLID, Rating.SOLID, last_review_days_ago=1),
        _stored(3, "New, never reviewed"),
    ]

    queue = [t.title for t in order_for_review(topics, now=NOW)]

    assert queue == ["Due: forgotten yesterday", "New, never reviewed"]


def test_due_topics_come_before_new_ones_most_overdue_first() -> None:
    topics = [
        _stored(1, "New one"),
        _stored(2, "Overdue by 1 day", Rating.SOLID, last_review_days_ago=2),
        _stored(3, "Overdue by 4 days", Rating.SOLID, last_review_days_ago=5),
        _stored(4, "Due today", Rating.SOLID, last_review_days_ago=1),
    ]

    queue = [t.title for t in order_for_review(topics, now=NOW)]

    assert queue == ["Overdue by 4 days", "Overdue by 1 day", "Due today", "New one"]


def test_include_upcoming_appends_not_yet_due_topics_soonest_first() -> None:
    topics = [
        _stored(1, "Due in 2 days", Rating.SOLID, Rating.SOLID, last_review_days_ago=1),
        _stored(2, "Due in 7 days", Rating.SOLID, Rating.SOLID, Rating.SOLID, last_review_days_ago=1),
        _stored(3, "New one"),
        _stored(4, "Due today", Rating.SOLID, last_review_days_ago=1),
    ]

    default = [t.title for t in order_for_review(topics, now=NOW)]
    everything = [t.title for t in order_for_review(topics, now=NOW, include_upcoming=True)]

    assert default == ["Due today", "New one"]
    assert everything == ["Due today", "New one", "Due in 2 days", "Due in 7 days"]


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


def test_context_line_shows_course_source_and_due_status() -> None:
    console = ScriptedConsole(["", "s", "", "s", "", "s"])
    topics = [
        _stored(1, "Overdue", Rating.SHAKY, last_review_days_ago=3),  # 1-day gap, 3 days ago
        _stored(2, "Fresh"),
        _stored(3, "Ahead", Rating.SOLID, Rating.SOLID, last_review_days_ago=1),  # due in 2 days
    ]

    ReviewSession(topics, FakeRepository(), console, now=NOW).run()

    transcript = console.transcript
    assert "core | Networking | from week-1.pdf | overdue by 2 days (last: shaky)" in transcript
    assert "core | Networking | from week-1.pdf | new" in transcript
    assert "core | Networking | from week-1.pdf | due in 2 days (last: solid)" in transcript


def test_each_rating_reports_when_the_topic_comes_back() -> None:
    console = ScriptedConsole(["", "s", "", "f", "", "h"])
    topics = [
        _stored(1, "First solid", ),  # new -> first solid = 1 day
        _stored(2, "Forgotten", Rating.SOLID, Rating.SOLID, Rating.SOLID),  # -> back to 1 day
        _stored(3, "Shaky", Rating.SOLID, Rating.SOLID),  # repeats the 3-day interval
    ]

    ReviewSession(topics, FakeRepository(), console, now=NOW).run()

    transcript = console.transcript
    assert "-> solid. Back in 1 day." in transcript
    assert "-> forgotten. Back in 1 day." in transcript
    assert "-> shaky. Back in 3 days." in transcript


def test_an_empty_queue_says_whether_anything_is_scheduled_for_later() -> None:
    nothing_at_all = ScriptedConsole([])
    ReviewSession([], FakeRepository(), nothing_at_all, total_topics=0).run()
    assert "Nothing to review yet" in nothing_at_all.transcript

    nothing_due = ScriptedConsole([])
    ReviewSession([], FakeRepository(), nothing_due, total_topics=6).run()
    assert "Nothing due right now (6 topic(s) scheduled for later)" in nothing_due.transcript
    assert "--all" in nothing_due.transcript
