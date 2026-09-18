from datetime import date, datetime, timedelta, timezone

from retrieve.schema import Difficulty, Rating, ReviewEvent, StoredTopic, TopicSource
from retrieve.stats import compute_stats, format_stats

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def _topic(topic_id: int, *reviews: tuple[Rating, int]) -> StoredTopic:
    """`reviews` are (rating, days_ago) pairs, given oldest first."""
    return StoredTopic(
        id=topic_id, title=f"Topic {topic_id}", summary="x" * 25, key_points=["k"],
        tags=["t"], difficulty=Difficulty.CORE,
        sources=[TopicSource(file_name="week-1.pdf", course="Networking", pages=[1])],
        reviews=[ReviewEvent(rating=r, reviewed_at=NOW - timedelta(days=d)) for r, d in reviews],
    )


def test_topics_are_bucketed_into_due_new_and_upcoming() -> None:
    topics = [
        _topic(1),  # new
        _topic(2, (Rating.SOLID, 3)),  # 1-day interval, 3 days ago -> overdue
        _topic(3, (Rating.SOLID, 2), (Rating.SOLID, 1)),  # 3-day interval from yesterday -> due in 2 days
        _topic(4, (Rating.FORGOTTEN, 0)),  # rated today, due tomorrow
    ]

    stats = compute_stats(topics, now=NOW)

    assert stats.total == 4
    assert stats.new == 1
    assert stats.due_now == 1
    assert stats.upcoming == 2
    assert stats.next_due == (NOW + timedelta(days=1)).date()


def test_last_rating_breakdown_and_review_counts() -> None:
    topics = [
        _topic(1, (Rating.FORGOTTEN, 2), (Rating.SOLID, 0)),
        _topic(2, (Rating.SHAKY, 0)),
        _topic(3),
    ]

    stats = compute_stats(topics, now=NOW)

    assert stats.by_last_rating == {Rating.SOLID: 1, Rating.SHAKY: 1, Rating.FORGOTTEN: 0}
    assert stats.reviews_total == 3
    assert stats.reviews_today == 2


def test_streak_counts_consecutive_days_and_survives_until_the_day_ends() -> None:
    # Reviews on each of the last three days, ending today: streak 3.
    three_day = compute_stats([_topic(1, (Rating.SOLID, 2), (Rating.SOLID, 1), (Rating.SOLID, 0))], now=NOW)
    assert three_day.streak_days == 3

    # Last review yesterday: still a streak of 1 -- today isn't over.
    yesterday = compute_stats([_topic(1, (Rating.SOLID, 1))], now=NOW)
    assert yesterday.streak_days == 1

    # A gap two days ago breaks it.
    broken = compute_stats([_topic(1, (Rating.SOLID, 4), (Rating.SOLID, 2))], now=NOW)
    assert broken.streak_days == 0


def test_format_stats_reads_naturally() -> None:
    topics = [
        _topic(1),
        _topic(2, (Rating.SOLID, 2), (Rating.SOLID, 1)),
        _topic(3, (Rating.SHAKY, 0)),
    ]

    text = format_stats(compute_stats(topics, now=NOW), course="Networking")

    assert text.splitlines()[0] == "Networking: 3 topic(s)"
    assert "due now: 0   new: 1   scheduled later: 2 (next due Thu 17 Sep)" in text
    assert "last rating: 1 solid, 1 shaky" in text
    assert "reviews: 3 total, 1 today; streak 3 days" in text


def test_format_stats_with_no_topics() -> None:
    text = format_stats(compute_stats([], now=NOW))

    assert text.startswith("All courses: 0 topic(s)")
    assert "Run `summarize`" in text
