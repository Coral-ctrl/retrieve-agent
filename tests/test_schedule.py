"""The spaced-repetition rules, checked step by step.

Each test replays a specific rating history and asserts the resulting
interval, ease and due date, so the algorithm's behaviour is pinned down
in plain examples rather than left to be inferred from the code.
"""

from datetime import datetime, timedelta, timezone

from retrieve.schedule import (
    INITIAL_EASE,
    MIN_EASE,
    next_interval_days,
    schedule_for,
)
from retrieve.schema import Rating, ReviewEvent

DAY_0 = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)


def _history(*ratings: Rating, gap_days: int = 1) -> list[ReviewEvent]:
    """A review on day 0, then one every `gap_days`, with the given ratings."""
    return [
        ReviewEvent(rating=rating, reviewed_at=DAY_0 + timedelta(days=i * gap_days))
        for i, rating in enumerate(ratings)
    ]


def test_a_never_reviewed_topic_is_new_and_due_now() -> None:
    schedule = schedule_for([])

    assert schedule.is_new
    assert schedule.is_due(DAY_0)
    assert schedule.interval_days == 0
    assert schedule.ease == INITIAL_EASE


def test_solid_ratings_grow_the_interval_1_3_then_times_ease() -> None:
    # SM-2's shape: two fixed short steps, then geometric growth by ease.
    assert schedule_for(_history(Rating.SOLID)).interval_days == 1
    assert schedule_for(_history(Rating.SOLID, Rating.SOLID)).interval_days == 3
    third = schedule_for(_history(Rating.SOLID, Rating.SOLID, Rating.SOLID))
    assert third.interval_days == round(3 * third.ease)  # ease has risen to 2.8 -> 8 days
    assert third.interval_days == 8
    fourth = schedule_for(_history(*[Rating.SOLID] * 4))
    assert fourth.interval_days == round(8 * fourth.ease)  # 2.9 -> 23 days


def test_solid_raises_ease_and_the_due_date_follows_the_last_review() -> None:
    history = _history(Rating.SOLID, Rating.SOLID)
    schedule = schedule_for(history)

    assert schedule.ease == INITIAL_EASE + 0.2
    assert schedule.streak == 2
    assert schedule.due_at == history[-1].reviewed_at + timedelta(days=3)


def test_forgotten_resets_to_one_day_and_lowers_ease() -> None:
    schedule = schedule_for(_history(Rating.SOLID, Rating.SOLID, Rating.SOLID, Rating.FORGOTTEN))

    assert schedule.interval_days == 1
    assert schedule.streak == 0
    assert schedule.ease == round(INITIAL_EASE + 0.3 - 0.2, 2)


def test_after_forgetting_the_climb_starts_again_from_the_short_steps() -> None:
    schedule = schedule_for(_history(Rating.SOLID, Rating.SOLID, Rating.SOLID,
                                     Rating.FORGOTTEN, Rating.SOLID, Rating.SOLID))

    assert schedule.interval_days == 3  # 1 then 3 again, not straight back to 8


def test_shaky_repeats_the_current_interval_and_lowers_ease_a_little() -> None:
    before = schedule_for(_history(Rating.SOLID, Rating.SOLID))
    after = schedule_for(_history(Rating.SOLID, Rating.SOLID, Rating.SHAKY))

    assert after.interval_days == before.interval_days == 3
    assert after.ease == round(before.ease - 0.15, 2)
    assert after.streak == before.streak  # shaky neither extends nor breaks the streak


def test_a_first_review_rated_shaky_still_gets_a_one_day_gap() -> None:
    assert schedule_for(_history(Rating.SHAKY)).interval_days == 1


def test_ease_never_drops_below_the_floor() -> None:
    schedule = schedule_for(_history(*[Rating.FORGOTTEN] * 20))

    assert schedule.ease == MIN_EASE


def test_interval_always_grows_on_solid_even_at_minimum_ease() -> None:
    # With ease at the floor (1.3), round(1 * 1.3) would be 1 again -- the
    # max(interval + 1, ...) guard stops a topic getting stuck at 1 day.
    history = _history(*[Rating.FORGOTTEN] * 10, Rating.SOLID, Rating.SOLID, Rating.SOLID)
    schedule = schedule_for(history)

    assert schedule.interval_days > 3


def test_is_due_and_days_until_due() -> None:
    history = _history(Rating.SOLID, Rating.SOLID)  # last review DAY_0 + 1, due 3 days later
    schedule = schedule_for(history)
    due_day = DAY_0 + timedelta(days=4)

    assert not schedule.is_due(due_day - timedelta(days=1))
    assert schedule.days_until_due(due_day - timedelta(days=1)) == 1
    assert schedule.is_due(due_day)
    assert schedule.days_until_due(due_day) == 0
    assert schedule.is_due(due_day + timedelta(days=2))
    assert schedule.days_until_due(due_day + timedelta(days=2)) == -2


def test_next_interval_days_previews_a_rating_without_recording_it() -> None:
    history = _history(Rating.SOLID, Rating.SOLID)

    assert next_interval_days(history, Rating.SOLID) == 8
    assert next_interval_days(history, Rating.SHAKY) == 3
    assert next_interval_days(history, Rating.FORGOTTEN) == 1
    assert next_interval_days([], Rating.SOLID) == 1  # a brand-new topic's first solid
