"""When should a topic come back? Spaced repetition, SM-2 style.

The idea behind spaced repetition: each time you successfully recall
something, the memory gets more durable, so the next review can wait longer.
Fail to recall it and the interval collapses back to short. Reviewing on
that schedule -- just before you'd forget -- is far more efficient than
re-reading everything on a fixed timetable.

This is an adaptation of the SM-2 algorithm (the one Anki grew out of) to
this project's three ratings:

    solid      recalled it fully   -> interval grows: 1d, 3d, then x ease
    shaky      got the gist        -> repeat the same interval, ease drops a little
    forgotten  couldn't recall it  -> back to 1 day, ease drops more

`ease` is a per-topic multiplier (starting 2.5, never below 1.3) that
captures how easy a topic has been for you historically: consistently solid
topics spread out faster, ones you keep fumbling spread out slower.

Design choice: the schedule is REPLAYED from the review history each time,
not stored. The reviews table stays an append-only log of what actually
happened, there are no extra columns to keep in sync, and changing these
rules later re-schedules every topic consistently. At a few hundred topics
the replay is instantaneous.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from retrieve.schema import Rating, ReviewEvent

INITIAL_EASE = 2.5
MIN_EASE = 1.3
FIRST_INTERVAL_DAYS = 1
SECOND_INTERVAL_DAYS = 3

# How much each rating moves the ease. SM-2's own constants, roughly.
EASE_CHANGE = {
    Rating.SOLID: +0.10,
    Rating.SHAKY: -0.15,
    Rating.FORGOTTEN: -0.20,
}


@dataclass(frozen=True)
class Schedule:
    """A topic's position in the spacing schedule."""

    interval_days: int  # gap that was set at the last review (0 = never reviewed)
    ease: float
    streak: int  # consecutive solid ratings since the last forgotten
    due_at: datetime | None  # None means never reviewed: due now

    @property
    def is_new(self) -> bool:
        return self.due_at is None

    def is_due(self, now: datetime) -> bool:
        return self.due_at is None or self.due_at <= now

    def days_until_due(self, now: datetime) -> int:
        """Negative when overdue, 0 when due today, positive when upcoming."""
        if self.due_at is None:
            return 0
        return (self.due_at.date() - now.date()).days


def schedule_for(reviews: list[ReviewEvent]) -> Schedule:
    """Replay a topic's rating history, oldest first, to find its schedule."""
    interval = 0
    ease = INITIAL_EASE
    streak = 0
    due_at: datetime | None = None

    for event in reviews:
        interval, ease, streak = _apply(event.rating, interval, ease, streak)
        due_at = event.reviewed_at + timedelta(days=interval)

    return Schedule(interval_days=interval, ease=round(ease, 2), streak=streak, due_at=due_at)


def next_interval_days(reviews: list[ReviewEvent], rating: Rating) -> int:
    """If the topic were rated `rating` right now, how many days until it
    comes back? Used to tell the reviewer what their rating just did."""
    current = schedule_for(reviews)
    interval, _ease, _streak = _apply(rating, current.interval_days, current.ease, current.streak)
    return interval


def _apply(rating: Rating, interval: int, ease: float, streak: int) -> tuple[int, float, int]:
    """One step of the algorithm. Returns the new (interval, ease, streak)."""
    ease = max(MIN_EASE, ease + EASE_CHANGE[rating])

    if rating is Rating.FORGOTTEN:
        return FIRST_INTERVAL_DAYS, ease, 0

    if rating is Rating.SHAKY:
        # Not confident enough to spread out further; see it again after the
        # same gap. A first-ever shaky review still needs some gap.
        return max(FIRST_INTERVAL_DAYS, interval), ease, streak

    # solid
    streak += 1
    if streak == 1:
        new_interval = FIRST_INTERVAL_DAYS
    elif streak == 2:
        new_interval = SECOND_INTERVAL_DAYS
    else:
        new_interval = max(interval + 1, round(interval * ease))
    return new_interval, ease, streak
