"""A one-screen summary of where your knowledge stands.

Pure functions over the topics already loaded from the database, so this is
free to run and easy to test: `compute_stats` turns topics into numbers,
`format_stats` turns numbers into text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from retrieve.schedule import schedule_for
from retrieve.schema import Rating, StoredTopic


@dataclass
class Stats:
    total: int = 0
    due_now: int = 0  # reviewed before and due (or overdue)
    new: int = 0  # never reviewed
    upcoming: int = 0  # reviewed, not due yet
    next_due: date | None = None  # soonest upcoming due date, if nothing is due now
    by_last_rating: dict[Rating, int] = field(default_factory=lambda: {r: 0 for r in Rating})
    reviews_total: int = 0
    reviews_today: int = 0
    streak_days: int = 0  # consecutive days, ending today or yesterday, with a review


def compute_stats(topics: list[StoredTopic], now: datetime | None = None) -> Stats:
    now = now or datetime.now(timezone.utc)
    today = now.date()
    stats = Stats(total=len(topics))
    review_days: set[date] = set()

    for topic in topics:
        schedule = schedule_for(topic.reviews)
        if schedule.is_new:
            stats.new += 1
        elif schedule.is_due(now):
            stats.due_now += 1
        else:
            stats.upcoming += 1
            due_date = schedule.due_at.date()
            if stats.next_due is None or due_date < stats.next_due:
                stats.next_due = due_date

        if topic.last_rating is not None:
            stats.by_last_rating[topic.last_rating] += 1

        for event in topic.reviews:
            stats.reviews_total += 1
            event_day = event.reviewed_at.astimezone(timezone.utc).date()
            review_days.add(event_day)
            if event_day == today:
                stats.reviews_today += 1

    stats.streak_days = _streak(review_days, today)
    return stats


def _streak(review_days: set[date], today: date) -> int:
    """Consecutive days with at least one review, counting back from today.
    A streak that ended yesterday still counts (you haven't broken it until
    today is over without a review)."""
    if today in review_days:
        day = today
    elif today - timedelta(days=1) in review_days:
        day = today - timedelta(days=1)
    else:
        return 0
    streak = 0
    while day in review_days:
        streak += 1
        day -= timedelta(days=1)
    return streak


def format_stats(stats: Stats, course: str | None = None) -> str:
    scope = course or "All courses"
    lines = [f"{scope}: {stats.total} topic(s)"]
    if stats.total == 0:
        lines.append("  Nothing here yet. Run `summarize` on some material first.")
        return "\n".join(lines)

    schedule_line = f"  due now: {stats.due_now}   new: {stats.new}   scheduled later: {stats.upcoming}"
    if stats.due_now == 0 and stats.next_due is not None:
        schedule_line += f" (next due {stats.next_due.strftime('%a %d %b')})"
    lines.append(schedule_line)

    rated = {r: n for r, n in stats.by_last_rating.items() if n}
    if rated:
        lines.append("  last rating: " + ", ".join(f"{n} {r.value}" for r, n in rated.items()))

    streak = f"{stats.streak_days} day" if stats.streak_days == 1 else f"{stats.streak_days} days"
    lines.append(f"  reviews: {stats.reviews_total} total, {stats.reviews_today} today; streak {streak}")
    return "\n".join(lines)
