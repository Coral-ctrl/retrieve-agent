"""The review session: recall a topic from its title, reveal it, rate it.

The order matters and is the whole point of the project's name. You see the
title *first* and try to explain the concept to yourself before the summary
is shown; only then do you rate how it went. Reading the answer first and
nodding along feels like studying but isn't retrieval practice.

Two things are kept separate here so each can be tested on its own:

- `order_for_review` is the queue policy: which topic comes up next. Phase 3
  ordered by last rating; Phase 4 orders by the spaced-repetition schedule
  in schedule.py, so only topics that are due (or new) come up.
- `ReviewSession` is the interaction loop. It talks to the terminal through
  a tiny `Console` protocol, so tests script the keypresses the same way
  tests/fakes.py scripts the Claude client for the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from retrieve.schedule import Schedule, next_interval_days, schedule_for
from retrieve.schema import Rating, StoredTopic

# The ratings a user can enter, keyed by the single letter they type.
RATING_KEYS: dict[str, Rating] = {
    "s": Rating.SOLID,
    "h": Rating.SHAKY,
    "f": Rating.FORGOTTEN,
}

def order_for_review(
    topics: list[StoredTopic],
    now: datetime | None = None,
    include_upcoming: bool = False,
) -> list[StoredTopic]:
    """Phase 4 queue policy: what's due, in the order it matters.

    1. Due and overdue topics, most overdue first. These are the ones whose
       memory is about to (or has started to) fade, so they're the most
       time-sensitive -- the same reason Anki shows reviews before new cards.
    2. New topics (never reviewed), oldest first.
    3. Only with `include_upcoming` (the --all flag): everything else, soonest
       due first. For cramming the night before an interview, when the
       schedule matters less than seeing it all once.
    """
    now = now or datetime.now(timezone.utc)
    scheduled = [(topic, schedule_for(topic.reviews)) for topic in topics]

    due = sorted(
        (t for t, s in scheduled if not s.is_new and s.is_due(now)),
        key=lambda t: (schedule_for(t.reviews).due_at, t.id),
    )
    new = sorted((t for t, s in scheduled if s.is_new), key=lambda t: t.id)
    queue = due + new

    if include_upcoming:
        upcoming = sorted(
            (t for t, s in scheduled if not s.is_new and not s.is_due(now)),
            key=lambda t: (schedule_for(t.reviews).due_at, t.id),
        )
        queue += upcoming
    return queue


class Console(Protocol):
    """What the session needs from a terminal: print, and ask for a line."""

    def write(self, text: str) -> None: ...
    def read(self, prompt: str) -> str: ...


class ReviewStore(Protocol):
    """What the session needs from the database."""

    def record_review(self, topic_id: int, rating: Rating) -> None: ...


@dataclass
class ReviewSummary:
    reviewed: int = 0
    skipped: int = 0
    ratings: dict[Rating, int] = field(default_factory=lambda: {r: 0 for r in Rating})

    def describe(self) -> str:
        if self.reviewed == 0:
            return "No topics rated."
        parts = [f"{count} {rating.value}" for rating, count in self.ratings.items() if count]
        skipped = f", {self.skipped} skipped" if self.skipped else ""
        return f"Rated {self.reviewed} topic(s): {', '.join(parts)}{skipped}."


class ReviewSession:
    def __init__(
        self,
        topics: list[StoredTopic],
        store: ReviewStore,
        console: Console,
        now: datetime | None = None,
        total_topics: int | None = None,
    ) -> None:
        self.topics = topics
        self.store = store
        self.console = console
        self.now = now or datetime.now(timezone.utc)
        # How many topics exist in total, so an empty queue can say "nothing
        # due" rather than "nothing at all".
        self.total_topics = len(topics) if total_topics is None else total_topics
        self.summary = ReviewSummary()

    def run(self) -> ReviewSummary:
        if not self.topics:
            if self.total_topics == 0:
                self.console.write("Nothing to review yet. Run `summarize` on some material first.")
            else:
                self.console.write(
                    f"Nothing due right now ({self.total_topics} topic(s) scheduled for later). "
                    "Use --all to review ahead of schedule."
                )
            return self.summary

        self.console.write(
            f"{len(self.topics)} topic(s) to review. For each: try to explain it from the "
            "title, press Enter to reveal, then rate it.\n"
            "  s = solid   h = shaky   f = forgotten   k = skip   q = quit\n"
        )
        for position, topic in enumerate(self.topics, start=1):
            if not self._review_one(topic, position):
                break

        self.console.write(f"\n{self.summary.describe()}")
        return self.summary

    def _review_one(self, topic: StoredTopic, position: int) -> bool:
        """Show one topic and record the answer. Returns False to stop."""
        self.console.write(f"\n[{position}/{len(self.topics)}] {topic.title}")
        self.console.write(f"    {self._context_line(topic)}")

        answer = self.console.read("Explain it to yourself, then press Enter to reveal (q to quit): ")
        if answer.strip().lower() == "q":
            return False

        self.console.write(f"\n{topic.summary}")
        for point in topic.key_points:
            self.console.write(f"  - {point}")

        while True:
            key = self.console.read("\nHow did you do? [s/h/f, k=skip, q=quit]: ").strip().lower()
            if key == "q":
                return False
            if key == "k":
                self.summary.skipped += 1
                return True
            rating = RATING_KEYS.get(key)
            if rating is not None:
                days = next_interval_days(topic.reviews, rating)
                self.store.record_review(topic.id, rating)
                self.summary.reviewed += 1
                self.summary.ratings[rating] += 1
                self.console.write(f"-> {rating.value}. Back in {_days_text(days)}.")
                return True
            self.console.write("Please type s, h, f, k or q.")

    def _context_line(self, topic: StoredTopic) -> str:
        parts = [topic.difficulty.value]
        if topic.courses:
            parts.append(", ".join(topic.courses))
        if topic.sources:
            parts.append("from " + ", ".join(s.file_name for s in topic.sources))
        parts.append(describe_due(schedule_for(topic.reviews), topic, self.now))
        return " | ".join(parts)


def describe_due(schedule: Schedule, topic: StoredTopic, now: datetime) -> str:
    """'new', 'due today (last: shaky)', 'overdue by 3 days (last: solid)',
    'due in 5 days (last: solid)'."""
    if schedule.is_new:
        return "new"
    days = schedule.days_until_due(now)
    last = f"last: {topic.last_rating.value}"
    if days < 0:
        return f"overdue by {_days_text(-days)} ({last})"
    if days == 0:
        return f"due today ({last})"
    return f"due in {_days_text(days)} ({last})"


def _days_text(days: int) -> str:
    return "1 day" if days == 1 else f"{days} days"


class TerminalConsole:
    """The real terminal."""

    def write(self, text: str) -> None:
        print(text)

    def read(self, prompt: str) -> str:
        try:
            return input(prompt)
        except EOFError:  # e.g. piped stdin ran out
            return "q"
