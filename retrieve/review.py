"""The review session: recall a topic from its title, reveal it, rate it.

The order matters and is the whole point of the project's name. You see the
title *first* and try to explain the concept to yourself before the summary
is shown; only then do you rate how it went. Reading the answer first and
nodding along feels like studying but isn't retrieval practice.

Two things are kept separate here so each can be tested on its own:

- `order_for_review` is the queue policy: which topic comes up next. Phase 3
  uses a deliberately simple rule; Phase 4 replaces this one function with a
  policy that uses the rating history.
- `ReviewSession` is the interaction loop. It talks to the terminal through
  a tiny `Console` protocol, so tests script the keypresses the same way
  tests/fakes.py scripts the Claude client for the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from retrieve.schema import Rating, StoredTopic

# The ratings a user can enter, keyed by the single letter they type.
RATING_KEYS: dict[str, Rating] = {
    "s": Rating.SOLID,
    "h": Rating.SHAKY,
    "f": Rating.FORGOTTEN,
}

# Lower comes first. Never-reviewed topics are placed first (they're what
# you most need to see once), then the ones you've been forgetting, then
# shaky ones; solid ones come last. Within a group, the least recently
# reviewed topic comes first, so nothing sits unseen forever.
_RATING_PRIORITY = {None: 0, Rating.FORGOTTEN: 1, Rating.SHAKY: 2, Rating.SOLID: 3}


def order_for_review(topics: list[StoredTopic]) -> list[StoredTopic]:
    """Phase 3 queue policy. Simple on purpose; see the module docstring."""
    return sorted(
        topics,
        key=lambda topic: (
            _RATING_PRIORITY[topic.last_rating],
            topic.last_reviewed_at.timestamp() if topic.last_reviewed_at else 0.0,
            topic.id,
        ),
    )


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
    def __init__(self, topics: list[StoredTopic], store: ReviewStore, console: Console) -> None:
        self.topics = topics
        self.store = store
        self.console = console
        self.summary = ReviewSummary()

    def run(self) -> ReviewSummary:
        if not self.topics:
            self.console.write("Nothing to review yet. Run `summarize` on some material first.")
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
                self.store.record_review(topic.id, rating)
                self.summary.reviewed += 1
                self.summary.ratings[rating] += 1
                return True
            self.console.write("Please type s, h, f, k or q.")

    @staticmethod
    def _context_line(topic: StoredTopic) -> str:
        parts = [topic.difficulty.value]
        if topic.courses:
            parts.append(", ".join(topic.courses))
        if topic.sources:
            parts.append("from " + ", ".join(s.file_name for s in topic.sources))
        if topic.last_rating:
            parts.append(f"last time: {topic.last_rating.value}")
        else:
            parts.append("not reviewed yet")
        return " | ".join(parts)


class TerminalConsole:
    """The real terminal."""

    def write(self, text: str) -> None:
        print(text)

    def read(self, prompt: str) -> str:
        try:
            return input(prompt)
        except EOFError:  # e.g. piped stdin ran out
            return "q"
