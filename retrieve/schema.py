"""The shape of the knowledge the agent produces.

Every topic is validated against these models before it's saved, so a
malformed tool call from the model becomes a clear error the agent can fix
instead of bad data in your notes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Difficulty(str, Enum):
    INTRO = "intro"  # definitions, vocabulary, "what is X"
    CORE = "core"  # the main ideas you'd be expected to explain
    ADVANCED = "advanced"  # edge cases, trade-offs, deeper mechanics


class Topic(BaseModel):
    """One concept = one note, so each topic becomes one review card later."""

    title: str = Field(min_length=3, max_length=80)
    summary: str = Field(min_length=20, max_length=800)
    key_points: list[str] = Field(min_length=1, max_length=7)
    tags: list[str] = Field(min_length=1, max_length=6)
    difficulty: Difficulty
    source_pages: list[int] = Field(min_length=1)

    @field_validator("tags")
    @classmethod
    def normalise_tags(cls, tags: list[str]) -> list[str]:
        # "TCP Handshake" and "tcp-handshake" should be the same tag.
        cleaned = [tag.strip().lower().replace(" ", "-") for tag in tags if tag.strip()]
        return list(dict.fromkeys(cleaned))  # de-duplicate, keep order

    @field_validator("source_pages")
    @classmethod
    def pages_are_positive(cls, pages: list[int]) -> list[int]:
        if any(page_number < 1 for page_number in pages):
            raise ValueError("page numbers start at 1")
        return sorted(set(pages))


class Rating(str, Enum):
    """How well you recalled a topic when it came up for review."""

    SOLID = "solid"  # explained it fully without looking
    SHAKY = "shaky"  # got the gist, missed details or order
    FORGOTTEN = "forgotten"  # couldn't recall it


class StudyDigest(BaseModel):
    """Everything the agent distilled from one source file."""

    source_file: str
    course: str | None = None
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    topics: list[Topic]
    agent_note: str | None = None  # the agent's own closing remark, e.g. gaps it noticed


class TopicSource(BaseModel):
    """Where a stored topic was seen: one source file and the pages in it."""

    file_name: str
    course: str | None
    pages: list[int]


class StoredTopic(BaseModel):
    """A topic as read back from the database, with its review history
    summarised. This is what the review screen shows and what the queue is
    ordered by."""

    id: int
    title: str
    summary: str
    key_points: list[str]
    tags: list[str]
    difficulty: Difficulty
    sources: list[TopicSource]
    review_count: int = 0
    last_rating: Rating | None = None
    last_reviewed_at: datetime | None = None

    @property
    def courses(self) -> list[str]:
        return list(dict.fromkeys(s.course for s in self.sources if s.course))
