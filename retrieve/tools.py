"""The tools the agent can call, and the workspace they operate on.

The model decides *which* tool to call and *when*. This module decides what
each tool actually does. Keeping them separate is what makes this an agent
rather than a fixed script: the model plans its own reading order, decides
where one topic ends and the next begins, and fixes its own mistakes when a
tool returns an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from retrieve.extract import Page
from retrieve.schema import Difficulty, Topic

# Reading a whole 200-page PDF in one call wastes context. Capping the page
# range nudges the agent to read in chunks and summarise as it goes.
MAX_PAGES_PER_READ = 8
OUTLINE_PREVIEW_CHARS = 90

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_document_outline",
        "description": (
            "Get the page count and the first line of every page. "
            "Call this first to plan how to read the document."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_pages",
        "description": (
            f"Read the full text of a page range (inclusive, 1-based). "
            f"At most {MAX_PAGES_PER_READ} pages per call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_page": {"type": "integer", "minimum": 1},
                "end_page": {"type": "integer", "minimum": 1},
            },
            "required": ["start_page", "end_page"],
        },
    },
    {
        "name": "save_topic",
        "description": (
            "Save one distilled topic. One concept per topic. Write the summary "
            "so a learner could explain the idea in an interview from it alone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short, specific concept name."},
                "summary": {"type": "string", "description": "2-4 plain-English sentences."},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "1-7 facts, steps or terms worth remembering exactly.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "1-6 lowercase tags, e.g. module name and concept area.",
                },
                "difficulty": {"type": "string", "enum": [level.value for level in Difficulty]},
                "source_pages": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Pages this topic came from.",
                },
            },
            "required": ["title", "summary", "key_points", "tags", "difficulty", "source_pages"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Call once every page has been read and every topic saved. "
            "Include a short note on anything unclear or missing in the source."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": ["note"],
        },
    },
]


class ToolError(Exception):
    """A problem the agent caused and can fix, e.g. a bad page range."""


@dataclass
class DocumentWorkspace:
    """Holds one document's pages and the topics saved from it so far."""

    pages: list[Page]
    saved_topics: list[Topic] = field(default_factory=list)
    pages_read: set[int] = field(default_factory=set)
    finished: bool = False
    finish_note: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def run_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Dispatch a tool call. Raises ToolError for mistakes the agent should correct."""
        handlers = {
            "get_document_outline": self._get_document_outline,
            "read_pages": self._read_pages,
            "save_topic": self._save_topic,
            "finish": self._finish,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            raise ToolError(f"Unknown tool '{tool_name}'.")
        return handler(**tool_input)

    # --- individual tools -------------------------------------------------

    def _get_document_outline(self) -> str:
        outline_lines = [f"Document has {self.page_count} page(s)."]
        for page in self.pages:
            if page.is_empty:
                preview = "(no extractable text - possibly an image or scanned slide)"
            else:
                preview = page.text.splitlines()[0][:OUTLINE_PREVIEW_CHARS]
            outline_lines.append(f"p{page.number}: {preview}")
        return "\n".join(outline_lines)

    def _read_pages(self, start_page: int, end_page: int) -> str:
        if start_page > end_page:
            raise ToolError("start_page must be less than or equal to end_page.")
        if start_page < 1 or end_page > self.page_count:
            raise ToolError(f"Page range must be within 1-{self.page_count}.")
        if end_page - start_page + 1 > MAX_PAGES_PER_READ:
            raise ToolError(f"Read at most {MAX_PAGES_PER_READ} pages per call.")

        page_blocks = []
        for page in self.pages[start_page - 1 : end_page]:
            self.pages_read.add(page.number)
            body = page.text if not page.is_empty else "(no extractable text)"
            page_blocks.append(f"--- Page {page.number} ---\n{body}")
        return "\n\n".join(page_blocks)

    def _save_topic(self, **topic_fields: Any) -> str:
        try:
            topic = Topic(**topic_fields)
        except ValidationError as validation_error:
            # Hand the exact problems back so the model can retry correctly.
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in validation_error.errors()
            )
            raise ToolError(f"Topic not saved. Fix these fields: {problems}") from None

        invalid_pages = [n for n in topic.source_pages if n > self.page_count]
        if invalid_pages:
            raise ToolError(f"source_pages {invalid_pages} don't exist in this document.")

        duplicate = next(
            (t for t in self.saved_topics if t.title.lower() == topic.title.lower()), None
        )
        if duplicate:
            raise ToolError(
                f"A topic titled '{duplicate.title}' already exists. "
                "Use a more specific title or fold this into that topic."
            )

        self.saved_topics.append(topic)
        return f"Saved topic #{len(self.saved_topics)}: {topic.title}"

    def _finish(self, note: str) -> str:
        unread_pages = [
            page.number
            for page in self.pages
            if page.number not in self.pages_read and not page.is_empty
        ]
        if unread_pages:
            raise ToolError(f"Pages {unread_pages} haven't been read yet. Read them first.")
        if not self.saved_topics:
            raise ToolError("No topics saved yet. Save at least one topic before finishing.")

        self.finished = True
        self.finish_note = note
        return "Finished."
