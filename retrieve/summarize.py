"""Turn one source file into study topics and save them to disk -- the
orchestration shared by `retrieve summarize` (CLI) and the web app's "New
material" page.

Reading the file and talking to Claude happen first; only once topics come
back is anything written. Database sync is a separate, optional step the
caller does afterwards (sync.sync_digest), since a CLI run and a web request
open the database connection differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from retrieve.agent import MessagesClient, run_agent
from retrieve.extract import extract_pages
from retrieve.run_log import RunLogger
from retrieve.schema import StudyDigest
from retrieve.storage import save_digest, slugify
from retrieve.tools import DocumentWorkspace

DEFAULT_MAX_TURNS = 20


class SummarizeError(ValueError):
    """A file was read fine but nothing usable came out of it: no
    extractable text, or the agent finished without saving any topics."""


@dataclass
class SummarizeResult:
    digest: StudyDigest
    output_folder: Path
    run_summary: dict  # RunLogger.summary(model)
    finished: bool  # False if the agent hit --max-turns before finishing
    turns_used: int
    log_path: Path


def summarize_file(
    file_path: Path,
    *,
    original_name: str,
    course: str | None,
    model: str,
    max_turns: int,
    messages_client: MessagesClient,
    logs_dir: Path,
    knowledge_dir: Path,
) -> SummarizeResult:
    """Read `file_path`, distill it with Claude, and save the digest under
    `knowledge_dir`. Raises FileNotFoundError or UnsupportedFileError (from
    extract_pages) for problems reading the file, SummarizeError if there's
    nothing to work with, and whatever the Claude client raises on API
    errors -- callers decide how to present each of those."""
    pages = extract_pages(file_path)
    readable_pages = [page for page in pages if not page.is_empty]
    if not readable_pages:
        raise SummarizeError(
            "No text could be extracted. If this is a scanned PDF, it needs OCR first."
        )

    logger = RunLogger(logs_dir, run_label=slugify(Path(original_name).stem))
    workspace = DocumentWorkspace(pages=pages)
    result = run_agent(
        workspace=workspace,
        messages_client=messages_client,
        model=model,
        max_turns=max_turns,
        logger=logger,
        source_name=original_name,
        course=course,
    )
    if not result.topics:
        raise SummarizeError(
            f"The agent didn't save any topics. Check the run log for details: {logger.log_path}"
        )

    digest = StudyDigest(
        source_file=original_name,
        course=course,
        model=model,
        topics=result.topics,
        agent_note=result.finish_note,
    )
    output_folder = save_digest(digest, knowledge_dir)
    return SummarizeResult(
        digest=digest,
        output_folder=output_folder,
        run_summary=logger.summary(model),
        finished=result.finished,
        turns_used=result.turns_used,
        log_path=logger.log_path,
    )
