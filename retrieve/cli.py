"""Command-line entry point.

Usage:
    python -m retrieve summarize path/to/slides.pdf --course "Networking"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retrieve.agent import run_agent
from retrieve.config import Settings, load_settings
from retrieve.extract import SUPPORTED_SUFFIXES, UnsupportedFileError, extract_pages
from retrieve.run_log import RunLogger
from retrieve.schema import StudyDigest
from retrieve.storage import save_digest, slugify
from retrieve.tools import DocumentWorkspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrieve",
        description="Turn course material into reviewable study topics.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    summarize = subcommands.add_parser("summarize", help="Distill one file into topics.")
    summarize.add_argument("file", type=Path,
                           help=f"Source file ({', '.join(sorted(SUPPORTED_SUFFIXES))}).")
    summarize.add_argument("--course", help="Course or module name. Given to the agent as context and saved with each note.")
    summarize.add_argument("--model", help="Override the model set in .env.")
    summarize.add_argument("--max-turns", type=int, help="Override the turn limit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()

    if args.command == "summarize":
        return summarize_command(args, settings)
    return 1


def summarize_command(args: argparse.Namespace, settings: Settings) -> int:
    model = args.model or settings.model
    max_turns = args.max_turns or settings.max_turns

    # --- Read the file before spending any tokens ------------------------
    try:
        pages = extract_pages(args.file)
    except FileNotFoundError:
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1
    except UnsupportedFileError as error:
        print(error, file=sys.stderr)
        return 1

    readable_pages = [page for page in pages if not page.is_empty]
    if not readable_pages:
        print("No text could be extracted. If this is a scanned PDF, it needs OCR first.",
              file=sys.stderr)
        return 1

    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.",
              file=sys.stderr)
        return 1

    # Imported here so --help and the checks above work without the SDK configured.
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    logger = RunLogger(settings.logs_dir, run_label=slugify(args.file.stem))
    workspace = DocumentWorkspace(pages=pages)

    print(f"Reading {args.file.name} ({len(pages)} pages, "
          f"{len(pages) - len(readable_pages)} without text) with {model}...")

    try:
        result = run_agent(
            workspace=workspace,
            messages_client=client.messages,
            model=model,
            max_turns=max_turns,
            logger=logger,
            source_name=args.file.name,
            course=args.course,
        )
    except anthropic.APIError as api_error:
        print(f"Claude API error: {api_error}", file=sys.stderr)
        print(f"Run log: {logger.log_path}", file=sys.stderr)
        return 1

    if not result.topics:
        print("The agent didn't save any topics. Check the run log for details:",
              logger.log_path, file=sys.stderr)
        return 1

    digest = StudyDigest(
        source_file=args.file.name,
        course=args.course,
        model=model,
        topics=result.topics,
        agent_note=result.finish_note,
    )
    output_folder = save_digest(digest, settings.knowledge_dir)
    print_digest(digest)

    run_summary = logger.summary(model)
    cost = run_summary["estimated_cost_usd"]
    cost_text = f"~US${cost:.3f}" if cost is not None else "unknown cost"
    topic_count = len(result.topics)
    topic_word = "topic" if topic_count == 1 else "topics"
    print(f"\n{topic_count} {topic_word} saved to {output_folder}")
    print(f"{result.turns_used} turns, {run_summary['tool_calls']} tool calls "
          f"({run_summary['tool_errors']} corrected), {cost_text}")
    print(f"Run log: {logger.log_path}")

    if not result.finished:
        print(f"\nWarning: stopped after {max_turns} turns before the agent finished. "
              "Some pages may be missing; try --max-turns with a higher number.")
        return 2
    return 0


def print_digest(digest: StudyDigest) -> None:
    for index, topic in enumerate(digest.topics, start=1):
        pages_text = ", ".join(str(n) for n in topic.source_pages)
        print(f"\n{index}. {topic.title}  [{topic.difficulty.value}] (p. {pages_text})")
        print(f"   {topic.summary}")
        for point in topic.key_points:
            print(f"   - {point}")
        print(f"   tags: {', '.join(topic.tags)}")
    if digest.agent_note:
        print(f"\nAgent note: {digest.agent_note}")


if __name__ == "__main__":
    raise SystemExit(main())
