"""Command-line entry point.

Usage:
    python -m retrieve summarize path/to/slides.pdf --course "Networking"
    python -m retrieve init-db
    python -m retrieve import-knowledge      # backfill existing digests into the DB
    python -m retrieve review --course "Networking"
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
    summarize.add_argument("--no-db", action="store_true",
                           help="Skip syncing to the database even if DATABASE_URL is set.")

    subcommands.add_parser(
        "init-db",
        help="Create or update the database schema in DATABASE_URL.",
    )

    subcommands.add_parser(
        "import-knowledge",
        help="Load every digest.json already in the knowledge folder into the database.",
    )

    review = subcommands.add_parser("review", help="Recall, reveal and rate topics from the database.")
    review.add_argument("--course", help="Only review topics from this course.")
    review.add_argument("--limit", type=int, default=20,
                        help="Maximum topics in this session (default 20).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()

    if args.command == "summarize":
        return summarize_command(args, settings)
    if args.command == "init-db":
        return init_db_command(settings)
    if args.command == "import-knowledge":
        return import_knowledge_command(settings)
    if args.command == "review":
        return review_command(args, settings)
    return 1


def _open_database(settings: Settings):
    """Connect, or explain why not. Returns None when the caller should exit 1.
    Shared by every command that needs the database."""
    if not settings.database_url:
        print("DATABASE_URL is not set. Add it to .env first (see .env.example).", file=sys.stderr)
        return None

    from retrieve.db import connect

    try:
        return connect(settings.database_url)
    except Exception as connection_error:  # noqa: BLE001 - report any driver error plainly
        print(f"Could not connect to the database: {connection_error}", file=sys.stderr)
        return None


def _make_embedder(settings: Settings):
    """A Voyage client if a key is configured, else None (title-only dedup)."""
    if not settings.voyage_api_key:
        return None
    import voyageai

    return voyageai.Client(api_key=settings.voyage_api_key)


def init_db_command(settings: Settings) -> int:
    conn = _open_database(settings)
    if conn is None:
        return 1

    from retrieve.db import init_schema

    with conn:
        init_schema(conn)
    print("Database ready: sources, topics, topic_sources and reviews tables are in place.")
    return 0


def import_knowledge_command(settings: Settings) -> int:
    """Backfill: sync digests that were written before the database existed."""
    digest_paths = sorted(settings.knowledge_dir.glob("*/digest.json"))
    if not digest_paths:
        print(f"No digest.json files found under {settings.knowledge_dir}.", file=sys.stderr)
        return 1

    conn = _open_database(settings)
    if conn is None:
        return 1

    from retrieve.db import Repository
    from retrieve.sync import format_sync_report, sync_digest

    embedder = _make_embedder(settings)
    if embedder is None:
        print("(No VOYAGE_API_KEY: topics are stored without embeddings and matched by title only.)\n")

    imported = skipped = 0
    with conn:
        repository = Repository(conn)
        for digest_path in digest_paths:
            digest = StudyDigest.model_validate_json(digest_path.read_text(encoding="utf-8"))
            if repository.source_exists(digest.source_file, digest.course):
                print(f"- {digest.source_file}: already imported, skipped")
                skipped += 1
                continue
            results = sync_digest(
                digest, repository, embedder,
                embedding_model=settings.embedding_model,
                dedup_threshold=settings.dedup_threshold,
            )
            print(f"- {digest.source_file}: {format_sync_report(results)}")
            imported += 1

    print(f"\nImported {imported} digest(s), skipped {skipped} already in the database.")
    return 0


def review_command(args: argparse.Namespace, settings: Settings) -> int:
    conn = _open_database(settings)
    if conn is None:
        return 1

    from retrieve.db import Repository
    from retrieve.review import ReviewSession, TerminalConsole, order_for_review

    with conn:
        repository = Repository(conn)
        queue = order_for_review(repository.list_topics(course=args.course))[: args.limit]
        ReviewSession(queue, repository, TerminalConsole()).run()
    return 0


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

    if settings.database_url and not args.no_db:
        sync_to_database(digest, settings)
    elif not settings.database_url:
        print("\n(DATABASE_URL not set - topics saved to files only. "
              "See .env.example to enable database sync.)")

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


def sync_to_database(digest: StudyDigest, settings: Settings) -> None:
    """Embed and de-duplicate this digest's topics into DATABASE_URL.

    Best-effort by design: the notes are already safely on disk by the time
    this runs, so a database problem is reported and the run still exits
    0 rather than losing the agent's work over a sync step.
    """
    import psycopg

    from retrieve.db import Repository, connect
    from retrieve.sync import format_sync_report, sync_digest

    try:
        conn = connect(settings.database_url)
    except Exception as connection_error:  # noqa: BLE001 - report any driver error plainly
        print(f"\nCould not sync to the database: {connection_error}", file=sys.stderr)
        return

    embedder = _make_embedder(settings)
    if embedder is None:
        print("\n(No VOYAGE_API_KEY: topics stored without embeddings, de-duplication by title only.)")

    try:
        with conn:
            repository = Repository(conn)
            results = sync_digest(
                digest, repository, embedder,
                embedding_model=settings.embedding_model,
                dedup_threshold=settings.dedup_threshold,
            )
        print(f"\nSynced to database: {format_sync_report(results)}")
    except psycopg.errors.UndefinedTable:
        print("\nDatabase sync failed: schema not found. Run `python -m retrieve init-db` first.",
              file=sys.stderr)
    except Exception as sync_error:  # noqa: BLE001 - keep the run's exit code 0 on sync failure
        print(f"\nDatabase sync failed: {sync_error}", file=sys.stderr)
    finally:
        conn.close()


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
