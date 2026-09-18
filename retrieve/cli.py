"""Command-line entry point.

Usage:
    python -m retrieve summarize path/to/slides.pdf --course "Networking"
    python -m retrieve init-db
    python -m retrieve import-knowledge      # backfill existing digests into the DB
    python -m retrieve review --course "Networking"   # what's due today
    python -m retrieve review --all                    # everything, e.g. before an interview
    python -m retrieve stats
    python -m retrieve ask "how does slow start work?"   # answered from your notes
    python -m retrieve serve                             # the same, in your browser
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retrieve.config import Settings, load_settings
from retrieve.extract import SUPPORTED_SUFFIXES, UnsupportedFileError
from retrieve.run_log import RunLogger
from retrieve.schema import StudyDigest
from retrieve.summarize import SummarizeError, summarize_file


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

    import_knowledge = subcommands.add_parser(
        "import-knowledge",
        help="Load every digest.json already in the knowledge folder into the database.",
    )
    import_knowledge.add_argument(
        "--force", action="store_true",
        help="Re-sync digests whose source is already in the database (e.g. after a "
             "failed sync). Safe: existing topics are merged into, not duplicated.",
    )

    review = subcommands.add_parser("review", help="Recall, reveal and rate the topics that are due.")
    review.add_argument("--course", help="Only review topics from this course.")
    review.add_argument("--limit", type=int, default=20,
                        help="Maximum topics in this session (default 20).")
    review.add_argument("--all", action="store_true",
                        help="Include topics not due yet, soonest first (cramming mode).")

    stats = subcommands.add_parser("stats", help="What's due, what's new, and how reviewing is going.")
    stats.add_argument("--course", help="Only count topics from this course.")

    ask = subcommands.add_parser("ask", help="Ask a question; the answer comes from your notes only.")
    ask.add_argument("question", help="A plain-English question, in quotes.")
    ask.add_argument("--course", help="Only search notes from this course.")
    ask.add_argument("--top-k", type=int, default=5, help="How many notes to give the model (default 5).")
    ask.add_argument("--model", help="Override the model set in .env.")

    serve = subcommands.add_parser("serve", help="Open the dashboard, review and ask pages in your browser.")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-browser", action="store_true", help="Don't open a browser tab automatically.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()

    if args.command == "summarize":
        return summarize_command(args, settings)
    if args.command == "init-db":
        return init_db_command(settings)
    if args.command == "import-knowledge":
        return import_knowledge_command(settings, force=args.force)
    if args.command == "review":
        return review_command(args, settings)
    if args.command == "stats":
        return stats_command(args, settings)
    if args.command == "ask":
        return ask_command(args, settings)
    if args.command == "serve":
        return serve_command(args, settings)
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

    # The SDK retries rate-limit and transient errors with exponential
    # backoff, but only if asked: the default is max_retries=0. An account
    # with no payment method is capped at 3 requests/minute, which
    # import-knowledge over several files would otherwise trip over.
    return voyageai.Client(api_key=settings.voyage_api_key, max_retries=6)


def init_db_command(settings: Settings) -> int:
    conn = _open_database(settings)
    if conn is None:
        return 1

    from retrieve.db import init_schema

    with conn:
        init_schema(conn)
    print("Database ready: sources, topics, topic_sources and reviews tables are in place.")
    return 0


def import_knowledge_command(settings: Settings, force: bool = False) -> int:
    """Backfill: sync digests that were written before the database existed.

    Also the cheap way to repair a source that didn't fully sync: with
    --force it re-syncs from the digest.json on disk, costing one embedding
    request and no Claude call, instead of re-running summarize."""
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
            if not force and repository.source_exists(digest.source_file, digest.course):
                print(f"- {digest.source_file}: already imported, skipped (use --force to re-sync)")
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
        topics = repository.list_topics(course=args.course)
        queue = order_for_review(topics, include_upcoming=args.all)[: args.limit]
        ReviewSession(queue, repository, TerminalConsole(), total_topics=len(topics)).run()
    return 0


def stats_command(args: argparse.Namespace, settings: Settings) -> int:
    conn = _open_database(settings)
    if conn is None:
        return 1

    from retrieve.db import Repository
    from retrieve.stats import compute_stats, format_stats

    with conn:
        topics = Repository(conn).list_topics(course=args.course)
    print(format_stats(compute_stats(topics), course=args.course))
    return 0


def ask_command(args: argparse.Namespace, settings: Settings) -> int:
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.",
              file=sys.stderr)
        return 1
    conn = _open_database(settings)
    if conn is None:
        return 1

    import anthropic

    from retrieve.ask import answer_question, format_answer, retrieve_topics
    from retrieve.db import Repository

    model = args.model or settings.model
    embedder = _make_embedder(settings)
    logger = RunLogger(settings.logs_dir, run_label="ask")

    with conn:
        repository = Repository(conn)
        try:
            retrieved, used_vector = retrieve_topics(
                args.question, repository, embedder,
                top_k=args.top_k, course=args.course,
                embedding_model=settings.embedding_model,
            )
        except Exception as search_error:  # noqa: BLE001 - e.g. Voyage rate limit
            print(f"Could not search your notes: {search_error}", file=sys.stderr)
            return 1

    if not retrieved:
        scope = f" in course '{args.course}'" if args.course else ""
        print(f"No notes{scope} match that question. "
              "Try different wording, or `summarize` the material first.")
        return 0

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        answer = answer_question(args.question, retrieved, client.messages, model, logger,
                                 used_vector_search=used_vector)
    except anthropic.APIError as api_error:
        print(f"Claude API error: {api_error}", file=sys.stderr)
        return 1

    print(format_answer(answer, logger.estimated_cost_usd(model)))
    return 0


def serve_command(args: argparse.Namespace, settings: Settings) -> int:
    """Run the local web app. Every request opens its own short-lived
    connection, so a Neon database that idles between uses just wakes up."""
    if not settings.database_url:
        print("DATABASE_URL is not set. Add it to .env first (see .env.example).", file=sys.stderr)
        return 1

    import contextlib
    import threading
    import webbrowser

    import uvicorn

    from retrieve.db import Repository, connect
    from retrieve.web.app import WebDeps, create_app

    @contextlib.contextmanager
    def open_repository():
        conn = connect(settings.database_url)
        try:
            yield Repository(conn)
        finally:
            conn.close()

    messages_client = None
    if settings.anthropic_api_key:
        import anthropic
        messages_client = anthropic.Anthropic(api_key=settings.anthropic_api_key).messages
    else:
        print("(ANTHROPIC_API_KEY not set: the Ask page will be read-only.)")

    app = create_app(WebDeps(
        open_repository=open_repository,
        embedder=_make_embedder(settings),
        messages_client=messages_client,
        model=settings.model,
        embedding_model=settings.embedding_model,
        logs_dir=settings.logs_dir,
        knowledge_dir=settings.knowledge_dir,
        max_turns=settings.max_turns,
        dedup_threshold=settings.dedup_threshold,
    ))

    url = f"http://127.0.0.1:{args.port}"
    print(f"Retrieve is running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def summarize_command(args: argparse.Namespace, settings: Settings) -> int:
    model = args.model or settings.model
    max_turns = args.max_turns or settings.max_turns

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1

    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.",
              file=sys.stderr)
        return 1

    # Imported here so --help and the checks above work without the SDK configured.
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    print(f"Reading {args.file.name} with {model}...")

    try:
        result = summarize_file(
            args.file,
            original_name=args.file.name,
            course=args.course,
            model=model,
            max_turns=max_turns,
            messages_client=client.messages,
            logs_dir=settings.logs_dir,
            knowledge_dir=settings.knowledge_dir,
        )
    except UnsupportedFileError as error:
        print(error, file=sys.stderr)
        return 1
    except SummarizeError as error:
        print(error, file=sys.stderr)
        return 1
    except anthropic.APIError as api_error:
        print(f"Claude API error: {api_error}", file=sys.stderr)
        return 1

    digest = result.digest
    print_digest(digest)

    if settings.database_url and not args.no_db:
        sync_to_database(digest, settings)
    elif not settings.database_url:
        print("\n(DATABASE_URL not set - topics saved to files only. "
              "See .env.example to enable database sync.)")

    cost = result.run_summary["estimated_cost_usd"]
    cost_text = f"~US${cost:.3f}" if cost is not None else "unknown cost"
    topic_count = len(digest.topics)
    topic_word = "topic" if topic_count == 1 else "topics"
    print(f"\n{topic_count} {topic_word} saved to {result.output_folder}")
    print(f"{result.turns_used} turns, {result.run_summary['tool_calls']} tool calls "
          f"({result.run_summary['tool_errors']} corrected), {cost_text}")
    print(f"Run log: {result.log_path}")

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
