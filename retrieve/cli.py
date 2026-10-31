"""Command-line entry point.

Usage:
    python -m retrieve summarize path/to/slides.pdf --course "Networking"
    python -m retrieve init-db
    python -m retrieve import-knowledge      # backfill existing digests into the DB
    python -m retrieve review --course "Networking"   # what's due today
    python -m retrieve review --all                    # everything, e.g. before an interview
    python -m retrieve stats
    python -m retrieve sources                            # list synced files, for remove-source
    python -m retrieve remove-source "week-1.pdf"          # clean slate before re-summarizing
    python -m retrieve ask "how does slow start work?"   # answered from your notes
    python -m retrieve serve                             # the same, in your browser
    python -m retrieve setup-token                       # sign in with a Claude subscription
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retrieve.backends import make_backend
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

    subcommands.add_parser(
        "sources",
        help="List every source file that's been synced, with its exact file_name "
             "(for remove-source) and how many topics it's linked to.",
    )

    remove_source = subcommands.add_parser(
        "remove-source",
        help="Delete a source and any topics that exist only because of it -- e.g. "
             "to re-run summarize on the same file from a clean slate. Topics also "
             "linked to another source are kept, just unlinked from this one.",
    )
    remove_source.add_argument("file_name", help="Exact file_name as shown by `sources`.")
    remove_source.add_argument("--course", help="Only if the source was synced with a course.")

    ask = subcommands.add_parser("ask", help="Ask a question; the answer comes from your notes only.")
    ask.add_argument("question", help="A plain-English question, in quotes.")
    ask.add_argument("--course", help="Only search notes from this course.")
    ask.add_argument("--top-k", type=int, default=5, help="How many notes to give the model (default 5).")
    ask.add_argument("--model", help="Override the model set in .env.")

    subcommands.add_parser(
        "setup-token",
        help="Sign in with your Claude subscription: prints a token to put in .env as CLAUDE_CODE_OAUTH_TOKEN.",
    )

    serve = subcommands.add_parser("serve", help="Open the dashboard, review and ask pages in your browser.")
    serve.add_argument("--port", type=int, default=8000)
    # 127.0.0.1 = only this computer can open it. A server or container
    # needs 0.0.0.0 so requests from outside reach the app.
    serve.add_argument("--host", default="127.0.0.1",
                       help="Address to listen on. Use 0.0.0.0 when deploying (e.g. in Docker).")
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
    if args.command == "sources":
        return sources_command(settings)
    if args.command == "remove-source":
        return remove_source_command(args, settings)
    if args.command == "ask":
        return ask_command(args, settings)
    if args.command == "serve":
        return serve_command(args, settings)
    if args.command == "setup-token":
        return setup_token_command()
    return 1


def setup_token_command() -> int:
    """Run Claude Code's `setup-token` (from the binary bundled with the
    SDK, so nothing else needs installing). It opens a browser to sign in
    with your Claude account and prints a long-lived token."""
    import subprocess

    from retrieve.sdk_agent import bundled_cli_path

    cli = bundled_cli_path()
    if cli is None:
        print("Neither claude-agent-sdk nor a `claude` command is installed. "
              "Run: pip install claude-agent-sdk", file=sys.stderr)
        return 1
    print("Signing in with your Claude subscription. When the token is printed, "
          "copy it into .env as CLAUDE_CODE_OAUTH_TOKEN=...\n")
    return subprocess.call([cli, "setup-token"])


def _notice(message: str) -> None:
    """Backend notices (e.g. falling back to the API key) go to stderr so
    they never mix into piped output."""
    print(message, file=sys.stderr)


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
    """A Voyage client if a key is configured and Voyage isn't disabled,
    else None (title-only dedup, keyword search instead of vector search --
    the same graceful path as having no key at all)."""
    if settings.disable_voyage or not settings.voyage_api_key:
        return None
    import voyageai

    # The SDK retries rate-limit and transient errors with exponential
    # backoff, but only if asked: the default is max_retries=0. An account
    # with no payment method is capped at 3 requests/minute, which
    # import-knowledge over several files would otherwise trip over.
    return voyageai.Client(api_key=settings.voyage_api_key, max_retries=6)


def _no_embedder_reason(settings: Settings) -> str:
    """Why `_make_embedder` returned None, for the status lines below."""
    if settings.disable_voyage:
        return "RETRIEVE_DISABLE_VOYAGE is set"
    return "No VOYAGE_API_KEY"


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
        print(f"({_no_embedder_reason(settings)}: topics are stored without embeddings "
              "and matched by title only.)\n")

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


def sources_command(settings: Settings) -> int:
    conn = _open_database(settings)
    if conn is None:
        return 1

    from retrieve.db import Repository

    with conn:
        sources = Repository(conn).list_sources()

    if not sources:
        print("No sources synced yet.")
        return 0

    for source in sources:
        course = f"  [{source.course}]" if source.course else ""
        print(f"{source.file_name}{course}  ({source.topic_count} topic(s))")
    return 0


def remove_source_command(args: argparse.Namespace, settings: Settings) -> int:
    conn = _open_database(settings)
    if conn is None:
        return 1

    from retrieve.db import Repository

    with conn:
        repository = Repository(conn)
        with repository.transaction():
            result = repository.delete_source(args.file_name, args.course)

    if result is None:
        print(f"No source matches file_name={args.file_name!r} course={args.course!r}. "
              "Run `python -m retrieve sources` to see exact names.", file=sys.stderr)
        return 1

    print(f"Removed source {args.file_name!r} (id {result.source_id}).")
    if result.topics_deleted:
        print(f"Deleted {len(result.topics_deleted)} topic(s) that only came from this source:")
        for title in result.topics_deleted:
            print(f"  - {title}")
    if result.topics_unlinked:
        print(f"Kept {result.topics_unlinked} topic(s) also linked to another source "
              "(just unlinked from this one).")
    print("\nThe knowledge/ folder on disk isn't touched by this command -- "
          "delete that source's subfolder yourself if you want a fully clean slate.")
    return 0


def ask_command(args: argparse.Namespace, settings: Settings) -> int:
    backend = make_backend(settings, notify=_notice)
    if backend is None:
        return 1
    conn = _open_database(settings)
    if conn is None:
        return 1

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

    try:
        answer = answer_question(args.question, retrieved, backend.messages_client, model, logger,
                                 used_vector_search=used_vector)
    except Exception as model_error:  # noqa: BLE001 - API error, or subscription path with no backup
        print(f"Claude error: {model_error}", file=sys.stderr)
        return 1

    print(format_answer(answer, logger.estimated_cost_usd(model)))
    if backend.name == "subscription":
        print("(cost shown at API rates; covered by your Claude subscription)")
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

    if settings.demo_mode:
        # Don't even build a backend: a demo server never reads the Claude
        # token or API key, so it doesn't matter what's in its environment.
        backend = None
        embedder = None
        print("Demo mode: no Claude or Voyage calls. Uploads are off; Ask shows matching notes only.")
    else:
        backend = make_backend(settings, notify=_notice)
        embedder = _make_embedder(settings)
        if backend is None:
            print("(The Ask and New material pages will be disabled.)")
        else:
            print(f"Claude via {backend.describe}")

    app = create_app(WebDeps(
        open_repository=open_repository,
        embedder=embedder,
        messages_client=backend.messages_client if backend else None,
        agent_runner=backend.agent_runner if backend else None,
        model=settings.model,
        embedding_model=settings.embedding_model,
        logs_dir=settings.logs_dir,
        knowledge_dir=settings.knowledge_dir,
        max_turns=settings.max_turns,
        dedup_threshold=settings.dedup_threshold,
        demo_mode=settings.demo_mode,
    ))

    # 0.0.0.0 isn't an address a browser can open, so show localhost instead.
    shown_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{shown_host}:{args.port}"
    print(f"Retrieve is running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def summarize_command(args: argparse.Namespace, settings: Settings) -> int:
    model = args.model or settings.model
    max_turns = args.max_turns or settings.max_turns

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1

    backend = make_backend(settings, notify=_notice)
    if backend is None:
        return 1

    print(f"Reading {args.file.name} with {model} via {backend.describe}...")

    try:
        result = summarize_file(
            args.file,
            original_name=args.file.name,
            course=args.course,
            model=model,
            max_turns=max_turns,
            agent_runner=backend.agent_runner,
            logs_dir=settings.logs_dir,
            knowledge_dir=settings.knowledge_dir,
        )
    except UnsupportedFileError as error:
        print(error, file=sys.stderr)
        return 1
    except SummarizeError as error:
        print(error, file=sys.stderr)
        return 1
    except Exception as model_error:  # noqa: BLE001 - API error, or subscription path with no backup
        print(f"Claude error: {model_error}", file=sys.stderr)
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
    if backend.name == "subscription":
        cost_text += " at API rates, covered by your subscription"
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
        print(f"\n({_no_embedder_reason(settings)}: topics stored without embeddings, "
              "de-duplication by title only.)")

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
