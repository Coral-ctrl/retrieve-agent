"""The local web app: a dashboard, the review flow, ask, and adding new
material -- in a browser.

Everything here is a thin adapter over functions that already exist and are
already tested: stats.compute_stats, review.order_for_review,
schedule.next_interval_days, ask.retrieve_topics/answer_question,
summarize.summarize_file, sync.sync_digest, and the Repository. No study
logic lives in this file; a route reads the request, calls the same code
the CLI calls, and renders a template.

The New material page (POST /new) runs the same agent loop as
`retrieve summarize`, synchronously, in the request -- there's no background
job or polling. That's a deliberate choice for a single-user local app: a
file typically takes seconds to a couple of minutes, and a blocking request
with a "working..." state is simpler and easier to reason about than a job
queue. If uploads ever needed to handle much larger files, that's the first
thing to revisit.

Server-rendered HTML with a few lines of plain JavaScript (no framework,
no build step): the review page fetches the reveal and swaps it in, so the
title stays put while the answer appears beneath it -- all the
interactivity a recall-and-rate loop needs.
It's a single-user app on localhost, so there's no auth and no session
state -- every request goes to the database, the one source of truth.

Demo mode (RETRIEVE_DEMO_MODE=true) is for putting the app on the public
internet for interviewers to try. Rather than adding a login, it removes
what makes a public copy risky: create_app drops the Claude and Voyage
clients before any route can see them, so New material is switched off and
Ask shows the notes it found (free keyword search) without the paid answer
step. Dashboard and review work as normal against whatever DATABASE_URL
points at -- which, for a demo, should be a separate database holding
sample notes only.

The app is built by a factory that takes its dependencies, so tests can
pass a fake repository and a scripted Claude client, exactly as the CLI
tests do. The real wiring (Neon, Voyage, Anthropic) happens in cli.py's
`serve` command.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import tempfile

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from retrieve.ask import Answer, answer_question, retrieve_topics
from retrieve.extract import SUPPORTED_SUFFIXES, UnsupportedFileError
from retrieve.review import RATING_KEYS, describe_due, order_for_review
from retrieve.run_log import RunLogger
from retrieve.schedule import next_interval_days, schedule_for
from retrieve.schema import Rating
from retrieve.stats import compute_stats
from retrieve.summarize import SummarizeError, summarize_file
from retrieve.sync import format_sync_report, sync_digest

WEB_DIR = Path(__file__).resolve().parent


@dataclass
class WebDeps:
    """What the app needs from the outside world. cli.py builds the real
    one; tests build one with fakes."""

    open_repository: Callable[[], Any]  # returns a context manager yielding a Repository
    embedder: Any | None  # Voyage client, or None for keyword search
    messages_client: Any | None  # Backend.messages_client (ask), or None if Claude is unreachable
    agent_runner: Any | None  # Backend.agent_runner (new material), or None likewise
    model: str
    embedding_model: str
    logs_dir: Path
    knowledge_dir: Path
    max_turns: int
    dedup_threshold: float
    database_available: bool = True  # False mirrors "DATABASE_URL not set" in the CLI
    demo_mode: bool = False  # public demo: no paid calls, whatever clients are passed in
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


# Shown where the full app would call Claude. Kept here, not in a template,
# so the tests can check the exact wording the page promises.
DEMO_ASK_NOTE = (
    "This is the demo, so the answer step is switched off. In the full app, "
    "Claude now writes an answer using only the notes below, citing them as [1], [2]..."
)


def create_app(deps: WebDeps) -> FastAPI:
    if deps.demo_mode:
        # The guarantee demo mode makes ("no paid calls") is enforced here,
        # in one place, rather than by every route remembering to check:
        # with the clients gone, there is nothing a request could call.
        # The embedder goes too, so Ask uses Postgres keyword search.
        deps = replace(deps, messages_client=None, agent_runner=None, embedder=None)

    app = FastAPI(title="Retrieve")
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=WEB_DIR / "templates")

    def render(request: Request, name: str, status_code: int = 200, **context: Any) -> HTMLResponse:
        # demo_mode goes to every template so base.html can show the banner.
        return templates.TemplateResponse(
            request, name, {"request": request, "demo_mode": deps.demo_mode, **context},
            status_code=status_code,
        )

    # --- dashboard --------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, course: str | None = None):
        with deps.open_repository() as repo:
            topics = repo.list_topics(course=course or None)
        now = deps.now()
        stats = compute_stats(topics, now=now)
        courses = sorted({c for t in topics for c in t.courses})
        queue = order_for_review(topics, now=now)
        return render(
            request, "dashboard.html",
            stats=stats, course=course or "", courses=courses,
            ratings={r.value: n for r, n in stats.by_last_rating.items()},
            due_titles=[t.title for t in queue[:10]], due_total=len(queue),
            topics=sorted(topics, key=lambda t: t.title.lower()),
            due_text={t.id: describe_due(schedule_for(t.reviews), t, now) for t in topics},
        )

    # --- review -----------------------------------------------------------

    @app.get("/review", response_class=HTMLResponse)
    def review(request: Request, course: str | None = None, all: bool = False):
        """Show the next due topic, title only (the recall step)."""
        with deps.open_repository() as repo:
            topics = repo.list_topics(course=course or None)
        now = deps.now()
        queue = order_for_review(topics, now=now, include_upcoming=all)
        if not queue:
            return render(request, "review_done.html", total=len(topics),
                          course=course or "", all=all)
        topic = queue[0]
        return render(
            request, "review.html",
            topic=topic, remaining=len(queue), course=course or "", all=all,
            due=describe_due(schedule_for(topic.reviews), topic, now),
            preview={key: next_interval_days(topic.reviews, rating)
                     for key, rating in RATING_KEYS.items()},
        )

    @app.get("/review/{topic_id}/reveal", response_class=HTMLResponse)
    def reveal(request: Request, topic_id: int):
        """The reveal step: summary, key points and the rating buttons. Fetched
        by the review page's script and swapped in beneath the title."""
        with deps.open_repository() as repo:
            [topic] = repo.get_topics_by_ids([topic_id]) or [None]
        if topic is None:
            return HTMLResponse("Topic not found.", status_code=404)
        return render(
            request, "_reveal.html", topic=topic,
            preview={key: next_interval_days(topic.reviews, rating)
                     for key, rating in RATING_KEYS.items()},
        )

    @app.post("/review/{topic_id}/rate")
    def rate(request: Request, topic_id: int, rating: str = Form(...),
             course: str = Form(""), all: bool = Form(False)):
        """Record a rating, then go to the next due topic."""
        if rating not in Rating.__members__.values() and rating not in RATING_KEYS:
            return HTMLResponse("Unknown rating.", status_code=400)
        chosen = RATING_KEYS.get(rating) or Rating(rating)
        with deps.open_repository() as repo:
            [topic] = repo.get_topics_by_ids([topic_id]) or [None]
            if topic is None:
                return HTMLResponse("Topic not found.", status_code=404)
            days = next_interval_days(topic.reviews, chosen)
            repo.record_review(topic_id, chosen)
        next_url = f"/review?course={course}" + ("&all=true" if all else "")
        next_url += f"&just={topic_id}&rated={chosen.value}&days={days}"
        return RedirectResponse(next_url, status_code=303)

    # --- ask --------------------------------------------------------------

    @app.get("/ask", response_class=HTMLResponse)
    def ask_page(request: Request, course: str | None = None):
        # In the demo the form stays usable: searching the notes is free.
        return render(request, "ask.html", course=course or "", answer=None,
                      can_answer=deps.demo_mode or deps.messages_client is not None)

    @app.post("/ask", response_class=HTMLResponse)
    def ask_submit(request: Request, question: str = Form(...), course: str = Form("")):
        if deps.demo_mode:
            # Retrieval only: the same search the real Ask runs first (keyword,
            # since the embedder was removed above), then the matched notes
            # are shown in place of Claude's answer.
            with deps.open_repository() as repo:
                retrieved, used_vector = retrieve_topics(question, repo, None, course=course or None)
            text = DEMO_ASK_NOTE if retrieved else "No notes in the demo match that question."
            answer = Answer(question=question, text=text, retrieved=retrieved,
                            used_vector_search=used_vector)
            return render(request, "ask.html", course=course, question=question, answer=answer,
                          can_answer=True, cost=None)

        if deps.messages_client is None:
            return render(request, "ask.html", course=course, answer=None, can_answer=False,
                          question=question)
        with deps.open_repository() as repo:
            retrieved, used_vector = retrieve_topics(
                question, repo, deps.embedder, course=course or None,
                embedding_model=deps.embedding_model,
            )
        logger = RunLogger(deps.logs_dir, run_label="ask")
        try:
            answer = answer_question(question, retrieved, deps.messages_client, deps.model, logger,
                                     used_vector_search=used_vector)
        except Exception as model_error:  # noqa: BLE001 - API error, or subscription path with no backup
            return render(request, "ask.html", course=course, question=question, answer=None,
                          can_answer=True, error=f"Claude error: {model_error}")
        return render(request, "ask.html", course=course, question=question, answer=answer,
                      can_answer=True, cost=logger.estimated_cost_usd(deps.model))

    # --- new material -------------------------------------------------------

    @app.get("/new", response_class=HTMLResponse)
    def new_page(request: Request):
        return render(
            request, "new.html",
            can_summarize=deps.agent_runner is not None,
            suffixes=", ".join(sorted(SUPPORTED_SUFFIXES)),
        )

    @app.post("/new", response_class=HTMLResponse)
    def new_submit(request: Request, file: UploadFile, course: str = Form("")):
        if deps.demo_mode:
            # Refuse before reading the upload: in the demo, nothing a visitor
            # sends is processed or stored. 403 = understood, but not allowed here.
            return render(request, "new.html", status_code=403, can_summarize=False,
                          suffixes=", ".join(sorted(SUPPORTED_SUFFIXES)))
        if deps.agent_runner is None:
            return render(request, "new.html", can_summarize=False,
                          suffixes=", ".join(sorted(SUPPORTED_SUFFIXES)))

        original_name = file.filename or "upload"
        suffix = Path(original_name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(file.file.read())
            tmp.flush()

            try:
                result = summarize_file(
                    Path(tmp.name), original_name=original_name, course=course or None,
                    model=deps.model, max_turns=deps.max_turns,
                    agent_runner=deps.agent_runner, logs_dir=deps.logs_dir,
                    knowledge_dir=deps.knowledge_dir,
                )
            except UnsupportedFileError as error:
                return render(request, "new.html", can_summarize=True, error=str(error),
                              suffixes=", ".join(sorted(SUPPORTED_SUFFIXES)))
            except SummarizeError as error:
                return render(request, "new.html", can_summarize=True, error=str(error),
                              suffixes=", ".join(sorted(SUPPORTED_SUFFIXES)))
            except Exception as model_error:  # noqa: BLE001 - surface any Claude error plainly
                return render(request, "new.html", can_summarize=True,
                              error=f"Claude error: {model_error}",
                              suffixes=", ".join(sorted(SUPPORTED_SUFFIXES)))

        sync_report = None
        sync_error = None
        if deps.database_available:
            try:
                with deps.open_repository() as repo:
                    sync_results = sync_digest(
                        result.digest, repo, deps.embedder,
                        embedding_model=deps.embedding_model,
                        dedup_threshold=deps.dedup_threshold,
                    )
                sync_report = format_sync_report(sync_results)
            except Exception as db_error:  # noqa: BLE001 - the notes are already saved to disk
                sync_error = str(db_error)

        cost = result.run_summary["estimated_cost_usd"]
        return render(
            request, "new_result.html",
            digest=result.digest, output_folder=str(result.output_folder),
            finished=result.finished, sync_report=sync_report, sync_error=sync_error,
            database_available=deps.database_available, cost=cost,
        )

    return app
