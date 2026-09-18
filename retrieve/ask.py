"""Ask a question and get an answer grounded in your own notes (RAG).

Retrieval-augmented generation in its simplest honest form:

    1. Turn the question into a vector and find the topics whose vectors are
       closest -- the same pgvector index de-duplication uses, queried from
       the other side (Voyage's 'query' input type, paired with the
       'document' type the notes were stored with).
    2. Hand those topics to Claude with one firm rule: answer from these
       notes only, cite them, and say so if they don't cover the question.

Step 2 is a single model call, not an agent loop: there's nothing to
iterate on. The value is in what goes in (your notes, not the open
internet) and in the model being told it may say "your notes don't cover
this" -- which is exactly what you want to hear before an interview, rather
than a fluent answer to a question you never actually studied.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from retrieve.embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingClient, embed_query
from retrieve.run_log import RunLogger
from retrieve.schema import StoredTopic

DEFAULT_TOP_K = 5
MAX_ANSWER_TOKENS = 1024

SYSTEM_PROMPT = """\
You are Retrieve, a study assistant. The user is preparing for job interviews \
and asks questions about material they've studied. You are given excerpts \
from THEIR OWN notes, numbered [1], [2], ...

Rules:
- Answer only from the notes provided. Do not add facts from elsewhere, even \
if you know them. The point is to tell the user what THEIR notes say.
- Cite the notes you draw on inline, like [1] or [2][3].
- If the notes don't contain the answer, say so plainly in one sentence \
("Your notes don't cover X.") and, if some note is partly relevant, say what \
it does cover. Don't guess.
- Be concise: this is revision, not an essay. Use the notes' exact terms, \
since the user needs to recall precise wording under pressure."""


class MessagesClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class SearchStore(Protocol):
    """What `ask` needs from the database."""

    def find_similar_topics(self, embedding: list[float], limit: int,
                            course: str | None = None) -> list[tuple[StoredTopic, float]]: ...

    def search_topics_by_text(self, query: str, limit: int,
                              course: str | None = None) -> list[StoredTopic]: ...


@dataclass
class RetrievedTopic:
    topic: StoredTopic
    distance: float | None  # None when found by text search rather than by vector


@dataclass
class Answer:
    question: str
    text: str
    retrieved: list[RetrievedTopic]
    used_vector_search: bool
    input_tokens: int = 0
    output_tokens: int = 0
    unused_notes: list[str] = field(default_factory=list)


def retrieve_topics(
    question: str,
    store: SearchStore,
    embedder: EmbeddingClient | None,
    top_k: int = DEFAULT_TOP_K,
    course: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[list[RetrievedTopic], bool]:
    """Find the notes most likely to answer the question.

    With an embedder: nearest neighbours by meaning. Without one: Postgres
    full-text search on title/summary/key points, which catches questions
    that share vocabulary with the note but not ones phrased differently.
    Returns (topics, used_vector_search)."""
    if embedder is not None:
        query_vector = embed_query(question, embedder, model=embedding_model)
        matches = store.find_similar_topics(query_vector, limit=top_k, course=course)
        return [RetrievedTopic(topic, distance) for topic, distance in matches], True

    topics = store.search_topics_by_text(question, limit=top_k, course=course)
    return [RetrievedTopic(topic, None) for topic in topics], False


def format_notes_for_prompt(retrieved: list[RetrievedTopic]) -> str:
    blocks = []
    for index, item in enumerate(retrieved, start=1):
        topic = item.topic
        sources = "; ".join(
            f"{s.file_name} p.{','.join(map(str, s.pages))}" for s in topic.sources
        ) or "unknown source"
        blocks.append(
            f"[{index}] {topic.title}  ({sources})\n"
            f"{topic.summary}\n"
            + "\n".join(f"- {point}" for point in topic.key_points)
        )
    return "\n\n".join(blocks)


def answer_question(
    question: str,
    retrieved: list[RetrievedTopic],
    messages_client: MessagesClient,
    model: str,
    logger: RunLogger,
    used_vector_search: bool = True,
) -> Answer:
    """One grounded model call. Returns the answer plus what it was built from."""
    if not retrieved:
        return Answer(question=question, text="No notes found to answer from.",
                      retrieved=[], used_vector_search=used_vector_search)

    user_message = (
        f"Notes:\n\n{format_notes_for_prompt(retrieved)}\n\n"
        f"Question: {question}"
    )
    logger.log("ask_start", question=question, notes=[r.topic.title for r in retrieved],
               model=model, vector_search=used_vector_search)

    started = time.perf_counter()
    response = messages_client.create(
        model=model,
        max_tokens=MAX_ANSWER_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    logger.record_model_call(
        turn=1,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        stop_reason=response.stop_reason,
        duration_ms=(time.perf_counter() - started) * 1000,
    )

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    cited = {index for index in range(1, len(retrieved) + 1) if f"[{index}]" in text}
    unused = [r.topic.title for i, r in enumerate(retrieved, start=1) if i not in cited]

    logger.log("ask_end", cited=sorted(cited), **logger.summary(model))
    return Answer(
        question=question,
        text=text,
        retrieved=retrieved,
        used_vector_search=used_vector_search,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        unused_notes=unused,
    )


def format_answer(answer: Answer, cost_usd: float | None) -> str:
    lines = []
    if answer.retrieved:
        how = "by meaning" if answer.used_vector_search else "by keyword (no VOYAGE_API_KEY)"
        lines.append(f"Found {len(answer.retrieved)} relevant note(s) {how}:")
        for index, item in enumerate(answer.retrieved, start=1):
            distance = f"  distance {item.distance:.2f}" if item.distance is not None else ""
            lines.append(f"  [{index}] {item.topic.title}{distance}")
        lines.append("")

    lines.append(answer.text)

    if answer.retrieved:
        lines.append("")
        lines.append("Sources:")
        for index, item in enumerate(answer.retrieved, start=1):
            for source in item.topic.sources:
                pages = ", ".join(str(p) for p in source.pages)
                lines.append(f"  [{index}] {source.file_name} p.{pages}")
        cost = f"~US${cost_usd:.4f}" if cost_usd is not None else "unknown cost"
        lines.append(f"\n{answer.input_tokens} input / {answer.output_tokens} output tokens, {cost}")
    return "\n".join(lines)
