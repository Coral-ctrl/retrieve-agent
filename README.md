# Retrieve

An AI agent that reads your course slides, PDFs and notes, distills them into
one-concept-per-note study topics, and (in later phases) quizzes you on the
ones you're forgetting.

> **Status:** Phase 5 in progress. Distill course material from the command
> line, store topics in Postgres, review them on a spaced-repetition schedule
> driven by your own ratings, and ask questions answered from your notes
> alone. Practice questions are next on the [roadmap](#roadmap).

## Why

Knowing something and being able to explain it precisely in an interview are
two different skills. After months of courses, the knowledge is there, but the
exact wording and the order of steps fade. Retrieve turns course material into
short, structured notes you can review quickly. The goal is an agent that
tracks what you remember and brings back what you don't.

## How it works

```mermaid
flowchart LR
    A[PDF / PPTX / Markdown] --> B[extract.py<br/>text per page]
    B --> C{Agent loop<br/>agent.py}
    C -- get_document_outline --> D[tools.py<br/>DocumentWorkspace]
    C -- read_pages --> D
    C -- save_topic --> D
    C -- finish --> D
    D -- results / errors --> C
    D --> E[storage.py<br/>digest.json + one .md per topic]
    D -. if DATABASE_URL set .-> H[sync.py<br/>new or duplicate?]
    H -. with a Voyage key .-> G[embeddings.py]
    G --> H
    H --> I[(Postgres + pgvector<br/>topics, sources, reviews)]
    I --> K[schedule.py<br/>what's due?]
    K --> J[review.py<br/>recall, reveal, rate]
    J -- rating --> I
    I -- nearest notes --> L[ask.py<br/>grounded answer]
    L -- one Claude call --> L
    C -. every call .-> F[run_log.py<br/>JSONL run log]
```

The agent loop is written by hand against the Claude API, with no agent
framework, so every step is visible:

1. Claude receives the task and four tool definitions.
2. It decides what to do: look at the outline, read a page range, save a topic, or finish.
3. The code runs the tool and sends the result back. **Mistakes come back as
   errors the model can fix**: a bad page range, a malformed topic, a
   duplicate title, or trying to finish before every page has been read.
4. The loop ends when `finish` succeeds or the turn limit is reached.

Every topic is validated with Pydantic before it's saved, and every model call
and tool call is logged with token counts and an estimated cost.

## Quick start

Requires Python 3.10+ and an [Anthropic API key](https://platform.claude.com).

```bash
git clone https://github.com/<your-username>/retrieve.git
cd retrieve

# Using uv (recommended)
uv venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
uv pip install -r requirements-dev.txt

# Or using plain pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env               # then paste your API key into .env
```

## Usage

```bash
# Try the included example notes
python -m retrieve summarize examples/tcp-basics.md --course "Networking"

# Your own material (keep it in materials/, which git ignores)
python -m retrieve summarize materials/week-3-slides.pdf --course "Networking"

# Use a cheaper model for simple slides
python -m retrieve summarize materials/week-3-slides.pdf --model claude-haiku-4-5
```

The command prints each topic, then a run summary with turns, tool calls,
corrected errors and estimated cost. Output is written to:

```
knowledge/week-3-slides/
├── digest.json                  # all topics, for code to load later
├── tcp-three-way-handshake.md   # one note per concept
└── ...
logs/20260916-021500-week-3-slides.jsonl
```

Each note has YAML frontmatter (source, pages, difficulty, tags, course), so
the `knowledge/` folder also opens directly as an Obsidian vault.

## The database (Phases 2 to 4)

Set `DATABASE_URL` in `.env` (a free [Neon](https://neon.tech) Postgres
project works; so does a local Postgres with the pgvector extension), then
create the schema once:

```bash
python -m retrieve init-db
```

From then on, every `summarize` run also stores its topics in the database,
and any digests you produced *before* setting this up can be backfilled:

```bash
python -m retrieve import-knowledge
```

### Reviewing (Phases 3 and 4)

```bash
python -m retrieve review                      # what's due today, 20 topics max
python -m retrieve review --course Networking --limit 10
python -m retrieve review --all                # everything, soonest due first (cramming)
python -m retrieve stats                       # what's due, what's new, streak
```

Each topic shows its **title first**. You try to explain the concept to
yourself, then press Enter to reveal the summary and key points, then rate
how it went: `s` solid, `h` shaky, `f` forgotten (`k` skips, `q` quits).

```
[1/2] TCP three-way handshake
    core | Networking | from week-1.pdf | overdue by 2 days (last: shaky)
Explain it to yourself, then press Enter to reveal (q to quit):

TCP opens a connection with SYN, SYN-ACK, ACK before any data is sent.
  - SYN
  - SYN-ACK
  - ACK

How did you do? [s/h/f, k=skip, q=quit]: s
-> solid. Back in 3 days.
```

**When a topic comes back is decided by spaced repetition** (`schedule.py`,
an adaptation of the SM-2 algorithm behind Anki): each time you rate a
topic *solid* the gap before its next review grows -- 1 day, 3 days, then
multiplied by a per-topic *ease* factor -- so well-known material fades
into the background. *Shaky* repeats the current gap and lowers ease a
little; *forgotten* drops the gap back to 1 day and lowers ease more. The
`review` queue shows what's due (most overdue first), then new topics --
the same order Anki uses, because a fading memory is more urgent than a
fresh one.

The schedule isn't stored anywhere. It's replayed from the append-only
`reviews` log each time, so there are no extra columns to keep in sync and
changing the rules later re-schedules everything consistently.

`stats` shows where things stand:

```
Networking: 24 topic(s)
  due now: 4   new: 2   scheduled later: 18
  last rating: 15 solid, 5 shaky, 2 forgotten
  reviews: 61 total, 4 today; streak 6 days
```

### Asking questions (Phase 5)

```bash
python -m retrieve ask "how does slow start work?"
python -m retrieve ask "what's the difference between flow and congestion control?" --course Networking
```

Retrieval-augmented generation over your own notes, in its simplest honest
form. The question is embedded (Voyage's `query` input type, the partner of
the `document` type the notes were stored with) and the nearest topics are
pulled from the same pgvector index de-duplication uses. Those notes go to
Claude with one firm rule: **answer only from these notes, cite them, and
say plainly if they don't cover the question.**

```
Found 2 relevant note(s) by meaning:
  [1] Flow control vs congestion control  distance 0.31
  [2] TCP reliability mechanism  distance 0.47

Slow start is part of congestion control: TCP starts slowly and backs off
when it detects packet loss [1]. Your notes don't go into how the window
grows during slow start.

Sources:
  [1] week-3-slides.pdf p.5
  [2] week-3-slides.pdf p.4

812 input / 61 output tokens, ~US$0.0014
```

That last sentence is the point. A general assistant would happily explain
slow start's window doubling from its own knowledge; `ask` tells you your
notes are thin there, which is what you need to know before an interview.
Each question costs one Claude call (a cent or so) and, with a Voyage key,
one embedding request. Without a Voyage key it falls back to Postgres
full-text search, which finds notes that share words with the question but
not ones phrased differently.

### De-duplication across sources (Phase 2, needs a Voyage key)

Add `VOYAGE_API_KEY` (from [Voyage AI](https://dashboard.voyageai.com),
Anthropic's recommended embeddings partner -- Claude itself doesn't serve
embeddings) and each topic is also embedded and checked against everything
already saved, by meaning rather than by title text. The default model,
`voyage-4-lite`, comes with a one-time allowance of 200M free tokens per
account. A topic is a few hundred tokens, so that is on the order of a
million topics -- this project won't get near it, so in practice embeddings
cost nothing:

```
Synced to database: 3 new, 1 merged with existing topics
  - 'The TCP handshake' merged into 'TCP three-way handshake' (distance 0.041, now in 2 source(s))
```

A near-duplicate isn't inserted as a second row. Instead the existing topic
gains a link to the new source (and its page numbers) and its tags are
unioned with the new ones, so the same concept taught across different
weeks' slides converges on one topic instead of splitting into near-copies.
"Near" is a cosine-distance threshold (`RETRIEVE_DEDUP_THRESHOLD`, default
`0.10`) found by checking real reworded duplicates against genuinely
different topics from the same course; lower it if your material tends to
already be phrased close together.

Without a Voyage key, topics are stored with no vector and matching falls
back to exact title, which is enough to stop a re-run of the same file from
doubling everything but won't catch the same idea worded differently. You
can add the key later; only topics stored after that will have embeddings.

A Voyage account with no payment method is limited to 3 requests a minute.
That's fine here: a whole file's topics are embedded in **one** request, and
the client retries with backoff if `import-knowledge` over many files does
hit the limit. Adding a card lifts the limit and doesn't change the price
(the free tokens still apply). Syncing is all-or-nothing per file -- if it
fails for any reason, nothing is written. To repair or retry, use
`python -m retrieve import-knowledge --force`: it re-syncs from the
`digest.json` already on disk, so it costs one embedding request and no
Claude call, and merges straight back into any rows that did get saved.

File output from Phase 1 is unaffected -- `knowledge/` is written the same
way whether or not `DATABASE_URL` is set. Pass `--no-db` to skip syncing
for one run without touching `.env`.

## Project structure

```
retrieve/
├── agent.py       # the agent loop and system prompt
├── tools.py       # tool definitions and what each tool does
├── schema.py      # Pydantic models for topics and digests
├── extract.py     # PDF / PPTX / Markdown -> text per page
├── storage.py     # writes digest.json and Markdown notes
├── run_log.py     # JSONL logging, token and cost tracking
├── embeddings.py  # topic -> vector, via Voyage AI
├── db.py          # Repository: Postgres/pgvector reads and writes
├── sync.py        # the de-duplication decision (new vs. merge)
├── schedule.py    # spaced repetition: when is each topic due?
├── review.py      # the due queue + the recall/reveal/rate session
├── stats.py       # what's due, what's new, streak
├── ask.py         # RAG: nearest notes -> one grounded Claude call
├── config.py      # settings from .env
└── cli.py         # command-line interface
migrations/        # SQL schema, applied by `python -m retrieve init-db`
tests/             # 100 tests; all but the Postgres integration tests run
                    # fully offline, without an API key
examples/          # sample notes to try the agent on
```

## Testing

```bash
pytest
```

The tests never call a real paid API. A fake client replays scripted
responses to test the loop's logic, including error recovery and nudging. A
separate test runs the loop through the real Anthropic SDK against a mocked
HTTP server, checking that the requests are valid.

The de-duplication decision in `sync.py` is tested the same way, against a
fake in-memory repository and a fake embedding client (`tests/test_sync.py`),
so the "is this the same topic?" logic is checked with no database or Voyage
key needed. Separately, `tests/test_db_integration.py` runs `db.py`'s actual
SQL -- cosine-distance ordering, the source-merge upsert, tag unioning --
against a real local Postgres with pgvector, because that's the layer a fake
can't safely stand in for. It skips cleanly if no test database is reachable;
locally that's:

```bash
createdb retrieve_test
psql retrieve_test -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

## Design decisions

- **A hand-written loop instead of a framework.** The loop is short and fully
  explainable, and the project exists to demonstrate agent design.
- **Tools that return fixable errors.** Validation happens in the tools, and
  errors go back to the model instead of crashing the run. The agent corrects
  itself, and the log counts how often that happens.
- **Chunked reading.** `read_pages` is capped at 8 pages per call, so the
  agent summarises as it goes instead of loading a whole textbook at once.
- **A completion check.** `finish` refuses to succeed until every page with
  text has been read, so the agent can't silently skip material.
- **Plain files first, database additive.** Phase 1 stores JSON and
  Markdown, and Phase 2 doesn't replace that -- it adds Postgres alongside
  it. `knowledge/` stays a readable, portable Obsidian vault regardless of
  whether `DATABASE_URL` is set; the database exists for what files can't
  do well, like vector search and rating history.
- **Title before answer.** The review screen makes you attempt recall
  before it shows the summary, because reading an answer and recognising it
  is a different (weaker) skill from producing it. The rating is of the
  attempt, not of whether the note looked familiar.
- **The schedule is derived, not stored.** The `reviews` table is an
  append-only log of what happened; `schedule.py` replays it to work out
  what's due. No `due_at` column to keep in sync, no migration when the
  algorithm changes, and the whole history is there for a better policy
  later. At a few hundred topics the replay is instantaneous.
- **`ask` may say "your notes don't cover this".** The prompt forbids
  outside knowledge and requires citations, so an answer reflects what you
  actually studied. Retrieval is one vector query; generation is one model
  call, not an agent loop, because there's nothing to iterate on.
- **Spaced repetition, not a fixed timetable.** A topic you keep getting
  right comes back less and less often; one you keep forgetting comes back
  tomorrow. That's what makes the review load stay small as the number of
  topics grows.
- **De-duplicate by meaning, not by title string.** Course material repeats
  itself across weeks with different wording ("the TCP handshake" vs. "TCP's
  three-way handshake"). Matching on embeddings catches that; matching on
  exact or fuzzy title text wouldn't.
- **The dedup decision is a separate, fake-testable module.** `sync.py`
  only depends on two small protocols (an embedder, a repository), so the
  question "should this be a new topic or a merge?" is tested without a
  database. `db.py`'s SQL is verified separately, against a real Postgres,
  because a hand-rolled cosine-distance calculation in a fake wouldn't prove
  the actual `<=>` query and index behave correctly.

## Roadmap

- [x] **Phase 1:** distillation agent (CLI), structured topics, run logging, tests
- [x] **Phase 2:** Postgres + pgvector storage (Neon), de-duplication across sources
- [x] **Phase 3:** terminal review: recall from the title, reveal, rate it *solid / shaky / forgotten*
- [x] **Phase 4:** spaced repetition (SM-2 style): ratings decide when each topic comes back; `stats`
- [x] **Phase 5a:** Q&A over your own notes (RAG) with citations
- [ ] **Phase 5b:** generated practice questions, graded against the note
- [ ] **Phase 5c:** interview-story practice from project write-ups
- [ ] Evaluation set: hand-labelled slides to measure tagging accuracy

## Known limitations

- Scanned PDFs with no text layer need OCR first.
- The whole conversation is resent on every turn, so very long documents use
  more tokens. Compressing earlier turns is a planned improvement.
- Cost estimates use a hard-coded price table in `run_log.py`.

## Acknowledgements

The note structure (one concept per note, metadata in frontmatter, distill
before saving, keep an agent log) was inspired by the *Knowledge Agent
Workshop* run by Make AI Practical in Melbourne. This project is an
independent implementation.
