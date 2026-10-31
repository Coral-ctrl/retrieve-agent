from pathlib import Path
from types import SimpleNamespace

import anthropic

from retrieve import cli
from fakes import FakeMessagesClient, fake_response, tool_use_block


def run_cli(monkeypatch, tmp_path: Path, argv: list[str], api_key: str | None = "test-key"):
    """Run the CLI from inside tmp_path so knowledge/ and logs/ land there."""
    monkeypatch.chdir(tmp_path)
    if api_key:
        monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)
    else:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return cli.main(argv)


def test_missing_api_key_is_explained(monkeypatch, tmp_path, capsys) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("# UDP\nConnectionless.", encoding="utf-8")

    exit_code = run_cli(monkeypatch, tmp_path, ["summarize", str(notes)], api_key=None)

    assert exit_code == 1
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err  # RETRIEVE_BACKEND=api in tests


def test_unsupported_file_is_explained(monkeypatch, tmp_path, capsys) -> None:
    image = tmp_path / "slide.png"
    image.write_bytes(b"png")

    exit_code = run_cli(monkeypatch, tmp_path, ["summarize", str(image)])

    assert exit_code == 1
    assert "Can't read '.png'" in capsys.readouterr().err


def test_full_run_writes_notes(monkeypatch, tmp_path, sample_pdf, capsys) -> None:
    fake_messages = FakeMessagesClient([
        fake_response(tool_use_block("t1", "read_pages", {"start_page": 1, "end_page": 2})),
        fake_response(tool_use_block("t2", "save_topic", {
            "title": "TCP vs UDP",
            "summary": "TCP is reliable and ordered, while UDP trades reliability for speed.",
            "key_points": ["TCP: reliable, ordered", "UDP: faster, unreliable"],
            "tags": ["networking", "transport-layer"],
            "difficulty": "intro",
            "source_pages": [2],
        })),
        fake_response(tool_use_block("t3", "finish", {"note": "Short source."})),
    ])
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: SimpleNamespace(messages=fake_messages))

    exit_code = run_cli(monkeypatch, tmp_path,
                        ["summarize", str(sample_pdf), "--course", "Networking"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "1. TCP vs UDP  [intro] (p. 2)" in output
    assert "1 topic saved" in output
    assert (tmp_path / "knowledge" / "networking-week-1" / "tcp-vs-udp.md").exists()
    assert list((tmp_path / "logs").glob("*.jsonl"))
    assert "DATABASE_URL not set - topics saved to files only" in output


def _scripted_messages() -> FakeMessagesClient:
    return FakeMessagesClient([
        fake_response(tool_use_block("t1", "read_pages", {"start_page": 1, "end_page": 2})),
        fake_response(tool_use_block("t2", "save_topic", {
            "title": "TCP vs UDP",
            "summary": "TCP is reliable and ordered, while UDP trades reliability for speed.",
            "key_points": ["TCP: reliable, ordered", "UDP: faster, unreliable"],
            "tags": ["networking", "transport-layer"],
            "difficulty": "intro",
            "source_pages": [2],
        })),
        fake_response(tool_use_block("t3", "finish", {"note": "Short source."})),
    ])


def test_no_db_flag_skips_sync_even_with_database_url_set(monkeypatch, tmp_path, sample_pdf, capsys) -> None:
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: SimpleNamespace(messages=_scripted_messages()))
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/wouldnt-actually-connect")
    called = []
    monkeypatch.setattr(cli, "sync_to_database", lambda digest, settings: called.append(digest))

    exit_code = run_cli(monkeypatch, tmp_path, ["summarize", str(sample_pdf), "--no-db"])

    assert exit_code == 0
    assert called == []
    assert "DATABASE_URL not set" not in capsys.readouterr().out


def test_database_url_set_triggers_sync(monkeypatch, tmp_path, sample_pdf, capsys) -> None:
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: SimpleNamespace(messages=_scripted_messages()))
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/wouldnt-actually-connect")
    called = []
    monkeypatch.setattr(cli, "sync_to_database", lambda digest, settings: called.append(digest))

    exit_code = run_cli(monkeypatch, tmp_path, ["summarize", str(sample_pdf)])

    assert exit_code == 0
    assert len(called) == 1
    assert called[0].topics[0].title == "TCP vs UDP"


def test_database_commands_without_database_url_are_explained(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # `ask` checks this first
    (tmp_path / "knowledge" / "x").mkdir(parents=True)
    (tmp_path / "knowledge" / "x" / "digest.json").write_text("{}")

    for argv in (["init-db"], ["review"], ["import-knowledge"], ["stats"], ["ask", "q?"]):
        assert cli.main(argv) == 1
        assert "DATABASE_URL is not set" in capsys.readouterr().err


def test_import_knowledge_with_nothing_to_import_is_explained(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["import-knowledge"])

    assert exit_code == 1
    assert "No digest.json files found" in capsys.readouterr().err


def test_import_knowledge_syncs_each_digest_once(monkeypatch, tmp_path, sample_pdf, capsys) -> None:
    """Runs summarize (files only), then imports the resulting digest into a
    fake repository, then imports again to check it's skipped."""
    from tests.fakes import FakeRepository

    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: SimpleNamespace(messages=_scripted_messages()))
    assert run_cli(monkeypatch, tmp_path, ["summarize", str(sample_pdf), "--course", "Networking"]) == 0
    capsys.readouterr()

    repository = FakeRepository()
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/fake")
    monkeypatch.setattr(cli, "_open_database", lambda settings: _FakeConnection())
    monkeypatch.setattr("retrieve.db.Repository", lambda conn: repository)

    assert cli.main(["import-knowledge"]) == 0
    first_output = capsys.readouterr().out
    assert "networking-week-1.pdf: 1 new, 0 merged" in first_output
    assert "Imported 1 digest(s), skipped 0" in first_output
    assert "No VOYAGE_API_KEY" in first_output
    assert repository.topics[1]["embedding"] is None

    assert cli.main(["import-knowledge"]) == 0
    assert "Imported 0 digest(s), skipped 1" in capsys.readouterr().out
    assert len(repository.topics) == 1

    # --force re-syncs an already-imported source without duplicating anything:
    # the cheap repair path after a failed sync, since it needs no Claude call.
    assert cli.main(["import-knowledge", "--force"]) == 0
    forced_output = capsys.readouterr().out
    assert "networking-week-1.pdf: 0 new, 1 merged" in forced_output
    assert "Imported 1 digest(s), skipped 0" in forced_output
    assert len(repository.topics) == 1
    assert len(repository.sources) == 1


def test_disable_voyage_skips_embeddings_even_with_a_key(monkeypatch, tmp_path, sample_pdf, capsys) -> None:
    """RETRIEVE_DISABLE_VOYAGE=true must win over a configured VOYAGE_API_KEY:
    no voyageai.Client is even constructed, so there's no way this setting
    accidentally still calls out to Voyage."""
    from tests.fakes import FakeRepository

    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: SimpleNamespace(messages=_scripted_messages()))
    assert run_cli(monkeypatch, tmp_path, ["summarize", str(sample_pdf), "--course", "Networking"]) == 0
    capsys.readouterr()

    monkeypatch.setenv("DATABASE_URL", "postgresql://example/fake")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test-key")
    monkeypatch.setenv("RETRIEVE_DISABLE_VOYAGE", "true")
    monkeypatch.setattr(cli, "_open_database", lambda settings: _FakeConnection())
    monkeypatch.setattr("retrieve.db.Repository", lambda conn: FakeRepository())

    import voyageai
    monkeypatch.setattr(voyageai, "Client",
                        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Voyage must not be called")))

    assert cli.main(["import-knowledge"]) == 0
    output = capsys.readouterr().out
    assert "RETRIEVE_DISABLE_VOYAGE is set" in output
    assert "1 new, 0 merged" in output


def _repository_with_two_sources():
    """One topic unique to source A, one shared between A and B."""
    from tests.fakes import FakeRepository

    repository = FakeRepository()
    source_a = repository.get_or_create_source("a.pptx", None, "claude-sonnet-5", None)
    source_b = repository.get_or_create_source("b.pptx", None, "claude-sonnet-5", None)
    only_in_a = repository.insert_topic(
        SimpleNamespace(title="Course Agenda", tags=["logistics"]), None)
    shared = repository.insert_topic(
        SimpleNamespace(title="OSI Model", tags=["networking"]), None)
    repository.link_source(only_in_a, source_a, [1])
    repository.link_source(shared, source_a, [2])
    repository.link_source(shared, source_b, [3])
    return repository


def test_remove_source_deletes_orphaned_topics_but_keeps_shared_ones(monkeypatch, tmp_path, capsys) -> None:
    repository = _repository_with_two_sources()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/fake")
    monkeypatch.setattr(cli, "_open_database", lambda settings: _FakeConnection())
    monkeypatch.setattr("retrieve.db.Repository", lambda conn: repository)

    assert cli.main(["remove-source", "a.pptx"]) == 0

    output = capsys.readouterr().out
    assert "Removed source 'a.pptx'" in output
    assert "Deleted 1 topic(s)" in output
    assert "Course Agenda" in output
    assert "Kept 1 topic(s) also linked to another source" in output
    # a.pptx and the orphaned topic are gone; the shared topic survives.
    assert len(repository.sources) == 1
    assert [t["title"] for t in repository.topics.values()] == ["OSI Model"]


def test_remove_source_with_no_match_is_explained(monkeypatch, tmp_path, capsys) -> None:
    repository = _repository_with_two_sources()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/fake")
    monkeypatch.setattr(cli, "_open_database", lambda settings: _FakeConnection())
    monkeypatch.setattr("retrieve.db.Repository", lambda conn: repository)

    assert cli.main(["remove-source", "nope.pptx"]) == 1
    assert "No source matches" in capsys.readouterr().err
    assert len(repository.sources) == 2  # nothing touched


def test_sources_command_lists_file_names_and_topic_counts(monkeypatch, tmp_path, capsys) -> None:
    repository = _repository_with_two_sources()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/fake")
    monkeypatch.setattr(cli, "_open_database", lambda settings: _FakeConnection())
    monkeypatch.setattr("retrieve.db.Repository", lambda conn: repository)

    assert cli.main(["sources"]) == 0

    output = capsys.readouterr().out
    assert "a.pptx" in output and "(2 topic(s))" in output
    assert "b.pptx" in output and "(1 topic(s))" in output


class _FakeConnection:
    """Enough of a psycopg connection for the CLI's `with conn:` blocks."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def close(self):
        pass


def test_stats_command_prints_a_summary_from_the_repository(monkeypatch, tmp_path, capsys) -> None:
    from datetime import datetime, timezone
    from retrieve.schema import Difficulty, Rating, ReviewEvent, StoredTopic, TopicSource

    class StubRepository:
        def list_topics(self, course=None):
            return [
                StoredTopic(id=1, title="Fresh", summary="x" * 25, key_points=["k"], tags=["t"],
                            difficulty=Difficulty.CORE,
                            sources=[TopicSource(file_name="a.pdf", course="Networking", pages=[1])]),
                StoredTopic(id=2, title="Rated", summary="x" * 25, key_points=["k"], tags=["t"],
                            difficulty=Difficulty.CORE,
                            sources=[TopicSource(file_name="a.pdf", course="Networking", pages=[2])],
                            reviews=[ReviewEvent(rating=Rating.SOLID,
                                                 reviewed_at=datetime.now(timezone.utc))]),
            ]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/fake")
    monkeypatch.setattr(cli, "_open_database", lambda settings: _FakeConnection())
    monkeypatch.setattr("retrieve.db.Repository", lambda conn: StubRepository())

    assert cli.main(["stats", "--course", "Networking"]) == 0

    output = capsys.readouterr().out
    assert "Networking: 2 topic(s)" in output
    assert "due now: 0   new: 1   scheduled later: 1" in output
    assert "last rating: 1 solid" in output
    assert "1 today; streak 1 day" in output


def test_ask_without_api_key_is_explained(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/fake")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert cli.main(["ask", "how does tcp connect?"]) == 1
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_ask_answers_from_retrieved_notes(monkeypatch, tmp_path, capsys) -> None:
    from retrieve.schema import Difficulty, StoredTopic, TopicSource
    from tests.fakes import text_block

    handshake = StoredTopic(
        id=1, title="TCP three-way handshake", summary="TCP opens with SYN, SYN-ACK, ACK.",
        key_points=["SYN"], tags=["tcp"], difficulty=Difficulty.CORE,
        sources=[TopicSource(file_name="week-1.pdf", course="Networking", pages=[2])],
    )

    class StubRepository:
        def find_similar_topics(self, embedding, limit, course=None):
            raise AssertionError("no Voyage key configured, so vector search must not be used")

        def search_topics_by_text(self, query, limit, course=None):
            return [handshake]

    fake_messages = FakeMessagesClient([
        fake_response(text_block("SYN, then SYN-ACK, then ACK [1]."), stop_reason="end_turn"),
    ])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/fake")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_open_database", lambda settings: _FakeConnection())
    monkeypatch.setattr("retrieve.db.Repository", lambda conn: StubRepository())
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key: SimpleNamespace(messages=fake_messages))

    assert cli.main(["ask", "how does tcp connect?", "--model", "claude-haiku-4-5"]) == 0

    output = capsys.readouterr().out
    assert "by keyword (no VOYAGE_API_KEY)" in output
    assert "[1] TCP three-way handshake" in output
    assert "SYN, then SYN-ACK, then ACK [1]." in output
    assert "[1] week-1.pdf p.2" in output
    assert "~US$" in output
    assert fake_messages.received_calls[0]["model"] == "claude-haiku-4-5"
    assert list((tmp_path / "logs").glob("*-ask.jsonl"))


def test_ask_with_no_matching_notes_says_so_without_calling_claude(monkeypatch, tmp_path, capsys) -> None:
    class EmptyRepository:
        def search_topics_by_text(self, query, limit, course=None):
            return []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/fake")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_open_database", lambda settings: _FakeConnection())
    monkeypatch.setattr("retrieve.db.Repository", lambda conn: EmptyRepository())
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: (_ for _ in ()).throw(AssertionError("must not call Claude")))

    assert cli.main(["ask", "quantum entanglement", "--course", "Networking"]) == 0
    assert "No notes in course 'Networking' match" in capsys.readouterr().out


def test_summarize_on_the_subscription_backend_uses_the_sdk_runner(monkeypatch, tmp_path, sample_pdf, capsys) -> None:
    """RETRIEVE_BACKEND=subscription routes summarize through sdk_agent.run_agent_sdk
    (stubbed here) and never builds an Anthropic client."""
    from retrieve.agent import AgentResult
    from retrieve.schema import Difficulty, Topic

    seen = {}

    def fake_run_agent_sdk(*, workspace, model, max_turns, logger, source_name, course, oauth_token=None, **_):
        seen.update(model=model, source=source_name, course=course, token=oauth_token, pages=workspace.page_count)
        return AgentResult(
            topics=[Topic(title="TCP vs UDP", summary="TCP is reliable and ordered; UDP is faster but unreliable.",
                          key_points=["TCP: reliable"], tags=["networking"], difficulty=Difficulty.INTRO,
                          source_pages=[2])],
            finished=True, finish_note=None, turns_used=3, end_reason="finished")

    monkeypatch.setattr("retrieve.sdk_agent.run_agent_sdk", fake_run_agent_sdk)
    monkeypatch.setattr("retrieve.sdk_agent.sdk_available", lambda: True)
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: (_ for _ in ()).throw(AssertionError("must not build an API client")))
    monkeypatch.setenv("RETRIEVE_BACKEND", "subscription")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")

    exit_code = run_cli(monkeypatch, tmp_path, ["summarize", str(sample_pdf), "--course", "Networking"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "via Claude subscription (API key as backup)" in output
    assert "covered by your subscription" in output
    assert seen == {"model": "claude-sonnet-5", "source": "networking-week-1.pdf", "course": "Networking",
                    "token": "sk-ant-oat01-test", "pages": 2}
    assert (tmp_path / "knowledge" / "networking-week-1" / "digest.json").exists()


def test_serve_in_demo_mode_never_builds_claude_or_voyage_clients(monkeypatch, tmp_path, capsys) -> None:
    """Even with an API key and a Voyage key in the environment, a demo
    server must not create the clients that would spend them."""
    import uvicorn

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/demo")  # never connected: no request is made
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test-key")
    monkeypatch.setenv("RETRIEVE_DEMO_MODE", "true")

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("demo mode built a paid client")

    monkeypatch.setattr(cli, "make_backend", must_not_be_called)
    monkeypatch.setattr(cli, "_make_embedder", must_not_be_called)
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: started.update(kwargs))

    exit_code = cli.main(["serve", "--no-browser", "--host", "0.0.0.0", "--port", "9000"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert started["host"] == "0.0.0.0" and started["port"] == 9000
    assert "Demo mode" in output
    assert "http://127.0.0.1:9000" in output  # 0.0.0.0 shown as an address a browser can open
