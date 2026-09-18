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
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


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
    (tmp_path / "knowledge" / "x").mkdir(parents=True)
    (tmp_path / "knowledge" / "x" / "digest.json").write_text("{}")

    for argv in (["init-db"], ["review"], ["import-knowledge"], ["stats"]):
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
