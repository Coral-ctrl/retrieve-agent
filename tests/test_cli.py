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
