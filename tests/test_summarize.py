"""summarize_file: the orchestration shared by `retrieve summarize` and the
web app's upload page. run_agent itself is covered in test_agent.py; this
checks the wiring around it (reading the file, saving the digest, the two
error cases) with a fake Claude client, so no network or API key involved.
"""

from pathlib import Path

import pytest

from retrieve.extract import UnsupportedFileError
from retrieve.summarize import SummarizeError, summarize_file

from fakes import FakeMessagesClient, fake_response, tool_use_block

READ_PAGE_1 = tool_use_block("t0", "read_pages", {"start_page": 1, "end_page": 1})

TOPIC_INPUT = {
    "title": "TCP three-way handshake",
    "summary": "TCP opens a connection with three messages before any data is sent.",
    "key_points": ["SYN", "SYN-ACK", "ACK"],
    "tags": ["networking"],
    "difficulty": "core",
    "source_pages": [1],
}


def _notes_file(tmp_path: Path, text: str = "TCP opens with SYN, SYN-ACK, ACK.") -> Path:
    path = tmp_path / "week-1.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_summarize_file_saves_a_digest_from_a_successful_run(tmp_path) -> None:
    client = FakeMessagesClient([
        fake_response(READ_PAGE_1),
        fake_response(tool_use_block("t1", "save_topic", TOPIC_INPUT)),
        fake_response(tool_use_block("t2", "finish", {"note": "All done."})),
    ])

    result = summarize_file(
        _notes_file(tmp_path), original_name="week-1.txt", course="Networking",
        model="claude-sonnet-5", max_turns=10, messages_client=client,
        logs_dir=tmp_path / "logs", knowledge_dir=tmp_path / "knowledge",
    )

    assert result.digest.source_file == "week-1.txt"
    assert result.digest.course == "Networking"
    assert [t.title for t in result.digest.topics] == ["TCP three-way handshake"]
    assert result.finished is True
    assert result.turns_used == 3
    assert result.output_folder == tmp_path / "knowledge" / "week-1"
    assert (result.output_folder / "digest.json").exists()
    assert result.log_path.exists()


def test_summarize_file_rejects_an_unsupported_extension(tmp_path) -> None:
    bad_file = tmp_path / "slides.key"
    bad_file.write_text("not readable", encoding="utf-8")

    with pytest.raises(UnsupportedFileError):
        summarize_file(
            bad_file, original_name="slides.key", course=None, model="claude-sonnet-5",
            max_turns=10, messages_client=FakeMessagesClient([]),
            logs_dir=tmp_path / "logs", knowledge_dir=tmp_path / "knowledge",
        )


def test_summarize_file_raises_when_the_file_has_no_extractable_text(tmp_path) -> None:
    empty_file = _notes_file(tmp_path, text="   \n\n  ")

    with pytest.raises(SummarizeError, match="No text could be extracted"):
        summarize_file(
            empty_file, original_name="week-1.txt", course=None, model="claude-sonnet-5",
            max_turns=10, messages_client=FakeMessagesClient([]),
            logs_dir=tmp_path / "logs", knowledge_dir=tmp_path / "knowledge",
        )


def test_summarize_file_raises_when_the_agent_saves_nothing(tmp_path) -> None:
    # Never reads the page or saves a topic, so every finish attempt is
    # rejected until max_turns runs out -- no topics, nothing to save.
    client = FakeMessagesClient([
        fake_response(tool_use_block("t1", "finish", {"note": "done"})),
        fake_response(tool_use_block("t2", "finish", {"note": "done"})),
    ])

    with pytest.raises(SummarizeError, match="didn't save any topics"):
        summarize_file(
            _notes_file(tmp_path), original_name="week-1.txt", course=None,
            model="claude-sonnet-5", max_turns=2, messages_client=client,
            logs_dir=tmp_path / "logs", knowledge_dir=tmp_path / "knowledge",
        )
