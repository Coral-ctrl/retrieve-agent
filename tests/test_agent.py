import json
from pathlib import Path

from retrieve.agent import run_agent
from retrieve.run_log import RunLogger
from retrieve.tools import DocumentWorkspace

from fakes import FakeMessagesClient, fake_response, text_block, tool_use_block

TOPIC_INPUT = {
    "title": "TCP three-way handshake",
    "summary": "TCP opens a connection with three messages before any data is sent.",
    "key_points": ["SYN", "SYN-ACK", "ACK"],
    "tags": ["networking"],
    "difficulty": "core",
    "source_pages": [1, 2],
}


def make_logger(tmp_path: Path) -> RunLogger:
    return RunLogger(tmp_path / "logs", run_label="test")


def test_agent_completes_a_full_run(sample_pages, tmp_path) -> None:
    client = FakeMessagesClient([
        fake_response(tool_use_block("t1", "get_document_outline", {})),
        fake_response(
            text_block("Reading pages 1-2."),
            tool_use_block("t2", "read_pages", {"start_page": 1, "end_page": 2}),
        ),
        fake_response(tool_use_block("t3", "save_topic", TOPIC_INPUT)),
        fake_response(tool_use_block("t4", "finish", {"note": "Page 3 is a diagram."})),
    ])
    logger = make_logger(tmp_path)

    result = run_agent(DocumentWorkspace(pages=sample_pages), client, "claude-sonnet-5",
                       max_turns=10, logger=logger, source_name="tcp.pdf")

    assert result.finished and result.end_reason == "finished"
    assert result.turns_used == 4
    assert [topic.title for topic in result.topics] == ["TCP three-way handshake"]
    assert result.finish_note == "Page 3 is a diagram."

    # Every tool result is sent back with the matching tool_use_id.
    last_call_messages = client.received_calls[-1]["messages"]
    tool_result = last_call_messages[-1]["content"][0]
    assert tool_result["tool_use_id"] == "t3"
    assert tool_result["is_error"] is False


def test_agent_sees_errors_and_can_recover(sample_pages, tmp_path) -> None:
    client = FakeMessagesClient([
        # Tries to finish too early -> error is returned to the model.
        fake_response(tool_use_block("t1", "finish", {"note": "done"})),
        fake_response(tool_use_block("t2", "read_pages", {"start_page": 1, "end_page": 2})),
        fake_response(tool_use_block("t3", "save_topic", TOPIC_INPUT)),
        fake_response(tool_use_block("t4", "finish", {"note": "done"})),
    ])
    logger = make_logger(tmp_path)

    result = run_agent(DocumentWorkspace(pages=sample_pages), client, "claude-sonnet-5",
                       max_turns=10, logger=logger, source_name="tcp.pdf")

    first_tool_result = client.received_calls[1]["messages"][-1]["content"][0]
    assert first_tool_result["is_error"] is True
    assert "haven't been read" in first_tool_result["content"]
    assert result.finished
    assert logger.tool_error_count == 1


def test_missing_arguments_are_reported_not_raised(sample_pages, tmp_path) -> None:
    client = FakeMessagesClient([
        fake_response(tool_use_block("t1", "read_pages", {"start_page": 1})),
        fake_response(text_block("Stopping."), stop_reason="end_turn"),
    ])

    result = run_agent(DocumentWorkspace(pages=sample_pages), client, "claude-sonnet-5",
                       max_turns=2, logger=make_logger(tmp_path), source_name="tcp.pdf")

    tool_result = client.received_calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "Invalid arguments" in tool_result["content"]
    assert result.end_reason == "max_turns"


def test_text_only_reply_gets_a_nudge(sample_pages, tmp_path) -> None:
    client = FakeMessagesClient([
        fake_response(text_block("I think I'm done."), stop_reason="end_turn"),
        fake_response(text_block("Still done."), stop_reason="end_turn"),
    ])

    result = run_agent(DocumentWorkspace(pages=sample_pages), client, "claude-sonnet-5",
                       max_turns=2, logger=make_logger(tmp_path), source_name="tcp.pdf")

    nudge = client.received_calls[1]["messages"][-1]
    assert nudge["role"] == "user" and "haven't called finish" in nudge["content"]
    assert not result.finished


def test_run_log_records_every_step(sample_pages, tmp_path) -> None:
    client = FakeMessagesClient([
        fake_response(tool_use_block("t1", "read_pages", {"start_page": 1, "end_page": 2})),
        fake_response(tool_use_block("t2", "save_topic", TOPIC_INPUT)),
        fake_response(tool_use_block("t3", "finish", {"note": "ok"})),
    ])
    logger = make_logger(tmp_path)

    run_agent(DocumentWorkspace(pages=sample_pages), client, "claude-sonnet-5",
              max_turns=10, logger=logger, source_name="tcp.pdf")

    events = [json.loads(line) for line in logger.log_path.read_text().splitlines()]
    event_types = [event["event"] for event in events]
    assert event_types[0] == "run_start" and event_types[-1] == "run_end"
    assert event_types.count("model_call") == 3
    assert event_types.count("tool_call") == 3

    run_end = events[-1]
    assert run_end["input_tokens"] == 3000
    # 3000 input * $2/M + 600 output * $10/M = $0.012
    assert abs(run_end["estimated_cost_usd"] - 0.012) < 1e-9
