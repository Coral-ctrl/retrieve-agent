"""The subscription backend, offline.

`run_agent_sdk` and `SdkMessagesClient` reach the Claude Agent SDK through
one `driver` callable, so these tests pass a fake driver: an async
generator that plays the model's part, calling the wrapped tools the way
the SDK would and yielding the same message types the SDK yields. No SDK
subprocess, no sign-in, no network -- the same deal as FakeMessagesClient
for the hand-written loop.
"""

from types import SimpleNamespace

import pytest

from retrieve.run_log import RunLogger
from retrieve.sdk_agent import (
    MCP_SERVER_NAME,
    SdkMessagesClient,
    SubscriptionError,
    _sdk_env,
    run_agent_sdk,
)
from retrieve.tools import DocumentWorkspace

TOPIC_INPUT = {
    "title": "TCP three-way handshake",
    "summary": "TCP opens a connection with three messages before any data is sent.",
    "key_points": ["SYN", "SYN-ACK", "ACK"],
    "tags": ["networking"],
    "difficulty": "core",
    "source_pages": [1, 2],
}


# --- message stand-ins: same class names the runner switches on ---------------

class AssistantMessage(SimpleNamespace):
    pass


class ResultMessage(SimpleNamespace):
    pass


_ids = iter(range(1, 10_000))


def assistant(*blocks, input_tokens=1000, output_tokens=100, message_id=None):
    """One AssistantMessage per content block, as the SDK yields them. Blocks
    of the same model turn share a message_id (see _TurnTracker)."""
    return AssistantMessage(content=list(blocks), model="claude-sonnet-5",
                            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
                            stop_reason="end_turn", message_id=message_id or f"msg_{next(_ids)}")


def text(t):
    return SimpleNamespace(type="text", text=t)


def result(*, is_error=False, subtype="success", errors=None, text_result=None,
           usage=None, num_turns=1, cost=0.01):
    return ResultMessage(subtype=subtype, is_error=is_error, errors=errors, result=text_result,
                         usage=usage, num_turns=num_turns, total_cost_usd=cost,
                         stop_reason="end_turn", duration_api_ms=500)


def scripted_driver(script):
    """A fake driver. `script` is a list of steps; each is either
    ("call", tool_name, args) -- the model calls a tool, we run the handler
    like the SDK would -- or a message to yield as-is. Records everything
    it was given so tests can check the wiring."""
    calls = {"invocations": []}

    async def driver(*, prompt, system_prompt, tools, model, max_turns, oauth_token):
        calls["invocations"].append(dict(prompt=prompt, system_prompt=system_prompt,
                                         tool_names=[t.name for t in tools], model=model,
                                         max_turns=max_turns, oauth_token=oauth_token))
        by_name = {t.name: t for t in tools}
        for step in script:
            if isinstance(step, tuple) and step[0] == "call":
                _, name, args = step
                yield assistant(SimpleNamespace(type="tool_use", name=name, input=args))
                outcome = await by_name[name].handler(args)
                calls.setdefault("tool_results", []).append((name, outcome))
            else:
                yield step

    driver.calls = calls
    return driver


# --- run_agent_sdk ---------------------------------------------------------------

def test_sdk_runner_drives_the_same_tools_and_returns_the_topics(sample_pages, tmp_path) -> None:
    driver = scripted_driver([
        ("call", "get_document_outline", {}),
        ("call", "read_pages", {"start_page": 1, "end_page": 2}),
        ("call", "save_topic", TOPIC_INPUT),
        ("call", "finish", {"note": "Page 3 is a diagram."}),
        assistant(text("Done.")),
        result(num_turns=5, usage={"input_tokens": 9000, "cache_read_input_tokens": 1000,
                                   "output_tokens": 700}),
    ])
    logger = RunLogger(tmp_path / "logs", run_label="test")

    outcome = run_agent_sdk(DocumentWorkspace(pages=sample_pages), "claude-sonnet-5", 10,
                            logger, "tcp.pdf", course="Networking", driver=driver,
                            oauth_token="sk-ant-oat01-test")

    assert outcome.finished and outcome.end_reason == "finished"
    assert [t.title for t in outcome.topics] == ["TCP three-way handshake"]
    assert outcome.finish_note == "Page 3 is a diagram."
    assert outcome.turns_used == 5  # one per distinct model turn (message_id)

    # Same tools, same prompt, and the token given to the SDK.
    [call] = driver.calls["invocations"]
    assert call["tool_names"] == ["get_document_outline", "read_pages", "save_topic", "finish"]
    assert "Distill the document 'tcp.pdf' (3 pages)" in call["prompt"]
    assert "course 'Networking'" in call["prompt"]
    assert f"mcp__{MCP_SERVER_NAME}__" in call["system_prompt"]
    assert call["model"] == "claude-sonnet-5" and call["max_turns"] == 10
    assert call["oauth_token"] == "sk-ant-oat01-test"

    # Tool results flow back in the SDK's shape, and every call is logged.
    name, saved = driver.calls["tool_results"][2]
    assert name == "save_topic" and saved["is_error"] is False
    assert "Saved topic #1" in saved["content"][0]["text"]
    assert logger.tool_call_count == 4 and logger.tool_error_count == 0
    # Token totals come from the final ResultMessage (cached input counts
    # as input), never from summing the per-block snapshots.
    assert (logger.total_input_tokens, logger.total_output_tokens) == (10000, 700)


def test_sdk_runner_hands_tool_errors_back_for_the_model_to_correct(sample_pages, tmp_path) -> None:
    driver = scripted_driver([
        ("call", "finish", {"note": "too early"}),  # pages unread -> ToolError
        ("call", "read_pages", {"start_page": 1}),  # missing end_page -> TypeError
        ("call", "read_pages", {"start_page": 1, "end_page": 2}),
        ("call", "save_topic", TOPIC_INPUT),
        ("call", "finish", {"note": "done"}),
        result(),
    ])
    logger = RunLogger(tmp_path / "logs", run_label="test")

    outcome = run_agent_sdk(DocumentWorkspace(pages=sample_pages), "claude-sonnet-5", 10,
                            logger, "tcp.pdf", driver=driver)

    results = driver.calls["tool_results"]
    assert results[0][1]["is_error"] is True and "haven't been read yet" in results[0][1]["content"][0]["text"]
    assert results[1][1]["is_error"] is True and "Invalid arguments" in results[1][1]["content"][0]["text"]
    assert outcome.finished and logger.tool_error_count == 2


def test_sdk_runner_reports_max_turns_when_the_model_never_finishes(sample_pages, tmp_path) -> None:
    driver = scripted_driver([
        ("call", "read_pages", {"start_page": 1, "end_page": 2}),
        result(subtype="error_max_turns", num_turns=1),  # the SDK stopped it, no error flag
    ])
    outcome = run_agent_sdk(DocumentWorkspace(pages=sample_pages), "claude-sonnet-5", 1,
                            RunLogger(tmp_path / "logs", run_label="test"), "tcp.pdf", driver=driver)

    assert outcome.finished is False and outcome.end_reason == "max_turns"
    assert outcome.topics == []


def test_sdk_runner_raises_subscription_error_when_the_sdk_fails(sample_pages, tmp_path) -> None:
    async def broken_driver(**kwargs):
        raise RuntimeError("Not logged in. Run `claude` to sign in.")
        yield  # noqa: unreachable, makes this an async generator

    with pytest.raises(SubscriptionError, match="Not logged in"):
        run_agent_sdk(DocumentWorkspace(pages=sample_pages), "claude-sonnet-5", 10,
                      RunLogger(tmp_path / "logs", run_label="test"), "tcp.pdf",
                      driver=broken_driver)


def test_sdk_runner_raises_subscription_error_when_the_run_ends_in_error(sample_pages, tmp_path) -> None:
    driver = scripted_driver([
        result(is_error=True, subtype="error_during_execution",
               errors=["Usage limit reached. Your limit will reset at 3pm."]),
    ])
    with pytest.raises(SubscriptionError, match="Usage limit reached"):
        run_agent_sdk(DocumentWorkspace(pages=sample_pages), "claude-sonnet-5", 10,
                      RunLogger(tmp_path / "logs", run_label="test"), "tcp.pdf", driver=driver)


def test_sdk_runner_counts_one_turn_per_model_message_not_per_block(sample_pages, tmp_path) -> None:
    # The SDK streams a thinking block and a text block of the same model
    # turn as two AssistantMessages with one message_id: that is one turn.
    driver = scripted_driver([
        assistant(SimpleNamespace(type="thinking", thinking="..."), message_id="msg_A"),
        assistant(text("Reading."), message_id="msg_A"),
        ("call", "read_pages", {"start_page": 1, "end_page": 2}),
        result(usage={"input_tokens": 4200, "output_tokens": 300}, num_turns=2),
    ])
    logger = RunLogger(tmp_path / "logs", run_label="test")

    outcome = run_agent_sdk(DocumentWorkspace(pages=sample_pages), "claude-sonnet-5", 10, logger,
                            "tcp.pdf", driver=driver)

    assert outcome.turns_used == 2
    assert (logger.total_input_tokens, logger.total_output_tokens) == (4200, 300)
    assert logger.estimated_cost_usd("claude-sonnet-5") == pytest.approx(4200 * 2 / 1e6 + 300 * 10 / 1e6)


# --- SdkMessagesClient (ask) -----------------------------------------------------

def test_sdk_messages_client_answers_like_the_anthropic_client() -> None:
    driver = scripted_driver([
        assistant(SimpleNamespace(type="thinking", thinking="..."), message_id="msg_1"),
        assistant(text("Slow start doubles cwnd [1]."), message_id="msg_1"),
        result(usage={"input_tokens": 800, "output_tokens": 50},
               text_result="Slow start doubles cwnd [1]."),  # the SDK's final reply
    ])
    client = SdkMessagesClient(driver=driver, oauth_token="tok")

    response = client.create(model="claude-haiku-4-5", max_tokens=600, system="Answer from notes.",
                             messages=[{"role": "user", "content": "Notes: ... Question: slow start?"}])

    # The exact shape ask.answer_question reads.
    assert [b.type for b in response.content] == ["text"]
    assert response.content[0].text == "Slow start doubles cwnd [1]."
    assert (response.usage.input_tokens, response.usage.output_tokens) == (800, 50)
    assert response.stop_reason == "end_turn"

    [call] = driver.calls["invocations"]
    assert call["tool_names"] == [] and call["max_turns"] == 1
    assert call["system_prompt"] == "Answer from notes." and call["model"] == "claude-haiku-4-5"
    assert call["prompt"].startswith("Notes: ...")


def test_sdk_messages_client_raises_subscription_error_on_a_failed_call() -> None:
    driver = scripted_driver([result(is_error=True, subtype="error_during_execution",
                                     errors=["Invalid OAuth token"])])
    with pytest.raises(SubscriptionError, match="Invalid OAuth token"):
        SdkMessagesClient(driver=driver).create(
            model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])


# --- environment for the SDK subprocess ------------------------------------------

def test_sdk_env_blanks_the_api_key_so_the_subscription_is_used() -> None:
    # An ANTHROPIC_API_KEY in the environment would silently win over the
    # sign-in and bill the API -- the whole point of this backend is that
    # it doesn't.
    assert _sdk_env(None) == {"ANTHROPIC_API_KEY": ""}
    assert _sdk_env("sk-ant-oat01-x") == {"ANTHROPIC_API_KEY": "",
                                          "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-x"}


def test_sdk_messages_client_uses_streamed_text_when_the_result_has_none() -> None:
    driver = scripted_driver([assistant(text("From the stream.")), result(text_result=None)])
    response = SdkMessagesClient(driver=driver).create(
        model="m", max_tokens=10, system="s", messages=[{"role": "user", "content": "q"}])
    assert response.content[0].text == "From the stream."
