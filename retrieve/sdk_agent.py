"""The subscription backend: the same agent, driven by the Claude Agent SDK.

Phase 1's `agent.run_agent` is a loop written by hand against the raw
Messages API -- it decides when to call the model, runs each tool, and feeds
results back. That loop needs an API key billed per token.

This module drives the *same tools* with the Claude Agent SDK instead. The
SDK ships its own agent loop (the one inside Claude Code) and, when signed
in with a Claude Pro/Max plan, its usage comes out of the plan's monthly
credit rather than a metered API key. So nothing about the task changes --
`DocumentWorkspace`, `TOOL_DEFINITIONS` and the system prompt are reused
as-is -- only the loop that drives them is swapped out.

What that swap costs, honestly:

- The loop is the SDK's, not ours, so the run log records less per turn
  than the hand-written version (we still log every tool call and the
  token totals the SDK reports).
- The SDK launches a bundled Claude Code binary as a subprocess. Nothing
  extra to install, but it is a second process and a few seconds of
  startup per run.

Two things live here: `run_agent_sdk`, the counterpart of `agent.run_agent`
for `summarize`; and `SdkMessagesClient`, which lets `ask` (a single,
tool-free call) run through the SDK without `ask.py` changing at all --
it presents the SDK's answer in the shape `ask.py` already reads from the
Anthropic client.

Everything that touches the SDK goes through a `driver` callable so the
tests can substitute a fake and run the whole thing offline.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any

from retrieve.agent import SYSTEM_PROMPT, AgentResult
from retrieve.run_log import RunLogger
from retrieve.tools import TOOL_DEFINITIONS, DocumentWorkspace, ToolError

# The SDK exposes in-process tools under an MCP server name. The model sees
# them as mcp__<server>__<tool>, so the server name becomes part of every
# tool name the model calls.
MCP_SERVER_NAME = "retrieve"

SDK_PROMPT_SUFFIX = (
    f"\n\nThe tools are provided by the '{MCP_SERVER_NAME}' MCP server, so their "
    f"full names are prefixed mcp__{MCP_SERVER_NAME}__ (e.g. "
    f"mcp__{MCP_SERVER_NAME}__read_pages). Use only these tools."
)


class SubscriptionError(RuntimeError):
    """The subscription path could not complete: the SDK couldn't start or
    sign in, the plan's usage limit was hit, or the run ended in an error.
    The caller may fall back to the API key, if one is configured."""


# A driver takes everything the SDK needs and yields its messages. The real
# one wraps claude_agent_sdk.query; tests pass an async generator instead.
Driver = Callable[..., AsyncIterator[Any]]


def _sdk_env(oauth_token: str | None) -> dict[str, str]:
    """Environment overrides for the SDK subprocess.

    An ANTHROPIC_API_KEY in the environment silently wins over the
    subscription sign-in and bills the API instead -- and `load_dotenv`
    puts the .env key there. So the subscription path always blanks it
    for the subprocess, and passes the OAuth token explicitly if we have
    one (otherwise the CLI uses whatever `claude` is signed in with).
    """
    env = {"ANTHROPIC_API_KEY": ""}
    if oauth_token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
    return env


def default_driver(*, prompt: str, system_prompt: str, tools: list[Any], model: str,
                   max_turns: int, oauth_token: str | None) -> AsyncIterator[Any]:
    """The real thing: build SDK options and stream `claude_agent_sdk.query`."""
    # Imported here so the rest of the package (and the API backend) works
    # without the SDK installed.
    from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query

    mcp_servers = {}
    allowed_tools: list[str] = []
    if tools:
        mcp_servers[MCP_SERVER_NAME] = create_sdk_mcp_server(MCP_SERVER_NAME, tools=tools)
        allowed_tools = [f"mcp__{MCP_SERVER_NAME}__{t.name}" for t in tools]

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=model,
        max_turns=max_turns,
        tools=[],  # no Claude Code built-ins (Bash, Read...): our tools only
        allowed_tools=allowed_tools,  # ours run without a permission prompt
        mcp_servers=mcp_servers,
        permission_mode="dontAsk",  # nobody is there to answer a prompt
        setting_sources=[],  # ignore any CLAUDE.md / settings on this machine
        skills=[],
        env=_sdk_env(oauth_token),
    )
    return query(prompt=prompt, options=options)


def _run_async(coroutine: Any) -> Any:
    """asyncio.run, but tolerant of being called from a thread that already
    has an event loop (FastAPI runs sync routes in a worker thread; a plain
    asyncio.run is fine there, but this keeps the CLI and tests simple too)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    # Already inside a loop: run in a fresh thread with its own loop.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


# --- summarize: the tool-using agent ------------------------------------------


def build_sdk_tools(workspace: DocumentWorkspace, logger: RunLogger,
                    turn_counter: dict[str, int]) -> list[Any]:
    """Wrap the workspace's tools for the SDK, one per TOOL_DEFINITIONS entry.

    The name, description and JSON schema come straight from the same list
    the hand-written loop sends to the API, so the model sees identical
    tools either way. Each wrapper just calls `workspace.run_tool` and logs
    it -- exactly what the hand-written loop does in its step 3.
    """
    from claude_agent_sdk import tool

    def make(definition: dict[str, Any]) -> Any:
        name = definition["name"]

        @tool(name, definition["description"], definition["input_schema"])
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            tool_input = dict(args or {})
            try:
                result_text = workspace.run_tool(name, tool_input)
                is_error = False
            except ToolError as tool_error:
                result_text = str(tool_error)
                is_error = True
            except TypeError as argument_error:
                result_text = f"Invalid arguments for {name}: {argument_error}"
                is_error = True
            logger.record_tool_call(turn_counter["turn"], name, tool_input, is_error, result_text)
            return {"content": [{"type": "text", "text": result_text}], "is_error": is_error}

        return handler

    return [make(definition) for definition in TOOL_DEFINITIONS]


def run_agent_sdk(
    workspace: DocumentWorkspace,
    model: str,
    max_turns: int,
    logger: RunLogger,
    source_name: str,
    course: str | None = None,
    *,
    driver: Driver = default_driver,
    oauth_token: str | None = None,
) -> AgentResult:
    """Distill `workspace` into topics, with the SDK driving the loop.

    Same inputs and result as `agent.run_agent`, minus the messages client
    (the SDK brings its own). Raises SubscriptionError if the SDK path
    can't run or the run ends in an error.
    """
    course_line = f" It belongs to the course '{course}'." if course else ""
    prompt = (
        f"Distill the document '{source_name}' ({workspace.page_count} pages)"
        f" into study topics.{course_line}"
    )
    logger.log("run_start", source=source_name, pages=workspace.page_count, model=model,
               backend="subscription")

    turn_counter = {"turn": 0}  # shared with the tool wrappers for logging
    tools = build_sdk_tools(workspace, logger, turn_counter)
    tracker = _TurnTracker(logger, turn_counter)

    async def consume() -> Any:
        final = None
        stream = driver(prompt=prompt, system_prompt=SYSTEM_PROMPT + SDK_PROMPT_SUFFIX,
                        tools=tools, model=model, max_turns=max_turns, oauth_token=oauth_token)
        async for message in stream:
            kind = type(message).__name__
            if kind == "AssistantMessage":
                tracker.saw(message)
            elif kind == "ResultMessage":
                final = message
        return final

    try:
        result = _run_async(consume())
    except SubscriptionError:
        raise
    except Exception as sdk_error:  # noqa: BLE001 - CLI not found / not signed in / crashed
        logger.log("run_error", backend="subscription", error=str(sdk_error))
        raise SubscriptionError(f"Subscription path failed: {sdk_error}") from sdk_error

    _record_result_usage(result, logger, turn_counter["turn"])

    if result is not None and getattr(result, "is_error", False):
        detail = "; ".join(getattr(result, "errors", None) or []) or (result.result or "unknown error")
        logger.log("run_error", backend="subscription", subtype=result.subtype, error=detail)
        raise SubscriptionError(f"Subscription run ended in an error ({result.subtype}): {detail}")

    end_reason = "finished" if workspace.finished else "max_turns"
    turns_used = turn_counter["turn"] or int(getattr(result, "num_turns", 0) or 0)
    logger.log("run_end", end_reason=end_reason, turns=turns_used,
               topics=len(workspace.saved_topics), backend="subscription",
               sdk_cost_usd=getattr(result, "total_cost_usd", None), **logger.summary(model))

    return AgentResult(
        topics=list(workspace.saved_topics),
        finished=workspace.finished,
        finish_note=workspace.finish_note,
        turns_used=turns_used,
        end_reason=end_reason,
    )


class _TurnTracker:
    """Counts model turns from the SDK's stream.

    The SDK yields one AssistantMessage per content block (a thinking block,
    then a text block, then a tool call...), all sharing one `message_id`
    and carrying the *same* usage snapshot. So a turn is a distinct
    message_id, and per-message usage is logged for the record but never
    summed -- the totals come from the final ResultMessage instead
    (`_record_result_usage`), which is what the SDK itself bills on.
    """

    def __init__(self, logger: RunLogger, turn_counter: dict[str, int]) -> None:
        self._logger = logger
        self._turn_counter = turn_counter
        self._seen_ids: set[str] = set()
        self._started = time.perf_counter()

    def saw(self, message: Any) -> None:
        message_id = getattr(message, "message_id", None) or f"anon-{self._turn_counter['turn'] + 1}"
        if message_id in self._seen_ids:
            return
        self._seen_ids.add(message_id)
        self._turn_counter["turn"] += 1
        usage = getattr(message, "usage", None) or {}
        self._logger.log(
            "model_message",
            turn=self._turn_counter["turn"],
            input_tokens_snapshot=int(usage.get("input_tokens", 0)),
            stop_reason=getattr(message, "stop_reason", None),
            duration_ms=round((time.perf_counter() - self._started) * 1000),
        )
        self._started = time.perf_counter()


def _usage_totals(usage: dict[str, Any]) -> tuple[int, int]:
    """(input, output) tokens from an SDK usage dict. Cached input tokens are
    counted as input: they were still sent, and the estimate stays honest
    at the "API rates" the CLI reports (slightly high, never low)."""
    input_tokens = (int(usage.get("input_tokens", 0))
                    + int(usage.get("cache_creation_input_tokens", 0))
                    + int(usage.get("cache_read_input_tokens", 0)))
    return input_tokens, int(usage.get("output_tokens", 0))


def _record_result_usage(result: Any, logger: RunLogger, turns_seen: int) -> None:
    """Record the run's token totals from the final ResultMessage -- the one
    place the SDK reports them reliably -- as a single model_call so
    RunLogger's totals and cost estimate add up."""
    if result is None:
        return
    usage = getattr(result, "usage", None) or {}
    if not usage:
        return
    input_tokens, output_tokens = _usage_totals(usage)
    logger.record_model_call(
        turn=turns_seen or int(getattr(result, "num_turns", 0) or 0),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason=getattr(result, "stop_reason", None),
        duration_ms=float(getattr(result, "duration_api_ms", 0) or 0),
    )


# --- ask: a single tool-free call -----------------------------------------------


class SdkMessagesClient:
    """Quacks like `anthropic.Anthropic().messages` for one-shot calls.

    `ask.answer_question` calls `.create(model=, max_tokens=, system=,
    messages=[one user message])` and reads `.content[].text`,
    `.usage.input_tokens`, `.usage.output_tokens` and `.stop_reason` off the
    response. This class answers that call through the SDK and hands back an
    object with exactly those attributes, so `ask.py` and the web app's Ask
    page work on the subscription without knowing it.
    """

    def __init__(self, driver: Driver = default_driver, oauth_token: str | None = None) -> None:
        self._driver = driver
        self._oauth_token = oauth_token

    def create(self, **kwargs: Any) -> SimpleNamespace:
        messages = kwargs.get("messages") or []
        if len(messages) != 1 or messages[0].get("role") != "user":
            raise ValueError("SdkMessagesClient handles a single user message per call.")
        prompt = messages[0]["content"]
        if not isinstance(prompt, str):
            raise ValueError("SdkMessagesClient expects the user message content as text.")

        async def consume() -> tuple[str, Any]:
            text_parts: list[str] = []
            final = None
            stream = self._driver(prompt=prompt, system_prompt=kwargs.get("system") or "",
                                  tools=[], model=kwargs["model"], max_turns=1,
                                  oauth_token=self._oauth_token)
            async for message in stream:
                kind = type(message).__name__
                if kind == "AssistantMessage":
                    for block in getattr(message, "content", []) or []:
                        text = getattr(block, "text", None)
                        if text:
                            text_parts.append(text)
                elif kind == "ResultMessage":
                    final = message
            return "".join(text_parts), final

        try:
            text, result = _run_async(consume())
        except Exception as sdk_error:  # noqa: BLE001
            raise SubscriptionError(f"Subscription path failed: {sdk_error}") from sdk_error

        if result is not None and getattr(result, "is_error", False):
            detail = "; ".join(getattr(result, "errors", None) or []) or (result.result or "unknown error")
            raise SubscriptionError(f"Subscription call ended in an error ({result.subtype}): {detail}")
        # The SDK's final `result` is the complete reply; the streamed text
        # blocks are the fallback if a driver doesn't provide it.
        if result is not None and result.result:
            text = result.result

        usage = (getattr(result, "usage", None) or {}) if result is not None else {}
        input_tokens, output_tokens = _usage_totals(usage)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
            stop_reason=getattr(result, "stop_reason", None) or "end_turn",
        )


def bundled_cli_path() -> str | None:
    """The Claude Code binary to run `setup-token` with: the one bundled in
    the SDK wheel, else a `claude` on PATH, else None."""
    import shutil
    import sys
    from pathlib import Path

    try:
        import claude_agent_sdk
    except ImportError:
        return shutil.which("claude")
    name = "claude.exe" if sys.platform == "win32" else "claude"
    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / name
    if bundled.is_file():
        return str(bundled)
    return shutil.which("claude")


def sdk_available() -> bool:
    """True if the Claude Agent SDK is installed (it bundles the CLI)."""
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        return False
    return True

