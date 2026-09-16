"""The agent loop.

This is the heart of the project, written by hand on purpose. The loop:

    1. Send the conversation plus tool definitions to Claude.
    2. If Claude asks to use tools, run them and send the results back.
    3. Repeat until Claude calls `finish`, or we hit the turn limit.

No framework is involved, so every step can be explained line by line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from retrieve.run_log import RunLogger
from retrieve.schema import Topic
from retrieve.tools import TOOL_DEFINITIONS, DocumentWorkspace, ToolError

MAX_OUTPUT_TOKENS = 4096

SYSTEM_PROMPT = """\
You are Retrieve, a study agent. Your job is to turn course material into \
concise, well-organised knowledge that a learner can review before a job \
interview.

How to work:
- Start with get_document_outline and plan your reading order.
- Read the document in page ranges with read_pages. Read every page that has text.
- After each range, save the distinct concepts you found with save_topic.
- One concept per topic. Prefer specific titles ("TCP three-way handshake") \
over broad ones ("Networking").
- Summaries must be accurate to the source. Do not add facts the source \
doesn't support; if the source is vague, say so in your finish note.
- Keep the exact terms, steps and order the source uses in key_points, \
because learners need to recall precise wording under pressure.
- If a tool returns an error, read it and correct your call.
- When everything is covered, call finish with a short note on gaps or \
unclear parts of the source."""


class MessagesClient(Protocol):
    """The one method we need from anthropic.Anthropic().messages.

    Typing against this protocol lets the tests pass in a fake client, so
    the loop can be tested without an API key or network access.
    """

    def create(self, **kwargs: Any) -> Any: ...


@dataclass
class AgentResult:
    topics: list[Topic]
    finished: bool
    finish_note: str | None
    turns_used: int
    end_reason: str  # "finished" or "max_turns"


def run_agent(
    workspace: DocumentWorkspace,
    messages_client: MessagesClient,
    model: str,
    max_turns: int,
    logger: RunLogger,
    source_name: str,
    course: str | None = None,
) -> AgentResult:
    course_line = f" It belongs to the course '{course}'." if course else ""
    conversation: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Distill the document '{source_name}' ({workspace.page_count} pages)"
                f" into study topics.{course_line}"
            ),
        }
    ]
    logger.log("run_start", source=source_name, pages=workspace.page_count, model=model)

    end_reason = "max_turns"
    turns_used = 0

    for turn in range(1, max_turns + 1):
        turns_used = turn

        # --- 1. Ask the model what to do next ---------------------------
        call_started = time.perf_counter()
        response = messages_client.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=conversation,
        )
        logger.record_model_call(
            turn=turn,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
            duration_ms=(time.perf_counter() - call_started) * 1000,
        )
        conversation.append({"role": "assistant", "content": response.content})

        tool_requests = [block for block in response.content if block.type == "tool_use"]

        # --- 2. The model replied with text only: nudge it back on task ---
        if not tool_requests:
            conversation.append({
                "role": "user",
                "content": "You haven't called finish yet. Keep reading and saving "
                           "topics, then call finish.",
            })
            continue

        # --- 3. Run each requested tool and collect the results ---------
        tool_results = []
        for tool_request in tool_requests:
            tool_input = dict(tool_request.input or {})
            try:
                result_text = workspace.run_tool(tool_request.name, tool_input)
                is_error = False
            except ToolError as tool_error:
                result_text = str(tool_error)
                is_error = True
            except TypeError as argument_error:
                # Missing or unexpected arguments, e.g. read_pages without end_page.
                result_text = f"Invalid arguments for {tool_request.name}: {argument_error}"
                is_error = True

            logger.record_tool_call(turn, tool_request.name, tool_input, is_error, result_text)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_request.id,
                "content": result_text,
                "is_error": is_error,
            })

        conversation.append({"role": "user", "content": tool_results})

        # --- 4. Stop once the agent has successfully called finish -------
        if workspace.finished:
            end_reason = "finished"
            break

    logger.log("run_end", end_reason=end_reason, turns=turns_used,
               topics=len(workspace.saved_topics), **logger.summary(model))

    return AgentResult(
        topics=list(workspace.saved_topics),
        finished=workspace.finished,
        finish_note=workspace.finish_note,
        turns_used=turns_used,
        end_reason=end_reason,
    )
