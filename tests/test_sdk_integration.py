"""Runs the agent loop through the real Anthropic SDK with a fake HTTP server.

The fake-client tests prove the loop's logic. This test proves the requests
we build are ones the real SDK accepts, including sending the SDK's own
response blocks back in the next request, without calling the real API.
"""

import json

import anthropic
import httpx2

from retrieve.agent import run_agent
from retrieve.run_log import RunLogger
from retrieve.tools import DocumentWorkspace

SCRIPTED_TOOL_CALLS = [
    ("read_pages", {"start_page": 1, "end_page": 2}),
    ("save_topic", {
        "title": "TCP three-way handshake",
        "summary": "TCP opens a connection with three messages before any data is sent.",
        "key_points": ["SYN", "SYN-ACK", "ACK"],
        "tags": ["networking"],
        "difficulty": "core",
        "source_pages": [2],
    }),
    ("finish", {"note": "ok"}),
]


def test_agent_runs_through_real_sdk(sample_pages, tmp_path) -> None:
    captured_request_bodies: list[dict] = []

    def fake_api(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        captured_request_bodies.append(body)
        tool_name, tool_input = SCRIPTED_TOOL_CALLS[len(captured_request_bodies) - 1]
        return httpx2.Response(200, json={
            "id": f"msg_{len(captured_request_bodies)}",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [{
                "type": "tool_use",
                "id": f"toolu_{len(captured_request_bodies)}",
                "name": tool_name,
                "input": tool_input,
            }],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 500, "output_tokens": 50},
        })

    client = anthropic.Anthropic(
        api_key="test-key",
        http_client=httpx2.Client(transport=httpx2.MockTransport(fake_api)),
    )

    result = run_agent(
        DocumentWorkspace(pages=sample_pages),
        client.messages,
        model="claude-sonnet-5",
        max_turns=5,
        logger=RunLogger(tmp_path, "sdk"),
        source_name="tcp.pdf",
    )

    assert result.finished
    assert len(captured_request_bodies) == 3

    # The second request must contain the first assistant turn, serialised
    # by the SDK, followed by our tool_result.
    second_request_messages = captured_request_bodies[1]["messages"]
    assert second_request_messages[1]["content"][0]["type"] == "tool_use"
    assert second_request_messages[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": second_request_messages[2]["content"][0]["content"],
        "is_error": False,
    }
    assert {tool["name"] for tool in captured_request_bodies[0]["tools"]} == {
        "get_document_outline", "read_pages", "save_topic", "finish",
    }
