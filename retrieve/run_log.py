"""Structured logging for agent runs.

Every model call and tool call is written as one JSON line. This is how you
debug an agent ("why did it skip page 12?") and how you answer the interview
question "how did you know it was working?" with evidence.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# USD per million tokens (input, output). Check the pricing page and update
# these if they change: https://platform.claude.com/docs/en/about-claude/pricing
PRICE_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
}


class RunLogger:
    def __init__(self, logs_dir: Path, run_label: str) -> None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.log_path = logs_dir / f"{timestamp}-{run_label}.jsonl"
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.tool_call_count = 0
        self.tool_error_count = 0
        self._started_at = time.perf_counter()

    def log(self, event_type: str, **details: Any) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **details,
        }
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, default=str) + "\n")

    def record_model_call(self, turn: int, input_tokens: int, output_tokens: int,
                          stop_reason: str | None, duration_ms: float) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.log(
            "model_call",
            turn=turn,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            duration_ms=round(duration_ms),
        )

    def record_tool_call(self, turn: int, tool_name: str, tool_input: dict[str, Any],
                         is_error: bool, result_preview: str) -> None:
        self.tool_call_count += 1
        self.tool_error_count += int(is_error)
        self.log(
            "tool_call",
            turn=turn,
            tool=tool_name,
            input=tool_input,
            is_error=is_error,
            result_preview=result_preview[:200],
        )

    def estimated_cost_usd(self, model: str) -> float | None:
        prices = PRICE_PER_MILLION_TOKENS.get(model)
        if prices is None:
            return None
        input_price, output_price = prices
        return (self.total_input_tokens * input_price
                + self.total_output_tokens * output_price) / 1_000_000

    def summary(self, model: str) -> dict[str, Any]:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "tool_calls": self.tool_call_count,
            "tool_errors": self.tool_error_count,
            "estimated_cost_usd": self.estimated_cost_usd(model),
            "duration_s": round(time.perf_counter() - self._started_at, 1),
        }
