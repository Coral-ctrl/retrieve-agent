"""Choosing how to reach Claude: subscription first, API key as backup.

A `Backend` bundles the two things the rest of the app needs to talk to
Claude: an `agent_runner` for `summarize` (the loop that drives the tools)
and a `messages_client` for `ask` (one tool-free call). `cli.py` and the
web app build one from Settings and never care which kind they got.

    subscription  -> retrieve.sdk_agent (Claude Agent SDK, plan credit)
    api           -> retrieve.agent + anthropic.Anthropic (metered key)

With RETRIEVE_BACKEND=subscription (the default) and an ANTHROPIC_API_KEY
also set, the key is the fallback: if the subscription path can't run --
the SDK isn't installed, `claude` isn't signed in, the plan's usage limit
is hit -- the same job is re-run on the API and a notice says so, because
that is the moment metered billing starts. Nothing is written to disk or
the database until a run succeeds, so re-running is safe.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from retrieve.agent import AgentResult
from retrieve.config import BACKEND_API, BACKEND_SUBSCRIPTION, Settings
from retrieve.sdk_agent import SubscriptionError


class AgentRunner(Protocol):
    """What `summarize_file` needs: run the agent over a workspace."""

    def __call__(self, *, workspace: Any, model: str, max_turns: int, logger: Any,
                 source_name: str, course: str | None) -> AgentResult: ...


@dataclass
class Backend:
    name: str  # "subscription" or "api"
    agent_runner: AgentRunner
    messages_client: Any  # has .create(**kwargs) like anthropic's client.messages
    fallback: Backend | None = None  # used only by the subscription backend

    @property
    def describe(self) -> str:
        if self.name == BACKEND_SUBSCRIPTION:
            extra = " (API key as backup)" if self.fallback else " (no API key backup)"
            return "Claude subscription" + extra
        return "Anthropic API (pay per token)"


Notify = Callable[[str], None]


class _LazyMessages:
    """anthropic's client.messages, created on first use.

    Building the backend must stay free of side effects: `ask` with no
    matching notes never talks to Claude, and as the fallback behind the
    subscription path the API client may never be needed at all."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._messages: Any = None

    def create(self, **kwargs: Any) -> Any:
        if self._messages is None:
            import anthropic

            self._messages = anthropic.Anthropic(api_key=self._api_key).messages
        return self._messages.create(**kwargs)


def api_backend(api_key: str) -> Backend:
    """The Phase 1 path: the hand-written loop on the Anthropic API."""
    from retrieve.agent import run_agent

    messages = _LazyMessages(api_key)
    return Backend(
        name=BACKEND_API,
        agent_runner=functools.partial(run_agent, messages_client=messages),
        messages_client=messages,
    )


def subscription_backend(oauth_token: str | None, fallback: Backend | None,
                         notify: Notify) -> Backend:
    """The SDK path, with optional automatic fallback to the API backend."""
    from retrieve.sdk_agent import SdkMessagesClient, run_agent_sdk

    primary_runner = functools.partial(run_agent_sdk, oauth_token=oauth_token)
    primary_messages = SdkMessagesClient(oauth_token=oauth_token)

    if fallback is None:
        return Backend(name=BACKEND_SUBSCRIPTION, agent_runner=primary_runner,
                       messages_client=primary_messages)

    def runner_with_fallback(**kwargs: Any) -> AgentResult:
        try:
            return primary_runner(**kwargs)
        except SubscriptionError as error:
            notify(f"Subscription path unavailable ({error}). "
                   "Falling back to the Anthropic API key -- this run is billed per token.")
            # A fresh workspace over the same pages, so a half-done
            # subscription run leaves nothing behind for the API run.
            from retrieve.tools import DocumentWorkspace

            kwargs["workspace"] = DocumentWorkspace(pages=kwargs["workspace"].pages)
            return fallback.agent_runner(**kwargs)

    class MessagesWithFallback:
        def create(self, **kwargs: Any) -> Any:
            try:
                return primary_messages.create(**kwargs)
            except SubscriptionError as error:
                notify(f"Subscription path unavailable ({error}). "
                       "Falling back to the Anthropic API key -- this call is billed per token.")
                return fallback.messages_client.create(**kwargs)

    return Backend(name=BACKEND_SUBSCRIPTION, agent_runner=runner_with_fallback,
                   messages_client=MessagesWithFallback(), fallback=fallback)


def make_backend(settings: Settings, notify: Notify = print) -> Backend | None:
    """Build the backend Settings asks for, or None (with a printed reason)
    if neither path is usable."""
    from retrieve.sdk_agent import sdk_available

    api = api_backend(settings.anthropic_api_key) if settings.anthropic_api_key else None

    if settings.backend == BACKEND_API:
        if api is None:
            notify("RETRIEVE_BACKEND=api but ANTHROPIC_API_KEY is not set. "
                   "Add the key to .env, or switch to RETRIEVE_BACKEND=subscription.")
        return api

    # subscription (default)
    if sdk_available():
        return subscription_backend(settings.claude_oauth_token, fallback=api, notify=notify)
    if api is not None:
        notify("claude-agent-sdk is not installed, so the subscription path can't run. "
               "Using the Anthropic API key instead (billed per token). "
               "Install it with: pip install claude-agent-sdk")
        return api
    notify("No way to reach Claude: install claude-agent-sdk and sign in with `claude` "
           "(or set CLAUDE_CODE_OAUTH_TOKEN) to use your subscription, or set "
           "ANTHROPIC_API_KEY to use the API.")
    return None
