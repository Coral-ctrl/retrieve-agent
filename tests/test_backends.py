"""Backend selection and the subscription -> API fallback.

`make_backend` decides how Claude is reached from Settings; the subscription
backend retries on the API key when the subscription path raises
SubscriptionError. Both are checked here with the SDK path stubbed out, so
nothing needs a sign-in or a key that works.
"""

import functools
from pathlib import Path

import pytest

from retrieve import backends
from retrieve.backends import Backend, make_backend, subscription_backend
from retrieve.config import Settings
from retrieve.run_log import RunLogger
from retrieve.sdk_agent import SubscriptionError
from retrieve.tools import DocumentWorkspace


def settings(backend="subscription", api_key=None, token=None) -> Settings:
    return Settings(
        backend=backend, anthropic_api_key=api_key, claude_oauth_token=token,
        model="claude-sonnet-5", max_turns=5, knowledge_dir=Path("k"), logs_dir=Path("l"),
        database_url=None, voyage_api_key=None, embedding_model="voyage-4-lite",
        dedup_threshold=0.1,
    )


# --- selection -------------------------------------------------------------------

def test_subscription_is_chosen_by_default_with_the_api_key_as_backup(monkeypatch) -> None:
    monkeypatch.setattr("retrieve.sdk_agent.sdk_available", lambda: True)
    notices = []

    backend = make_backend(settings(api_key="sk-ant-api-x", token="tok"), notify=notices.append)

    assert backend.name == "subscription"
    assert backend.fallback is not None and backend.fallback.name == "api"
    assert backend.describe == "Claude subscription (API key as backup)"
    assert notices == []


def test_subscription_without_a_key_has_no_backup(monkeypatch) -> None:
    monkeypatch.setattr("retrieve.sdk_agent.sdk_available", lambda: True)

    backend = make_backend(settings(), notify=lambda m: None)

    assert backend.name == "subscription" and backend.fallback is None
    assert "no API key backup" in backend.describe


def test_api_backend_is_used_when_asked_for(monkeypatch) -> None:
    backend = make_backend(settings(backend="api", api_key="sk-ant-api-x"), notify=lambda m: None)
    assert backend.name == "api" and backend.fallback is None
    assert "pay per token" in backend.describe


def test_api_backend_without_a_key_is_explained() -> None:
    notices = []
    assert make_backend(settings(backend="api"), notify=notices.append) is None
    assert "ANTHROPIC_API_KEY is not set" in notices[0]


def test_missing_sdk_falls_back_to_the_key_with_a_notice(monkeypatch) -> None:
    monkeypatch.setattr("retrieve.sdk_agent.sdk_available", lambda: False)
    notices = []

    backend = make_backend(settings(api_key="sk-ant-api-x"), notify=notices.append)

    assert backend.name == "api"
    assert "claude-agent-sdk is not installed" in notices[0]


def test_nothing_configured_is_explained(monkeypatch) -> None:
    monkeypatch.setattr("retrieve.sdk_agent.sdk_available", lambda: False)
    notices = []
    assert make_backend(settings(), notify=notices.append) is None
    assert "subscription" in notices[0] and "ANTHROPIC_API_KEY" in notices[0]


def test_invalid_backend_setting_is_rejected(monkeypatch, tmp_path) -> None:
    from retrieve.config import load_settings

    monkeypatch.setenv("RETRIEVE_BACKEND", "magic")
    with pytest.raises(ValueError, match="RETRIEVE_BACKEND"):
        load_settings(tmp_path)


# --- fallback -------------------------------------------------------------------

class _ApiStub:
    """A fake API backend that records what reached it."""

    def __init__(self) -> None:
        self.runs: list[dict] = []
        self.creates: list[dict] = []

    def backend(self) -> Backend:
        def run(**kwargs):
            self.runs.append(kwargs)
            from retrieve.agent import AgentResult
            return AgentResult(topics=[], finished=True, finish_note="api", turns_used=1,
                               end_reason="finished")

        def create(**kwargs):
            self.creates.append(kwargs)
            return "api-answer"

        return Backend(name="api", agent_runner=run,
                       messages_client=type("M", (), {"create": staticmethod(create)})())


def _failing_sdk(monkeypatch, error="Usage limit reached"):
    def run_agent_sdk(**kwargs):
        # Simulate a half-done subscription run before it died.
        kwargs["workspace"].pages_read.add(1)
        raise SubscriptionError(error)

    class FailingMessages:
        def __init__(self, oauth_token=None):
            pass

        def create(self, **kwargs):
            raise SubscriptionError(error)

    monkeypatch.setattr("retrieve.sdk_agent.run_agent_sdk", run_agent_sdk)
    monkeypatch.setattr("retrieve.sdk_agent.SdkMessagesClient", FailingMessages)


def test_summarize_falls_back_to_the_api_with_a_fresh_workspace(monkeypatch, sample_pages, tmp_path) -> None:
    _failing_sdk(monkeypatch)
    api = _ApiStub()
    notices = []
    backend = subscription_backend(None, fallback=api.backend(), notify=notices.append)
    workspace = DocumentWorkspace(pages=sample_pages)

    outcome = backend.agent_runner(workspace=workspace, model="m", max_turns=5,
                                   logger=RunLogger(tmp_path, "t"), source_name="x.pdf", course=None)

    assert outcome.finish_note == "api"
    assert "billed per token" in notices[0] and "Usage limit reached" in notices[0]
    # The API run got a clean workspace over the same pages, not the half-done one.
    [api_run] = api.runs
    assert api_run["workspace"] is not workspace
    assert api_run["workspace"].pages == sample_pages and api_run["workspace"].pages_read == set()


def test_ask_falls_back_to_the_api(monkeypatch) -> None:
    _failing_sdk(monkeypatch, error="Not logged in")
    api = _ApiStub()
    notices = []
    backend = subscription_backend("tok", fallback=api.backend(), notify=notices.append)

    answer = backend.messages_client.create(model="m", messages=[{"role": "user", "content": "q"}])

    assert answer == "api-answer" and api.creates[0]["model"] == "m"
    assert "Not logged in" in notices[0]


def test_without_a_backup_the_subscription_error_surfaces(monkeypatch, sample_pages, tmp_path) -> None:
    _failing_sdk(monkeypatch)
    backend = subscription_backend(None, fallback=None, notify=lambda m: None)

    with pytest.raises(SubscriptionError, match="Usage limit reached"):
        backend.agent_runner(workspace=DocumentWorkspace(pages=sample_pages), model="m", max_turns=5,
                             logger=RunLogger(tmp_path, "t"), source_name="x.pdf", course=None)


def test_api_backend_builds_the_anthropic_client_lazily(monkeypatch) -> None:
    """Building a backend must not touch the network or the SDK: `ask` with
    no matching notes never calls Claude, and the backup may never be used."""
    import anthropic

    built = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key: built.append(api_key) or
                        type("C", (), {"messages": type("M", (), {"create": staticmethod(lambda **kw: "ok")})()})())

    backend = backends.api_backend("sk-ant-api-x")
    assert built == []
    assert backend.messages_client.create(model="m", messages=[]) == "ok"
    assert built == ["sk-ant-api-x"]
    assert isinstance(backend.agent_runner, functools.partial)
