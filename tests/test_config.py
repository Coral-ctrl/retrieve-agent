"""Settings parsing that the rest of the suite relies on but doesn't test
directly. load_settings is pointed at an empty tmp_path, so a real .env on
the machine running the tests can't change the result."""

from retrieve.config import load_settings


def test_demo_mode_is_off_unless_switched_on(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("RETRIEVE_DEMO_MODE", raising=False)
    assert load_settings(project_root=tmp_path).demo_mode is False

    monkeypatch.setenv("RETRIEVE_DEMO_MODE", "true")
    assert load_settings(project_root=tmp_path).demo_mode is True

    # Anything that isn't a clear "on" stays off: a typo shouldn't publish a demo.
    monkeypatch.setenv("RETRIEVE_DEMO_MODE", "ture")
    assert load_settings(project_root=tmp_path).demo_mode is False
