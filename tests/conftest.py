"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from retrieve.extract import Page


@pytest.fixture(autouse=True)
def api_backend_by_default(monkeypatch) -> None:
    """Tests that go through cli.py patch `anthropic.Anthropic`, i.e. they
    test the hand-written loop on the API. Pin that backend so the default
    (subscription) doesn't quietly route them to the Claude Agent SDK.
    Subscription tests set RETRIEVE_BACKEND themselves."""
    monkeypatch.setenv("RETRIEVE_BACKEND", "api")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)


@pytest.fixture
def sample_pages() -> list[Page]:
    return [
        Page(number=1, text="TCP basics\nTCP is a connection-oriented transport protocol."),
        Page(number=2, text="Three-way handshake\nSYN, then SYN-ACK, then ACK."),
        Page(number=3, text=""),  # e.g. a diagram-only slide
    ]


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """A real two-page PDF, generated fresh for each test."""
    pdf_path = tmp_path / "networking-week-1.pdf"
    pdf_canvas = canvas.Canvas(str(pdf_path), pagesize=A4)
    pdf_canvas.drawString(72, 750, "Packets and Layers")
    pdf_canvas.drawString(72, 730, "Data is split into packets before transmission.")
    pdf_canvas.showPage()
    pdf_canvas.drawString(72, 750, "TCP vs UDP")
    pdf_canvas.drawString(72, 730, "TCP is reliable and ordered; UDP is faster but unreliable.")
    pdf_canvas.showPage()
    pdf_canvas.save()
    return pdf_path
