"""Turn source files into plain text, one entry per page or slide.

The agent never sees the raw file. It reads these pages through a tool,
so this module is the only place that knows about file formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from pptx import Presentation

SUPPORTED_SUFFIXES = {".pdf", ".pptx", ".md", ".txt"}


class UnsupportedFileError(ValueError):
    """Raised when we don't know how to read a file type."""


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, matches what you see in a PDF viewer
    text: str

    @property
    def is_empty(self) -> bool:
        # Scanned slides often have no extractable text at all.
        return not self.text.strip()


def extract_pages(file_path: Path) -> list[Page]:
    """Extract text from a supported file, returning one Page per page/slide."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix == ".pptx":
        return _extract_pptx(file_path)
    if suffix in {".md", ".txt"}:
        return _extract_text_file(file_path)

    supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
    raise UnsupportedFileError(f"Can't read '{suffix}' files yet. Supported: {supported}")


def _extract_pdf(file_path: Path) -> list[Page]:
    pages: list[Page] = []
    with pdfplumber.open(file_path) as pdf:
        for page_index, pdf_page in enumerate(pdf.pages, start=1):
            page_text = pdf_page.extract_text() or ""
            pages.append(Page(number=page_index, text=page_text.strip()))
    return pages


def _extract_pptx(file_path: Path) -> list[Page]:
    presentation = Presentation(str(file_path))
    pages: list[Page] = []
    for slide_index, slide in enumerate(presentation.slides, start=1):
        text_parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                text_parts.append(shape.text_frame.text.strip())
        # Speaker notes often hold the real explanation behind a sparse slide.
        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text.strip()
            if notes_text:
                text_parts.append(f"[Speaker notes] {notes_text}")
        pages.append(Page(number=slide_index, text="\n".join(text_parts)))
    return pages


def _extract_text_file(file_path: Path) -> list[Page]:
    # Plain notes have no pages, so treat the whole file as page 1.
    return [Page(number=1, text=file_path.read_text(encoding="utf-8").strip())]
