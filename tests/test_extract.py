from pathlib import Path

import pytest

from retrieve.extract import UnsupportedFileError, extract_pages


def test_pdf_pages_are_extracted_in_order(sample_pdf: Path) -> None:
    pages = extract_pages(sample_pdf)

    assert [page.number for page in pages] == [1, 2]
    assert "Packets and Layers" in pages[0].text
    assert "TCP vs UDP" in pages[1].text


def test_markdown_file_becomes_a_single_page(tmp_path: Path) -> None:
    notes_file = tmp_path / "notes.md"
    notes_file.write_text("# OSI model\nSeven layers.", encoding="utf-8")

    pages = extract_pages(notes_file)

    assert len(pages) == 1
    assert pages[0].text.startswith("# OSI model")


def test_unsupported_file_type_is_rejected(tmp_path: Path) -> None:
    image_file = tmp_path / "diagram.png"
    image_file.write_bytes(b"not really a png")

    with pytest.raises(UnsupportedFileError):
        extract_pages(image_file)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_pages(tmp_path / "does-not-exist.pdf")
