import pytest

from retrieve.tools import MAX_PAGES_PER_READ, DocumentWorkspace, ToolError

VALID_TOPIC = {
    "title": "TCP three-way handshake",
    "summary": "TCP opens a connection with three messages before any data is sent.",
    "key_points": ["SYN", "SYN-ACK", "ACK"],
    "tags": ["Networking", "tcp", "TCP"],
    "difficulty": "core",
    "source_pages": [2],
}


def test_outline_flags_pages_without_text(sample_pages) -> None:
    workspace = DocumentWorkspace(pages=sample_pages)

    outline = workspace.run_tool("get_document_outline", {})

    assert "3 page(s)" in outline
    assert "p1: TCP basics" in outline
    assert "p3: (no extractable text" in outline


def test_read_pages_tracks_what_was_read(sample_pages) -> None:
    workspace = DocumentWorkspace(pages=sample_pages)

    text = workspace.run_tool("read_pages", {"start_page": 1, "end_page": 2})

    assert "--- Page 2 ---" in text
    assert workspace.pages_read == {1, 2}


@pytest.mark.parametrize(
    "page_range",
    [
        {"start_page": 2, "end_page": 1},  # backwards
        {"start_page": 1, "end_page": 99},  # past the end
    ],
)
def test_bad_page_ranges_return_fixable_errors(sample_pages, page_range) -> None:
    workspace = DocumentWorkspace(pages=sample_pages)

    with pytest.raises(ToolError):
        workspace.run_tool("read_pages", page_range)


def test_read_pages_enforces_chunk_size() -> None:
    from retrieve.extract import Page

    many_pages = [Page(number=n, text=f"page {n}") for n in range(1, 21)]
    workspace = DocumentWorkspace(pages=many_pages)

    with pytest.raises(ToolError, match=str(MAX_PAGES_PER_READ)):
        workspace.run_tool("read_pages", {"start_page": 1, "end_page": 20})


def test_save_topic_normalises_tags(sample_pages) -> None:
    workspace = DocumentWorkspace(pages=sample_pages)

    workspace.run_tool("save_topic", VALID_TOPIC)

    assert workspace.saved_topics[0].tags == ["networking", "tcp"]


def test_save_topic_explains_validation_errors(sample_pages) -> None:
    workspace = DocumentWorkspace(pages=sample_pages)
    broken_topic = {**VALID_TOPIC, "difficulty": "impossible", "key_points": []}

    with pytest.raises(ToolError) as error_info:
        workspace.run_tool("save_topic", broken_topic)

    message = str(error_info.value)
    assert "difficulty" in message and "key_points" in message
    assert workspace.saved_topics == []


def test_save_topic_rejects_pages_outside_document(sample_pages) -> None:
    workspace = DocumentWorkspace(pages=sample_pages)

    with pytest.raises(ToolError, match="don't exist"):
        workspace.run_tool("save_topic", {**VALID_TOPIC, "source_pages": [7]})


def test_duplicate_titles_are_rejected(sample_pages) -> None:
    workspace = DocumentWorkspace(pages=sample_pages)
    workspace.run_tool("save_topic", VALID_TOPIC)

    with pytest.raises(ToolError, match="already exists"):
        workspace.run_tool("save_topic", {**VALID_TOPIC, "title": "TCP Three-Way Handshake"})


def test_finish_requires_all_text_pages_read(sample_pages) -> None:
    workspace = DocumentWorkspace(pages=sample_pages)
    workspace.run_tool("read_pages", {"start_page": 1, "end_page": 1})
    workspace.run_tool("save_topic", VALID_TOPIC)

    with pytest.raises(ToolError, match=r"\[2\]"):
        workspace.run_tool("finish", {"note": "done"})

    # Page 3 has no text, so it doesn't need to be read.
    workspace.run_tool("read_pages", {"start_page": 2, "end_page": 2})
    assert workspace.run_tool("finish", {"note": "done"}) == "Finished."
    assert workspace.finished


def test_unknown_tool_is_an_error(sample_pages) -> None:
    workspace = DocumentWorkspace(pages=sample_pages)

    with pytest.raises(ToolError, match="Unknown tool"):
        workspace.run_tool("delete_everything", {})
