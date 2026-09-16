from retrieve.schema import Difficulty, StudyDigest, Topic
from retrieve.storage import save_digest, slugify


def test_slugify() -> None:
    assert slugify("TCP Three-Way Handshake!") == "tcp-three-way-handshake"
    assert slugify("???") == "untitled"


def test_digest_and_topic_notes_are_written(tmp_path) -> None:
    topic = Topic(
        title='HTTP: the "request" line',
        summary="Every HTTP request starts with a method, a path and a version.",
        key_points=["GET /index.html HTTP/1.1"],
        tags=["http"],
        difficulty=Difficulty.INTRO,
        source_pages=[3],
    )
    digest = StudyDigest(source_file="Week 2 Slides.pdf", course="Networking",
                         model="claude-sonnet-5", topics=[topic])

    output_folder = save_digest(digest, tmp_path / "knowledge")

    assert output_folder.name == "week-2-slides"
    assert (output_folder / "digest.json").exists()

    note = (output_folder / "http-the-request-line.md").read_text()
    # Quotes and colons in the title are escaped, so the frontmatter stays valid.
    assert 'title: "HTTP: the \\"request\\" line"' in note
    assert 'course: "Networking"' in note
    assert "- GET /index.html HTTP/1.1" in note
