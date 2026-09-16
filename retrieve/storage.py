"""Save distilled knowledge to disk.

Phase 1 stores plain files on purpose: a JSON digest per source (easy for
code to load later) and one Markdown note per topic (easy for you to read,
and it opens directly in Obsidian). A database replaces this in Phase 2.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from retrieve.schema import StudyDigest, Topic


def slugify(text: str) -> str:
    """'TCP Three-Way Handshake!' -> 'tcp-three-way-handshake'"""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def save_digest(digest: StudyDigest, knowledge_dir: Path) -> Path:
    """Write the digest and its topic notes. Returns the folder written to."""
    source_folder = knowledge_dir / slugify(Path(digest.source_file).stem)
    source_folder.mkdir(parents=True, exist_ok=True)

    digest_path = source_folder / "digest.json"
    digest_path.write_text(digest.model_dump_json(indent=2), encoding="utf-8")

    for topic in digest.topics:
        note_path = source_folder / f"{slugify(topic.title)}.md"
        note_path.write_text(_topic_to_markdown(topic, digest), encoding="utf-8")

    return source_folder


def _topic_to_markdown(topic: Topic, digest: StudyDigest) -> str:
    # YAML frontmatter holds the metadata; the body is what you actually read.
    # json.dumps quotes values safely, and JSON is valid YAML, so a title
    # containing quotes or colons can't break the frontmatter.
    frontmatter_lines = [
        "---",
        f"title: {json.dumps(topic.title)}",
        f"source: {json.dumps(digest.source_file)}",
        f"pages: {json.dumps(topic.source_pages)}",
        f"difficulty: {topic.difficulty.value}",
        f"tags: {json.dumps(topic.tags)}",
    ]
    if digest.course:
        frontmatter_lines.append(f"course: {json.dumps(digest.course)}")
    frontmatter_lines.append(f"created: {digest.created_at.date().isoformat()}")
    frontmatter_lines.append("---")

    key_point_lines = [f"- {point}" for point in topic.key_points]
    body_lines = [
        f"# {topic.title}",
        "",
        topic.summary,
        "",
        "## Key points",
        *key_point_lines,
        "",
    ]
    return "\n".join(frontmatter_lines + [""] + body_lines)
