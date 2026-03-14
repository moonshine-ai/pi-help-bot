"""
Fetch asciidoc content from the documentation tree by source path#section.

Source format: path/to/file.faq#section-slug or path#whole-document.
Resolves includes and returns either the full document or the named section
as asciidoc text.
"""

from __future__ import annotations

import re
from pathlib import Path

# Reuse parsing helpers from docs-to-faqs
try:
    from docs_to_faqs import (
        _ANCHOR_RE,
        _HEADING_RE,
        _INCLUDE_RE,
        _asciidoc_slug,
        _resolve_includes,
    )
except ImportError:
    _HEADING_RE = re.compile(r"^(={1,6})\s+(.+)$")
    _ANCHOR_RE = re.compile(r"^\[#([\w\-]+)\]")
    _INCLUDE_RE = re.compile(r"^include::([^\[]+)\[")

    def _asciidoc_slug(text: str) -> str:
        s = text.lower()
        s = re.sub(r"[^\w\s-]", "", s)
        s = re.sub(r"[\s_]+", "-", s)
        s = re.sub(r"-{2,}", "-", s)
        return s.strip("-")

    def _resolve_includes(adoc_path: Path, depth: int = 0) -> list[str]:
        if depth > 8 or not adoc_path.exists():
            return []
        lines = adoc_path.read_text(encoding="utf-8", errors="replace").splitlines()
        result = []
        for line in lines:
            m = _INCLUDE_RE.match(line)
            if m:
                include_path = adoc_path.parent / m.group(1)
                result.extend(_resolve_includes(include_path, depth + 1))
            else:
                result.append(line)
        return result


def _adoc_path_from_source(doc_root: Path, source_path: str) -> Path | None:
    """Resolve source path (e.g. 'accessories/ai-camera/about.faq') to .adoc file."""
    path_part = source_path.replace("\\", "/").strip("/")
    # Embeddings use .faq from .faq.txt stem; doc is .adoc
    if path_part.endswith(".faq"):
        path_part = path_part[:-4]  # drop .faq
    if not path_part.endswith(".adoc"):
        path_part = path_part + ".adoc"
    full = doc_root / path_part
    if full.exists():
        return full
    # Try with .adoc appended if path had no extension
    alt = doc_root / (source_path.strip("/").replace(".faq", "") + ".adoc")
    return alt if alt.exists() else None


def _extract_section_as_adoc(lines: list[str], section_slug: str) -> str | None:
    """
    Find the section with the given anchor slug and return its raw asciidoc
    (heading line + body lines until the next same- or higher-level heading).
    """
    pending_anchor: str | None = None
    in_literal = False
    i = 0
    section_start: int | None = None
    section_level = 0

    while i < len(lines):
        line = lines[i]

        if line.strip() in ("----", "...."):
            in_literal = not in_literal
            i += 1
            continue
        if in_literal:
            i += 1
            continue

        m_anchor = _ANCHOR_RE.match(line.strip())
        if m_anchor:
            pending_anchor = m_anchor.group(1)
            i += 1
            continue

        m_head = _HEADING_RE.match(line)
        if m_head:
            level = len(m_head.group(1))
            heading = m_head.group(2).strip()
            anchor = pending_anchor or _asciidoc_slug(heading)
            pending_anchor = None

            if section_start is not None:
                # End of previous section; we've found the next heading
                return "\n".join(lines[section_start:i])

            if anchor == section_slug:
                section_start = i
                section_level = level
            i += 1
            continue

        pending_anchor = None
        i += 1

    if section_start is not None:
        return "\n".join(lines[section_start:])
    return None


def get_doc_content(
    doc_root: str | Path,
    source: str,
) -> tuple[str | None, str | None]:
    """
    Fetch asciidoc content for the given source.

    source: "path/to/file.faq#section-slug" or "path#whole-document"

    Returns (content_adoc, error). On success content_adoc is the asciidoc
    string and error is None. On failure content_adoc is None and error
    is a short message.
    """
    doc_root = Path(doc_root)
    if "#" in source:
        path_part, section_slug = source.split("#", 1)
        section_slug = section_slug.strip()
    else:
        path_part = source
        section_slug = "whole-document"

    path_part = path_part.strip()
    adoc_path = _adoc_path_from_source(doc_root, path_part)
    if not adoc_path or not adoc_path.is_file():
        return None, "File not found"

    try:
        lines = _resolve_includes(adoc_path)
    except Exception as e:
        return None, str(e)

    if not lines:
        return None, "Empty or unreadable file"

    if section_slug == "whole-document":
        return "\n".join(lines), None

    content = _extract_section_as_adoc(lines, section_slug)
    if content is None:
        return None, f"Section '{section_slug}' not found"
    return content, None
