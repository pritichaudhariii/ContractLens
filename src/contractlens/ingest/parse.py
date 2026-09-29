"""Parsers that turn uploaded files into plain text with page markers."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from pypdf import PdfReader

PAGE_BREAK = "\f"


@dataclass
class ParsedDocument:
    text: str
    pages: int = 1
    title_hint: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def parse_bytes(data: bytes, filename: str) -> ParsedDocument:
    """Dispatch on file extension. PDF pages are joined with form-feed characters so the
    chunker can recover page numbers; text and markdown are returned as-is."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return parse_pdf(data)
    text = data.decode("utf-8", errors="replace")
    return ParsedDocument(text=text, pages=max(1, text.count(PAGE_BREAK) + 1), title_hint=_title_from_markdown(text))


def parse_pdf(data: bytes) -> ParsedDocument:
    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    meta = {k.lstrip("/"): str(v) for k, v in (reader.metadata or {}).items() if v}
    return ParsedDocument(text=PAGE_BREAK.join(pages), pages=len(pages), title_hint=meta.get("Title"), metadata=meta)


def _title_from_markdown(text: str) -> str | None:
    for line in text.splitlines()[:20]:
        if line.startswith("# "):
            return line[2:].strip()
    return None
