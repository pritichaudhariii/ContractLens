"""Section-aware chunking.

Contracts and filings are hierarchical: numbered clauses, articles, headed sections. A chunk
that respects those boundaries is both a better retrieval unit and a better citation, so the
chunker starts a new chunk at every heading, packs paragraphs up to a character budget, and
carries a sentence-aligned overlap between consecutive chunks of the same section.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from contractlens.ingest.parse import PAGE_BREAK
from contractlens.models import Chunk, DocumentType

HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$"),  # markdown headings
    re.compile(r"^(?P<title>(?:ARTICLE|Article|SECTION|Section|PART|Part)\s+[\dIVXLC]+[.:]?\s*[-–—]?\s*.*)$"),
    re.compile(r"^(?P<title>\d{1,2}\.\s+[A-Z][^.]{2,80})$"),  # "9. Limitation of Liability"
    re.compile(r"^(?P<title>\d{1,2}\.\d{1,2}\.?\s+[A-Z][^.]{2,80})$"),  # "9.1 Cap on Damages"
    re.compile(r"^(?P<title>Item\s+\d+[A-Z]?\.\s+.*)$"),  # SEC filing items
]
SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(\"'\[])")


@dataclass
class _Block:
    text: str
    start: int
    section: str


def detect_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    for pattern in HEADING_PATTERNS:
        m = pattern.match(stripped)
        if m:
            return m.group("title").strip()
    return None


def chunk_text(
    text: str,
    *,
    document_id: str,
    document_title: str,
    doc_type: DocumentType = "other",
    chunk_size: int = 1400,
    overlap: int = 200,
) -> list[Chunk]:
    page_starts = _page_offsets(text)
    blocks = _blocks(text)
    chunks: list[Chunk] = []
    buffer: list[_Block] = []
    buffer_len = 0
    carry = ""  # overlap text carried into the next chunk of the same section
    current_section = ""

    def flush() -> None:
        nonlocal buffer, buffer_len, carry
        if not buffer:
            return
        body = "\n\n".join(b.text for b in buffer)
        start = buffer[0].start
        end = buffer[-1].start + len(buffer[-1].text)
        prefix = f"{carry}\n\n" if carry else ""
        chunk_body = prefix + body
        chunks.append(
            Chunk(
                id=_chunk_id(document_id, len(chunks)),
                document_id=document_id,
                document_title=document_title,
                doc_type=doc_type,
                ordinal=len(chunks),
                section=buffer[0].section,
                page=_page_for(start, page_starts),
                text=chunk_body.strip(),
                char_start=start,
                char_end=end,
            )
        )
        carry = _tail(body, overlap)
        buffer, buffer_len = [], 0

    for block in blocks:
        if block.section != current_section:
            flush()
            carry = ""  # never carry text across section boundaries
            current_section = block.section
        pieces = [block] if len(block.text) <= chunk_size else _split_long(block, chunk_size)
        for piece in pieces:
            if buffer and buffer_len + len(piece.text) > chunk_size:
                flush()
            buffer.append(piece)
            buffer_len += len(piece.text) + 2
    flush()
    return chunks


def _blocks(text: str) -> list[_Block]:
    """Paragraph blocks tagged with their nearest preceding heading."""
    blocks: list[_Block] = []
    section = ""
    offset = 0
    para: list[str] = []
    para_start = 0

    def close() -> None:
        nonlocal para
        if para:
            body = "\n".join(para).strip()
            if body:
                blocks.append(_Block(text=body, start=para_start, section=section))
        para = []

    for line in text.splitlines(keepends=True):
        stripped = line.strip(" \t\r\n" + PAGE_BREAK)
        if not stripped:
            close()
        else:
            heading = detect_heading(stripped)
            if heading:
                close()
                section = heading
                blocks.append(_Block(text=heading, start=offset, section=section))
            else:
                if not para:
                    para_start = offset
                para.append(stripped)
        offset += len(line)
    close()
    return blocks


def _split_long(block: _Block, chunk_size: int) -> list[_Block]:
    sentences = SENTENCE_END.split(block.text)
    pieces: list[_Block] = []
    current: list[str] = []
    current_len = 0
    start = block.start
    for sentence in sentences:
        if current and current_len + len(sentence) > chunk_size:
            joined = " ".join(current)
            pieces.append(_Block(text=joined, start=start, section=block.section))
            start += len(joined) + 1
            current, current_len = [], 0
        current.append(sentence)
        current_len += len(sentence) + 1
    if current:
        pieces.append(_Block(text=" ".join(current), start=start, section=block.section))
    return pieces


def _tail(body: str, overlap: int) -> str:
    if overlap <= 0 or len(body) <= overlap:
        return ""
    tail = body[-overlap:]
    # Align to a sentence start so the overlap reads naturally.
    m = SENTENCE_END.search(tail)
    return tail[m.end() :] if m else tail


def _page_offsets(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == PAGE_BREAK:
            starts.append(i + 1)
    return starts


def _page_for(offset: int, page_starts: list[int]) -> int | None:
    if len(page_starts) <= 1:
        return None
    page = 1
    for i, start in enumerate(page_starts, start=1):
        if offset >= start:
            page = i
    return page


def _chunk_id(document_id: str, ordinal: int) -> str:
    digest = hashlib.sha1(f"{document_id}:{ordinal}".encode()).hexdigest()[:12]
    return f"{document_id}-{ordinal:03d}-{digest}"
