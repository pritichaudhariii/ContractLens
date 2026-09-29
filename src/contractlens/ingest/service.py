"""Ingestion service: file → parsed text → section-aware chunks → embeddings → store."""

from __future__ import annotations

import re
from dataclasses import dataclass

from contractlens.embeddings import Embedder
from contractlens.ingest.chunk import chunk_text
from contractlens.ingest.parse import parse_bytes
from contractlens.models import Document, DocumentType
from contractlens.store.base import DocumentStore


@dataclass
class IngestResult:
    document: Document
    chunks: int


class Ingestor:
    def __init__(self, store: DocumentStore, embedder: Embedder, *, chunk_size: int = 1400, overlap: int = 200) -> None:
        self.store = store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest_bytes(
        self,
        data: bytes,
        *,
        filename: str,
        title: str | None = None,
        doc_type: DocumentType = "other",
        document_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> IngestResult:
        parsed = parse_bytes(data, filename)
        return self.ingest_text(
            parsed.text,
            source=filename,
            title=title or parsed.title_hint or _title_from_filename(filename),
            doc_type=doc_type,
            document_id=document_id,
            metadata={**parsed.metadata, **(metadata or {})},
        )

    def ingest_text(
        self,
        text: str,
        *,
        source: str,
        title: str,
        doc_type: DocumentType = "other",
        document_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> IngestResult:
        doc_id = document_id or slugify(title)
        chunks = chunk_text(
            text,
            document_id=doc_id,
            document_title=title,
            doc_type=doc_type,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )
        if not chunks:
            raise ValueError(f"No text could be extracted from {source}")
        embeddings = self.embedder.embed_documents([f"{c.section}\n{c.text}" if c.section else c.text for c in chunks])
        document = Document(
            id=doc_id, title=title, doc_type=doc_type, source=source, metadata=metadata or {}, chunk_count=len(chunks)
        )
        self.store.upsert_document(document, chunks, embeddings)
        return IngestResult(document=document, chunks=len(chunks))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "document"


def _title_from_filename(filename: str) -> str:
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename.rsplit("/", 1)[-1])
    return re.sub(r"[_-]+", " ", stem).strip().title() or "Untitled"
