"""Core domain models shared by ingestion, retrieval, the graph and the API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

DocumentType = Literal["contract", "regulatory_filing", "policy", "other"]


class Document(BaseModel):
    id: str
    title: str
    doc_type: DocumentType = "other"
    source: str = Field(default="", description="File name or origin")
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    chunk_count: int = 0


class Chunk(BaseModel):
    """A retrievable unit of a document with enough metadata to cite it precisely."""

    id: str
    document_id: str
    document_title: str
    doc_type: DocumentType = "other"
    ordinal: int
    section: str = Field(default="", description="Nearest heading, e.g. '9. Limitation of Liability'")
    page: int | None = None
    text: str
    char_start: int = 0
    char_end: int = 0


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float = Field(description="Fused relevance score (higher is better)")
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    marker: int = Field(description="The [n] used in the answer text")
    chunk_id: str
    document_id: str
    document_title: str
    section: str = ""
    page: int | None = None
    quote: str = Field(description="A short supporting excerpt from the chunk")
    supported: bool = Field(default=True, description="Whether the grounding check found the citation supported")


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    grounded: bool = Field(description="True when every citation passed the grounding check")
    unsupported_markers: list[int] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    trace: dict[str, float | int | str] = Field(default_factory=dict)
