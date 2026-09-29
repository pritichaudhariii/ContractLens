"""Request and response schemas for the HTTP API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from contractlens.models import Citation, DocumentType


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3, max_length=2000, examples=["What is the notice period for termination for convenience?"]
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="Passages to pass to the model")
    doc_types: list[DocumentType] | None = Field(default=None, description="Restrict retrieval to these document types")
    include_passages: bool = Field(default=False, description="Return the retrieved passages alongside the answer")


class PassageOut(BaseModel):
    marker: int
    chunk_id: str
    document_id: str
    document_title: str
    section: str
    page: int | None
    score: float
    vector_rank: int | None
    keyword_rank: int | None
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    grounded: bool
    unsupported_markers: list[int]
    passages: list[PassageOut] | None = None
    trace: dict[str, float | int | str]


class DocumentOut(BaseModel):
    id: str
    title: str
    doc_type: str
    source: str
    chunk_count: int
    created_at: datetime
    metadata: dict[str, str]


class IngestTextRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=20)
    doc_type: DocumentType = "other"
    document_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document: DocumentOut
    chunks: int


class HealthResponse(BaseModel):
    status: str
    version: str
    documents: int
    chunks: int
    components: dict[str, str | int]
