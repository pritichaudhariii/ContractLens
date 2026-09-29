"""Storage interface for documents and chunks, with vector and keyword search."""

from __future__ import annotations

from abc import ABC, abstractmethod

from contractlens.models import Chunk, Document


class DocumentStore(ABC):
    name: str

    @abstractmethod
    def upsert_document(self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def list_documents(self) -> list[Document]: ...

    @abstractmethod
    def get_document(self, document_id: str) -> Document | None: ...

    @abstractmethod
    def delete_document(self, document_id: str) -> bool: ...

    @abstractmethod
    def count_chunks(self) -> int: ...

    @abstractmethod
    def vector_search(
        self,
        query_embedding: list[float],
        k: int,
        *,
        doc_types: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity (higher is better), optionally filtered by type or document."""

    @abstractmethod
    def keyword_search(
        self, query: str, k: int, *, doc_types: list[str] | None = None, document_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        """Top-k chunks by lexical relevance (higher is better), optionally filtered by type or document."""

    @abstractmethod
    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]: ...

    @abstractmethod
    def document_chunks(self, document_id: str) -> list[Chunk]:
        """All chunks of one document in reading order."""

    def close(self) -> None:  # pragma: no cover - default no-op
        return None
