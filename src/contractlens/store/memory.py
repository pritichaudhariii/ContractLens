"""In-memory store for development and tests: exact cosine search plus a small BM25 index."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from contractlens.models import Chunk, Document
from contractlens.store.base import DocumentStore

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "was",
    "be",
    "by",
    "with",
    "as",
    "at",
    "it",
    "its",
    "this",
    "that",
    "which",
    "what",
    "who",
    "whom",
    "does",
    "do",
    "did",
    "how",
    "when",
    "where",
    "any",
    "all",
    "under",
    "from",
    "than",
    "then",
    "if",
    "not",
    "no",
    "will",
    "shall",
    "may",
    "can",
    "has",
    "have",
    "had",
    "been",
    "into",
    "such",
    "per",
    "vs",
    "s",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class InMemoryStore(DocumentStore):
    name = "memory"

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.embeddings: dict[str, list[float]] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._lengths: dict[str, int] = {}

    # ---- writes ---------------------------------------------------------------

    def upsert_document(self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.delete_document(document.id)
        self.documents[document.id] = document.model_copy(update={"chunk_count": len(chunks)})
        for chunk, emb in zip(chunks, embeddings, strict=True):
            self.chunks[chunk.id] = chunk
            self.embeddings[chunk.id] = emb
            tokens = tokenize(f"{chunk.section} {chunk.text}")
            tf = Counter(tokens)
            self._tf[chunk.id] = tf
            self._lengths[chunk.id] = len(tokens)
            for term in tf:
                self._df[term] += 1

    def delete_document(self, document_id: str) -> bool:
        if document_id not in self.documents:
            return False
        for cid in [c for c in self.chunks if self.chunks[c].document_id == document_id]:
            for term in self._tf[cid]:
                self._df[term] -= 1
            del self._tf[cid], self._lengths[cid], self.chunks[cid], self.embeddings[cid]
        del self.documents[document_id]
        return True

    # ---- reads ------------------------------------------------------------------

    def list_documents(self) -> list[Document]:
        return sorted(self.documents.values(), key=lambda d: d.title)

    def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    def count_chunks(self) -> int:
        return len(self.chunks)

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self.chunks[c] for c in chunk_ids if c in self.chunks]

    def document_chunks(self, document_id: str) -> list[Chunk]:
        return sorted((c for c in self.chunks.values() if c.document_id == document_id), key=lambda c: c.ordinal)

    def vector_search(
        self,
        query_embedding: list[float],
        k: int,
        *,
        doc_types: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        scored: list[tuple[Chunk, float]] = []
        for cid, emb in self.embeddings.items():
            chunk = self.chunks[cid]
            if doc_types and chunk.doc_type not in doc_types:
                continue
            if document_ids and chunk.document_id not in document_ids:
                continue
            scored.append((chunk, _cosine(query_embedding, emb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def keyword_search(
        self, query: str, k: int, *, doc_types: list[str] | None = None, document_ids: list[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        n = len(self.chunks)
        avg_len = sum(self._lengths.values()) / n
        k1, b = 1.5, 0.75
        scores: dict[str, float] = defaultdict(float)
        for term in set(terms):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for cid, tf in self._tf.items():
                f = tf.get(term)
                if not f:
                    continue
                denom = f + k1 * (1 - b + b * self._lengths[cid] / avg_len)
                scores[cid] += idf * (f * (k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out: list[tuple[Chunk, float]] = []
        for cid, score in ranked:
            chunk = self.chunks[cid]
            if doc_types and chunk.doc_type not in doc_types:
                continue
            if document_ids and chunk.document_id not in document_ids:
                continue
            out.append((chunk, score))
            if len(out) >= k:
                break
        return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
