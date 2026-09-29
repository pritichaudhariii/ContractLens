"""Hybrid retrieval: dense (pgvector cosine) + sparse (full-text) fused with Reciprocal Rank Fusion.

RRF is rank-based, so it needs no score calibration between the two retrievers and is
robust when one side returns nothing (e.g. an exact clause number that has no semantic
signal, or a paraphrased question with no lexical overlap).
"""

from __future__ import annotations

from dataclasses import dataclass

from contractlens.embeddings import Embedder
from contractlens.models import Chunk, RetrievedChunk
from contractlens.store.base import DocumentStore


@dataclass
class HybridRetriever:
    store: DocumentStore
    embedder: Embedder
    candidates: int = 24
    rrf_k: int = 60

    def retrieve(
        self, query: str, *, k: int, doc_types: list[str] | None = None, document_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query)
        dense = self.store.vector_search(
            query_embedding, self.candidates, doc_types=doc_types, document_ids=document_ids
        )
        sparse = self.store.keyword_search(query, self.candidates, doc_types=doc_types, document_ids=document_ids)
        return reciprocal_rank_fusion(dense, sparse, k=k, rrf_k=self.rrf_k)


def reciprocal_rank_fusion(
    dense: list[tuple[Chunk, float]],
    sparse: list[tuple[Chunk, float]],
    *,
    k: int,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    fused: dict[str, RetrievedChunk] = {}

    for rank, (chunk, _score) in enumerate(dense, start=1):
        entry = fused.get(chunk.id) or RetrievedChunk(chunk=chunk, score=0.0)
        entry.score += 1.0 / (rrf_k + rank)
        entry.vector_rank = rank
        fused[chunk.id] = entry

    for rank, (chunk, _score) in enumerate(sparse, start=1):
        entry = fused.get(chunk.id) or RetrievedChunk(chunk=chunk, score=0.0)
        entry.score += 1.0 / (rrf_k + rank)
        entry.keyword_rank = rank
        fused[chunk.id] = entry

    ranked = sorted(fused.values(), key=lambda r: (-r.score, r.chunk.document_id, r.chunk.ordinal))
    return ranked[:k]
