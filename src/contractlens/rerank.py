"""Rerankers used in the context-processing step of the graph."""

from __future__ import annotations

from abc import ABC, abstractmethod

from contractlens.models import RetrievedChunk


class Reranker(ABC):
    name: str

    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


class NoopReranker(Reranker):
    """Keeps the fused retrieval order."""

    name = "none"

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return candidates[:top_k]


class VoyageReranker(Reranker):
    name = "voyage"

    def __init__(self, api_key: str, model: str = "rerank-2.5") -> None:
        import voyageai

        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        result = self.client.rerank(query, [c.chunk.text for c in candidates], model=self.model, top_k=top_k)
        out: list[RetrievedChunk] = []
        for item in result.results:
            rc = candidates[item.index].model_copy(update={"rerank_score": float(item.relevance_score)})
            out.append(rc)
        return out


def build_reranker(kind: str, *, api_key: str | None, model: str) -> Reranker:
    if kind == "voyage":
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY is required for the voyage reranker")
        return VoyageReranker(api_key, model=model)
    return NoopReranker()
