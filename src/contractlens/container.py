"""Wires configuration into concrete components. One place to construct the store, embedder,
LLM, reranker, ingestor and the compiled graph so the API, CLI and eval harness share it."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from contractlens.config import Settings, get_settings
from contractlens.embeddings import Embedder, build_embedder
from contractlens.graph import ContractLens, GraphDeps
from contractlens.ingest.service import Ingestor
from contractlens.llm import LLM, build_llm
from contractlens.rerank import Reranker, build_reranker
from contractlens.store.base import DocumentStore
from contractlens.store.memory import InMemoryStore
from contractlens.store.pgvector import PgVectorStore

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    store: DocumentStore
    embedder: Embedder
    llm: LLM
    reranker: Reranker
    ingestor: Ingestor
    lens: ContractLens

    def describe(self) -> dict[str, str | int]:
        return {
            "store": self.store.name,
            "embedding_provider": self.embedder.name,
            "embedding_dimension": self.embedder.dimension,
            "llm": f"{self.llm.name}:{self.llm.model}",
            "reranker": self.reranker.name,
            "environment": self.settings.environment,
        }

    def close(self) -> None:
        self.store.close()


def build_container(settings: Settings | None = None, *, store: DocumentStore | None = None) -> Container:
    settings = settings or get_settings()

    embedder = build_embedder(
        settings.resolved_embedding_provider,
        api_key=settings.voyage_api_key,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )
    if store is None:
        if settings.resolved_store_backend == "pgvector":
            assert settings.database_url, "DATABASE_URL is required for the pgvector store"
            store = PgVectorStore(settings.database_url, dimension=embedder.dimension)
        else:
            store = InMemoryStore()
    llm = build_llm(settings.resolved_llm_provider, api_key=settings.anthropic_api_key, model=settings.anthropic_model)
    reranker = build_reranker(settings.resolved_reranker, api_key=settings.voyage_api_key, model=settings.rerank_model)
    ingestor = Ingestor(store, embedder, chunk_size=settings.chunk_size_chars, overlap=settings.chunk_overlap_chars)
    lens = ContractLens(
        GraphDeps(
            store=store,
            embedder=embedder,
            llm=llm,
            reranker=reranker,
            top_k=settings.retrieval_top_k,
            candidates=settings.retrieval_candidates,
            rrf_k=settings.rrf_k,
            max_context_chars=settings.max_context_chars,
            answer_max_tokens=settings.answer_max_tokens,
        )
    )
    container = Container(settings, store, embedder, llm, reranker, ingestor, lens)
    log.info("ContractLens components: %s", container.describe())
    return container
