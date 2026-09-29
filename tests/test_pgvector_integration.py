"""Integration tests against a real PostgreSQL 16 + pgvector (embedded via `pgserver`)."""

from contractlens.container import build_container
from contractlens.evals.harness import ingest_folder
from contractlens.retrieval import HybridRetriever
from contractlens.store.pgvector import PgVectorStore
from tests.conftest import CORPUS, offline_settings


def test_pgvector_hybrid_search_and_graph(embedded_pg_url):
    settings = offline_settings(store_backend="pgvector", database_url=embedded_pg_url)
    container = build_container(settings)
    try:
        assert isinstance(container.store, PgVectorStore)
        ingest_folder(container, CORPUS)
        assert container.store.count_chunks() > 100

        retriever = HybridRetriever(container.store, container.embedder)
        hits = retriever.retrieve("liability cap", k=5)
        assert hits and any("Liability" in h.chunk.section for h in hits)
        assert any(h.keyword_rank is not None for h in hits) and any(h.vector_rank is not None for h in hits)

        # Filters push down to SQL
        only_filings = container.store.keyword_search("revenue", 10, doc_types=["regulatory_filing"])
        assert only_filings and all(c.doc_type == "regulatory_filing" for c, _ in only_filings)
        one_doc = container.store.vector_search(
            container.embedder.embed_query("rent"), 10, document_ids=["office-lease-agreement-riverbend-northwind"]
        )
        assert one_doc and all(c.document_id == "office-lease-agreement-riverbend-northwind" for c, _ in one_doc)

        answer = container.lens.ask("What is the maximum leverage ratio permitted under the credit agreement?")
        assert "3.50" in answer.answer and answer.grounded

        # Re-ingest is idempotent (upsert replaces chunks)
        before = container.store.count_chunks()
        ingest_folder(container, CORPUS)
        assert container.store.count_chunks() == before
        assert container.store.delete_document("credit-agreement-northwind-first-meridian")
        assert container.store.count_chunks() < before
    finally:
        container.close()


def test_dimension_mismatch_is_detected(embedded_pg_url):
    store = PgVectorStore(embedded_pg_url, dimension=384)
    store.close()
    import pytest

    with pytest.raises(RuntimeError, match="dimensional"):
        PgVectorStore(embedded_pg_url, dimension=1024)
