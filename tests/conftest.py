"""Shared fixtures: an offline container (memory store, hashing embeddings, fake model) and,
for integration tests, a throwaway PostgreSQL 16 + pgvector started by the `pgserver` package."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pytest

from contractlens.config import Settings
from contractlens.container import Container, build_container
from contractlens.evals.harness import ingest_folder

warnings.filterwarnings("ignore", message="XDG_RUNTIME_DIR")

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
GOLD = Path(__file__).resolve().parents[1] / "data" / "gold" / "questions.jsonl"


def offline_settings(**overrides) -> Settings:
    values = dict(
        environment="test",
        llm_provider="fake",
        embedding_provider="hashing",
        reranker="none",
        store_backend="memory",
        anthropic_api_key=None,
        voyage_api_key=None,
        database_url=None,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture(scope="session")
def corpus_container() -> Container:
    container = build_container(offline_settings())
    ingest_folder(container, CORPUS)
    yield container
    container.close()


@pytest.fixture(scope="session")
def embedded_pg_url() -> str:
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tempfile.mkdtemp(prefix="contractlens-test-pg-"))
    yield server.get_uri()
    server.cleanup()
