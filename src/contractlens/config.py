"""Application configuration, loaded from environment variables (and a local .env file)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_DIMENSIONS: dict[str, int] = {
    "hashing": 384,
    "voyage-law-2": 1024,
    "voyage-4": 1024,
    "voyage-4-lite": 1024,
    "voyage-4-large": 1024,
    "voyage-finance-2": 1024,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- Service --------------------------------------------------------------
    app_name: str = "ContractLens"
    environment: Literal["development", "test", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    api_key: str | None = Field(default=None, description="If set, requests must send X-API-Key")
    cors_origins: list[str] = ["*"]

    # ---- Storage ----------------------------------------------------------------
    database_url: str | None = Field(default=None, description="postgresql://user:pass@host:5432/db")
    store_backend: Literal["auto", "pgvector", "memory"] = "auto"

    # ---- Providers --------------------------------------------------------------
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    judge_model: str | None = Field(default=None, description="Model for eval judges; defaults to anthropic_model")
    llm_provider: Literal["auto", "anthropic", "fake"] = "auto"

    voyage_api_key: str | None = None
    embedding_provider: Literal["auto", "voyage", "hashing"] = "auto"
    embedding_model: str = "voyage-law-2"
    reranker: Literal["auto", "voyage", "none"] = "auto"
    rerank_model: str = "rerank-2.5"

    # ---- Retrieval & generation ------------------------------------------------------
    chunk_size_chars: int = 1400
    chunk_overlap_chars: int = 200
    retrieval_top_k: int = 8
    retrieval_candidates: int = 24
    rrf_k: int = 60
    max_context_chars: int = 12000
    answer_max_tokens: int = 1024

    # ---- Eval ------------------------------------------------------------------------------
    eval_gold_path: str = "data/gold/questions.jsonl"
    eval_reports_dir: str = "reports"

    # ---- Derived ------------------------------------------------------------------------
    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        return "anthropic" if self.anthropic_api_key else "fake"

    @property
    def resolved_embedding_provider(self) -> str:
        if self.embedding_provider != "auto":
            return self.embedding_provider
        return "voyage" if self.voyage_api_key else "hashing"

    @property
    def resolved_reranker(self) -> str:
        if self.reranker != "auto":
            return self.reranker
        return "voyage" if self.voyage_api_key else "none"

    @property
    def resolved_store_backend(self) -> str:
        if self.store_backend != "auto":
            return self.store_backend
        return "pgvector" if self.database_url else "memory"

    @property
    def embedding_dimension(self) -> int:
        if self.resolved_embedding_provider == "hashing":
            return EMBEDDING_DIMENSIONS["hashing"]
        return EMBEDDING_DIMENSIONS.get(self.embedding_model, 1024)

    @property
    def resolved_judge_model(self) -> str:
        return self.judge_model or self.anthropic_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
