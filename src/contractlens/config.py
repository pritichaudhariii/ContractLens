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
    # Azure AI Foundry is the primary provider for generation, embeddings and the eval judge.
    # Endpoint styles: "azure_openai" (https://<resource>.openai.azure.com, deployment names) or
    # "inference" (https://<resource>.services.ai.azure.com/models, any deployed model).
    azure_foundry_endpoint: str | None = None
    azure_foundry_api_key: str | None = None
    azure_foundry_api_style: Literal["azure_openai", "inference"] = "azure_openai"
    azure_foundry_api_version: str = "2024-10-21"
    azure_foundry_chat_deployment: str = "gpt-4o-mini"
    azure_foundry_judge_deployment: str | None = Field(
        default=None, description="Deployment for the eval judge; defaults to the chat deployment"
    )
    azure_foundry_embedding_deployment: str = "text-embedding-3-small"
    azure_foundry_embedding_dimension: int = 1536
    azure_foundry_timeout_s: float = 60.0

    # Alternatives: Anthropic for generation, Voyage for embeddings and reranking.
    anthropic_api_key: str | None = None
    anthropic_model: str | None = None
    judge_model: str | None = Field(default=None, description="Anthropic judge model; defaults to anthropic_model")
    llm_provider: Literal["auto", "foundry", "anthropic", "fake"] = "auto"

    voyage_api_key: str | None = None
    embedding_provider: Literal["auto", "foundry", "voyage", "hashing"] = "auto"
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
    def foundry_configured(self) -> bool:
        return bool(self.azure_foundry_endpoint and self.azure_foundry_api_key)

    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.foundry_configured:
            return "foundry"
        return "anthropic" if self.anthropic_api_key else "fake"

    @property
    def resolved_embedding_provider(self) -> str:
        if self.embedding_provider != "auto":
            return self.embedding_provider
        if self.foundry_configured:
            return "foundry"
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
    def resolved_embedding_model(self) -> str:
        if self.resolved_embedding_provider == "foundry":
            return self.azure_foundry_embedding_deployment
        return self.embedding_model

    @property
    def embedding_dimension(self) -> int:
        provider = self.resolved_embedding_provider
        if provider == "hashing":
            return EMBEDDING_DIMENSIONS["hashing"]
        if provider == "foundry":
            return self.azure_foundry_embedding_dimension
        return EMBEDDING_DIMENSIONS.get(self.embedding_model, 1024)

    @property
    def resolved_llm_model(self) -> str | None:
        if self.resolved_llm_provider == "foundry":
            return self.azure_foundry_chat_deployment
        return self.anthropic_model

    @property
    def resolved_judge_model(self) -> str | None:
        if self.resolved_llm_provider == "foundry":
            return self.azure_foundry_judge_deployment or self.azure_foundry_chat_deployment
        return self.judge_model or self.anthropic_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
