"""Embedding providers behind one interface.

`FoundryEmbedder` is the production choice (an embedding deployment in Azure AI Foundry,
e.g. text-embedding-3-small). `VoyageEmbedder` is an alternative (voyage-law-2 is trained
for legal text).
`HashingEmbedder` is a dependency-free, deterministic fallback used in tests, CI and
offline development: it hashes word and character n-grams into a fixed-size vector,
which is good enough to make hybrid retrieval meaningful without any model download.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

from contractlens.foundry import FoundryClient

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


class Embedder(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder(Embedder):
    """Feature-hashing embedder: unigrams, bigrams and character trigrams, L2-normalised."""

    name = "hashing"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        tokens = TOKEN.findall(text.lower())
        features: list[tuple[str, float]] = []
        features.extend((f"w:{t}", 1.0) for t in tokens)
        features.extend((f"b:{a}_{b}", 0.8) for a, b in zip(tokens, tokens[1:], strict=False))
        for t in tokens:
            if len(t) >= 5:
                padded = f"#{t}#"
                features.extend((f"c:{padded[i : i + 3]}", 0.3) for i in range(len(padded) - 2))
        for feat, weight in features:
            h = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dimension
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FoundryEmbedder(Embedder):
    """Embeddings from an Azure AI Foundry deployment, batched, returned in input order."""

    name = "foundry"

    def __init__(self, client: FoundryClient, deployment: str, dimension: int = 1536, batch_size: int = 64) -> None:
        self.client = client
        self.model = deployment
        self.dimension = dimension
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed(texts[i : i + self.batch_size]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, batch: list[str]) -> list[list[float]]:
        payload: dict = {"input": batch}
        # text-embedding-3 models can return shortened vectors; ask for the configured size.
        if self.model.startswith("text-embedding-3"):
            payload["dimensions"] = self.dimension
        out = self.client.post(self.model, "embeddings", payload)
        rows = sorted(out["data"], key=lambda r: r["index"])
        vectors = [r["embedding"] for r in rows]
        if vectors and len(vectors[0]) != self.dimension:
            raise RuntimeError(
                f"Embedding deployment {self.model} returned {len(vectors[0])} dimensions; "
                f"set AZURE_FOUNDRY_EMBEDDING_DIMENSION={len(vectors[0])}"
            )
        return vectors


class VoyageEmbedder(Embedder):
    name = "voyage"

    def __init__(self, api_key: str, model: str = "voyage-law-2", dimension: int = 1024, batch_size: int = 64) -> None:
        import voyageai

        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            out.extend(self.client.embed(batch, model=self.model, input_type="document").embeddings)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed([text], model=self.model, input_type="query").embeddings[0]


def build_embedder(
    provider: str,
    *,
    api_key: str | None = None,
    model: str,
    dimension: int,
    foundry: FoundryClient | None = None,
) -> Embedder:
    if provider == "foundry":
        if foundry is None:
            raise RuntimeError("AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_API_KEY are required for Foundry embeddings")
        return FoundryEmbedder(foundry, deployment=model, dimension=dimension)
    if provider == "voyage":
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY is required for the voyage embedding provider")
        return VoyageEmbedder(api_key, model=model, dimension=dimension)
    if provider == "hashing":
        return HashingEmbedder(dimension=dimension)
    raise ValueError(f"Unknown embedding provider: {provider}")
