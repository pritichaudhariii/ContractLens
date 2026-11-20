"""Azure AI Foundry HTTP client shared by the Foundry LLM and embedder.

Two endpoint styles are supported:

* ``azure_openai``: Azure OpenAI models in a Foundry resource, addressed by deployment name.
      POST {endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...
      POST {endpoint}/openai/deployments/{deployment}/embeddings?api-version=...
* ``inference``: the Foundry model inference API, for any model deployed in the project
  (OpenAI, Llama, Mistral, Phi, Cohere...), addressed by the ``model`` field in the body.
      POST {endpoint}/chat/completions?api-version=...
      POST {endpoint}/embeddings?api-version=...

Both authenticate with the ``api-key`` header. Transient failures (408/409/429/5xx, timeouts,
connection errors) are retried with exponential backoff, honouring ``Retry-After``.
The client uses only the standard library so it adds no dependency.
"""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class FoundryError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Azure AI Foundry request failed ({status}): {body[:500]}")
        self.status = status
        self.body = body


@dataclass(frozen=True)
class FoundryConfig:
    endpoint: str
    api_key: str
    api_style: Literal["azure_openai", "inference"] = "azure_openai"
    api_version: str = "2024-10-21"
    timeout_s: float = 60.0
    max_retries: int = 4


class FoundryClient:
    def __init__(self, config: FoundryConfig) -> None:
        if not config.endpoint or not config.api_key:
            raise RuntimeError("AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_API_KEY are required for Azure AI Foundry")
        if config.api_style not in ("azure_openai", "inference"):
            raise ValueError(f"Unknown AZURE_FOUNDRY_API_STYLE: {config.api_style}")
        self.config = config
        self.base_url = config.endpoint.rstrip("/")

    def url(self, deployment: str, operation: str) -> str:
        if self.config.api_style == "inference":
            return f"{self.base_url}/{operation}?api-version={self.config.api_version}"
        return f"{self.base_url}/openai/deployments/{deployment}/{operation}?api-version={self.config.api_version}"

    def post(self, deployment: str, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.config.api_style == "inference":
            payload = {**payload, "model": deployment}
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "api-key": self.config.api_key}
        url = self.url(deployment, operation)
        attempt = 0
        while True:
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code not in RETRYABLE_STATUS or attempt >= self.config.max_retries:
                    raise FoundryError(e.code, body) from None
                delay = _backoff(e.headers.get("Retry-After"), attempt)
                log.warning("Azure AI Foundry returned %s; retrying in %.1fs", e.code, delay)
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt >= self.config.max_retries:
                    raise FoundryError(0, str(e)) from None
                delay = _backoff(None, attempt)
                log.warning("Azure AI Foundry connection error (%s); retrying in %.1fs", e, delay)
            time.sleep(delay)
            attempt += 1


def _backoff(retry_after: str | None, attempt: int) -> float:
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return min(0.5 * 2**attempt + random.random() * 0.25, 20.0)
