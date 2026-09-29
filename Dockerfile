# syntax=docker/dockerfile:1.7
# ContractLens API image. Multi-stage: build wheels, then a slim runtime that runs as non-root.
#   docker build -t contractlens .
#   docker run -p 8000:8000 -e DATABASE_URL=... -e ANTHROPIC_API_KEY=... contractlens

FROM python:3.12-slim AS build
WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-deps --wheel-dir /wheels . \
    && pip wheel --wheel-dir /wheels .

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    PORT=8000 ENVIRONMENT=production LOG_LEVEL=INFO
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links=/wheels contractlens && rm -rf /wheels
# Sample corpus, gold set and thresholds ship in the image so `contractlens eval` and the
# first-run ingest work in any environment.
COPY --chown=app:app data ./data
COPY --chown=app:app evals ./evals
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1
CMD ["sh", "-c", "uvicorn contractlens.api.app:app --host 0.0.0.0 --port ${PORT} --workers 2 --timeout-keep-alive 30"]
