"""FastAPI application: document ingestion, citation-grounded Q&A, health and the demo UI."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from importlib.metadata import PackageNotFoundError, version

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from contractlens.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentOut,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    PassageOut,
)
from contractlens.config import Settings, get_settings
from contractlens.container import Container, build_container
from contractlens.models import Document, DocumentType

log = logging.getLogger(__name__)

try:
    VERSION = version("contractlens")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    VERSION = "0.0.0"


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or build_container(settings)
        try:
            yield
        finally:
            app.state.container.close()

    app = FastAPI(
        title="ContractLens",
        version=VERSION,
        description="Citation-grounded Q&A over contracts and regulatory filings.",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])

    def get_container(request: Request) -> Container:
        return request.app.state.container

    def require_api_key(request: Request) -> None:
        if settings.api_key and request.headers.get("x-api-key") != settings.api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key")

    protected = [Depends(require_api_key)]

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        if request.url.path not in {"/health", "/"}:
            log.info("%s %s -> %s in %.1f ms", request.method, request.url.path, response.status_code, elapsed)
        response.headers["x-response-time-ms"] = f"{elapsed:.1f}"
        return response

    # ---- health ------------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(c: Container = Depends(get_container)) -> HealthResponse:
        docs = await run_in_threadpool(c.store.list_documents)
        chunks = await run_in_threadpool(c.store.count_chunks)
        return HealthResponse(status="ok", version=VERSION, documents=len(docs), chunks=chunks, components=c.describe())

    @app.get("/graph", response_class=PlainTextResponse, tags=["ops"], summary="The LangGraph workflow as Mermaid")
    async def graph(c: Container = Depends(get_container)) -> str:
        return c.lens.mermaid()

    # ---- documents -------------------------------------------------------------------

    @app.get("/documents", response_model=list[DocumentOut], tags=["documents"], dependencies=protected)
    async def list_documents(c: Container = Depends(get_container)) -> list[DocumentOut]:
        return [_doc_out(d) for d in await run_in_threadpool(c.store.list_documents)]

    @app.post(
        "/documents",
        response_model=IngestResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
        dependencies=protected,
        summary="Upload a PDF, text or markdown file",
    )
    async def upload_document(
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        doc_type: DocumentType = Form(default="other"),
        document_id: str | None = Form(default=None),
        c: Container = Depends(get_container),
    ) -> IngestResponse:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        try:
            result = await run_in_threadpool(
                c.ingestor.ingest_bytes,
                data,
                filename=file.filename or "upload",
                title=title,
                doc_type=doc_type,
                document_id=document_id,
            )
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        return IngestResponse(document=_doc_out(result.document), chunks=result.chunks)

    @app.post(
        "/documents/text",
        response_model=IngestResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
        dependencies=protected,
        summary="Ingest raw text",
    )
    async def ingest_text(body: IngestTextRequest, c: Container = Depends(get_container)) -> IngestResponse:
        try:
            result = await run_in_threadpool(
                c.ingestor.ingest_text,
                body.text,
                source="api",
                title=body.title,
                doc_type=body.doc_type,
                document_id=body.document_id,
                metadata=body.metadata,
            )
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        return IngestResponse(document=_doc_out(result.document), chunks=result.chunks)

    @app.delete(
        "/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["documents"], dependencies=protected
    )
    async def delete_document(document_id: str, c: Container = Depends(get_container)) -> None:
        if not await run_in_threadpool(c.store.delete_document, document_id):
            raise HTTPException(status_code=404, detail="Document not found")

    # ---- q&a --------------------------------------------------------------------------------

    @app.post("/ask", response_model=AskResponse, tags=["qa"], dependencies=protected)
    async def ask(body: AskRequest, c: Container = Depends(get_container)) -> AskResponse:
        answer = await run_in_threadpool(c.lens.ask, body.question, top_k=body.top_k, doc_types=body.doc_types)
        passages = None
        if body.include_passages:
            passages = [
                PassageOut(
                    marker=i,
                    chunk_id=rc.chunk.id,
                    document_id=rc.chunk.document_id,
                    document_title=rc.chunk.document_title,
                    section=rc.chunk.section,
                    page=rc.chunk.page,
                    score=round(rc.score, 5),
                    vector_rank=rc.vector_rank,
                    keyword_rank=rc.keyword_rank,
                    text=rc.chunk.text,
                )
                for i, rc in enumerate(answer.retrieved, start=1)
            ]
        return AskResponse(
            question=answer.question,
            answer=answer.answer,
            citations=answer.citations,
            grounded=answer.grounded,
            unsupported_markers=answer.unsupported_markers,
            passages=passages,
            trace=answer.trace,
        )

    # ---- ui ----------------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return resources.files("contractlens.ui").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


def _doc_out(d: Document) -> DocumentOut:
    return DocumentOut(
        id=d.id,
        title=d.title,
        doc_type=d.doc_type,
        source=d.source,
        chunk_count=d.chunk_count,
        created_at=d.created_at,
        metadata=d.metadata,
    )


app = create_app()
