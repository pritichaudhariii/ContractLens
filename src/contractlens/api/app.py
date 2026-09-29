"""FastAPI application: document ingestion, citation-grounded Q&A (JSON and streaming), eval
reports, health and the demo UI."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from contractlens.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkOut,
    DocumentDetail,
    DocumentOut,
    EvalSummary,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    PassageOut,
    StatsResponse,
)
from contractlens.config import Settings, get_settings
from contractlens.container import Container, build_container
from contractlens.models import Answer, Chunk, Document, DocumentType, RetrievedChunk

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
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{elapsed:.1f}"
        if request.url.path not in {"/health", "/", "/favicon.ico"}:
            log.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "ms": round(elapsed, 1),
                    }
                )
            )
        return response

    # ---- ops ----------------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(c: Container = Depends(get_container)) -> HealthResponse:
        docs = await run_in_threadpool(c.store.list_documents)
        chunks = await run_in_threadpool(c.store.count_chunks)
        return HealthResponse(status="ok", version=VERSION, documents=len(docs), chunks=chunks, components=c.describe())

    @app.get("/stats", response_model=StatsResponse, tags=["ops"])
    async def stats(c: Container = Depends(get_container)) -> StatsResponse:
        docs = await run_in_threadpool(c.store.list_documents)
        by_type: dict[str, int] = {}
        sections = 0
        chars = 0
        chunks = 0
        for d in docs:
            by_type[d.doc_type] = by_type.get(d.doc_type, 0) + 1
            doc_chunks = await run_in_threadpool(c.store.document_chunks, d.id)
            chunks += len(doc_chunks)
            chars += sum(len(ch.text) for ch in doc_chunks)
            sections += len({ch.section for ch in doc_chunks})
        return StatsResponse(
            documents=len(docs),
            chunks=chunks,
            by_type=by_type,
            sections=sections,
            avg_chunk_chars=int(chars / chunks) if chunks else 0,
            components=c.describe(),
        )

    @app.get("/graph", response_class=PlainTextResponse, tags=["ops"], summary="The LangGraph workflow as Mermaid")
    async def graph(c: Container = Depends(get_container)) -> str:
        return c.lens.mermaid()

    # ---- documents ----------------------------------------------------------------------

    @app.get("/documents", response_model=list[DocumentOut], tags=["documents"], dependencies=protected)
    async def list_documents(c: Container = Depends(get_container)) -> list[DocumentOut]:
        return [_doc_out(d) for d in await run_in_threadpool(c.store.list_documents)]

    @app.get("/documents/{document_id}", response_model=DocumentDetail, tags=["documents"], dependencies=protected)
    async def get_document(document_id: str, c: Container = Depends(get_container)) -> DocumentDetail:
        doc = await run_in_threadpool(c.store.get_document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        chunks = await run_in_threadpool(c.store.document_chunks, document_id)
        return DocumentDetail(**_doc_out(doc).model_dump(), chunks=[_chunk_out(ch) for ch in chunks])

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

    # ---- q&a ------------------------------------------------------------------------------

    @app.post("/ask", response_model=AskResponse, tags=["qa"], dependencies=protected)
    async def ask(body: AskRequest, c: Container = Depends(get_container)) -> AskResponse:
        answer = await run_in_threadpool(c.lens.ask, body.question, top_k=body.top_k, doc_types=body.doc_types)
        return _ask_response(answer, include_passages=body.include_passages)

    @app.post(
        "/ask/stream",
        tags=["qa"],
        dependencies=protected,
        summary="Ask with the workflow streamed as server-sent events",
        response_class=StreamingResponse,
    )
    async def ask_stream(body: AskRequest, c: Container = Depends(get_container)) -> StreamingResponse:
        def events() -> Iterator[bytes]:
            try:
                for ev in c.lens.ask_stream(body.question, top_k=body.top_k, doc_types=body.doc_types):
                    if ev.type == "node":
                        payload = {"node": ev.node, "elapsed_ms": ev.elapsed_ms, "detail": ev.detail}
                        yield _sse("node", payload)
                    elif ev.answer is not None:
                        yield _sse("answer", _ask_response(ev.answer, include_passages=True).model_dump(mode="json"))
            except Exception as err:  # noqa: BLE001 - report to the client, then log
                log.exception("streamed ask failed")
                yield _sse("error", {"detail": str(err)})

        return StreamingResponse(
            iterate_in_threadpool(events()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- evals --------------------------------------------------------------------------

    @app.get("/evals/latest", response_model=EvalSummary, tags=["evals"])
    async def latest_eval(c: Container = Depends(get_container)) -> EvalSummary:
        path = Path(c.settings.eval_reports_dir) / "latest.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="No eval report yet. Run `contractlens eval`.")
        report = json.loads(path.read_text(encoding="utf-8"))
        misses = [
            {
                k: r[k]
                for k in (
                    "id",
                    "question",
                    "category",
                    "fact_correctness",
                    "faithfulness",
                    "answer_relevance",
                    "retrieval_hit",
                    "answer",
                )
            }
            for r in report.get("results", [])
            if r.get("fact_correctness", 1.0) < 1.0 or r.get("faithfulness", 1.0) < 0.99 or r.get("error")
        ]
        return EvalSummary(**{k: report[k] for k in EvalSummary.model_fields if k in report}, misses=misses[:60])

    @app.get("/evals/history", tags=["evals"])
    async def eval_history(c: Container = Depends(get_container)) -> list[dict]:
        out: list[dict] = []
        for path in sorted(Path(c.settings.eval_reports_dir).glob("eval-*.json")):
            try:
                r = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append(
                {
                    "run_id": r["run_id"],
                    "started_at": r["started_at"],
                    "passed": r["passed"],
                    "profile": r["profile"],
                    "aggregates": r["aggregates"],
                }
            )
        return out[-30:]

    # ---- ui -----------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return resources.files("contractlens.ui").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


# ---- helpers -------------------------------------------------------------------------------


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _ask_response(answer: Answer, *, include_passages: bool) -> AskResponse:
    passages = [_passage_out(i, rc) for i, rc in enumerate(answer.retrieved, start=1)] if include_passages else None
    return AskResponse(
        question=answer.question,
        answer=answer.answer,
        citations=answer.citations,
        grounded=answer.grounded,
        unsupported_markers=answer.unsupported_markers,
        passages=passages,
        trace=answer.trace,
    )


def _passage_out(marker: int, rc: RetrievedChunk) -> PassageOut:
    return PassageOut(
        marker=marker,
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


def _chunk_out(ch: Chunk) -> ChunkOut:
    return ChunkOut(
        id=ch.id,
        ordinal=ch.ordinal,
        section=ch.section,
        page=ch.page,
        text=ch.text,
        char_start=ch.char_start,
        char_end=ch.char_end,
    )


app = create_app()
