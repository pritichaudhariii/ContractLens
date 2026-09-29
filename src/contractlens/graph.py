"""The ContractLens workflow as a LangGraph state machine.

    START ─▶ analyze_query ─▶ retrieve ─▶ process_context ─▶ generate ─▶ ground_check ─▶ END
                                   │                                          │
                                   └──(no candidates)──▶ no_context ─▶ END      └──(unsupported citations,
                                                                                    attempts left)──▶ generate

Each node is a plain function over a typed state so it can be unit-tested on its own; the
graph wiring lives in `build_graph`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from contractlens.embeddings import Embedder
from contractlens.grounding import check_grounding
from contractlens.llm import LLM
from contractlens.models import Answer, Citation, RetrievedChunk
from contractlens.rerank import Reranker
from contractlens.retrieval import HybridRetriever
from contractlens.scope import scope_documents
from contractlens.store.base import DocumentStore

SYSTEM_PROMPT = """You are ContractLens, an analyst that answers questions about contracts and regulatory filings.

Rules:
- Answer only from the numbered CONTEXT passages. Do not use outside knowledge.
- Cite every factual statement with the passage number in square brackets, e.g. "The notice period is 30 days [2]." Use one or more markers per sentence; never invent a marker that is not in the context.
- Quote amounts, dates, durations and party names exactly as written in the passages.
- If the passages do not contain the answer, say so plainly in one sentence and do not guess.
- Be concise: a direct answer first, then only the qualifications the documents actually state."""

REVISION_NOTE = """Your previous draft cited passages that do not support the sentences they were attached to: markers {markers}.
Rewrite the answer so that every sentence is supported by the passage it cites, or drop the unsupported sentences.

PREVIOUS DRAFT:
{draft}
"""


class GraphState(TypedDict, total=False):
    question: str
    doc_types: list[str] | None
    document_ids: list[str] | None
    top_k: int
    search_query: str
    candidates: list[RetrievedChunk]
    context: list[RetrievedChunk]
    draft: str
    citations: list[Citation]
    unsupported: list[int]
    grounded: bool
    attempts: int
    trace: dict[str, float | int | str]


@dataclass
class GraphDeps:
    store: DocumentStore
    embedder: Embedder
    llm: LLM
    reranker: Reranker
    top_k: int = 8
    candidates: int = 24
    rrf_k: int = 60
    max_context_chars: int = 12000
    answer_max_tokens: int = 1024
    max_revisions: int = 1
    grounding_threshold: float = 0.3


def build_graph(deps: GraphDeps):
    retriever = HybridRetriever(deps.store, deps.embedder, candidates=deps.candidates, rrf_k=deps.rrf_k)

    # ---- nodes ----------------------------------------------------------------------

    def analyze_query(state: GraphState) -> GraphState:
        question = state["question"].strip()
        # Callers may filter by document type explicitly; the question itself decides document scope.
        doc_types = state.get("doc_types") or None
        document_ids = state.get("document_ids") or None
        if not document_ids:
            document_ids = scope_documents(question, deps.store.list_documents()).document_ids
        trace: dict[str, float | int | str] = {"scoped_document": document_ids[0] if document_ids else ""}
        return {
            "search_query": question,
            "doc_types": doc_types,
            "document_ids": document_ids,
            "attempts": 0,
            "trace": trace,
        }

    def retrieve(state: GraphState) -> GraphState:
        started = time.perf_counter()
        k = state.get("top_k") or deps.top_k
        n = max(k, deps.candidates)
        query = state["search_query"]
        document_ids = state.get("document_ids")
        candidates = retriever.retrieve(query, k=n, doc_types=state.get("doc_types"), document_ids=document_ids)
        if document_ids and len(candidates) < k:
            # Scoped retrieval came up short (small document); top up from the whole corpus.
            seen = {rc.chunk.id for rc in candidates}
            extra = retriever.retrieve(query, k=n, doc_types=state.get("doc_types"))
            candidates.extend(rc for rc in extra if rc.chunk.id not in seen)
        if not candidates and state.get("doc_types"):
            # The doc-type hint was wrong; widen the search before giving up.
            candidates = retriever.retrieve(query, k=n, doc_types=None)
        trace = dict(state.get("trace", {}))
        trace["retrieve_ms"] = round((time.perf_counter() - started) * 1000, 1)
        trace["candidates"] = len(candidates)
        return {"candidates": candidates, "trace": trace}

    def process_context(state: GraphState) -> GraphState:
        started = time.perf_counter()
        k = state.get("top_k") or deps.top_k
        ranked = deps.reranker.rerank(state["search_query"], state["candidates"], top_k=k)
        context: list[RetrievedChunk] = []
        used = 0
        seen_text: set[str] = set()
        for rc in ranked:
            fingerprint = rc.chunk.text[:200]
            if fingerprint in seen_text:
                continue
            if used + len(rc.chunk.text) > deps.max_context_chars and context:
                break
            seen_text.add(fingerprint)
            context.append(rc)
            used += len(rc.chunk.text)
        trace = dict(state.get("trace", {}))
        trace["context_chunks"] = len(context)
        trace["context_chars"] = used
        trace["rerank"] = deps.reranker.name
        trace["process_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return {"context": context, "trace": trace}

    def generate(state: GraphState) -> GraphState:
        started = time.perf_counter()
        prompt = _build_prompt(state["question"], state["context"])
        if state.get("attempts", 0) > 0 and state.get("draft"):
            prompt = (
                REVISION_NOTE.format(markers=", ".join(map(str, state.get("unsupported", []))), draft=state["draft"])
                + "\n"
                + prompt
            )
        draft = deps.llm.complete(SYSTEM_PROMPT, prompt, max_tokens=deps.answer_max_tokens)
        trace = dict(state.get("trace", {}))
        trace["generate_ms"] = round(trace.get("generate_ms", 0) + (time.perf_counter() - started) * 1000, 1)
        trace["llm"] = f"{deps.llm.name}:{deps.llm.model}"
        return {"draft": draft, "attempts": state.get("attempts", 0) + 1, "trace": trace}

    def ground_check(state: GraphState) -> GraphState:
        result = check_grounding(state["draft"], state["context"], threshold=deps.grounding_threshold)
        trace = dict(state.get("trace", {}))
        trace["citations"] = len(result.citations)
        trace["unsupported"] = len(result.unsupported)
        trace["attempts"] = state.get("attempts", 0)
        return {
            "citations": result.citations,
            "unsupported": result.unsupported,
            "grounded": result.grounded,
            "trace": trace,
        }

    def no_context(state: GraphState) -> GraphState:
        trace = dict(state.get("trace", {}))
        trace["citations"] = 0
        return {
            "context": [],
            "draft": "I couldn't find any passage in the indexed documents that relates to this question.",
            "citations": [],
            "unsupported": [],
            "grounded": True,
            "trace": trace,
        }

    # ---- routing --------------------------------------------------------------------

    def after_retrieve(state: GraphState) -> Literal["process_context", "no_context"]:
        return "process_context" if state.get("candidates") else "no_context"

    def after_ground_check(state: GraphState) -> Literal["generate", "__end__"]:
        if state.get("unsupported") and state.get("attempts", 0) <= deps.max_revisions:
            return "generate"
        return END

    graph = StateGraph(GraphState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("process_context", process_context)
    graph.add_node("generate", generate)
    graph.add_node("ground_check", ground_check)
    graph.add_node("no_context", no_context)

    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "retrieve")
    graph.add_conditional_edges(
        "retrieve", after_retrieve, {"process_context": "process_context", "no_context": "no_context"}
    )
    graph.add_edge("process_context", "generate")
    graph.add_edge("generate", "ground_check")
    graph.add_conditional_edges("ground_check", after_ground_check, {"generate": "generate", END: END})
    graph.add_edge("no_context", END)
    return graph.compile()


def _build_prompt(question: str, context: list[RetrievedChunk]) -> str:
    blocks = []
    for i, rc in enumerate(context, start=1):
        c = rc.chunk
        label = c.document_title + (f" — {c.section}" if c.section else "") + (f", p. {c.page}" if c.page else "")
        blocks.append(f"[{i}] ({label})\n{c.text}")
    return "CONTEXT:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}\n"


@dataclass
class GraphEvent:
    """One step of a streamed run: which node finished, when, and a small human-readable detail."""

    type: Literal["node", "answer"]
    node: str
    elapsed_ms: float
    detail: dict[str, float | int | str | bool] = field(default_factory=dict)
    answer: Answer | None = None


def _node_detail(node: str, state: dict) -> dict[str, float | int | str | bool]:
    trace = state.get("trace", {}) or {}
    if node == "analyze_query":
        return {"scoped_document": trace.get("scoped_document", "") or ""}
    if node == "retrieve":
        return {"candidates": trace.get("candidates", 0), "ms": trace.get("retrieve_ms", 0)}
    if node == "process_context":
        return {
            "passages": trace.get("context_chunks", 0),
            "chars": trace.get("context_chars", 0),
            "rerank": trace.get("rerank", ""),
        }
    if node == "generate":
        return {"attempt": state.get("attempts", 0), "model": trace.get("llm", ""), "ms": trace.get("generate_ms", 0)}
    if node == "ground_check":
        return {
            "citations": trace.get("citations", 0),
            "unsupported": trace.get("unsupported", 0),
            "grounded": bool(state.get("grounded", False)),
        }
    if node == "no_context":
        return {"abstained": True}
    return {}


class ContractLens:
    """Convenience wrapper that runs the compiled graph and shapes the result as an `Answer`."""

    def __init__(self, deps: GraphDeps) -> None:
        self.deps = deps
        self.graph = build_graph(deps)

    def ask(self, question: str, *, top_k: int | None = None, doc_types: list[str] | None = None) -> Answer:
        started = time.perf_counter()
        state: GraphState = {"question": question, "top_k": top_k or self.deps.top_k, "doc_types": doc_types}
        final = self.graph.invoke(state)
        return self._to_answer(question, final, started)

    def ask_stream(
        self, question: str, *, top_k: int | None = None, doc_types: list[str] | None = None
    ) -> Iterator[GraphEvent]:
        """Runs the graph node by node, yielding a `GraphEvent` after each node completes and a
        final `answer` event. This is what the SSE endpoint and the UI's live trace consume."""
        started = time.perf_counter()
        state: GraphState = {"question": question, "top_k": top_k or self.deps.top_k, "doc_types": doc_types}
        merged: dict = dict(state)
        for update in self.graph.stream(state, stream_mode="updates"):
            for node, delta in update.items():
                merged.update(delta or {})
                yield GraphEvent(
                    type="node",
                    node=node,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                    detail=_node_detail(node, merged),
                )
        yield GraphEvent(
            type="answer",
            node="",
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            answer=self._to_answer(question, merged, started),
        )

    def _to_answer(self, question: str, final: dict, started: float) -> Answer:
        trace = dict(final.get("trace", {}))
        trace["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return Answer(
            question=question,
            answer=final.get("draft", ""),
            citations=final.get("citations", []),
            grounded=bool(final.get("grounded", False)),
            unsupported_markers=final.get("unsupported", []),
            retrieved=final.get("context", []),
            trace=trace,
        )

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
