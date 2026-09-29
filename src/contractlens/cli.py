"""Command-line entry points: ingest a folder, ask a question, serve the API, run the eval harness."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from contractlens.config import get_settings
from contractlens.container import build_container
from contractlens.models import DocumentType

DOC_TYPE_BY_PREFIX: dict[str, DocumentType] = {
    "contract": "contract",
    "filing": "regulatory_filing",
    "policy": "policy",
}


def infer_doc_type(path: Path, explicit: str | None) -> DocumentType:
    """`contract_msa.md` → contract, `filing_10k.md` → regulatory_filing; otherwise the flag or 'other'."""
    if explicit:
        return explicit  # type: ignore[return-value]
    for prefix, doc_type in DOC_TYPE_BY_PREFIX.items():
        if path.name.lower().startswith(prefix):
            return doc_type
    return "other"


def cmd_ingest(args: argparse.Namespace) -> int:
    container = build_container()
    paths: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(x for x in p.iterdir() if x.suffix.lower() in {".md", ".txt", ".pdf"}))
        elif p.exists():
            paths.append(p)
        else:
            print(f"skip: {p} does not exist", file=sys.stderr)
    if not paths:
        print("nothing to ingest", file=sys.stderr)
        return 1
    total = 0
    for p in paths:
        result = container.ingestor.ingest_bytes(
            p.read_bytes(), filename=p.name, doc_type=infer_doc_type(p, args.doc_type)
        )
        total += result.chunks
        print(f"{result.document.id:<40} {result.document.doc_type:<18} {result.chunks:>4} passages  ({p.name})")
    print(f"\n{len(paths)} documents, {total} passages → {container.store.name}")
    container.close()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    container = build_container()
    answer = container.lens.ask(args.question, top_k=args.top_k)
    if args.json:
        print(answer.model_dump_json(indent=2, exclude={"retrieved"}))
    else:
        print(answer.answer)
        print()
        for c in answer.citations:
            flag = "" if c.supported else "  (unsupported)"
            where = f"{c.document_title} — {c.section}" + (f", p. {c.page}" if c.page else "")
            print(f"[{c.marker}] {where}{flag}\n    “{c.quote}”")
        print(f"\ngrounded={answer.grounded}  {json.dumps(answer.trace)}")
    container.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run("contractlens.api.app:app", host=settings.host, port=args.port or settings.port, reload=args.reload)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from contractlens.evals.harness import run_from_args

    return run_from_args(args)


def cmd_graph(_: argparse.Namespace) -> int:
    from contractlens.embeddings import HashingEmbedder
    from contractlens.graph import ContractLens, GraphDeps
    from contractlens.llm import FakeLLM
    from contractlens.rerank import NoopReranker
    from contractlens.store.memory import InMemoryStore

    print(ContractLens(GraphDeps(InMemoryStore(), HashingEmbedder(), FakeLLM(), NoopReranker())).mermaid())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contractlens", description="ContractLens: citation-grounded document Q&A")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Index files or folders (PDF, .md, .txt)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--doc-type", choices=["contract", "regulatory_filing", "policy", "other"], default=None)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="Ask one question from the terminal")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("serve", help="Run the HTTP API")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="Run the evaluation harness against the gold dataset")
    p.add_argument("--gold", default=None, help="Path to the gold JSONL (default from settings)")
    p.add_argument("--reports-dir", default=None)
    p.add_argument("--thresholds", default="evals/thresholds.yaml")
    p.add_argument("--judge", choices=["auto", "llm", "heuristic"], default="auto")
    p.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    p.add_argument("--ingest", default=None, help="Ingest this folder into the store before evaluating")
    p.add_argument("--no-gate", action="store_true", help="Report only; never exit non-zero")
    p.add_argument(
        "--embedded-postgres",
        action="store_true",
        help="Run against a throwaway PostgreSQL+pgvector (needs the pgserver dev dependency)",
    )
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("graph", help="Print the LangGraph workflow as Mermaid")
    p.set_defaults(func=cmd_graph)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
