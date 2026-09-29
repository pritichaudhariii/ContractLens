"""The evaluation harness: run every gold question through the graph, score it, compare the
aggregates to thresholds, write a report, and fail the build when a threshold is missed."""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from contractlens.config import get_settings
from contractlens.container import Container, build_container
from contractlens.evals.gold import GoldQuestion, load_gold
from contractlens.evals.judge import Judge, build_judge
from contractlens.evals.metrics import citation_precision, fact_recall, is_abstention, retrieval_hit, token_f1
from contractlens.models import Answer

log = logging.getLogger(__name__)

METRIC_KEYS = (
    "faithfulness",
    "answer_relevance",
    "fact_correctness",
    "retrieval_hit_rate",
    "section_hit_rate",
    "citation_precision",
    "grounded_rate",
    "abstention_rate",
)


@dataclass
class QuestionResult:
    id: str
    question: str
    category: str
    difficulty: str
    expected_document: str
    answer: str
    citations: int
    grounded: bool
    abstained: bool
    retrieval_hit: bool
    section_hit: bool
    citation_precision: float | None
    fact_correctness: float
    token_f1: float
    faithfulness: float
    answer_relevance: float
    judge_reason: str
    latency_ms: float
    error: str | None = None


@dataclass
class EvalReport:
    run_id: str
    started_at: str
    duration_s: float
    profile: str
    judge: str
    components: dict
    questions: int
    aggregates: dict[str, float]
    thresholds: dict[str, float]
    failures: list[str]
    passed: bool
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    by_document: dict[str, dict[str, float]] = field(default_factory=dict)
    results: list[QuestionResult] = field(default_factory=list)


class EvalHarness:
    def __init__(self, container: Container, judge: Judge, *, thresholds: dict[str, float], profile: str) -> None:
        self.container = container
        self.judge = judge
        self.thresholds = thresholds
        self.profile = profile

    def evaluate_one(self, gold: GoldQuestion) -> QuestionResult:
        started = time.perf_counter()
        try:
            answer = self.container.lens.ask(gold.question)
        except Exception as err:  # noqa: BLE001 - a crash is a failed question, not a failed run
            log.exception("question %s crashed", gold.id)
            return QuestionResult(
                id=gold.id,
                question=gold.question,
                category=gold.category,
                difficulty=gold.difficulty,
                expected_document=gold.source.document_id,
                answer="",
                citations=0,
                grounded=False,
                abstained=False,
                retrieval_hit=False,
                section_hit=False,
                citation_precision=None,
                fact_correctness=0.0,
                token_f1=0.0,
                faithfulness=0.0,
                answer_relevance=0.0,
                judge_reason="",
                latency_ms=0.0,
                error=str(err),
            )
        latency = (time.perf_counter() - started) * 1000
        return self.score(gold, answer, latency)

    def score(self, gold: GoldQuestion, answer: Answer, latency_ms: float) -> QuestionResult:
        doc_hit, section_hit = retrieval_hit(answer.retrieved, gold.source.document_id, gold.source.section)
        judged = self.judge.score(gold, answer)
        return QuestionResult(
            id=gold.id,
            question=gold.question,
            category=gold.category,
            difficulty=gold.difficulty,
            expected_document=gold.source.document_id,
            answer=answer.answer,
            citations=len(answer.citations),
            grounded=answer.grounded,
            abstained=is_abstention(answer.answer),
            retrieval_hit=doc_hit,
            section_hit=section_hit,
            citation_precision=citation_precision(answer, gold.source.document_id),
            fact_correctness=round(fact_recall(gold.expected_facts, answer.answer), 3),
            token_f1=round(token_f1(answer.answer, gold.answer), 3),
            faithfulness=judged.faithfulness,
            answer_relevance=judged.relevance,
            judge_reason=judged.reason,
            latency_ms=round(latency_ms, 1),
        )

    def run(self, questions: list[GoldQuestion], *, progress: bool = True) -> EvalReport:
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()
        results: list[QuestionResult] = []
        for gold in questions:
            result = self.evaluate_one(gold)
            results.append(result)
            if progress:
                mark = "✓" if result.fact_correctness == 1.0 and result.faithfulness >= 0.99 else "·"
                print(
                    f"  {mark} {gold.id:<10} faith={result.faithfulness:.2f} rel={result.answer_relevance:.2f} "
                    f"facts={result.fact_correctness:.2f} hit={'y' if result.retrieval_hit else 'n'} {result.latency_ms:6.0f} ms",
                    flush=True,
                )
        aggregates = aggregate(results)
        failures = [
            f"{key} {aggregates[key]:.3f} < {threshold:.3f}"
            for key, threshold in self.thresholds.items()
            if key in aggregates and aggregates[key] < threshold
        ]
        return EvalReport(
            run_id=started_at.strftime("%Y%m%dT%H%M%SZ"),
            started_at=started_at.isoformat(),
            duration_s=round(time.perf_counter() - t0, 1),
            profile=self.profile,
            judge=self.judge.name,
            components=self.container.describe(),
            questions=len(results),
            aggregates=aggregates,
            thresholds=self.thresholds,
            failures=failures,
            passed=not failures,
            by_category=group_aggregates(results, lambda r: r.category),
            by_document=group_aggregates(results, lambda r: r.expected_document),
            results=results,
        )


def aggregate(results: list[QuestionResult]) -> dict[str, float]:
    if not results:
        return {k: 0.0 for k in METRIC_KEYS}
    n = len(results)
    cited = [r.citation_precision for r in results if r.citation_precision is not None]
    return {
        "faithfulness": round(statistics.fmean(r.faithfulness for r in results), 4),
        "answer_relevance": round(statistics.fmean(r.answer_relevance for r in results), 4),
        "fact_correctness": round(statistics.fmean(r.fact_correctness for r in results), 4),
        "retrieval_hit_rate": round(sum(r.retrieval_hit for r in results) / n, 4),
        "section_hit_rate": round(sum(r.section_hit for r in results) / n, 4),
        "citation_precision": round(statistics.fmean(cited), 4) if cited else 0.0,
        "grounded_rate": round(sum(r.grounded for r in results) / n, 4),
        "abstention_rate": round(sum(r.abstained for r in results) / n, 4),
        "error_rate": round(sum(1 for r in results if r.error) / n, 4),
        "p50_latency_ms": round(statistics.median(r.latency_ms for r in results), 1),
        "p95_latency_ms": round(sorted(r.latency_ms for r in results)[max(0, int(n * 0.95) - 1)], 1),
    }


def group_aggregates(results: list[QuestionResult], key) -> dict[str, dict[str, float]]:
    groups: dict[str, list[QuestionResult]] = {}
    for r in results:
        groups.setdefault(key(r), []).append(r)
    out: dict[str, dict[str, float]] = {}
    for name, rows in sorted(groups.items()):
        agg = aggregate(rows)
        out[name] = {
            "n": len(rows),
            **{k: agg[k] for k in ("faithfulness", "answer_relevance", "fact_correctness", "retrieval_hit_rate")},
        }
    return out


# ---- thresholds & reports -----------------------------------------------------------------------


def load_thresholds(path: str | Path, profile: str) -> dict[str, float]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    profiles = data.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"profile {profile!r} not found in {path}; available: {sorted(profiles)}")
    return {k: float(v) for k, v in profiles[profile].items()}


def choose_profile(container: Container) -> str:
    """Production thresholds apply when a real model answers; offline thresholds for the fake."""
    return "production" if container.llm.name != "fake" else "offline"


def write_reports(report: EvalReport, reports_dir: str | Path) -> tuple[Path, Path]:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"eval-{report.run_id}.json"
    json_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = out / "latest.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    (out / "latest.json").write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path, md_path


def render_markdown(report: EvalReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"# Eval report {report.run_id} — {status}",
        "",
        f"Profile: `{report.profile}` · Judge: `{report.judge}` · Questions: {report.questions} · Duration: {report.duration_s}s",
        "",
        "Components: " + ", ".join(f"{k}={v}" for k, v in report.components.items()),
        "",
        "| Metric | Value | Threshold | Status |",
        "| --- | ---: | ---: | :---: |",
    ]
    for key, value in report.aggregates.items():
        threshold = report.thresholds.get(key)
        if threshold is None:
            lines.append(f"| {key} | {value} | – | |")
        else:
            lines.append(f"| {key} | {value} | {threshold} | {'✅' if value >= threshold else '❌'} |")
    if report.failures:
        lines += ["", "Failures:", *[f"- {f}" for f in report.failures]]
    lines += [
        "",
        "## By category",
        "",
        "| Category | n | Faithfulness | Relevance | Facts | Retrieval hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, agg in report.by_category.items():
        lines.append(
            f"| {name} | {agg['n']} | {agg['faithfulness']} | {agg['answer_relevance']} | {agg['fact_correctness']} | {agg['retrieval_hit_rate']} |"
        )
    lines += [
        "",
        "## By document",
        "",
        "| Document | n | Faithfulness | Relevance | Facts | Retrieval hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, agg in report.by_document.items():
        lines.append(
            f"| {name} | {agg['n']} | {agg['faithfulness']} | {agg['answer_relevance']} | {agg['fact_correctness']} | {agg['retrieval_hit_rate']} |"
        )
    misses = [r for r in report.results if r.fact_correctness < 1.0 or r.faithfulness < 0.99 or r.error]
    if misses:
        lines += ["", f"## Questions needing attention ({len(misses)})", ""]
        for r in misses[:40]:
            why = (
                r.error
                or f"facts={r.fact_correctness} faith={r.faithfulness} rel={r.answer_relevance} hit={r.retrieval_hit}"
            )
            lines.append(f"- **{r.id}** {r.question}  \n  {why}  \n  _{r.answer[:220].replace(chr(10), ' ')}_")
    return "\n".join(lines) + "\n"


# ---- CLI entry -----------------------------------------------------------------------------------


def run_from_args(args: argparse.Namespace) -> int:
    settings = get_settings()
    embedded = None
    if getattr(args, "embedded_postgres", False):
        embedded = start_embedded_postgres()
        os.environ["DATABASE_URL"] = embedded.get_uri()
        get_settings.cache_clear()
        settings = get_settings()

    container = build_container(settings)
    try:
        if args.ingest:
            ingest_folder(container, Path(args.ingest))
        if container.store.count_chunks() == 0:
            print("The store is empty. Pass --ingest data/corpus (or ingest first).", file=sys.stderr)
            return 2

        judge_kind = args.judge
        if judge_kind == "auto":
            judge_kind = "llm" if container.llm.name != "fake" else "heuristic"
        judge = build_judge(judge_kind, container.llm)
        profile = choose_profile(container)
        thresholds = load_thresholds(args.thresholds, profile)

        gold_path = args.gold or settings.eval_gold_path
        questions = load_gold(gold_path)
        if args.limit:
            questions = questions[: args.limit]

        print(f"ContractLens eval — {len(questions)} questions from {gold_path}")
        print(f"  components: {container.describe()}")
        print(f"  judge: {judge.name}  profile: {profile}  thresholds: {thresholds}\n")

        harness = EvalHarness(container, judge, thresholds=thresholds, profile=profile)
        report = harness.run(questions)
        json_path, md_path = write_reports(report, args.reports_dir or settings.eval_reports_dir)

        print()
        for key, value in report.aggregates.items():
            threshold = thresholds.get(key)
            flag = "" if threshold is None else ("  ✅" if value >= threshold else f"  ❌ (< {threshold})")
            print(f"  {key:<20} {value:>8}{flag}")
        print(f"\n  report: {json_path}\n  summary: {md_path}")
        if report.passed:
            print("\nRESULT: PASS — all release thresholds met")
            return 0
        print("\nRESULT: FAIL — " + "; ".join(report.failures))
        return 0 if args.no_gate else 1
    finally:
        container.close()
        if embedded is not None:
            embedded.cleanup()


def ingest_folder(container: Container, folder: Path) -> None:
    from contractlens.cli import infer_doc_type

    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
    for p in files:
        result = container.ingestor.ingest_bytes(p.read_bytes(), filename=p.name, doc_type=infer_doc_type(p, None))
        log.info("ingested %s (%d passages)", result.document.id, result.chunks)
    print(f"ingested {len(files)} documents, {container.store.count_chunks()} passages into {container.store.name}")


def start_embedded_postgres():
    """Starts a throwaway PostgreSQL 16 + pgvector via the `pgserver` package (dev dependency)."""
    import tempfile

    import pgserver

    return pgserver.get_server(tempfile.mkdtemp(prefix="contractlens-pg-"))
