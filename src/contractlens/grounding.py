"""Grounding utilities: parse citation markers from an answer and check each against its chunk."""

from __future__ import annotations

import re
from dataclasses import dataclass

from contractlens.models import Chunk, Citation, RetrievedChunk
from contractlens.store.memory import tokenize

MARKER = re.compile(r"\[(\d{1,2})\]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'\[])")
NUMBERISH = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


@dataclass
class GroundingResult:
    citations: list[Citation]
    unsupported: list[int]

    @property
    def grounded(self) -> bool:
        return not self.unsupported


def claim_sentences(answer: str) -> list[tuple[str, list[int]]]:
    """Split the answer into sentences and attach the markers each sentence cites."""
    out: list[tuple[str, list[int]]] = []
    for sentence in SENTENCE_SPLIT.split(answer.strip()):
        markers = [int(m) for m in MARKER.findall(sentence)]
        if markers:
            cleaned = re.sub(r"\s+([.,;:!?])", r"\1", MARKER.sub("", sentence)).strip()
            out.append((cleaned, markers))
    return out


def support_score(claim: str, chunk_text: str) -> float:
    """Cheap, deterministic support estimate: share of the claim's content terms found in the
    chunk, with numbers and amounts weighted heavily because they carry the meaning in
    contracts (a 30-day notice period is not a 60-day one)."""
    claim_terms = set(tokenize(claim))
    chunk_terms = set(tokenize(chunk_text))
    if not claim_terms:
        return 0.0
    term_score = len(claim_terms & chunk_terms) / len(claim_terms)
    claim_numbers = set(NUMBERISH.findall(claim))
    if claim_numbers:
        chunk_numbers = set(NUMBERISH.findall(chunk_text))
        number_score = len(claim_numbers & chunk_numbers) / len(claim_numbers)
        return 0.5 * term_score + 0.5 * number_score
    return term_score


def best_quote(claim: str, chunk_text: str, max_len: int = 240) -> str:
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(chunk_text) if s.strip()]
    if not sentences:
        return chunk_text[:max_len]
    best = max(sentences, key=lambda s: support_score(claim, s))
    return best if len(best) <= max_len else best[: max_len - 1].rstrip() + "…"


def check_grounding(answer: str, context: list[RetrievedChunk], *, threshold: float = 0.3) -> GroundingResult:
    by_marker: dict[int, Chunk] = {i + 1: rc.chunk for i, rc in enumerate(context)}
    citations: dict[int, Citation] = {}
    unsupported: set[int] = set()

    for claim, markers in claim_sentences(answer):
        for marker in markers:
            chunk = by_marker.get(marker)
            if chunk is None:
                unsupported.add(marker)
                continue
            score = support_score(claim, chunk.text)
            supported = score >= threshold
            if not supported:
                unsupported.add(marker)
            existing = citations.get(marker)
            if existing is None or (supported and not existing.supported):
                citations[marker] = Citation(
                    marker=marker,
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    section=chunk.section,
                    page=chunk.page,
                    quote=best_quote(claim, chunk.text),
                    supported=supported,
                )

    # A marker is only unsupported if none of its claims were supported.
    final_unsupported = sorted(m for m in unsupported if not (citations.get(m) and citations[m].supported))
    return GroundingResult(citations=[citations[m] for m in sorted(citations)], unsupported=final_unsupported)
