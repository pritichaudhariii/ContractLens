"""Deterministic metrics for the eval harness.

These are the metrics that never need a model: whether the right document was retrieved,
whether the citations point at it, and whether the answer contains the facts a human wrote
down as required. They run in CI on every commit with no API keys.
"""

from __future__ import annotations

import re
from collections import Counter

from contractlens.grounding import claim_sentences, support_score
from contractlens.models import Answer, RetrievedChunk
from contractlens.store.memory import tokenize

NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    "seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    "billion|and|a|an|half|quarter|dollars?|cents?|percent"
)
# "thirty (30) days" → "30 days"; "four hundred thirty-two thousand dollars ($432,000)" → "$432,000"
SPELLED_NUMBER = re.compile(rf"(?:\b(?:{NUMBER_WORDS})\b[\s\-]*){{1,14}}\(\s*(\$?\s?\d[\d,\.]*\s*%?)\s*\)", re.I)
PAREN_NUMBER = re.compile(r"\(\s*(\$?\d[\d,\.]*\s*%?)\s*\)")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def normalize(text: str) -> str:
    """Canonical form for substring matching of facts in answers and documents."""
    t = text.lower()
    t = SPELLED_NUMBER.sub(lambda m: f" {m.group(1)} ", t)
    t = PAREN_NUMBER.sub(lambda m: f" {m.group(1)} ", t)
    t = t.replace("percent", "%").replace("per cent", "%")
    t = t.replace("$ ", "$").replace(" %", "%")
    t = THOUSANDS.sub("", t)
    t = t.replace("$", "")
    t = re.sub(r"[–—\-–—]", "-", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fact_present(fact: str, text: str) -> bool:
    """A fact is present if any of its `|`-separated alternatives appears in the normalized text."""
    norm_text = normalize(text)
    return any(normalize(alt) in norm_text for alt in fact.split("|") if alt.strip())


def fact_recall(expected_facts: list[str], text: str) -> float:
    if not expected_facts:
        return 1.0
    hits = sum(1 for f in expected_facts if fact_present(f, text))
    return hits / len(expected_facts)


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token overlap between the answer and the reference answer."""
    p = Counter(tokenize(normalize(prediction)))
    r = Counter(tokenize(normalize(reference)))
    if not p or not r:
        return 0.0
    common = sum((p & r).values())
    if common == 0:
        return 0.0
    precision = common / sum(p.values())
    recall = common / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def retrieval_hit(retrieved: list[RetrievedChunk], document_id: str, section: str = "") -> tuple[bool, bool]:
    """(document retrieved, exact section retrieved) over the passages handed to the model."""
    doc_hit = any(rc.chunk.document_id == document_id for rc in retrieved)
    section_hit = bool(section) and any(
        rc.chunk.document_id == document_id and rc.chunk.section == section for rc in retrieved
    )
    return doc_hit, section_hit


def citation_precision(answer: Answer, document_id: str) -> float | None:
    """Share of citations that point at the expected document. None when nothing was cited."""
    if not answer.citations:
        return None
    on_target = sum(1 for c in answer.citations if c.document_id == document_id)
    return on_target / len(answer.citations)


def heuristic_faithfulness(answer: Answer, *, threshold: float = 0.3) -> float:
    """Share of cited claim sentences whose cited passage lexically supports them.

    An abstention (no claims) is perfectly faithful: it asserts nothing. Claims without any
    citation count as unsupported, because the product's contract is that every claim cites.
    """
    text = answer.answer.strip()
    if not text:
        return 0.0
    by_marker = {i + 1: rc.chunk.text for i, rc in enumerate(answer.retrieved)}
    claims = claim_sentences(text)
    sentences_total = len([s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()])
    if not claims:
        return 1.0 if sentences_total <= 1 and not by_marker else 0.0 if by_marker else 1.0
    supported = 0
    for claim, markers in claims:
        best = max((support_score(claim, by_marker[m]) for m in markers if m in by_marker), default=0.0)
        if best >= threshold:
            supported += 1
    uncited = max(sentences_total - len(claims), 0)
    return supported / (len(claims) + uncited)


def is_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "do not contain",
            "does not contain",
            "couldn't find",
            "could not find",
            "no passage",
            "not addressed",
        )
    )
