"""Query analysis: work out which document a question is about.

Questions about contracts usually name the document ("the NDA", "the Helix subscription
agreement", "the 10-K"). Those words dominate lexical retrieval and drag in preambles instead
of the clause that answers the question. Detecting the document up front and filtering
retrieval to it (metadata filtering) is the single most effective retrieval improvement for
this kind of corpus, and it is what the `analyze_query` node feeds into `retrieve`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from contractlens.models import Document
from contractlens.store.memory import tokenize

# Phrases that strongly imply a kind of document. Keys are matched against the lowercased title;
# values are matched against the lowercased question. A phrase match is worth 2 points.
TITLE_ALIASES: dict[str, tuple[str, ...]] = {
    "non-disclosure": ("nda", "non-disclosure", "nondisclosure", "confidentiality agreement", "standstill"),
    "master services": ("msa", "master services agreement", "services agreement", "service credit"),
    "data processing": (
        "dpa",
        "data processing addendum",
        "processing addendum",
        "sub-processor",
        "subprocessor",
        "data subject",
        "personal data",
    ),
    "10-k": (
        "10-k",
        "10k",
        "annual report",
        "form 10",
        "risk factor",
        "legal proceedings",
        "fiscal 2025",
        "fiscal 2024",
        "net loss",
        "revenue",
    ),
    "8-k": (
        "8-k",
        "8k",
        "current report",
        "form 8",
        "merger",
        "acquisition",
        "acquire",
        "earn-out",
        "earnout",
        "escrow",
        "closing date",
    ),
    "proxy": (
        "proxy",
        "def 14a",
        "14a",
        "annual meeting",
        "stockholder",
        "shareholder",
        "board of directors",
        "director",
        "nominee",
        "say-on-pay",
        "audit fee",
        "pay ratio",
        "record date",
        "retainer",
        "total compensation",
    ),
    "lease": (
        "lease",
        "landlord",
        "tenant",
        "premises",
        "rent",
        "square feet",
        "square foot",
        "floor",
        "parking",
        "holdover",
        "security deposit",
        "tenant improvement",
    ),
    "credit": (
        "credit agreement",
        "credit facility",
        "revolver",
        "revolving",
        "lender",
        "loan",
        "covenant",
        "sofr",
        "leverage",
        "commitment fee",
        "maturity",
        "cross-default",
    ),
    "employment": (
        "employment agreement",
        "executive agreement",
        "severance",
        "cfo",
        "chief financial officer",
        "signing bonus",
        "non-compete",
        "vesting",
        "paid time off",
    ),
    "subscription": (
        "saas",
        "subscription agreement",
        "subscription",
        "named user",
        "uptime",
        "premier support",
        "aws",
    ),
}
# Title words that appear across many documents and must not decide scope on their own.
GENERIC = {
    "agreement",
    "form",
    "inc",
    "corp",
    "llc",
    "analytics",
    "report",
    "statement",
    "current",
    "annual",
    "northwind",
    "first",
    "february",
    "2026",
    "fy2025",
    "meeting",
    "office",
    "mutual",
    "executive",
    "data",
}


@dataclass
class ScopeDecision:
    document_ids: list[str] | None
    scores: dict[str, int]

    @property
    def scoped(self) -> bool:
        return bool(self.document_ids)


def title_tokens(doc: Document) -> set[str]:
    title = doc.title.lower()
    tokens = set(tokenize(title)) - GENERIC
    tokens -= {t for t in tokens if t.isdigit() or len(t) <= 1}
    return tokens


def alias_phrases(doc: Document) -> tuple[str, ...]:
    title = doc.title.lower()
    phrases: list[str] = []
    for key, aliases in TITLE_ALIASES.items():
        if key in title:
            phrases.extend(aliases)
    return tuple(phrases)


def score_document(question: str, doc: Document) -> int:
    q_lower = question.lower()
    q_tokens = set(tokenize(q_lower))
    score = len(title_tokens(doc) & q_tokens)
    for phrase in alias_phrases(doc):
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", q_lower):
            score += 2
    return score


def scope_documents(question: str, documents: list[Document]) -> ScopeDecision:
    scores = {doc.id: s for doc in documents if (s := score_document(question, doc)) > 0}
    if not scores:
        return ScopeDecision(None, scores)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_id, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    # Scope only when one document is clearly named: a strong match and a clear lead over the next.
    if best >= 2 and best > runner_up:
        return ScopeDecision([best_id], scores)
    return ScopeDecision(None, scores)
