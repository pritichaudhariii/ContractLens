"""LLM providers behind one interface, plus an offline fake for tests, CI and demos without keys."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod

from contractlens.store.memory import tokenize

ABBREVIATIONS = re.compile(
    r"\b(?:a\.m|p\.m|Dr|Mr|Ms|Mrs|Inc|Corp|Co|Ltd|No|U\.S|N\.A|L\.P|LLC|e\.g|i\.e|vs|Jr|Sr|St)\.", re.I
)
CONTEXT_BLOCK = re.compile(r"^\[(\d+)\]\s+\((?P<label>[^\n]*)\)[ \t]*\n(?P<body>.*?)(?=^\[\d+\]\s+\(|\Z)", re.S | re.M)


class LLM(ABC):
    name: str
    model: str

    @abstractmethod
    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str: ...


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()


class FakeLLM(LLM):
    """Deterministic stand-in for a model.

    For answer generation it is *extractive*: it parses the numbered context blocks out of the
    prompt, picks the sentences that best match the question, and returns them with citation
    markers. That is enough to drive the whole graph, the grounding check and the eval harness
    end to end without a network call. For anything else it echoes a short acknowledgement.
    """

    name = "fake"
    model = "fake-extractive"

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        if "QUESTION:" in user and "[1]" in user:
            return self._extractive_answer(user)
        return "OK"

    def _extractive_answer(self, prompt: str) -> str:
        question = prompt.split("QUESTION:", 1)[1].strip().splitlines()[0]
        context_part = prompt.split("CONTEXT:", 1)[1] if "CONTEXT:" in prompt else prompt
        context_part = context_part.split("QUESTION:", 1)[0]
        blocks = list(CONTEXT_BLOCK.finditer(context_part))

        # Terms that name the document (its title, before the " — section" suffix) are scoping
        # words, not the fact being asked for, so they do not count towards a sentence's score.
        q_terms = set(tokenize(question))
        title_terms: set[str] = set()
        for m in blocks:
            title_terms |= set(tokenize(m.group("label").split(" — ", 1)[0]))
        core_terms = q_terms - title_terms or q_terms
        wants_number = bool(
            re.search(
                r"\b(how (much|many|long|often|soon)|when|what (is|was|are|were) the .*"
                r"(fee|rate|price|amount|term|period|cap|date|number|percentage|ratio|revenue|salary|bonus|deposit|allowance|limit|threshold|margin))\b",
                question.lower(),
            )
        )

        # Sentences per block. Headings are separate lines, so split on newlines as well as on
        # sentence punctuation, and protect common abbreviations from being split.
        # A sentence inherits its section heading's terms ("10. Standstill" describes what follows).
        sentences: list[tuple[int, str, set[str]]] = []
        for m in blocks:
            marker = int(m.group(1))
            label_parts = m.group("label").split(" — ", 1)
            section_terms = set(tokenize(label_parts[1])) if len(label_parts) > 1 else set()
            body = ABBREVIATIONS.sub(lambda a: a.group(0).replace(".", "\u2024"), m.group("body"))
            for raw in re.split(r"(?<=[.;])\s+|\n+", body):
                sentence = raw.strip().replace("\u2024", ".")
                s_terms = set(tokenize(sentence))
                if len(s_terms) < 4 or not re.search(r"[.;]\s*$", sentence):
                    continue
                sentences.append((marker, sentence, s_terms | section_terms))
        if not sentences:
            return "The documents provided do not contain information that answers this question."

        # Rare terms carry more signal than common ones ("standstill" vs "agreement").
        df: dict[str, int] = {}
        for _, _, s_terms in sentences:
            for t in s_terms:
                df[t] = df.get(t, 0) + 1
        n = len(sentences)
        weight = {t: 1.0 + math.log((n + 1) / (df.get(t, 0) + 1)) for t in core_terms}
        total_weight = sum(weight.values()) or 1.0

        candidates: list[tuple[float, int, str]] = []
        for marker, sentence, s_terms in sentences:
            matched = sum(weight[t] for t in core_terms & s_terms)
            if matched == 0:
                continue
            overlap = matched / total_weight
            digits = len(re.findall(r"\d", sentence))
            bonus = min(0.03 * digits, 0.25) if wants_number else min(0.01 * digits, 0.1)
            candidates.append((overlap + bonus, marker, sentence))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        picked: list[tuple[int, str]] = []
        seen: set[str] = set()
        for _score, marker, sentence in candidates:
            if sentence in seen:
                continue
            seen.add(sentence)
            picked.append((marker, sentence))
            if len(picked) == 2:
                break
        if not picked:
            return "The documents provided do not contain information that answers this question."
        return " ".join(f"{sentence.rstrip('.;')} [{marker}]." for marker, sentence in picked)


def build_llm(provider: str, *, api_key: str | None, model: str) -> LLM:
    if provider == "anthropic":
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for the anthropic LLM provider")
        return AnthropicLLM(api_key, model=model)
    if provider == "fake":
        return FakeLLM()
    raise ValueError(f"Unknown LLM provider: {provider}")
