"""Judges score faithfulness and answer relevance.

`LLMJudge` asks a model to grade the answer against the retrieved passages and the reference
answer (LLM-as-judge). `HeuristicJudge` is the zero-dependency fallback used in CI without keys:
faithfulness from lexical support of each cited claim, relevance from fact recall and token F1.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from contractlens.evals.gold import GoldQuestion
from contractlens.evals.metrics import fact_recall, heuristic_faithfulness, is_abstention, token_f1
from contractlens.llm import LLM
from contractlens.models import Answer

JUDGE_SYSTEM = """You are a strict evaluator of a document question-answering system for contracts and regulatory filings.
You will be given the question, the numbered passages the system retrieved, the system's answer, and a human-written reference answer.
Score two things on a 0.0 to 1.0 scale:

faithfulness: is every factual claim in the answer supported by the retrieved passages? 1.0 means every claim is directly supported; 0.0 means the answer contradicts or invents facts not in the passages. An answer that says the documents do not contain the information is faithful (1.0) if the passages indeed do not answer the question.

relevance: does the answer actually answer the question with the same substance as the reference? 1.0 means it gives the same facts (wording may differ); 0.5 means partially; 0.0 means it is off-topic, wrong, or missing the key fact.

Respond with only a JSON object: {"faithfulness": <number>, "relevance": <number>, "reason": "<one short sentence>"}"""


@dataclass
class JudgeScores:
    faithfulness: float
    relevance: float
    reason: str = ""


class Judge(ABC):
    name: str

    @abstractmethod
    def score(self, gold: GoldQuestion, answer: Answer) -> JudgeScores: ...


class HeuristicJudge(Judge):
    name = "heuristic"

    def score(self, gold: GoldQuestion, answer: Answer) -> JudgeScores:
        faithfulness = heuristic_faithfulness(answer)
        if is_abstention(answer.answer):
            relevance = 0.0
        else:
            relevance = 0.6 * fact_recall(gold.expected_facts, answer.answer) + 0.4 * token_f1(
                answer.answer, gold.answer
            )
        return JudgeScores(round(faithfulness, 3), round(relevance, 3), "lexical support / fact recall + token F1")


class LLMJudge(Judge):
    name = "llm"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def score(self, gold: GoldQuestion, answer: Answer) -> JudgeScores:
        passages = (
            "\n\n".join(
                f"[{i}] ({rc.chunk.document_title} — {rc.chunk.section})\n{rc.chunk.text}"
                for i, rc in enumerate(answer.retrieved, start=1)
            )
            or "(no passages were retrieved)"
        )
        user = (
            f"QUESTION:\n{gold.question}\n\nRETRIEVED PASSAGES:\n{passages}\n\n"
            f"SYSTEM ANSWER:\n{answer.answer}\n\nREFERENCE ANSWER:\n{gold.answer}\n\nJSON:"
        )
        raw = self.llm.complete(JUDGE_SYSTEM, user, max_tokens=300)
        return parse_judge_json(raw)


def parse_judge_json(raw: str) -> JudgeScores:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"judge returned no JSON: {raw[:200]!r}")
    data = json.loads(match.group(0))
    return JudgeScores(
        faithfulness=_clamp(float(data.get("faithfulness", 0))),
        relevance=_clamp(float(data.get("relevance", 0))),
        reason=str(data.get("reason", "")),
    )


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def build_judge(kind: str, llm: LLM | None) -> Judge:
    if kind == "llm":
        if llm is None or llm.name == "fake":
            raise RuntimeError("The LLM judge needs a real model; set ANTHROPIC_API_KEY or use --judge heuristic")
        return LLMJudge(llm)
    return HeuristicJudge()
