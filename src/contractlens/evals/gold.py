"""Gold dataset loading."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldSource(BaseModel):
    document_id: str
    document_title: str = ""
    section: str = ""


class GoldQuestion(BaseModel):
    id: str
    question: str
    answer: str = Field(description="Reference answer written by a human")
    expected_facts: list[str] = Field(description="Facts a correct answer must contain; alternatives separated by '|'")
    source: GoldSource
    category: str = "factual"
    difficulty: str = "medium"


def load_gold(path: str | Path) -> list[GoldQuestion]:
    rows: list[GoldQuestion] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(GoldQuestion.model_validate(json.loads(line)))
            except Exception as err:  # noqa: BLE001 - surface the offending line
                raise ValueError(f"{path}:{line_no}: {err}") from err
    ids = [r.id for r in rows]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate gold ids: {sorted(duplicates)}")
    return rows
