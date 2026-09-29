from contractlens.evals.metrics import fact_present, fact_recall, heuristic_faithfulness, normalize, token_f1
from contractlens.grounding import check_grounding, claim_sentences
from contractlens.models import Answer, Chunk, RetrievedChunk


def rc(i: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=f"c{i}", document_id="doc", document_title="Doc", ordinal=i, section="S", text=text), score=1.0
    )


CONTEXT = [
    rc(
        0,
        "Invoices are due within thirty (30) days of the invoice date. Late amounts accrue interest at 1.5% per month.",
    ),
    rc(1, "The Agreement is governed by the laws of the State of Delaware."),
]


def test_normalize_expands_spelled_numbers_and_currency():
    assert normalize("within thirty (30) days") == "within 30 days"
    assert normalize("four hundred thirty-two thousand dollars ($432,000)") == "432000"
    assert normalize("one and one-half percent (1.5%) per month") == "1.5% per month"
    assert fact_present("$432,000|432,000", "The fee is four hundred thirty-two thousand dollars ($432,000) per year.")
    assert fact_present("30 days", "Payment is due within thirty (30) days.")
    assert not fact_present("60 days", "Payment is due within thirty (30) days.")


def test_fact_recall_and_token_f1():
    assert fact_recall(["30 days", "1.5%"], "Due within 30 days; interest is 1.5% monthly") == 1.0
    assert fact_recall(["30 days", "99.9%"], "Due within 30 days") == 0.5
    assert 0.5 < token_f1("Invoices are due within 30 days", "Payment is due within 30 days of invoice") < 1.0


def test_grounding_flags_unsupported_and_invalid_markers():
    good = "Invoices are due within 30 days [1]. Delaware law governs [2]."
    result = check_grounding(good, CONTEXT)
    assert result.grounded and [c.marker for c in result.citations] == [1, 2]
    assert "Delaware" in result.citations[1].quote

    bad = "The notice period is ninety days for termination [2]. Something cited nowhere [7]."
    result = check_grounding(bad, CONTEXT)
    assert not result.grounded
    assert 7 in result.unsupported and 2 in result.unsupported


def test_claim_sentences_parse_markers():
    claims = claim_sentences("First claim [1][2]. No citation here. Third [3].")
    assert claims == [("First claim.", [1, 2]), ("Third.", [3])]


def test_heuristic_faithfulness():
    answer = Answer(
        question="q",
        answer="Invoices are due within 30 days [1]. Delaware law governs [2].",
        citations=[],
        grounded=True,
        retrieved=CONTEXT,
    )
    assert heuristic_faithfulness(answer) == 1.0
    fabricated = Answer(
        question="q", answer="The cap is nine million dollars [1].", citations=[], grounded=False, retrieved=CONTEXT
    )
    assert heuristic_faithfulness(fabricated) < 0.5
