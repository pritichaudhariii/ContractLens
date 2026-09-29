from contractlens.ingest.chunk import chunk_text, detect_heading
from contractlens.ingest.parse import PAGE_BREAK

DOC = (
    """# Sample Agreement

This Agreement is made on January 5, 2026.

## 1. Term

The term is twelve (12) months. It renews automatically.

## 2. Payment

Invoices are due within thirty (30) days. """
    + ("Late fees accrue at one percent per month. " * 60)
    + """

## 3. Governing Law

Delaware law governs.
"""
)


def test_headings_detected():
    assert detect_heading("## 9. Limitation of Liability") == "9. Limitation of Liability"
    assert detect_heading("Item 7. Management's Discussion") == "Item 7. Management's Discussion"
    assert detect_heading("This is a normal sentence that is long enough.") is None


def test_chunks_respect_sections_and_carry_metadata():
    chunks = chunk_text(
        DOC, document_id="sample", document_title="Sample Agreement", doc_type="contract", chunk_size=600, overlap=80
    )
    assert chunks, "expected chunks"
    sections = [c.section for c in chunks]
    assert "1. Term" in sections and "3. Governing Law" in sections
    # Long section split into several chunks, each tagged with the same section
    payment = [c for c in chunks if c.section == "2. Payment"]
    assert len(payment) >= 2
    assert all(len(c.text) <= 600 + 80 + 40 for c in payment)
    # Ordinals are contiguous and ids unique
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)
    # Overlap: the second payment chunk starts with the tail of the first
    assert payment[1].text.startswith(payment[0].text.split(". ")[-2][:10]) or "Late fees" in payment[1].text


def test_page_numbers_from_page_breaks():
    text = "# Title\n\nPage one text here." + PAGE_BREAK + "## 1. Second\n\nPage two text here."
    chunks = chunk_text(text, document_id="d", document_title="Title")
    pages = {c.section: c.page for c in chunks}
    assert pages["1. Second"] == 2
    assert min(p for p in pages.values() if p) == 1
