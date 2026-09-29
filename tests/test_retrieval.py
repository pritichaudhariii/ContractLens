from contractlens.models import Chunk, Document
from contractlens.retrieval import reciprocal_rank_fusion
from contractlens.scope import scope_documents


def chunk(i: str, doc: str = "d") -> Chunk:
    return Chunk(id=i, document_id=doc, document_title="D", ordinal=0, section="", text=f"text {i}")


def test_rrf_prefers_chunks_ranked_by_both_retrievers():
    dense = [(chunk("a"), 0.9), (chunk("b"), 0.8), (chunk("c"), 0.7)]
    sparse = [(chunk("c"), 3.0), (chunk("b"), 2.0), (chunk("z"), 1.0)]
    fused = reciprocal_rank_fusion(dense, sparse, k=3, rrf_k=60)
    ids = [r.chunk.id for r in fused]
    assert ids[0] in {"b", "c"}  # both appear in both lists
    assert set(ids) <= {"a", "b", "c", "z"}
    b = next(r for r in fused if r.chunk.id == "b")
    assert b.vector_rank == 2 and b.keyword_rank == 2


def test_rrf_handles_empty_side():
    dense = [(chunk("a"), 0.9)]
    fused = reciprocal_rank_fusion(dense, [], k=5)
    assert [r.chunk.id for r in fused] == ["a"]
    assert fused[0].keyword_rank is None


def docs() -> list[Document]:
    return [
        Document(id="nda", title="Mutual Non-Disclosure Agreement (Acme–Bluewater)", doc_type="contract"),
        Document(id="msa", title="Master Services Agreement (Northwind–Acme)", doc_type="contract"),
        Document(id="10k", title="Form 10-K Annual Report FY2025 (Northwind Analytics)", doc_type="regulatory_filing"),
        Document(id="lease", title="Office Lease Agreement (Riverbend–Northwind)", doc_type="contract"),
    ]


def test_scope_picks_named_document():
    assert scope_documents("How long is the standstill period in the NDA?", docs()).document_ids == ["nda"]
    assert scope_documents("What uptime does the MSA commit to?", docs()).document_ids == ["msa"]
    assert scope_documents("What is the base rent under the lease?", docs()).document_ids == ["lease"]


def test_scope_stays_open_when_ambiguous():
    # "Northwind" and "agreement" appear in several titles and must not decide on their own.
    assert scope_documents("What did Northwind agree to?", docs()).document_ids is None
    assert scope_documents("Which agreements mention Acme?", docs()).document_ids is None
