import json

from fastapi.testclient import TestClient

from contractlens.api.app import create_app
from contractlens.evals.gold import load_gold
from contractlens.evals.harness import EvalHarness
from contractlens.evals.judge import HeuristicJudge
from tests.conftest import GOLD, offline_settings


def test_graph_answers_with_citations(corpus_container):
    answer = corpus_container.lens.ask("How long is the standstill period in the Acme–Bluewater NDA?")
    assert "18" in answer.answer
    assert answer.citations and answer.grounded
    assert answer.citations[0].document_id == "mutual-non-disclosure-agreement-acme-bluewater"
    assert answer.trace["scoped_document"] == "mutual-non-disclosure-agreement-acme-bluewater"


def test_graph_abstains_without_context(corpus_container):
    answer = corpus_container.lens.ask("What is the airspeed velocity of an unladen swallow?")
    assert answer.grounded
    assert not answer.citations or "do not contain" in answer.answer.lower() or "couldn't find" in answer.answer.lower()


def test_graph_mermaid_lists_nodes(corpus_container):
    mermaid = corpus_container.lens.mermaid()
    for node in ("analyze_query", "retrieve", "process_context", "generate", "ground_check", "no_context"):
        assert node in mermaid


def test_eval_harness_subset_meets_offline_gate(corpus_container):
    questions = load_gold(GOLD)[:20]
    harness = EvalHarness(
        corpus_container,
        HeuristicJudge(),
        thresholds={"retrieval_hit_rate": 0.8, "grounded_rate": 0.9},
        profile="offline",
    )
    report = harness.run(questions, progress=False)
    assert report.questions == 20
    assert report.aggregates["retrieval_hit_rate"] >= 0.8
    assert report.passed, report.failures


def test_api_end_to_end(corpus_container):
    app = create_app(offline_settings(api_key="secret"), container=corpus_container)
    with TestClient(app) as client:
        assert client.get("/health").json()["documents"] == 10
        assert client.get("/documents").status_code == 401  # api key required
        headers = {"x-api-key": "secret"}
        docs = client.get("/documents", headers=headers).json()
        assert len(docs) == 10 and all(d["chunk_count"] > 0 for d in docs)

        res = client.post(
            "/ask",
            json={"question": "What was Northwind's revenue for fiscal 2025?", "include_passages": True},
            headers=headers,
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert "412.6" in body["answer"]
        assert body["citations"][0]["document_id"].startswith("form-10-k")
        assert body["passages"] and body["passages"][0]["marker"] == 1

        created = client.post(
            "/documents/text",
            json={
                "title": "Test Policy",
                "text": "# Test Policy\n\n## 1. Scope\n\nThis policy applies to all vendors and requires a forty-five (45) day review.",
                "doc_type": "policy",
            },
            headers=headers,
        )
        assert created.status_code == 201 and created.json()["chunks"] >= 1
        assert client.delete("/documents/test-policy", headers=headers).status_code == 204
        assert client.delete("/documents/test-policy", headers=headers).status_code == 404

        assert "analyze_query" in client.get("/graph").text
        assert "ContractLens" in client.get("/").text

        bad = client.post("/ask", json={"question": "hi"}, headers=headers)
        assert bad.status_code == 422


def test_streaming_ask_emits_nodes_then_answer(corpus_container):
    app = create_app(offline_settings(), container=corpus_container)
    with TestClient(app) as client:
        with client.stream(
            "POST", "/ask/stream", json={"question": "What is the base rent under the office lease?"}
        ) as res:
            assert res.status_code == 200
            assert res.headers["content-type"].startswith("text/event-stream")
            body = "".join(res.iter_text())
    events = [f for f in body.split("\n\n") if f.strip()]
    kinds = [f.split("\n")[0].removeprefix("event: ") for f in events]
    assert kinds[:2] == ["node", "node"] and kinds[-1] == "answer"
    nodes = [json.loads(f.split("\ndata: ", 1)[1])["node"] for f in events if f.startswith("event: node")]
    assert nodes[:3] == ["analyze_query", "retrieve", "process_context"]
    answer = json.loads(events[-1].split("\ndata: ", 1)[1])
    assert "38.50" in answer["answer"] and answer["passages"]
    assert all("highlight" in c and "support_score" in c for c in answer["citations"])


def test_document_detail_and_stats(corpus_container):
    app = create_app(offline_settings(), container=corpus_container)
    with TestClient(app) as client:
        detail = client.get("/documents/office-lease-agreement-riverbend-northwind").json()
        assert detail["chunks"] and detail["chunks"][0]["ordinal"] == 0
        assert [c["ordinal"] for c in detail["chunks"]] == sorted(c["ordinal"] for c in detail["chunks"])
        assert client.get("/documents/nope").status_code == 404
        stats = client.get("/stats").json()
        assert stats["documents"] == 10 and stats["chunks"] > 100 and stats["by_type"]["contract"] == 7
        assert client.get("/evals/latest").status_code in (200, 404)
        assert "x-request-id" in client.get("/stats").headers
