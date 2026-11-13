"""End-to-end integration: HTTP API -> pipeline -> retrieval -> generation.

Uses the committed corpus (data/processed/chunks.jsonl) with the BM25 retriever
and the extractive generator, so no model is downloaded and the test runs in CI.
"""

from __future__ import annotations

import pytest

from docqa.cli import main as cli_main
from docqa.config import REPO_ROOT, Config

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

NO_DOWNLOAD = {
    "retrieval.method": "bm25",
    "generation.generator": "extractive",
    "generation.min_retrieval_score": 0,
}
NO_DOWNLOAD_ARGS = [
    arg for k, v in NO_DOWNLOAD.items() for arg in ("--set", f"{k}={v}")
]

CHUNKS = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
needs_corpus = pytest.mark.skipif(
    not CHUNKS.exists(), reason="processed corpus not present"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg = Config.load("configs/default.json", overrides=NO_DOWNLOAD)
    cfg_path = tmp_path / "config.json"
    cfg.dump(cfg_path)
    monkeypatch.setenv("DOCQA_CONFIG", str(cfg_path))

    from server.app import app

    with TestClient(app) as test_client:
        yield test_client


@needs_corpus
def test_health_reports_loaded_pipeline(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0
    assert body["retriever"] == "bm25"
    assert body["generator"] == "extractive"


@needs_corpus
def test_ask_returns_grounded_answer_with_citations(client):
    res = client.post(
        "/ask", json={"question": "How do I make a query parameter optional?"}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["abstained"] is False
    assert body["answer"].strip()
    assert body["citations"], "an answered question must cite at least one passage"
    assert any("query-param" in c["source"] for c in body["citations"])
    assert {"retrieval", "generation"} <= set(body["timings"])


@needs_corpus
@pytest.mark.parametrize("payload", [{"question": "   "}, {"question": ""}, {}])
def test_ask_rejects_empty_question(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


@needs_corpus
def test_frontend_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "/ask" in res.text


@needs_corpus
def test_cli_ask_end_to_end(capsys):
    code = cli_main(["ask", "How do I add CORS middleware?", *NO_DOWNLOAD_ARGS])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("Q: How do I add CORS middleware?")
    assert "Sources:" in out or "abstained" in out


def test_cli_empty_question_exits_with_error(capsys):
    assert cli_main(["ask", "  ", *NO_DOWNLOAD_ARGS]) == 2
    assert "question is empty" in capsys.readouterr().err
