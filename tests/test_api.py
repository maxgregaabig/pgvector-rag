from uuid import uuid4

from tests.conftest import FINTECH_FAQ


def test_healthz_reports_the_active_backends(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "store": "memory",
        "cache": "memory",
        "provider": "fake",
    }


def test_ingest_search_ask_roundtrip(client):
    ingest = client.post("/v1/documents", json={"source": "faq.md", "text": FINTECH_FAQ})
    assert ingest.status_code == 201
    body = ingest.json()
    assert body["chunks"] >= 1
    assert body["embedding_tokens"] > 0

    search = client.post("/v1/search", json={"query": "payout cutoff time", "top_k": 2})
    assert search.status_code == 200
    results = search.json()["results"]
    assert 1 <= len(results) <= 2
    assert results[0]["document_id"] == body["document_id"]

    ask = client.post("/v1/ask", json={"question": "When is the payout cutoff?", "top_k": 3})
    assert ask.status_code == 200
    answer = ask.json()
    assert answer["citations"]
    assert answer["cached"] is False
    assert answer["usage"]["chat_input_tokens"] > 0


def test_second_identical_ask_is_cached_and_skips_the_provider(client, fake_provider_of):
    client.post("/v1/documents", json={"source": "faq.md", "text": FINTECH_FAQ})
    question = {"question": "How long is the chargeback window?", "top_k": 3}

    first = client.post("/v1/ask", json=question).json()
    calls = (fake_provider_of.chat_calls, fake_provider_of.embed_calls)
    second = client.post("/v1/ask", json=question).json()

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"]
    assert (fake_provider_of.chat_calls, fake_provider_of.embed_calls) == calls


def test_delete_document_removes_it_from_search(client):
    document_id = client.post(
        "/v1/documents", json={"source": "faq.md", "text": FINTECH_FAQ}
    ).json()["document_id"]

    assert client.delete(f"/v1/documents/{document_id}").status_code == 204
    assert client.delete(f"/v1/documents/{document_id}").status_code == 404
    assert client.post("/v1/search", json={"query": "payout cutoff"}).json()["results"] == []


def test_missing_document_scope_returns_no_results(client):
    client.post("/v1/documents", json={"source": "faq.md", "text": FINTECH_FAQ})
    response = client.post("/v1/search", json={"query": "payout", "document_ids": [str(uuid4())]})
    assert response.json()["results"] == []


def test_blank_document_is_rejected(client):
    assert client.post("/v1/documents", json={"source": "x.md", "text": "  "}).status_code == 422


def test_invalid_payloads_are_rejected(client):
    assert client.post("/v1/documents", json={"source": "x.md"}).status_code == 422
    assert client.post("/v1/search", json={"query": ""}).status_code == 422
    assert client.post("/v1/search", json={"query": "q", "top_k": 0}).status_code == 422
    assert client.post("/v1/ask", json={"question": "q", "top_k": 999}).status_code == 422
