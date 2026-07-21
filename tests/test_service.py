from uuid import uuid4

import pytest

from app.models import SearchHit
from app.service import EmptyDocumentError, RagService, build_user_prompt, extract_citations
from tests.conftest import FINTECH_FAQ


async def test_ingest_then_search_finds_the_right_chunk(service: RagService):
    await service.ingest(source="faq.md", text=FINTECH_FAQ)
    response = await service.search("chargeback deadline for the cardholder", top_k=3)
    assert response.results
    assert "Chargebacks" in response.results[0].content
    assert response.results[0].score > 0


async def test_search_scores_are_ordered(service: RagService):
    await service.ingest(source="faq.md", text=FINTECH_FAQ)
    scores = [r.score for r in (await service.search("payout cutoff", top_k=5)).results]
    assert scores == sorted(scores, reverse=True)


async def test_search_can_be_scoped_to_one_document(service: RagService):
    first = await service.ingest(source="a.md", text="Alpha covers settlement timing rules.")
    await service.ingest(source="b.md", text="Bravo covers settlement timing rules too.")
    response = await service.search("settlement timing", top_k=5, document_ids=[first.document_id])
    assert {r.document_id for r in response.results} == {first.document_id}


async def test_empty_document_is_rejected(service: RagService):
    with pytest.raises(EmptyDocumentError):
        await service.ingest(source="empty.md", text="   ")


async def test_ask_returns_citations_that_point_at_retrieved_chunks(service: RagService):
    ingested = await service.ingest(source="faq.md", text=FINTECH_FAQ)
    response = await service.ask("What is the payout cutoff?", top_k=3)
    assert response.citations
    assert not response.cached
    retrieved = await service.search("What is the payout cutoff?", 3)
    chunk_ids = {result.chunk_id for result in retrieved.results}
    for citation in response.citations:
        assert citation.chunk_id in chunk_ids
        assert citation.document_id == ingested.document_id
        assert f"[{citation.marker}]" in response.answer


async def test_ask_with_no_documents_does_not_invent_citations(service: RagService):
    response = await service.ask("Anything at all?")
    assert response.citations == []


async def test_repeated_ingest_of_the_same_text_reuses_cached_embeddings(service: RagService):
    first = await service.ingest(source="faq.md", text=FINTECH_FAQ)
    calls_after_first = service.provider.embed_calls
    second = await service.ingest(source="faq-copy.md", text=FINTECH_FAQ)
    assert second.cached_embeddings == second.chunks == first.chunks
    assert second.embedding_tokens == 0
    assert service.provider.embed_calls == calls_after_first


async def test_repeated_question_is_served_from_cache_without_calling_the_provider(
    service: RagService,
):
    await service.ingest(source="faq.md", text=FINTECH_FAQ)
    first = await service.ask("How long is the chargeback window?", top_k=3)
    embed_calls = service.provider.embed_calls
    chat_calls = service.provider.chat_calls

    second = await service.ask("How long is the chargeback window?", top_k=3)

    assert second.cached is True
    assert first.cached is False
    assert second.answer == first.answer
    assert [c.chunk_id for c in second.citations] == [c.chunk_id for c in first.citations]
    assert service.provider.chat_calls == chat_calls
    assert service.provider.embed_calls == embed_calls


async def test_a_different_top_k_is_a_different_cache_entry(service: RagService):
    await service.ingest(source="faq.md", text=FINTECH_FAQ)
    await service.ask("How long is the chargeback window?", top_k=2)
    chat_calls = service.provider.chat_calls
    response = await service.ask("How long is the chargeback window?", top_k=3)
    assert response.cached is False
    assert service.provider.chat_calls == chat_calls + 1


def test_prompt_numbers_the_context_blocks():
    hits = [
        SearchHit(uuid4(), uuid4(), "a.md", None, 0, "First body.", 0.9),
        SearchHit(uuid4(), uuid4(), "b.md", None, 2, "Second body.", 0.5),
    ]
    prompt = build_user_prompt("q?", hits)
    assert "[1] source=a.md#0" in prompt
    assert "[2] source=b.md#2" in prompt


def test_out_of_range_markers_are_dropped():
    hits = [SearchHit(uuid4(), uuid4(), "a.md", None, 0, "Body.", 0.9)]
    citations = extract_citations("Grounded [1] but invented [7] and repeated [1].", hits)
    assert [c.marker for c in citations] == [1]
