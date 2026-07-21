import math

from app.providers.base import estimate_tokens
from app.providers.fake import FakeProvider, hash_embedding
from app.store.memory import cosine_similarity


def test_embedding_is_deterministic_and_unit_length():
    first = hash_embedding("settlement happens on T plus two", 256)
    second = hash_embedding("settlement happens on T plus two", 256)
    assert first == second
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-9)


def test_empty_text_still_yields_a_valid_vector():
    vector = hash_embedding("   ", 64)
    assert len(vector) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_shared_vocabulary_scores_higher_than_unrelated_text():
    document = hash_embedding("Chargebacks can be raised within 120 days.", 1536)
    related = hash_embedding("chargeback deadline in days", 1536)
    unrelated = hash_embedding("The office kitchen has a coffee machine.", 1536)
    assert cosine_similarity(document, related) > cosine_similarity(document, unrelated)


async def test_embed_returns_one_vector_per_input_and_counts_calls():
    provider = FakeProvider(dim=128)
    result = await provider.embed(["alpha", "bravo", "charlie"])
    assert len(result.vectors) == 3
    assert all(len(vector) == 128 for vector in result.vectors)
    assert result.tokens > 0
    assert provider.embed_calls == 1
    assert provider.embedded_texts == 3


async def test_chat_cites_the_context_blocks_it_used():
    provider = FakeProvider(dim=64)
    prompt = (
        "Question: when is the cutoff?\n\nContext:\n"
        "[1] source=faq.md#0\nThe daily payout cutoff is 16:00 UTC. More text.\n\n"
        "[2] source=faq.md#1\nChargebacks run for 120 days.\n"
    )
    result = await provider.chat("system", prompt)
    assert "[1]" in result.text
    assert "[2]" in result.text
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert provider.chat_calls == 1


async def test_chat_without_context_says_so():
    provider = FakeProvider(dim=64)
    prompt = "Question: anything?\n\nContext:\n(no matching documents)"
    result = await provider.chat("system", prompt)
    assert "could not find" in result.text


def test_token_estimate_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100
