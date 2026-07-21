import pytest

from app.chunking import chunk_text, split_sentences
from tests.conftest import FINTECH_FAQ


def test_blank_input_produces_no_chunks():
    # Compared against [] rather than checked for falsiness, so returning None
    # instead of an empty list would fail this test.
    # pylint: disable=use-implicit-booleaness-not-comparison
    assert chunk_text("   \n\n  ") == []


def test_split_sentences_normalises_whitespace():
    assert split_sentences("One   thing.\nTwo things!") == ["One thing.", "Two things!"]


def test_chunks_stay_near_the_target_size():
    chunks = chunk_text(FINTECH_FAQ, target_chars=200, overlap_chars=40)
    assert len(chunks) > 1
    # Packing is done in whole sentences, so a chunk can overshoot by at most
    # the length of the sentence that closed it.
    assert all(len(chunk) <= 400 for chunk in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_text(FINTECH_FAQ, target_chars=200, overlap_chars=80)
    shared = [
        any(sentence in chunks[index + 1] for sentence in split_sentences(chunks[index]))
        for index in range(len(chunks) - 1)
    ]
    assert any(shared)


def test_no_text_is_lost():
    text = "Alpha one. Bravo two. Charlie three. Delta four."
    joined = " ".join(chunk_text(text, target_chars=30, overlap_chars=10))
    for sentence in split_sentences(text):
        assert sentence in joined


def test_oversized_sentence_is_hard_split():
    sentence = " ".join(f"word{index}" for index in range(200))
    chunks = chunk_text(sentence, target_chars=100, overlap_chars=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_overlap_must_be_smaller_than_target():
    with pytest.raises(ValueError):
        chunk_text("anything", target_chars=100, overlap_chars=100)
