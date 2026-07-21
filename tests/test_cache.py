from uuid import uuid4

from app.cache import MemoryCache, answer_key, content_hash, embedding_key


def test_embedding_key_depends_on_model_dim_and_text():
    base = embedding_key("text-embedding-3-small", 1536, "hello")
    assert base == embedding_key("text-embedding-3-small", 1536, "hello")
    assert base != embedding_key("text-embedding-3-large", 1536, "hello")
    assert base != embedding_key("text-embedding-3-small", 256, "hello")
    assert base != embedding_key("text-embedding-3-small", 1536, "hello there")


def test_answer_key_ignores_case_and_document_id_order():
    ids = [uuid4(), uuid4()]
    assert answer_key("m", "What is the cutoff?", 5, ids) == answer_key(
        "m", "  what is the CUTOFF? ", 5, list(reversed(ids))
    )


def test_answer_key_changes_with_top_k_and_scope():
    ids = [uuid4()]
    assert answer_key("m", "q", 5, None) != answer_key("m", "q", 6, None)
    assert answer_key("m", "q", 5, None) != answer_key("m", "q", 5, ids)


def test_content_hash_is_stable():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


async def test_memory_cache_roundtrip():
    cache = MemoryCache()
    assert await cache.get("missing") is None
    await cache.set("k", "v", 60)
    assert await cache.get("k") == "v"
    await cache.aclose()
    assert await cache.get("k") is None


async def test_memory_cache_honours_ttl():
    cache = MemoryCache()
    await cache.set("k", "v", 0)
    assert await cache.get("k") is None


async def test_memory_cache_is_bounded():
    # Without a bound this dict would grow for the life of the process, since
    # every distinct question and chunk body produces a new key.
    cache = MemoryCache(max_entries=10)
    for i in range(50):
        await cache.set(f"k{i}", str(i), 60)

    assert len(cache._data) == 10  # pylint: disable=protected-access
    assert await cache.get("k0") is None
    assert await cache.get("k49") == "49"


async def test_memory_cache_eviction_is_least_recently_used():
    cache = MemoryCache(max_entries=2)
    await cache.set("a", "1", 60)
    await cache.set("b", "2", 60)
    await cache.get("a")  # makes "b" the least recently used
    await cache.set("c", "3", 60)

    assert await cache.get("a") == "1"
    assert await cache.get("b") is None
    assert await cache.get("c") == "3"
