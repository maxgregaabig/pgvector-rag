# pgvector-rag

[![CI](https://github.com/H4mid2019/pgvector-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/H4mid2019/pgvector-rag/actions/workflows/ci.yml)

A retrieval augmented generation API. It ingests documents, chunks and embeds
them, stores the vectors in PostgreSQL with pgvector, and answers questions
with citations that point back at the exact chunk each claim came from. Redis
caches embeddings and answers. An eval script reports retrieval hit rate,
latency and an estimated token cost.

Python 3.12, FastAPI, asyncpg, Redis.

## What this repo demonstrates

- Vector retrieval in PostgreSQL instead of a separate vector database, with a
  real schema, migrations and an hnsw index.
- A provider interface with two implementations, so the test suite and the
  eval script run with no API key and no network. See `app/providers/base.py`,
  `app/providers/fake.py`, `app/providers/openai.py`.
- Answers that carry citations, and a citation extractor that drops markers the
  model invented
  (`tests/test_service.py::test_out_of_range_markers_are_dropped`).
- Caching with a test that counts provider calls and proves the second
  identical question does not reach the model
  (`tests/test_api.py::test_second_identical_ask_is_cached_and_skips_the_provider`).
- Integration tests against a real pgvector and a real Redis, which skip with a
  message when those are not configured.
- One command to start the whole stack, checked by the `compose` job in CI.

## Run it

```bash
docker compose up
```

That starts PostgreSQL with pgvector, Redis, and the API on
<http://localhost:8000>. Migrations run on startup. The default provider is the
fake one, so it works with no API key. For real models:

```bash
PROVIDER=openai OPENAI_API_KEY=sk-... docker compose up
```

Then:

```bash
curl -s localhost:8000/healthz

curl -s -X POST localhost:8000/v1/documents \
  -H 'content-type: application/json' \
  -d '{"source":"payments-faq.md","text":"The daily payout cutoff is 16:00 UTC. Chargebacks can be raised within 120 days of the transaction date."}'

curl -s -X POST localhost:8000/v1/search \
  -H 'content-type: application/json' \
  -d '{"query":"chargeback deadline","top_k":3}'

curl -s -X POST localhost:8000/v1/ask \
  -H 'content-type: application/json' \
  -d '{"question":"When is the payout cutoff?","top_k":3}'
```

Interactive docs are at <http://localhost:8000/docs>.

## API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/healthz` | none | status and the active store, cache and provider |
| POST | `/v1/documents` | `source`, `text`, optional `title` and `metadata` | `201` with `document_id`, chunk count, embedding tokens billed, embeddings served from cache |
| DELETE | `/v1/documents/{id}` | none | `204`, or `404` if it does not exist |
| POST | `/v1/search` | `query`, optional `top_k` and `document_ids` | ranked chunks with cosine scores and `took_ms` |
| POST | `/v1/ask` | `question`, optional `top_k` and `document_ids` | answer text, citations, token usage, `cached` flag |

`top_k` defaults to `DEFAULT_TOP_K` (5) and is capped at `MAX_TOP_K` (50).
`document_ids` scopes retrieval to a subset of documents.

The `/v1/ask` response to the call above, against the compose stack with the
fake provider, so the wording of the answer is a template:

```json
{
  "question": "When is the payout cutoff?",
  "answer": "Based on the indexed documents: The daily payout cutoff is 16:00 UTC. [1]",
  "citations": [
    {
      "marker": 1,
      "chunk_id": "380a00b1-228c-492f-ba0d-63c5dd673a37",
      "document_id": "1e5c0088-ad3f-4699-9e67-8992adcad3fe",
      "source": "payments-faq.md",
      "ordinal": 0,
      "score": 0.50023
    }
  ],
  "cached": false,
  "usage": {"embedding_tokens": 7, "chat_input_tokens": 108, "chat_output_tokens": 19},
  "took_ms": 2.008
}
```

Repeating that request returns the same body with `"cached": true` and, in
this run, `took_ms` of 0.326 instead of 2.008.

## Architecture

```text
client -> FastAPI (app/main.py)
             |
             +-- RagService (app/service.py)
                   |
                   +-- Provider     embeddings and chat   (openai | fake)
                   +-- Cache        embeddings, answers   (redis | memory)
                   +-- VectorStore  chunks and search     (postgres | memory)
```

Each of the three collaborators is a `Protocol` with two implementations, and
the service never imports a concrete one. That is what makes the default test
run independent of Postgres, Redis and the network, and it is also how you
would swap in a different model vendor.

Layout:

```text
app/
  main.py        FastAPI app factory, routes, lifespan wiring
  service.py     ingest, search, ask, prompt building, citation extraction
  chunking.py    sentence packing with overlap
  cache.py       Cache protocol, Redis and memory, key derivation
  config.py      pydantic-settings
  models.py      request and response schemas, internal records
  providers/     Provider protocol, OpenAI over httpx, deterministic fake
  store/         VectorStore protocol, pgvector store, memory store
migrations/      001_init.sql, applied on startup, tracked in schema_migrations
eval/            eval runner plus the fixture corpus and question set
tests/           unit tests plus opt-in integration tests
```

### Data model

```sql
documents(id uuid pk, source text, title text, metadata jsonb, created_at timestamptz)

chunks(id uuid pk,
       document_id uuid references documents on delete cascade,
       ordinal int,
       content text,
       content_hash text,
       token_estimate int,
       embedding vector(1536),
       created_at timestamptz,
       unique (document_id, ordinal))
```

Indexes: `chunks(document_id)`, `chunks(content_hash)`, and
`chunks USING hnsw (embedding vector_cosine_ops)` with `m = 16` and
`ef_construction = 64`.

Search is `ORDER BY embedding <=> $query LIMIT k`, and the reported score is
`1 - cosine_distance`. `ordinal` keeps the chunk order inside a document and is
what the `source#ordinal` citation label refers to.

## Design decisions

**pgvector rather than a dedicated vector database.** For a corpus that lives
next to data already in Postgres, keeping the vectors there means one backup,
one connection pool, one transaction and metadata filters that are ordinary
SQL rather than a second query language. `add_document` writes the document row
and all its chunk rows in one transaction, so a failed ingest leaves nothing
behind. A dedicated engine wins on scale and on index features once you are
past roughly ten million vectors, or when you need sharding or filtered ANN
with complex predicates. Below that, the operational cost of a second stateful
system is the larger number.

**hnsw rather than ivfflat.** ivfflat has to see representative rows before it
can pick centroids, so it must be built after the data is loaded and rebuilt as
the distribution shifts. That does not fit a migration that runs against an
empty database on startup. hnsw builds incrementally, gives better recall per
unit of query latency, and its recall knob (`hnsw.ef_search`) is set per query
rather than baked into the index. The cost is a slower build and more memory.
If ingest throughput mattered more than query recall, ivfflat with a post load
build step would be the better trade.

**Sentence packing with overlap for chunking.** Chunks are built by packing
whole sentences up to a target size, then carrying trailing whole sentences
into the next chunk as overlap. A fact split across a boundary still appears
intact in one chunk, and no chunk starts mid sentence, which keeps the text
that reaches the model readable. Fixed size character windows are simpler but
cut sentences in half. Semantic chunking with an embedding based boundary
detector is better but costs extra model calls per document. The size is a
setting rather than a constant because it is the main retrieval quality knob;
the eval section below shows what changing it does.

**Cache at the service layer, not at HTTP or SQL.** Two things are cached:
chunk embeddings keyed by model, dimension and content hash, and whole answers
keyed by model, question, `top_k` and document scope. Both are the points where
money is spent. An HTTP response cache in front of the API would key on the
whole request and would miss the cross document reuse of identical paragraphs.
A query cache below the store would only save the vector search, which is the
cheap part. Search results are deliberately not cached: they change whenever a
document is ingested, and invalidating them correctly costs more than the few
milliseconds it saves. Answer entries expire after an hour and embeddings after
a week, so new documents reach answers without explicit invalidation.

**A provider interface with a deterministic fake.** The fake embeds text with
the hashing trick: every word, plus character n-grams of that word at a lower
weight, hashes to one dimension with a stable sign, and the vector is L2
normalised. It is lexical, not semantic, so "chargeback deadline" matches a
chunk about chargebacks but "when do I get paid" will not match "payout
cutoff". That is enough to exercise ranking, citations and caching end to end
without a key, and the fake counts its own calls so tests can assert that a
cache hit skipped it.

**Raw asyncpg and httpx rather than an ORM and a vendor SDK.** The queries here
are four statements, one of which uses an operator no ORM models well, and the
OpenAI surface used is two endpoints. Both dependencies would cost more than
they save at this size.

**Token counts are estimated at four characters per token.** A real tokenizer
would download vocabulary files at runtime, which breaks the offline
requirement. With the OpenAI provider the counts come from the API response
instead of the estimate.

## Eval

`eval/run_eval.py` ingests four fixture documents, runs 15 fixture questions and
checks whether the chunk holding the expected answer snippet appears in the top
k. Unless the environment says otherwise it uses the memory store, the memory
cache and the fake provider, so it runs offline.

```bash
make eval
# or
python -m eval.run_eval
```

Output from a run in this repo, at the default chunk size:

```text
store=memory cache=memory provider=fake
chunking=900/150 chars, top_k=5
corpus: 4 documents, 5 chunks, 15 questions (ingest 4.5 ms)

  hit rate @1   100.00%
  hit rate @3   100.00%
  hit rate @5   100.00%
  MRR         1.0000

  search  p50     1.01   p95     1.37
  ask     p50     0.95   p95     1.51
  ask (cached) p50     0.02

  embedding tokens        885
  chat input tokens     15040
  chat output tokens      992
  estimated cost     $0.002869
```

Read that with care. At 900 character chunks the four fixture documents produce
only 5 chunks, so the top 5 is the whole corpus and a perfect hit rate says
very little. The smaller chunk size is the informative run:

```text
$ python -m eval.run_eval --chunk-chars 300 --chunk-overlap 60
chunking=300/60 chars, top_k=5
corpus: 4 documents, 16 chunks, 15 questions

  hit rate @1    93.33%
  hit rate @3   100.00%
  hit rate @5   100.00%
  MRR         0.9667
```

The same run against the real stack, with `docker compose up -d postgres redis`
running:

```text
store=postgres cache=redis provider=fake
chunking=300/60 chars, top_k=5
corpus: 4 documents, 16 chunks, 15 questions (ingest 225.3 ms)

  hit rate @1    93.33%
  hit rate @3   100.00%
  hit rate @5   100.00%
  MRR         0.9667

  search  p50     2.29   p95     7.18
  ask     p50     2.69   p95     3.05
  ask (cached) p50     0.44
```

Retrieval quality is the same against pgvector and against the brute force
memory store, which is the point: the memory store is a faithful stand-in for
the tests. The latency numbers come from a Windows laptop running Docker
Desktop against 16 chunks, they measure the in process call rather than HTTP,
and they say nothing about pgvector at any real scale. The cost figure is what
these token counts would cost at the prices in `app/config.py`, applied to
counts the fake provider estimated.

## Tests

The default suite needs no database, no Redis, no API key and no network:

```bash
make test
# or
pytest
```

```text
37 passed, 7 skipped
```

The 7 skips are the integration tests, and they say why:

```text
SKIPPED tests/test_integration_postgres.py:69: DATABASE_URL is not set, skipping
PostgreSQL integration tests. Run 'docker compose up -d postgres redis' and
export DATABASE_URL to enable them.
```

To run everything, including the tests that use a real pgvector and a real
Redis:

```bash
make test-integration
# or
docker compose up -d postgres redis
DATABASE_URL=postgresql://rag:rag@localhost:5432/rag \
REDIS_URL=redis://localhost:6379/0 pytest
```

```text
44 passed
```

Lint and format:

```bash
make lint     # ruff check . and ruff format --check .
make format   # apply both
```

CI runs four jobs: lint, the unit suite with no services configured, the full
suite against `pgvector/pgvector:pg17` and `redis:7-alpine` service containers,
and a job that runs `docker compose up -d --build` and then ingests, searches
and asks over HTTP.

## Configuration

Set through the environment or a `.env` file. See `.env.example`.

| Variable | Default | Notes |
| --- | --- | --- |
| `STORE_BACKEND` | `postgres` | `postgres` or `memory` |
| `CACHE_BACKEND` | `redis` | `redis` or `memory` |
| `PROVIDER` | `fake` | `openai` or `fake` |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` | |
| `REDIS_URL` | `redis://localhost:6379/0` | |
| `OPENAI_API_KEY` | empty | required when `PROVIDER=openai` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | part of the embedding cache key |
| `CHAT_MODEL` | `gpt-4o-mini` | part of the answer cache key |
| `EMBEDDING_DIM` | `1536` | must match `vector(1536)` in the migration |
| `CHUNK_TARGET_CHARS` | `900` | |
| `CHUNK_OVERLAP_CHARS` | `150` | must be smaller than the target |
| `DEFAULT_TOP_K` | `5` | |
| `ANSWER_CACHE_TTL_SECONDS` | `3600` | |
| `EMBEDDING_CACHE_TTL_SECONDS` | `604800` | |
| `MIGRATIONS_DIR` | repo `migrations/` | set to `/app/migrations` in the image |

## Not implemented

Gaps a real deployment would have to close:

- No authentication, authorisation or tenant isolation. Every caller sees every
  document.
- No streaming. `/v1/ask` returns the whole answer at once.
- No reranking and no hybrid search. Retrieval is pure dense vector search, so
  exact keyword and identifier lookups are weaker than a `tsvector` plus vector
  hybrid would be.
- No document update path. Re-ingesting the same source creates a second
  document, and you have to delete the old one yourself.
- Ingestion is synchronous and embeds every chunk in a single provider call.
  Large documents belong on a queue with batching and retries.
- No retry, timeout budget or circuit breaker around the provider. A provider
  outage surfaces as a 500.
- Token counts under the fake provider are a four characters per token
  estimate, so the cost figure is an approximation.
- `hnsw.ef_search` is left at the pgvector default, not tuned and not exposed.
- No text extraction. `/v1/documents` takes plain text, not PDF or HTML.
- No metrics or tracing endpoint. There is structured logging only.
- The eval corpus is 4 documents and 15 questions. It is a regression check on
  the pipeline, not a measurement of retrieval quality at scale.
