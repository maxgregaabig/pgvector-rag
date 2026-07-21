"""Retrieval and cost evaluation.

Ingests the fixture documents, runs every fixture question through search and
ask, and reports:

* hit rate at k: the share of questions where a chunk containing the expected
  snippet, from the expected source, appears in the top k results.
* p50 and p95 latency for the search and ask calls, measured in this process.
* token counts and an estimated cost from the per million token prices in the
  settings.

Unless the environment says otherwise this uses the memory store, the memory
cache and the fake provider, so it runs offline. Set STORE_BACKEND,
CACHE_BACKEND and PROVIDER to evaluate the real backends.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --k 1 3 5 --json eval/out/report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.cache import build_cache
from app.config import Settings
from app.providers import build_provider
from app.service import RagService
from app.store import build_store

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The service defaults to Postgres and Redis. The eval defaults to the in
# process backends instead, so that running it needs nothing.
OFFLINE_DEFAULTS = {
    "STORE_BACKEND": "memory",
    "CACHE_BACKEND": "memory",
    "PROVIDER": "fake",
}


def apply_offline_defaults() -> None:
    """Fill in the in process backends for any variable the caller left unset."""
    for key, value in OFFLINE_DEFAULTS.items():
        os.environ.setdefault(key, value)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest rank percentile. Returns 0.0 for an empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(fraction * len(ordered) + 0.5)))
    return ordered[rank - 1]


@dataclass
class IngestStats:
    chunks: int = 0
    embedding_tokens: int = 0
    elapsed_ms: float = 0.0


@dataclass
class QuestionStats:  # pylint: disable=too-many-instance-attributes
    """Everything measured while running the question set.

    A flat record of counters and samples. Splitting it into retrieval, latency
    and token sub-objects would satisfy the attribute count and make every read
    site longer for nothing.
    """

    hits: dict[int, int] = field(default_factory=dict)
    reciprocal_rank_sum: float = 0.0
    found: int = 0
    misses: list[str] = field(default_factory=list)
    search_latencies: list[float] = field(default_factory=list)
    ask_latencies: list[float] = field(default_factory=list)
    cached_ask_latencies: list[float] = field(default_factory=list)
    embedding_tokens: int = 0
    chat_input_tokens: int = 0
    chat_output_tokens: int = 0


async def ingest_corpus(service: RagService, documents: list[dict]) -> IngestStats:
    """Load the fixture documents and report what it cost."""
    stats = IngestStats()
    started = time.perf_counter()
    for document in documents:
        result = await service.ingest(
            source=document["source"],
            text=document["text"],
            title=document.get("title"),
        )
        stats.chunks += result.chunks
        stats.embedding_tokens += result.embedding_tokens
    stats.elapsed_ms = (time.perf_counter() - started) * 1000
    return stats


def rank_of_expected(results, expected_source: str, expected_snippet: str) -> int | None:
    """1-based position of the first result matching the fixture, or None."""
    for index, result in enumerate(results, start=1):
        if result.source == expected_source and expected_snippet in result.content:
            return index
    return None


async def measure_questions(
    service: RagService,
    questions: list[dict],
    k_values: list[int],
    warm_cache: bool,
) -> QuestionStats:
    """Run search and ask for every question, timing both."""
    stats = QuestionStats(hits=dict.fromkeys(k_values, 0))
    max_k = max(k_values)

    for item in questions:
        started = time.perf_counter()
        search = await service.search(item["question"], top_k=max_k)
        stats.search_latencies.append((time.perf_counter() - started) * 1000)

        rank = rank_of_expected(search.results, item["source"], item["snippet"])
        if rank is None:
            stats.misses.append(item["question"])
        else:
            stats.found += 1
            stats.reciprocal_rank_sum += 1.0 / rank
            for k in k_values:
                if rank <= k:
                    stats.hits[k] += 1

        started = time.perf_counter()
        answer = await service.ask(item["question"], top_k=max_k)
        stats.ask_latencies.append((time.perf_counter() - started) * 1000)
        stats.embedding_tokens += answer.usage.embedding_tokens
        stats.chat_input_tokens += answer.usage.chat_input_tokens
        stats.chat_output_tokens += answer.usage.chat_output_tokens

        if warm_cache:
            started = time.perf_counter()
            repeat = await service.ask(item["question"], top_k=max_k)
            stats.cached_ask_latencies.append((time.perf_counter() - started) * 1000)
            if not repeat.cached:
                raise RuntimeError(f"expected a cache hit for {item['question']!r}")

    return stats


def build_report(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    settings: Settings,
    names: dict[str, str],
    k_values: list[int],
    document_count: int,
    question_count: int,
    ingest: IngestStats,
    stats: QuestionStats,
) -> dict:
    """Assemble the JSON report. Pure: every input is a value, nothing is read
    from the environment or the network here."""
    embedding_tokens = ingest.embedding_tokens + stats.embedding_tokens
    cost = (
        embedding_tokens * settings.embedding_price_per_mtok
        + stats.chat_input_tokens * settings.chat_input_price_per_mtok
        + stats.chat_output_tokens * settings.chat_output_price_per_mtok
    ) / 1_000_000

    return {
        "config": {
            **names,
            "embedding_model": settings.embedding_model,
            "chat_model": settings.chat_model,
            "chunk_target_chars": settings.chunk_target_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
            "top_k": max(k_values),
        },
        "corpus": {
            "documents": document_count,
            "chunks": ingest.chunks,
            "questions": question_count,
            "ingest_ms": round(ingest.elapsed_ms, 1),
        },
        "retrieval": {
            "hit_rate": {str(k): round(stats.hits[k] / question_count, 4) for k in k_values},
            "mrr": round(stats.reciprocal_rank_sum / question_count, 4),
            "found": stats.found,
            "misses": stats.misses,
        },
        "latency_ms": {
            "search_p50": round(statistics.median(stats.search_latencies), 2),
            "search_p95": round(percentile(stats.search_latencies, 0.95), 2),
            "ask_p50": round(statistics.median(stats.ask_latencies), 2),
            "ask_p95": round(percentile(stats.ask_latencies, 0.95), 2),
            "ask_cached_p50": (
                round(statistics.median(stats.cached_ask_latencies), 2)
                if stats.cached_ask_latencies
                else None
            ),
        },
        "tokens": {
            "embedding": embedding_tokens,
            "chat_input": stats.chat_input_tokens,
            "chat_output": stats.chat_output_tokens,
        },
        "estimated_cost_usd": round(cost, 6),
    }


def resolve_settings(chunk_chars: int | None, chunk_overlap: int | None) -> Settings:
    """Settings from the environment, with the chunking flags applied on top."""
    settings = Settings()
    overrides = {}
    if chunk_chars is not None:
        overrides["chunk_target_chars"] = chunk_chars
    if chunk_overlap is not None:
        overrides["chunk_overlap_chars"] = chunk_overlap
    return settings.model_copy(update=overrides) if overrides else settings


async def run(
    k_values: list[int],
    warm_cache: bool,
    chunk_chars: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    """Run the whole evaluation and return the report."""
    settings = resolve_settings(chunk_chars, chunk_overlap)
    store = build_store(settings)
    cache = build_cache(settings)
    provider = build_provider(settings)

    documents = json.loads((FIXTURES / "documents.json").read_text(encoding="utf-8"))
    questions = json.loads((FIXTURES / "questions.json").read_text(encoding="utf-8"))

    try:
        await store.setup()
        service = RagService(settings, store, cache, provider)
        ingest = await ingest_corpus(service, documents)
        stats = await measure_questions(service, questions, k_values, warm_cache)
        return build_report(
            settings,
            {"store": store.name, "cache": cache.name, "provider": provider.name},
            k_values,
            len(documents),
            len(questions),
            ingest,
            stats,
        )
    finally:
        await store.aclose()
        await cache.aclose()
        await provider.aclose()


def format_report(report: dict) -> str:
    config = report["config"]
    corpus = report["corpus"]
    latency = report["latency_ms"]
    tokens = report["tokens"]
    lines = [
        "pgvector-rag eval",
        "=================",
        f"store={config['store']} cache={config['cache']} provider={config['provider']}",
        f"chunking={config['chunk_target_chars']}/{config['chunk_overlap_chars']} chars, "
        f"top_k={config['top_k']}",
        f"corpus: {corpus['documents']} documents, {corpus['chunks']} chunks, "
        f"{corpus['questions']} questions (ingest {corpus['ingest_ms']} ms)",
        "",
        "retrieval",
        "---------",
    ]
    for k, rate in report["retrieval"]["hit_rate"].items():
        lines.append(f"  hit rate @{k:<3} {rate:.2%}")
    lines.append(f"  MRR         {report['retrieval']['mrr']:.4f}")
    for miss in report["retrieval"]["misses"]:
        lines.append(f"  miss: {miss}")
    lines += [
        "",
        "latency (ms, in process, excludes HTTP)",
        "---------------------------------------",
        f"  search  p50 {latency['search_p50']:>8.2f}   p95 {latency['search_p95']:>8.2f}",
        f"  ask     p50 {latency['ask_p50']:>8.2f}   p95 {latency['ask_p95']:>8.2f}",
    ]
    if latency["ask_cached_p50"] is not None:
        lines.append(f"  ask (cached) p50 {latency['ask_cached_p50']:>8.2f}")
    lines += [
        "",
        "tokens and cost",
        "---------------",
        f"  embedding tokens   {tokens['embedding']:>8}",
        f"  chat input tokens  {tokens['chat_input']:>8}",
        f"  chat output tokens {tokens['chat_output']:>8}",
        f"  estimated cost     ${report['estimated_cost_usd']:.6f}",
        "  token counts from the fake provider are a 4 chars per token estimate",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality, latency and cost.")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5], help="k values to report")
    parser.add_argument(
        "--no-warm-cache",
        action="store_true",
        help="skip the repeated question that measures cached answer latency",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    parser.add_argument(
        "--chunk-chars", type=int, default=None, help="override CHUNK_TARGET_CHARS for this run"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=None, help="override CHUNK_OVERLAP_CHARS for this run"
    )
    args = parser.parse_args()

    apply_offline_defaults()
    report = asyncio.run(
        run(
            sorted(set(args.k)),
            warm_cache=not args.no_warm_cache,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
        )
    )
    print(format_report(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
