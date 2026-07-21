# pgvector-rag

A retrieval augmented generation API. It ingests documents, chunks and embeds
them, stores the vectors in PostgreSQL with pgvector, and answers questions
with citations. Redis caches embeddings and answers.

Python 3.12, FastAPI, asyncpg, Redis.
