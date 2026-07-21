"""OpenAI provider over the REST API.

This calls the two endpoints it needs with httpx instead of pulling in the
vendor SDK. The surface used here is small and stable, and httpx is already a
dependency through FastAPI's test client.
"""

from __future__ import annotations

import httpx

from app.providers.base import ChatResult, EmbeddingResult, estimate_tokens


class OpenAIError(RuntimeError):
    pass


class OpenAIProvider:
    name = "openai"

    # Five of the six are keyword-only with defaults, mapping settings onto the
    # client. Wrapping them in a config object for the single call site in
    # build_provider would add a type and remove nothing.
    def __init__(  # pylint: disable=too-many-arguments
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        dim: int = 1536,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise OpenAIError("OPENAI_API_KEY is required when PROVIDER=openai")
        self.dim = dim
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], tokens=0)
        payload = {
            "model": self.embedding_model,
            "input": texts,
            "dimensions": self.dim,
        }
        data = await self._post("/embeddings", payload)
        rows = sorted(data["data"], key=lambda row: row["index"])
        vectors = [row["embedding"] for row in rows]
        for vector in vectors:
            if len(vector) != self.dim:
                raise OpenAIError(f"model returned {len(vector)} dimensions, expected {self.dim}")
        tokens = int(data.get("usage", {}).get("total_tokens", 0))
        return EmbeddingResult(vectors=vectors, tokens=tokens)

    async def chat(self, system: str, user: str) -> ChatResult:
        payload = {
            "model": self.chat_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = await self._post("/chat/completions", payload)
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        return ChatResult(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", estimate_tokens(system + user))),
            output_tokens=int(usage.get("completion_tokens", estimate_tokens(text))),
        )

    async def _post(self, path: str, payload: dict) -> dict:
        response = await self._client.post(path, json=payload)
        if response.status_code >= 400:
            raise OpenAIError(f"{path} returned {response.status_code}: {response.text[:500]}")
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()
