from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache
from typing import Protocol

import requests

class EmbeddingProvider(Protocol):
    """질문을 vector DB 검색용 embedding으로 바꾸는 경계."""

    dimensions: int

    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_document(self, text: str) -> list[float]:
        ...


class DeterministicEmbeddingProvider:
    """LLM/embedding API가 붙기 전까지 테스트와 로컬 개발에 쓰는 결정적 embedding."""

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for index in range(self.dimensions):
            byte = digest[index % len(digest)]
            values.append((byte / 255.0) - 0.5)

        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [round(value / norm, 6) for value in values]


class E5EmbeddingProvider:
    """SentenceTransformers 기반 multilingual-e5 embedding provider."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", dimensions: int = 384):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.dimensions = dimensions
        self._model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self._encode(f"query: {text}")

    def embed_document(self, text: str) -> list[float]:
        return self._encode(f"passage: {text}")

    def _encode(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        values = [float(value) for value in vector.tolist()]
        if len(values) != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimensions}, got {len(values)}"
            )
        return values


class GeminiEmbeddingProvider:
    """Google Gemini embedding provider."""

    def __init__(self, model_name: str = "gemini-embedding-2", dimensions: int = 1536):
        from google import genai

        self.model_name = model_name
        self.dimensions = dimensions
        resolved_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for GeminiEmbeddingProvider.")
        self._client = genai.Client(api_key=resolved_api_key)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        response = self._client.models.embed_content(
            model=self.model_name,
            contents=text,
            config={"output_dimensionality": self.dimensions},
        )
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise RuntimeError("Gemini embedding response did not include embeddings.")
        values = [float(value) for value in embeddings[0].values]
        if len(values) != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimensions}, got {len(values)}"
            )
        return values


class OpenAIEmbeddingProvider:
    """OpenAI Embeddings API provider."""

    def __init__(self, model_name: str = "text-embedding-3-small", dimensions: int = 1536):
        self.model_name = model_name
        self.dimensions = dimensions
        resolved_api_key = os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider.")
        self._api_key = resolved_api_key

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "input": text,
                "dimensions": self.dimensions,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        if not data or "embedding" not in data[0]:
            raise RuntimeError("OpenAI embedding response did not include embedding.")
        values = [float(value) for value in data[0]["embedding"]]
        if len(values) != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimensions}, got {len(values)}"
            )
        return values


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """환경변수 기준으로 Main Agent embedding provider를 선택한다."""
    provider = os.getenv("MAIN_AGENT_EMBEDDING_PROVIDER", "deterministic").lower()
    if provider == "e5":
        return E5EmbeddingProvider(
            model_name=os.getenv("MAIN_AGENT_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            dimensions=int(os.getenv("MAIN_AGENT_EMBEDDING_DIMENSIONS", "384")),
        )
    if provider == "gemini":
        return GeminiEmbeddingProvider(
            model_name=os.getenv("MAIN_AGENT_EMBEDDING_MODEL", "gemini-embedding-2"),
            dimensions=int(os.getenv("MAIN_AGENT_EMBEDDING_DIMENSIONS", "1536")),
        )
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            model_name=os.getenv("MAIN_AGENT_EMBEDDING_MODEL", "text-embedding-3-small"),
            dimensions=int(os.getenv("MAIN_AGENT_EMBEDDING_DIMENSIONS", "1536")),
        )

    return DeterministicEmbeddingProvider(
        dimensions=int(os.getenv("MAIN_AGENT_EMBEDDING_DIMENSIONS", "1536"))
    )
