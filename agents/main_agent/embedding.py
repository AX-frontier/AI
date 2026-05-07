from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache
from typing import Protocol


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


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """환경변수 기준으로 Main Agent embedding provider를 선택한다."""
    provider = os.getenv("MAIN_AGENT_EMBEDDING_PROVIDER", "deterministic").lower()
    if provider == "e5":
        return E5EmbeddingProvider(
            model_name=os.getenv("MAIN_AGENT_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            dimensions=int(os.getenv("MAIN_AGENT_EMBEDDING_DIMENSIONS", "384")),
        )

    return DeterministicEmbeddingProvider(
        dimensions=int(os.getenv("MAIN_AGENT_EMBEDDING_DIMENSIONS", "1536"))
    )
