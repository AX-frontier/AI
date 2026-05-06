from __future__ import annotations

import hashlib
import math
from typing import Protocol


class EmbeddingProvider(Protocol):
    """질문을 vector DB 검색용 embedding으로 바꾸는 경계."""

    dimensions: int

    def embed_query(self, text: str) -> list[float]:
        ...


class DeterministicEmbeddingProvider:
    """LLM/embedding API가 붙기 전까지 테스트와 로컬 개발에 쓰는 결정적 embedding."""

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for index in range(self.dimensions):
            byte = digest[index % len(digest)]
            values.append((byte / 255.0) - 0.5)

        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [round(value / norm, 6) for value in values]
