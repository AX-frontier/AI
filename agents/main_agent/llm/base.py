from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """검색 context 기반 답변 생성을 위한 LLM 경계."""

    def generate(self, prompt: str) -> str:
        ...
