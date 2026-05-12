from __future__ import annotations

import os
from typing import Iterator


class GeminiLLMClient:
    """Google Gemini API 기반 LLM client."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai

        self.model = model or os.getenv("MAIN_AGENT_LLM_MODEL", "gemini-2.5-flash")
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for GeminiLLMClient.")
        self._client = genai.Client(api_key=resolved_api_key)

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini response did not include text.")
        return text.strip()

    def generate_stream(self, prompt: str) -> Iterator[str]:
        for chunk in self._client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield text
