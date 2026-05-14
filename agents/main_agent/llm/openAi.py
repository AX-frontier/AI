from __future__ import annotations

import os
from typing import Iterator

import requests


class OpenAILLMClient:
    """OpenAI Responses API 기반 LLM client."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.model = model or os.getenv("MAIN_AGENT_LLM_MODEL", "gpt-4.1-mini")
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAILLMClient.")
        self._api_key = resolved_api_key

    def generate(self, prompt: str) -> str:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": prompt,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("output_text")
        if not text:
            output = payload.get("output") or []
            parts: list[str] = []
            for item in output:
                for content in item.get("content", []) or []:
                    value = content.get("text")
                    if value:
                        parts.append(str(value))
            text = "".join(parts).strip()
        if not text:
            raise RuntimeError("OpenAI response did not include output text.")
        return text.strip()

    def generate_stream(self, prompt: str) -> Iterator[str]:
        # Keep streaming contract without adding SSE parser complexity.
        yield self.generate(prompt)
