from __future__ import annotations

import os
from functools import lru_cache

from agents.main_agent.llm.base import LLMClient
from agents.main_agent.llm.openAi import OpenAILLMClient
from agents.main_agent.llm.mock import MockLLMClient


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """환경변수 기준으로 Main Agent LLM client를 선택한다."""
    provider = os.getenv("MAIN_AGENT_LLM_PROVIDER", "mock").lower()
    if provider == "openai":
        return OpenAILLMClient()
    return MockLLMClient()
