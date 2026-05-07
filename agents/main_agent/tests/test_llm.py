from __future__ import annotations

from agents.main_agent.llm.mock import MockLLMClient


def test_mock_llm_client_returns_configured_answer() -> None:
    client = MockLLMClient("테스트 답변")

    assert client.generate("prompt") == "테스트 답변"
