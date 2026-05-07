from __future__ import annotations


class MockLLMClient:
    """테스트와 fallback에 사용하는 deterministic LLM client."""

    def __init__(self, answer: str | None = None, should_fail: bool = False):
        self.answer = answer or "검색된 근거를 바탕으로 답변합니다."
        self.should_fail = should_fail

    def generate(self, prompt: str) -> str:
        if self.should_fail:
            raise RuntimeError("mock llm failure")
        return self.answer
