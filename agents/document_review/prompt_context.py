from __future__ import annotations

from functools import lru_cache
from pathlib import Path


PROMPT_CONTEXT_FILES = (
    ("전자결재 프롬프트 ver1", "electronic_approval_prompt_ver1.txt"),
    ("전자결재 프롬프트 ver1-gpt가 작성", "electronic_approval_prompt_ver1_gpt.txt"),
)


@lru_cache(maxsize=1)
def load_manual_prompt_context() -> str:
    """HWP 매뉴얼에서 추출한 원문 텍스트를 LLM 최종 검토 컨텍스트로 로드한다."""
    prompt_dir = Path(__file__).with_name("prompts")
    sections: list[str] = []
    for title, filename in PROMPT_CONTEXT_FILES:
        path = prompt_dir / filename
        try:
            text = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if text:
            sections.append(f"## {title}\n{text}")
    return "\n\n".join(sections)
