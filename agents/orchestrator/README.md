# Orchestrator

사용자 질의를 받아 적절한 에이전트로 라우팅하는 최상위 제어 영역입니다.

## Scope

- 질의 의도 분석
- 대상 에이전트 선택
- 실행 순서 제어
- fallback 처리
- `query_id`, `trace_id` 흐름 관리

## Folders

- `api/`: BE 연동용 query/route 계약 초안
- `routing/`: 규칙 기반/LLM 기반 라우팅 로직
- `pipeline/`: 에이전트 실행 흐름
- `prompts/`: 의도 분석, 라우팅 보정 프롬프트
- `tests/`: 라우팅 및 fallback 테스트

