# Main Agent

학교 공지, 학사 정보, 주요 안내 페이지를 기반으로 일반 학교 정보 질문에 답하는 에이전트입니다.

## Scope

- JSON/Markdown 크롤링 데이터 수신
- 학교 정보 RAG 검색
- 공식 정보 요약
- 출처 링크 제공
- 캐싱 및 검색 속도 개선

## Folders

- `api/`: 일반 학교 정보 QA 계약 초안
- `pipeline/`: 검색, 생성, 출처 정리 흐름
- `prompts/`: 학교 정보 응답 프롬프트
- `retrieval/`: 검색, rerank, cache 관련 코드
- `tests/`: 검색/응답 테스트

