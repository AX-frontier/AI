# Week 1-2 Plan

## Schedule

| 영역 | 1주차 (5/6~5/13) | 2주차 (5/13~5/20) |
| --- | --- | --- |
| 메인 오케스트레이터 | 라우팅 기준, 계약, `query_id`/`trace_id` 흐름 설계 | 서브 에이전트 연결 및 fallback 보완 |
| 메인 에이전트 | JSON/Markdown 크롤링 데이터 수신, 질의 파이프라인 구축 | 서버 임베딩, 응답 속도 보완, 캐싱 전략 검토 |
| 도서 에이전트 | RAG 파이프라인 구축, ERD 정제 | API 완성 및 Orchestrator 연결 |
| 문서 에이전트 | 문서 규칙 데이터 구조 설계, 검토 항목 정의 | 검토 API 초안 및 수정 제안 흐름 구축 |
| 크롤링 | 한성 공지 6개월, 학술정보관 전체 6개월 크롤링 | 증분 수집, 청킹, 임베딩 상태 관리 |

## Week 1 Deliverables

- Orchestrator 라우팅 기준 초안
- Main Agent 질의 파이프라인 초안
- Library Agent RAG 파이프라인 초안
- Document Review 규칙 스키마 초안
- 한성 공지/학술정보관 크롤링 샘플 데이터
- BE 연동용 API 요청/응답 계약 초안

## Week 2 Deliverables

- Orchestrator fallback 정책
- Main Agent 임베딩/캐싱 전략
- Library Agent 도서 검색 API 연결안
- Document Review 검토 결과/수정 제안 응답 포맷
- Ingestion 증분 수집 및 재색인 작업 흐름

