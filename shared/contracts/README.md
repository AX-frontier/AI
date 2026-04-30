# Shared Contracts

BE 레포와 AI 레포가 맞춰야 하는 요청/응답 스키마를 둡니다.

## Naming

- `orchestrator.query.schema.json`
- `orchestrator.route.schema.json`
- `main-agent.answer.schema.json`
- `document-review.check.schema.json`
- `library.qa.schema.json`
- `library.book-search.schema.json`
- `ingestion.job.schema.json`

## Common Fields

모든 요청/응답에는 가능한 한 아래 키를 포함합니다.

```json
{
  "query_id": "q_...",
  "trace_id": "tr_...",
  "user_id": "u_..."
}
```

## Rule

- 계약 파일에는 비즈니스 로직을 넣지 않습니다.
- 에이전트 내부 모델과 외부 API 계약을 섞지 않습니다.
- breaking change가 생기면 `docs/architecture`에 변경 이유를 남깁니다.

