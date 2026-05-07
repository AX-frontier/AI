# Ingestion

학교와 학술정보관 데이터를 수집하고 검색 가능한 형태로 변환하는 영역입니다.

## Scope

- 한성대학교 공지 6개월 크롤링
- 학술정보관 전체 6개월 크롤링
- 원문 저장
- 문서 청킹
- 임베딩 생성
- 증분 업데이트
- 재색인 작업

## Folders

- `crawlers/`: 웹 크롤러
- `chunking/`: 문서 분할 및 정제
- `embedding/`: 임베딩 생성과 vector store 적재
- `jobs/`: 수집/재색인 작업 단위
- `tests/`: 수집/정제 테스트

## Sources

- `crawlers/hansung-notice/sources.json`: 한성대학교 공지사항 카테고리 URL 목록

## Hansung Notice Crawler

이미지 본문만 있는 공지는 우선 제외하고, 텍스트/표 기반 공지를 Markdown과 JSON으로 저장합니다.

```bash
python -m ingestion.crawlers.hansung_notice.main --category academic --max-pages 1
```

출력 위치:

```text
data/raw/hansung-notice/
├── errors.jsonl
├── json/
│   └── <category>/
│       ├── <notice_id>.json
│       └── <notice_id>.md.metadata.json
└── markdown/
    └── <category>/
        └── <notice_id>.md
```

## Load Hansung Notices to pgvector

크롤링 산출물의 Markdown과 metadata JSON을 chunk로 바꾼 뒤, Spring BE Docker PostgreSQL의
`main_agent.document_chunks` 테이블에 upsert합니다.

DB 없이 chunk 변환만 확인:

```bash
python -m ingestion.jobs.load_hansung_notice_vectors \
  --input-dir data/raw/hansung-notice \
  --dry-run
```

BE PostgreSQL에 적재:

```bash
export DATABASE_URL=postgresql://ax_prontier:ax_prontier@localhost:5432/ax_prontier

python -m ingestion.jobs.load_hansung_notice_vectors \
  --input-dir data/raw/hansung-notice \
  --init-schema
```

현재 임베딩은 실제 모델 연동 전 로컬 검증용 `DeterministicEmbeddingProvider(1536)`를 사용합니다.
실제 서비스 임베딩 모델이 정해지면 provider만 교체하고, `data/schemas/main_agent.sql`의
`vector(1536)` 차원도 모델 차원에 맞춥니다.
