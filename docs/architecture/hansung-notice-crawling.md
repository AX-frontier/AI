# Hansung Notice Crawling

한성대학교 공지사항은 같은 게시판 엔진을 사용하고, 카테고리는 `findCclSeq` 값으로 구분한다.

## Source

- 게시판: 한성대학교 공지사항
- 목록 URL: `https://www.hansung.ac.kr/hansung/6172/subview.do`
- 상세 URL 패턴: `https://www.hansung.ac.kr/bbs/hansung/2127/{notice_id}/artclView.do`
- 카테고리 설정: `ingestion/crawlers/hansung-notice/sources.json`

## Categories

| Key | Name | findCclSeq |
| --- | --- | --- |
| `all` | 전체공지 | |
| `stay_study` | Stay&Study in School 장학사업 | `333` |
| `hansung` | 한성공지 | `334` |
| `academic` | 학사공지 | `335` |
| `extracurricular` | 비교과공지 | `336` |
| `field_training` | 현장실습공지 | `337` |
| `career` | 진로공지 | `338` |
| `employment` | 취업공지 | `339` |
| `scholarship` | 장학공지 | `346` |
| `startup` | 창업공지 | `355` |
| `international` | 국제공지 | `360` |
| `covid19` | 코로나19공지 | `361` |

## Output Contract

공지 1건은 JSON 메타데이터와 Markdown 본문으로 분리한다.

```text
data/raw/hansung-notice/
├── errors.jsonl
├── json/
│   └── {category_key}/
│       ├── {notice_id}.json
│       └── {notice_id}.md.metadata.json
└── markdown/
    └── {category_key}/
        └── {notice_id}.md
```

## Run

의존성은 `uv`가 `.venv`에 설치한다.

```bash
uv run python -m ingestion.crawlers.hansung_notice.main --category academic --max-pages 1
```

최근 6개월처럼 기간을 제한할 때는 `--since-date`를 사용한다.

```bash
uv run python -m ingestion.crawlers.hansung_notice.main \
  --category all \
  --since-date 2025-11-02 \
  --max-pages 999
```

여러 카테고리를 한 번에 돌릴 때는 `--category`를 반복한다.

```bash
uv run python -m ingestion.crawlers.hansung_notice.main \
  --category hansung \
  --category academic \
  --category scholarship \
  --max-pages 3
```

이미지 본문만 있는 공지는 기본적으로 제외한다. 필요하면 아래 옵션으로 포함할 수 있다.

```bash
uv run python -m ingestion.crawlers.hansung_notice.main --include-image-only
```

## Metadata Fields

```json
{
  "notice_id": "221894",
  "source": "hansung_notice",
  "category": "장학공지",
  "source_category_key": "scholarship",
  "source_category_name": "장학공지",
  "title": "공지 제목",
  "department": "학생복지팀",
  "published_at": "2026-04-29",
  "views": 103,
  "source_url": "https://www.hansung.ac.kr/bbs/hansung/2127/221894/artclView.do",
  "content_path": "markdown/scholarship/221894.md",
  "attachments": [
    {
      "name": "첨부파일.pdf",
      "url": "https://www.hansung.ac.kr/..."
    }
  ],
  "collected_at": "2026-04-30T00:00:00+09:00"
}
```

## Markdown Fields

Markdown 파일에는 아래 정보를 포함한다.

- 제목
- 공지 ID
- 카테고리
- 작성 부서
- 작성일
- 조회수
- 원문 URL
- 수집 시각
- 첨부파일
- 본문 Markdown

## Parser Notes

- 목록의 상단 고정 공지와 일반 공지를 모두 수집한다.
- 6개월 종료 판단에서는 모든 페이지에 반복되는 상단 고정 공지를 제외하고 일반 공지 날짜만 본다.
- `category`는 상세 페이지에 표시된 실제 카테고리이고, `source_category_key`는 해당 공지를 발견한 목록 카테고리이다.
- `notice_id`는 목록 번호가 아니라 상세 URL의 `{notice_id}`에서 추출한다.
- 이미지 본문만 있고 텍스트가 없는 공지는 우선 제외한다.
- 상세 페이지의 원문 HTML과 변환된 Markdown을 모두 보관할 수 있게 설계한다.
- `td`/`th`가 1개뿐인 `table`은 안내 박스일 가능성이 높으므로 Markdown 변환 전에 일반 블록으로 바꾼다.
- `th`가 없는 실제 표는 첫 번째 행을 헤더로 승격해 Markdown 표 깨짐을 줄인다.

## Incremental Crawl

- 이미 저장된 `notice_id`는 skip한다.
- 이미 저장된 공지가 연속으로 일정 개수 이상 나오면 종료한다.
- 기본 종료 기준은 `existing_notice_stop_threshold = 5`로 둔다.
- 실패 항목은 `errors.jsonl`에 기록하고 retry 작업에서 재처리한다.
