# 전자결재 문서검토 매뉴얼 이식 현황

기준 문서:

- `전자결재 프롬프트 ver1.hwp`: 원 매뉴얼
- `전자결재 프롬프트 ver1-gpt가 작성.hwp`: 원 매뉴얼을 프롬프트화한 지침

이 에이전트는 복붙 텍스트와 웹 에디터 HTML로 확인 가능한 규칙은 자동 수정 또는 확인 항목으로 반환하고, 글꼴/줄간격/표 위치처럼 텍스트만으로 확정할 수 없는 항목은 `서식 참고` 또는 `확인 필요`로 반환한다.

| 매뉴얼 항목 | 처리 방식 | 구현 위치 |
| --- | --- | --- |
| 문서는 어문규범에 맞는 한글 표현으로 작성 | 확인 필요. 완전한 맞춤법 검사는 외부 교정기 없이 확정하지 않음 | `문서 목적과 표현`, `맞춤법과 띄어쓰기` 기준 |
| 간결하고 명확한 문장 | 매우 긴 문장의 첫 의심 지점을 확인 필요로 표시 | `_review_basic_principles` |
| 일반화되지 않은 약어/전문용어 지양 | 설명 없는 영문 약어/외국어 의심 지점을 확인 필요로 표시 | `_review_basic_principles` |
| 숫자는 특별한 사유가 없으면 아라비아 숫자 | `두 건`, `열 명`처럼 한글 수사와 단위가 분리된 표현을 확인 필요로 표시 | `_review_basic_principles` |
| 항목 기호 순서 `1. → 가. → 1) → 가) → (1) → (가) → ① → ㉮` | 자동 수정 제안 | `_review_item_marker_styles` |
| 항목 기호와 내용 사이 1타 | 자동 수정 제안 | `_review_item_spacing` |
| 하위 항목 2타 들여쓰기, 두 줄 이상 둘째 줄 정렬 | 텍스트 복붙 한계로 서식 확인 필요 | `_review_item_marker_hierarchy`, `format_notices` |
| 항목이 하나뿐이면 항목기호 미부여 | 확인 필요 | `_review_single_item_sections` |
| `재가하여 주시기 바랍니다.` 표현 | 문맥 키워드 기준 자동 수정 제안 | `_review_approval_phrase` |
| 날짜 `yyyy. m. d.` 및 0 미표기 | 자동 수정 제안 | `_review_hyphen_dates`, `_review_dot_dates` |
| 요일은 날짜 뒤에 붙여 `yyyy. m. d.(목)` | 자동 수정 제안 | `_review_weekday_dates` |
| 시간은 24시간제 `HH:MM` | 자동 수정 제안 | `_review_time` |
| 금액은 숫자 금액과 한글 금액 병기 | 확인 필요 | `_review_amounts` |
| 숫자 금액과 한글 금액 일치 | 확인 필요 | `_review_amounts` |
| 시간 외 쌍점은 앞말에 붙이고 뒤를 띄움 | 자동 수정 제안 | `_review_colon` |
| 물결표는 앞뒤에 붙임 | 자동 수정 제안 | `_review_tilde` |
| 관련문서는 문서번호/날짜/제목 포함 | 확인 필요 | `_review_related_documents` |
| 타기관 문서는 기관명 포함 | 문맥 확정 불가. 관련문서 형식 확인 항목에 포함 | `_review_related_documents` |
| 법령명은 홑낫표, 조문 번호/조문명 표시 | 확인 필요 | `_review_law_references` |
| 예산 사용 문서에는 소요예산 표 필요 | 확인 필요 | `_review_budget_tables` |
| 소요예산 표 필수 열: 회계연도, 회계구분/예산구분, 세목, 세목코드, 소요예산, 합계 | 확인 필요 및 표 검토 | `_review_budget_tables`, `_review_budget_amount_table` |
| 세목코드 2개 이상이면 산출근거/예산승인 기준 분리 | 사실관계 확인 필요. 표 검토 코멘트로 안내 | `_review_budget_tables` |
| 끝표시는 마지막 글자 뒤 2타 후 `끝.` | 자동 수정 제안 | `_review_end_marker_spacing` |
| 마침표 없이 끝나면 마침표 후 2타 `끝.` | 자동 수정 제안 | `_review_end_marker_spacing` |
| 표로 끝나는 문서의 끝표시 위치 | 텍스트/HTML만으로 위치 확정 불가. 확인 필요 | `format_notices`, `tableChecks` |
| 첨부파일은 `붙임`으로 표시 | 자동 수정 제안 | `_review_attachment_spacing` |
| `붙임` 뒤 2타, 파일명, 부수, 마침표 | 자동 수정 제안 및 확인 필요 | `_review_attachment_spacing`, `_review_attachment_list` |
| 두 번째 붙임부터 `붙임` 반복 금지 | 확인 필요 | `_review_attachment_list` |
| 붙임 파일명과 실제 첨부파일명 일치 | 확인 필요 | `_review_attachment_list` |
| 글꼴 굴림 11pt, 줄간격 180% | 텍스트 복붙으로 확정 불가. 서식 참고로 안내 | `format_notices` |
| 표 내용 검토 | 표는 자동 수정하지 않고 별도 검토 카드로 반환 | `review_table_checks` |

## 의도적으로 자동 수정하지 않는 항목

- 표 내부 텍스트: 웹한글기안기/HWP 표 레이아웃 손상을 막기 위해 자동 수정하지 않고, 표 검토 결과로만 안내한다.
- 사실관계: 날짜, 금액, 문서번호, 기관명, 부서명, 세목코드 등은 임의 보완하지 않고 확인 필요로 반환한다.
- 서식: 글꼴, 글자 크기, 줄간격, 표의 물리적 위치는 복붙 텍스트/HTML만으로 신뢰성 있게 판정하지 않는다.
