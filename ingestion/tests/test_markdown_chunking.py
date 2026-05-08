from __future__ import annotations

import json

from ingestion.chunking.markdown import load_markdown_chunks, split_markdown_text


def test_load_markdown_chunks_cleans_metadata_and_image_noise(tmp_path):
    markdown_root = tmp_path / "markdown" / "notices"
    metadata_root = tmp_path / "json" / "notices"
    markdown_root.mkdir(parents=True)
    metadata_root.mkdir(parents=True)
    (markdown_root / "1.md").write_text(
        "\n".join(
            [
                "# 운영 시간 안내",
                "",
                "- 문서 ID: 1",
                "- 문서 유형: notices",
                "- 조회수: 10",
                "- 원문 URL: https://example.com/notice/1",
                "",
                "## 본문",
                "",
                "![](https://example.com/poster.png)",
                "",
                "- [개관시간](#hours)",
                "",
                "학기 중 평일은 09:00부터 21:00까지 운영합니다.",
            ]
        ),
        encoding="utf-8",
    )
    (metadata_root / "1.json").write_text(
        json.dumps(
            {
                "notice_id": 1,
                "title": "운영 시간 안내",
                "source_url": "https://example.com/notice/1",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (metadata_root / "1.md.metadata.json").write_text("{}", encoding="utf-8")

    chunks = load_markdown_chunks(
        markdown_root=tmp_path / "markdown",
        metadata_root=tmp_path / "json",
        clean_for_embedding=True,
        max_chars=300,
        overlap_chars=50,
    )

    assert len(chunks) == 1
    text = chunks[0].text
    assert "문서 제목: 운영 시간 안내" in text
    assert "문서 ID" not in text
    assert "조회수" not in text
    assert "원문 URL" not in text
    assert "![](" not in text
    assert "https://example.com" not in text
    assert "개관시간](#hours)" not in text


def test_split_markdown_text_does_not_repeat_same_title_when_packing_small_chunks():
    chunks = split_markdown_text(
        "별도 참여 신청없이 당일 행사 장소에 오시면 됩니다~!!\n\n"
        "문의: 학술정보팀 (02-760-5671)",
        title="KERIS 특성화 사업 성과 공유 워크숍 참여 안내",
        max_chars=300,
        overlap_chars=50,
    )

    assert len(chunks) == 1
    assert chunks[0].count("문서 제목: KERIS 특성화 사업 성과 공유 워크숍 참여 안내") == 1
    assert "별도 참여 신청없이 당일 행사 장소에 오시면 됩니다~!!" in chunks[0]
    assert "문의: 학술정보팀 (02-760-5671)" in chunks[0]


def test_split_markdown_text_repeats_table_header_without_cutting_rows():
    table = "\n".join(
        [
            "# 대출 안내",
            "",
            "## 대출 권수",
            "",
            "| 구분 | 권수 | 기간 |",
            "| --- | --- | --- |",
            "| 학부생 | 10권 | 14일 동안 대출 가능 |",
            "| 대학원생 | 20권 | 30일 동안 대출 가능 |",
            "| 교직원 | 30권 | 90일 동안 대출 가능 |",
        ]
    )

    chunks = split_markdown_text(
        table,
        title="대출 안내",
        max_chars=135,
        overlap_chars=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 135 for chunk in chunks)
    table_chunks = [chunk for chunk in chunks if "| 구분 | 권수 | 기간 |" in chunk]
    assert len(table_chunks) == len(chunks)
    assert all("| --- | --- | --- |" in chunk for chunk in table_chunks)
    assert any("| 학부생 | 10권 | 14일 동안 대출 가능 |" in chunk for chunk in chunks)
    assert any("| 대학원생 | 20권 | 30일 동안 대출 가능 |" in chunk for chunk in chunks)
    assert any("| 교직원 | 30권 | 90일 동안 대출 가능 |" in chunk for chunk in chunks)


def test_split_markdown_text_keeps_chunks_within_max_chars_for_long_paragraph():
    text = (
        "# 연체 안내\n\n"
        "## 반납 및 연체\n\n"
        + " ".join(["반납일을 지나면 연체일수만큼 대출이 제한됩니다."] * 40)
    )

    chunks = split_markdown_text(
        text,
        title="대출/반납/연장/예약/연체",
        max_chars=260,
        overlap_chars=40,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 260 for chunk in chunks)
    assert all(chunk.startswith("문서 제목: 대출/반납/연장/예약/연체") for chunk in chunks)


def test_load_markdown_chunks_keeps_title_chunk_for_image_only_document(tmp_path):
    markdown_root = tmp_path / "markdown" / "notices"
    metadata_root = tmp_path / "json" / "notices"
    markdown_root.mkdir(parents=True)
    metadata_root.mkdir(parents=True)
    (markdown_root / "2.md").write_text(
        "# 기말고사 기간 열람실 이용시간 연장 안내\n\n"
        "- 문서 ID: 2\n\n"
        "## 본문\n\n"
        "![](https://example.com/hours.png)\n",
        encoding="utf-8",
    )
    (metadata_root / "2.json").write_text(
        json.dumps(
            {
                "notice_id": 2,
                "title": "기말고사 기간 열람실 이용시간 연장 안내",
                "source_url": "https://example.com/notice/2",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (metadata_root / "2.md.metadata.json").write_text("{}", encoding="utf-8")

    chunks = load_markdown_chunks(
        markdown_root=tmp_path / "markdown",
        metadata_root=tmp_path / "json",
        clean_for_embedding=True,
        max_chars=300,
        overlap_chars=50,
    )

    assert [chunk.text for chunk in chunks] == ["문서 제목: 기말고사 기간 열람실 이용시간 연장 안내"]
