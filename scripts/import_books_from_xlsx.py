import json
import os

import openpyxl
import psycopg2
from psycopg2.extras import Json, execute_values


XLSX_PATH = os.path.expanduser("data/raw/books.xlsx")


HEADER_MAP = {
    "번호": "excel_no",
    "임시서지번호(중복가능)": "bib_no",
    "임시등록번호 중복불가": "reg_no",
    "서명": "title",
    "저자": "author",
    "출판사": "publisher",
    "출판년도": "publish_year",
    "소장처": "holding_location",
    "청구기호": "holding_call_no",
    "자료유형": "material_type",
    "별치기호": "location_symbol",
    "보존서가 소장처": "stack_location",
    "보존서가-칸": "stack_shelf",
    "ISBN": "isbn",
}

INSERT_COLUMNS = [
    "excel_no",
    "bib_no",
    "reg_no",
    "title",
    "author",
    "publisher",
    "publish_year",
    "holding_location",
    "holding_call_no",
    "material_type",
    "location_symbol",
    "stack_location",
    "stack_shelf",
    "isbn",
    "raw_data",
]


def clean_text(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def clean_int(value):
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    value = str(value).strip()
    if not value:
        return None

    return int(float(value))


def read_books_from_xlsx(path):
    workbook = openpyxl.load_workbook(path, data_only=True)
    sheet = workbook.active

    headers = [str(cell.value).strip() if cell.value else None for cell in sheet[1]]

    indexes = {}
    for index, header in enumerate(headers):
        if header in HEADER_MAP:
            indexes[HEADER_MAP[header]] = index

    if "reg_no" not in indexes:
        raise ValueError("엑셀에 '임시등록번호 중복불가' 컬럼이 없습니다.")

    if "title" not in indexes:
        raise ValueError("엑셀에 '서명' 컬럼이 없습니다.")

    books = []

    for row in sheet.iter_rows(min_row=2, values_only=True):
        book = {column: None for column in INSERT_COLUMNS}
        raw_data = {}

        for index, header in enumerate(headers):
            if header:
                raw_data[header] = row[index] if index < len(row) else None

        for column, index in indexes.items():
            value = row[index] if index < len(row) else None

            if column in ("excel_no", "publish_year"):
                book[column] = clean_int(value)
            else:
                book[column] = clean_text(value)

        if not book["reg_no"] or not book["title"]:
            continue

        book["raw_data"] = Json(
            raw_data,
            dumps=lambda value: json.dumps(value, ensure_ascii=False, default=str),
        )

        books.append(tuple(book[column] for column in INSERT_COLUMNS))

    return books


def import_books(books):
    sql = f"""
        INSERT INTO library.books ({", ".join(INSERT_COLUMNS)})
        VALUES %s
        ON CONFLICT (reg_no) DO UPDATE SET
            excel_no = EXCLUDED.excel_no,
            bib_no = EXCLUDED.bib_no,
            title = EXCLUDED.title,
            author = EXCLUDED.author,
            publisher = EXCLUDED.publisher,
            publish_year = EXCLUDED.publish_year,
            holding_location = EXCLUDED.holding_location,
            holding_call_no = EXCLUDED.holding_call_no,
            material_type = EXCLUDED.material_type,
            location_symbol = EXCLUDED.location_symbol,
            stack_location = EXCLUDED.stack_location,
            stack_shelf = EXCLUDED.stack_shelf,
            isbn = EXCLUDED.isbn,
            raw_data = EXCLUDED.raw_data,
            updated_at = now()
    """

    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="jokids",
        user="jokids",
        password="jokids1234",
    )

    try:
        with conn:
            with conn.cursor() as cursor:
                execute_values(cursor, sql, books, page_size=1000)
    finally:
        conn.close()


if __name__ == "__main__":
    books = read_books_from_xlsx(XLSX_PATH)
    import_books(books)
    print(f"완료: {len(books)}건 import")
