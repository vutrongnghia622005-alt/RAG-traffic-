from pathlib import Path
import json
import re
import unicodedata


INPUT_FILE = Path("data/interim/pages_raw.jsonl")
OUTPUT_FILE = Path("data/interim/pages_clean_c0.jsonl")


def is_structure_line(line: str) -> bool:
    """
    Nhận diện các dòng cấu trúc pháp luật cần giữ riêng.
    """
    line = line.strip()

    patterns = [
        r"^Chương\s+[IVXLCDM]+",
        r"^Điều\s+\d+[a-zA-Z]?\.",
        r"^\d+\.",
        r"^[a-zA-Zđ]\)",
    ]

    return any(
        re.match(pattern, line)
        for pattern in patterns
    )


def is_heading_line(line: str) -> bool:
    """
    Nhận diện heading viết hoa.
    """
    line = line.strip()

    return (
        line.isupper()
        and len(line) < 150
    )


def clean_page_text(text: str) -> str:
    """
    Cleaning cơ bản cho baseline C0.

    Không xử lý structure-aware nâng cao.
    Không xóa footnote.
    Không xóa số điều/khoản.
    """

    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    raw_lines = text.splitlines()

    lines = []

    # 2. Normalize whitespace
    for line in raw_lines:

        line = line.strip()

        line = re.sub(r"\s+", " ", line)

        if line:
            lines.append(line)

    cleaned = []

    # 3. Nối các dòng bị wrap bởi PDF
    for line in lines:

        if not cleaned:
            cleaned.append(line)
            continue

        previous = cleaned[-1]

        # Dòng hiện tại là Chương / Điều / Khoản / Điểm
        if is_structure_line(line):
            cleaned.append(line)
            continue

        # Dòng trước là Chương / Điều / Khoản / Điểm
        if is_structure_line(previous):
            cleaned.append(line)
            continue

        # Dòng hiện tại là heading
        if is_heading_line(line):
            cleaned.append(line)
            continue

        # Dòng trước là heading
        if is_heading_line(previous):
            cleaned.append(line)
            continue

        # Dòng trước đã kết thúc câu
        if re.search(r'[.!?:;”"]$', previous):
            cleaned.append(line)
            continue

        # Nếu không thuộc các trường hợp trên:
        # coi là line-wrap của cùng một câu
        cleaned[-1] = previous + " " + line

    return "\n".join(cleaned)


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    page_count = 0
    raw_total = 0
    clean_total = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as fin, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as fout:

        for line in fin:

            page = json.loads(line)

            raw_text = page["text"]
            clean_text = clean_page_text(raw_text)

            raw_count = page["char_count"]
            clean_count = len(clean_text)

            result = {
                "document_id": page["document_id"],
                "page": page["page"],
                "text": clean_text,
                "raw_char_count": raw_count,
                "clean_char_count": clean_count,
            }

            fout.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + "\n"
            )

            page_count += 1
            raw_total += raw_count
            clean_total += clean_count

    print("=" * 50)
    print("C0 CLEANING COMPLETED")
    print("=" * 50)

    print(f"Pages processed: {page_count}")
    print(f"Raw characters: {raw_total}")
    print(f"Clean characters: {clean_total}")
    print(f"Removed characters: {raw_total - clean_total}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()