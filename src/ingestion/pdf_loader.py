from pathlib import Path
import json
import pymupdf as fitz


def load_pdf(pdf_path: str):
    """
    Đọc PDF theo từng trang và giữ nguyên text raw.
    Chưa cleaning, chưa chunking.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    pages = []

    for page_index, page in enumerate(doc):
        text = page.get_text("text")

        pages.append(
            {
                "document_id": pdf_path.stem,
                "page": page_index + 1,
                "text": text,
                "char_count": len(text),
            }
        )

    doc.close()

    return pages


def save_jsonl(pages, output_path: str):
    """
    Lưu mỗi trang thành một dòng JSON.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(
                json.dumps(
                    page,
                    ensure_ascii=False
                )
                + "\n"
            )


if __name__ == "__main__":

    input_pdf = "data/raw/traffic_law.pdf"

    output_file = "data/interim/pages_raw.jsonl"

    pages = load_pdf(input_pdf)

    save_jsonl(
        pages,
        output_file
    )

    print("=" * 50)
    print("PDF INGESTION COMPLETED")
    print("=" * 50)

    print(f"Total pages: {len(pages)}")

    empty_pages = [
        page["page"]
        for page in pages
        if page["char_count"] == 0
    ]

    print(f"Empty pages: {empty_pages}")

    total_chars = sum(
        page["char_count"]
        for page in pages
    )

    print(f"Total characters: {total_chars}")

    print(f"Saved to: {output_file}")