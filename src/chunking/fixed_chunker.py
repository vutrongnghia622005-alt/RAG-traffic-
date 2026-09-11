from pathlib import Path
import json


INPUT_FILE = Path("data/interim/pages_clean_c0.jsonl")
OUTPUT_FILE = Path("data/processed/chunks_c0.jsonl")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def load_pages(input_path: Path):
    """
    Đọc dữ liệu page-level sau bước cleaning C0.
    Mỗi dòng trong file JSONL tương ứng với một trang PDF.
    """
    pages = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))

    return pages


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):
    """
    Fixed character chunking cho baseline C0.

    Ví dụ:
    Chunk 1: 0 -> 512
    Chunk 2: 462 -> 974
    Chunk 3: 924 -> 1436

    Với:
    chunk_size = 512
    overlap = 50

    Không dùng semantic splitting.
    Không dùng structure-aware splitting.
    Không nhận diện Chương / Điều / Khoản.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        # Không dùng .strip() ở đây
        # để đảm bảo offset và overlap chính xác tuyệt đối
        chunk = text[start:end]

        # Chỉ bỏ qua nếu chunk hoàn toàn không có nội dung
        if chunk.strip():
            chunks.append(
                {
                    "text": chunk,
                    "start_char": start,
                    "end_char": end,
                    "char_count": len(chunk),
                }
            )

        # Nếu đã tới cuối text thì dừng
        if end >= text_length:
            break

        # Tạo overlap
        start = end - overlap

    return chunks


def create_chunks(pages):
    """
    Chunk từng page riêng biệt.

    Việc chunk từng page riêng giúp:
    - giữ metadata page chính xác
    - dễ citation
    - tránh chunk chạy qua hai trang khác nhau
    """

    all_chunks = []

    for page in pages:

        document_id = page["document_id"]
        page_number = page["page"]
        text = page["text"]

        page_chunks = chunk_text(
            text=text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for chunk_index, chunk in enumerate(
            page_chunks,
            start=1
        ):

            chunk_id = (
                f"{document_id}"
                f"_p{page_number:03d}"
                f"_c{chunk_index:03d}"
            )

            result = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "page": page_number,
                "chunk_index": chunk_index,
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "char_count": chunk["char_count"],
                "text": chunk["text"],
            }

            all_chunks.append(result)

    return all_chunks


def save_chunks(chunks, output_path: Path):
    """
    Lưu chunks thành JSONL.
    Một dòng = một chunk.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                )
                + "\n"
            )


def validate_chunks(chunks):
    """
    Kiểm tra toàn bộ chunk:
    - offset có đúng không
    - overlap có đúng 50 ký tự không
    """

    pages = {}

    for chunk in chunks:
        page_number = chunk["page"]

        if page_number not in pages:
            pages[page_number] = []

        pages[page_number].append(chunk)

    errors = []

    for page_number, page_chunks in pages.items():

        page_chunks.sort(
            key=lambda x: x["chunk_index"]
        )

        for i in range(len(page_chunks) - 1):

            current = page_chunks[i]
            next_chunk = page_chunks[i + 1]

            # Kiểm tra start_char của chunk tiếp theo
            expected_next_start = (
                current["end_char"]
                - CHUNK_OVERLAP
            )

            if (
                next_chunk["start_char"]
                != expected_next_start
            ):
                errors.append(
                    {
                        "page": page_number,
                        "current": current["chunk_id"],
                        "next": next_chunk["chunk_id"],
                        "error": "offset_mismatch",
                    }
                )

            # Kiểm tra nội dung overlap
            current_overlap = (
                current["text"][
                    -CHUNK_OVERLAP:
                ]
            )

            next_overlap = (
                next_chunk["text"][
                    :CHUNK_OVERLAP
                ]
            )

            if current_overlap != next_overlap:
                errors.append(
                    {
                        "page": page_number,
                        "current": current["chunk_id"],
                        "next": next_chunk["chunk_id"],
                        "error": "overlap_mismatch",
                    }
                )

    return errors


def print_statistics(
    chunks,
    total_pages,
    validation_errors,
):
    """
    In thống kê chunking.
    """

    char_counts = [
        chunk["char_count"]
        for chunk in chunks
    ]

    print("=" * 60)
    print("C0 FIXED CHUNKING COMPLETED")
    print("=" * 60)

    print(
        f"Pages processed : {total_pages}"
    )

    print(
        f"Total chunks    : {len(chunks)}"
    )

    if char_counts:

        average_size = (
            sum(char_counts)
            / len(char_counts)
        )

        print(
            f"Average size    : "
            f"{average_size:.2f}"
        )

        print(
            f"Min chunk size  : "
            f"{min(char_counts)}"
        )

        print(
            f"Max chunk size  : "
            f"{max(char_counts)}"
        )

    print(
        f"Chunk size      : {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap   : {CHUNK_OVERLAP}"
    )

    print(
        f"Validation errs : "
        f"{len(validation_errors)}"
    )

    if validation_errors:

        print("\nFirst validation errors:")

        for error in validation_errors[:10]:
            print(error)

    else:

        print(
            "\nALL CHUNKS VALID"
        )

    print(
        f"\nSaved to        : {OUTPUT_FILE}"
    )


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    pages = load_pages(INPUT_FILE)

    if not pages:
        raise ValueError(
            "No pages found in input file"
        )

    chunks = create_chunks(pages)

    if not chunks:
        raise ValueError(
            "No chunks were created"
        )

    save_chunks(
        chunks,
        OUTPUT_FILE
    )

    validation_errors = (
        validate_chunks(chunks)
    )

    print_statistics(
        chunks=chunks,
        total_pages=len(pages),
        validation_errors=validation_errors,
    )


if __name__ == "__main__":
    main()