from pathlib import Path
from collections import defaultdict, Counter
import json
import math
import re


PAGES_FILE = Path("data/interim/pages_clean_c0.jsonl")
CHUNKS_FILE = Path("data/processed/chunks_c0.jsonl")

REPORT_DIR = Path("results/chunk_qc")
REPORT_FILE = REPORT_DIR / "chunk_validation_report.json"
REVIEW_FILE = REPORT_DIR / "all_chunks_review.txt"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

SHORT_CHUNK_WARNING = 100


def load_jsonl(path: Path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                data.append(json.loads(line))

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_number} "
                    f"in {path}: {e}"
                )

    return data


def expected_chunk_count(text_length: int):
    """
    Tính số chunk mong đợi theo:
    chunk_size = 512
    overlap = 50
    """

    if text_length <= 0:
        return 0

    if text_length <= CHUNK_SIZE:
        return 1

    step = CHUNK_SIZE - CHUNK_OVERLAP

    return 1 + math.ceil(
        (text_length - CHUNK_SIZE) / step
    )


def has_control_characters(text: str):
    """
    Phát hiện control characters bất thường.
    Cho phép newline và tab.
    """

    bad_chars = []

    for char in text:
        code = ord(char)

        if code < 32 and char not in ("\n", "\t", "\r"):
            bad_chars.append(code)

    return bad_chars


def validate_chunks(pages, chunks):

    errors = []
    warnings = []

    page_lookup = {
        page["page"]: page
        for page in pages
    }

    chunks_by_page = defaultdict(list)

    for chunk in chunks:
        chunks_by_page[chunk["page"]].append(chunk)

    # =========================================================
    # 1. Kiểm tra tổng số page
    # =========================================================

    page_numbers = sorted(page_lookup.keys())

    if len(page_numbers) != len(pages):
        errors.append(
            "Duplicate page numbers found in clean page dataset."
        )

    # =========================================================
    # 2. Kiểm tra chunk_id duplicate
    # =========================================================

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    duplicate_ids = [
        chunk_id
        for chunk_id, count
        in Counter(chunk_ids).items()
        if count > 1
    ]

    if duplicate_ids:
        errors.append(
            {
                "type": "duplicate_chunk_id",
                "ids": duplicate_ids,
            }
        )

    # =========================================================
    # 3. Kiểm tra từng page
    # =========================================================

    for page_number in page_numbers:

        page = page_lookup[page_number]
        source_text = page["text"]

        page_chunks = chunks_by_page.get(
            page_number,
            []
        )

        page_chunks.sort(
            key=lambda x: x["chunk_index"]
        )

        if not page_chunks:
            errors.append(
                {
                    "type": "page_without_chunks",
                    "page": page_number,
                }
            )
            continue

        # -----------------------------------------------------
        # Expected number of chunks
        # -----------------------------------------------------

        expected_count = expected_chunk_count(
            len(source_text)
        )

        if len(page_chunks) != expected_count:
            errors.append(
                {
                    "type": "wrong_chunk_count",
                    "page": page_number,
                    "expected": expected_count,
                    "actual": len(page_chunks),
                }
            )

        # -----------------------------------------------------
        # First chunk must start at 0
        # -----------------------------------------------------

        if page_chunks[0]["start_char"] != 0:
            errors.append(
                {
                    "type": "first_chunk_not_zero",
                    "page": page_number,
                    "actual": page_chunks[0]["start_char"],
                }
            )

        # -----------------------------------------------------
        # Last chunk must end at end of page
        # -----------------------------------------------------

        if page_chunks[-1]["end_char"] != len(source_text):
            errors.append(
                {
                    "type": "last_chunk_wrong_end",
                    "page": page_number,
                    "expected": len(source_text),
                    "actual": page_chunks[-1]["end_char"],
                }
            )

        # =====================================================
        # 4. Kiểm tra từng chunk
        # =====================================================

        for expected_index, chunk in enumerate(
            page_chunks,
            start=1
        ):

            chunk_id = chunk["chunk_id"]

            start = chunk["start_char"]
            end = chunk["end_char"]
            text = chunk["text"]

            # ---------------------------------------------
            # chunk_index
            # ---------------------------------------------

            if chunk["chunk_index"] != expected_index:
                errors.append(
                    {
                        "type": "wrong_chunk_index",
                        "chunk_id": chunk_id,
                        "expected": expected_index,
                        "actual": chunk["chunk_index"],
                    }
                )

            # ---------------------------------------------
            # Expected chunk ID
            # ---------------------------------------------

            expected_id = (
                f"{chunk['document_id']}"
                f"_p{page_number:03d}"
                f"_c{expected_index:03d}"
            )

            if chunk_id != expected_id:
                errors.append(
                    {
                        "type": "wrong_chunk_id",
                        "expected": expected_id,
                        "actual": chunk_id,
                    }
                )

            # ---------------------------------------------
            # Offset bounds
            # ---------------------------------------------

            if start < 0:
                errors.append(
                    {
                        "type": "negative_start",
                        "chunk_id": chunk_id,
                    }
                )

            if end > len(source_text):
                errors.append(
                    {
                        "type": "end_out_of_bounds",
                        "chunk_id": chunk_id,
                        "end": end,
                        "page_length": len(source_text),
                    }
                )

            if end <= start:
                errors.append(
                    {
                        "type": "invalid_range",
                        "chunk_id": chunk_id,
                        "start": start,
                        "end": end,
                    }
                )

            # ---------------------------------------------
            # Text phải chính xác bằng source[start:end]
            # ---------------------------------------------

            expected_text = source_text[start:end]

            if text != expected_text:
                errors.append(
                    {
                        "type": "text_source_mismatch",
                        "chunk_id": chunk_id,
                    }
                )

            # ---------------------------------------------
            # char_count
            # ---------------------------------------------

            if chunk["char_count"] != len(text):
                errors.append(
                    {
                        "type": "wrong_char_count",
                        "chunk_id": chunk_id,
                        "metadata": chunk["char_count"],
                        "actual": len(text),
                    }
                )

            # ---------------------------------------------
            # Max chunk size
            # ---------------------------------------------

            if len(text) > CHUNK_SIZE:
                errors.append(
                    {
                        "type": "chunk_too_large",
                        "chunk_id": chunk_id,
                        "size": len(text),
                    }
                )

            # ---------------------------------------------
            # Empty chunk
            # ---------------------------------------------

            if not text.strip():
                errors.append(
                    {
                        "type": "empty_chunk",
                        "chunk_id": chunk_id,
                    }
                )

            # ---------------------------------------------
            # Short chunk warning
            # ---------------------------------------------

            if len(text) < SHORT_CHUNK_WARNING:
                warnings.append(
                    {
                        "type": "short_chunk",
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "size": len(text),
                    }
                )

            # ---------------------------------------------
            # Control character warning
            # ---------------------------------------------

            bad_chars = has_control_characters(text)

            if bad_chars:
                warnings.append(
                    {
                        "type": "control_characters",
                        "chunk_id": chunk_id,
                        "codes": sorted(set(bad_chars)),
                    }
                )

            # ---------------------------------------------
            # Detect word split at chunk boundary
            # Đây KHÔNG phải error đối với C0.
            # ---------------------------------------------

            if start > 0:

                before = source_text[start - 1]
                current = source_text[start]

                if (
                    before.isalnum()
                    and current.isalnum()
                ):
                    warnings.append(
                        {
                            "type": "starts_inside_word",
                            "chunk_id": chunk_id,
                            "page": page_number,
                            "position": start,
                        }
                    )

            if end < len(source_text):

                before_end = source_text[end - 1]
                after_end = source_text[end]

                if (
                    before_end.isalnum()
                    and after_end.isalnum()
                ):
                    warnings.append(
                        {
                            "type": "ends_inside_word",
                            "chunk_id": chunk_id,
                            "page": page_number,
                            "position": end,
                        }
                    )

        # =====================================================
        # 5. Kiểm tra overlap / continuity
        # =====================================================

        for i in range(len(page_chunks) - 1):

            current = page_chunks[i]
            nxt = page_chunks[i + 1]

            expected_next_start = (
                current["end_char"]
                - CHUNK_OVERLAP
            )

            if nxt["start_char"] != expected_next_start:
                errors.append(
                    {
                        "type": "wrong_overlap_offset",
                        "page": page_number,
                        "current": current["chunk_id"],
                        "next": nxt["chunk_id"],
                        "expected_next_start": expected_next_start,
                        "actual_next_start": nxt["start_char"],
                    }
                )

            current_overlap = (
                current["text"][-CHUNK_OVERLAP:]
            )

            next_overlap = (
                nxt["text"][:CHUNK_OVERLAP]
            )

            if current_overlap != next_overlap:
                errors.append(
                    {
                        "type": "overlap_content_mismatch",
                        "page": page_number,
                        "current": current["chunk_id"],
                        "next": nxt["chunk_id"],
                    }
                )

    # =========================================================
    # 6. Check chunks pointing to unknown page
    # =========================================================

    for chunk in chunks:

        if chunk["page"] not in page_lookup:
            errors.append(
                {
                    "type": "unknown_page",
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                }
            )

    # =========================================================
    # 7. Exact duplicate text warnings
    # =========================================================

    text_counter = Counter(
        chunk["text"]
        for chunk in chunks
    )

    duplicate_texts = [
        text
        for text, count
        in text_counter.items()
        if count > 1
    ]

    if duplicate_texts:

        for duplicated_text in duplicate_texts:

            ids = [
                chunk["chunk_id"]
                for chunk in chunks
                if chunk["text"] == duplicated_text
            ]

            warnings.append(
                {
                    "type": "duplicate_chunk_text",
                    "chunk_ids": ids,
                    "size": len(duplicated_text),
                }
            )

    return errors, warnings


def create_review_file(chunks):
    """
    Xuất toàn bộ 412 chunk ra file text
    để đọc thủ công.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REVIEW_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for i, chunk in enumerate(
            chunks,
            start=1
        ):

            f.write(
                "=" * 80
                + "\n"
            )

            f.write(
                f"GLOBAL CHUNK: {i}\n"
            )

            f.write(
                f"CHUNK ID    : {chunk['chunk_id']}\n"
            )

            f.write(
                f"PAGE        : {chunk['page']}\n"
            )

            f.write(
                f"CHUNK INDEX : {chunk['chunk_index']}\n"
            )

            f.write(
                f"RANGE       : "
                f"{chunk['start_char']} -> "
                f"{chunk['end_char']}\n"
            )

            f.write(
                f"CHAR COUNT  : "
                f"{chunk['char_count']}\n"
            )

            f.write(
                "-" * 80
                + "\n"
            )

            f.write(
                chunk["text"]
            )

            f.write(
                "\n\n"
            )


def save_report(
    pages,
    chunks,
    errors,
    warnings
):
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    warning_types = Counter(
        warning["type"]
        for warning in warnings
    )

    chunk_sizes = [
        chunk["char_count"]
        for chunk in chunks
    ]

    report = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "dataset": {
            "total_pages": len(pages),
            "total_chunks": len(chunks),
        },
        "chunk_statistics": {
            "min_size": min(chunk_sizes),
            "max_size": max(chunk_sizes),
            "average_size": (
                sum(chunk_sizes)
                / len(chunk_sizes)
            ),
        },
        "validation": {
            "total_errors": len(errors),
            "total_warnings": len(warnings),
            "status": (
                "PASS"
                if len(errors) == 0
                else "FAIL"
            ),
        },
        "warning_summary": dict(
            warning_types
        ),
        "errors": errors,
        "warnings": warnings,
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )


def print_summary(
    pages,
    chunks,
    errors,
    warnings
):

    warning_types = Counter(
        warning["type"]
        for warning in warnings
    )

    print("=" * 70)
    print("C0 CHUNK QUALITY CONTROL")
    print("=" * 70)

    print(
        f"Pages checked       : {len(pages)}"
    )

    print(
        f"Chunks checked      : {len(chunks)}"
    )

    print(
        f"Validation errors   : {len(errors)}"
    )

    print(
        f"Warnings            : {len(warnings)}"
    )

    print()

    if warning_types:

        print("Warning summary:")

        for warning_type, count in (
            warning_types.items()
        ):
            print(
                f"  {warning_type:<25}: {count}"
            )

    print()

    if errors:

        print("STATUS: FAIL")
        print()

        print("First errors:")

        for error in errors[:10]:
            print(error)

    else:

        print("STATUS: PASS")
        print(
            "All chunks passed integrity validation."
        )

    print()
    print(
        f"Report saved : {REPORT_FILE}"
    )

    print(
        f"Review file  : {REVIEW_FILE}"
    )


def main():

    if not PAGES_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {PAGES_FILE}"
        )

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {CHUNKS_FILE}"
        )

    pages = load_jsonl(
        PAGES_FILE
    )

    chunks = load_jsonl(
        CHUNKS_FILE
    )

    errors, warnings = validate_chunks(
        pages,
        chunks
    )

    create_review_file(
        chunks
    )

    save_report(
        pages,
        chunks,
        errors,
        warnings,
    )

    print_summary(
        pages,
        chunks,
        errors,
        warnings,
    )

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()