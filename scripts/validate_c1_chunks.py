from pathlib import Path
from collections import Counter, defaultdict
import json
import re


# ============================================================
# CONFIG
# ============================================================

PAGES_FILE = Path(
    "data/interim/pages_clean_c0.jsonl"
)

CHUNKS_FILE = Path(
    "data/processed/chunks_c1.jsonl"
)

REPORT_DIR = Path(
    "results/chunk_qc"
)

REPORT_FILE = (
    REPORT_DIR
    / "c1_chunk_validation_report.json"
)

REVIEW_FILE = (
    REPORT_DIR
    / "c1_all_chunks_review.txt"
)

MAX_CHUNK_SIZE = 800
FALLBACK_OVERLAP = 120

EXPECTED_PAGES = 74
EXPECTED_ARTICLES = 89
EXPECTED_CHAPTERS = 9


# ============================================================
# REGEX
# ============================================================

CHAPTER_RE = re.compile(
    r"(?m)^Chương\s+([IVXLCDM]+)\s*$"
)

ARTICLE_RE = re.compile(
    r"(?m)^Điều\s+(\d+[a-zA-Z]?)\."
)

FOOTNOTE_RE = re.compile(
    r"(?m)^"
    r"\d{1,3}\s+"
    r"(?:Khoản|Điểm|Điều|Đoạn|"
    r"Cụm từ|Nội dung|Tên)"
    r"\s+này\b"
)


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path: Path):

    items = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            if not line.strip():
                continue

            try:
                items.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}: {e}"
                )

    return items


# ============================================================
# GLOBAL SOURCE
# ============================================================

def build_global_document(pages):

    pages = sorted(
        pages,
        key=lambda x: x["page"]
    )

    parts = []
    page_ranges = []

    current_offset = 0

    for page in pages:

        text = page["text"]

        start = current_offset
        end = start + len(text)

        page_ranges.append(
            {
                "page": page["page"],
                "start": start,
                "end": end,
            }
        )

        parts.append(text)

        current_offset = end

        parts.append("\n")

        current_offset += 1

    return (
        "".join(parts),
        page_ranges
    )


# ============================================================
# POSITION -> PAGE
# ============================================================

def position_to_page(
    position,
    page_ranges
):

    previous_page = None

    for item in page_ranges:

        if (
            item["start"]
            <= position
            < item["end"]
        ):
            return item["page"]

        if item["start"] > position:
            break

        previous_page = item["page"]

    return previous_page


# ============================================================
# VALIDATE
# ============================================================

def validate(
    pages,
    chunks
):

    errors = []
    warnings = []

    (
        full_text,
        page_ranges
    ) = build_global_document(
        pages
    )

    # ========================================================
    # Dataset level
    # ========================================================

    if len(pages) != EXPECTED_PAGES:

        errors.append(
            {
                "type": "wrong_page_count",
                "expected": EXPECTED_PAGES,
                "actual": len(pages),
            }
        )

    # --------------------------------------------------------
    # Duplicate chunk IDs
    # --------------------------------------------------------

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    duplicate_ids = [
        chunk_id
        for chunk_id, count
        in Counter(ids).items()
        if count > 1
    ]

    if duplicate_ids:

        errors.append(
            {
                "type": "duplicate_chunk_ids",
                "ids": duplicate_ids,
            }
        )

    # --------------------------------------------------------
    # Articles represented
    # --------------------------------------------------------

    articles = {
        chunk["article_number"]
        for chunk in chunks
    }

    if len(articles) != EXPECTED_ARTICLES:

        errors.append(
            {
                "type": "wrong_article_count",
                "expected": EXPECTED_ARTICLES,
                "actual": len(articles),
            }
        )

    # --------------------------------------------------------
    # Chapters represented
    # --------------------------------------------------------

    chapters = {
        chunk["chapter"]
        for chunk in chunks
        if chunk["chapter"] is not None
    }

    if len(chapters) != EXPECTED_CHAPTERS:

        errors.append(
            {
                "type": "wrong_chapter_count",
                "expected": EXPECTED_CHAPTERS,
                "actual": len(chapters),
                "chapters": sorted(chapters),
            }
        )

    # ========================================================
    # Individual chunks
    # ========================================================

    for chunk in chunks:

        chunk_id = (
            chunk["chunk_id"]
        )

        text = (
            chunk["text"]
        )

        article_number = (
            chunk["article_number"]
        )

        article_title = (
            chunk["article_title"]
        )

        source_start = (
            chunk["source_start"]
        )

        source_end = (
            chunk["source_end"]
        )

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        required = {
            "chunk_id",
            "document_id",
            "chapter",
            "article_number",
            "article_title",
            "clause",
            "piece_index",
            "page_start",
            "page_end",
            "source_start",
            "source_end",
            "char_count",
            "text",
        }

        missing = (
            required
            - set(chunk.keys())
        )

        if missing:

            errors.append(
                {
                    "type": "missing_fields",
                    "chunk_id": chunk_id,
                    "fields": sorted(missing),
                }
            )

            continue

        # ----------------------------------------------------
        # Empty
        # ----------------------------------------------------

        if not text.strip():

            errors.append(
                {
                    "type": "empty_chunk",
                    "chunk_id": chunk_id,
                }
            )

        # ----------------------------------------------------
        # char_count
        # ----------------------------------------------------

        if (
            chunk["char_count"]
            != len(text)
        ):

            errors.append(
                {
                    "type": "char_count_mismatch",
                    "chunk_id": chunk_id,
                }
            )

        # ----------------------------------------------------
        # Maximum size
        # ----------------------------------------------------

        if len(text) > MAX_CHUNK_SIZE:

            errors.append(
                {
                    "type": "chunk_too_large",
                    "chunk_id": chunk_id,
                    "size": len(text),
                }
            )

        # ----------------------------------------------------
        # Source bounds
        # ----------------------------------------------------

        if (
            source_start < 0
            or source_end
            > len(full_text)
            or source_end
            <= source_start
        ):

            errors.append(
                {
                    "type": "invalid_source_range",
                    "chunk_id": chunk_id,
                    "start": source_start,
                    "end": source_end,
                }
            )

            continue

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        expected_heading = (
            f"Điều {article_number}. "
            f"{article_title}"
        )

        if not text.startswith(
            expected_heading
        ):

            errors.append(
                {
                    "type": "wrong_heading",
                    "chunk_id": chunk_id,
                    "expected": expected_heading,
                }
            )

        # ----------------------------------------------------
        # Body must match source span exactly
        # ----------------------------------------------------

        prefix = (
            expected_heading
            + "\n"
        )

        if text.startswith(prefix):

            chunk_body = text[
                len(prefix):
            ]

            source_body = full_text[
                source_start:
                source_end
            ]

            if chunk_body != source_body:

                errors.append(
                    {
                        "type": "source_text_mismatch",
                        "chunk_id": chunk_id,
                    }
                )

        # ----------------------------------------------------
        # Page start
        # ----------------------------------------------------

        expected_page_start = (
            position_to_page(
                source_start,
                page_ranges
            )
        )

        expected_page_end = (
            position_to_page(
                max(
                    source_start,
                    source_end - 1
                ),
                page_ranges
            )
        )

        if (
            chunk["page_start"]
            != expected_page_start
        ):

            errors.append(
                {
                    "type": "page_start_mismatch",
                    "chunk_id": chunk_id,
                    "expected": expected_page_start,
                    "actual": chunk["page_start"],
                }
            )

        if (
            chunk["page_end"]
            != expected_page_end
        ):

            errors.append(
                {
                    "type": "page_end_mismatch",
                    "chunk_id": chunk_id,
                    "expected": expected_page_end,
                    "actual": chunk["page_end"],
                }
            )

        if (
            chunk["page_start"]
            > chunk["page_end"]
        ):

            errors.append(
                {
                    "type": "invalid_page_range",
                    "chunk_id": chunk_id,
                }
            )

        # ----------------------------------------------------
        # Chapter contamination
        # ----------------------------------------------------

        body = text[
            len(expected_heading):
        ]

        if CHAPTER_RE.search(
            body
        ):

            errors.append(
                {
                    "type": "chapter_heading_contamination",
                    "chunk_id": chunk_id,
                }
            )

        # ----------------------------------------------------
        # Footnote contamination
        # ----------------------------------------------------

        if FOOTNOTE_RE.search(
            body
        ):

            warnings.append(
                {
                    "type": "footnote_in_chunk",
                    "chunk_id": chunk_id,
                }
            )

        # ----------------------------------------------------
        # Bad title
        # ----------------------------------------------------

        if (
            "\n"
            in article_title
        ):

            errors.append(
                {
                    "type": "newline_in_article_title",
                    "chunk_id": chunk_id,
                }
            )

        if (
            "Khoản này được sửa đổi"
            in article_title
        ):

            errors.append(
                {
                    "type": "footnote_in_title",
                    "chunk_id": chunk_id,
                }
            )

        # ----------------------------------------------------
        # Short chunks
        # ----------------------------------------------------

        if len(text) < 100:

            warnings.append(
                {
                    "type": "short_chunk",
                    "chunk_id": chunk_id,
                    "size": len(text),
                }
            )

    # ========================================================
    # Chunk numbering inside each Article
    # ========================================================

    chunks_by_article = defaultdict(
        list
    )

    for chunk in chunks:

        chunks_by_article[
            chunk["article_number"]
        ].append(chunk)

    for article_number, items in (
        chunks_by_article.items()
    ):

        def chunk_number(item):

            match = re.search(
                r"_c(\d+)$",
                item["chunk_id"]
            )

            if not match:
                return -1

            return int(
                match.group(1)
            )

        items.sort(
            key=chunk_number
        )

        actual_numbers = [
            chunk_number(item)
            for item in items
        ]

        expected_numbers = list(
            range(
                1,
                len(items) + 1
            )
        )

        if (
            actual_numbers
            != expected_numbers
        ):

            errors.append(
                {
                    "type": "non_sequential_chunk_ids",
                    "article": article_number,
                    "actual": actual_numbers,
                }
            )

    # ========================================================
    # Piece validation inside same clause
    # ========================================================

    clause_groups = defaultdict(
        list
    )

    for chunk in chunks:

        key = (
            chunk["article_number"],
            chunk["clause"],
        )

        clause_groups[
            key
        ].append(chunk)

    for (
        article_number,
        clause
    ), items in clause_groups.items():

        # Intro/None cũng có thể chỉ 1 piece
        if len(items) <= 1:
            continue

        items.sort(
            key=lambda x: (
                x["source_start"],
                x["piece_index"],
            )
        )

        # Piece indexes
        piece_indexes = [
            item["piece_index"]
            for item in items
        ]

        expected = list(
            range(
                1,
                len(items) + 1
            )
        )

        if piece_indexes != expected:

            errors.append(
                {
                    "type": "piece_index_sequence",
                    "article": article_number,
                    "clause": clause,
                    "actual": piece_indexes,
                }
            )

        # Overlap
        for i in range(
            len(items) - 1
        ):

            current = items[i]
            nxt = items[i + 1]

            overlap = (
                current["source_end"]
                - nxt["source_start"]
            )

            if overlap <= 0:

                errors.append(
                    {
                        "type": "missing_fallback_overlap",
                        "article": article_number,
                        "clause": clause,
                        "current": current["chunk_id"],
                        "next": nxt["chunk_id"],
                    }
                )

            elif overlap > FALLBACK_OVERLAP:

                errors.append(
                    {
                        "type": "overlap_too_large",
                        "article": article_number,
                        "clause": clause,
                        "overlap": overlap,
                    }
                )

            elif overlap != FALLBACK_OVERLAP:

                warnings.append(
                    {
                        "type": "nonstandard_overlap",
                        "article": article_number,
                        "clause": clause,
                        "overlap": overlap,
                    }
                )

    return (
        errors,
        warnings
    )


# ============================================================
# REVIEW FILE
# ============================================================

def create_review_file(
    chunks
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REVIEW_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            f.write(
                "=" * 100
                + "\n"
            )

            f.write(
                f"GLOBAL CHUNK : {index}\n"
            )

            f.write(
                f"CHUNK ID     : "
                f"{chunk['chunk_id']}\n"
            )

            f.write(
                f"CHAPTER      : "
                f"{chunk['chapter']}\n"
            )

            f.write(
                f"ARTICLE      : "
                f"{chunk['article_number']}\n"
            )

            f.write(
                f"CLAUSE       : "
                f"{chunk['clause']}\n"
            )

            f.write(
                f"PIECE        : "
                f"{chunk['piece_index']}\n"
            )

            f.write(
                f"PAGES        : "
                f"{chunk['page_start']} "
                f"-> "
                f"{chunk['page_end']}\n"
            )

            f.write(
                f"SOURCE RANGE : "
                f"{chunk['source_start']} "
                f"-> "
                f"{chunk['source_end']}\n"
            )

            f.write(
                f"SIZE         : "
                f"{chunk['char_count']}\n"
            )

            f.write(
                "-" * 100
                + "\n"
            )

            f.write(
                chunk["text"]
            )

            f.write(
                "\n\n"
            )


# ============================================================
# SAVE REPORT
# ============================================================

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

    sizes = [
        chunk["char_count"]
        for chunk in chunks
    ]

    warning_summary = Counter(
        warning["type"]
        for warning in warnings
    )

    report = {
        "config": {
            "name": "C1_structure_aware",
            "max_chunk_size": (
                MAX_CHUNK_SIZE
            ),
            "fallback_overlap": (
                FALLBACK_OVERLAP
            ),
        },
        "dataset": {
            "pages": len(pages),
            "chunks": len(chunks),
            "articles": len(
                {
                    chunk["article_number"]
                    for chunk in chunks
                }
            ),
            "chapters": len(
                {
                    chunk["chapter"]
                    for chunk in chunks
                    if chunk["chapter"]
                    is not None
                }
            ),
        },
        "statistics": {
            "min_chunk_size": (
                min(sizes)
            ),
            "max_chunk_size": (
                max(sizes)
            ),
            "average_chunk_size": (
                sum(sizes)
                / len(sizes)
            ),
        },
        "validation": {
            "status": (
                "PASS"
                if len(errors) == 0
                else "FAIL"
            ),
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "warning_summary": dict(
            warning_summary
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
            indent=2
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

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

    print("=" * 75)
    print(
        "C1 FULL CHUNK QUALITY CONTROL"
    )
    print("=" * 75)

    print(
        f"Pages checked       : "
        f"{len(pages)}"
    )

    print(
        f"Chunks checked      : "
        f"{len(chunks)}"
    )

    print(
        f"Articles represented: "
        f"{len(set(c['article_number'] for c in chunks))}"
    )

    print(
        f"Validation errors   : "
        f"{len(errors)}"
    )

    print(
        f"Warnings            : "
        f"{len(warnings)}"
    )

    if warning_types:

        print()
        print(
            "Warning summary:"
        )

        for (
            warning_type,
            count
        ) in warning_types.items():

            print(
                f"  "
                f"{warning_type:<28}"
                f": {count}"
            )

    print()

    if errors:

        print(
            "STATUS: FAIL"
        )

        print()

        print(
            "First errors:"
        )

        for error in errors[:15]:

            print(
                error
            )

    else:

        print(
            "STATUS: PASS"
        )

        print(
            "All C1 chunks passed "
            "integrity validation."
        )

    print()

    print(
        f"Report saved : "
        f"{REPORT_FILE}"
    )

    print(
        f"Review file  : "
        f"{REVIEW_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not PAGES_FILE.exists():

        raise FileNotFoundError(
            f"Missing: {PAGES_FILE}"
        )

    if not CHUNKS_FILE.exists():

        raise FileNotFoundError(
            f"Missing: {CHUNKS_FILE}"
        )

    pages = load_jsonl(
        PAGES_FILE
    )

    chunks = load_jsonl(
        CHUNKS_FILE
    )

    errors, warnings = validate(
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
        warnings
    )

    print_summary(
        pages,
        chunks,
        errors,
        warnings
    )

    if errors:

        raise SystemExit(1)


if __name__ == "__main__":
    main()