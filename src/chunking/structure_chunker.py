from pathlib import Path
import json
import re


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/interim/pages_clean_c0.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/chunks_c1.jsonl"
)

# C1 structure-aware chunking
MAX_CHUNK_SIZE = 800

# Chỉ dùng khi một legal unit quá dài
FALLBACK_OVERLAP = 120


# ============================================================
# REGEX
# ============================================================

# ------------------------------------------------------------
# Chapter:
#
# Chương I
# Chương II
# ...
# ------------------------------------------------------------

CHAPTER_RE = re.compile(
    r"(?m)^Chương\s+([IVXLCDM]+)\s*$"
)


# ------------------------------------------------------------
# Article:
#
# Điều 1. ...
# Điều 72. ...
# ------------------------------------------------------------

ARTICLE_RE = re.compile(
    r"(?m)^Điều\s+(\d+[a-zA-Z]?)\.\s*(.*)$"
)


# ------------------------------------------------------------
# Clause:
#
# 1. ...
# 2. ...
# 2a. ...
#
# Amendment marker bị dính:
#
# 2.43 Bộ Công an...
# 3.44 Bộ Xây dựng...
#
# group(1):
# 1
# 2
# 2a
# 3
# ------------------------------------------------------------

CLAUSE_RE = re.compile(
    r"(?m)^(\d+[a-zA-Z]?)\.(?:\d+)?\s+"
)


# ------------------------------------------------------------
# Footnote line.
#
# Ví dụ:
#
# 39 Khoản này được sửa đổi...
# 43 Khoản này được sửa đổi...
#
# hoặc:
#
# 45 Khoản 1 và khoản 2 Điều 11...
# ------------------------------------------------------------

FOOTNOTE_START_RE = re.compile(
    r"(?mi)^"
    r"\d{1,3}\s+"
    r"(?="
    r"(?:"
    r"Khoản(?:\s+này|\s+\d+[a-zA-Z]?)"
    r"|Điểm(?:\s+này|\s+[a-zA-Z])"
    r"|Điều(?:\s+này|\s+\d+)"
    r"|Đoạn\s+này"
    r"|Cụm\s+từ"
    r"|Nội\s+dung"
    r"|Tên\s+"
    r")"
    r")"
)


# ------------------------------------------------------------
# Page number đứng riêng.
#
# 58
# 71
# 72
# ------------------------------------------------------------

PAGE_NUMBER_RE = re.compile(
    r"(?m)^\s*\d{1,3}\s*$"
)


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path: Path):
    """
    Đọc file JSONL.
    """

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

                item = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}: "
                    f"{exc}"
                )

            items.append(
                item
            )

    return items


# ============================================================
# BUILD GLOBAL DOCUMENT
# ============================================================

def build_global_document(pages):
    """
    Ghép toàn bộ page clean thành một document logic.

    Giữa hai page thêm đúng 1 newline.

    page_ranges giữ global character offset
    để map chunk về trang PDF.
    """

    pages = sorted(
        pages,
        key=lambda item: item["page"]
    )

    parts = []
    page_ranges = []

    current_offset = 0

    for page in pages:

        page_number = (
            page["page"]
        )

        text = (
            page["text"]
        )

        page_start = (
            current_offset
        )

        page_end = (
            page_start
            + len(text)
        )

        page_ranges.append(
            {
                "page": page_number,
                "start": page_start,
                "end": page_end,
            }
        )

        parts.append(
            text
        )

        current_offset = (
            page_end
        )

        # separator giữa hai page
        parts.append(
            "\n"
        )

        current_offset += 1

    full_text = "".join(
        parts
    )

    return (
        full_text,
        page_ranges,
    )


# ============================================================
# POSITION -> PAGE
# ============================================================

def position_to_page(
    position,
    page_ranges,
):
    """
    Map global character position
    về page number.
    """

    previous_page = None

    for item in page_ranges:

        if (
            item["start"]
            <= position
            < item["end"]
        ):

            return item["page"]

        if (
            item["start"]
            > position
        ):

            break

        previous_page = (
            item["page"]
        )

    # Nếu position nằm đúng separator giữa 2 trang
    # thì sử dụng trang trước đó.
    return previous_page


# ============================================================
# QUOTED SPANS
# ============================================================

def find_quoted_spans(text):
    """
    Tìm các vùng nằm trong dấu ngoặc kép.

    Hỗ trợ:

        “ ... ”
        " ... "

    Cần thiết cho Điều 89.

    Footnote 45 trích:

        “Điều 11...
        1. ...
        2. ...”

    Hai số 1. và 2. trong quote không phải
    Khoản của Điều 89.
    """

    spans = []

    # --------------------------------------------------------
    # Vietnamese smart quote: “ ... ”
    # --------------------------------------------------------

    start = None

    for index, char in enumerate(
        text
    ):

        if char == "“":

            if start is None:

                start = index

        elif char == "”":

            if start is not None:

                spans.append(
                    (
                        start,
                        index + 1,
                    )
                )

                start = None

    # --------------------------------------------------------
    # ASCII quote: " ... "
    # --------------------------------------------------------

    start = None

    for index, char in enumerate(
        text
    ):

        if char != '"':
            continue

        if start is None:

            start = index

        else:

            spans.append(
                (
                    start,
                    index + 1,
                )
            )

            start = None

    # --------------------------------------------------------
    # Không có quote
    # --------------------------------------------------------

    if not spans:

        return []

    # --------------------------------------------------------
    # Merge nếu spans overlap
    # --------------------------------------------------------

    spans.sort(
        key=lambda item: item[0]
    )

    merged = [
        spans[0]
    ]

    for start, end in spans[1:]:

        (
            previous_start,
            previous_end,
        ) = merged[-1]

        if start <= previous_end:

            merged[-1] = (
                previous_start,
                max(
                    previous_end,
                    end
                ),
            )

        else:

            merged.append(
                (
                    start,
                    end
                )
            )

    return merged


def position_inside_spans(
    position,
    spans,
):
    """
    True nếu position nằm bên trong
    một quoted span.
    """

    for start, end in spans:

        if (
            start
            <= position
            < end
        ):

            return True

    return False


# ============================================================
# CLAUSE SORT KEY
# ============================================================

def clause_sort_key(label):
    """
    Chuyển label thành key để so sánh.

    1   -> (1, 0)
    2   -> (2, 0)
    2a  -> (2, 1)
    2b  -> (2, 2)
    3   -> (3, 0)
    """

    match = re.fullmatch(
        r"(\d+)([a-zA-Z]?)",
        label
    )

    if not match:

        return None

    number = int(
        match.group(1)
    )

    suffix = (
        match.group(2)
        .lower()
    )

    if suffix:

        suffix_order = (
            ord(suffix)
            - ord("a")
            + 1
        )

    else:

        suffix_order = 0

    return (
        number,
        suffix_order,
    )


# ============================================================
# FILTER STRUCTURAL CLAUSES
# ============================================================

def filter_structural_clauses(
    matches,
    quoted_spans,
):
    """
    Loại các false-positive clause.

    Hai lớp bảo vệ:

    1. Không nhận số nằm trong quote.
    2. Top-level clause phải tăng dần.

    Ví dụ Điều 89 source có thể trông như:

        1. ...
        2. ...

        footnote:
        “Điều 11...
        1. ...
        2. ...”

        3. ...
        4. ...

    Candidate regex:
        1, 2, 1, 2, 3, 4

    Sau filter:
        1, 2, 3, 4
    """

    valid_matches = []

    last_key = None

    for match in matches:

        # ----------------------------------------------------
        # Clause nằm trong quotation
        # ----------------------------------------------------

        if position_inside_spans(
            match.start(),
            quoted_spans,
        ):

            continue

        label = (
            match.group(1)
        )

        key = (
            clause_sort_key(
                label
            )
        )

        if key is None:

            continue

        # ----------------------------------------------------
        # Clause đầu tiên
        # ----------------------------------------------------

        if last_key is None:

            valid_matches.append(
                match
            )

            last_key = key

            continue

        # ----------------------------------------------------
        # Top-level clause phải tăng dần.
        #
        # 1 -> 2       OK
        # 2 -> 2a      OK
        # 2a -> 3      OK
        #
        # 2 -> 1       reject
        # 2 -> 2       reject
        # ----------------------------------------------------

        if key <= last_key:

            continue

        valid_matches.append(
            match
        )

        last_key = key

    return valid_matches


# ============================================================
# EXTRACT CHAPTERS
# ============================================================

def extract_chapters(
    full_text
):
    """
    Detect Chapter headings.
    """

    matches = list(
        CHAPTER_RE.finditer(
            full_text
        )
    )

    chapters = []

    for index, match in enumerate(
        matches
    ):

        start = (
            match.start()
        )

        if (
            index + 1
            < len(matches)
        ):

            end = (
                matches[
                    index + 1
                ].start()
            )

        else:

            end = (
                len(full_text)
            )

        chapters.append(
            {
                "chapter": (
                    match.group(1)
                ),
                "start": start,
                "end": end,
            }
        )

    return chapters


# ============================================================
# FIND CHAPTER
# ============================================================

def find_chapter(
    article_start,
    chapters,
):
    """
    Tìm Chapter gần nhất phía trước Article.
    """

    result = None

    for chapter in chapters:

        if (
            chapter["start"]
            <= article_start
        ):

            result = (
                chapter["chapter"]
            )

        else:

            break

    return result


# ============================================================
# TITLE CONTINUATION
# ============================================================

def is_title_continuation(
    line
):
    """
    Nhận biết title bị PDF wrap.

    Ví dụ:

    Điều 72. Quyền và trách nhiệm của người
    điều khiển phương tiện tham gia giao thông đường bộ

    Dòng thứ 2 bắt đầu bằng chữ thường,
    nên thuộc title.
    """

    line = (
        line.strip()
    )

    if not line:

        return False

    # Clause bắt đầu
    if CLAUSE_RE.match(
        line
    ):

        return False

    # Footnote
    if FOOTNOTE_START_RE.match(
        line
    ):

        return False

    # Page number
    if re.fullmatch(
        r"\d{1,3}",
        line
    ):

        return False

    # Chapter heading
    if CHAPTER_RE.match(
        line
    ):

        return False

    first_alpha = None

    for char in line:

        if char.isalpha():

            first_alpha = char

            break

    if first_alpha is None:

        return False

    return (
        first_alpha.islower()
    )


# ============================================================
# CLEAN ARTICLE TITLE
# ============================================================

def clean_article_title(
    title
):
    """
    Làm sạch artefact ở title.

    Ví dụ lỗi trước đây:

    Điều 69. Di chuyển ... giao thông đường bộ
    39 Khoản này được sửa đổi...
    58

    Không được để footnote/page number
    trở thành article_title.
    """

    title = re.sub(
        r"\s+",
        " ",
        title
    ).strip()

    # --------------------------------------------------------
    # Footnote dính vào title
    # --------------------------------------------------------

    footnote_pattern = re.compile(
        r"\s+\d{1,3}\s+"
        r"(?="
        r"(?:"
        r"Khoản(?:\s+này|\s+\d+[a-zA-Z]?)"
        r"|Điểm(?:\s+này|\s+[a-zA-Z])"
        r"|Điều(?:\s+này|\s+\d+)"
        r"|Đoạn\s+này"
        r"|Cụm\s+từ"
        r"|Nội\s+dung"
        r"|Tên\s+"
        r")"
        r")",
        re.IGNORECASE,
    )

    match = (
        footnote_pattern.search(
            title
        )
    )

    if match:

        title = (
            title[
                :match.start()
            ]
            .strip()
        )

    # --------------------------------------------------------
    # Page number dính cuối title
    # --------------------------------------------------------

    title = re.sub(
        r"\s+\d{1,3}\s*$",
        "",
        title
    ).strip()

    return title


# ============================================================
# PARSE ARTICLE TITLE
# ============================================================

def parse_article_title(
    raw_article_text,
    first_title_part,
):
    """
    Ghép article title bị wrap.

    Returns:
        article_title
        body_start_local
    """

    title_parts = []

    first_title_part = (
        first_title_part.strip()
    )

    if first_title_part:

        title_parts.append(
            first_title_part
        )

    first_line_end = (
        raw_article_text.find(
            "\n"
        )
    )

    # --------------------------------------------------------
    # Article chỉ có 1 dòng
    # --------------------------------------------------------

    if first_line_end == -1:

        title = (
            clean_article_title(
                " ".join(
                    title_parts
                )
            )
        )

        return (
            title,
            len(
                raw_article_text
            ),
        )

    cursor = (
        first_line_end + 1
    )

    body_start = (
        cursor
    )

    while (
        cursor
        < len(raw_article_text)
    ):

        line_end = (
            raw_article_text.find(
                "\n",
                cursor
            )
        )

        if line_end == -1:

            line_end = (
                len(
                    raw_article_text
                )
            )

        raw_line = (
            raw_article_text[
                cursor:
                line_end
            ]
        )

        line = (
            raw_line.strip()
        )

        # ----------------------------------------------------
        # Empty line
        # ----------------------------------------------------

        if not line:

            cursor = min(
                line_end + 1,
                len(
                    raw_article_text
                ),
            )

            body_start = (
                cursor
            )

            continue

        # ----------------------------------------------------
        # Wrapped title continuation
        # ----------------------------------------------------

        if is_title_continuation(
            line
        ):

            title_parts.append(
                line
            )

            cursor = min(
                line_end + 1,
                len(
                    raw_article_text
                ),
            )

            body_start = (
                cursor
            )

            continue

        # ----------------------------------------------------
        # Body bắt đầu
        # ----------------------------------------------------

        body_start = (
            cursor
        )

        break

    article_title = (
        " ".join(
            title_parts
        )
    )

    article_title = (
        clean_article_title(
            article_title
        )
    )

    return (
        article_title,
        body_start,
    )


# ============================================================
# EXTRACT ARTICLES
# ============================================================

def extract_articles(
    full_text,
    chapters,
):
    """
    Article boundary kết thúc tại vị trí sớm hơn:

        Article tiếp theo
        hoặc
        Chapter tiếp theo.

    Nhờ vậy Article cuối Chapter không nuốt
    heading Chapter mới.
    """

    article_matches = list(
        ARTICLE_RE.finditer(
            full_text
        )
    )

    chapter_matches = list(
        CHAPTER_RE.finditer(
            full_text
        )
    )

    articles = []

    for index, match in enumerate(
        article_matches
    ):

        article_start = (
            match.start()
        )

        # ----------------------------------------------------
        # Next Article
        # ----------------------------------------------------

        if (
            index + 1
            < len(
                article_matches
            )
        ):

            next_article_start = (
                article_matches[
                    index + 1
                ].start()
            )

        else:

            next_article_start = (
                len(full_text)
            )

        # ----------------------------------------------------
        # Next Chapter
        # ----------------------------------------------------

        next_chapter_start = (
            len(full_text)
        )

        for chapter_match in (
            chapter_matches
        ):

            if (
                chapter_match.start()
                > article_start
            ):

                next_chapter_start = (
                    chapter_match.start()
                )

                break

        # ----------------------------------------------------
        # Correct Article end
        # ----------------------------------------------------

        article_end = min(
            next_article_start,
            next_chapter_start,
        )

        raw_article_text = (
            full_text[
                article_start:
                article_end
            ]
        )

        article_number = (
            match.group(1)
        )

        first_title_part = (
            match.group(2)
        )

        (
            article_title,
            body_start_local,
        ) = parse_article_title(
            raw_article_text,
            first_title_part,
        )

        chapter = (
            find_chapter(
                article_start,
                chapters,
            )
        )

        articles.append(
            {
                "article_number": (
                    article_number
                ),
                "article_title": (
                    article_title
                ),
                "chapter": (
                    chapter
                ),
                "start": (
                    article_start
                ),
                "end": (
                    article_end
                ),
                "text": (
                    raw_article_text
                ),
                "body_start_local": (
                    body_start_local
                ),
            }
        )

    return articles


# ============================================================
# TRIM SOURCE SPAN
# ============================================================

def trim_span(
    text,
    start,
    end,
):
    """
    Trim whitespace nhưng giữ chính xác offsets.
    """

    while (
        start < end
        and text[
            start
        ].isspace()
    ):

        start += 1

    while (
        end > start
        and text[
            end - 1
        ].isspace()
    ):

        end -= 1

    return (
        start,
        end,
    )


# ============================================================
# FIND BODY END
# ============================================================

def find_body_end(
    text,
    body_start,
    clause_matches,
):
    """
    Loại trailing footnote khỏi legal body.

    Chỉ cắt footnote khi nó nằm sau
    top-level clause cuối cùng.

    Footnote nằm giữa Article không bị cắt,
    vì sau nó có thể còn legal text.
    """

    body_end = (
        len(text)
    )

    if clause_matches:

        last_clause_start = (
            clause_matches[
                -1
            ].start()
        )

    else:

        last_clause_start = (
            body_start
        )

    quoted_spans = (
        find_quoted_spans(
            text
        )
    )

    footnote_matches = list(
        FOOTNOTE_START_RE.finditer(
            text,
            body_start,
        )
    )

    for footnote in (
        footnote_matches
    ):

        # Footnote-like content inside quote
        if position_inside_spans(
            footnote.start(),
            quoted_spans,
        ):

            continue

        # Có clause thật ở phía sau
        # => footnote này nằm giữa Article,
        # không được dùng làm body boundary.
        if (
            footnote.start()
            <= last_clause_start
        ):

            continue

        legal_before = (
            text[
                last_clause_start:
                footnote.start()
            ]
            .rstrip()
        )

        if legal_before.endswith(
            (
                ".",
                ";",
                ":",
                "”",
                '"',
            )
        ):

            body_end = min(
                body_end,
                footnote.start(),
            )

            break

    # --------------------------------------------------------
    # Page number đứng riêng ở cuối
    # --------------------------------------------------------

    page_matches = list(
        PAGE_NUMBER_RE.finditer(
            text,
            body_start,
            body_end,
        )
    )

    for page_match in reversed(
        page_matches
    ):

        remaining = (
            text[
                page_match.end():
                body_end
            ]
            .strip()
        )

        if not remaining:

            body_end = (
                page_match.start()
            )

        else:

            break

    return body_end


# ============================================================
# CLEAN INTRO
# ============================================================

def clean_intro_range(
    text,
    start,
    end,
):
    """
    Giữ intro hợp lệ.

    Ví dụ Điều 22:

        Khi đến gần đường giao nhau...
        theo quy định sau đây:

    Nhưng loại footnote/page-number giả.
    """

    footnote = (
        FOOTNOTE_START_RE.search(
            text,
            start,
            end,
        )
    )

    if footnote:

        end = min(
            end,
            footnote.start(),
        )

    (
        start,
        end,
    ) = trim_span(
        text,
        start,
        end,
    )

    if start >= end:

        return (
            start,
            start,
        )

    content = (
        text[
            start:end
        ]
        .strip()
    )

    # Chỉ là số trang
    if re.fullmatch(
        r"\d{1,3}",
        content
    ):

        return (
            start,
            start,
        )

    return (
        start,
        end,
    )


# ============================================================
# SPLIT ARTICLE INTO LEGAL UNITS
# ============================================================

def split_article_units(
    article
):
    """
    Tách Article:

        Article
          ↓
        intro (nếu có)
          ↓
        clause 1
        clause 2
        clause 2a
        clause 3
        ...

    False-positive clauses trong quote
    hoặc numbering bị reset được loại bỏ.
    """

    text = (
        article["text"]
    )

    body_start = (
        article[
            "body_start_local"
        ]
    )

    # --------------------------------------------------------
    # Quote spans
    # --------------------------------------------------------

    quoted_spans = (
        find_quoted_spans(
            text
        )
    )

    # --------------------------------------------------------
    # Raw clause candidates
    # --------------------------------------------------------

    raw_clause_matches = list(
        CLAUSE_RE.finditer(
            text,
            body_start,
        )
    )

    # --------------------------------------------------------
    # Remove quoted / non-monotonic clauses
    # --------------------------------------------------------

    all_clause_matches = (
        filter_structural_clauses(
            raw_clause_matches,
            quoted_spans,
        )
    )

    # --------------------------------------------------------
    # Legal body end
    # --------------------------------------------------------

    body_end = (
        find_body_end(
            text=text,
            body_start=body_start,
            clause_matches=(
                all_clause_matches
            ),
        )
    )

    # --------------------------------------------------------
    # Clauses inside legal body only
    # --------------------------------------------------------

    clause_matches = [
        match
        for match in (
            all_clause_matches
        )
        if (
            match.start()
            < body_end
        )
    ]

    units = []

    # ========================================================
    # ARTICLE WITHOUT CLAUSES
    # ========================================================

    if not clause_matches:

        (
            start,
            end,
        ) = trim_span(
            text,
            body_start,
            body_end,
        )

        if start < end:

            # Nếu body cuối cùng bắt đầu chứa
            # trailing footnote thì cắt nó.
            footnote = (
                FOOTNOTE_START_RE.search(
                    text,
                    start,
                    end,
                )
            )

            if (
                footnote
                and not position_inside_spans(
                    footnote.start(),
                    quoted_spans,
                )
            ):

                end = min(
                    end,
                    footnote.start(),
                )

                (
                    start,
                    end,
                ) = trim_span(
                    text,
                    start,
                    end,
                )

        if start < end:

            units.append(
                {
                    "clause": None,
                    "text": (
                        text[
                            start:end
                        ]
                    ),
                    "local_start": (
                        start
                    ),
                    "local_end": (
                        end
                    ),
                }
            )

        return units

    # ========================================================
    # INTRO BEFORE FIRST CLAUSE
    # ========================================================

    intro_start = (
        body_start
    )

    intro_end = (
        clause_matches[
            0
        ].start()
    )

    (
        intro_start,
        intro_end,
    ) = clean_intro_range(
        text,
        intro_start,
        intro_end,
    )

    if (
        intro_start
        < intro_end
    ):

        units.append(
            {
                "clause": "intro",
                "text": (
                    text[
                        intro_start:
                        intro_end
                    ]
                ),
                "local_start": (
                    intro_start
                ),
                "local_end": (
                    intro_end
                ),
            }
        )

    # ========================================================
    # STRUCTURAL CLAUSES
    # ========================================================

    for index, match in enumerate(
        clause_matches
    ):

        start = (
            match.start()
        )

        if (
            index + 1
            < len(
                clause_matches
            )
        ):

            end = (
                clause_matches[
                    index + 1
                ].start()
            )

        else:

            end = (
                body_end
            )

        (
            start,
            end,
        ) = trim_span(
            text,
            start,
            end,
        )

        if (
            start
            >= end
        ):

            continue

        units.append(
            {
                "clause": (
                    match.group(1)
                ),
                "text": (
                    text[
                        start:end
                    ]
                ),
                "local_start": (
                    start
                ),
                "local_end": (
                    end
                ),
            }
        )

    return units


# ============================================================
# FALLBACK SPLIT
# ============================================================

def fallback_split(
    text,
    max_size,
    overlap,
):
    """
    Character fallback chỉ dùng nếu một
    legal unit vượt MAX_CHUNK_SIZE.

    Không dùng để chia toàn document.
    """

    if (
        max_size
        <= 0
    ):

        raise ValueError(
            "max_size must be > 0"
        )

    if overlap < 0:

        raise ValueError(
            "overlap cannot be negative"
        )

    if (
        overlap
        >= max_size
    ):

        raise ValueError(
            "overlap must be smaller "
            "than max_size"
        )

    pieces = []

    start = 0

    while (
        start
        < len(text)
    ):

        end = min(
            start + max_size,
            len(text),
        )

        piece = (
            text[
                start:end
            ]
        )

        if piece.strip():

            pieces.append(
                {
                    "text": piece,
                    "start": start,
                    "end": end,
                }
            )

        if (
            end
            >= len(text)
        ):

            break

        start = (
            end
            - overlap
        )

    return pieces


# ============================================================
# CREATE STRUCTURE CHUNKS
# ============================================================

def create_structure_chunks(
    articles,
    page_ranges,
    document_id,
):
    """
    C1:

        Chapter
           ↓
        Article
           ↓
        Clause
           ↓
        fallback pieces
    """

    chunks = []

    for article in articles:

        article_number = (
            article[
                "article_number"
            ]
        )

        article_title = (
            article[
                "article_title"
            ]
        )

        chapter = (
            article[
                "chapter"
            ]
        )

        article_heading = (
            f"Điều {article_number}. "
            f"{article_title}"
        ).strip()

        legal_units = (
            split_article_units(
                article
            )
        )

        chunk_counter = 0

        for unit in legal_units:

            unit_text = (
                unit[
                    "text"
                ]
            )

            clause = (
                unit[
                    "clause"
                ]
            )

            # ------------------------------------------------
            # Heading cũng tính vào MAX_CHUNK_SIZE
            # ------------------------------------------------

            available_size = (
                MAX_CHUNK_SIZE
                - len(
                    article_heading
                )
                - 1
            )

            if (
                available_size
                <= 0
            ):

                raise ValueError(
                    "Article heading too long: "
                    f"Điều {article_number}"
                )

            # =================================================
            # UNIT FITS
            # =================================================

            if (
                len(unit_text)
                <= available_size
            ):

                pieces = [
                    {
                        "text": (
                            unit_text
                        ),
                        "start": 0,
                        "end": (
                            len(
                                unit_text
                            )
                        ),
                    }
                ]

            # =================================================
            # FALLBACK SPLIT
            # =================================================

            else:

                overlap = min(
                    FALLBACK_OVERLAP,
                    available_size - 1,
                )

                pieces = (
                    fallback_split(
                        text=unit_text,
                        max_size=(
                            available_size
                        ),
                        overlap=(
                            overlap
                        ),
                    )
                )

            # =================================================
            # BUILD CHUNKS
            # =================================================

            for (
                piece_index,
                piece
            ) in enumerate(
                pieces,
                start=1
            ):

                chunk_counter += 1

                # --------------------------------------------
                # Source offsets inside Article
                # --------------------------------------------

                source_local_start = (
                    unit[
                        "local_start"
                    ]
                    + piece[
                        "start"
                    ]
                )

                source_local_end = (
                    unit[
                        "local_start"
                    ]
                    + piece[
                        "end"
                    ]
                )

                # --------------------------------------------
                # Global source offsets
                # --------------------------------------------

                source_global_start = (
                    article[
                        "start"
                    ]
                    + source_local_start
                )

                source_global_end = (
                    article[
                        "start"
                    ]
                    + source_local_end
                )

                # --------------------------------------------
                # Page metadata
                # --------------------------------------------

                page_start = (
                    position_to_page(
                        source_global_start,
                        page_ranges,
                    )
                )

                page_end = (
                    position_to_page(
                        max(
                            source_global_start,
                            source_global_end
                            - 1,
                        ),
                        page_ranges,
                    )
                )

                # --------------------------------------------
                # Text được embed.
                #
                # Luôn prepend Article heading để giữ context.
                # --------------------------------------------

                chunk_text = (
                    article_heading
                    + "\n"
                    + piece[
                        "text"
                    ]
                )

                # --------------------------------------------
                # ID
                # --------------------------------------------

                chunk_id = (
                    f"{document_id}"
                    f"_a{article_number}"
                    f"_c{chunk_counter:03d}"
                )

                chunks.append(
                    {
                        "chunk_id": (
                            chunk_id
                        ),
                        "document_id": (
                            document_id
                        ),
                        "chapter": (
                            chapter
                        ),
                        "article_number": (
                            article_number
                        ),
                        "article_title": (
                            article_title
                        ),
                        "clause": (
                            clause
                        ),
                        "piece_index": (
                            piece_index
                        ),
                        "page_start": (
                            page_start
                        ),
                        "page_end": (
                            page_end
                        ),
                        "source_start": (
                            source_global_start
                        ),
                        "source_end": (
                            source_global_end
                        ),
                        "char_count": (
                            len(
                                chunk_text
                            )
                        ),
                        "text": (
                            chunk_text
                        ),
                    }
                )

    return chunks


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_chunks(
    chunks
):
    """
    Validation nhanh ngay sau build.

    Full validation vẫn chạy bằng:
        scripts/validate_c1_chunks.py
    """

    errors = []

    seen_ids = set()

    for chunk in chunks:

        chunk_id = (
            chunk[
                "chunk_id"
            ]
        )

        # ----------------------------------------------------
        # Duplicate ID
        # ----------------------------------------------------

        if (
            chunk_id
            in seen_ids
        ):

            errors.append(
                f"Duplicate ID: "
                f"{chunk_id}"
            )

        seen_ids.add(
            chunk_id
        )

        # ----------------------------------------------------
        # Empty
        # ----------------------------------------------------

        if not (
            chunk[
                "text"
            ]
            .strip()
        ):

            errors.append(
                f"Empty chunk: "
                f"{chunk_id}"
            )

        # ----------------------------------------------------
        # char_count
        # ----------------------------------------------------

        if (
            chunk[
                "char_count"
            ]
            != len(
                chunk[
                    "text"
                ]
            )
        ):

            errors.append(
                f"char_count mismatch: "
                f"{chunk_id}"
            )

        # ----------------------------------------------------
        # Maximum size
        # ----------------------------------------------------

        if (
            chunk[
                "char_count"
            ]
            > MAX_CHUNK_SIZE
        ):

            errors.append(
                f"Chunk too large: "
                f"{chunk_id} "
                f"({chunk['char_count']})"
            )

        # ----------------------------------------------------
        # Source range
        # ----------------------------------------------------

        if (
            chunk[
                "source_start"
            ]
            < 0
        ):

            errors.append(
                f"Negative source_start: "
                f"{chunk_id}"
            )

        if (
            chunk[
                "source_end"
            ]
            <= chunk[
                "source_start"
            ]
        ):

            errors.append(
                f"Invalid source range: "
                f"{chunk_id}"
            )

        # ----------------------------------------------------
        # Page range
        # ----------------------------------------------------

        if (
            chunk[
                "page_start"
            ]
            is None
        ):

            errors.append(
                f"No page_start: "
                f"{chunk_id}"
            )

        if (
            chunk[
                "page_end"
            ]
            is None
        ):

            errors.append(
                f"No page_end: "
                f"{chunk_id}"
            )

        if (
            chunk[
                "page_start"
            ]
            is not None
            and chunk[
                "page_end"
            ]
            is not None
            and chunk[
                "page_start"
            ]
            > chunk[
                "page_end"
            ]
        ):

            errors.append(
                f"Invalid page range: "
                f"{chunk_id}"
            )

        # ----------------------------------------------------
        # Chapter leakage
        # ----------------------------------------------------

        heading = (
            f"Điều "
            f"{chunk['article_number']}. "
            f"{chunk['article_title']}"
        )

        body = (
            chunk[
                "text"
            ][
                len(heading):
            ]
        )

        if CHAPTER_RE.search(
            body
        ):

            errors.append(
                "Chapter heading leaked "
                f"into chunk: {chunk_id}"
            )

        # ----------------------------------------------------
        # Title contamination
        # ----------------------------------------------------

        if (
            "Khoản này được sửa đổi"
            in chunk[
                "article_title"
            ]
        ):

            errors.append(
                "Footnote leaked into "
                f"title: {chunk_id}"
            )

    return errors


# ============================================================
# SAVE
# ============================================================

def save_chunks(
    chunks,
    output_path,
):
    """
    Save chunks JSONL.
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


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    pages,
    chapters,
    articles,
    chunks,
    errors,
):
    """
    Print C1 summary.
    """

    sizes = [
        chunk[
            "char_count"
        ]
        for chunk in chunks
    ]

    intro_chunks = [
        chunk
        for chunk in chunks
        if (
            chunk[
                "clause"
            ]
            == "intro"
        )
    ]

    multi_page_chunks = [
        chunk
        for chunk in chunks
        if (
            chunk[
                "page_start"
            ]
            != chunk[
                "page_end"
            ]
        )
    ]

    fallback_pieces = [
        chunk
        for chunk in chunks
        if (
            chunk[
                "piece_index"
            ]
            > 1
        )
    ]

    articles_represented = {
        chunk[
            "article_number"
        ]
        for chunk in chunks
    }

    print(
        "=" * 70
    )

    print(
        "C1 STRUCTURE-AWARE CHUNKING"
    )

    print(
        "=" * 70
    )

    print(
        f"Pages processed     : "
        f"{len(pages)}"
    )

    print(
        f"Chapters detected   : "
        f"{len(chapters)}"
    )

    print(
        f"Articles detected   : "
        f"{len(articles)}"
    )

    print(
        f"Articles represented: "
        f"{len(articles_represented)}"
    )

    print(
        f"Total chunks        : "
        f"{len(chunks)}"
    )

    if sizes:

        print(
            f"Average chunk size  : "
            f"{sum(sizes) / len(sizes):.2f}"
        )

        print(
            f"Min chunk size      : "
            f"{min(sizes)}"
        )

        print(
            f"Max chunk size      : "
            f"{max(sizes)}"
        )

    print(
        f"Intro chunks        : "
        f"{len(intro_chunks)}"
    )

    print(
        f"Fallback pieces     : "
        f"{len(fallback_pieces)}"
    )

    print(
        f"Multi-page chunks   : "
        f"{len(multi_page_chunks)}"
    )

    print(
        f"Max chunk size conf : "
        f"{MAX_CHUNK_SIZE}"
    )

    print(
        f"Fallback overlap    : "
        f"{FALLBACK_OVERLAP}"
    )

    print(
        f"Validation errors   : "
        f"{len(errors)}"
    )

    if errors:

        print()
        print(
            "First errors:"
        )

        for error in (
            errors[:20]
        ):

            print(
                " -",
                error
            )

    else:

        print()

        print(
            "ALL C1 CHUNKS VALID"
        )

    print()

    print(
        f"Saved to            : "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # INPUT CHECK
    # ========================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input not found: "
            f"{INPUT_FILE}"
        )

    # ========================================================
    # LOAD CLEAN DATA
    # ========================================================

    pages = (
        load_jsonl(
            INPUT_FILE
        )
    )

    if not pages:

        raise ValueError(
            "No clean pages found"
        )

    document_id = (
        pages[0][
            "document_id"
        ]
    )

    # ========================================================
    # GLOBAL DOCUMENT
    # ========================================================

    (
        full_text,
        page_ranges,
    ) = build_global_document(
        pages
    )

    # ========================================================
    # CHAPTERS
    # ========================================================

    chapters = (
        extract_chapters(
            full_text
        )
    )

    # ========================================================
    # ARTICLES
    # ========================================================

    articles = (
        extract_articles(
            full_text,
            chapters,
        )
    )

    # ========================================================
    # STRUCTURE-AWARE CHUNKS
    # ========================================================

    chunks = (
        create_structure_chunks(
            articles=articles,
            page_ranges=page_ranges,
            document_id=document_id,
        )
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    errors = (
        validate_chunks(
            chunks
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_chunks(
        chunks,
        OUTPUT_FILE,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print_statistics(
        pages=pages,
        chapters=chapters,
        articles=articles,
        chunks=chunks,
        errors=errors,
    )

    if errors:

        raise SystemExit(1)


if __name__ == "__main__":
    main()