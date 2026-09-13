import re
import unicodedata
from dataclasses import dataclass, asdict
from typing import List


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa Unicode NFC và khoảng trắng.
    """

    text = unicodedata.normalize(
        "NFC",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_lower(text: str) -> str:
    """
    Normalize + lowercase.
    """

    return normalize_text(
        text
    ).lower()


# ============================================================
# LANGUAGE DETECTION
# ============================================================

VIETNAMESE_CHARS = set(
    "ăâđêôơư"
    "áàảãạ"
    "ắằẳẵặ"
    "ấầẩẫậ"
    "éèẻẽẹ"
    "ếềểễệ"
    "íìỉĩị"
    "óòỏõọ"
    "ốồổỗộ"
    "ớờởỡợ"
    "úùủũụ"
    "ứừửữự"
    "ýỳỷỹỵ"
)


VIETNAMESE_COMMON_WORDS = {
    "là",
    "và",
    "của",
    "được",
    "phải",
    "không",
    "người",
    "những",
    "các",
    "trong",
    "khi",
    "nào",
    "giao",
    "thông",
    "đường",
    "bộ",
    "trách",
    "nhiệm",
    "điều",
    "khoản",
    "quy",
    "định",
    "pháp",
    "luật",
    "xe",
    "phương",
    "tiện",
    "quyền",
    "hoạt",
    "động",
}


ENGLISH_COMMON_WORDS = {
    "the",
    "is",
    "are",
    "what",
    "which",
    "who",
    "when",
    "where",
    "how",
    "must",
    "should",
    "road",
    "traffic",
    "vehicle",
    "driver",
    "law",
    "responsibility",
    "right",
    "rights",
    "condition",
    "purpose",
}


def detect_language(text: str) -> dict:
    """
    Language detector heuristic.

    Hiện tại chủ yếu phục vụ:
        vi
        en
        unknown

    confidence ở đây là heuristic score,
    không phải calibrated probability.
    """

    normalized = normalize_lower(
        text
    )

    tokens = re.findall(
        r"(?u)\b\w+\b",
        normalized
    )

    vi_char_count = sum(
        char in VIETNAMESE_CHARS
        for char in normalized
    )

    vi_word_count = sum(
        token in VIETNAMESE_COMMON_WORDS
        for token in tokens
    )

    en_word_count = sum(
        token in ENGLISH_COMMON_WORDS
        for token in tokens
    )

    vi_score = (
        vi_char_count * 2.0
        + vi_word_count
    )

    en_score = float(
        en_word_count
    )

    # --------------------------------------------------------
    # VIETNAMESE
    # --------------------------------------------------------

    if vi_score >= 2:

        language = "vi"

        confidence = min(
            1.0,
            0.70
            + vi_score / 20.0
        )

    # --------------------------------------------------------
    # ENGLISH
    # --------------------------------------------------------

    elif en_score >= 2:

        language = "en"

        confidence = min(
            1.0,
            0.65
            + en_score / 20.0
        )

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    else:

        language = "unknown"

        confidence = 0.50

    return {
        "language": language,

        "confidence": round(
            confidence,
            4
        ),

        "vi_score": round(
            vi_score,
            4
        ),

        "en_score": round(
            en_score,
            4
        ),
    }


# ============================================================
# EXPLICIT ARTICLE / CLAUSE REFERENCES
# ============================================================

ARTICLE_PATTERN = re.compile(
    r"\bđiều\s+(\d+)\b",
    re.IGNORECASE
)


CLAUSE_PATTERN = re.compile(
    r"\bkhoản\s+(\d+[a-zA-Z]?)\b",
    re.IGNORECASE
)


def extract_article_numbers(
    text: str
) -> List[int]:
    """
    Ví dụ:
        "Theo Điều 87..."
            -> [87]
    """

    values = []

    for value in ARTICLE_PATTERN.findall(
        text
    ):

        number = int(
            value
        )

        if number not in values:

            values.append(
                number
            )

    return values


def extract_clause_numbers(
    text: str
) -> List[str]:
    """
    Ví dụ:
        "Điều 87 khoản 2"
            -> ["2"]
    """

    values = []

    for value in CLAUSE_PATTERN.findall(
        text
    ):

        value = value.lower()

        if value not in values:

            values.append(
                value
            )

    return values


# ============================================================
# LEGAL ENTITY EXTRACTION
# ============================================================

KNOWN_ENTITIES = {
    "chính phủ":
        "Chính phủ",

    "bộ công an":
        "Bộ Công an",

    "bộ xây dựng":
        "Bộ Xây dựng",

    "bộ quốc phòng":
        "Bộ Quốc phòng",

    "ủy ban nhân dân":
        "Ủy ban nhân dân",

    "uỷ ban nhân dân":
        "Ủy ban nhân dân",

    "cảnh sát giao thông":
        "Cảnh sát giao thông",

    "bộ, cơ quan ngang bộ":
        "Bộ, cơ quan ngang Bộ",
}


def extract_entities(
    text: str
) -> List[str]:
    """
    Entity extractor rule-based.

    Hiện tại chỉ nhận các entity có ích trực tiếp
    cho traffic-law dataset.
    """

    lowered = normalize_lower(
        text
    )

    found = []

    for phrase, canonical in (
        KNOWN_ENTITIES.items()
    ):

        if phrase in lowered:

            if canonical not in found:

                found.append(
                    canonical
                )

    return found


# ============================================================
# QUERY INTENT RULES
# ============================================================

INTENT_RULES = {

    # --------------------------------------------------------
    # RESPONSIBILITY / DUTY
    # --------------------------------------------------------

    "responsibility": [
        r"\btrách nhiệm\b",
        r"\bchịu trách nhiệm\b",
        r"\bcơ quan nào\b",
        r"\bai chịu trách nhiệm\b",
        r"\bquản lý nhà nước\b",
        r"\bnhiệm vụ gì\b",
        r"\bcó nhiệm vụ\b",
    ],

    # --------------------------------------------------------
    # RIGHTS
    # --------------------------------------------------------

    "rights": [
        r"\bquyền gì\b",
        r"\bcó những quyền\b",
        r"\bquyền của\b",
        r"\bđược quyền\b",
    ],

    # --------------------------------------------------------
    # PROHIBITION
    # --------------------------------------------------------

    "prohibition": [
        r"\bcấm\b",
        r"\bkhông được\b",
        r"\bcó được .* không\b",
        r"\bhành vi bị nghiêm cấm\b",
    ],

    # --------------------------------------------------------
    # CONDITION
    # --------------------------------------------------------

    "condition": [
        r"\bđiều kiện\b",
        r"\bcần đáp ứng\b",
        r"\bđủ điều kiện\b",
        r"\bphải có\b",
        r"\bđủ tuổi\b",
        r"\bsức khỏe\b",
    ],

    # --------------------------------------------------------
    # PROCEDURE
    # --------------------------------------------------------

    "procedure": [
        r"\bthủ tục\b",
        r"\btrình tự\b",
        r"\bhồ sơ\b",
        r"\bcấp lại\b",
        r"\bđổi giấy phép\b",
        r"\bthu hồi\b",
    ],

    # --------------------------------------------------------
    # FUNCTION / PURPOSE
    # --------------------------------------------------------

    "function_purpose": [
        r"\bdùng để làm gì\b",
        r"\bđược dùng để\b",
        r"\bđể làm gì\b",
        r"\bcó tác dụng gì\b",
        r"\bmục đích\b",
    ],

    # --------------------------------------------------------
    # DEFINITION / SCOPE
    # --------------------------------------------------------

    "definition_scope": [
        r"\blà gì\b",
        r"\blà hoạt động gì\b",
        r"\bđược hiểu là\b",
        r"\bgiải thích\b",
        r"\bđiều chỉnh những nội dung gì\b",
        r"\bphạm vi điều chỉnh\b",
        r"\bđối tượng áp dụng\b",
    ],

    # --------------------------------------------------------
    # RULE / OBLIGATION
    # --------------------------------------------------------

    "rule_obligation": [
        r"\bphải làm gì\b",
        r"\bphải thực hiện\b",
        r"\bphải\b",
        r"\bnhư thế nào\b",
        r"\bkhi nào\b",
        r"\bnhường đường\b",
        r"\bgiảm tốc độ\b",
        r"\bquy định\b",
    ],
}


# ============================================================
# INTENT PRIORITY
# ============================================================

INTENT_PRIORITY = [
    "responsibility",
    "rights",
    "prohibition",
    "condition",
    "procedure",
    "function_purpose",
    "definition_scope",
    "rule_obligation",
]


def classify_intent(
    text: str
) -> dict:
    """
    Rule-based query intent classification.

    Trả về:
        intent
        confidence
        scores
        matched_rules
    """

    lowered = normalize_lower(
        text
    )

    scores = {}

    matches = {}

    # --------------------------------------------------------
    # SCORE EACH INTENT
    # --------------------------------------------------------

    for intent, patterns in (
        INTENT_RULES.items()
    ):

        intent_matches = []

        for pattern in patterns:

            if re.search(
                pattern,
                lowered,
                flags=re.IGNORECASE,
            ):

                intent_matches.append(
                    pattern
                )

        matches[
            intent
        ] = intent_matches

        scores[
            intent
        ] = len(
            intent_matches
        )

    # --------------------------------------------------------
    # CHOOSE BEST INTENT
    # --------------------------------------------------------

    best_intent = "general"

    best_score = 0

    for intent in INTENT_PRIORITY:

        score = scores[
            intent
        ]

        if score > best_score:

            best_score = score

            best_intent = intent

    # --------------------------------------------------------
    # HEURISTIC CONFIDENCE
    # --------------------------------------------------------

    if best_score == 0:

        confidence = 0.50

    elif best_score == 1:

        confidence = 0.75

    elif best_score == 2:

        confidence = 0.90

    else:

        confidence = 0.97

    return {
        "intent": (
            best_intent
        ),

        "confidence": (
            confidence
        ),

        "scores": (
            scores
        ),

        "matched_rules": (
            matches.get(
                best_intent,
                []
            )
        ),
    }


# ============================================================
# MULTI-EVIDENCE DETECTION
# ============================================================

MULTI_EVIDENCE_PATTERNS = [
    r"\bnhững cơ quan nào\b",
    r"\bcác cơ quan\b",

    r"\bnhững trường hợp nào\b",
    r"\bcác trường hợp\b",

    r"\bbao gồm những\b",
    r"\bgồm những\b",

    r"\bcó những loại nào\b",

    r"\bliệt kê\b",

    r"\bnhững đối tượng nào\b",
    r"\bcác đối tượng\b",

    r"\bnhững ai\b",
]


def detect_multi_evidence(
    text: str
) -> dict:
    """
    Phát hiện câu hỏi có khả năng cần nhiều evidence.

    Lưu ý:
    Không phải mọi câu chứa "những" đều multi-evidence.
    """

    lowered = normalize_lower(
        text
    )

    matched = []

    for pattern in (
        MULTI_EVIDENCE_PATTERNS
    ):

        if re.search(
            pattern,
            lowered,
            flags=re.IGNORECASE,
        ):

            matched.append(
                pattern
            )

    multi_evidence = (
        len(matched) > 0
    )

    if len(matched) >= 2:

        confidence = 0.95

    elif len(matched) == 1:

        confidence = 0.85

    else:

        confidence = 0.70

    return {
        "multi_evidence": (
            multi_evidence
        ),

        "confidence": (
            confidence
        ),

        "matched_rules": (
            matched
        ),
    }


# ============================================================
# FILTER STRATEGY
# ============================================================

def choose_filter_strategy(
    article_numbers,
    clause_numbers,
    entities,
    router_confidence,
):
    """
    Safe routing principle:

    HARD FILTER
        chỉ khi user nêu Điều cụ thể.

    SOFT FILTER
        khi có entity rõ ràng.

    NONE
        nếu router chỉ đang suy đoán semantic intent.

    Không bao giờ hard-filter chỉ dựa trên intent.
    """

    # --------------------------------------------------------
    # HARD FILTER
    # --------------------------------------------------------

    if article_numbers:

        return {
            "strategy": "hard",

            "reason": (
                "explicit_article_reference"
            ),

            "article_numbers": (
                article_numbers
            ),

            "clause_numbers": (
                clause_numbers
            ),

            "entities": (
                entities
            ),
        }

    # --------------------------------------------------------
    # SOFT FILTER / BOOST
    # --------------------------------------------------------

    if (
        entities
        and router_confidence >= 0.75
    ):

        return {
            "strategy": "soft",

            "reason": (
                "recognized_entity"
            ),

            "article_numbers": [],

            "clause_numbers": [],

            "entities": (
                entities
            ),
        }

    # --------------------------------------------------------
    # NO FILTER
    # --------------------------------------------------------

    return {
        "strategy": "none",

        "reason": (
            "insufficient_safe_filter_signal"
        ),

        "article_numbers": [],

        "clause_numbers": [],

        "entities": [],
    }


# ============================================================
# ROUTE RESULT DATA MODEL
# ============================================================

@dataclass
class RouteResult:

    question: str

    language: str

    language_confidence: float

    intent: str

    intent_confidence: float

    multi_evidence: bool

    multi_evidence_confidence: float

    article_numbers: List[int]

    clause_numbers: List[str]

    entities: List[str]

    filter_strategy: str

    filter_reason: str

    candidate_top_k: int

    final_top_k: int


# ============================================================
# QUERY ROUTER
# ============================================================

class QueryRouter:

    def __init__(
        self,
        default_candidate_top_k=20,
        multi_evidence_candidate_top_k=30,
        final_top_k=5,
    ):

        self.default_candidate_top_k = (
            default_candidate_top_k
        )

        self.multi_evidence_candidate_top_k = (
            multi_evidence_candidate_top_k
        )

        self.final_top_k = (
            final_top_k
        )


    def route(
        self,
        question: str,
    ) -> dict:
        """
        Main routing function.
        """

        question = normalize_text(
            question
        )

        if not question:

            raise ValueError(
                "Question is empty."
            )

        # ====================================================
        # LANGUAGE
        # ====================================================

        language_result = (
            detect_language(
                question
            )
        )

        # ====================================================
        # QUERY INTENT
        # ====================================================

        intent_result = (
            classify_intent(
                question
            )
        )

        # ====================================================
        # MULTI-EVIDENCE
        # ====================================================

        multi_result = (
            detect_multi_evidence(
                question
            )
        )

        # ====================================================
        # ARTICLE REFERENCES
        # ====================================================

        article_numbers = (
            extract_article_numbers(
                question
            )
        )

        # ====================================================
        # CLAUSE REFERENCES
        # ====================================================

        clause_numbers = (
            extract_clause_numbers(
                question
            )
        )

        # ====================================================
        # ENTITIES
        # ====================================================

        entities = (
            extract_entities(
                question
            )
        )

        # ====================================================
        # FILTER STRATEGY
        # ====================================================

        filter_result = (
            choose_filter_strategy(
                article_numbers=(
                    article_numbers
                ),

                clause_numbers=(
                    clause_numbers
                ),

                entities=(
                    entities
                ),

                router_confidence=(
                    intent_result[
                        "confidence"
                    ]
                ),
            )
        )

        # ====================================================
        # CANDIDATE BUDGET
        # ====================================================

        if multi_result[
            "multi_evidence"
        ]:

            candidate_top_k = (
                self
                .multi_evidence_candidate_top_k
            )

        else:

            candidate_top_k = (
                self
                .default_candidate_top_k
            )

        # ====================================================
        # FINAL ROUTE RESULT
        # ====================================================

        result = RouteResult(

            question=(
                question
            ),

            language=(
                language_result[
                    "language"
                ]
            ),

            language_confidence=(
                language_result[
                    "confidence"
                ]
            ),

            intent=(
                intent_result[
                    "intent"
                ]
            ),

            intent_confidence=(
                intent_result[
                    "confidence"
                ]
            ),

            multi_evidence=(
                multi_result[
                    "multi_evidence"
                ]
            ),

            multi_evidence_confidence=(
                multi_result[
                    "confidence"
                ]
            ),

            article_numbers=(
                article_numbers
            ),

            clause_numbers=(
                clause_numbers
            ),

            entities=(
                entities
            ),

            filter_strategy=(
                filter_result[
                    "strategy"
                ]
            ),

            filter_reason=(
                filter_result[
                    "reason"
                ]
            ),

            candidate_top_k=(
                candidate_top_k
            ),

            final_top_k=(
                self.final_top_k
            ),
        )

        output = asdict(
            result
        )

        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        output[
            "diagnostics"
        ] = {

            "intent_scores": (
                intent_result[
                    "scores"
                ]
            ),

            "intent_matched_rules": (
                intent_result[
                    "matched_rules"
                ]
            ),

            "multi_evidence_rules": (
                multi_result[
                    "matched_rules"
                ]
            ),

            "language_scores": {

                "vi": (
                    language_result[
                        "vi_score"
                    ]
                ),

                "en": (
                    language_result[
                        "en_score"
                    ]
                ),
            },
        }

        return output


# ============================================================
# DISPLAY ROUTE RESULT
# ============================================================

def display_route(
    route
):

    print()

    print(
        "=" * 85
    )

    print(
        "C4 QUERY ROUTER"
    )

    print(
        "=" * 85
    )

    print(
        f"Question          : "
        f"{route['question']}"
    )

    print(
        f"Language          : "
        f"{route['language']} "
        f"({route['language_confidence']:.2f})"
    )

    print(
        f"Intent            : "
        f"{route['intent']} "
        f"({route['intent_confidence']:.2f})"
    )

    print(
        f"Multi evidence    : "
        f"{route['multi_evidence']} "
        f"({route['multi_evidence_confidence']:.2f})"
    )

    print(
        f"Articles          : "
        f"{route['article_numbers']}"
    )

    print(
        f"Clauses           : "
        f"{route['clause_numbers']}"
    )

    print(
        f"Entities          : "
        f"{route['entities']}"
    )

    print(
        f"Filter strategy   : "
        f"{route['filter_strategy']}"
    )

    print(
        f"Filter reason     : "
        f"{route['filter_reason']}"
    )

    print(
        f"Candidate Top-K   : "
        f"{route['candidate_top_k']}"
    )

    print(
        f"Final Top-K       : "
        f"{route['final_top_k']}"
    )


# ============================================================
# INTERACTIVE TEST
# ============================================================

def main():

    router = (
        QueryRouter()
    )

    print(
        "=" * 85
    )

    print(
        "C4 QUERY ROUTER READY"
    )

    print(
        "=" * 85
    )

    print(
        "Nhập câu hỏi. Gõ 'exit' để thoát."
    )

    while True:

        print()

        question = input(
            "Câu hỏi > "
        ).strip()

        if not question:

            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:

            print(
                "Đã thoát."
            )

            break

        try:

            route = (
                router.route(
                    question
                )
            )

            display_route(
                route
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()