import re
from dataclasses import dataclass, asdict
from typing import List


# ============================================================
# CITATION REGEX
# ============================================================

CITATION_PATTERN = re.compile(
    r"\[S(\d+)\]",
    re.IGNORECASE,
)


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass
class CitationValidationResult:

    valid: bool

    has_citation: bool

    cited_source_ids: List[str]

    allowed_source_ids: List[str]

    invalid_source_ids: List[str]

    missing_citation: bool

    reason: str


# ============================================================
# EXTRACT CITATIONS
# ============================================================

def extract_citations(
    answer: str
) -> List[str]:

    if not answer:
        return []

    matches = (
        CITATION_PATTERN.findall(
            answer
        )
    )

    output = []

    for value in matches:

        source_id = (
            f"S{int(value)}"
        )

        if source_id not in output:
            output.append(
                source_id
            )

    return output


# ============================================================
# CITATION VALIDATOR
# ============================================================

class CitationValidator:
    """
    Kiểm tra citation integrity.

    Hiện tại validator kiểm tra:

    1. Answer có citation hay không.
    2. Citation có nằm trong source được cung cấp không.
    3. Model có tạo source ID không tồn tại không.

    Lưu ý:
    Đây chưa phải semantic entailment validation.
    Nó chưa xác minh câu được citation có thực sự được
    source hỗ trợ về mặt ngữ nghĩa.
    """

    def validate(
        self,
        answer,
        sources,
        require_citation=True,
    ):

        allowed_source_ids = [
            source[
                "source_id"
            ]
            for source
            in sources
        ]

        cited_source_ids = (
            extract_citations(
                answer
            )
        )

        has_citation = (
            len(
                cited_source_ids
            )
            > 0
        )

        invalid_source_ids = [
            source_id
            for source_id
            in cited_source_ids
            if source_id
            not in allowed_source_ids
        ]

        missing_citation = (
            require_citation
            and not has_citation
        )

        # ----------------------------------------------------
        # INVALID SOURCE
        # ----------------------------------------------------

        if invalid_source_ids:

            result = CitationValidationResult(

                valid=False,

                has_citation=(
                    has_citation
                ),

                cited_source_ids=(
                    cited_source_ids
                ),

                allowed_source_ids=(
                    allowed_source_ids
                ),

                invalid_source_ids=(
                    invalid_source_ids
                ),

                missing_citation=(
                    missing_citation
                ),

                reason=(
                    "invalid_source_reference"
                ),
            )

            return asdict(
                result
            )

        # ----------------------------------------------------
        # MISSING CITATION
        # ----------------------------------------------------

        if missing_citation:

            result = CitationValidationResult(

                valid=False,

                has_citation=False,

                cited_source_ids=[],

                allowed_source_ids=(
                    allowed_source_ids
                ),

                invalid_source_ids=[],

                missing_citation=True,

                reason=(
                    "missing_required_citation"
                ),
            )

            return asdict(
                result
            )

        # ----------------------------------------------------
        # PASS
        # ----------------------------------------------------

        result = CitationValidationResult(

            valid=True,

            has_citation=(
                has_citation
            ),

            cited_source_ids=(
                cited_source_ids
            ),

            allowed_source_ids=(
                allowed_source_ids
            ),

            invalid_source_ids=[],

            missing_citation=False,

            reason=(
                "citation_integrity_pass"
            ),
        )

        return asdict(
            result
        )


# ============================================================
# BUILD RETRY FEEDBACK
# ============================================================

def build_citation_retry_feedback(
    validation,
):

    allowed = (
        validation[
            "allowed_source_ids"
        ]
    )

    allowed_text = (
        ", ".join(
            f"[{source_id}]"
            for source_id
            in allowed
        )
    )

    if validation[
        "reason"
    ] == "missing_required_citation":

        return (
            "Câu trả lời trước chưa có citation. "
            "Hãy tạo lại câu trả lời và trích dẫn ít nhất "
            "một nguồn phù hợp. "
            f"Chỉ được sử dụng: {allowed_text}."
        )

    if validation[
        "reason"
    ] == "invalid_source_reference":

        invalid = (
            ", ".join(
                f"[{source_id}]"
                for source_id
                in validation[
                    "invalid_source_ids"
                ]
            )
        )

        return (
            f"Câu trả lời trước đã sử dụng citation "
            f"không tồn tại: {invalid}. "
            f"Hãy tạo lại câu trả lời. "
            f"Chỉ được sử dụng: {allowed_text}."
        )

    return (
        "Hãy tạo lại câu trả lời chỉ dựa trên "
        "các nguồn được cung cấp."
    )


# ============================================================
# SIMPLE TEST
# ============================================================

def main():

    validator = (
        CitationValidator()
    )

    sources = [
        {
            "source_id": "S1"
        },
        {
            "source_id": "S2"
        },
        {
            "source_id": "S3"
        },
    ]

    examples = [

        (
            "Thông tin này được quy định trong tài liệu [S1].",
            True,
        ),

        (
            "Thông tin này có trong [S2] và [S3].",
            True,
        ),

        (
            "Thông tin này có trong [S7].",
            False,
        ),

        (
            "Thông tin này được quy định trong tài liệu.",
            False,
        ),
    ]

    print(
        "=" * 72
    )

    print(
        "CITATION VALIDATOR TEST"
    )

    print(
        "=" * 72
    )

    for answer, expected in examples:

        result = (
            validator.validate(
                answer=answer,
                sources=sources,
            )
        )

        print()

        print(
            f"Answer   : {answer}"
        )

        print(
            f"Valid    : {result['valid']}"
        )

        print(
            f"Expected : {expected}"
        )

        print(
            f"Reason   : {result['reason']}"
        )


if __name__ == "__main__":
    main()