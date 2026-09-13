from dataclasses import dataclass, asdict
from typing import Optional


# ============================================================
# CONFIG
# ============================================================

DEFAULT_RERANK_THRESHOLD = 0.7690


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass
class GuardrailDecision:

    supported: bool

    action: str

    reason: str

    top1_score: float

    threshold: float

    margin_to_threshold: float

    top_chunk_id: Optional[str]

    top_article: Optional[object]

    top_clause: Optional[object]


# ============================================================
# EVIDENCE GUARDRAIL
# ============================================================

class EvidenceGuardrail:
    """
    C5 evidence sufficiency guardrail.

    Threshold được calibration trên DEV set:

        threshold = +0.7690

    Không dùng threshold được tối ưu trên test set.

    Decision:

        rerank_score >= threshold
            -> SUPPORTED

        rerank_score < threshold
            -> ABSTAIN
    """

    def __init__(
        self,
        threshold=DEFAULT_RERANK_THRESHOLD,
    ):

        self.threshold = float(
            threshold
        )


    # ========================================================
    # DECIDE FROM SCORE
    # ========================================================

    def decide_score(
        self,
        top1_score,
        top_chunk_id=None,
        top_article=None,
        top_clause=None,
    ):

        score = float(
            top1_score
        )

        supported = (
            score
            >= self.threshold
        )

        if supported:

            action = "answer"

            reason = (
                "sufficient_reranker_evidence"
            )

        else:

            action = "abstain"

            reason = (
                "insufficient_reranker_evidence"
            )

        decision = GuardrailDecision(

            supported=(
                supported
            ),

            action=(
                action
            ),

            reason=(
                reason
            ),

            top1_score=(
                score
            ),

            threshold=(
                self.threshold
            ),

            margin_to_threshold=(
                score
                - self.threshold
            ),

            top_chunk_id=(
                top_chunk_id
            ),

            top_article=(
                top_article
            ),

            top_clause=(
                top_clause
            ),
        )

        return asdict(
            decision
        )


    # ========================================================
    # DECIDE FROM RERANKED RESULTS
    # ========================================================

    def decide(
        self,
        reranked_results,
    ):
        """
        reranked_results phải được sắp xếp score giảm dần.
        """

        if not reranked_results:

            return asdict(
                GuardrailDecision(

                    supported=False,

                    action="abstain",

                    reason=(
                        "no_retrieval_results"
                    ),

                    top1_score=(
                        float("-inf")
                    ),

                    threshold=(
                        self.threshold
                    ),

                    margin_to_threshold=(
                        float("-inf")
                    ),

                    top_chunk_id=None,

                    top_article=None,

                    top_clause=None,
                )
            )

        top1 = (
            reranked_results[0]
        )

        metadata = (
            top1.get(
                "metadata",
                {}
            )
        )

        return self.decide_score(

            top1_score=(
                top1[
                    "rerank_score"
                ]
            ),

            top_chunk_id=(
                top1.get(
                    "chunk_id"
                )
            ),

            top_article=(
                metadata.get(
                    "article_number"
                )
            ),

            top_clause=(
                metadata.get(
                    "clause"
                )
            ),
        )


# ============================================================
# USER-FACING ABSTENTION
# ============================================================

def build_abstention_message(
    decision,
):

    if decision[
        "supported"
    ]:

        return None

    return (
        "Tôi chưa tìm thấy đủ bằng chứng trong tài liệu "
        "để trả lời câu hỏi này một cách đáng tin cậy. "
        "Bạn có thể đặt lại câu hỏi hoặc cung cấp thêm "
        "tài liệu có liên quan."
    )


# ============================================================
# SIMPLE TEST
# ============================================================

def main():

    guardrail = (
        EvidenceGuardrail()
    )

    examples = [
        7.5,
        4.9,
        0.769,
        0.5,
        -1.07,
        -5.0,
    ]

    print(
        "=" * 72
    )

    print(
        "C5 EVIDENCE GUARDRAIL TEST"
    )

    print(
        "=" * 72
    )

    print(
        f"Threshold: "
        f"{guardrail.threshold:+.4f}"
    )

    print()

    for score in examples:

        result = (
            guardrail.decide_score(
                score
            )
        )

        print(
            f"score={score:+.4f} "
            f"-> "
            f"{result['action'].upper()} "
            f"| margin="
            f"{result['margin_to_threshold']:+.4f}"
        )


if __name__ == "__main__":
    main()