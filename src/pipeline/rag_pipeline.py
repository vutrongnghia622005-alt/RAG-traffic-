from pathlib import Path
import sys
import time


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IMPORTS
# ============================================================

from src.retrieval.routed_retriever import (
    RoutedHybridRetriever,
)

from src.retrieval.reranker import (
    BGEReranker,
)

from src.guardrail.evidence_guardrail import (
    EvidenceGuardrail,
    build_abstention_message,
)

from src.generation.context_builder import (
    build_labeled_context,
)

from src.generation.qwen_generator import (
    QwenGenerator,
)

from src.generation.citation_validator import (
    CitationValidator,
    build_citation_retry_feedback,
)


# ============================================================
# CONFIG
# ============================================================

FINAL_TOP_K = 5

# Generation lần đầu + 1 lần sửa citation.
MAX_GENERATION_ATTEMPTS = 2

# Mac Intel chạy Qwen3-8B CPU khá chậm.
GENERATOR_TIMEOUT_SECONDS = 600


# ============================================================
# ADVANCED RAG PIPELINE
# ============================================================

class AdvancedRAGPipeline:

    def __init__(
        self,
        enable_generation=True,
    ):

        print(
            "=" * 90
        )
        print(
            "ADVANCED RAG PIPELINE INITIALIZATION"
        )
        print(
            "=" * 90
        )

        # ----------------------------------------------------
        # C4 ROUTED HYBRID RETRIEVER
        # ----------------------------------------------------

        print()
        print(
            "[1/5] Loading routed hybrid retriever..."
        )

        self.retriever = (
            RoutedHybridRetriever()
        )

        # ----------------------------------------------------
        # C3 RERANKER
        # ----------------------------------------------------

        print()
        print(
            "[2/5] Loading BGE reranker..."
        )

        self.reranker = (
            BGEReranker()
        )

        # ----------------------------------------------------
        # C5 GUARDRAIL
        # ----------------------------------------------------

        print()
        print(
            "[3/5] Loading evidence guardrail..."
        )

        self.guardrail = (
            EvidenceGuardrail()
        )

        # ----------------------------------------------------
        # CITATION VALIDATOR
        # ----------------------------------------------------

        print()
        print(
            "[4/5] Loading citation validator..."
        )

        self.citation_validator = (
            CitationValidator()
        )

        # ----------------------------------------------------
        # QWEN GENERATOR
        # ----------------------------------------------------

        print()
        print(
            "[5/5] Generation configuration..."
        )

        self.enable_generation = (
            enable_generation
        )

        self.generator = None

        if self.enable_generation:

            self.generator = (
                QwenGenerator(
                    timeout=(
                        GENERATOR_TIMEOUT_SECONDS
                    )
                )
            )

            print(
                "Qwen generation : ENABLED"
            )

            print(
                f"Qwen timeout    : "
                f"{GENERATOR_TIMEOUT_SECONDS}s"
            )

        else:

            print(
                "Qwen generation : DISABLED"
            )

        print()
        print(
            "=" * 90
        )

        print(
            "Pipeline ready."
        )

        print(
            "=" * 90
        )


    # ========================================================
    # GET TEXT FROM RETRIEVAL CANDIDATE
    # ========================================================

    @staticmethod
    def _get_candidate_text(
        candidate,
    ):

        metadata = (
            candidate.get(
                "metadata",
                {},
            )
            or {}
        )

        text = (
            metadata.get(
                "text"
            )
        )

        if isinstance(
            text,
            str,
        ) and text.strip():

            return text

        text = (
            candidate.get(
                "text"
            )
        )

        if isinstance(
            text,
            str,
        ):

            return text

        return ""


    # ========================================================
    # RERANK CANDIDATES
    # ========================================================

    def rerank_candidates(
        self,
        question,
        candidates,
    ):

        if not candidates:

            return (
                [],
                0.0,
            )

        texts = [

            self._get_candidate_text(
                candidate
            )

            for candidate
            in candidates
        ]

        start = (
            time.perf_counter()
        )

        reranker_output = (
            self.reranker.score_texts(
                question,
                texts,
            )
        )

        measured_elapsed = (
            time.perf_counter()
            - start
        )

        # ----------------------------------------------------
        # Expected output:
        #
        # scores, probabilities, elapsed
        # ----------------------------------------------------

        if (
            isinstance(
                reranker_output,
                tuple,
            )
            and len(
                reranker_output
            ) == 3
        ):

            (
                scores,
                probabilities,
                model_elapsed,
            ) = reranker_output

        else:

            raise RuntimeError(
                "Unexpected reranker output. "
                "Expected (scores, probabilities, elapsed)."
            )

        if model_elapsed is None:

            rerank_elapsed = (
                measured_elapsed
            )

        else:

            rerank_elapsed = (
                float(
                    model_elapsed
                )
            )

        reranked_results = []

        for (
            candidate,
            score,
            probability,
        ) in zip(
            candidates,
            scores,
            probabilities,
        ):

            item = dict(
                candidate
            )

            item[
                "pre_rerank_rank"
            ] = (
                candidate.get(
                    "rank"
                )
            )

            item[
                "rerank_score"
            ] = float(
                score
            )

            item[
                "rerank_probability"
            ] = float(
                probability
            )

            reranked_results.append(
                item
            )

        # ----------------------------------------------------
        # SORT BY RERANK SCORE DESC
        # ----------------------------------------------------

        reranked_results.sort(
            key=lambda item: (
                -item.get(
                    "rerank_score",
                    float("-inf"),
                ),
                str(
                    item.get(
                        "chunk_id",
                        "",
                    )
                ),
            )
        )

        # ----------------------------------------------------
        # ASSIGN FINAL RANK
        # ----------------------------------------------------

        for rank, item in enumerate(
            reranked_results,
            start=1,
        ):

            item[
                "rank"
            ] = rank

        return (
            reranked_results,
            rerank_elapsed,
        )


    # ========================================================
    # GENERATE GROUNDED ANSWER
    # ========================================================

    def generate_grounded_answer(
        self,
        question,
        context,
        sources,
    ):

        if self.generator is None:

            raise RuntimeError(
                "Qwen generator is not initialized."
            )

        allowed_source_ids = [

            source[
                "source_id"
            ]

            for source
            in sources
        ]

        attempts = []

        correction_feedback = None

        final_answer = None

        final_validation = None

        total_generation_time = (
            0.0
        )

        # ----------------------------------------------------
        # Attempt 1:
        # normal generation
        #
        # Attempt 2:
        # citation repair
        # ----------------------------------------------------

        for attempt_number in range(
            1,
            MAX_GENERATION_ATTEMPTS + 1,
        ):

            generation_start = (
                time.perf_counter()
            )

            answer = (
                self.generator.generate(
                    question=(
                        question
                    ),
                    context=(
                        context
                    ),
                    allowed_source_ids=(
                        allowed_source_ids
                    ),
                    correction_feedback=(
                        correction_feedback
                    ),
                )
            )

            generation_elapsed = (
                time.perf_counter()
                - generation_start
            )

            total_generation_time += (
                generation_elapsed
            )

            # ------------------------------------------------
            # STRUCTURAL CITATION VALIDATION
            # ------------------------------------------------

            validation = (
                self.citation_validator
                .validate(
                    answer=(
                        answer
                    ),
                    sources=(
                        sources
                    ),
                    require_citation=True,
                )
            )

            attempts.append(
                {
                    "attempt":
                        attempt_number,

                    "answer":
                        answer,

                    "validation":
                        validation,

                    "elapsed":
                        generation_elapsed,
                }
            )

            final_answer = (
                answer
            )

            final_validation = (
                validation
            )

            # ------------------------------------------------
            # PASS
            # ------------------------------------------------

            if validation[
                "valid"
            ]:

                break

            # ------------------------------------------------
            # PREPARE CITATION REPAIR PROMPT
            # ------------------------------------------------

            correction_feedback = (
                build_citation_retry_feedback(
                    validation
                )
            )

        return {

            "answer":
                final_answer,

            "citation_validation":
                final_validation,

            "generation_attempts":
                attempts,

            "generation_elapsed":
                total_generation_time,
        }


    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        question,
    ):

        total_start = (
            time.perf_counter()
        )

        question = (
            question.strip()
        )

        if not question:

            raise ValueError(
                "Question is empty."
            )


        # ====================================================
        # STEP 1 - C4 RETRIEVAL
        # ====================================================

        retrieval_start = (
            time.perf_counter()
        )

        retrieval = (
            self.retriever
            .retrieve_candidates(
                question
            )
        )

        measured_retrieval_elapsed = (
            time.perf_counter()
            - retrieval_start
        )

        candidates = (
            retrieval.get(
                "candidates",
                [],
            )
            or []
        )


        # ====================================================
        # STEP 2 - C3 RERANK
        # ====================================================

        (
            reranked_results,
            rerank_elapsed,
        ) = self.rerank_candidates(
            question=(
                question
            ),
            candidates=(
                candidates
            ),
        )

        top_results = (
            reranked_results[
                :FINAL_TOP_K
            ]
        )


        # ====================================================
        # STEP 3 - C5 GUARDRAIL
        # ====================================================

        decision = (
            self.guardrail.decide(
                top_results
            )
        )


        # ====================================================
        # STEP 4A - ABSTAIN
        # ====================================================

        if not decision[
            "supported"
        ]:

            total_elapsed = (
                time.perf_counter()
                - total_start
            )

            return {

                "question":
                    question,

                "status":
                    "abstain",

                "answer":
                    build_abstention_message(
                        decision
                    ),

                "generation_error":
                    None,

                "guardrail":
                    decision,

                "route":
                    retrieval.get(
                        "route"
                    ),

                "context":
                    None,

                "sources":
                    [],

                "top_results":
                    top_results,

                "citation_validation":
                    None,

                "generation_attempts":
                    [],

                "timing": {

                    "retrieval":
                        retrieval.get(
                            "timing",
                            measured_retrieval_elapsed,
                        ),

                    "reranker":
                        rerank_elapsed,

                    "generation":
                        0.0,

                    "total":
                        total_elapsed,
                },
            }


        # ====================================================
        # STEP 4B - BUILD LABELED CONTEXT
        # ====================================================

        (
            context,
            sources,
        ) = build_labeled_context(
            top_results,
            max_sources=(
                FINAL_TOP_K
            ),
        )


        # ====================================================
        # STEP 5A - RETRIEVAL ONLY MODE
        # ====================================================

        if not self.enable_generation:

            total_elapsed = (
                time.perf_counter()
                - total_start
            )

            return {

                "question":
                    question,

                "status":
                    "supported",

                "answer":
                    None,

                "generation_error":
                    None,

                "guardrail":
                    decision,

                "route":
                    retrieval.get(
                        "route"
                    ),

                "context":
                    context,

                "sources":
                    sources,

                "top_results":
                    top_results,

                "citation_validation":
                    None,

                "generation_attempts":
                    [],

                "timing": {

                    "retrieval":
                        retrieval.get(
                            "timing",
                            measured_retrieval_elapsed,
                        ),

                    "reranker":
                        rerank_elapsed,

                    "generation":
                        0.0,

                    "total":
                        total_elapsed,
                },
            }


        # ====================================================
        # STEP 5B - CHECK GENERATOR
        # ====================================================

        if (
            self.generator is None
            or
            not self.generator.is_available()
        ):

            total_elapsed = (
                time.perf_counter()
                - total_start
            )

            return {

                "question":
                    question,

                "status":
                    "generator_unavailable",

                "answer":
                    None,

                "generation_error":
                    (
                        "Qwen3 server is unavailable."
                    ),

                "guardrail":
                    decision,

                "route":
                    retrieval.get(
                        "route"
                    ),

                "context":
                    context,

                "sources":
                    sources,

                "top_results":
                    top_results,

                "citation_validation":
                    None,

                "generation_attempts":
                    [],

                "timing": {

                    "retrieval":
                        retrieval.get(
                            "timing",
                            measured_retrieval_elapsed,
                        ),

                    "reranker":
                        rerank_elapsed,

                    "generation":
                        0.0,

                    "total":
                        total_elapsed,
                },
            }


        # ====================================================
        # STEP 6 - GENERATION
        # ====================================================

        generation_stage_start = (
            time.perf_counter()
        )

        try:

            generation_result = (
                self.generate_grounded_answer(
                    question=(
                        question
                    ),
                    context=(
                        context
                    ),
                    sources=(
                        sources
                    ),
                )
            )

        except Exception as exc:

            generation_elapsed = (
                time.perf_counter()
                - generation_stage_start
            )

            total_elapsed = (
                time.perf_counter()
                - total_start
            )

            return {

                "question":
                    question,

                "status":
                    "generation_error",

                "answer":
                    None,

                "generation_error":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),

                # IMPORTANT:
                # Preserve Guardrail result.
                "guardrail":
                    decision,

                "route":
                    retrieval.get(
                        "route"
                    ),

                "context":
                    context,

                "sources":
                    sources,

                "top_results":
                    top_results,

                "citation_validation":
                    None,

                "generation_attempts":
                    [],

                "timing": {

                    "retrieval":
                        retrieval.get(
                            "timing",
                            measured_retrieval_elapsed,
                        ),

                    "reranker":
                        rerank_elapsed,

                    "generation":
                        generation_elapsed,

                    "total":
                        total_elapsed,
                },
            }


        # ====================================================
        # STEP 7 - CITATION DECISION
        # ====================================================

        citation_validation = (
            generation_result.get(
                "citation_validation"
            )
        )

        if (
            citation_validation
            and
            citation_validation.get(
                "valid"
            )
        ):

            status = (
                "answered"
            )

            answer = (
                generation_result[
                    "answer"
                ]
            )

            generation_error = (
                None
            )

        else:

            status = (
                "citation_validation_failed"
            )

            answer = (
                "Hệ thống đã tìm thấy bằng chứng phù hợp, "
                "nhưng câu trả lời sinh ra không vượt qua "
                "kiểm tra trích dẫn nguồn nên đã bị từ chối."
            )

            generation_error = (
                None
            )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        total_elapsed = (
            time.perf_counter()
            - total_start
        )

        return {

            "question":
                question,

            "status":
                status,

            "answer":
                answer,

            "generation_error":
                generation_error,

            "guardrail":
                decision,

            "route":
                retrieval.get(
                    "route"
                ),

            "context":
                context,

            "sources":
                sources,

            "top_results":
                top_results,

            "citation_validation":
                citation_validation,

            "generation_attempts":
                generation_result.get(
                    "generation_attempts",
                    [],
                ),

            "timing": {

                "retrieval":
                    retrieval.get(
                        "timing",
                        measured_retrieval_elapsed,
                    ),

                "reranker":
                    rerank_elapsed,

                "generation":
                    generation_result.get(
                        "generation_elapsed",
                        0.0,
                    ),

                "total":
                    total_elapsed,
            },
        }


# ============================================================
# DISPLAY HELPERS
# ============================================================

def _safe_value(
    source,
    key,
    default="-",
):

    value = (
        source.get(
            key,
            default,
        )
    )

    if value is None:

        return default

    return value


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    result,
):

    print()

    print(
        "=" * 90
    )

    print(
        "ADVANCED RAG RESULT"
    )

    print(
        "=" * 90
    )

    print()

    print(
        f"Question : "
        f"{result.get('question')}"
    )

    print(
        f"Status   : "
        f"{result.get('status')}"
    )


    # ========================================================
    # ANSWER
    # ========================================================

    print()

    print(
        "ANSWER"
    )

    print(
        "-" * 90
    )

    answer = (
        result.get(
            "answer"
        )
    )

    if answer is None:

        print(
            "None"
        )

    else:

        print(
            answer
        )


    # ========================================================
    # GENERATION ERROR
    # ========================================================

    generation_error = (
        result.get(
            "generation_error"
        )
    )

    if generation_error:

        print()

        print(
            "GENERATION ERROR"
        )

        print(
            "-" * 90
        )

        print(
            generation_error
        )


    # ========================================================
    # GUARDRAIL
    # ========================================================

    guardrail = (
        result.get(
            "guardrail",
            {},
        )
        or {}
    )

    print()

    print(
        "GUARDRAIL"
    )

    print(
        "-" * 90
    )

    print(
        f"Supported : "
        f"{guardrail.get('supported')}"
    )

    top1_score = (
        guardrail.get(
            "top1_score"
        )
    )

    threshold = (
        guardrail.get(
            "threshold"
        )
    )

    if top1_score is not None:

        print(
            f"Score     : "
            f"{float(top1_score):+.4f}"
        )

    else:

        print(
            "Score     : -"
        )

    if threshold is not None:

        print(
            f"Threshold : "
            f"{float(threshold):+.4f}"
        )

    else:

        print(
            "Threshold : -"
        )


    # ========================================================
    # CITATION
    # ========================================================

    citation = (
        result.get(
            "citation_validation"
        )
    )

    if citation is not None:

        print()

        print(
            "CITATION VALIDATION"
        )

        print(
            "-" * 90
        )

        print(
            f"Valid    : "
            f"{citation.get('valid')}"
        )

        print(
            f"Reason   : "
            f"{citation.get('reason')}"
        )

        print(
            f"Cited    : "
            f"{citation.get('cited_source_ids')}"
        )

        print(
            f"Allowed  : "
            f"{citation.get('allowed_source_ids')}"
        )

        invalid_ids = (
            citation.get(
                "invalid_source_ids",
                [],
            )
        )

        if invalid_ids:

            print(
                f"Invalid  : "
                f"{invalid_ids}"
            )


    # ========================================================
    # GENERATION ATTEMPTS
    # ========================================================

    attempts = (
        result.get(
            "generation_attempts",
            [],
        )
        or []
    )

    if attempts:

        print()

        print(
            "GENERATION ATTEMPTS"
        )

        print(
            "-" * 90
        )

        for attempt in attempts:

            validation = (
                attempt.get(
                    "validation",
                    {},
                )
                or {}
            )

            print(
                f"Attempt "
                f"{attempt.get('attempt')} "
                f"| Valid="
                f"{validation.get('valid')} "
                f"| Reason="
                f"{validation.get('reason')} "
                f"| Time="
                f"{float(attempt.get('elapsed', 0.0)):.2f}s"
            )


    # ========================================================
    # SOURCES
    # ========================================================

    sources = (
        result.get(
            "sources",
            [],
        )
        or []
    )

    if sources:

        print()

        print(
            "SOURCES"
        )

        print(
            "-" * 90
        )

        for source in sources:

            source_id = (
                _safe_value(
                    source,
                    "source_id",
                )
            )

            article = (
                _safe_value(
                    source,
                    "article_number",
                )
            )

            clause = (
                _safe_value(
                    source,
                    "clause",
                )
            )

            page_start = (
                _safe_value(
                    source,
                    "page_start",
                )
            )

            page_end = (
                _safe_value(
                    source,
                    "page_end",
                )
            )

            rerank_score = (
                source.get(
                    "rerank_score"
                )
            )

            if rerank_score is None:

                score_text = (
                    "-"
                )

            else:

                score_text = (
                    f"{float(rerank_score):+.4f}"
                )

            print(
                f"[{source_id}] "
                f"Điều {article} "
                f"| Khoản {clause} "
                f"| Trang "
                f"{page_start}-{page_end} "
                f"| Score "
                f"{score_text}"
            )


    # ========================================================
    # TIMING
    # ========================================================

    timing = (
        result.get(
            "timing",
            {},
        )
        or {}
    )

    print()

    print(
        "TIMING"
    )

    print(
        "-" * 90
    )

    retrieval_timing = (
        timing.get(
            "retrieval"
        )
    )

    if isinstance(
        retrieval_timing,
        (int, float),
    ):

        print(
            f"Retrieval  : "
            f"{float(retrieval_timing):.4f}s"
        )

    else:

        print(
            f"Retrieval  : "
            f"{retrieval_timing}"
        )

    print(
        f"Reranker   : "
        f"{float(timing.get('reranker', 0.0)):.4f}s"
    )

    print(
        f"Generation : "
        f"{float(timing.get('generation', 0.0)):.4f}s"
    )

    print(
        f"Total      : "
        f"{float(timing.get('total', 0.0)):.4f}s"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    pipeline = (
        AdvancedRAGPipeline(
            enable_generation=True
        )
    )

    print()
    print(
        "Nhập câu hỏi."
    )

    print(
        "Gõ 'exit' để thoát."
    )

    while True:

        print()

        question = input(
            "Câu hỏi > "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:

            print(
                "Đã thoát."
            )

            break

        if not question:

            continue

        try:

            result = (
                pipeline.ask(
                    question
                )
            )

            display_result(
                result
            )

        except KeyboardInterrupt:

            print()
            print(
                "Đã dừng câu hỏi hiện tại."
            )

        except Exception as exc:

            print()
            print(
                "=" * 90
            )

            print(
                "PIPELINE ERROR"
            )

            print(
                "=" * 90
            )

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            raise


if __name__ == "__main__":
    main()