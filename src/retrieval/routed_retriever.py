from pathlib import Path
import sys
import time
import unicodedata
import re

import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================
#
# File:
#   advanced-rag/src/retrieval/routed_retriever.py
#
# parents[0] = src/retrieval
# parents[1] = src
# parents[2] = advanced-rag
#
# Thêm project root vào sys.path để chạy được cả:
#
#   python src/retrieval/routed_retriever.py
#
# và:
#
#   python -m src.retrieval.routed_retriever
#
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    RRF_K,
)

from src.routing.query_router import (
    QueryRouter,
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_RETRIEVAL_TOP_K = 20

DEFAULT_CANDIDATE_TOP_K = 20

MULTI_EVIDENCE_CANDIDATE_TOP_K = 30

FINAL_TOP_K = 5


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_lower(text):
    """
    Normalize Unicode + lowercase + normalize whitespace.
    """

    text = unicodedata.normalize(
        "NFC",
        str(text)
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().lower()


# ============================================================
# C4 ROUTED HYBRID RETRIEVER
# ============================================================

class RoutedHybridRetriever:
    """
    C4 retrieval pipeline.

    C3:
        Dense Top-20
            +
        BM25 Top-20
            ↓
        RRF Top-20
            ↓
        Reranker

    C4 bổ sung:
        Query Router
            ↓
        Language detection
        Intent detection
        Multi-evidence detection
        Explicit article/clause detection
        Entity detection
            ↓
        Routed candidate generation

    Routing rules:

    1. Explicit Article
       -> hard structural filter

    2. Explicit entity
       -> soft routing
       -> không loại semantic candidates

    3. Multi-evidence query
       -> candidate budget 20 -> 30

    4. Normal query
       -> giữ pipeline C3
    """

    def __init__(self):

        print("=" * 78)
        print(
            "C4 ROUTED HYBRID RETRIEVER INITIALIZATION"
        )
        print("=" * 78)

        # ====================================================
        # BASE RETRIEVER
        # ====================================================

        self.base = HybridRetriever()

        # ====================================================
        # QUERY ROUTER
        # ====================================================

        self.router = QueryRouter(
            default_candidate_top_k=(
                DEFAULT_CANDIDATE_TOP_K
            ),
            multi_evidence_candidate_top_k=(
                MULTI_EVIDENCE_CANDIDATE_TOP_K
            ),
            final_top_k=(
                FINAL_TOP_K
            ),
        )

        print()
        print(
            "Query router       : PASS"
        )

        print(
            "Default candidate K: "
            f"{DEFAULT_CANDIDATE_TOP_K}"
        )

        print(
            "Multi-evidence K   : "
            f"{MULTI_EVIDENCE_CANDIDATE_TOP_K}"
        )

        print(
            "Final Top-K        : "
            f"{FINAL_TOP_K}"
        )

        print()
        print(
            "C4 routed retriever ready."
        )


    # ========================================================
    # METADATA PROPERTY
    # ========================================================

    @property
    def metadata(self):

        return self.base.metadata


    # ========================================================
    # EMBEDDING PROPERTY
    # ========================================================

    @property
    def embeddings(self):

        return self.base.embeddings


    # ========================================================
    # QUERY ENCODING
    # ========================================================

    def encode_query(
        self,
        question
    ):

        return self.base.encode_query(
            question
        )


    # ========================================================
    # HARD FILTER INDICES
    # ========================================================

    def get_hard_filter_indices(
        self,
        route
    ):
        """
        Trả về vector indices được phép search.

        HARD filter chỉ xảy ra khi user nêu rõ Điều.

        Ví dụ:

            "Theo Điều 87..."

        Nếu user đồng thời nêu Khoản:

            "Điều 87 khoản 2"

        thì tiếp tục filter xuống Khoản 2.
        """

        article_numbers = set(
            int(value)
            for value
            in route[
                "article_numbers"
            ]
        )

        clause_numbers = set(
            str(value)
            .lower()
            .strip()
            for value
            in route[
                "clause_numbers"
            ]
        )

        allowed_indices = []

        for index, item in enumerate(
            self.metadata
        ):

            # ------------------------------------------------
            # ARTICLE
            # ------------------------------------------------

            article = item.get(
                "article_number"
            )

            try:

                article = int(
                    article
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if article not in article_numbers:
                continue

            # ------------------------------------------------
            # Nếu không nêu Khoản
            # -> giữ toàn bộ Điều
            # ------------------------------------------------

            if not clause_numbers:

                allowed_indices.append(
                    index
                )

                continue

            # ------------------------------------------------
            # CLAUSE
            # ------------------------------------------------

            clause = item.get(
                "clause"
            )

            if clause is None:
                continue

            clause = (
                str(clause)
                .lower()
                .strip()
            )

            if clause in clause_numbers:

                allowed_indices.append(
                    index
                )

        return allowed_indices


    # ========================================================
    # DENSE SEARCH ON FILTERED INDICES
    # ========================================================

    def dense_search_filtered(
        self,
        query_vector,
        allowed_indices,
        top_k
    ):
        """
        Dense search chỉ trên subset metadata/vector.
        """

        start = (
            time.perf_counter()
        )

        if not allowed_indices:

            elapsed = (
                time.perf_counter()
                - start
            )

            return (
                [],
                elapsed,
            )

        indices = np.asarray(
            allowed_indices,
            dtype=np.int64
        )

        # ----------------------------------------------------
        # COSINE / DOT PRODUCT
        #
        # Embeddings và query đều normalized.
        # ----------------------------------------------------

        scores = (
            self.embeddings[
                indices
            ]
            @ query_vector
        )

        order = (
            np.argsort(
                scores
            )[::-1]
        )

        order = order[
            :top_k
        ]

        results = []

        for rank, local_position in enumerate(
            order,
            start=1
        ):

            local_position = int(
                local_position
            )

            vector_index = int(
                indices[
                    local_position
                ]
            )

            score = float(
                scores[
                    local_position
                ]
            )

            results.append(
                {
                    "rank": (
                        rank
                    ),

                    "vector_index": (
                        vector_index
                    ),

                    "chunk_id": (
                        self.metadata[
                            vector_index
                        ][
                            "chunk_id"
                        ]
                    ),

                    "score": (
                        score
                    ),

                    "metadata": (
                        self.metadata[
                            vector_index
                        ]
                    ),
                }
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        return (
            results,
            elapsed,
        )


    # ========================================================
    # BM25 SEARCH ON FILTERED INDICES
    # ========================================================

    def bm25_search_filtered(
        self,
        question,
        allowed_indices,
        top_k
    ):
        """
        BM25 filter implementation.

        Dataset chỉ có 521 chunks nên cách đơn giản và
        reproducible là:

            BM25 toàn index
                ↓
            filter theo allowed vector indices
                ↓
            re-rank lại
        """

        start = (
            time.perf_counter()
        )

        if not allowed_indices:

            elapsed = (
                time.perf_counter()
                - start
            )

            return (
                [],
                elapsed,
            )

        allowed_set = set(
            allowed_indices
        )

        # ----------------------------------------------------
        # Lấy BM25 ranking toàn index.
        # ----------------------------------------------------

        (
            global_results,
            _
        ) = self.base.bm25_search(
            question,
            top_k=len(
                self.metadata
            ),
        )

        # ----------------------------------------------------
        # Structural filter
        # ----------------------------------------------------

        filtered = [
            dict(item)
            for item
            in global_results
            if (
                item[
                    "vector_index"
                ]
                in allowed_set
            )
        ]

        filtered = filtered[
            :top_k
        ]

        # ----------------------------------------------------
        # Reset rank sau filtering
        # ----------------------------------------------------

        for rank, item in enumerate(
            filtered,
            start=1
        ):

            item[
                "rank"
            ] = rank

        elapsed = (
            time.perf_counter()
            - start
        )

        return (
            filtered,
            elapsed,
        )


    # ========================================================
    # ENTITY MATCHING
    # ========================================================

    def get_entity_matching_indices(
        self,
        entities
    ):
        """
        Tìm các chunks chứa entity được user nêu.

        Ví dụ:

            Bộ Công an

        Đây không phải hard filter.
        Chỉ dùng cho soft routing.
        """

        if not entities:

            return []

        normalized_entities = [
            normalize_lower(
                entity
            )
            for entity in entities
        ]

        matching_indices = []

        for index, item in enumerate(
            self.metadata
        ):

            text = normalize_lower(
                item.get(
                    "text",
                    ""
                )
            )

            if any(
                entity in text
                for entity
                in normalized_entities
            ):

                matching_indices.append(
                    index
                )

        return matching_indices


    # ========================================================
    # SOFT ENTITY ROUTING
    # ========================================================

    def apply_soft_entity_routing(
        self,
        candidates,
        route,
        query_vector,
        target_k
    ):
        """
        Soft entity routing.

        Nguyên tắc:

        - Không hard-filter theo entity.
        - Không loại bỏ semantic retrieval.
        - Nếu Top-N chưa có chunk nào chứa entity user nêu,
          thêm candidate entity-related tốt nhất.
        - Candidate cuối cùng vẫn phải qua reranker.

        Ví dụ:

            "Bộ Công an có trách nhiệm gì?"

        Router nhận:
            entities = ["Bộ Công an"]

        Nếu candidate pool chưa có Bộ Công an:
            tìm chunk chứa Bộ Công an
            có dense similarity cao nhất
            và đảm bảo nó có mặt trong candidate pool.
        """

        entities = route[
            "entities"
        ]

        if not entities:

            return candidates

        output = [
            dict(item)
            for item in candidates
        ]

        current_ids = {
            item[
                "chunk_id"
            ]
            for item in output
        }

        # ====================================================
        # PROCESS EACH ENTITY
        # ====================================================

        for entity in entities:

            entity_normalized = (
                normalize_lower(
                    entity
                )
            )

            # ------------------------------------------------
            # Check entity already exists
            # ------------------------------------------------

            already_present = False

            for candidate in output:

                candidate_text = (
                    normalize_lower(
                        candidate[
                            "metadata"
                        ].get(
                            "text",
                            ""
                        )
                    )
                )

                if (
                    entity_normalized
                    in candidate_text
                ):

                    already_present = True

                    break

            if already_present:
                continue

            # ------------------------------------------------
            # Find entity chunks
            # ------------------------------------------------

            matching_indices = (
                self.get_entity_matching_indices(
                    [entity]
                )
            )

            if not matching_indices:
                continue

            # ------------------------------------------------
            # Select entity chunk with highest dense score
            # ------------------------------------------------

            matching_array = np.asarray(
                matching_indices,
                dtype=np.int64
            )

            candidate_embeddings = (
                self.embeddings[
                    matching_array
                ]
            )

            scores = (
                candidate_embeddings
                @ query_vector
            )

            best_local_index = int(
                np.argmax(
                    scores
                )
            )

            best_vector_index = int(
                matching_array[
                    best_local_index
                ]
            )

            chunk_id = (
                self.metadata[
                    best_vector_index
                ][
                    "chunk_id"
                ]
            )

            if chunk_id in current_ids:
                continue

            # ------------------------------------------------
            # Add soft-routed candidate
            # ------------------------------------------------

            added_candidate = {

                "rank": None,

                "vector_index": (
                    best_vector_index
                ),

                "chunk_id": (
                    chunk_id
                ),

                "metadata": (
                    self.metadata[
                        best_vector_index
                    ]
                ),

                "rrf_score": 0.0,

                "dense_rank": None,

                "dense_score": float(
                    scores[
                        best_local_index
                    ]
                ),

                "bm25_rank": None,

                "bm25_score": None,

                "routing_added": True,

                "routing_reason": (
                    f"soft_entity:{entity}"
                ),
            }

            output.append(
                added_candidate
            )

            current_ids.add(
                chunk_id
            )

        # ====================================================
        # KEEP FIXED CANDIDATE BUDGET
        # ====================================================

        while (
            len(output)
            > target_k
        ):

            removed = False

            # ------------------------------------------------
            # Remove lowest ordinary candidate first.
            # ------------------------------------------------

            for index in range(
                len(output) - 1,
                -1,
                -1
            ):

                if not output[
                    index
                ].get(
                    "routing_added",
                    False
                ):

                    output.pop(
                        index
                    )

                    removed = True

                    break

            # ------------------------------------------------
            # Defensive fallback
            # ------------------------------------------------

            if not removed:

                output.pop()

        # ====================================================
        # RESET CANDIDATE RANK
        # ====================================================

        for rank, item in enumerate(
            output,
            start=1
        ):

            item[
                "rank"
            ] = rank

        return output


    # ========================================================
    # RETRIEVE C4 CANDIDATES
    # ========================================================

    def retrieve_candidates(
        self,
        question,
        query_vector=None,
    ):
        """
        Main C4 retrieval function.

        Steps:

            Question
               ↓
            Router
               ↓
            Determine:
                - hard filter
                - soft entity routing
                - candidate budget
               ↓
            Dense
               +
            BM25
               ↓
            RRF
               ↓
            Routed candidate pool

        Reranker được chạy ở bước tiếp theo của pipeline.
        """

        total_start = (
            time.perf_counter()
        )

        # ====================================================
        # ROUTER
        # ====================================================

        routing_start = (
            time.perf_counter()
        )

        route = (
            self.router.route(
                question
            )
        )

        routing_elapsed = (
            time.perf_counter()
            - routing_start
        )

        # ====================================================
        # QUERY EMBEDDING
        # ====================================================

        embedding_elapsed = 0.0

        if query_vector is None:

            embedding_start = (
                time.perf_counter()
            )

            query_vector = (
                self.encode_query(
                    question
                )
            )

            embedding_elapsed = (
                time.perf_counter()
                - embedding_start
            )

        # ====================================================
        # ROUTED CANDIDATE BUDGET
        # ====================================================

        candidate_top_k = int(
            route[
                "candidate_top_k"
            ]
        )

        retrieval_top_k = max(
            DEFAULT_RETRIEVAL_TOP_K,
            candidate_top_k,
        )

        # ====================================================
        # HARD FILTER
        # ====================================================

        allowed_indices = None

        if (
            route[
                "filter_strategy"
            ]
            == "hard"
        ):

            allowed_indices = (
                self.get_hard_filter_indices(
                    route
                )
            )

            (
                dense_results,
                dense_elapsed,
            ) = self.dense_search_filtered(
                query_vector=(
                    query_vector
                ),
                allowed_indices=(
                    allowed_indices
                ),
                top_k=(
                    retrieval_top_k
                ),
            )

            (
                bm25_results,
                bm25_elapsed,
            ) = self.bm25_search_filtered(
                question=(
                    question
                ),
                allowed_indices=(
                    allowed_indices
                ),
                top_k=(
                    retrieval_top_k
                ),
            )

        # ====================================================
        # NORMAL / SOFT SEARCH
        # ====================================================

        else:

            (
                dense_results,
                dense_elapsed,
            ) = self.base.dense_search(
                query_vector,
                retrieval_top_k,
            )

            (
                bm25_results,
                bm25_elapsed,
            ) = self.base.bm25_search(
                question,
                retrieval_top_k,
            )

        # ====================================================
        # RRF
        # ====================================================

        fusion_start = (
            time.perf_counter()
        )

        candidates = (
            self.base
            .reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                final_top_k=(
                    candidate_top_k
                ),
                rrf_k=(
                    RRF_K
                ),
            )
        )

        fusion_elapsed = (
            time.perf_counter()
            - fusion_start
        )

        # ====================================================
        # SOFT ENTITY ROUTING
        # ====================================================

        soft_routing_start = (
            time.perf_counter()
        )

        if (
            route[
                "filter_strategy"
            ]
            == "soft"
        ):

            candidates = (
                self.apply_soft_entity_routing(
                    candidates=(
                        candidates
                    ),

                    route=(
                        route
                    ),

                    query_vector=(
                        query_vector
                    ),

                    target_k=(
                        candidate_top_k
                    ),
                )
            )

        soft_routing_elapsed = (
            time.perf_counter()
            - soft_routing_start
        )

        # ====================================================
        # TOTAL TIMING
        # ====================================================

        total_elapsed = (
            time.perf_counter()
            - total_start
        )

        return {

            "question": (
                question
            ),

            "route": (
                route
            ),

            "allowed_indices": (
                allowed_indices
            ),

            "dense_results": (
                dense_results
            ),

            "bm25_results": (
                bm25_results
            ),

            "candidates": (
                candidates
            ),

            "timing": {

                "routing": (
                    routing_elapsed
                ),

                "query_embedding": (
                    embedding_elapsed
                ),

                "dense": (
                    dense_elapsed
                ),

                "bm25": (
                    bm25_elapsed
                ),

                "rrf": (
                    fusion_elapsed
                ),

                "soft_routing": (
                    soft_routing_elapsed
                ),

                "total": (
                    total_elapsed
                ),
            },
        }


# ============================================================
# DISPLAY ROUTE
# ============================================================

def display_route(
    route
):

    print()
    print(
        "ROUTING"
    )

    print(
        "-" * 90
    )

    print(
        f"Language        : "
        f"{route['language']}"
    )

    print(
        f"Intent          : "
        f"{route['intent']}"
    )

    print(
        f"Multi evidence  : "
        f"{route['multi_evidence']}"
    )

    print(
        f"Articles        : "
        f"{route['article_numbers']}"
    )

    print(
        f"Clauses         : "
        f"{route['clause_numbers']}"
    )

    print(
        f"Entities        : "
        f"{route['entities']}"
    )

    print(
        f"Filter strategy : "
        f"{route['filter_strategy']}"
    )

    print(
        f"Filter reason   : "
        f"{route['filter_reason']}"
    )

    print(
        f"Candidate K     : "
        f"{route['candidate_top_k']}"
    )


# ============================================================
# DISPLAY TIMING
# ============================================================

def display_timing(
    timing
):

    print()
    print(
        "TIMING"
    )

    print(
        "-" * 90
    )

    print(
        f"Routing          : "
        f"{timing['routing']:.6f}s"
    )

    print(
        f"Query embedding  : "
        f"{timing['query_embedding']:.4f}s"
    )

    print(
        f"Dense            : "
        f"{timing['dense']:.6f}s"
    )

    print(
        f"BM25             : "
        f"{timing['bm25']:.6f}s"
    )

    print(
        f"RRF              : "
        f"{timing['rrf']:.6f}s"
    )

    print(
        f"Soft routing     : "
        f"{timing['soft_routing']:.6f}s"
    )

    print(
        f"Total            : "
        f"{timing['total']:.4f}s"
    )


# ============================================================
# DISPLAY CANDIDATES
# ============================================================

def display_candidates(
    candidates
):

    print()
    print(
        "CANDIDATES"
    )

    print(
        "=" * 90
    )

    for item in candidates:

        metadata = (
            item[
                "metadata"
            ]
        )

        print()
        print(
            "-" * 90
        )

        print(
            f"Candidate Rank : "
            f"{item.get('rank')}"
        )

        print(
            f"Chunk ID       : "
            f"{item.get('chunk_id')}"
        )

        print(
            f"Article        : "
            f"{metadata.get('article_number')}"
        )

        print(
            f"Clause         : "
            f"{metadata.get('clause')}"
        )

        print(
            f"Pages          : "
            f"{metadata.get('page_start')} "
            f"-> "
            f"{metadata.get('page_end')}"
        )

        print(
            f"RRF score      : "
            f"{item.get('rrf_score')}"
        )

        print(
            f"Dense rank     : "
            f"{item.get('dense_rank')}"
        )

        print(
            f"BM25 rank      : "
            f"{item.get('bm25_rank')}"
        )

        if item.get(
            "routing_added",
            False
        ):

            print(
                f"Routing added  : "
                f"YES"
            )

            print(
                f"Routing reason : "
                f"{item.get('routing_reason')}"
            )


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    result
):

    print()
    print(
        "=" * 90
    )

    print(
        "C4 ROUTED RETRIEVAL RESULT"
    )

    print(
        "=" * 90
    )

    print()
    print(
        "QUESTION:"
    )

    print(
        result[
            "question"
        ]
    )

    display_route(
        result[
            "route"
        ]
    )

    display_timing(
        result[
            "timing"
        ]
    )

    print()
    print(
        f"Candidate count : "
        f"{len(result['candidates'])}"
    )

    if (
        result[
            "allowed_indices"
        ]
        is not None
    ):

        print(
            f"Hard-filter pool: "
            f"{len(result['allowed_indices'])}"
        )

    display_candidates(
        result[
            "candidates"
        ]
    )


# ============================================================
# INTERACTIVE TEST
# ============================================================

def main():

    retriever = (
        RoutedHybridRetriever()
    )

    print()
    print(
        "=" * 90
    )

    print(
        "C4 ROUTED HYBRID RETRIEVER READY"
    )

    print(
        "=" * 90
    )

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

            result = (
                retriever
                .retrieve_candidates(
                    question
                )
            )

            display_result(
                result
            )

        except Exception as exc:

            print()
            print(
                f"ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()