from pathlib import Path
from collections import Counter, defaultdict
import json
import math
import re
import time
import unicodedata

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

EMBEDDINGS_FILE = Path(
    "data/processed/embeddings_c1.npy"
)

METADATA_FILE = Path(
    "data/processed/chunk_metadata_c1.jsonl"
)

MODEL_NAME = "BAAI/bge-m3"

DEVICE = "cpu"

EXPECTED_DIMENSION = 1024

DENSE_TOP_K = 20
BM25_TOP_K = 20

RRF_K = 60

FINAL_TOP_K = 5

BM25_K1 = 1.5
BM25_B = 0.75


# ============================================================
# TOKENIZER
# ============================================================

TOKEN_RE = re.compile(
    r"(?u)\b\w+\b"
)


def tokenize(text):
    """
    Tokenizer đơn giản cho BM25.

    - Unicode-aware
    - lowercase
    - NFC normalization

    Không stopword removal để giữ baseline C2
    đơn giản và reproducible.
    """

    text = unicodedata.normalize(
        "NFC",
        text
    )

    text = text.lower()

    return TOKEN_RE.findall(
        text
    )


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

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

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON "
                    f"{path}:{line_number}: "
                    f"{exc}"
                )

    return items


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:

    def __init__(
        self,
        embeddings_file=EMBEDDINGS_FILE,
        metadata_file=METADATA_FILE,
        model_name=MODEL_NAME,
        device=DEVICE,
    ):

        self.embeddings_file = Path(
            embeddings_file
        )

        self.metadata_file = Path(
            metadata_file
        )

        self.model_name = (
            model_name
        )

        self.device = (
            device
        )

        print("=" * 78)
        print(
            "C2 HYBRID RETRIEVER INITIALIZATION"
        )
        print("=" * 78)

        # ====================================================
        # LOAD INDEX
        # ====================================================

        if not self.embeddings_file.exists():

            raise FileNotFoundError(
                f"Missing: "
                f"{self.embeddings_file}"
            )

        if not self.metadata_file.exists():

            raise FileNotFoundError(
                f"Missing: "
                f"{self.metadata_file}"
            )

        self.embeddings = np.load(
            self.embeddings_file
        )

        self.metadata = load_jsonl(
            self.metadata_file
        )

        self.validate_dense_index()

        print(
            f"Chunks loaded       : "
            f"{len(self.metadata)}"
        )

        print(
            f"Vector shape        : "
            f"{self.embeddings.shape}"
        )

        print(
            "Dense index         : PASS"
        )

        # ====================================================
        # BM25 INDEX
        # ====================================================

        print()
        print(
            "Building BM25 index..."
        )

        bm25_start = (
            time.perf_counter()
        )

        self.build_bm25_index()

        bm25_elapsed = (
            time.perf_counter()
            - bm25_start
        )

        print(
            f"BM25 documents      : "
            f"{self.num_documents}"
        )

        print(
            f"BM25 vocabulary     : "
            f"{len(self.document_frequency)}"
        )

        print(
            f"BM25 avg doc length : "
            f"{self.average_document_length:.2f}"
        )

        print(
            f"BM25 build time     : "
            f"{bm25_elapsed:.4f}s"
        )

        print(
            "BM25 index          : PASS"
        )

        # ====================================================
        # QUERY ENCODER
        # ====================================================

        print()
        print(
            f"Loading encoder     : "
            f"{self.model_name}"
        )

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )

        print(
            "Query encoder       : PASS"
        )

        # ====================================================
        # WARM-UP
        # ====================================================

        print()
        print(
            "Running warm-up..."
        )

        self.encode_query(
            "quy định giao thông đường bộ"
        )

        print(
            "Warm-up             : PASS"
        )

        print()
        print(
            "C2 retriever ready."
        )


    # ========================================================
    # VALIDATE DENSE INDEX
    # ========================================================

    def validate_dense_index(self):

        if (
            self.embeddings.ndim
            != 2
        ):

            raise ValueError(
                "Embeddings must be 2D."
            )

        if (
            self.embeddings.shape[0]
            != len(
                self.metadata
            )
        ):

            raise ValueError(
                "Embedding / metadata "
                "count mismatch."
            )

        if (
            self.embeddings.shape[1]
            != EXPECTED_DIMENSION
        ):

            raise ValueError(
                "Wrong embedding dimension: "
                f"{self.embeddings.shape[1]}"
            )

        if np.isnan(
            self.embeddings
        ).any():

            raise ValueError(
                "NaN detected in embeddings."
            )

        if np.isinf(
            self.embeddings
        ).any():

            raise ValueError(
                "Inf detected in embeddings."
            )

        norms = np.linalg.norm(
            self.embeddings,
            axis=1
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-4
        ):

            raise ValueError(
                "Embeddings are not normalized."
            )

        seen_ids = set()

        for expected_index, item in enumerate(
            self.metadata
        ):

            if (
                item["vector_index"]
                != expected_index
            ):

                raise ValueError(
                    "vector_index mismatch "
                    f"at {expected_index}"
                )

            chunk_id = (
                item["chunk_id"]
            )

            if chunk_id in seen_ids:

                raise ValueError(
                    f"Duplicate chunk ID: "
                    f"{chunk_id}"
                )

            seen_ids.add(
                chunk_id
            )


    # ========================================================
    # BUILD BM25 INDEX
    # ========================================================

    def build_bm25_index(self):

        self.num_documents = len(
            self.metadata
        )

        self.document_lengths = (
            np.zeros(
                self.num_documents,
                dtype=np.float32
            )
        )

        self.document_frequency = Counter()

        self.inverted_index = defaultdict(
            list
        )

        total_length = 0

        for document_index, item in enumerate(
            self.metadata
        ):

            tokens = tokenize(
                item["text"]
            )

            token_counts = Counter(
                tokens
            )

            document_length = len(
                tokens
            )

            self.document_lengths[
                document_index
            ] = (
                document_length
            )

            total_length += (
                document_length
            )

            for term, frequency in (
                token_counts.items()
            ):

                self.document_frequency[
                    term
                ] += 1

                self.inverted_index[
                    term
                ].append(
                    (
                        document_index,
                        frequency,
                    )
                )

        if (
            self.num_documents
            == 0
        ):

            raise ValueError(
                "No documents for BM25."
            )

        self.average_document_length = (
            total_length
            / self.num_documents
        )

        if (
            self.average_document_length
            <= 0
        ):

            raise ValueError(
                "Invalid BM25 average "
                "document length."
            )


    # ========================================================
    # ENCODE QUERY
    # ========================================================

    def encode_query(
        self,
        question
    ):

        vector = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        vector = np.asarray(
            vector,
            dtype=np.float32
        )

        if (
            vector.shape[0]
            != EXPECTED_DIMENSION
        ):

            raise ValueError(
                "Unexpected query "
                "embedding dimension."
            )

        return vector


    # ========================================================
    # DENSE SEARCH
    # ========================================================

    def dense_search(
        self,
        query_vector,
        top_k=DENSE_TOP_K,
    ):

        start = (
            time.perf_counter()
        )

        scores = (
            self.embeddings
            @ query_vector
        )

        indices = (
            np.argsort(
                scores
            )[::-1][:top_k]
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        results = []

        for rank, index in enumerate(
            indices,
            start=1
        ):

            index = int(
                index
            )

            results.append(
                {
                    "rank": rank,
                    "vector_index": (
                        index
                    ),
                    "chunk_id": (
                        self.metadata[
                            index
                        ]["chunk_id"]
                    ),
                    "score": float(
                        scores[index]
                    ),
                    "metadata": (
                        self.metadata[
                            index
                        ]
                    ),
                }
            )

        return (
            results,
            elapsed,
        )


    # ========================================================
    # BM25 IDF
    # ========================================================

    def bm25_idf(
        self,
        term
    ):

        document_frequency = (
            self.document_frequency.get(
                term,
                0
            )
        )

        if (
            document_frequency
            == 0
        ):

            return 0.0

        return math.log(
            1.0
            + (
                self.num_documents
                - document_frequency
                + 0.5
            )
            / (
                document_frequency
                + 0.5
            )
        )


    # ========================================================
    # BM25 SEARCH
    # ========================================================

    def bm25_search(
        self,
        question,
        top_k=BM25_TOP_K,
    ):

        start = (
            time.perf_counter()
        )

        query_tokens = tokenize(
            question
        )

        scores = np.zeros(
            self.num_documents,
            dtype=np.float32
        )

        # Query term frequencies.
        query_counts = Counter(
            query_tokens
        )

        for term, query_frequency in (
            query_counts.items()
        ):

            postings = (
                self.inverted_index.get(
                    term
                )
            )

            if not postings:
                continue

            idf = (
                self.bm25_idf(
                    term
                )
            )

            for (
                document_index,
                term_frequency,
            ) in postings:

                document_length = (
                    self.document_lengths[
                        document_index
                    ]
                )

                denominator = (
                    term_frequency
                    + BM25_K1
                    * (
                        1.0
                        - BM25_B
                        + BM25_B
                        * (
                            document_length
                            / self.average_document_length
                        )
                    )
                )

                term_score = (
                    idf
                    * (
                        term_frequency
                        * (
                            BM25_K1
                            + 1.0
                        )
                    )
                    / denominator
                )

                # Nếu query lặp một term,
                # contribution tăng tương ứng.
                scores[
                    document_index
                ] += (
                    term_score
                    * query_frequency
                )

        # Chỉ rank documents có lexical match.
        positive_indices = np.where(
            scores > 0
        )[0]

        if (
            len(
                positive_indices
            )
            == 0
        ):

            elapsed = (
                time.perf_counter()
                - start
            )

            return (
                [],
                elapsed,
            )

        ranked_indices = (
            positive_indices[
                np.argsort(
                    scores[
                        positive_indices
                    ]
                )[::-1]
            ]
        )

        ranked_indices = (
            ranked_indices[:top_k]
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        results = []

        for rank, index in enumerate(
            ranked_indices,
            start=1
        ):

            index = int(
                index
            )

            results.append(
                {
                    "rank": rank,
                    "vector_index": (
                        index
                    ),
                    "chunk_id": (
                        self.metadata[
                            index
                        ]["chunk_id"]
                    ),
                    "score": float(
                        scores[index]
                    ),
                    "metadata": (
                        self.metadata[
                            index
                        ]
                    ),
                }
            )

        return (
            results,
            elapsed,
        )


    # ========================================================
    # RRF FUSION
    # ========================================================

    def reciprocal_rank_fusion(
        self,
        dense_results,
        bm25_results,
        final_top_k=FINAL_TOP_K,
        rrf_k=RRF_K,
    ):

        fused = {}

        # ----------------------------------------------------
        # Dense contribution
        # ----------------------------------------------------

        for result in dense_results:

            chunk_id = (
                result["chunk_id"]
            )

            if chunk_id not in fused:

                fused[
                    chunk_id
                ] = {
                    "chunk_id": (
                        chunk_id
                    ),
                    "vector_index": (
                        result[
                            "vector_index"
                        ]
                    ),
                    "metadata": (
                        result[
                            "metadata"
                        ]
                    ),
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                }

            fused[
                chunk_id
            ]["dense_rank"] = (
                result["rank"]
            )

            fused[
                chunk_id
            ]["dense_score"] = (
                result["score"]
            )

            fused[
                chunk_id
            ]["rrf_score"] += (
                1.0
                / (
                    rrf_k
                    + result["rank"]
                )
            )

        # ----------------------------------------------------
        # BM25 contribution
        # ----------------------------------------------------

        for result in bm25_results:

            chunk_id = (
                result["chunk_id"]
            )

            if chunk_id not in fused:

                fused[
                    chunk_id
                ] = {
                    "chunk_id": (
                        chunk_id
                    ),
                    "vector_index": (
                        result[
                            "vector_index"
                        ]
                    ),
                    "metadata": (
                        result[
                            "metadata"
                        ]
                    ),
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                }

            fused[
                chunk_id
            ]["bm25_rank"] = (
                result["rank"]
            )

            fused[
                chunk_id
            ]["bm25_score"] = (
                result["score"]
            )

            fused[
                chunk_id
            ]["rrf_score"] += (
                1.0
                / (
                    rrf_k
                    + result["rank"]
                )
            )

        # ----------------------------------------------------
        # Deterministic ranking
        # ----------------------------------------------------

        def best_source_rank(item):

            ranks = [
                rank
                for rank in [
                    item[
                        "dense_rank"
                    ],
                    item[
                        "bm25_rank"
                    ],
                ]
                if rank is not None
            ]

            if not ranks:
                return 10**9

            return min(
                ranks
            )

        ranked = sorted(
            fused.values(),
            key=lambda item: (
                -item[
                    "rrf_score"
                ],
                best_source_rank(
                    item
                ),
                item[
                    "chunk_id"
                ],
            )
        )

        final_results = []

        for rank, item in enumerate(
            ranked[
                :final_top_k
            ],
            start=1
        ):

            result = dict(
                item
            )

            result[
                "rank"
            ] = rank

            final_results.append(
                result
            )

        return final_results


    # ========================================================
    # HYBRID SEARCH
    # ========================================================

    def search(
        self,
        question,
        dense_top_k=DENSE_TOP_K,
        bm25_top_k=BM25_TOP_K,
        final_top_k=FINAL_TOP_K,
    ):

        total_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        encode_start = (
            time.perf_counter()
        )

        query_vector = (
            self.encode_query(
                question
            )
        )

        encode_elapsed = (
            time.perf_counter()
            - encode_start
        )

        # ----------------------------------------------------
        # Dense
        # ----------------------------------------------------

        (
            dense_results,
            dense_elapsed,
        ) = self.dense_search(
            query_vector,
            dense_top_k,
        )

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        (
            bm25_results,
            bm25_elapsed,
        ) = self.bm25_search(
            question,
            bm25_top_k,
        )

        # ----------------------------------------------------
        # RRF
        # ----------------------------------------------------

        fusion_start = (
            time.perf_counter()
        )

        fused_results = (
            self.reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                final_top_k=(
                    final_top_k
                ),
                rrf_k=RRF_K,
            )
        )

        fusion_elapsed = (
            time.perf_counter()
            - fusion_start
        )

        total_elapsed = (
            time.perf_counter()
            - total_start
        )

        timing = {
            "query_embedding": (
                encode_elapsed
            ),
            "dense_search": (
                dense_elapsed
            ),
            "bm25_search": (
                bm25_elapsed
            ),
            "rrf_fusion": (
                fusion_elapsed
            ),
            "total": (
                total_elapsed
            ),
        }

        return {
            "question": (
                question
            ),
            "dense_results": (
                dense_results
            ),
            "bm25_results": (
                bm25_results
            ),
            "results": (
                fused_results
            ),
            "timing": (
                timing
            ),
        }


# ============================================================
# DISPLAY
# ============================================================

def display_results(
    search_result
):

    print()
    print("=" * 90)
    print(
        "C2 HYBRID RETRIEVAL RESULT"
    )
    print("=" * 90)

    print()
    print(
        "QUESTION:"
    )

    print(
        search_result[
            "question"
        ]
    )

    print()

    timing = (
        search_result[
            "timing"
        ]
    )

    print(
        f"Query embedding : "
        f"{timing['query_embedding']:.4f}s"
    )

    print(
        f"Dense search    : "
        f"{timing['dense_search']:.6f}s"
    )

    print(
        f"BM25 search     : "
        f"{timing['bm25_search']:.6f}s"
    )

    print(
        f"RRF fusion      : "
        f"{timing['rrf_fusion']:.6f}s"
    )

    print(
        f"Total           : "
        f"{timing['total']:.4f}s"
    )

    for result in (
        search_result[
            "results"
        ]
    ):

        chunk = (
            result[
                "metadata"
            ]
        )

        print()
        print("-" * 90)

        print(
            f"FINAL RANK : "
            f"{result['rank']}"
        )

        print(
            f"RRF SCORE  : "
            f"{result['rrf_score']:.8f}"
        )

        print(
            f"DENSE RANK : "
            f"{result['dense_rank']}"
        )

        print(
            f"BM25 RANK  : "
            f"{result['bm25_rank']}"
        )

        print(
            f"CHUNK ID   : "
            f"{result['chunk_id']}"
        )

        print(
            f"ARTICLE    : "
            f"{chunk['article_number']}"
        )

        print(
            f"CLAUSE     : "
            f"{chunk['clause']}"
        )

        print(
            f"PAGES      : "
            f"{chunk['page_start']} "
            f"-> "
            f"{chunk['page_end']}"
        )

        print()

        print(
            chunk["text"]
        )


# ============================================================
# INTERACTIVE TEST
# ============================================================

def main():

    retriever = (
        HybridRetriever()
    )

    print()
    print("=" * 90)
    print(
        "C2 HYBRID RETRIEVER READY"
    )
    print("=" * 90)

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

        result = (
            retriever.search(
                question
            )
        )

        display_results(
            result
        )


if __name__ == "__main__":
    main()