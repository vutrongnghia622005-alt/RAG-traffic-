from pathlib import Path
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

GOLD_FILE = Path(
    "evaluation/gold_questions_span.jsonl"
)

PAGES_FILE = Path(
    "data/interim/pages_clean_c0.jsonl"
)

EMBEDDINGS_FILE = Path(
    "data/processed/embeddings_c1.npy"
)

METADATA_FILE = Path(
    "data/processed/chunk_metadata_c1.jsonl"
)

OUTPUT_DIR = Path(
    "results/evaluation"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "c1_span_results.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "c1_span_summary.json"
)


MODEL_NAME = "BAAI/bge-m3"

DEVICE = "cpu"

TOP_K = 5

EXPECTED_DIMENSION = 1024


# Evidence ↔ chunk mapping
MIN_OVERLAP_CHARS = 20
MIN_OVERLAP_RATIO = 0.50


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
    Ghép 74 clean pages thành một global document.

    Phải giống chính xác cách C1 chunker
    xây global source:

        page1
        + "\\n"
        + page2
        + "\\n"
        + ...
    """

    pages = sorted(
        pages,
        key=lambda x: x["page"]
    )

    parts = []

    current_offset = 0

    page_ranges = []

    for page in pages:

        text = (
            page["text"]
        )

        start = (
            current_offset
        )

        end = (
            start
            + len(text)
        )

        page_ranges.append(
            {
                "page": (
                    page["page"]
                ),
                "start": (
                    start
                ),
                "end": (
                    end
                ),
            }
        )

        parts.append(
            text
        )

        current_offset = (
            end
        )

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
# FIND GOLD EVIDENCE
# ============================================================

def locate_evidence(
    full_text,
    gold_questions,
):
    """
    Tìm vị trí global của từng evidence anchor.

    Cho phép whitespace trong gold khác với
    whitespace trong source.

    Ví dụ:

        giao thông đường bộ

    vẫn match:

        giao thông
        đường bộ
    """

    all_evidence = {}

    for item in gold_questions:

        question_id = (
            item["question_id"]
        )

        anchors = (
            item["evidence_anchors"]
        )

        evidence_units = []

        for evidence_index, anchor in enumerate(
            anchors,
            start=1
        ):

            anchor = (
                unicodedata.normalize(
                    "NFC",
                    anchor
                )
                .strip()
            )

            words = re.split(
                r"\s+",
                anchor
            )

            pattern_text = (
                r"\s+".join(
                    re.escape(word)
                    for word in words
                )
            )

            pattern = re.compile(
                pattern_text
            )

            matches = list(
                pattern.finditer(
                    full_text
                )
            )

            if len(matches) != 1:

                raise ValueError(
                    f"{question_id}: "
                    f"evidence anchor must "
                    f"appear exactly once. "
                    f"Found {len(matches)}.\n"
                    f"Anchor: {anchor}"
                )

            match = (
                matches[0]
            )

            evidence_units.append(
                {
                    "evidence_id": (
                        f"{question_id}"
                        f"_e{evidence_index:02d}"
                    ),
                    "text": (
                        anchor
                    ),
                    "start": (
                        match.start()
                    ),
                    "end": (
                        match.end()
                    ),
                    "length": (
                        match.end()
                        - match.start()
                    ),
                }
            )

        all_evidence[
            question_id
        ] = (
            evidence_units
        )

    return all_evidence


# ============================================================
# BUILD C1 CHUNK SPANS
# ============================================================

def build_chunk_spans(
    metadata
):
    """
    C1 đã có global offsets:

        source_start
        source_end

    nên không cần convert từ page-local offsets
    như C0.
    """

    spans = {}

    for chunk in metadata:

        chunk_id = (
            chunk["chunk_id"]
        )

        spans[
            chunk_id
        ] = {
            "start": (
                chunk["source_start"]
            ),
            "end": (
                chunk["source_end"]
            ),
        }

    return spans


# ============================================================
# OVERLAP
# ============================================================

def overlap_size(
    start_a,
    end_a,
    start_b,
    end_b,
):

    return max(
        0,
        min(
            end_a,
            end_b
        )
        - max(
            start_a,
            start_b
        )
    )


# ============================================================
# MAP EVIDENCE TO C1 CHUNKS
# ============================================================

def map_evidence_to_chunks(
    evidence_units,
    chunk_spans,
):
    """
    Một chunk được coi là relevant với evidence
    nếu overlap đủ lớn.

    Điều này giúp gold độc lập với chunk ID.
    """

    evidence_map = {}

    for evidence in evidence_units:

        evidence_length = (
            evidence["length"]
        )

        required_overlap = max(
            MIN_OVERLAP_CHARS,
            math.ceil(
                evidence_length
                * MIN_OVERLAP_RATIO
            ),
        )

        matched_chunks = set()

        for chunk_id, span in (
            chunk_spans.items()
        ):

            overlap = overlap_size(
                evidence["start"],
                evidence["end"],
                span["start"],
                span["end"],
            )

            if (
                overlap
                >= required_overlap
            ):

                matched_chunks.add(
                    chunk_id
                )

        if not matched_chunks:

            raise ValueError(
                "No C1 chunk mapped to "
                f"{evidence['evidence_id']}"
            )

        evidence_map[
            evidence["evidence_id"]
        ] = (
            matched_chunks
        )

    return evidence_map


# ============================================================
# VALIDATE INDEX
# ============================================================

def validate_index(
    embeddings,
    metadata,
):

    if (
        embeddings.ndim
        != 2
    ):

        raise ValueError(
            "Embeddings must be 2D."
        )

    if (
        embeddings.shape[0]
        != len(metadata)
    ):

        raise ValueError(
            "Embeddings / metadata "
            "count mismatch."
        )

    if (
        embeddings.shape[1]
        != EXPECTED_DIMENSION
    ):

        raise ValueError(
            "Wrong embedding dimension: "
            f"{embeddings.shape[1]}"
        )

    if np.isnan(
        embeddings
    ).any():

        raise ValueError(
            "NaN detected in embeddings."
        )

    if np.isinf(
        embeddings
    ).any():

        raise ValueError(
            "Inf detected in embeddings."
        )

    norms = np.linalg.norm(
        embeddings,
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

    for vector_index, item in enumerate(
        metadata
    ):

        if (
            item["vector_index"]
            != vector_index
        ):

            raise ValueError(
                "vector_index mismatch "
                f"at {vector_index}"
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


# ============================================================
# DENSE SEARCH
# ============================================================

def dense_search(
    query_vector,
    embeddings,
    metadata,
    top_k,
):

    scores = (
        embeddings
        @ query_vector
    )

    indices = (
        np.argsort(
            scores
        )[::-1][:top_k]
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
                "rank": (
                    rank
                ),
                "score": float(
                    scores[
                        index
                    ]
                ),
                "metadata": (
                    metadata[
                        index
                    ]
                ),
            }
        )

    return results


# ============================================================
# PRECISION@K
# ============================================================

def precision_at_k(
    retrieved,
    relevant,
    k,
):

    hits = sum(
        chunk_id in relevant
        for chunk_id in (
            retrieved[:k]
        )
    )

    return (
        hits / k
    )


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(
    retrieved,
    relevant,
    k,
):

    if not relevant:

        return 0.0

    hits = sum(
        chunk_id in relevant
        for chunk_id in (
            retrieved[:k]
        )
    )

    return (
        hits
        / len(relevant)
    )


# ============================================================
# HIT@K
# ============================================================

def hit_at_k(
    retrieved,
    relevant,
    k,
):

    return float(
        any(
            chunk_id in relevant
            for chunk_id in (
                retrieved[:k]
            )
        )
    )


# ============================================================
# MRR
# ============================================================

def reciprocal_rank(
    retrieved,
    relevant,
):

    for rank, chunk_id in enumerate(
        retrieved,
        start=1
    ):

        if chunk_id in relevant:

            return (
                1.0 / rank
            )

    return 0.0


# ============================================================
# nDCG@K
# ============================================================

def ndcg_at_k(
    retrieved,
    relevant,
    k,
):

    dcg = 0.0

    for rank, chunk_id in enumerate(
        retrieved[:k],
        start=1
    ):

        relevance = (
            1.0
            if chunk_id in relevant
            else 0.0
        )

        dcg += (
            relevance
            / math.log2(
                rank + 1
            )
        )

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:

        return 0.0

    idcg = sum(
        1.0
        / math.log2(
            rank + 1
        )
        for rank in range(
            1,
            ideal_hits + 1
        )
    )

    return (
        dcg / idcg
    )


# ============================================================
# EVIDENCE RECALL@K
# ============================================================

def evidence_recall_at_k(
    retrieved,
    evidence_map,
    k,
):
    """
    Tỷ lệ evidence units được cover bởi Top-K.

    Đây là metric quan trọng nhất
    cho span-level evaluation.
    """

    if not evidence_map:

        return 0.0

    retrieved_set = set(
        retrieved[:k]
    )

    covered = 0

    for chunk_ids in (
        evidence_map.values()
    ):

        if (
            retrieved_set
            & chunk_ids
        ):

            covered += 1

    return (
        covered
        / len(
            evidence_map
        )
    )


# ============================================================
# MEAN
# ============================================================

def mean(values):

    if not values:

        return 0.0

    return float(
        np.mean(
            values
        )
    )


# ============================================================
# LATENCY SUMMARY
# ============================================================

def latency_summary(
    values
):

    return {
        "mean": float(
            np.mean(
                values
            )
        ),
        "p50": float(
            np.percentile(
                values,
                50
            )
        ),
        "p95": float(
            np.percentile(
                values,
                95
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)

    print(
        "C1 STRUCTURE-AWARE DENSE RETRIEVAL EVALUATION"
    )

    print("=" * 75)

    # ========================================================
    # CHECK FILES
    # ========================================================

    required_files = [
        GOLD_FILE,
        PAGES_FILE,
        EMBEDDINGS_FILE,
        METADATA_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing: {path}"
            )

    # ========================================================
    # LOAD DATA
    # ========================================================

    pages = (
        load_jsonl(
            PAGES_FILE
        )
    )

    gold_questions = (
        load_jsonl(
            GOLD_FILE
        )
    )

    metadata = (
        load_jsonl(
            METADATA_FILE
        )
    )

    embeddings = (
        np.load(
            EMBEDDINGS_FILE
        )
    )

    print(
        f"Pages              : "
        f"{len(pages)}"
    )

    print(
        f"Chunks             : "
        f"{len(metadata)}"
    )

    print(
        f"Questions          : "
        f"{len(gold_questions)}"
    )

    # ========================================================
    # VALIDATE INDEX
    # ========================================================

    validate_index(
        embeddings,
        metadata,
    )

    print(
        "C1 index validation: PASS"
    )

    # ========================================================
    # GLOBAL SOURCE
    # ========================================================

    (
        full_text,
        _
    ) = build_global_document(
        pages
    )

    # ========================================================
    # GOLD EVIDENCE
    # ========================================================

    all_evidence = (
        locate_evidence(
            full_text,
            gold_questions,
        )
    )

    total_evidence = sum(
        len(items)
        for items in (
            all_evidence.values()
        )
    )

    print(
        f"Evidence units     : "
        f"{total_evidence}"
    )

    print(
        "Evidence validation: PASS"
    )

    # ========================================================
    # CHUNK SPANS
    # ========================================================

    chunk_spans = (
        build_chunk_spans(
            metadata
        )
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()

    print(
        f"Loading model      : "
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE
    )

    print(
        "Model loaded       : PASS"
    )

    # ========================================================
    # WARM-UP
    # ========================================================

    print()

    print(
        "Running warm-up..."
    )

    warmup = model.encode(
        [
            "quy định giao thông đường bộ"
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].astype(
        np.float32
    )

    dense_search(
        warmup,
        embeddings,
        metadata,
        TOP_K,
    )

    print(
        "Warm-up            : PASS"
    )

    # ========================================================
    # METRIC STORAGE
    # ========================================================

    precision_values = []
    recall_values = []
    evidence_recall_values = []
    hit_values = []
    mrr_values = []
    ndcg_values = []
    latency_values = []

    result_rows = []

    # ========================================================
    # EVALUATION
    # ========================================================

    print()

    print("=" * 75)

    print(
        "RUNNING C1 GOLD BENCHMARK"
    )

    print("=" * 75)

    for question_number, item in enumerate(
        gold_questions,
        start=1
    ):

        question_id = (
            item["question_id"]
        )

        question = (
            item["question"]
        )

        evidence_units = (
            all_evidence[
                question_id
            ]
        )

        evidence_map = (
            map_evidence_to_chunks(
                evidence_units,
                chunk_spans,
            )
        )

        # ----------------------------------------------------
        # Relevant chunk set
        # ----------------------------------------------------

        relevant_chunks = set()

        for chunk_ids in (
            evidence_map.values()
        ):

            relevant_chunks.update(
                chunk_ids
            )

        # ----------------------------------------------------
        # End-to-end retrieval latency
        #
        # Query embedding + dense search
        # ----------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        query_vector = model.encode(
            [
                question
            ],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(
            np.float32
        )

        results = dense_search(
            query_vector,
            embeddings,
            metadata,
            TOP_K,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        latency_values.append(
            elapsed
        )

        # ----------------------------------------------------
        # Retrieved IDs
        # ----------------------------------------------------

        retrieved_ids = [
            result[
                "metadata"
            ][
                "chunk_id"
            ]
            for result in results
        ]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        precision = precision_at_k(
            retrieved_ids,
            relevant_chunks,
            TOP_K,
        )

        recall = recall_at_k(
            retrieved_ids,
            relevant_chunks,
            TOP_K,
        )

        evidence_recall = (
            evidence_recall_at_k(
                retrieved_ids,
                evidence_map,
                TOP_K,
            )
        )

        hit = hit_at_k(
            retrieved_ids,
            relevant_chunks,
            TOP_K,
        )

        mrr = reciprocal_rank(
            retrieved_ids,
            relevant_chunks,
        )

        ndcg = ndcg_at_k(
            retrieved_ids,
            relevant_chunks,
            TOP_K,
        )

        precision_values.append(
            precision
        )

        recall_values.append(
            recall
        )

        evidence_recall_values.append(
            evidence_recall
        )

        hit_values.append(
            hit
        )

        mrr_values.append(
            mrr
        )

        ndcg_values.append(
            ndcg
        )

        # ----------------------------------------------------
        # Retrieval details
        # ----------------------------------------------------

        retrieved_details = []

        for result in results:

            chunk = (
                result[
                    "metadata"
                ]
            )

            chunk_id = (
                chunk[
                    "chunk_id"
                ]
            )

            matched_evidence = []

            for (
                evidence_id,
                chunk_ids
            ) in evidence_map.items():

                if (
                    chunk_id
                    in chunk_ids
                ):

                    matched_evidence.append(
                        evidence_id
                    )

            retrieved_details.append(
                {
                    "rank": (
                        result[
                            "rank"
                        ]
                    ),
                    "chunk_id": (
                        chunk_id
                    ),
                    "score": (
                        result[
                            "score"
                        ]
                    ),
                    "article_number": (
                        chunk[
                            "article_number"
                        ]
                    ),
                    "clause": (
                        chunk[
                            "clause"
                        ]
                    ),
                    "page_start": (
                        chunk[
                            "page_start"
                        ]
                    ),
                    "page_end": (
                        chunk[
                            "page_end"
                        ]
                    ),
                    "relevant": (
                        chunk_id
                        in relevant_chunks
                    ),
                    "evidence_ids": (
                        matched_evidence
                    ),
                    "text_preview": (
                        chunk[
                            "text"
                        ][
                            :300
                        ]
                    ),
                }
            )

        # ----------------------------------------------------
        # Save question row
        # ----------------------------------------------------

        result_rows.append(
            {
                "question_id": (
                    question_id
                ),
                "question": (
                    question
                ),
                "evidence_units": (
                    evidence_units
                ),
                "gold_chunk_count": (
                    len(
                        relevant_chunks
                    )
                ),
                "metrics": {
                    "precision@5": (
                        precision
                    ),
                    "recall@5": (
                        recall
                    ),
                    "evidence_recall@5": (
                        evidence_recall
                    ),
                    "hit@5": (
                        hit
                    ),
                    "mrr": (
                        mrr
                    ),
                    "ndcg@5": (
                        ndcg
                    ),
                },
                "latency_seconds": (
                    elapsed
                ),
                "retrieved": (
                    retrieved_details
                ),
            }
        )

        print(
            f"[{question_number:02d}/"
            f"{len(gold_questions):02d}] "
            f"{question_id} "
            f"| P@5={precision:.2f} "
            f"R@5={recall:.2f} "
            f"ER@5={evidence_recall:.2f} "
            f"MRR={mrr:.2f} "
            f"nDCG={ndcg:.2f}"
        )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    final_metrics = {
        "precision@5": (
            mean(
                precision_values
            )
        ),
        "recall@5": (
            mean(
                recall_values
            )
        ),
        "evidence_recall@5": (
            mean(
                evidence_recall_values
            )
        ),
        "hit@5": (
            mean(
                hit_values
            )
        ),
        "mrr": (
            mean(
                mrr_values
            )
        ),
        "ndcg@5": (
            mean(
                ndcg_values
            )
        ),
    }

    latency = (
        latency_summary(
            latency_values
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for row in result_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    summary = {
        "config": (
            "C1_structure_aware"
        ),
        "chunking": (
            "structure-aware"
        ),
        "embedding_model": (
            MODEL_NAME
        ),
        "retrieval": (
            "dense_top_5"
        ),
        "questions": (
            len(
                gold_questions
            )
        ),
        "evidence_units": (
            total_evidence
        ),
        "chunks": (
            len(
                metadata
            )
        ),
        "metrics": (
            final_metrics
        ),
        "latency": (
            latency
        ),
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()

    print("=" * 75)

    print(
        "C1 RETRIEVAL EVALUATION SUMMARY"
    )

    print("=" * 75)

    print(
        f"Questions         : "
        f"{len(gold_questions)}"
    )

    print(
        f"Evidence units    : "
        f"{total_evidence}"
    )

    print(
        f"Chunks            : "
        f"{len(metadata)}"
    )

    print()

    print(
        f"Precision@5       : "
        f"{final_metrics['precision@5']:.4f}"
    )

    print(
        f"Recall@5          : "
        f"{final_metrics['recall@5']:.4f}"
    )

    print(
        f"Evidence Recall@5 : "
        f"{final_metrics['evidence_recall@5']:.4f}"
    )

    print(
        f"Hit@5             : "
        f"{final_metrics['hit@5']:.4f}"
    )

    print(
        f"MRR               : "
        f"{final_metrics['mrr']:.4f}"
    )

    print(
        f"nDCG@5            : "
        f"{final_metrics['ndcg@5']:.4f}"
    )

    print()

    print(
        f"Mean latency      : "
        f"{latency['mean']:.4f}s"
    )

    print(
        f"p50 latency       : "
        f"{latency['p50']:.4f}s"
    )

    print(
        f"p95 latency       : "
        f"{latency['p95']:.4f}s"
    )

    print()

    print(
        f"Results saved     : "
        f"{RESULTS_FILE}"
    )

    print(
        f"Summary saved     : "
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()