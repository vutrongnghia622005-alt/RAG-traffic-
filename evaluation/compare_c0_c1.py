from pathlib import Path
from collections import defaultdict
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

C0_EMBEDDINGS = Path(
    "data/processed/embeddings_c0.npy"
)

C0_METADATA = Path(
    "data/processed/chunk_metadata_c0.jsonl"
)

C1_EMBEDDINGS = Path(
    "data/processed/embeddings_c1.npy"
)

C1_METADATA = Path(
    "data/processed/chunk_metadata_c1.jsonl"
)

OUTPUT_DIR = Path(
    "results/evaluation"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "c0_vs_c1_results.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "c0_vs_c1_summary.json"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "c0_vs_c1_report.md"
)

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cpu"

TOP_K = 5

EXPECTED_DIMENSION = 1024

MIN_OVERLAP_CHARS = 20
MIN_OVERLAP_RATIO = 0.50


# ============================================================
# JSONL
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
# BUILD GLOBAL SOURCE
# ============================================================

def build_global_document(pages):
    """
    Ghép source giống cách C1 chunker đã làm.

    page1 + "\\n" + page2 + "\\n" + ...
    """

    pages = sorted(
        pages,
        key=lambda x: x["page"]
    )

    parts = []

    page_offsets = {}

    current_offset = 0

    for page in pages:

        page_number = (
            page["page"]
        )

        text = (
            page["text"]
        )

        start = (
            current_offset
        )

        end = (
            start + len(text)
        )

        page_offsets[
            page_number
        ] = {
            "start": start,
            "end": end,
        }

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

    return (
        "".join(parts),
        page_offsets,
    )


# ============================================================
# LOCATE GOLD EVIDENCE
# ============================================================

def locate_evidence(
    full_text,
    gold_questions,
):
    """
    Tìm evidence anchors bằng flexible whitespace.

    Ví dụ source:
        "giao thông\\nđường bộ"

    Gold:
        "giao thông đường bộ"

    vẫn match được.
    """

    all_evidence = {}

    for item in gold_questions:

        question_id = (
            item["question_id"]
        )

        evidence_units = []

        for evidence_index, anchor in enumerate(
            item["evidence_anchors"],
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
                    f"evidence anchor must occur "
                    f"exactly once; "
                    f"found {len(matches)}.\n"
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
                    "text": anchor,
                    "start": match.start(),
                    "end": match.end(),
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
# BUILD C0 GLOBAL SPANS
# ============================================================

def build_c0_spans(
    metadata,
    page_offsets,
):
    """
    C0 metadata:

        page
        start_char
        end_char

    start/end là local offset của page.

    Chuyển thành global offsets để so
    với cùng gold evidence spans.
    """

    spans = {}

    for chunk in metadata:

        page = (
            chunk["page"]
        )

        if page not in page_offsets:

            raise ValueError(
                f"Unknown page in C0: "
                f"{page}"
            )

        page_start = (
            page_offsets[
                page
            ]["start"]
        )

        global_start = (
            page_start
            + chunk["start_char"]
        )

        global_end = (
            page_start
            + chunk["end_char"]
        )

        spans[
            chunk["chunk_id"]
        ] = {
            "start": global_start,
            "end": global_end,
        }

    return spans


# ============================================================
# BUILD C1 GLOBAL SPANS
# ============================================================

def build_c1_spans(
    metadata
):
    """
    C1 đã lưu trực tiếp:

        source_start
        source_end

    theo global source offsets.
    """

    spans = {}

    for chunk in metadata:

        spans[
            chunk["chunk_id"]
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
# MAP EVIDENCE -> CONFIG CHUNKS
# ============================================================

def map_evidence_to_chunks(
    evidence_units,
    chunk_spans,
):
    """
    Same evidence span được map độc lập
    sang chunks của C0 hoặc C1.

    Đây là điểm quan trọng để ablation
    không phụ thuộc chunk ID.
    """

    evidence_chunk_map = {}

    for evidence in evidence_units:

        evidence_id = (
            evidence["evidence_id"]
        )

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

        matched = set()

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

                matched.add(
                    chunk_id
                )

        if not matched:

            raise ValueError(
                f"No chunks mapped to "
                f"{evidence_id}"
            )

        evidence_chunk_map[
            evidence_id
        ] = matched

    return evidence_chunk_map


# ============================================================
# RETRIEVAL INDEX
# ============================================================

class DenseIndex:

    def __init__(
        self,
        name,
        embeddings_path,
        metadata_path,
    ):

        self.name = name

        print()
        print(
            f"Loading {name}..."
        )

        self.embeddings = (
            np.load(
                embeddings_path
            )
        )

        self.metadata = (
            load_jsonl(
                metadata_path
            )
        )

        self.validate()

        print(
            f"{name} vectors        : "
            f"{self.embeddings.shape[0]}"
        )

        print(
            f"{name} dimension      : "
            f"{self.embeddings.shape[1]}"
        )

        print(
            f"{name} validation     : PASS"
        )

    def validate(self):

        if (
            self.embeddings.ndim
            != 2
        ):

            raise ValueError(
                f"{self.name}: "
                "embeddings must be 2D"
            )

        if (
            self.embeddings.shape[1]
            != EXPECTED_DIMENSION
        ):

            raise ValueError(
                f"{self.name}: "
                "wrong embedding dimension"
            )

        if (
            self.embeddings.shape[0]
            != len(
                self.metadata
            )
        ):

            raise ValueError(
                f"{self.name}: "
                "metadata/vector count mismatch"
            )

        if np.isnan(
            self.embeddings
        ).any():

            raise ValueError(
                f"{self.name}: "
                "NaN detected"
            )

        if np.isinf(
            self.embeddings
        ).any():

            raise ValueError(
                f"{self.name}: "
                "Inf detected"
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
                f"{self.name}: "
                "embeddings not normalized"
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
                    f"{self.name}: "
                    "vector_index mismatch at "
                    f"{expected_index}"
                )

            chunk_id = (
                item["chunk_id"]
            )

            if chunk_id in seen_ids:

                raise ValueError(
                    f"{self.name}: "
                    f"duplicate chunk ID: "
                    f"{chunk_id}"
                )

            seen_ids.add(
                chunk_id
            )

    def search(
        self,
        query_vector,
        top_k,
    ):
        """
        Dot product = cosine similarity
        vì document/query embeddings
        đều L2 normalized.
        """

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

        for rank, vector_index in enumerate(
            indices,
            start=1
        ):

            item = (
                self.metadata[
                    int(vector_index)
                ]
            )

            results.append(
                {
                    "rank": rank,
                    "vector_index": (
                        int(vector_index)
                    ),
                    "score": float(
                        scores[
                            vector_index
                        ]
                    ),
                    "metadata": item,
                }
            )

        return (
            results,
            elapsed,
        )


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    retrieved,
    relevant,
    k,
):

    hits = sum(
        chunk_id in relevant
        for chunk_id
        in retrieved[:k]
    )

    return (
        hits / k
    )


def recall_at_k(
    retrieved,
    relevant,
    k,
):

    if not relevant:

        return 0.0

    hits = sum(
        chunk_id in relevant
        for chunk_id
        in retrieved[:k]
    )

    return (
        hits
        / len(relevant)
    )


def hit_at_k(
    retrieved,
    relevant,
    k,
):

    return float(
        any(
            chunk_id in relevant
            for chunk_id
            in retrieved[:k]
        )
    )


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


def evidence_recall_at_k(
    retrieved,
    evidence_chunk_map,
    k,
):

    if not evidence_chunk_map:

        return 0.0

    retrieved_set = set(
        retrieved[:k]
    )

    covered = 0

    for chunk_ids in (
        evidence_chunk_map.values()
    ):

        if (
            retrieved_set
            & chunk_ids
        ):

            covered += 1

    return (
        covered
        / len(
            evidence_chunk_map
        )
    )


# ============================================================
# SCORE ONE CONFIG
# ============================================================

def score_results(
    retrieval_results,
    evidence_chunk_map,
):

    relevant_chunks = set()

    for chunk_ids in (
        evidence_chunk_map.values()
    ):

        relevant_chunks.update(
            chunk_ids
        )

    retrieved_ids = [
        result[
            "metadata"
        ]["chunk_id"]
        for result in retrieval_results
    ]

    metrics = {
        "precision@5": (
            precision_at_k(
                retrieved_ids,
                relevant_chunks,
                TOP_K,
            )
        ),
        "recall@5": (
            recall_at_k(
                retrieved_ids,
                relevant_chunks,
                TOP_K,
            )
        ),
        "evidence_recall@5": (
            evidence_recall_at_k(
                retrieved_ids,
                evidence_chunk_map,
                TOP_K,
            )
        ),
        "hit@5": (
            hit_at_k(
                retrieved_ids,
                relevant_chunks,
                TOP_K,
            )
        ),
        "mrr": (
            reciprocal_rank(
                retrieved_ids,
                relevant_chunks,
            )
        ),
        "ndcg@5": (
            ndcg_at_k(
                retrieved_ids,
                relevant_chunks,
                TOP_K,
            )
        ),
    }

    details = []

    for result in (
        retrieval_results
    ):

        metadata = (
            result["metadata"]
        )

        chunk_id = (
            metadata[
                "chunk_id"
            ]
        )

        evidence_ids = []

        for (
            evidence_id,
            chunk_ids
        ) in evidence_chunk_map.items():

            if (
                chunk_id
                in chunk_ids
            ):

                evidence_ids.append(
                    evidence_id
                )

        if "page" in metadata:

            page_info = (
                str(
                    metadata["page"]
                )
            )

        else:

            page_info = (
                f"{metadata['page_start']}"
                f"->{metadata['page_end']}"
            )

        details.append(
            {
                "rank": (
                    result["rank"]
                ),
                "chunk_id": (
                    chunk_id
                ),
                "page": (
                    page_info
                ),
                "score": (
                    result["score"]
                ),
                "relevant": (
                    chunk_id
                    in relevant_chunks
                ),
                "evidence_ids": (
                    evidence_ids
                ),
                "text_preview": (
                    metadata["text"][
                        :300
                    ]
                ),
            }
        )

    return (
        metrics,
        details,
        relevant_chunks,
    )


# ============================================================
# MEAN METRICS
# ============================================================

def mean_metrics(rows):

    metric_names = [
        "precision@5",
        "recall@5",
        "evidence_recall@5",
        "hit@5",
        "mrr",
        "ndcg@5",
    ]

    result = {}

    for metric in metric_names:

        result[
            metric
        ] = float(
            np.mean(
                [
                    row[
                        "metrics"
                    ][metric]
                    for row in rows
                ]
            )
        )

    return result


# ============================================================
# LATENCY SUMMARY
# ============================================================

def latency_summary(values):

    return {
        "mean_seconds": float(
            np.mean(values)
        ),
        "p50_seconds": float(
            np.percentile(
                values,
                50
            )
        ),
        "p95_seconds": float(
            np.percentile(
                values,
                95
            )
        ),
    }


# ============================================================
# BUILD MARKDOWN REPORT
# ============================================================

def build_markdown_report(
    summary,
    results,
):

    lines = []

    lines.append(
        "# C0 vs C1 Retrieval Comparison"
    )

    lines.append("")

    lines.append(
        "## Overall metrics"
    )

    lines.append("")

    lines.append(
        "| Metric | C0 | C1 | Delta C1-C0 |"
    )

    lines.append(
        "|---|---:|---:|---:|"
    )

    metric_order = [
        "precision@5",
        "recall@5",
        "evidence_recall@5",
        "hit@5",
        "mrr",
        "ndcg@5",
    ]

    for metric in metric_order:

        c0 = (
            summary[
                "C0"
            ]["metrics"][metric]
        )

        c1 = (
            summary[
                "C1"
            ]["metrics"][metric]
        )

        delta = (
            c1 - c0
        )

        lines.append(
            f"| {metric} "
            f"| {c0:.4f} "
            f"| {c1:.4f} "
            f"| {delta:+.4f} |"
        )

    lines.append("")
    lines.append(
        "## Per-question comparison"
    )
    lines.append("")

    lines.append(
        "| QID | C0 ER@5 | C1 ER@5 | "
        "C0 MRR | C1 MRR | "
        "C0 nDCG | C1 nDCG |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )

    for row in results:

        c0 = (
            row["C0"]["metrics"]
        )

        c1 = (
            row["C1"]["metrics"]
        )

        lines.append(
            f"| {row['question_id']} "
            f"| {c0['evidence_recall@5']:.2f} "
            f"| {c1['evidence_recall@5']:.2f} "
            f"| {c0['mrr']:.2f} "
            f"| {c1['mrr']:.2f} "
            f"| {c0['ndcg@5']:.2f} "
            f"| {c1['ndcg@5']:.2f} |"
        )

    lines.append("")
    lines.append(
        "## Interpretation"
    )
    lines.append("")

    er_delta = (
        summary["C1"]["metrics"][
            "evidence_recall@5"
        ]
        - summary["C0"]["metrics"][
            "evidence_recall@5"
        ]
    )

    mrr_delta = (
        summary["C1"]["metrics"]["mrr"]
        - summary["C0"]["metrics"]["mrr"]
    )

    ndcg_delta = (
        summary["C1"]["metrics"]["ndcg@5"]
        - summary["C0"]["metrics"]["ndcg@5"]
    )

    lines.append(
        f"- Evidence Recall@5 delta: "
        f"{er_delta:+.4f}"
    )

    lines.append(
        f"- MRR delta: "
        f"{mrr_delta:+.4f}"
    )

    lines.append(
        f"- nDCG@5 delta: "
        f"{ndcg_delta:+.4f}"
    )

    return (
        "\n".join(lines)
        + "\n"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "C0 vs C1 DENSE RETRIEVAL ABLATION"
    )
    print("=" * 78)

    # ========================================================
    # LOAD SOURCE + GOLD
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

    (
        full_text,
        page_offsets,
    ) = build_global_document(
        pages
    )

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
        f"Pages                 : "
        f"{len(pages)}"
    )

    print(
        f"Questions             : "
        f"{len(gold_questions)}"
    )

    print(
        f"Evidence units        : "
        f"{total_evidence}"
    )

    print(
        "Evidence validation  : PASS"
    )

    # ========================================================
    # LOAD INDEXES
    # ========================================================

    c0_index = DenseIndex(
        name="C0",
        embeddings_path=(
            C0_EMBEDDINGS
        ),
        metadata_path=(
            C0_METADATA
        ),
    )

    c1_index = DenseIndex(
        name="C1",
        embeddings_path=(
            C1_EMBEDDINGS
        ),
        metadata_path=(
            C1_METADATA
        ),
    )

    # ========================================================
    # BUILD SOURCE SPANS
    # ========================================================

    c0_spans = (
        build_c0_spans(
            c0_index.metadata,
            page_offsets,
        )
    )

    c1_spans = (
        build_c1_spans(
            c1_index.metadata
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Loading query encoder..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE
    )

    print(
        "Query encoder         : PASS"
    )

    # ========================================================
    # WARMUP
    # ========================================================

    print()
    print(
        "Running warm-up..."
    )

    warmup_vector = (
        model.encode(
            [
                "quy định giao thông đường bộ"
            ],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        .astype(
            np.float32
        )
    )

    c0_index.search(
        warmup_vector,
        TOP_K
    )

    c1_index.search(
        warmup_vector,
        TOP_K
    )

    print(
        "Warm-up               : PASS"
    )

    # ========================================================
    # RUN
    # ========================================================

    print()
    print("=" * 78)
    print(
        "RUNNING SAME GOLD SET ON C0 AND C1"
    )
    print("=" * 78)

    output_rows = []

    c0_rows = []
    c1_rows = []

    c0_latencies = []
    c1_latencies = []

    query_embedding_latencies = []

    for question_index, item in enumerate(
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

        # ----------------------------------------------------
        # SAME evidence -> C0 chunks
        # ----------------------------------------------------

        c0_evidence_map = (
            map_evidence_to_chunks(
                evidence_units,
                c0_spans,
            )
        )

        # ----------------------------------------------------
        # SAME evidence -> C1 chunks
        # ----------------------------------------------------

        c1_evidence_map = (
            map_evidence_to_chunks(
                evidence_units,
                c1_spans,
            )
        )

        # ----------------------------------------------------
        # Encode query ONCE
        # ----------------------------------------------------

        encode_start = (
            time.perf_counter()
        )

        query_vector = (
            model.encode(
                [question],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            .astype(
                np.float32
            )
        )

        encode_elapsed = (
            time.perf_counter()
            - encode_start
        )

        query_embedding_latencies.append(
            encode_elapsed
        )

        # ----------------------------------------------------
        # C0 search
        # ----------------------------------------------------

        (
            c0_results,
            c0_latency,
        ) = c0_index.search(
            query_vector,
            TOP_K,
        )

        (
            c0_metrics,
            c0_details,
            c0_relevant,
        ) = score_results(
            c0_results,
            c0_evidence_map,
        )

        c0_latencies.append(
            c0_latency
        )

        # ----------------------------------------------------
        # C1 search
        # ----------------------------------------------------

        (
            c1_results,
            c1_latency,
        ) = c1_index.search(
            query_vector,
            TOP_K,
        )

        (
            c1_metrics,
            c1_details,
            c1_relevant,
        ) = score_results(
            c1_results,
            c1_evidence_map,
        )

        c1_latencies.append(
            c1_latency
        )

        c0_rows.append(
            {
                "metrics": (
                    c0_metrics
                )
            }
        )

        c1_rows.append(
            {
                "metrics": (
                    c1_metrics
                )
            }
        )

        output_rows.append(
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
                "query_embedding_latency_seconds": (
                    encode_elapsed
                ),
                "C0": {
                    "gold_chunk_count": (
                        len(c0_relevant)
                    ),
                    "metrics": (
                        c0_metrics
                    ),
                    "retrieval_latency_seconds": (
                        c0_latency
                    ),
                    "retrieved": (
                        c0_details
                    ),
                },
                "C1": {
                    "gold_chunk_count": (
                        len(c1_relevant)
                    ),
                    "metrics": (
                        c1_metrics
                    ),
                    "retrieval_latency_seconds": (
                        c1_latency
                    ),
                    "retrieved": (
                        c1_details
                    ),
                },
            }
        )

        print(
            f"[{question_index:02d}/"
            f"{len(gold_questions):02d}] "
            f"{question_id} "
            f"| C0 ER={c0_metrics['evidence_recall@5']:.2f} "
            f"MRR={c0_metrics['mrr']:.2f} "
            f"nDCG={c0_metrics['ndcg@5']:.2f} "
            f"|| "
            f"C1 ER={c1_metrics['evidence_recall@5']:.2f} "
            f"MRR={c1_metrics['mrr']:.2f} "
            f"nDCG={c1_metrics['ndcg@5']:.2f}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    c0_metrics = (
        mean_metrics(
            c0_rows
        )
    )

    c1_metrics = (
        mean_metrics(
            c1_rows
        )
    )

    metric_deltas = {
        metric: (
            c1_metrics[metric]
            - c0_metrics[metric]
        )
        for metric in (
            c0_metrics.keys()
        )
    }

    # --------------------------------------------------------
    # Per-question win/tie/loss
    # --------------------------------------------------------

    win_tie_loss = {}

    for metric in [
        "evidence_recall@5",
        "mrr",
        "ndcg@5",
    ]:

        wins = 0
        ties = 0
        losses = 0

        for row in output_rows:

            c0_value = (
                row[
                    "C0"
                ]["metrics"][metric]
            )

            c1_value = (
                row[
                    "C1"
                ]["metrics"][metric]
            )

            if (
                c1_value
                > c0_value
                + 1e-12
            ):

                wins += 1

            elif (
                c1_value
                < c0_value
                - 1e-12
            ):

                losses += 1

            else:

                ties += 1

        win_tie_loss[
            metric
        ] = {
            "C1_wins": wins,
            "ties": ties,
            "C1_losses": losses,
        }

    summary = {
        "experiment": (
            "C0_fixed_vs_C1_structure_aware"
        ),
        "questions": (
            len(gold_questions)
        ),
        "evidence_units": (
            total_evidence
        ),
        "top_k": (
            TOP_K
        ),
        "embedding_model": (
            MODEL_NAME
        ),
        "C0": {
            "chunks": int(
                c0_index.embeddings.shape[0]
            ),
            "metrics": (
                c0_metrics
            ),
            "retrieval_latency": (
                latency_summary(
                    c0_latencies
                )
            ),
        },
        "C1": {
            "chunks": int(
                c1_index.embeddings.shape[0]
            ),
            "metrics": (
                c1_metrics
            ),
            "retrieval_latency": (
                latency_summary(
                    c1_latencies
                )
            ),
        },
        "delta_C1_minus_C0": (
            metric_deltas
        ),
        "win_tie_loss": (
            win_tie_loss
        ),
        "query_embedding_latency": (
            latency_summary(
                query_embedding_latencies
            )
        ),
    }

    # ========================================================
    # SAVE
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

        for row in output_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

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

    markdown = (
        build_markdown_report(
            summary,
            output_rows,
        )
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            markdown
        )

    # ========================================================
    # PRINT FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 78)
    print(
        "C0 vs C1 SUMMARY"
    )
    print("=" * 78)

    print(
        f"{'Metric':<22}"
        f"{'C0':>10}"
        f"{'C1':>10}"
        f"{'Delta':>12}"
    )

    print(
        "-" * 54
    )

    for metric in [
        "precision@5",
        "recall@5",
        "evidence_recall@5",
        "hit@5",
        "mrr",
        "ndcg@5",
    ]:

        print(
            f"{metric:<22}"
            f"{c0_metrics[metric]:>10.4f}"
            f"{c1_metrics[metric]:>10.4f}"
            f"{metric_deltas[metric]:>+12.4f}"
        )

    print()
    print(
        "C1 win/tie/loss:"
    )

    for metric, values in (
        win_tie_loss.items()
    ):

        print(
            f"  {metric:<20}"
            f"{values['C1_wins']} / "
            f"{values['ties']} / "
            f"{values['C1_losses']}"
        )

    print()
    print(
        f"C0 chunks            : "
        f"{summary['C0']['chunks']}"
    )

    print(
        f"C1 chunks            : "
        f"{summary['C1']['chunks']}"
    )

    print()

    print(
        f"C0 retrieval mean    : "
        f"{summary['C0']['retrieval_latency']['mean_seconds']:.6f}s"
    )

    print(
        f"C1 retrieval mean    : "
        f"{summary['C1']['retrieval_latency']['mean_seconds']:.6f}s"
    )

    print(
        f"Query embedding mean : "
        f"{summary['query_embedding_latency']['mean_seconds']:.4f}s"
    )

    print()

    print(
        f"Results saved        : "
        f"{RESULTS_FILE}"
    )

    print(
        f"Summary saved        : "
        f"{SUMMARY_FILE}"
    )

    print(
        f"Markdown report      : "
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()