from pathlib import Path
from collections import defaultdict
import json
import math
import re
import sys
import time

import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.retrieval.dense_retriever import DenseRetriever


# ============================================================
# CONFIG
# ============================================================

GOLD_FILE = Path(
    "evaluation/gold_questions.jsonl"
)

PAGES_FILE = Path(
    "data/interim/pages_clean_c0.jsonl"
)

METADATA_FILE = Path(
    "data/processed/chunk_metadata_c0.jsonl"
)

EMBEDDINGS_FILE = Path(
    "data/processed/embeddings_c0.npy"
)

OUTPUT_DIR = Path(
    "results/evaluation"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "c0_retrieval_results.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "c0_retrieval_summary.json"
)

TOP_K = 5

# Chunk được coi là liên quan nếu overlap
# ít nhất 30 ký tự với span của Điều gold.
MIN_OVERLAP_CHARS = 30


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
# BUILD GLOBAL DOCUMENT
# ============================================================

def build_global_document(pages):
    """
    Ghép toàn bộ 74 trang thành một document logic.

    Đồng thời lưu global offset của từng page.

    Không thay đổi nội dung page.
    """

    pages = sorted(
        pages,
        key=lambda x: x["page"]
    )

    parts = []

    page_offsets = {}

    current_offset = 0

    for page in pages:

        page_number = page["page"]
        text = page["text"]

        page_offsets[page_number] = {
            "global_start": current_offset,
            "global_end": (
                current_offset
                + len(text)
            ),
            "length": len(text),
        }

        parts.append(text)

        current_offset += len(text)

        # Separator giữa hai page
        parts.append("\n")

        current_offset += 1

    full_text = "".join(parts)

    return full_text, page_offsets


# ============================================================
# EXTRACT ARTICLE SPANS
# ============================================================

def extract_article_spans(full_text):
    """
    Tìm các heading dạng:

        Điều 1.
        Điều 12.
        Điều 56.

    Gold label dựa trên span của Điều,
    không phụ thuộc chunk_id của C0.
    """

    pattern = re.compile(
        r"(?m)^Điều\s+(\d+)[a-zA-Z]?\."
    )

    matches = list(
        pattern.finditer(
            full_text
        )
    )

    articles = {}

    for i, match in enumerate(
        matches
    ):

        article_number = int(
            match.group(1)
        )

        start = match.start()

        if i + 1 < len(matches):
            end = matches[
                i + 1
            ].start()
        else:
            end = len(
                full_text
            )

        articles[
            article_number
        ] = {
            "start": start,
            "end": end,
        }

    return articles


# ============================================================
# BUILD CHUNK GLOBAL SPANS
# ============================================================

def build_chunk_spans(
    metadata,
    page_offsets
):
    """
    Chuyển offset local trong page
    thành offset global document.
    """

    chunk_spans = {}

    for chunk in metadata:

        page = chunk["page"]

        page_global_start = (
            page_offsets[
                page
            ]["global_start"]
        )

        global_start = (
            page_global_start
            + chunk["start_char"]
        )

        global_end = (
            page_global_start
            + chunk["end_char"]
        )

        chunk_spans[
            chunk["chunk_id"]
        ] = {
            "start": global_start,
            "end": global_end,
        }

    return chunk_spans


# ============================================================
# INTERSECTION SIZE
# ============================================================

def overlap_size(
    start_a,
    end_a,
    start_b,
    end_b
):

    return max(
        0,
        min(end_a, end_b)
        - max(start_a, start_b)
    )


# ============================================================
# MAP GOLD ARTICLES -> GOLD CHUNKS
# ============================================================

def get_gold_chunk_ids(
    gold_articles,
    articles,
    chunk_spans,
):
    """
    Map article-level labels sang chunk IDs
    của cấu hình hiện tại.

    Đây là điểm quan trọng:
    gold_articles không đổi khi sang C1.
    """

    gold_chunks = set()

    for article_number in gold_articles:

        if article_number not in articles:

            raise ValueError(
                f"Article {article_number} "
                f"not found in clean dataset"
            )

        article_span = articles[
            article_number
        ]

        for chunk_id, chunk_span in (
            chunk_spans.items()
        ):

            size = overlap_size(
                article_span["start"],
                article_span["end"],
                chunk_span["start"],
                chunk_span["end"],
            )

            if (
                size
                >= MIN_OVERLAP_CHARS
            ):
                gold_chunks.add(
                    chunk_id
                )

    return gold_chunks


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    retrieved,
    relevant,
    k
):

    top_k = retrieved[:k]

    hits = sum(
        chunk_id in relevant
        for chunk_id in top_k
    )

    return hits / k


def recall_at_k(
    retrieved,
    relevant,
    k
):

    if not relevant:
        return 0.0

    top_k = retrieved[:k]

    hits = sum(
        chunk_id in relevant
        for chunk_id in top_k
    )

    return (
        hits
        / len(relevant)
    )


def hit_at_k(
    retrieved,
    relevant,
    k
):

    top_k = retrieved[:k]

    return float(
        any(
            chunk_id in relevant
            for chunk_id in top_k
        )
    )


def reciprocal_rank(
    retrieved,
    relevant
):

    for rank, chunk_id in enumerate(
        retrieved,
        start=1
    ):

        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved,
    relevant,
    k
):
    """
    Binary relevance nDCG@k.
    """

    dcg = 0.0

    for rank, chunk_id in enumerate(
        retrieved[:k],
        start=1
    ):

        rel = (
            1.0
            if chunk_id in relevant
            else 0.0
        )

        dcg += (
            rel
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

    return dcg / idcg


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("C0 RETRIEVAL BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    gold_questions = load_jsonl(
        GOLD_FILE
    )

    pages = load_jsonl(
        PAGES_FILE
    )

    metadata = load_jsonl(
        METADATA_FILE
    )

    print(
        f"Gold questions      : "
        f"{len(gold_questions)}"
    )

    print(
        f"Clean pages         : "
        f"{len(pages)}"
    )

    print(
        f"Indexed chunks      : "
        f"{len(metadata)}"
    )

    # --------------------------------------------------------
    # Global spans
    # --------------------------------------------------------

    full_text, page_offsets = (
        build_global_document(
            pages
        )
    )

    articles = (
        extract_article_spans(
            full_text
        )
    )

    print(
        f"Articles detected   : "
        f"{len(articles)}"
    )

    print(
        f"Article range       : "
        f"{min(articles)}"
        f" -> "
        f"{max(articles)}"
    )

    chunk_spans = (
        build_chunk_spans(
            metadata,
            page_offsets
        )
    )

    # --------------------------------------------------------
    # Validate gold labels
    # --------------------------------------------------------

    for item in gold_questions:

        for article in (
            item["gold_articles"]
        ):

            if article not in articles:

                raise ValueError(
                    f"{item['question_id']}: "
                    f"Article {article} "
                    f"not detected"
                )

    print(
        "Gold validation     : PASS"
    )

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    print()

    retriever = DenseRetriever(
        embeddings_path=(
            EMBEDDINGS_FILE
        ),
        metadata_path=(
            METADATA_FILE
        ),
    )

    # --------------------------------------------------------
    # Metrics containers
    # --------------------------------------------------------

    all_precision = []
    all_recall = []
    all_hit = []
    all_mrr = []
    all_ndcg = []
    all_latency = []

    output_rows = []

    print()
    print("=" * 70)
    print("RUNNING QUESTIONS")
    print("=" * 70)

    # --------------------------------------------------------
    # Evaluate each question
    # --------------------------------------------------------

    for index, item in enumerate(
        gold_questions,
        start=1
    ):

        question_id = (
            item["question_id"]
        )

        question = (
            item["question"]
        )

        gold_articles = (
            item["gold_articles"]
        )

        gold_chunks = (
            get_gold_chunk_ids(
                gold_articles,
                articles,
                chunk_spans,
            )
        )

        results, elapsed = (
            retriever.search(
                query=question,
                top_k=TOP_K,
            )
        )

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        precision = precision_at_k(
            retrieved_ids,
            gold_chunks,
            TOP_K,
        )

        recall = recall_at_k(
            retrieved_ids,
            gold_chunks,
            TOP_K,
        )

        hit = hit_at_k(
            retrieved_ids,
            gold_chunks,
            TOP_K,
        )

        mrr = reciprocal_rank(
            retrieved_ids,
            gold_chunks,
        )

        ndcg = ndcg_at_k(
            retrieved_ids,
            gold_chunks,
            TOP_K,
        )

        all_precision.append(
            precision
        )

        all_recall.append(
            recall
        )

        all_hit.append(
            hit
        )

        all_mrr.append(
            mrr
        )

        all_ndcg.append(
            ndcg
        )

        all_latency.append(
            elapsed
        )

        retrieved_detail = []

        for result in results:

            retrieved_detail.append(
                {
                    "rank": result["rank"],
                    "chunk_id": result[
                        "chunk_id"
                    ],
                    "page": result["page"],
                    "score": result["score"],
                    "relevant": (
                        result["chunk_id"]
                        in gold_chunks
                    ),
                }
            )

        row = {
            "question_id": question_id,
            "question": question,
            "gold_articles": (
                gold_articles
            ),
            "gold_chunk_count": (
                len(gold_chunks)
            ),
            "gold_chunk_ids": sorted(
                gold_chunks
            ),
            "retrieved": (
                retrieved_detail
            ),
            "metrics": {
                "precision@5": (
                    precision
                ),
                "recall@5": recall,
                "hit@5": hit,
                "mrr": mrr,
                "ndcg@5": ndcg,
            },
            "latency_seconds": (
                elapsed
            ),
        }

        output_rows.append(
            row
        )

        print(
            f"[{index:02d}/"
            f"{len(gold_questions):02d}] "
            f"{question_id} "
            f"| P@5={precision:.2f} "
            f"| R@5={recall:.2f} "
            f"| Hit={hit:.0f} "
            f"| MRR={mrr:.2f} "
            f"| nDCG={ndcg:.2f}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "config": "C0",
        "retrieval": (
            "dense_bge_m3"
        ),
        "top_k": TOP_K,
        "questions": len(
            gold_questions
        ),
        "metrics": {
            "precision@5": float(
                np.mean(
                    all_precision
                )
            ),
            "recall@5": float(
                np.mean(
                    all_recall
                )
            ),
            "hit@5": float(
                np.mean(
                    all_hit
                )
            ),
            "mrr": float(
                np.mean(
                    all_mrr
                )
            ),
            "ndcg@5": float(
                np.mean(
                    all_ndcg
                )
            ),
        },
        "latency": {
            "mean_seconds": float(
                np.mean(
                    all_latency
                )
            ),
            "p50_seconds": float(
                np.percentile(
                    all_latency,
                    50
                )
            ),
            "p95_seconds": float(
                np.percentile(
                    all_latency,
                    95
                )
            ),
        },
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("C0 BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        f"Questions       : "
        f"{summary['questions']}"
    )

    print(
        f"Precision@5     : "
        f"{summary['metrics']['precision@5']:.4f}"
    )

    print(
        f"Recall@5        : "
        f"{summary['metrics']['recall@5']:.4f}"
    )

    print(
        f"Hit@5           : "
        f"{summary['metrics']['hit@5']:.4f}"
    )

    print(
        f"MRR             : "
        f"{summary['metrics']['mrr']:.4f}"
    )

    print(
        f"nDCG@5          : "
        f"{summary['metrics']['ndcg@5']:.4f}"
    )

    print()
    print(
        f"Mean latency    : "
        f"{summary['latency']['mean_seconds']:.4f}s"
    )

    print(
        f"p50 latency     : "
        f"{summary['latency']['p50_seconds']:.4f}s"
    )

    print(
        f"p95 latency     : "
        f"{summary['latency']['p95_seconds']:.4f}s"
    )

    print()
    print(
        f"Results saved   : "
        f"{RESULTS_FILE}"
    )

    print(
        f"Summary saved   : "
        f"{SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()