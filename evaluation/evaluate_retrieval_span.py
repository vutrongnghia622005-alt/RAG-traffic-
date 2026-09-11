from pathlib import Path
import json
import math
import sys
import time

import numpy as np


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
    "evaluation/gold_questions_span.jsonl"
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
    / "c0_span_results.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "c0_span_summary.json"
)

TOP_K = 5

MIN_OVERLAP_CHARS = 20
MIN_OVERLAP_RATIO = 0.50


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

    pages = sorted(
        pages,
        key=lambda x: x["page"]
    )

    parts = []
    page_offsets = {}

    current_offset = 0

    for page in pages:

        text = page["text"]
        page_number = page["page"]

        page_offsets[
            page_number
        ] = {
            "global_start": current_offset,
            "global_end": (
                current_offset
                + len(text)
            )
        }

        parts.append(text)

        current_offset += len(text)

        # separator giữa hai page
        parts.append("\n")

        current_offset += 1

    return (
        "".join(parts),
        page_offsets
    )


# ============================================================
# BUILD CHUNK SPANS
# ============================================================

def build_chunk_spans(
    metadata,
    page_offsets
):

    spans = {}

    for chunk in metadata:

        page_number = chunk["page"]

        page_start = (
            page_offsets[
                page_number
            ]["global_start"]
        )

        spans[
            chunk["chunk_id"]
        ] = {
            "start": (
                page_start
                + chunk["start_char"]
            ),
            "end": (
                page_start
                + chunk["end_char"]
            ),
        }

    return spans


# ============================================================
# LOCATE GOLD EVIDENCE
# ============================================================

def locate_evidence(
    full_text,
    gold_questions
):
    """
    Tìm evidence anchor trong source.

    Cho phép khác biệt whitespace:
    - space
    - newline
    - nhiều spaces

    Nhưng vẫn yêu cầu mỗi anchor
    xuất hiện đúng 1 lần.
    """

    import re
    import unicodedata

    full_text = unicodedata.normalize(
        "NFC",
        full_text
    )

    all_evidence = {}

    for item in gold_questions:

        question_id = (
            item["question_id"]
        )

        all_evidence[
            question_id
        ] = []

        for index, anchor in enumerate(
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

            # ------------------------------------
            # Flexible whitespace regex
            # ------------------------------------

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

                print()
                print("=" * 70)
                print(
                    "EVIDENCE MATCH ERROR"
                )
                print("=" * 70)

                print(
                    "Question ID:",
                    question_id
                )

                print(
                    "Anchor:",
                    anchor
                )

                print(
                    "Matches found:",
                    len(matches)
                )

                raise ValueError(
                    f"{question_id}: "
                    f"anchor must occur "
                    f"exactly once, "
                    f"found {len(matches)}"
                )

            match = matches[0]

            start = match.start()
            end = match.end()

            evidence_id = (
                f"{question_id}_e"
                f"{index:02d}"
            )

            all_evidence[
                question_id
            ].append(
                {
                    "evidence_id": (
                        evidence_id
                    ),
                    "text": anchor,
                    "start": start,
                    "end": end,
                    "length": (
                        end - start
                    ),
                }
            )

    return all_evidence
    """
    Tìm exact evidence anchors trong source.

    Anchor phải xuất hiện đúng 1 lần.
    Nếu 0 hoặc >1 lần thì benchmark dừng.
    """

    all_evidence = {}

    for item in gold_questions:

        question_id = (
            item["question_id"]
        )

        all_evidence[
            question_id
        ] = []

        for index, anchor in enumerate(
            item["evidence_anchors"],
            start=1
        ):

            positions = []

            start_search = 0

            while True:

                position = full_text.find(
                    anchor,
                    start_search
                )

                if position == -1:
                    break

                positions.append(
                    position
                )

                start_search = (
                    position + 1
                )

            if len(positions) != 1:

                raise ValueError(
                    f"{question_id}: "
                    f"anchor must occur exactly once, "
                    f"found {len(positions)}:\n"
                    f"{anchor}"
                )

            start = positions[0]
            end = start + len(anchor)

            evidence_id = (
                f"{question_id}_e"
                f"{index:02d}"
            )

            all_evidence[
                question_id
            ].append(
                {
                    "evidence_id": evidence_id,
                    "text": anchor,
                    "start": start,
                    "end": end,
                    "length": len(anchor),
                }
            )

    return all_evidence


# ============================================================
# OVERLAP
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
# MAP EVIDENCE -> CHUNKS
# ============================================================

def map_evidence_to_chunks(
    evidence_units,
    chunk_spans
):
    """
    Một chunk được xem là chứa evidence nếu:
    - overlap ít nhất 20 ký tự
    - và overlap ít nhất 50% anchor
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
            )
        )

        matched_chunks = set()

        for chunk_id, chunk_span in (
            chunk_spans.items()
        ):

            overlap = overlap_size(
                evidence["start"],
                evidence["end"],
                chunk_span["start"],
                chunk_span["end"],
            )

            if overlap >= required_overlap:

                matched_chunks.add(
                    chunk_id
                )

        if not matched_chunks:

            raise ValueError(
                "No chunk mapped to "
                f"{evidence_id}"
            )

        evidence_chunk_map[
            evidence_id
        ] = matched_chunks

    return evidence_chunk_map


# ============================================================
# STANDARD METRICS
# ============================================================

def precision_at_k(
    retrieved,
    relevant,
    k
):

    hits = sum(
        chunk_id in relevant
        for chunk_id in retrieved[:k]
    )

    return hits / k


def recall_at_k(
    retrieved,
    relevant,
    k
):

    if not relevant:
        return 0.0

    hits = sum(
        chunk_id in relevant
        for chunk_id in retrieved[:k]
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

    return float(
        any(
            chunk_id in relevant
            for chunk_id in retrieved[:k]
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
# EVIDENCE RECALL
# ============================================================

def evidence_recall_at_k(
    retrieved,
    evidence_chunk_map,
    k
):
    """
    Mỗi evidence anchor = 1 evidence unit.

    Unit được coi là retrieved nếu ít nhất
    một chunk trong Top-K chứa evidence đó.
    """

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
        / len(evidence_chunk_map)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("C0 SPAN-LEVEL RETRIEVAL BENCHMARK")
    print("=" * 72)

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
    # Build source coordinate system
    # --------------------------------------------------------

    full_text, page_offsets = (
        build_global_document(
            pages
        )
    )

    chunk_spans = (
        build_chunk_spans(
            metadata,
            page_offsets
        )
    )

    all_evidence = locate_evidence(
        full_text,
        gold_questions
    )

    total_evidence = sum(
        len(items)
        for items in (
            all_evidence.values()
        )
    )

    print(
        f"Evidence units      : "
        f"{total_evidence}"
    )

    print(
        "Evidence validation : PASS"
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

    all_precision = []
    all_recall = []
    all_hit = []
    all_mrr = []
    all_ndcg = []
    all_evidence_recall = []
    all_latency = []

    output_rows = []

    print()
    print("=" * 72)
    print("RUNNING QUESTIONS")
    print("=" * 72)

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

        evidence_units = (
            all_evidence[
                question_id
            ]
        )

        evidence_chunk_map = (
            map_evidence_to_chunks(
                evidence_units,
                chunk_spans,
            )
        )

        relevant_chunks = set()

        for chunk_ids in (
            evidence_chunk_map.values()
        ):

            relevant_chunks.update(
                chunk_ids
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
            relevant_chunks,
            TOP_K,
        )

        recall = recall_at_k(
            retrieved_ids,
            relevant_chunks,
            TOP_K,
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

        evidence_recall = (
            evidence_recall_at_k(
                retrieved_ids,
                evidence_chunk_map,
                TOP_K,
            )
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

        all_evidence_recall.append(
            evidence_recall
        )

        all_latency.append(
            elapsed
        )

        retrieved_detail = []

        for result in results:

            covered_evidence = []

            for evidence_id, chunk_ids in (
                evidence_chunk_map.items()
            ):

                if (
                    result["chunk_id"]
                    in chunk_ids
                ):

                    covered_evidence.append(
                        evidence_id
                    )

            retrieved_detail.append(
                {
                    "rank": result["rank"],
                    "chunk_id": result["chunk_id"],
                    "page": result["page"],
                    "score": result["score"],
                    "relevant": (
                        result["chunk_id"]
                        in relevant_chunks
                    ),
                    "evidence_ids": (
                        covered_evidence
                    ),
                }
            )

        output_rows.append(
            {
                "question_id": question_id,
                "question": question,
                "evidence_units": (
                    evidence_units
                ),
                "relevant_chunk_ids": sorted(
                    relevant_chunks
                ),
                "retrieved": (
                    retrieved_detail
                ),
                "metrics": {
                    "precision@5": precision,
                    "recall@5": recall,
                    "hit@5": hit,
                    "mrr": mrr,
                    "ndcg@5": ndcg,
                    "evidence_recall@5": (
                        evidence_recall
                    ),
                },
                "latency_seconds": (
                    elapsed
                ),
            }
        )

        print(
            f"[{index:02d}/"
            f"{len(gold_questions):02d}] "
            f"{question_id} "
            f"| P@5={precision:.2f} "
            f"| R@5={recall:.2f} "
            f"| ER@5={evidence_recall:.2f} "
            f"| Hit={hit:.0f} "
            f"| MRR={mrr:.2f} "
            f"| nDCG={ndcg:.2f}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "config": "C0",
        "label_type": "evidence_span",
        "retrieval": "dense_bge_m3",
        "top_k": TOP_K,
        "questions": len(
            gold_questions
        ),
        "evidence_units": (
            total_evidence
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
            "evidence_recall@5": float(
                np.mean(
                    all_evidence_recall
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

    print()
    print("=" * 72)
    print("C0 SPAN BENCHMARK SUMMARY")
    print("=" * 72)

    print(
        f"Questions         : "
        f"{summary['questions']}"
    )

    print(
        f"Evidence units    : "
        f"{summary['evidence_units']}"
    )

    print(
        f"Precision@5       : "
        f"{summary['metrics']['precision@5']:.4f}"
    )

    print(
        f"Recall@5          : "
        f"{summary['metrics']['recall@5']:.4f}"
    )

    print(
        f"Evidence Recall@5 : "
        f"{summary['metrics']['evidence_recall@5']:.4f}"
    )

    print(
        f"Hit@5             : "
        f"{summary['metrics']['hit@5']:.4f}"
    )

    print(
        f"MRR               : "
        f"{summary['metrics']['mrr']:.4f}"
    )

    print(
        f"nDCG@5            : "
        f"{summary['metrics']['ndcg@5']:.4f}"
    )

    print()
    print(
        f"Mean latency      : "
        f"{summary['latency']['mean_seconds']:.4f}s"
    )

    print(
        f"p50 latency       : "
        f"{summary['latency']['p50_seconds']:.4f}s"
    )

    print(
        f"p95 latency       : "
        f"{summary['latency']['p95_seconds']:.4f}s"
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