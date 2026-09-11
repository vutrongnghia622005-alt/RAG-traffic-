from pathlib import Path
import json
import time

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

TOP_K = 5

EXPECTED_DIMENSION = 1024


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
                    f"Invalid JSON at "
                    f"{path}:{line_number}: "
                    f"{exc}"
                )

    return items


# ============================================================
# VALIDATE INDEX
# ============================================================

def validate_index(
    embeddings,
    metadata
):

    if embeddings.ndim != 2:

        raise ValueError(
            "Embeddings must be 2D."
        )

    if (
        embeddings.shape[0]
        != len(metadata)
    ):

        raise ValueError(
            "Vector count and metadata "
            "count do not match."
        )

    if (
        embeddings.shape[1]
        != EXPECTED_DIMENSION
    ):

        raise ValueError(
            f"Wrong vector dimension: "
            f"{embeddings.shape[1]}"
        )

    if np.isnan(
        embeddings
    ).any():

        raise ValueError(
            "NaN detected."
        )

    if np.isinf(
        embeddings
    ).any():

        raise ValueError(
            "Inf detected."
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


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    question,
    model,
    embeddings,
    metadata,
    top_k=5
):

    # --------------------------------------------------------
    # Encode question
    # --------------------------------------------------------

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].astype(
        np.float32
    )

    # --------------------------------------------------------
    # Cosine similarity
    #
    # Vì vectors đã normalize:
    #
    # cosine(a,b) = dot(a,b)
    # --------------------------------------------------------

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

        chunk = (
            metadata[
                index
            ]
        )

        results.append(
            {
                "rank": rank,
                "score": float(
                    scores[index]
                ),
                "chunk": chunk,
            }
        )

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    question,
    results,
    elapsed
):

    print()
    print("=" * 90)
    print("C1 STRUCTURE-AWARE RAG TEST")
    print("=" * 90)

    print()
    print("CÂU HỎI:")
    print(question)

    print()
    print(
        f"Retrieval time: "
        f"{elapsed:.4f} seconds"
    )

    print()
    print("=" * 90)
    print("TOP RETRIEVED CONTEXT")
    print("=" * 90)

    for result in results:

        chunk = (
            result["chunk"]
        )

        print()
        print("-" * 90)

        print(
            f"RANK       : "
            f"{result['rank']}"
        )

        print(
            f"SCORE      : "
            f"{result['score']:.6f}"
        )

        print(
            f"CHUNK ID   : "
            f"{chunk['chunk_id']}"
        )

        print(
            f"CHAPTER    : "
            f"{chunk['chapter']}"
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

        print("NỘI DUNG:")

        print()

        print(
            chunk["text"]
        )

    print()
    print("=" * 90)


# ============================================================
# DISPLAY CONTEXT FOR FUTURE LLM
# ============================================================

def display_llm_context(
    question,
    results
):

    print()
    print("=" * 90)
    print("CONTEXT SẼ ĐƯỢC ĐƯA VÀO LLM")
    print("=" * 90)

    print()

    print(
        "Question:"
    )

    print(
        question
    )

    print()

    for result in results:

        chunk = (
            result["chunk"]
        )

        print(
            f"[Nguồn {result['rank']}] "
            f"Điều {chunk['article_number']}, "
            f"Khoản {chunk['clause']}, "
            f"Trang {chunk['page_start']}"
        )

        print(
            chunk["text"]
        )

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("LOADING C1 RETRIEVAL SYSTEM")
    print("=" * 90)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not EMBEDDINGS_FILE.exists():

        raise FileNotFoundError(
            f"Missing: "
            f"{EMBEDDINGS_FILE}"
        )

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            f"Missing: "
            f"{METADATA_FILE}"
        )

    # --------------------------------------------------------
    # Load index
    # --------------------------------------------------------

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    metadata = load_jsonl(
        METADATA_FILE
    )

    print(
        f"Vectors loaded : "
        f"{embeddings.shape}"
    )

    print(
        f"Chunks loaded  : "
        f"{len(metadata)}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_index(
        embeddings,
        metadata
    )

    print(
        "Index validation: PASS"
    )

    # --------------------------------------------------------
    # Load BGE-M3
    # --------------------------------------------------------

    print()
    print(
        f"Loading model: "
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE
    )

    print(
        "Model loaded: PASS"
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print()
    print(
        "Running warm-up..."
    )

    model.encode(
        [
            "giao thông đường bộ"
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    print(
        "Warm-up: PASS"
    )

    # --------------------------------------------------------
    # Interactive loop
    # --------------------------------------------------------

    print()
    print("=" * 90)

    print(
        "HỆ THỐNG ĐÃ SẴN SÀNG"
    )

    print("=" * 90)

    print()

    print(
        "Nhập câu hỏi để test."
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

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        start = (
            time.perf_counter()
        )

        results = retrieve(
            question=question,
            model=model,
            embeddings=embeddings,
            metadata=metadata,
            top_k=TOP_K,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        # ----------------------------------------------------
        # Display retrieval
        # ----------------------------------------------------

        display_results(
            question,
            results,
            elapsed,
        )

        # ----------------------------------------------------
        # Show future LLM context
        # ----------------------------------------------------

        display_llm_context(
            question,
            results,
        )


if __name__ == "__main__":
    main()