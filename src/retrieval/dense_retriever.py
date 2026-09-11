from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

EMBEDDINGS_FILE = Path(
    "data/processed/embeddings_c0.npy"
)

METADATA_FILE = Path(
    "data/processed/chunk_metadata_c0.jsonl"
)

MODEL_NAME = "BAAI/bge-m3"

DEVICE = "cpu"

DEFAULT_TOP_K = 5

EXPECTED_DIMENSION = 1024


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata(path: Path):
    """
    Đọc metadata theo đúng thứ tự vector_index.
    """

    metadata = []

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
                item = json.loads(line)

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {e}"
                )

            metadata.append(item)

    return metadata


# ============================================================
# VALIDATE INDEX
# ============================================================

def validate_index(
    embeddings,
    metadata
):
    """
    Kiểm tra embeddings và metadata
    trước khi retrieval.
    """

    errors = []

    # ----------------------------------------
    # Embedding shape
    # ----------------------------------------

    if embeddings.ndim != 2:

        errors.append(
            f"Embeddings must be 2D, "
            f"got {embeddings.shape}"
        )

        return errors

    # ----------------------------------------
    # Vector count
    # ----------------------------------------

    if (
        embeddings.shape[0]
        != len(metadata)
    ):

        errors.append(
            "Embeddings / metadata count "
            "mismatch: "
            f"{embeddings.shape[0]} "
            f"!= {len(metadata)}"
        )

    # ----------------------------------------
    # Dimension
    # ----------------------------------------

    if (
        embeddings.shape[1]
        != EXPECTED_DIMENSION
    ):

        errors.append(
            f"Unexpected dimension: "
            f"{embeddings.shape[1]}"
        )

    # ----------------------------------------
    # NaN / Inf
    # ----------------------------------------

    if np.isnan(
        embeddings
    ).any():

        errors.append(
            "Embeddings contain NaN"
        )

    if np.isinf(
        embeddings
    ).any():

        errors.append(
            "Embeddings contain Inf"
        )

    # ----------------------------------------
    # Metadata vector_index
    # ----------------------------------------

    for i, item in enumerate(
        metadata
    ):

        if (
            item.get("vector_index")
            != i
        ):

            errors.append(
                "Metadata vector_index "
                f"mismatch at position {i}"
            )

            break

    return errors


# ============================================================
# DENSE RETRIEVER
# ============================================================

class DenseRetriever:

    def __init__(
        self,
        embeddings_path: Path,
        metadata_path: Path,
        model_name: str = MODEL_NAME,
        device: str = DEVICE,
    ):

        print("=" * 65)
        print("C0 DENSE RETRIEVER")
        print("=" * 65)

        # ----------------------------------------
        # File check
        # ----------------------------------------

        if not embeddings_path.exists():

            raise FileNotFoundError(
                f"Embeddings not found: "
                f"{embeddings_path}"
            )

        if not metadata_path.exists():

            raise FileNotFoundError(
                f"Metadata not found: "
                f"{metadata_path}"
            )

        # ----------------------------------------
        # Load embeddings
        # ----------------------------------------

        print(
            "Loading embeddings..."
        )

        self.embeddings = np.load(
            embeddings_path
        )

        # ----------------------------------------
        # Load metadata
        # ----------------------------------------

        self.metadata = load_metadata(
            metadata_path
        )

        print(
            f"Vectors loaded      : "
            f"{self.embeddings.shape[0]}"
        )

        print(
            f"Vector dimension    : "
            f"{self.embeddings.shape[1]}"
        )

        print(
            f"Metadata loaded     : "
            f"{len(self.metadata)}"
        )

        # ----------------------------------------
        # Validate index
        # ----------------------------------------

        errors = validate_index(
            self.embeddings,
            self.metadata
        )

        if errors:

            print()
            print("INDEX VALIDATION: FAIL")

            for error in errors:
                print(" -", error)

            sys.exit(1)

        print(
            "Index validation    : PASS"
        )

        # ----------------------------------------
        # Load embedding model
        # ----------------------------------------

        print()
        print(
            "Loading BGE-M3..."
        )

        self.model = SentenceTransformer(
            model_name,
            device=device
        )

        print(
            "Model loaded        : PASS"
        )


    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def embed_query(
        self,
        query: str
    ):
        """
        Embed query bằng cùng model
        đã dùng để embed documents.
        """

        query_embedding = (
            self.model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
        )

        return query_embedding.astype(
            np.float32
        )


    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ):
        """
        Dense cosine retrieval.

        Vì document embeddings và query embedding
        đều được L2 normalize:

            cosine_similarity
            =
            dot product
        """

        if not query.strip():

            raise ValueError(
                "Query cannot be empty"
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be > 0"
            )

        top_k = min(
            top_k,
            len(self.metadata)
        )

        start_time = time.time()

        # ----------------------------------------
        # Embed query
        # ----------------------------------------

        query_embedding = (
            self.embed_query(
                query
            )
        )

        # ----------------------------------------
        # Cosine similarity
        # ----------------------------------------

        scores = (
            self.embeddings
            @ query_embedding
        )

        # ----------------------------------------
        # Top K
        # ----------------------------------------

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            item = self.metadata[
                int(index)
            ]

            result = {
                "rank": rank,
                "score": float(
                    scores[index]
                ),
                "vector_index": int(
                    index
                ),
                "chunk_id": item[
                    "chunk_id"
                ],
                "document_id": item[
                    "document_id"
                ],
                "page": item[
                    "page"
                ],
                "chunk_index": item[
                    "chunk_index"
                ],
                "text": item[
                    "text"
                ],
            }

            results.append(
                result
            )

        elapsed = (
            time.time()
            - start_time
        )

        return results, elapsed


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    query,
    results,
    elapsed
):

    print()
    print("=" * 80)
    print("QUERY")
    print("=" * 80)

    print(query)

    print()
    print(
        f"Retrieval time: "
        f"{elapsed:.4f} seconds"
    )

    print(
        f"Top-K          : "
        f"{len(results)}"
    )

    for result in results:

        print()
        print("=" * 80)

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
            f"{result['chunk_id']}"
        )

        print(
            f"PAGE       : "
            f"{result['page']}"
        )

        print(
            f"CHUNK INDEX: "
            f"{result['chunk_index']}"
        )

        print("-" * 80)

        print(
            result["text"]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "C0 Dense Retriever "
            "using BGE-M3"
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        required=False,
        help="User query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved chunks",
    )

    args = parser.parse_args()

    retriever = DenseRetriever(
        embeddings_path=EMBEDDINGS_FILE,
        metadata_path=METADATA_FILE,
    )

    # ----------------------------------------
    # Query from CLI
    # ----------------------------------------

    if args.query:

        query = args.query

    else:

        print()
        query = input(
            "Enter query: "
        )

    # ----------------------------------------
    # Search
    # ----------------------------------------

    results, elapsed = (
        retriever.search(
            query=query,
            top_k=args.top_k,
        )
    )

    # ----------------------------------------
    # Print
    # ----------------------------------------

    print_results(
        query=query,
        results=results,
        elapsed=elapsed,
    )


if __name__ == "__main__":
    main()