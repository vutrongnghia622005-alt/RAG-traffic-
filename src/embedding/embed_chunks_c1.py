from pathlib import Path
import hashlib
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/processed/chunks_c1.jsonl"
)

OUTPUT_EMBEDDINGS = Path(
    "data/processed/embeddings_c1.npy"
)

OUTPUT_METADATA = Path(
    "data/processed/chunk_metadata_c1.jsonl"
)

OUTPUT_MANIFEST = Path(
    "data/processed/embedding_manifest_c1.json"
)

MODEL_NAME = "BAAI/bge-m3"

DEVICE = "cpu"

BATCH_SIZE = 4

EXPECTED_DIMENSION = 1024

EXPECTED_ARTICLES = 89


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
# FILE HASH
# ============================================================

def sha256_file(path):

    hasher = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            hasher.update(
                block
            )

    return hasher.hexdigest()


# ============================================================
# VALIDATE CHUNKS
# ============================================================

def validate_chunks(chunks):

    if not chunks:

        raise ValueError(
            "No C1 chunks found."
        )

    required_fields = {
        "chunk_id",
        "document_id",
        "chapter",
        "article_number",
        "article_title",
        "clause",
        "piece_index",
        "page_start",
        "page_end",
        "source_start",
        "source_end",
        "char_count",
        "text",
    }

    ids = set()

    articles = set()

    for index, chunk in enumerate(
        chunks
    ):

        missing = (
            required_fields
            - set(chunk.keys())
        )

        if missing:

            raise ValueError(
                f"Chunk {index} "
                f"missing fields: "
                f"{sorted(missing)}"
            )

        chunk_id = (
            chunk["chunk_id"]
        )

        if chunk_id in ids:

            raise ValueError(
                f"Duplicate chunk_id: "
                f"{chunk_id}"
            )

        ids.add(
            chunk_id
        )

        articles.add(
            chunk["article_number"]
        )

        if not (
            chunk["text"]
            .strip()
        ):

            raise ValueError(
                f"Empty text: "
                f"{chunk_id}"
            )

        if (
            chunk["char_count"]
            != len(
                chunk["text"]
            )
        ):

            raise ValueError(
                f"char_count mismatch: "
                f"{chunk_id}"
            )

        if (
            chunk["source_end"]
            <= chunk["source_start"]
        ):

            raise ValueError(
                f"Invalid source range: "
                f"{chunk_id}"
            )

    if (
        len(articles)
        != EXPECTED_ARTICLES
    ):

        raise ValueError(
            "Wrong number of articles: "
            f"{len(articles)}"
        )

    return True


# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

def validate_embeddings(
    embeddings,
    expected_count
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
        != expected_count
    ):

        raise ValueError(
            "Vector count mismatch: "
            f"{embeddings.shape[0]} "
            f"!= {expected_count}"
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
            "Embeddings are not "
            "L2 normalized."
        )

    return norms


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    chunks,
    path
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for vector_index, chunk in enumerate(
            chunks
        ):

            metadata = {
                "vector_index": (
                    vector_index
                ),
                **chunk,
            }

            f.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "C1 BGE-M3 EMBEDDING"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing: {INPUT_FILE}"
        )

    chunks = load_jsonl(
        INPUT_FILE
    )

    print(
        f"Chunks loaded       : "
        f"{len(chunks)}"
    )

    validate_chunks(
        chunks
    )

    print(
        "Chunk validation    : PASS"
    )

    # --------------------------------------------------------
    # Dataset hash
    # --------------------------------------------------------

    dataset_hash = (
        sha256_file(
            INPUT_FILE
        )
    )

    print(
        f"Dataset SHA256      : "
        f"{dataset_hash[:16]}..."
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        f"Model               : "
        f"{MODEL_NAME}"
    )

    print(
        f"Device              : "
        f"{DEVICE}"
    )

    print(
        f"Batch size          : "
        f"{BATCH_SIZE}"
    )

    print()

    print(
        "Loading BGE-M3..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE
    )

    print(
        "Model loaded        : PASS"
    )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    print()

    print(
        "Generating embeddings..."
    )

    start_time = (
        time.perf_counter()
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Validate vectors
    # --------------------------------------------------------

    norms = validate_embeddings(
        embeddings,
        expected_count=len(chunks),
    )

    print(
        "Embedding validation: PASS"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_EMBEDDINGS.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_EMBEDDINGS,
        embeddings
    )

    save_metadata(
        chunks,
        OUTPUT_METADATA
    )

    manifest = {
        "config": "C1_structure_aware",
        "source_file": (
            str(INPUT_FILE)
        ),
        "source_sha256": (
            dataset_hash
        ),
        "model": (
            MODEL_NAME
        ),
        "device": (
            DEVICE
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "normalize_embeddings": True,
        "vector_count": int(
            embeddings.shape[0]
        ),
        "vector_dimension": int(
            embeddings.shape[1]
        ),
        "dtype": (
            str(
                embeddings.dtype
            )
        ),
        "embedding_time_seconds": (
            elapsed
        ),
        "min_l2_norm": float(
            norms.min()
        ),
        "max_l2_norm": float(
            norms.max()
        ),
    }

    with open(
        OUTPUT_MANIFEST,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "C1 EMBEDDING COMPLETED"
    )
    print("=" * 72)

    print(
        f"Total vectors       : "
        f"{embeddings.shape[0]}"
    )

    print(
        f"Vector dimension    : "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Embedding dtype     : "
        f"{embeddings.dtype}"
    )

    print(
        f"Min L2 norm         : "
        f"{norms.min():.8f}"
    )

    print(
        f"Max L2 norm         : "
        f"{norms.max():.8f}"
    )

    print(
        f"NaN detected        : "
        f"{np.isnan(embeddings).any()}"
    )

    print(
        f"Inf detected        : "
        f"{np.isinf(embeddings).any()}"
    )

    print(
        "Normalization       : PASS"
    )

    print(
        f"Embedding time      : "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Embeddings saved    : "
        f"{OUTPUT_EMBEDDINGS}"
    )

    print(
        f"Metadata saved      : "
        f"{OUTPUT_METADATA}"
    )

    print(
        f"Manifest saved      : "
        f"{OUTPUT_MANIFEST}"
    )


if __name__ == "__main__":
    main()