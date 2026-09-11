from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/processed/chunks_c0.jsonl"
)

OUTPUT_EMBEDDINGS = Path(
    "data/processed/embeddings_c0.npy"
)

OUTPUT_METADATA = Path(
    "data/processed/chunk_metadata_c0.jsonl"
)

OUTPUT_MANIFEST = Path(
    "data/processed/embedding_manifest_c0.json"
)

MODEL_NAME = "BAAI/bge-m3"

DEVICE = "cpu"

# Máy hiện tại chạy CPU.
# Batch 4 an toàn hơn cho BGE-M3.
BATCH_SIZE = 4

EXPECTED_CHUNKS = 412

EXPECTED_DIMENSION = 1024


# ============================================================
# LOAD JSONL
# ============================================================

def load_chunks(path: Path):
    """
    Đọc toàn bộ chunks_c0.jsonl.
    """

    chunks = []

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
                chunk = json.loads(line)

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {e}"
                )

            chunks.append(chunk)

    return chunks


# ============================================================
# HASH SOURCE DATA
# ============================================================

def calculate_sha256(path: Path):
    """
    SHA256 giúp xác định embeddings được tạo
    từ đúng phiên bản chunks nào.
    """

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


# ============================================================
# VALIDATE CHUNKS
# ============================================================

def validate_chunks(chunks):
    """
    Kiểm tra lại dữ liệu ngay trước embedding.
    """

    errors = []

    required_fields = {
        "chunk_id",
        "document_id",
        "page",
        "chunk_index",
        "start_char",
        "end_char",
        "char_count",
        "text",
    }

    seen_ids = set()

    for chunk in chunks:

        chunk_id = chunk.get(
            "chunk_id",
            "UNKNOWN"
        )

        # ----------------------------------------
        # Required fields
        # ----------------------------------------

        missing_fields = (
            required_fields
            - set(chunk.keys())
        )

        if missing_fields:

            errors.append(
                f"{chunk_id}: missing fields "
                f"{sorted(missing_fields)}"
            )

            continue

        # ----------------------------------------
        # Duplicate ID
        # ----------------------------------------

        if chunk_id in seen_ids:

            errors.append(
                f"Duplicate chunk_id: "
                f"{chunk_id}"
            )

        seen_ids.add(chunk_id)

        # ----------------------------------------
        # Empty text
        # ----------------------------------------

        if not chunk["text"].strip():

            errors.append(
                f"{chunk_id}: empty text"
            )

        # ----------------------------------------
        # char_count
        # ----------------------------------------

        actual_length = len(
            chunk["text"]
        )

        if (
            chunk["char_count"]
            != actual_length
        ):

            errors.append(
                f"{chunk_id}: "
                f"char_count mismatch"
            )

        # ----------------------------------------
        # Offset length
        # ----------------------------------------

        offset_length = (
            chunk["end_char"]
            - chunk["start_char"]
        )

        if offset_length != actual_length:

            errors.append(
                f"{chunk_id}: "
                f"offset mismatch"
            )

    return errors


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    chunks,
    output_path: Path
):
    """
    Metadata được lưu đúng thứ tự với embeddings.

    embeddings[0]
        <->
    metadata vector_index = 0
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        for vector_index, chunk in enumerate(
            chunks
        ):

            metadata = {
                "vector_index": vector_index,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "char_count": chunk["char_count"],
                "text": chunk["text"],
            }

            f.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

def validate_embeddings(
    embeddings,
    expected_count
):
    """
    Kiểm tra vector sau embedding.
    """

    errors = []

    # ----------------------------------------
    # Must be 2D
    # ----------------------------------------

    if embeddings.ndim != 2:

        errors.append(
            f"Expected 2D embeddings, "
            f"got shape {embeddings.shape}"
        )

        return errors, None

    # ----------------------------------------
    # Number of vectors
    # ----------------------------------------

    if (
        embeddings.shape[0]
        != expected_count
    ):

        errors.append(
            "Vector count mismatch: "
            f"{embeddings.shape[0]} "
            f"!= {expected_count}"
        )

    # ----------------------------------------
    # Dimension
    # ----------------------------------------

    if (
        embeddings.shape[1]
        != EXPECTED_DIMENSION
    ):

        errors.append(
            "Unexpected vector dimension: "
            f"{embeddings.shape[1]} "
            f"!= {EXPECTED_DIMENSION}"
        )

    # ----------------------------------------
    # NaN
    # ----------------------------------------

    if np.isnan(
        embeddings
    ).any():

        errors.append(
            "Embeddings contain NaN"
        )

    # ----------------------------------------
    # Inf
    # ----------------------------------------

    if np.isinf(
        embeddings
    ).any():

        errors.append(
            "Embeddings contain Inf"
        )

    # ----------------------------------------
    # L2 norm
    # ----------------------------------------

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-3
    ):

        errors.append(
            "Embeddings are not "
            "properly L2-normalized"
        )

    return errors, norms


# ============================================================
# SAVE MANIFEST
# ============================================================

def save_manifest(
    chunks,
    embeddings,
    source_hash,
    elapsed_seconds,
    norms
):
    """
    Lưu thông tin cấu hình embedding
    phục vụ benchmark reproducibility.
    """

    manifest = {
        "benchmark": "C0",
        "source": {
            "chunk_file": str(
                INPUT_FILE
            ),
            "chunk_file_sha256": (
                source_hash
            ),
            "total_chunks": len(
                chunks
            ),
        },
        "embedding": {
            "model": MODEL_NAME,
            "type": "dense",
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
            "normalize_embeddings": True,
            "similarity": "cosine",
            "dimension": int(
                embeddings.shape[1]
            ),
            "dtype": str(
                embeddings.dtype
            ),
        },
        "validation": {
            "min_l2_norm": float(
                norms.min()
            ),
            "max_l2_norm": float(
                norms.max()
            ),
            "nan_found": bool(
                np.isnan(
                    embeddings
                ).any()
            ),
            "inf_found": bool(
                np.isinf(
                    embeddings
                ).any()
            ),
            "status": "PASS",
        },
        "runtime": {
            "embedding_seconds": (
                round(
                    elapsed_seconds,
                    2
                )
            ),
        },
        "outputs": {
            "embeddings": str(
                OUTPUT_EMBEDDINGS
            ),
            "metadata": str(
                OUTPUT_METADATA
            ),
        },
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("C0 BGE-M3 EMBEDDING")
    print("=" * 65)

    # ========================================================
    # INPUT CHECK
    # ========================================================

    if not INPUT_FILE.exists():

        print(
            "ERROR: input file "
            f"not found: {INPUT_FILE}"
        )

        sys.exit(1)

    # ========================================================
    # LOAD CHUNKS
    # ========================================================

    chunks = load_chunks(
        INPUT_FILE
    )

    print(
        f"Chunks loaded       : "
        f"{len(chunks)}"
    )

    # ========================================================
    # EXPECTED NUMBER
    # ========================================================

    if (
        len(chunks)
        != EXPECTED_CHUNKS
    ):

        print(
            "ERROR: expected "
            f"{EXPECTED_CHUNKS} chunks "
            f"but found {len(chunks)}"
        )

        sys.exit(1)

    # ========================================================
    # VALIDATE CHUNKS
    # ========================================================

    chunk_errors = (
        validate_chunks(
            chunks
        )
    )

    if chunk_errors:

        print(
            f"Chunk errors        : "
            f"{len(chunk_errors)}"
        )

        for error in (
            chunk_errors[:10]
        ):
            print(
                " -",
                error
            )

        sys.exit(1)

    print(
        "Chunk validation    : PASS"
    )

    # ========================================================
    # SOURCE HASH
    # ========================================================

    source_hash = (
        calculate_sha256(
            INPUT_FILE
        )
    )

    print(
        "Dataset SHA256      : "
        f"{source_hash[:16]}..."
    )

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

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print(
        "Loading BGE-M3 model..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE
    )

    print(
        "Model loaded        : PASS"
    )

    # ========================================================
    # PREPARE TEXT
    # ========================================================

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # ========================================================
    # EMBEDDING
    # ========================================================

    print()
    print(
        "Generating embeddings..."
    )

    start_time = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed_seconds = (
        time.time()
        - start_time
    )

    # ========================================================
    # FORCE FLOAT32
    # ========================================================

    embeddings = (
        embeddings.astype(
            np.float32
        )
    )

    # ========================================================
    # VALIDATE EMBEDDINGS
    # ========================================================

    embedding_errors, norms = (
        validate_embeddings(
            embeddings,
            len(chunks)
        )
    )

    if embedding_errors:

        print()
        print(
            "Embedding validation: FAIL"
        )

        for error in (
            embedding_errors
        ):
            print(
                " -",
                error
            )

        sys.exit(1)

    print()
    print(
        "Embedding validation: PASS"
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_EMBEDDINGS.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE EMBEDDINGS
    # ========================================================

    np.save(
        OUTPUT_EMBEDDINGS,
        embeddings
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    save_metadata(
        chunks,
        OUTPUT_METADATA
    )

    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    save_manifest(
        chunks=chunks,
        embeddings=embeddings,
        source_hash=source_hash,
        elapsed_seconds=elapsed_seconds,
        norms=norms,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 65)
    print("EMBEDDING COMPLETED")
    print("=" * 65)

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
        "NaN detected        : "
        f"{np.isnan(embeddings).any()}"
    )

    print(
        "Inf detected        : "
        f"{np.isinf(embeddings).any()}"
    )

    print(
        "Normalization       : PASS"
    )

    print(
        f"Embedding time      : "
        f"{elapsed_seconds:.2f} seconds"
    )

    print()
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