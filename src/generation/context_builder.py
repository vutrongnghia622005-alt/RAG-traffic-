# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_labeled_context(
    reranked_results,
    max_sources=5,
):
    """
    Chuyển Top-K retrieval results thành:

        [S1]
        Điều: ...
        Khoản: ...
        Trang: ...
        Nội dung: ...

    Generator chỉ nhìn thấy các source đã được label.
    """

    selected = (
        reranked_results[
            :max_sources
        ]
    )

    blocks = []

    sources = []

    for index, item in enumerate(
        selected,
        start=1,
    ):

        source_id = (
            f"S{index}"
        )

        metadata = (
            item.get(
                "metadata",
                {}
            )
        )

        article = (
            metadata.get(
                "article_number"
            )
        )

        clause = (
            metadata.get(
                "clause"
            )
        )

        page_start = (
            metadata.get(
                "page_start"
            )
        )

        page_end = (
            metadata.get(
                "page_end"
            )
        )

        text = (
            metadata.get(
                "text",
                ""
            ).strip()
        )

        block = (
            f"[{source_id}]\n"
            f"Điều: {article}\n"
            f"Khoản: {clause}\n"
            f"Trang: {page_start}-{page_end}\n"
            f"Chunk ID: {item.get('chunk_id')}\n"
            f"Nội dung:\n{text}"
        )

        blocks.append(
            block
        )

        sources.append(
            {
                "source_id": (
                    source_id
                ),

                "chunk_id": (
                    item.get(
                        "chunk_id"
                    )
                ),

                "article_number": (
                    article
                ),

                "clause": (
                    clause
                ),

                "page_start": (
                    page_start
                ),

                "page_end": (
                    page_end
                ),

                "rerank_score": (
                    item.get(
                        "rerank_score"
                    )
                ),
            }
        )

    context = (
        "\n\n"
        "==============================\n\n"
        .join(
            blocks
        )
    )

    return (
        context,
        sources,
    )