from pathlib import Path
import json
from collections import Counter, defaultdict


INPUT_FILE = Path(
    "data/processed/chunks_c1.jsonl"
)


def load_chunks(path):
    chunks = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                chunks.append(
                    json.loads(line)
                )

    return chunks


def print_article(
    chunks,
    article_number
):
    selected = [
        chunk
        for chunk in chunks
        if chunk["article_number"]
        == article_number
    ]

    print()
    print("=" * 90)
    print(
        f"ARTICLE {article_number}"
    )
    print("=" * 90)

    print(
        "Total chunks:",
        len(selected)
    )

    for chunk in selected:

        print()
        print(
            "Chunk ID   :",
            chunk["chunk_id"]
        )

        print(
            "Clause     :",
            chunk["clause"]
        )

        print(
            "Piece      :",
            chunk["piece_index"]
        )

        print(
            "Pages      :",
            chunk["page_start"],
            "->",
            chunk["page_end"]
        )

        print(
            "Size       :",
            chunk["char_count"]
        )

        print(
            "Title      :",
            chunk["article_title"]
        )

        print("-" * 90)

        print(
            chunk["text"][:1000]
        )


def inspect_intro_chunks(chunks):

    intro_chunks = [
        chunk
        for chunk in chunks
        if chunk["clause"] == "intro"
    ]

    print()
    print("=" * 90)
    print("INTRO CHUNKS")
    print("=" * 90)

    print(
        "Total:",
        len(intro_chunks)
    )

    for chunk in intro_chunks:

        print()
        print(
            chunk["chunk_id"],
            "| article:",
            chunk["article_number"],
            "| pages:",
            chunk["page_start"],
            "->",
            chunk["page_end"],
            "| size:",
            chunk["char_count"]
        )

        print("-" * 90)
        print(chunk["text"])


def inspect_multi_page(chunks):

    multi_page = [
        chunk
        for chunk in chunks
        if (
            chunk["page_start"]
            != chunk["page_end"]
        )
    ]

    print()
    print("=" * 90)
    print("MULTI-PAGE CHUNK SUMMARY")
    print("=" * 90)

    print(
        "Total:",
        len(multi_page)
    )

    print()

    for chunk in multi_page[:10]:

        print(
            chunk["chunk_id"],
            "| article:",
            chunk["article_number"],
            "| clause:",
            chunk["clause"],
            "| pages:",
            chunk["page_start"],
            "->",
            chunk["page_end"],
            "| size:",
            chunk["char_count"]
        )


def inspect_clause_distribution(chunks):

    article_clauses = defaultdict(list)

    for chunk in chunks:

        article_clauses[
            chunk["article_number"]
        ].append(
            chunk["clause"]
        )

    print()
    print("=" * 90)
    print("ARTICLE / CLAUSE SUMMARY")
    print("=" * 90)

    print(
        "Articles represented:",
        len(article_clauses)
    )

    article_numbers = sorted(
        article_clauses.keys(),
        key=lambda value: int(
            "".join(
                char
                for char in value
                if char.isdigit()
            )
        )
    )

    print(
        "First article:",
        article_numbers[0]
    )

    print(
        "Last article :",
        article_numbers[-1]
    )


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}"
        )

    chunks = load_chunks(
        INPUT_FILE
    )

    print("=" * 90)
    print("C1 MANUAL QUALITY CHECK")
    print("=" * 90)

    print(
        "Total chunks:",
        len(chunks)
    )

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    duplicates = [
        chunk_id
        for chunk_id, count
        in Counter(chunk_ids).items()
        if count > 1
    ]

    print(
        "Duplicate IDs:",
        len(duplicates)
    )

    print_article(
        chunks,
        "72"
    )

    print_article(
        chunks,
        "87"
    )

    inspect_intro_chunks(
        chunks
    )

    inspect_multi_page(
        chunks
    )

    inspect_clause_distribution(
        chunks
    )


if __name__ == "__main__":
    main()