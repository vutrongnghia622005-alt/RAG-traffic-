from pathlib import Path
import json


RESULTS_FILE = Path(
    "results/evaluation/c0_span_results.jsonl"
)


def load_results(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def main():

    rows = load_results(
        RESULTS_FILE
    )

    print("=" * 80)
    print("C0 FAILURE ANALYSIS")
    print("=" * 80)

    failure_count = 0

    for row in rows:

        metrics = row["metrics"]

        evidence_recall = (
            metrics["evidence_recall@5"]
        )

        mrr = metrics["mrr"]

        # Chỉ xem các câu chưa hoàn hảo
        if (
            evidence_recall >= 1.0
            and mrr >= 1.0
        ):
            continue

        failure_count += 1

        print()
        print("=" * 80)
        print(
            "QUESTION ID:",
            row["question_id"]
        )
        print(
            "QUESTION   :",
            row["question"]
        )

        print("-" * 80)

        print(
            "Evidence Recall@5:",
            evidence_recall
        )

        print(
            "MRR              :",
            mrr
        )

        print(
            "nDCG@5           :",
            metrics["ndcg@5"]
        )

        print()
        print("GOLD EVIDENCE:")

        for evidence in (
            row["evidence_units"]
        ):

            print(
                f"  [{evidence['evidence_id']}]"
            )

            print(
                " ",
                evidence["text"]
            )

        print()
        print("TOP-5 RETRIEVAL:")

        for result in (
            row["retrieved"]
        ):

            relevant = (
                "YES"
                if result["relevant"]
                else "NO"
            )

            print()
            print(
                f"Rank      : "
                f"{result['rank']}"
            )

            print(
                f"Chunk ID  : "
                f"{result['chunk_id']}"
            )

            print(
                f"Page      : "
                f"{result['page']}"
            )

            print(
                f"Score     : "
                f"{result['score']:.6f}"
            )

            print(
                f"Relevant  : "
                f"{relevant}"
            )

            print(
                f"Evidence  : "
                f"{result['evidence_ids']}"
            )

    print()
    print("=" * 80)
    print(
        "Questions needing analysis:",
        failure_count
    )
    print("=" * 80)


if __name__ == "__main__":
    main()