from pathlib import Path
import time

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DEVICE = "cpu"

MAX_LENGTH = 512

BATCH_SIZE = 4

DEFAULT_TOP_K = 5


# ============================================================
# BGE RERANKER
# ============================================================

class BGEReranker:

    def __init__(
        self,
        model_name=MODEL_NAME,
        device=DEVICE,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        warmup=True,
    ):

        self.model_name = model_name

        self.device = device

        self.max_length = max_length

        self.batch_size = batch_size

        print("=" * 78)
        print(
            "C3 BGE RERANKER INITIALIZATION"
        )
        print("=" * 78)

        print(
            f"Model        : "
            f"{self.model_name}"
        )

        print(
            f"Device       : "
            f"{self.device}"
        )

        print(
            f"Max length   : "
            f"{self.max_length}"
        )

        print(
            f"Batch size   : "
            f"{self.batch_size}"
        )

        # ====================================================
        # TOKENIZER
        # ====================================================

        print()
        print(
            "Loading tokenizer..."
        )

        tokenizer_start = (
            time.perf_counter()
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name
            )
        )

        tokenizer_elapsed = (
            time.perf_counter()
            - tokenizer_start
        )

        print(
            f"Tokenizer loaded : PASS "
            f"({tokenizer_elapsed:.2f}s)"
        )

        # ====================================================
        # MODEL
        # ====================================================

        print()
        print(
            "Loading reranker model..."
        )

        model_start = (
            time.perf_counter()
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        model_elapsed = (
            time.perf_counter()
            - model_start
        )

        print(
            f"Model loaded     : PASS "
            f"({model_elapsed:.2f}s)"
        )

        # ====================================================
        # WARM-UP
        # ====================================================

        if warmup:

            print()
            print(
                "Running reranker warm-up..."
            )

            self.score_texts(
                question=(
                    "Quy định giao thông "
                    "đường bộ là gì?"
                ),
                texts=[
                    (
                        "Quy định về trật tự, "
                        "an toàn giao thông "
                        "đường bộ."
                    )
                ],
            )

            print(
                "Warm-up          : PASS"
            )

        print()
        print(
            "C3 reranker ready."
        )


    # ========================================================
    # SCORE TEXTS
    # ========================================================

    def score_texts(
        self,
        question,
        texts,
    ):
        """
        Score từng cặp:

            (question, candidate_text)

        Raw logits được dùng để ranking.

        Sigmoid chỉ dùng để hiển thị dễ đọc.
        """

        if not texts:

            return (
                np.array(
                    [],
                    dtype=np.float32
                ),
                np.array(
                    [],
                    dtype=np.float32
                ),
                0.0,
            )

        all_logits = []
        all_probabilities = []

        start_time = (
            time.perf_counter()
        )

        # ====================================================
        # BATCHING
        # ====================================================

        for start_index in range(
            0,
            len(texts),
            self.batch_size,
        ):

            end_index = min(
                start_index
                + self.batch_size,
                len(texts),
            )

            batch_texts = (
                texts[
                    start_index:
                    end_index
                ]
            )

            pairs = [
                [
                    question,
                    text,
                ]
                for text in batch_texts
            ]

            inputs = (
                self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=(
                        self.max_length
                    ),
                )
            )

            inputs = {
                key: value.to(
                    self.device
                )
                for key, value
                in inputs.items()
            }

            # ================================================
            # FORWARD
            # ================================================

            with torch.no_grad():

                output = (
                    self.model(
                        **inputs,
                        return_dict=True,
                    )
                )

                logits = (
                    output.logits
                    .view(-1)
                    .float()
                )

                probabilities = (
                    torch.sigmoid(
                        logits
                    )
                )

            all_logits.extend(
                logits
                .cpu()
                .numpy()
                .tolist()
            )

            all_probabilities.extend(
                probabilities
                .cpu()
                .numpy()
                .tolist()
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return (
            np.asarray(
                all_logits,
                dtype=np.float32
            ),
            np.asarray(
                all_probabilities,
                dtype=np.float32
            ),
            elapsed,
        )


    # ========================================================
    # RERANK
    # ========================================================

    def rerank(
        self,
        question,
        candidates,
        top_k=DEFAULT_TOP_K,
    ):
        """
        candidates là output của RRF.

        Mỗi candidate phải có:
            chunk_id
            metadata["text"]

        Return:
            reranked candidates
            elapsed
        """

        if not candidates:

            return (
                [],
                0.0,
            )

        texts = []

        for candidate in candidates:

            if (
                "metadata"
                not in candidate
            ):

                raise ValueError(
                    "Candidate does not "
                    "contain metadata."
                )

            metadata = (
                candidate[
                    "metadata"
                ]
            )

            if (
                "text"
                not in metadata
            ):

                raise ValueError(
                    "Candidate metadata "
                    "does not contain text."
                )

            texts.append(
                metadata[
                    "text"
                ]
            )

        (
            logits,
            probabilities,
            elapsed,
        ) = self.score_texts(
            question,
            texts,
        )

        scored = []

        for (
            candidate,
            score,
            probability,
        ) in zip(
            candidates,
            logits,
            probabilities,
        ):

            item = dict(
                candidate
            )

            item[
                "pre_rerank_rank"
            ] = (
                candidate.get(
                    "rank"
                )
            )

            item[
                "rerank_score"
            ] = float(
                score
            )

            item[
                "rerank_probability"
            ] = float(
                probability
            )

            scored.append(
                item
            )

        # ====================================================
        # SORT
        # ====================================================

        scored.sort(
            key=lambda item: (
                -item[
                    "rerank_score"
                ],
                item[
                    "chunk_id"
                ],
            )
        )

        final_results = []

        for rank, item in enumerate(
            scored[:top_k],
            start=1
        ):

            result = dict(
                item
            )

            result[
                "rank"
            ] = rank

            result[
                "rerank_rank"
            ] = rank

            final_results.append(
                result
            )

        return (
            final_results,
            elapsed,
        )


# ============================================================
# STANDALONE SMOKE TEST
# ============================================================

def main():

    reranker = (
        BGEReranker()
    )

    question = (
        "Người lái xe cần đáp ứng "
        "điều kiện gì để tham gia "
        "giao thông đường bộ?"
    )

    texts = [
        (
            "Người lái xe tham gia giao thông "
            "đường bộ phải đủ tuổi, sức khỏe "
            "theo quy định và có giấy phép lái xe "
            "phù hợp."
        ),
        (
            "Người điều khiển phương tiện gây "
            "tai nạn giao thông phải dừng xe "
            "và báo cơ quan có thẩm quyền."
        ),
    ]

    (
        scores,
        probabilities,
        elapsed,
    ) = reranker.score_texts(
        question,
        texts,
    )

    print()
    print("=" * 78)
    print(
        "RERANKER SMOKE TEST"
    )
    print("=" * 78)

    for index, (
        score,
        probability
    ) in enumerate(
        zip(
            scores,
            probabilities
        ),
        start=1
    ):

        print(
            f"Candidate {index}: "
            f"logit={score:.6f} "
            f"sigmoid={probability:.6f}"
        )

    print()

    print(
        f"Elapsed: "
        f"{elapsed:.4f}s"
    )


if __name__ == "__main__":
    main()