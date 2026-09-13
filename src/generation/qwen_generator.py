import os
import requests


# ============================================================
# CONFIG
# ============================================================

DEFAULT_BASE_URL = os.getenv(
    "RAG_LLM_BASE_URL",
    "http://localhost:8000/v1",
)

DEFAULT_MODEL = os.getenv(
    "RAG_LLM_MODEL",
    "Qwen/Qwen3-8B",
)

DEFAULT_API_KEY = os.getenv(
    "RAG_LLM_API_KEY",
    "EMPTY",
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Bạn là trợ lý tra cứu văn bản pháp luật.

QUY TẮC BẮT BUỘC:

1. Chỉ được trả lời dựa trên CONTEXT được cung cấp.

2. Không sử dụng kiến thức bên ngoài CONTEXT để bổ sung
   quy định pháp luật.

3. Không suy đoán mức phạt, số tiền, thời hạn, điều kiện
   hoặc quy định nếu CONTEXT không nêu rõ.

4. Mỗi thông tin pháp lý quan trọng phải có citation
   theo dạng [S1], [S2], ...

5. Chỉ được sử dụng citation xuất hiện trong CONTEXT.

6. Tuyệt đối không tạo citation không tồn tại.

7. Nếu nhiều nguồn cùng hỗ trợ một nội dung,
   có thể sử dụng nhiều citation như [S1][S2].

8. Nếu CONTEXT không đủ để trả lời, phải nói rõ:
   "Tài liệu được cung cấp chưa đủ thông tin để trả lời."

9. Không được che giấu việc thiếu thông tin bằng suy luận.

10. Trả lời bằng cùng ngôn ngữ với câu hỏi.

11. Trả lời ngắn gọn, trực tiếp và chính xác.
""".strip()


# ============================================================
# QWEN GENERATOR
# ============================================================

class QwenGenerator:

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        model=DEFAULT_MODEL,
        api_key=DEFAULT_API_KEY,
        timeout=600,
        temperature=0.0,
        max_tokens=512,
        enable_thinking=False,
    ):

        self.base_url = (
            base_url.rstrip("/")
        )

        self.model = model

        self.api_key = api_key

        self.timeout = timeout

        self.temperature = temperature

        self.max_tokens = max_tokens

        self.enable_thinking = (
            enable_thinking
        )


    # ========================================================
    # HEADERS
    # ========================================================

    def headers(self):

        return {
            "Authorization":
                f"Bearer {self.api_key}",

            "Content-Type":
                "application/json",
        }


    # ========================================================
    # CHAT ENDPOINT
    # ========================================================

    @property
    def chat_endpoint(self):

        return (
            f"{self.base_url}"
            f"/chat/completions"
        )


    # ========================================================
    # MODELS ENDPOINT
    # ========================================================

    @property
    def models_endpoint(self):

        return (
            f"{self.base_url}"
            f"/models"
        )


    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def is_available(self):

        try:

            response = requests.get(
                self.models_endpoint,
                headers=(
                    self.headers()
                ),
                timeout=5,
            )

            return (
                response.status_code
                == 200
            )

        except requests.RequestException:

            return False


    # ========================================================
    # BUILD USER PROMPT
    # ========================================================

    def build_user_prompt(
        self,
        question,
        context,
        allowed_source_ids,
        correction_feedback=None,
    ):

        citation_list = (
            ", ".join(
                f"[{source_id}]"
                for source_id
                in allowed_source_ids
            )
        )

        parts = [

            "CONTEXT:",
            "",
            context,

            "",
            "==============================",
            "",

            "CÂU HỎI:",
            "",
            question,

            "",
            "==============================",
            "",

            (
                "Các citation hợp lệ: "
                f"{citation_list}"
            ),

            "",

            (
                "Hãy trả lời chỉ dựa trên CONTEXT. "
                "Mỗi thông tin pháp lý quan trọng "
                "phải có citation [Sx]."
            ),
        ]

        if correction_feedback:

            parts.extend(
                [
                    "",
                    "==============================",
                    "",
                    "YÊU CẦU SỬA:",
                    "",
                    correction_feedback,
                ]
            )

        return "\n".join(
            parts
        )


    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        question,
        context,
        allowed_source_ids,
        correction_feedback=None,
    ):

        user_prompt = (
            self.build_user_prompt(
                question=question,
                context=context,
                allowed_source_ids=(
                    allowed_source_ids
                ),
                correction_feedback=(
                    correction_feedback
                ),
            )
        )

        payload = {

            "model": (
                self.model
            ),

            "messages": [

                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT
                    ),
                },

                {
                    "role": "user",
                    "content": (
                        user_prompt
                    ),
                },
            ],

            "temperature": (
                self.temperature
            ),

            "max_tokens": (
                self.max_tokens
            ),

            # Qwen3 non-thinking mode
            "chat_template_kwargs": {
                "enable_thinking": (
                    self.enable_thinking
                )
            },
        }

        response = requests.post(
            self.chat_endpoint,
            headers=(
                self.headers()
            ),
            json=payload,
            timeout=(
                self.timeout
            ),
        )

        response.raise_for_status()

        data = (
            response.json()
        )

        answer = (
            data[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]
        )

        if answer is None:

            raise RuntimeError(
                "Generator returned empty content."
            )

        return (
            answer.strip()
        )


# ============================================================
# SIMPLE TEST
# ============================================================

def main():

    generator = (
        QwenGenerator()
    )

    print(
        "=" * 72
    )

    print(
        "QWEN GENERATOR"
    )

    print(
        "=" * 72
    )

    print(
        f"Base URL  : "
        f"{generator.base_url}"
    )

    print(
        f"Endpoint  : "
        f"{generator.chat_endpoint}"
    )

    print(
        f"Model     : "
        f"{generator.model}"
    )

    print(
        f"Thinking  : "
        f"{generator.enable_thinking}"
    )

    print(
        f"Available : "
        f"{generator.is_available()}"
    )


if __name__ == "__main__":
    main()