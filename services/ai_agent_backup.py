from __future__ import annotations

from typing import Any

import streamlit as st
from google import genai


DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiAgentError(RuntimeError):
    """Lỗi cấu hình hoặc gọi Gemini API."""


def _read_secret(
    key: str,
    default: str = "",
) -> str:
    """
    Đọc khóa từ st.secrets mà không làm ứng dụng dừng đột ngột
    khi secrets.toml chưa tồn tại.
    """

    try:
        value = st.secrets.get(
            key,
            default,
        )
    except Exception:
        value = default

    return str(value).strip()


def get_gemini_config() -> tuple[str, str]:
    """
    Lấy API key và tên model từ .streamlit/secrets.toml.
    """

    api_key = _read_secret(
        "GEMINI_API_KEY"
    )

    model = _read_secret(
        "GEMINI_MODEL",
        DEFAULT_MODEL,
    )

    if not api_key:
        raise GeminiAgentError(
            "Chưa cấu hình GEMINI_API_KEY. "
            "Hãy tạo file .streamlit/secrets.toml, "
            "sau đó khởi động lại Streamlit."
        )

    if not model:
        model = DEFAULT_MODEL

    return api_key, model


def gemini_is_configured() -> bool:
    """
    Kiểm tra website đã có API key hay chưa.
    """

    try:
        get_gemini_config()
        return True
    except GeminiAgentError:
        return False


def _format_mapping(
    values: dict[str, Any],
) -> str:
    """
    Chuyển dictionary thành nội dung dễ đọc trong prompt.
    """

    if not values:
        return "Không có dữ liệu."

    lines = []

    for key, value in values.items():
        lines.append(
            f"- {key}: {value}"
        )

    return "\n".join(lines)


def analyze_result(
    exercise_name: str,
    model_name: str,
    parameters: dict[str, Any],
    result_summary: str,
    policy_questions: str = "",
) -> str:
    """
    Gửi kết quả mô hình tới Gemini để diễn giải.

    Gemini chỉ nhận kết quả đã được Python tính toán.
    Hàm không yêu cầu Gemini tự tạo hay tính lại số liệu.
    """

    api_key, model = (
        get_gemini_config()
    )

    prompt = f"""
Bạn là chuyên gia về mô hình ra quyết định, kinh tế lượng
và chính sách phát triển kinh tế Việt Nam.

BÀI TOÁN
{exercise_name}

MÔ HÌNH
{model_name}

THAM SỐ ĐẦU VÀO
{_format_mapping(parameters)}

KẾT QUẢ DO PYTHON TÍNH TOÁN
{result_summary}

CÂU HỎI CHÍNH SÁCH
{policy_questions or "Không có câu hỏi bổ sung."}

Hãy phân tích bằng tiếng Việt theo đúng cấu trúc sau:

1. Tóm tắt 4-6 kết quả quan trọng nhất.
2. Giải thích ý nghĩa kinh tế của kết quả.
3. Đánh giá tính hợp lý và độ phù hợp của mô hình.
4. Xác định yếu tố hoặc ràng buộc có ảnh hưởng lớn nhất.
5. Trả lời trực tiếp các câu hỏi chính sách được cung cấp.
6. Đưa ra khuyến nghị chính sách khả thi.
7. Nêu rõ hạn chế của dữ liệu và mô hình.

Yêu cầu bắt buộc:
- Chỉ sử dụng các con số được cung cấp trong phần kết quả.
- Không tự tạo thêm số liệu, nguồn tài liệu hoặc kết quả mới.
- Không khẳng định đây là dự báo chính thức.
- Trình bày rõ ràng, có tiêu đề và gạch đầu dòng hợp lý.
"""

    try:
        client = genai.Client(
            api_key=api_key
        )

        response = (
            client.models.generate_content(
                model=model,
                contents=prompt,
            )
        )

    except Exception as error:
        raise GeminiAgentError(
            "Không gọi được Gemini API. "
            f"Chi tiết: {error}"
        ) from error

    response_text = getattr(
        response,
        "text",
        None,
    )

    if not response_text:
        raise GeminiAgentError(
            "Gemini không trả về nội dung văn bản."
        )

    return str(response_text).strip()
