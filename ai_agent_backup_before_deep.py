from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable

import streamlit as st

try:
    from google import genai
    from google.genai import types
except Exception as import_error:
    genai = None
    types = None
    _GENAI_IMPORT_ERROR = import_error
else:
    _GENAI_IMPORT_ERROR = None


DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_RETRY_BASE_SECONDS = 2.0
DEFAULT_TIMEOUT_MS = 90_000
DEFAULT_MAX_OUTPUT_TOKENS = 4_096
DEFAULT_TEMPERATURE = 0.20
DEFAULT_MAX_PROMPT_CHARS = 60_000


class GeminiAgentError(RuntimeError):
    """Lỗi cấu hình hoặc lỗi gọi Gemini API có thông báo thân thiện."""


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str
    primary_model: str
    fallback_models: tuple[str, ...]
    max_attempts: int
    retry_base_seconds: float
    timeout_ms: int
    max_output_tokens: int
    temperature: float
    max_prompt_chars: int

    @property
    def models(self) -> tuple[str, ...]:
        ordered: list[str] = []

        for model in (
            self.primary_model,
            *self.fallback_models,
        ):
            clean = str(model).strip()

            if clean and clean not in ordered:
                ordered.append(clean)

        return tuple(ordered)


def _safe_secret_get(
    key: str,
    default: Any = "",
) -> Any:
    """
    Đọc st.secrets an toàn.

    Hỗ trợ cả hai kiểu:
    GEMINI_API_KEY = "..."
    hoặc:
    [gemini]
    api_key = "..."
    """

    try:
        direct = st.secrets.get(
            key,
            None,
        )

        if direct not in (
            None,
            "",
        ):
            return direct
    except Exception:
        pass

    nested_keys = {
        "GEMINI_API_KEY":
            "api_key",
        "GEMINI_MODEL":
            "model",
        "GEMINI_FALLBACK_MODELS":
            "fallback_models",
        "GEMINI_MAX_ATTEMPTS":
            "max_attempts",
        "GEMINI_RETRY_BASE_SECONDS":
            "retry_base_seconds",
        "GEMINI_TIMEOUT_MS":
            "timeout_ms",
        "GEMINI_MAX_OUTPUT_TOKENS":
            "max_output_tokens",
        "GEMINI_TEMPERATURE":
            "temperature",
        "GEMINI_MAX_PROMPT_CHARS":
            "max_prompt_chars",
    }

    nested_key = nested_keys.get(
        key
    )

    if nested_key:
        try:
            section = st.secrets.get(
                "gemini",
                {},
            )

            value = section.get(
                nested_key,
                None,
            )

            if value not in (
                None,
                "",
            ):
                return value
        except Exception:
            pass

    return default


def _read_setting(
    key: str,
    default: Any = "",
) -> Any:
    """
    Ưu tiên st.secrets, sau đó biến môi trường.
    """

    secret_value = _safe_secret_get(
        key,
        None,
    )

    if secret_value not in (
        None,
        "",
    ):
        return secret_value

    env_value = os.getenv(
        key
    )

    if env_value not in (
        None,
        "",
    ):
        return env_value

    # Google SDK cũng thường đọc GOOGLE_API_KEY.
    if key == "GEMINI_API_KEY":
        google_key = os.getenv(
            "GOOGLE_API_KEY"
        )

        if google_key:
            return google_key

    return default


def _as_int(
    value: Any,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        parsed = default

    return max(
        minimum,
        min(
            maximum,
            parsed,
        ),
    )


def _as_float(
    value: Any,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        parsed = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        parsed = default

    return max(
        minimum,
        min(
            maximum,
            parsed,
        ),
    )


def _split_models(
    value: Any,
) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        candidates = [
            str(item).strip()
            for item in value
        ]
    else:
        text = str(
            value
        ).replace(
            ";",
            ",",
        )

        candidates = [
            item.strip()
            for item in text.split(
                ","
            )
        ]

    result: list[str] = []

    for model in candidates:
        if (
            model
            and model not in result
        ):
            result.append(
                model
            )

    return tuple(
        result
    )


def get_gemini_settings() -> GeminiSettings:
    if genai is None:
        raise GeminiAgentError(
            "Chưa cài thư viện google-genai. "
            "Chạy lệnh: python -m pip install -U google-genai. "
            f"Chi tiết import: {_GENAI_IMPORT_ERROR}"
        )

    api_key = str(
        _read_setting(
            "GEMINI_API_KEY",
            "",
        )
    ).strip()

    if not api_key:
        raise GeminiAgentError(
            "Chưa cấu hình GEMINI_API_KEY. "
            "Hãy tạo file .streamlit/secrets.toml, "
            "sau đó dừng và chạy lại Streamlit."
        )

    primary_model = str(
        _read_setting(
            "GEMINI_MODEL",
            DEFAULT_MODEL,
        )
    ).strip() or DEFAULT_MODEL

    fallback_models = _split_models(
        _read_setting(
            "GEMINI_FALLBACK_MODELS",
            "",
        )
    )

    return GeminiSettings(
        api_key=api_key,
        primary_model=(
            primary_model
        ),
        fallback_models=(
            fallback_models
        ),
        max_attempts=_as_int(
            _read_setting(
                "GEMINI_MAX_ATTEMPTS",
                DEFAULT_MAX_ATTEMPTS,
            ),
            DEFAULT_MAX_ATTEMPTS,
            1,
            8,
        ),
        retry_base_seconds=_as_float(
            _read_setting(
                "GEMINI_RETRY_BASE_SECONDS",
                DEFAULT_RETRY_BASE_SECONDS,
            ),
            DEFAULT_RETRY_BASE_SECONDS,
            0.2,
            15.0,
        ),
        timeout_ms=_as_int(
            _read_setting(
                "GEMINI_TIMEOUT_MS",
                DEFAULT_TIMEOUT_MS,
            ),
            DEFAULT_TIMEOUT_MS,
            10_000,
            300_000,
        ),
        max_output_tokens=_as_int(
            _read_setting(
                "GEMINI_MAX_OUTPUT_TOKENS",
                DEFAULT_MAX_OUTPUT_TOKENS,
            ),
            DEFAULT_MAX_OUTPUT_TOKENS,
            512,
            16_384,
        ),
        temperature=_as_float(
            _read_setting(
                "GEMINI_TEMPERATURE",
                DEFAULT_TEMPERATURE,
            ),
            DEFAULT_TEMPERATURE,
            0.0,
            1.0,
        ),
        max_prompt_chars=_as_int(
            _read_setting(
                "GEMINI_MAX_PROMPT_CHARS",
                DEFAULT_MAX_PROMPT_CHARS,
            ),
            DEFAULT_MAX_PROMPT_CHARS,
            5_000,
            200_000,
        ),
    )


def get_gemini_config() -> tuple[str, str]:
    """
    Giữ tương thích với code cũ.
    """

    settings = get_gemini_settings()

    return (
        settings.api_key,
        settings.primary_model,
    )


def gemini_is_configured() -> bool:
    try:
        get_gemini_settings()
        return True
    except GeminiAgentError:
        return False


def gemini_configuration_message() -> str:
    try:
        settings = get_gemini_settings()
    except GeminiAgentError as error:
        return str(
            error
        )

    fallback_text = (
        ", ".join(
            settings.fallback_models
        )
        if settings.fallback_models
        else "không đặt"
    )

    return (
        f"Gemini đã cấu hình. Model chính: "
        f"{settings.primary_model}; model dự phòng: "
        f"{fallback_text}; số lần thử: "
        f"{settings.max_attempts}."
    )


def _format_value(
    value: Any,
) -> str:
    if isinstance(
        value,
        dict,
    ):
        lines = []

        for key, item in value.items():
            lines.append(
                f"  - {key}: {_format_value(item)}"
            )

        return "\n" + "\n".join(
            lines
        )

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return ", ".join(
            str(item)
            for item in value
        )

    return str(
        value
    )


def _format_mapping(
    values: dict[str, Any],
) -> str:
    if not values:
        return "Không có dữ liệu."

    return "\n".join(
        f"- {key}: {_format_value(value)}"
        for key, value
        in values.items()
    )


def _truncate_text(
    text: str,
    maximum: int,
) -> str:
    clean = str(
        text
    ).strip()

    if len(
        clean
    ) <= maximum:
        return clean

    head_size = int(
        maximum * 0.72
    )
    tail_size = (
        maximum
        - head_size
    )

    return (
        clean[
            :head_size
        ]
        + "\n\n...[NỘI DUNG QUÁ DÀI ĐÃ ĐƯỢC RÚT GỌN]...\n\n"
        + clean[
            -tail_size:
        ]
    )


def _build_prompt(
    exercise_name: str,
    model_name: str,
    parameters: dict[str, Any],
    result_summary: str,
    policy_questions: str,
    maximum_chars: int,
) -> str:
    raw_prompt = f"""
Bạn là chuyên gia về mô hình ra quyết định, tối ưu hóa,
kinh tế lượng và chính sách phát triển kinh tế Việt Nam.

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

Hãy phân tích bằng tiếng Việt theo đúng cấu trúc:

## 1. Kết quả quan trọng
Tóm tắt 4–6 kết quả chính, dùng đúng số liệu được cung cấp.

## 2. Ý nghĩa kinh tế và quản trị
Giải thích kết quả bằng ngôn ngữ rõ ràng, không phóng đại.

## 3. Yếu tố và ràng buộc chi phối
Chỉ ra tham số, mục tiêu hoặc ràng buộc ảnh hưởng mạnh nhất.

## 4. Trả lời câu hỏi chính sách
Trả lời trực tiếp từng câu hỏi đã cung cấp.

## 5. Khuyến nghị
Đưa ra khuyến nghị ngắn hạn và dài hạn, có điều kiện áp dụng.

## 6. Hạn chế
Nêu rõ hạn chế dữ liệu, giả định mô hình và rủi ro diễn giải.

YÊU CẦU BẮT BUỘC
- Chỉ sử dụng các số liệu có trong nội dung được cung cấp.
- Không tự tạo số liệu, tài liệu tham khảo hoặc nguồn dẫn.
- Không tự tính lại rồi thay đổi kết quả Python.
- Không gọi đây là dự báo chính thức.
- Nếu kết quả có dấu hiệu bất thường, phải nói rõ.
- Trình bày bằng Markdown, tiêu đề rõ và dễ đưa vào báo cáo.
"""

    return _truncate_text(
        raw_prompt,
        maximum_chars,
    )


def _error_text(
    error: Exception,
) -> str:
    return (
        f"{type(error).__name__}: {error}"
    ).lower()


def _is_retryable(
    error: Exception,
) -> bool:
    text = _error_text(
        error
    )

    markers = (
        "429",
        "503",
        "500",
        "502",
        "504",
        "resource_exhausted",
        "unavailable",
        "overloaded",
        "high demand",
        "rate limit",
        "quota",
        "timeout",
        "timed out",
        "deadline exceeded",
        "connection reset",
        "connection aborted",
        "temporarily",
    )

    return any(
        marker in text
        for marker in markers
    )


def _is_model_error(
    error: Exception,
) -> bool:
    text = _error_text(
        error
    )

    markers = (
        "404",
        "not found",
        "model is not supported",
        "model not supported",
        "invalid model",
        "model name",
    )

    return any(
        marker in text
        for marker in markers
    )


def _friendly_error(
    error: Exception,
    model: str,
    attempts: int,
) -> str:
    text = _error_text(
        error
    )

    if (
        "429" in text
        or "resource_exhausted" in text
        or "rate limit" in text
        or "quota" in text
    ):
        return (
            "Gemini đang giới hạn tần suất hoặc đã hết hạn mức "
            "tạm thời. Hãy chờ 30–60 giây rồi bấm lại. "
            f"Model: {model}; đã thử {attempts} lần."
        )

    if (
        "503" in text
        or "unavailable" in text
        or "overloaded" in text
        or "high demand" in text
    ):
        return (
            "Gemini đang quá tải tạm thời (503 UNAVAILABLE). "
            "Website đã tự thử lại nhưng chưa thành công. "
            "Hãy đợi một lát rồi bấm lại. "
            f"Model: {model}; đã thử {attempts} lần."
        )

    if (
        "api key" in text
        or "permission_denied" in text
        or "401" in text
        or "403" in text
    ):
        return (
            "API key không hợp lệ, bị giới hạn hoặc không có quyền "
            "gọi model. Hãy kiểm tra GEMINI_API_KEY và GEMINI_MODEL "
            "trong .streamlit/secrets.toml."
        )

    if _is_model_error(
        error
    ):
        return (
            f"Không gọi được model '{model}'. "
            "Hãy đổi GEMINI_MODEL trong .streamlit/secrets.toml "
            "sang model mà API key của bạn được phép sử dụng."
        )

    return (
        "Không gọi được Gemini API. "
        f"Model: {model}. Chi tiết kỹ thuật: {error}"
    )


def _response_text(
    response: Any,
) -> str:
    direct_text = getattr(
        response,
        "text",
        None,
    )

    if direct_text:
        return str(
            direct_text
        ).strip()

    try:
        candidates = getattr(
            response,
            "candidates",
            None,
        ) or []

        chunks: list[str] = []

        for candidate in candidates:
            content = getattr(
                candidate,
                "content",
                None,
            )

            parts = getattr(
                content,
                "parts",
                None,
            ) or []

            for part in parts:
                text = getattr(
                    part,
                    "text",
                    None,
                )

                if text:
                    chunks.append(
                        str(text)
                    )

        return "\n".join(
            chunks
        ).strip()
    except Exception:
        return ""


def _sleep_before_retry(
    attempt_index: int,
    base_seconds: float,
) -> None:
    delay = (
        base_seconds
        * (
            2 ** attempt_index
        )
        + random.uniform(
            0.0,
            0.7,
        )
    )

    time.sleep(
        min(
            delay,
            20.0,
        )
    )


def analyze_result(
    exercise_name: str,
    model_name: str,
    parameters: dict[str, Any],
    result_summary: str,
    policy_questions: str = "",
) -> str:
    """
    Hàm dùng chung cho Bài 1–12.

    Tính năng:
    - đọc key từ secrets hoặc environment;
    - tự rút gọn prompt quá dài;
    - thử lại khi gặp 429/5xx/timeout;
    - có thể chuyển sang model dự phòng do người dùng cấu hình;
    - trả thông báo lỗi dễ hiểu.
    """

    settings = get_gemini_settings()

    prompt = _build_prompt(
        exercise_name=(
            exercise_name
        ),
        model_name=model_name,
        parameters=parameters,
        result_summary=(
            result_summary
        ),
        policy_questions=(
            policy_questions
        ),
        maximum_chars=(
            settings.max_prompt_chars
        ),
    )

    try:
        client = genai.Client(
            api_key=(
                settings.api_key
            ),
        )
    except Exception as error:
        raise GeminiAgentError(
            "Không khởi tạo được Gemini client. "
            f"Chi tiết: {error}"
        ) from error

    last_error: Exception | None = None
    last_model = (
        settings.primary_model
    )
    total_attempts = 0

    for model in settings.models:
        last_model = model

        for attempt_index in range(
            settings.max_attempts
        ):
            total_attempts += 1

            try:
                config = None

                if types is not None:
                    config = (
                        types.GenerateContentConfig(
                            temperature=(
                                settings.temperature
                            ),
                            max_output_tokens=(
                                settings.max_output_tokens
                            ),
                        )
                    )

                response = (
                    client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                )

                text = _response_text(
                    response
                )

                if not text:
                    raise RuntimeError(
                        "Gemini trả về phản hồi rỗng."
                    )

                return text

            except Exception as error:
                last_error = error

                model_error = _is_model_error(
                    error
                )
                retryable = _is_retryable(
                    error
                )

                # Model không tồn tại: chuyển ngay sang model dự phòng.
                if model_error:
                    break

                # Lỗi không tạm thời: không gọi lặp gây tốn quota.
                if not retryable:
                    raise GeminiAgentError(
                        _friendly_error(
                            error,
                            model,
                            total_attempts,
                        )
                    ) from error

                # Hết số lần thử của model hiện tại.
                if (
                    attempt_index
                    >= settings.max_attempts
                    - 1
                ):
                    break

                _sleep_before_retry(
                    attempt_index,
                    settings.retry_base_seconds,
                )

    if last_error is None:
        raise GeminiAgentError(
            "Không có model Gemini hợp lệ để gọi."
        )

    raise GeminiAgentError(
        _friendly_error(
            last_error,
            last_model,
            total_attempts,
        )
    ) from last_error


def test_gemini_connection() -> str:
    """
    Kiểm tra nhanh API key/model bằng prompt rất ngắn.
    """

    return analyze_result(
        exercise_name=(
            "Kiểm tra kết nối"
        ),
        model_name=(
            "Gemini API"
        ),
        parameters={
            "Yêu cầu":
                "Chỉ xác nhận kết nối",
        },
        result_summary=(
            "Không có số liệu mô hình."
        ),
        policy_questions=(
            "Chỉ trả lời đúng một câu: "
            "Kết nối Gemini thành công."
        ),
    )
