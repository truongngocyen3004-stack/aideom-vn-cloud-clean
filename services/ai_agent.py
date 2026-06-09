from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

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
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.22
DEFAULT_MAX_PROMPT_CHARS = 90000
DEFAULT_MIN_WORDS = 1200
DEFAULT_MIN_WORDS_BAI12 = 2200
DEFAULT_AUTO_EXPAND = True


class GeminiAgentError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str
    model: str
    max_attempts: int
    retry_base_seconds: float
    max_output_tokens: int
    temperature: float
    max_prompt_chars: int
    min_words: int
    min_words_bai12: int
    auto_expand: bool


def _secret_or_env(key: str, default: Any = "") -> Any:
    try:
        value = st.secrets.get(key, None)
        if value not in (None, ""):
            return value
    except Exception:
        pass

    value = os.getenv(key)
    if value not in (None, ""):
        return value

    if key == "GEMINI_API_KEY":
        value = os.getenv("GOOGLE_API_KEY")
        if value:
            return value

    return default


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(value)
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _as_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(value)
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "y", "on", "có", "co"}:
        return True

    if text in {"0", "false", "no", "n", "off", "không", "khong"}:
        return False

    return default


def get_gemini_settings() -> GeminiSettings:
    if genai is None:
        raise GeminiAgentError(
            "Chưa cài google-genai. Chạy lệnh: "
            "python -m pip install -U google-genai. "
            f"Chi tiết: {_GENAI_IMPORT_ERROR}"
        )

    api_key = str(_secret_or_env("GEMINI_API_KEY", "")).strip()

    if not api_key:
        raise GeminiAgentError(
            "Chưa cấu hình GEMINI_API_KEY trong .streamlit/secrets.toml"
        )

    return GeminiSettings(
        api_key=api_key,
        model=str(_secret_or_env("GEMINI_MODEL", DEFAULT_MODEL)).strip()
        or DEFAULT_MODEL,
        max_attempts=_as_int(
            _secret_or_env("GEMINI_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS),
            DEFAULT_MAX_ATTEMPTS,
            1,
            8,
        ),
        retry_base_seconds=_as_float(
            _secret_or_env(
                "GEMINI_RETRY_BASE_SECONDS",
                DEFAULT_RETRY_BASE_SECONDS,
            ),
            DEFAULT_RETRY_BASE_SECONDS,
            0.2,
            15.0,
        ),
        max_output_tokens=_as_int(
            _secret_or_env(
                "GEMINI_MAX_OUTPUT_TOKENS",
                DEFAULT_MAX_OUTPUT_TOKENS,
            ),
            DEFAULT_MAX_OUTPUT_TOKENS,
            1024,
            32768,
        ),
        temperature=_as_float(
            _secret_or_env("GEMINI_TEMPERATURE", DEFAULT_TEMPERATURE),
            DEFAULT_TEMPERATURE,
            0.0,
            1.0,
        ),
        max_prompt_chars=_as_int(
            _secret_or_env(
                "GEMINI_MAX_PROMPT_CHARS",
                DEFAULT_MAX_PROMPT_CHARS,
            ),
            DEFAULT_MAX_PROMPT_CHARS,
            10000,
            250000,
        ),
        min_words=_as_int(
            _secret_or_env("GEMINI_MIN_WORDS", DEFAULT_MIN_WORDS),
            DEFAULT_MIN_WORDS,
            500,
            5000,
        ),
        min_words_bai12=_as_int(
            _secret_or_env(
                "GEMINI_MIN_WORDS_BAI12",
                DEFAULT_MIN_WORDS_BAI12,
            ),
            DEFAULT_MIN_WORDS_BAI12,
            1000,
            8000,
        ),
        auto_expand=_as_bool(
            _secret_or_env("GEMINI_AUTO_EXPAND", DEFAULT_AUTO_EXPAND),
            DEFAULT_AUTO_EXPAND,
        ),
    )


def get_gemini_config() -> tuple[str, str]:
    settings = get_gemini_settings()
    return settings.api_key, settings.model


def gemini_is_configured() -> bool:
    try:
        get_gemini_settings()
        return True
    except GeminiAgentError:
        return False


def gemini_configuration_message() -> str:
    try:
        settings = get_gemini_settings()
        return (
            f"Gemini đã cấu hình. Model: {settings.model}; "
            f"min words: {settings.min_words}; "
            f"max output tokens: {settings.max_output_tokens}."
        )
    except GeminiAgentError as error:
        return str(error)


EXERCISE_RUBRIC = {
    "1": "Bài 1: Cobb-Douglas mở rộng, TFP A_t, MAPE, phân rã tăng trưởng, D-AI-H, kịch bản 2030.",
    "2": "Bài 2: LP 4 biến, nghiệm tối ưu, shadow price, độ nhạy ngân sách, ưu tiên nhân lực số.",
    "3": "Bài 3: Priority_i, chuẩn hóa min-max, Risk, trọng số, top ngành, độ nhạy AI Readiness.",
    "4": "Bài 4: LP ngành-vùng, PuLP/CVXPY, công bằng vùng, ma trận phân bổ, chi phí công bằng.",
    "5": "Bài 5: MIP chọn dự án, biến nhị phân, ngân sách đa năm, tiên quyết, loại trừ, rủi ro dự án.",
    "6": "Bài 6: TOPSIS, Entropy, ideal/anti-ideal, C_i*, độ nhạy AI Readiness, chọn vùng AI.",
    "7": "Bài 7: NSGA-II, Pareto frontier, TOPSIS, tăng trưởng, bao trùm, môi trường, an ninh dữ liệu.",
    "8": "Bài 8: tối ưu động, K-D-AI-H, tiêu dùng, đầu tư, chiết khấu, front-loaded/back-loaded.",
    "9": "Bài 9: NetJob, x_AI, x_H, đào tạo lại, lao động dễ tổn thương, ràng buộc mất việc 5%.",
    "10": "Bài 10: stochastic programming, RP, EV, EEV, VSS, EVPI, robust minimax regret.",
    "11": "Bài 11: Q-learning, MDP 81 trạng thái, policy π*, rule-based, DQN, đạo đức AI.",
    "12": "Bài 12: dashboard tích hợp 11 mô hình, GDP 2030, NetJob, readiness, rủi ro, kịch bản, bàn giao.",
}


def _exercise_number(exercise_name: str) -> str | None:
    match = re.search(r"bài\s*0?(\d{1,2})", str(exercise_name).lower())
    if not match:
        return None

    number = int(match.group(1))

    if 1 <= number <= 12:
        return str(number)

    return None


def _target_words(settings: GeminiSettings, exercise_name: str) -> int:
    if _exercise_number(exercise_name) == "12":
        return settings.min_words_bai12

    return settings.min_words


def _rubric(exercise_name: str) -> str:
    number = _exercise_number(exercise_name)

    if number and number in EXERCISE_RUBRIC:
        return EXERCISE_RUBRIC[number]

    return (
        "Bám sát mô hình, kết quả định lượng, biểu đồ, độ nhạy, "
        "hạn chế và câu hỏi chính sách."
    )


def _format_value(value: Any) -> str:
    if isinstance(value, dict):
        return "\n" + "\n".join(
            f"  - {key}: {_format_value(item)}"
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)

    return str(value)


def _format_mapping(values: dict[str, Any]) -> str:
    if not values:
        return "Không có tham số đầu vào."

    return "\n".join(
        f"- {key}: {_format_value(value)}"
        for key, value in values.items()
    )


def _truncate(text: str, max_chars: int) -> str:
    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    head = int(max_chars * 0.75)
    tail = max_chars - head

    return (
        text[:head]
        + "\n\n...[NỘI DUNG QUÁ DÀI ĐÃ RÚT GỌN Ở GIỮA]...\n\n"
        + text[-tail:]
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\wÀ-ỹ]+\b", str(text), flags=re.UNICODE))


def _build_prompt(
    exercise_name: str,
    model_name: str,
    parameters: dict[str, Any],
    result_summary: str,
    policy_questions: str,
    settings: GeminiSettings,
) -> str:
    target_words = _target_words(settings, exercise_name)

    prompt = f"""
Bạn là chuyên gia mô hình ra quyết định, tối ưu hóa, kinh tế lượng,
khoa học dữ liệu kinh tế và chính sách phát triển Việt Nam trong kỷ nguyên AI.

Nhiệm vụ:
Viết BÁO CÁO PHÂN TÍCH HỌC THUẬT DÀI, SÂU, CÓ CẤU TRÚC cho kết quả mô hình Python.

BÀI ĐANG PHÂN TÍCH
{exercise_name}

MÔ HÌNH / PHƯƠNG PHÁP
{model_name}

RUBRIC RIÊNG CỦA BÀI
{_rubric(exercise_name)}

THAM SỐ ĐẦU VÀO
{_format_mapping(parameters)}

KẾT QUẢ PYTHON ĐÃ TÍNH
{result_summary}

CÂU HỎI CHÍNH SÁCH
{policy_questions or "Không có câu hỏi riêng, hãy tự rút ra câu hỏi chính sách phù hợp từ kết quả."}

YÊU CẦU ĐỘ DÀI
- Viết tối thiểu khoảng {target_words} từ.
- Không được trả lời ngắn.
- Nếu dữ liệu ít, vẫn phải phân tích sâu mô hình, biến, ràng buộc, trade-off, độ nhạy và hạn chế.
- Chỉ dùng số liệu được truyền vào. Không bịa thêm số liệu.

CẤU TRÚC BẮT BUỘC
# 1. Tóm tắt điều hành
Viết 2-3 đoạn, nêu kết quả chính và ý nghĩa chính sách.

# 2. Bối cảnh và mục tiêu mô hình
Giải thích bài toán chính sách Việt Nam mà mô hình đang giải quyết.

# 3. Mô hình toán học và logic ra quyết định
Giải thích biến quyết định, hàm mục tiêu, ràng buộc, giả định.

# 4. Phân tích kết quả định lượng
Phân tích nghiệm/phương án/kết quả chính. Chỉ ra số liệu nổi bật.

# 5. Phân tích biểu đồ và bảng trong dashboard
Giải thích xu hướng, thứ hạng, phân bổ, heatmap, đường cong hoặc kịch bản nếu có.

# 6. Phân tích độ nhạy và kịch bản
Phân tích khi tham số thay đổi. Nếu thiếu dữ liệu độ nhạy, nêu rõ cần kiểm tra thêm.

# 7. Hàm ý kinh tế và chính sách công
Đánh giá tác động tới tăng trưởng, việc làm, chuyển đổi số, năng suất, công bằng, môi trường và rủi ro.

# 8. Rủi ro, hạn chế và kiểm định cần bổ sung
Nêu hạn chế dữ liệu, giả định mô hình, rủi ro trọng số, rủi ro tuyến tính hóa, rủi ro mô phỏng.

# 9. Khuyến nghị
Chia rõ:
- Ngắn hạn
- Trung hạn
- Dài hạn

# 10. Kết luận
Kết luận rõ mô hình hỗ trợ ra quyết định nhưng không thay thế quyết định chính trị, thể chế và phản biện xã hội.

YÊU CẦU CHẤT LƯỢNG
- Trả lời đầy đủ câu hỏi chính sách.
- Không tự tạo số liệu.
- Không tự thêm nguồn ngoài.
- Không nói kết quả là chắc chắn tuyệt đối.
- Văn phong học thuật, phù hợp đưa vào Word/PDF cuối kỳ.
"""

    return _truncate(prompt, settings.max_prompt_chars)


def _expand_prompt(
    exercise_name: str,
    old_prompt: str,
    old_answer: str,
    settings: GeminiSettings,
) -> str:
    target_words = _target_words(settings, exercise_name)

    prompt = f"""
Câu trả lời trước quá ngắn so với yêu cầu cuối kỳ.

Hãy viết lại thành bản phân tích học thuật đầy đủ hơn, tối thiểu khoảng {target_words} từ.
Bắt buộc giữ đủ 10 mục lớn, phân tích sâu hơn mô hình, kết quả, độ nhạy, rủi ro và khuyến nghị.
Không bịa số liệu. Không thêm nguồn ngoài. Chỉ dùng dữ liệu trong prompt gốc.

PROMPT GỐC
{old_prompt}

CÂU TRẢ LỜI CŨ
{old_answer}

Hãy trả về bản hoàn chỉnh, không xin lỗi, không giải thích rằng bạn đang mở rộng.
"""

    return _truncate(prompt, settings.max_prompt_chars)


def _is_retryable(error: Exception) -> bool:
    text = f"{type(error).__name__}: {error}".lower()

    markers = [
        "429",
        "500",
        "502",
        "503",
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
        "temporarily",
    ]

    return any(marker in text for marker in markers)


def _friendly_error(error: Exception, model: str) -> str:
    text = f"{type(error).__name__}: {error}".lower()

    if "429" in text or "quota" in text or "rate limit" in text:
        return (
            "Gemini đang giới hạn tần suất hoặc hết quota tạm thời. "
            "Hãy chờ 30-60 giây rồi bấm lại."
        )

    if "503" in text or "unavailable" in text or "overloaded" in text:
        return (
            "Gemini đang quá tải tạm thời. Website đã tự thử lại nhưng chưa thành công. "
            "Hãy đợi một lát rồi bấm lại."
        )

    if "api key" in text or "401" in text or "403" in text:
        return (
            "API key không hợp lệ hoặc không có quyền gọi model. "
            "Hãy kiểm tra GEMINI_API_KEY trong .streamlit/secrets.toml."
        )

    return f"Không gọi được Gemini API với model {model}. Chi tiết: {error}"


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)

    if text:
        return str(text).strip()

    try:
        chunks = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(str(part_text))
        return "\n".join(chunks).strip()
    except Exception:
        return ""


def _call_gemini(prompt: str, settings: GeminiSettings) -> str:
    client = genai.Client(api_key=settings.api_key)

    config = None

    if types is not None:
        config = types.GenerateContentConfig(
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
        )

    last_error = None

    for attempt in range(settings.max_attempts):
        try:
            response = client.models.generate_content(
                model=settings.model,
                contents=prompt,
                config=config,
            )

            text = _extract_text(response)

            if not text:
                raise GeminiAgentError("Gemini trả về phản hồi rỗng.")

            return text

        except Exception as error:
            last_error = error

            if not _is_retryable(error):
                raise GeminiAgentError(
                    _friendly_error(error, settings.model)
                ) from error

            if attempt < settings.max_attempts - 1:
                time.sleep(
                    min(
                        settings.retry_base_seconds * (2 ** attempt)
                        + random.uniform(0, 0.7),
                        20,
                    )
                )

    raise GeminiAgentError(
        _friendly_error(last_error, settings.model)
    ) from last_error


def analyze_result(
    exercise_name: str,
    model_name: str,
    parameters: dict[str, Any],
    result_summary: str,
    policy_questions: str = "",
) -> str:
    settings = get_gemini_settings()

    prompt = _build_prompt(
        exercise_name=exercise_name,
        model_name=model_name,
        parameters=parameters,
        result_summary=result_summary,
        policy_questions=policy_questions,
        settings=settings,
    )

    answer = _call_gemini(prompt, settings)

    target_words = _target_words(settings, exercise_name)

    too_short = _word_count(answer) < int(target_words * 0.72)

    missing_sections = not all(
        word in answer.lower()
        for word in [
            "tóm tắt",
            "mô hình",
            "kết quả",
            "độ nhạy",
            "khuyến nghị",
            "hạn chế",
            "kết luận",
        ]
    )

    if settings.auto_expand and (too_short or missing_sections):
        try:
            expanded = _call_gemini(
                _expand_prompt(exercise_name, prompt, answer, settings),
                settings,
            )

            if _word_count(expanded) > _word_count(answer) * 1.25:
                answer = expanded

        except Exception:
            pass

    return answer


def test_gemini_connection() -> str:
    settings = get_gemini_settings()

    return _call_gemini(
        "Hãy trả lời đúng một câu: Kết nối Gemini thành công.",
        settings,
    )