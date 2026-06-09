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


DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_MAX_ATTEMPTS = 1
DEFAULT_RETRY_BASE_SECONDS = 15.0
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.20
DEFAULT_MAX_PROMPT_CHARS = 60000
DEFAULT_MIN_WORDS = 850
DEFAULT_MIN_WORDS_BAI12 = 1800
DEFAULT_AUTO_EXPAND = False


class GeminiAgentError(RuntimeError):
    """Lỗi cấu hình hoặc lỗi gọi Gemini API."""


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

    if text in {"1", "true", "yes", "y", "on", "co", "có"}:
        return True

    if text in {"0", "false", "no", "n", "off", "khong", "không"}:
        return False

    return default


def get_gemini_settings() -> GeminiSettings:
    if genai is None:
        raise GeminiAgentError(
            "Chưa cài google-genai. Hãy thêm google-genai vào requirements.txt. "
            f"Chi tiết: {_GENAI_IMPORT_ERROR}"
        )

    api_key = str(_secret_or_env("GEMINI_API_KEY", "")).strip()

    if not api_key or api_key in {
        "API_KEY_THAT_CUA_BAN",
        "DAN_API_KEY_CUA_BAN_VAO_DAY",
    }:
        raise GeminiAgentError(
            "Chưa tìm thấy GEMINI_API_KEY trong Streamlit Secrets hoặc .streamlit/secrets.toml."
        )

    return GeminiSettings(
        api_key=api_key,
        model=str(_secret_or_env("GEMINI_MODEL", DEFAULT_MODEL)).strip()
        or DEFAULT_MODEL,
        max_attempts=_as_int(
            _secret_or_env("GEMINI_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS),
            DEFAULT_MAX_ATTEMPTS,
            1,
            4,
        ),
        retry_base_seconds=_as_float(
            _secret_or_env(
                "GEMINI_RETRY_BASE_SECONDS",
                DEFAULT_RETRY_BASE_SECONDS,
            ),
            DEFAULT_RETRY_BASE_SECONDS,
            3.0,
            30.0,
        ),
        max_output_tokens=_as_int(
            _secret_or_env(
                "GEMINI_MAX_OUTPUT_TOKENS",
                DEFAULT_MAX_OUTPUT_TOKENS,
            ),
            DEFAULT_MAX_OUTPUT_TOKENS,
            1024,
            8192,
        ),
        temperature=_as_float(
            _secret_or_env("GEMINI_TEMPERATURE", DEFAULT_TEMPERATURE),
            DEFAULT_TEMPERATURE,
            0.0,
            0.8,
        ),
        max_prompt_chars=_as_int(
            _secret_or_env(
                "GEMINI_MAX_PROMPT_CHARS",
                DEFAULT_MAX_PROMPT_CHARS,
            ),
            DEFAULT_MAX_PROMPT_CHARS,
            10000,
            120000,
        ),
        min_words=_as_int(
            _secret_or_env("GEMINI_MIN_WORDS", DEFAULT_MIN_WORDS),
            DEFAULT_MIN_WORDS,
            400,
            2500,
        ),
        min_words_bai12=_as_int(
            _secret_or_env(
                "GEMINI_MIN_WORDS_BAI12",
                DEFAULT_MIN_WORDS_BAI12,
            ),
            DEFAULT_MIN_WORDS_BAI12,
            800,
            4000,
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
            f"Gemini API đã được cấu hình. "
            f"Model: {settings.model}; "
            f"độ dài tối thiểu: {settings.min_words} từ; "
            f"Bài 12: {settings.min_words_bai12} từ."
        )
    except GeminiAgentError as error:
        return str(error)


EXERCISE_RUBRIC = {
    "1": """
Bài 1 — Cobb-Douglas + AI:
- Giải thích mô hình Cobb-Douglas mở rộng với K, L, D, AI, H và TFP.
- Phân tích kết quả ước lượng, MAPE, dự báo GDP 2025-2030.
- Phân rã tăng trưởng thành đóng góp vốn, lao động, số hóa, AI, nhân lực và TFP.
- Nêu ý nghĩa chính sách cho chuyển đổi số và tăng trưởng năng suất.
""",
    "2": """
Bài 2 — Quy hoạch tuyến tính phân bổ ngân sách số:
- Giải thích biến quyết định, hàm mục tiêu và ràng buộc.
- Phân tích nghiệm tối ưu, ngân sách phân bổ, giá trị mục tiêu.
- Phân tích shadow price/dual nếu có.
- So sánh các kịch bản ngân sách và kịch bản ưu tiên nhân lực số.
- Rút ra khuyến nghị phân bổ ngân sách.
""",
    "3": """
Bài 3 — Chỉ số ưu tiên ngành:
- Giải thích chuẩn hóa min-max, trọng số và cách đảo dấu Risk.
- Phân tích Priority_i của 10 ngành.
- Chỉ ra top ngành ưu tiên và ngành chưa nên ưu tiên.
- Phân tích độ nhạy theo trọng số AI Readiness, Growth, Inclusion, Spillover và Risk.
- Đưa ra khuyến nghị chính sách ngành.
""",
    "4": """
Bài 4 — LP ngành-vùng:
- Giải thích bài toán phân bổ ngân sách theo vùng và hạng mục.
- Phân tích ma trận phân bổ, nghiệm PuLP/CVXPY nếu có.
- So sánh mô hình có và không có ràng buộc công bằng.
- Phân tích chi phí công bằng và tác động tới vùng yếu.
- Đưa ra chính sách cân bằng giữa hiệu quả và công bằng vùng.
""",
    "5": """
Bài 5 — MIP chọn 15 dự án:
- Giải thích biến nhị phân, hàm mục tiêu, ngân sách, ràng buộc tiên quyết, loại trừ và bắt buộc.
- Phân tích danh mục dự án được chọn.
- Đánh giá lợi ích, chi phí, rủi ro và tính khả thi.
- So sánh kịch bản ngân sách nếu có.
- Đưa ra khuyến nghị danh mục đầu tư.
""",
    "6": """
Bài 6 — TOPSIS chọn vùng trung tâm AI:
- Giải thích phương pháp TOPSIS, ideal solution, anti-ideal solution.
- Phân tích điểm C_i* và thứ hạng các vùng.
- So sánh trọng số chuyên gia và Entropy nếu có.
- Phân tích độ nhạy theo AI Readiness.
- Khuyến nghị vùng nên ưu tiên phát triển trung tâm AI.
""",
    "7": """
Bài 7 — NSGA-II và Pareto:
- Giải thích tối ưu đa mục tiêu và Pareto frontier.
- Phân tích trade-off giữa tăng trưởng, bao trùm, môi trường và an ninh dữ liệu.
- Giải thích nghiệm thỏa hiệp theo TOPSIS.
- Phân tích chi phí cơ hội khi ưu tiên một mục tiêu.
- Đưa ra hàm ý chính sách cho lựa chọn cân bằng.
""",
    "8": """
Bài 8 — Tối ưu động:
- Giải thích mô hình tối ưu động theo thời gian.
- Phân tích đầu tư vào K, D, AI, H và ảnh hưởng tới TFP/GDP.
- So sánh front-loaded và back-loaded nếu có.
- Phân tích vai trò của chiết khấu, tiêu dùng và đầu tư.
- Đưa ra khuyến nghị lộ trình đầu tư.
""",
    "9": """
Bài 9 — Lao động và AI:
- Giải thích NetJob, đầu tư AI, đào tạo lại và lao động bị thay thế.
- Phân tích ngưỡng đào tạo để hạn chế mất việc.
- Đánh giá tác động tới nhóm lao động dễ tổn thương.
- Phân tích ràng buộc mất việc không quá 5% nếu có.
- Đưa ra khuyến nghị an sinh kỹ năng.
""",
    "10": """
Bài 10 — Stochastic programming:
- Giải thích mô hình hai giai đoạn dưới bất định.
- Phân tích RP, EV, EEV, WS, VSS, EVPI nếu có.
- So sánh quyết định theo kịch bản xác suất và robust minimax regret.
- Phân tích ý nghĩa của giá trị thông tin.
- Đưa ra khuyến nghị chính sách trong điều kiện bất định.
""",
    "11": """
Bài 11 — Q-learning/RL:
- Giải thích MDP, trạng thái, hành động, reward và policy.
- Phân tích chính sách học được bằng Q-learning.
- So sánh với rule-based hoặc random nếu có.
- Phân tích DQN nếu có.
- Nêu yêu cầu đạo đức, giải trình và kiểm soát AI trong ra quyết định.
""",
    "12": """
Bài 12 — Dashboard AIDEOM tích hợp:
- Phân tích dashboard tích hợp 11 mô hình trước.
- Phân tích GDP, CAGR, NetJob, training coverage, readiness, risk, ngân sách và cảnh báo rủi ro.
- So sánh các kịch bản chính sách, đặc biệt giải thích vì sao kịch bản được chọn đứng hạng cao.
- Phân tích biểu đồ/bảng trong dashboard: GDP, lao động, readiness, rủi ro, phân bổ ngân sách, xếp hạng kịch bản.
- Phân tích phân bổ vùng và nguy cơ bất bình đẳng vùng.
- Nêu hạn chế mô hình tích hợp và khuyến nghị dùng dashboard cho hoạch định chính sách.
""",
}


def _exercise_number(exercise_name: str) -> str | None:
    text = str(exercise_name).lower()
    match = re.search(r"bài\s*0?(\d{1,2})", text)
    if match:
        number = int(match.group(1))
        if 1 <= number <= 12:
            return str(number)
    return None


def _rubric(exercise_name: str) -> str:
    number = _exercise_number(exercise_name)
    if number and number in EXERCISE_RUBRIC:
        return EXERCISE_RUBRIC[number]

    return """
- Giải thích mô hình, dữ liệu đầu vào, biến và tham số.
- Phân tích kết quả định lượng.
- Phân tích biểu đồ, bảng và độ nhạy.
- Nêu hạn chế mô hình và khuyến nghị chính sách.
"""


def _target_words(settings: GeminiSettings, exercise_name: str) -> int:
    if _exercise_number(exercise_name) == "12":
        return settings.min_words_bai12
    return settings.min_words


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
        return "Không có tham số đầu vào được truyền sang Gemini."

    return "\n".join(
        f"- {key}: {_format_value(value)}"
        for key, value in values.items()
    )


def _truncate(text: str, max_chars: int) -> str:
    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    head = int(max_chars * 0.72)
    tail = max_chars - head

    return (
        text[:head]
        + "\n\n...[NỘI DUNG QUÁ DÀI ĐÃ ĐƯỢC RÚT GỌN Ở GIỮA]...\n\n"
        + text[-tail:]
    )


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
Bạn là chuyên gia mô hình ra quyết định, tối ưu hóa, kinh tế lượng, chính sách công và chuyển đổi số tại Việt Nam.

Nhiệm vụ của bạn là viết PHÂN TÍCH HỌC THUẬT cho kết quả mô hình trong web app AIDEOM-VN.

BÀI ĐANG PHÂN TÍCH
{exercise_name}

MÔ HÌNH / PHƯƠNG PHÁP
{model_name}

RUBRIC BẮT BUỘC CỦA BÀI
{_rubric(exercise_name)}

THAM SỐ ĐẦU VÀO
{_format_mapping(parameters)}

KẾT QUẢ PYTHON / DASHBOARD ĐÃ TÍNH
{result_summary}

CÂU HỎI CHÍNH SÁCH CẦN TRẢ LỜI
{policy_questions or "Tự rút ra câu hỏi chính sách phù hợp từ mô hình và kết quả."}

YÊU CẦU BẮT BUỘC
- Viết bằng tiếng Việt.
- Văn phong học thuật, phù hợp đưa vào báo cáo Word/PDF cuối kỳ.
- Viết tối thiểu khoảng {target_words} từ.
- Không trả lời ngắn.
- Không bịa số liệu.
- Chỉ dùng số liệu được cung cấp trong tham số và kết quả Python.
- Nếu không biết đơn vị của chỉ tiêu, ghi là “theo đơn vị mô hình”.
- Không nói kết quả là chắc chắn tuyệt đối; phải nhấn mạnh đây là mô hình hỗ trợ ra quyết định.
- Phân tích biểu đồ/bảng dựa trên tên chỉ tiêu và dữ liệu được cung cấp, không mô tả những biểu đồ không tồn tại.

CẤU TRÚC BẮT BUỘC

# 1. Tóm tắt điều hành
Nêu 3-5 phát hiện quan trọng nhất, kết quả nổi bật và ý nghĩa chính sách.

# 2. Bối cảnh và mục tiêu mô hình
Giải thích bài toán ra quyết định mà mô hình đang giải quyết trong bối cảnh Việt Nam.

# 3. Mô hình toán học và logic ra quyết định
Giải thích biến quyết định, tham số, hàm mục tiêu, ràng buộc hoặc logic xếp hạng/tối ưu.

# 4. Phân tích kết quả định lượng
Đọc các chỉ số chính, so sánh giá trị cao/thấp, giải thích ý nghĩa của nghiệm hoặc thứ hạng.

# 5. Phân tích biểu đồ và bảng trong dashboard
Giải thích các biểu đồ/bảng theo logic:
- Biểu đồ hoặc bảng thể hiện chỉ tiêu gì.
- Kết quả nào nổi bật.
- Kết quả đó có hàm ý chính sách gì.
- Có điểm bất thường hoặc cảnh báo nào không.

# 6. Phân tích độ nhạy, kịch bản và đánh đổi
Nếu có nhiều kịch bản, hãy so sánh kịch bản.
Nếu có trọng số/tham số, hãy giải thích khi thay đổi tham số thì kết luận có thể thay đổi thế nào.
Nêu rõ các trade-off giữa tăng trưởng, việc làm, công bằng, môi trường, rủi ro và năng lực thực thi.

# 7. Hàm ý kinh tế và chính sách công
Phân tích tác động tới:
- tăng trưởng
- năng suất
- chuyển đổi số
- việc làm
- công bằng vùng/ngành
- rủi ro môi trường, công nghệ hoặc vĩ mô

# 8. Hạn chế mô hình
Bắt buộc nêu:
- dữ liệu có thể là mô phỏng hoặc dữ liệu thứ cấp
- kết quả phụ thuộc vào giả định, trọng số và tham số
- mô hình chưa thay thế được phản biện chuyên gia
- cần kiểm định bằng dữ liệu thực tế trước khi áp dụng chính sách

# 9. Khuyến nghị chính sách
Chia thành:
- Ngắn hạn
- Trung hạn
- Dài hạn

# 10. Kết luận
Tóm tắt lại lựa chọn chính sách và nhấn mạnh mô hình là công cụ hỗ trợ ra quyết định, không thay thế quyết định thể chế.

LƯU Ý RIÊNG CHO BÀI 12
Nếu đây là Bài 12, bắt buộc phải có:
- So sánh kịch bản chính sách.
- Phân tích biểu đồ GDP, NetJob, readiness, risk, phân bổ ngân sách và cảnh báo rủi ro nếu dữ liệu có.
- Phân tích vì sao kịch bản cân bằng có ý nghĩa hơn kịch bản chỉ tối đa hóa một mục tiêu.
- Nêu hạn chế của dashboard tích hợp.
"""

    return _truncate(prompt, settings.max_prompt_chars)


def _expand_prompt(
    exercise_name: str,
    original_prompt: str,
    original_answer: str,
    settings: GeminiSettings,
) -> str:
    target_words = _target_words(settings, exercise_name)

    prompt = f"""
Câu trả lời trước chưa đủ sâu cho báo cáo cuối kỳ.

Hãy viết lại thành bản phân tích đầy đủ hơn, tối thiểu khoảng {target_words} từ.
Bắt buộc giữ đủ 10 mục lớn.
Không bịa số liệu. Không thêm nguồn ngoài. Chỉ dùng dữ liệu trong prompt gốc.

PROMPT GỐC
{original_prompt}

CÂU TRẢ LỜI CŨ
{original_answer}

Hãy trả về bản hoàn chỉnh, không xin lỗi, không giải thích quá trình.
"""

    return _truncate(prompt, settings.max_prompt_chars)


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)

    if text:
        return str(text).strip()

    chunks = []

    try:
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(str(part_text))
    except Exception:
        pass

    return "\n".join(chunks).strip()


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
        "rate limit",
        "quota",
        "timeout",
        "timed out",
        "deadline exceeded",
        "temporarily",
    ]

    return any(marker in text for marker in markers)


def _friendly_error(error: Exception, model: str) -> str:
    text = f"{type(error).__name__}: {error}".lower()

    if "429" in text or "quota" in text or "rate limit" in text:
        return (
            "Gemini đang hết quota hoặc bị giới hạn tần suất. "
            "Hãy chờ 30-60 phút rồi bấm lại, hoặc giảm GEMINI_MAX_OUTPUT_TOKENS."
        )

    if "503" in text or "unavailable" in text or "overloaded" in text:
        return (
            "Gemini đang quá tải tạm thời. Hãy chờ một lát rồi bấm lại."
        )

    if "api key" in text or "401" in text or "403" in text:
        return (
            "API key Gemini không hợp lệ hoặc chưa có quyền gọi model. "
            "Hãy kiểm tra GEMINI_API_KEY trong Streamlit Secrets."
        )

    if "model" in text and ("not found" in text or "invalid" in text):
        return (
            f"Không gọi được model {model}. "
            "Hãy đổi GEMINI_MODEL sang gemini-2.5-flash-lite hoặc gemini-2.5-flash."
        )

    return f"Không gọi được Gemini API. Chi tiết: {error}"


def _call_gemini(prompt: str, settings: GeminiSettings) -> str:
    try:
        client = genai.Client(api_key=settings.api_key)
    except Exception as error:
        raise GeminiAgentError(
            f"Không khởi tạo được Gemini client. Chi tiết: {error}"
        ) from error

    config = None

    if types is not None:
        config = types.GenerateContentConfig(
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
        )

    last_error: Exception | None = None

    for attempt in range(settings.max_attempts):
        try:
            response = client.models.generate_content(
                model=settings.model,
                contents=prompt,
                config=config,
            )

            text = _extract_text(response)

            if not text:
                raise RuntimeError("Gemini trả về phản hồi rỗng.")

            return text

        except Exception as error:
            last_error = error

            if not _is_retryable(error):
                raise GeminiAgentError(
                    _friendly_error(error, settings.model)
                ) from error

            if attempt < settings.max_attempts - 1:
                delay = min(
                    settings.retry_base_seconds * (2 ** attempt)
                    + random.uniform(0, 0.7),
                    30.0,
                )
                time.sleep(delay)

    raise GeminiAgentError(
        _friendly_error(
            last_error or RuntimeError("Không rõ lỗi Gemini."),
            settings.model,
        )
    ) from last_error


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\wÀ-ỹ]+\b", str(text), flags=re.UNICODE))


def _has_core_sections(text: str) -> bool:
    lower = str(text).lower()

    required = [
        "tóm tắt",
        "mô hình",
        "kết quả",
        "biểu đồ",
        "hạn chế",
        "khuyến nghị",
        "kết luận",
    ]

    return all(item in lower for item in required)


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

    too_short = _word_count(answer) < int(target_words * 0.65)
    missing_sections = not _has_core_sections(answer)

    if settings.auto_expand and (too_short or missing_sections):
        try:
            expanded = _call_gemini(
                _expand_prompt(
                    exercise_name=exercise_name,
                    original_prompt=prompt,
                    original_answer=answer,
                    settings=settings,
                ),
                settings,
            )

            if _word_count(expanded) > _word_count(answer):
                answer = expanded

        except Exception:
            pass

    return answer


def test_gemini_connection() -> str:
    settings = get_gemini_settings()

    return _call_gemini(
        "Hãy trả lời đúng một câu tiếng Việt: Kết nối Gemini thành công.",
        settings,
    )


def ask_gemini(prompt: str) -> str:
    settings = get_gemini_settings()
    return _call_gemini(prompt, settings)


def analyze_with_gemini(prompt: str) -> str:
    return ask_gemini(prompt)
