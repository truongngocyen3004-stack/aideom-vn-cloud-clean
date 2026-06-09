from __future__ import annotations

from typing import Any

import streamlit as st

from services.ai_agent import (
    GeminiAgentError,
    analyze_result,
    gemini_configuration_message,
    gemini_is_configured,
)


def clear_ai_result(
    page_key: str,
) -> None:
    st.session_state.pop(
        f"{page_key}_gemini_analysis",
        None,
    )


def render_ai_panel(
    *,
    page_key: str,
    exercise_name: str,
    model_name: str,
    parameters: dict[str, Any],
    result_summary: str,
    policy_questions: str = "",
    button_label: str = "✨ Phân tích kết quả bằng Gemini",
    can_analyze: bool = True,
) -> str | None:
    """
    Panel Gemini chuẩn dùng cho mọi trang.

    Các page hiện tại vẫn tương thích trực tiếp với
    services.ai_agent.analyze_result; file này được cung cấp
    để có thể rút gọn code trong các lần chỉnh tiếp theo.
    """

    configured = (
        gemini_is_configured()
    )

    if configured:
        st.success(
            gemini_configuration_message()
        )
    else:
        st.warning(
            gemini_configuration_message()
        )

    with st.expander(
        "Xem nội dung sẽ gửi cho Gemini",
        expanded=False,
    ):
        st.text_area(
            "Tóm tắt kết quả",
            value=result_summary.strip(),
            height=380,
            disabled=True,
            key=f"{page_key}_ai_preview",
        )

    clicked = st.button(
        button_label,
        disabled=(
            not configured
            or not can_analyze
        ),
        use_container_width=True,
        key=f"{page_key}_gemini_button",
    )

    if clicked:
        try:
            with st.spinner(
                "Gemini đang phân tích kết quả..."
            ):
                analysis = analyze_result(
                    exercise_name=(
                        exercise_name
                    ),
                    model_name=(
                        model_name
                    ),
                    parameters=(
                        parameters
                    ),
                    result_summary=(
                        result_summary
                    ),
                    policy_questions=(
                        policy_questions
                    ),
                )

                st.session_state[
                    f"{page_key}_gemini_analysis"
                ] = analysis

        except GeminiAgentError as error:
            st.error(
                str(error)
            )

    saved = st.session_state.get(
        f"{page_key}_gemini_analysis"
    )

    if saved:
        st.markdown(
            saved
        )

        st.download_button(
            "⬇️ Tải phân tích Gemini",
            data=saved.encode(
                "utf-8"
            ),
            file_name=(
                f"{page_key}_phan_tich_gemini.md"
            ),
            mime="text/markdown",
            use_container_width=True,
            key=f"{page_key}_gemini_download",
        )

    return saved
