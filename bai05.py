from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.bai05_model import (
    load_project_data,
    run_full_bai05,
)
from services.ai_agent import (
    GeminiAgentError,
    analyze_result,
    gemini_is_configured,
)
from ui.theme import page_header


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "vietnam_projects_bai5.csv"

PINK = "#D989A5"
ROSE = "#F4B8C8"
LAVENDER = "#CDB8E5"
MINT = "#A8D5D1"
YELLOW = "#F2D7A7"
BLUE = "#A9C9E8"
TEXT = "#503743"
GRID = "#EEDFE5"
BG = "#FFF9FB"

PASTEL_SEQUENCE = [
    PINK,
    ROSE,
    LAVENDER,
    MINT,
    YELLOW,
    BLUE,
    "#D6B5A5",
    "#BFD7B5",
    "#D8B4C8",
    "#B7C7E8",
]


def style_plotly(
    fig: go.Figure,
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 460,
) -> go.Figure:
    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
        },
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={
            "family": "Arial",
            "color": TEXT,
            "size": 13,
        },
        title_font={
            "size": 19,
            "color": TEXT,
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title_text="",
        height=height,
        margin={
            "l": 60,
            "r": 35,
            "t": 72,
            "b": 65,
        },
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "font_color": TEXT,
        },
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#DCCBD3",
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor="#DCCBD3",
    )

    return fig


def csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    return dataframe.to_csv(
        index=False
    ).encode("utf-8-sig")


def selection_label(
    selected: int,
) -> str:
    return (
        "Được chọn"
        if int(selected) == 1
        else "Không chọn"
    )


page_header(
    "Bài 5 — MIP lựa chọn danh mục 15 dự án chuyển đổi số",
    "Tối ưu danh mục dự án giai đoạn 2026–2030 với biến nhị phân, ngân sách đa năm, loại trừ, tiên quyết, an ninh mạng và rủi ro tiến độ.",
)

st.markdown(
    """
    <div style="
        background:#FFF1F6;
        border:1px solid #F0D5DF;
        border-radius:16px;
        padding:18px 20px;
        margin-bottom:16px;
        color:#503743;
    ">
        <b>Quyết định:</b>
        yᵢ = 1 nếu chọn dự án i, yᵢ = 0 nếu không chọn.
        <br>
        <b>Mục tiêu cơ sở:</b>
        tối đa hóa tổng lợi ích NPV của danh mục.
        <br>
        <b>Ràng buộc:</b>
        ngân sách 5 năm, ngân sách năm 1–2, loại trừ P1–P2,
        tiên quyết P8/P13 cần P12, chính phủ số, P14 bắt buộc
        và số lượng dự án.
    </div>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(
        f"Không tìm thấy file dữ liệu: {DATA_PATH}"
    )
    st.stop()

project_data = load_project_data(
    DATA_PATH
)

with st.expander(
    "⚙️ Thiết lập mô hình và kịch bản",
    expanded=True,
):
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        total_budget = st.number_input(
            "Ngân sách 5 năm (tỷ VND)",
            min_value=40000.0,
            max_value=150000.0,
            value=80000.0,
            step=5000.0,
        )

    with p2:
        early_budget = st.number_input(
            "Ngân sách năm 1–2 (tỷ VND)",
            min_value=20000.0,
            max_value=100000.0,
            value=40000.0,
            step=2500.0,
        )

    with p3:
        min_projects = st.number_input(
            "Số dự án tối thiểu",
            min_value=1,
            max_value=15,
            value=7,
            step=1,
        )

    with p4:
        max_projects = st.number_input(
            "Số dự án tối đa",
            min_value=1,
            max_value=15,
            value=11,
            step=1,
        )

    p5, p6 = st.columns(2)

    with p5:
        expanded_budget = st.number_input(
            "Ngân sách mở rộng 5.4.2",
            min_value=80000.0,
            max_value=150000.0,
            value=100000.0,
            step=5000.0,
        )

    with p6:
        synergy_bonus = st.number_input(
            "Bonus cộng hưởng P8–P13",
            min_value=0.0,
            max_value=30000.0,
            value=5000.0,
            step=1000.0,
        )

    run_clicked = st.button(
        "🌸 Chạy toàn bộ mô hình Bài 5",
        type="primary",
        use_container_width=True,
    )

signature = (
    total_budget,
    early_budget,
    int(min_projects),
    int(max_projects),
    expanded_budget,
    synergy_bonus,
)

if (
    run_clicked
    or "bai05_result"
    not in st.session_state
    or st.session_state.get(
        "bai05_signature"
    ) != signature
):
    try:
        with st.spinner(
            "Đang giải mô hình cơ sở và các kịch bản MIP..."
        ):
            st.session_state[
                "bai05_result"
            ] = run_full_bai05(
                csv_path=DATA_PATH,
                total_budget=total_budget,
                early_budget=early_budget,
                min_projects=int(
                    min_projects
                ),
                max_projects=int(
                    max_projects
                ),
                expanded_budget=(
                    expanded_budget
                ),
                synergy_bonus=(
                    synergy_bonus
                ),
            )

            st.session_state[
                "bai05_signature"
            ] = signature

    except ValueError as error:
        st.error(str(error))
        st.stop()

result = st.session_state[
    "bai05_result"
]

base = result["base"]
expanded = result["expanded"]
redundancy = result[
    "redundancy"
]
risk = result["risk"]
without_p14 = result[
    "without_p14"
]
force_p15 = result[
    "force_p15"
]
exclude_p15 = result[
    "exclude_p15"
]
synergy = result[
    "synergy"
]

tabs = st.tabs([
    "5.1 — Bối cảnh",
    "5.2 — Danh mục 15 dự án",
    "5.3 — Mô hình toán học",
    "5.4.1 — Mô hình cơ sở",
    "5.4.2 — Ngân sách 100.000",
    "5.4.3 — Redundancy P1–P2",
    "5.4.4 — Rủi ro tiến độ",
    "5.5 — Thảo luận chính sách",
    "✨ Phân tích AI",
])

with tabs[0]:
    st.subheader(
        "5.1 — Bối cảnh chương trình chuyển đổi số 2026–2030"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Dự án ứng cử",
        "15",
    )
    c2.metric(
        "Ngân sách cơ sở",
        f"{total_budget:,.0f} tỷ VND",
    )
    c3.metric(
        "Ngân sách năm 1–2",
        f"{early_budget:,.0f} tỷ VND",
    )
    c4.metric(
        "Số lĩnh vực",
        f"{project_data['field'].nunique()}",
    )

    context = project_data.copy()

    fig_context = px.scatter(
        context,
        x="cost_total",
        y="benefit_npv",
        size="benefit_cost_ratio",
        color="field",
        text="code",
        hover_name="project_name",
        hover_data=[
            "cost_year_1_2",
            "completion_probability",
            "benefit_cost_ratio",
        ],
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
        size_max=50,
    )

    fig_context.update_traces(
        textposition="top center"
    )

    fig_context = style_plotly(
        fig_context,
        title=(
            "Bản đồ chi phí–lợi ích của 15 dự án ứng cử"
        ),
        x_title="Chi phí 5 năm (tỷ VND)",
        y_title="NPV lợi ích (tỷ VND)",
        height=560,
    )

    st.plotly_chart(
        fig_context,
        use_container_width=True,
    )

    st.info(
        "Một dự án có NPV cao chưa chắc được chọn. "
        "MIP đánh giá đồng thời quy mô vốn, áp lực giải ngân sớm, "
        "quan hệ tiên quyết, loại trừ, an ninh mạng và số lượng dự án."
    )

with tabs[1]:
    st.subheader(
        "5.2 — Danh mục 15 dự án ứng cử"
    )

    display = project_data.rename(
        columns={
            "code": "Mã",
            "project_name": "Tên dự án",
            "field": "Lĩnh vực",
            "cost_total":
                "Chi phí 5 năm",
            "benefit_npv":
                "NPV lợi ích",
            "cost_year_1_2":
                "Chi phí năm 1–2",
            "cost_year_3_5":
                "Chi phí năm 3–5",
            "completion_probability":
                "Xác suất đúng tiến độ",
            "benefit_cost_ratio":
                "NPV/Chi phí",
            "expected_benefit":
                "Lợi ích kỳ vọng",
        }
    )

    st.dataframe(
        display.round(3),
        use_container_width=True,
        hide_index=True,
    )

    ratio_plot = (
        project_data.sort_values(
            "benefit_cost_ratio",
            ascending=True,
        )
    )

    fig_ratio = px.bar(
        ratio_plot,
        x="benefit_cost_ratio",
        y="code",
        orientation="h",
        color="field",
        text_auto=".2f",
        hover_name="project_name",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_ratio = style_plotly(
        fig_ratio,
        title=(
            "Xếp hạng tỷ suất NPV/Chi phí"
        ),
        x_title="NPV/Chi phí",
        y_title="Mã dự án",
        height=590,
    )

    st.plotly_chart(
        fig_ratio,
        use_container_width=True,
    )

    cost_long = project_data.melt(
        id_vars=[
            "code",
            "project_name",
        ],
        value_vars=[
            "cost_year_1_2",
            "cost_year_3_5",
        ],
        var_name="Giai đoạn",
        value_name="Chi phí",
    )

    cost_long["Giai đoạn"] = (
        cost_long["Giai đoạn"]
        .replace({
            "cost_year_1_2":
                "Năm 1–2",
            "cost_year_3_5":
                "Năm 3–5",
        })
    )

    fig_cost = px.bar(
        cost_long,
        x="code",
        y="Chi phí",
        color="Giai đoạn",
        barmode="stack",
        hover_name="project_name",
        color_discrete_sequence=[
            PINK,
            MINT,
        ],
    )

    fig_cost = style_plotly(
        fig_cost,
        title=(
            "Cơ cấu chi phí theo hai giai đoạn"
        ),
        x_title="Mã dự án",
        y_title="Chi phí (tỷ VND)",
        height=500,
    )

    st.plotly_chart(
        fig_cost,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Tải dữ liệu 15 dự án",
        data=csv_bytes(
            display
        ),
        file_name=(
            "bai05_du_lieu_15_du_an.csv"
        ),
        mime="text/csv",
    )

with tabs[2]:
    st.subheader(
        "5.3 — Mô hình toán học"
    )

    st.markdown(
        "**Biến quyết định nhị phân**"
    )

    st.latex(
        r"""
        y_i \in \{0,1\},
        \quad i=1,\ldots,15
        """
    )

    st.markdown(
        "**Hàm mục tiêu cơ sở**"
    )

    st.latex(
        r"""
        \max Z =
        \sum_{i=1}^{15}
        B_i y_i
        """
    )

    constraints = pd.DataFrame({
        "Mã": [
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
        ],
        "Nội dung": [
            "Ngân sách tổng 5 năm",
            "Ngân sách năm 1–2",
            "Chỉ chọn một trung tâm dữ liệu P1/P2",
            "P8 chỉ được chọn nếu có P12",
            "P13 chỉ được chọn nếu có P12",
            "Có ít nhất P4/P5 và bắt buộc P14",
            "Số dự án từ mức tối thiểu đến tối đa",
        ],
        "Biểu thức": [
            "Σ Cᵢyᵢ ≤ 80.000",
            "Σ C₁,ᵢyᵢ ≤ 40.000",
            "y₁ + y₂ ≤ 1",
            "y₈ ≤ y₁₂",
            "y₁₃ ≤ y₁₂",
            "y₄ + y₅ ≥ 1; y₁₄ = 1",
            "7 ≤ Σyᵢ ≤ 11",
        ],
    })

    st.dataframe(
        constraints,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "**Mở rộng rủi ro**"
    )

    st.latex(
        r"""
        \max E[Z] =
        \sum_i p_i B_i y_i
        """
    )

    st.markdown(
        "**Mở rộng cộng hưởng P8–P13**"
    )

    st.latex(
        r"""
        z_{8,13}\le y_8,\quad
        z_{8,13}\le y_{13},\quad
        z_{8,13}\ge y_8+y_{13}-1
        """
    )

    st.latex(
        r"""
        \max Z =
        \sum_i B_i y_i
        +
        Bonus_{8,13}z_{8,13}
        """
    )

    st.success(
        "Mô hình là knapsack tổng quát hóa có biến nhị phân. "
        "Tính khó không nằm ở số dự án mà ở quan hệ hệ thống giữa các dự án."
    )

with tabs[3]:
    st.subheader(
        "Câu 5.4.1 — Giải mô hình cơ sở bằng PuLP/CBC"
    )

    if not base["success"]:
        st.error(base["status"])
    else:
        b1, b2, b3, b4 = st.columns(4)

        b1.metric(
            "Trạng thái",
            base["status"],
        )
        b2.metric(
            "Z* NPV",
            f"{base['objective']:,.0f} tỷ VND",
        )
        b3.metric(
            "Tổng chi phí",
            f"{base['total_cost']:,.0f} tỷ VND",
        )
        b4.metric(
            "NPV/Chi phí",
            f"{base['npv_per_cost']:.3f}",
        )

        selected = base[
            "selected_df"
        ].copy()

        selected_display = selected[
            [
                "code",
                "project_name",
                "field",
                "cost_total",
                "cost_year_1_2",
                "benefit_npv",
                "benefit_cost_ratio",
            ]
        ].rename(
            columns={
                "code": "Mã",
                "project_name":
                    "Tên dự án",
                "field": "Lĩnh vực",
                "cost_total":
                    "Chi phí 5 năm",
                "cost_year_1_2":
                    "Chi phí năm 1–2",
                "benefit_npv":
                    "NPV lợi ích",
                "benefit_cost_ratio":
                    "NPV/Chi phí",
            }
        )

        st.dataframe(
            selected_display.round(3),
            use_container_width=True,
            hide_index=True,
        )

        full = base[
            "full_df"
        ].copy()

        full["Trạng thái"] = (
            full["selected"]
            .map({
                1: "Được chọn",
                0: "Không chọn",
            })
        )

        fig_selected = px.bar(
            full,
            x="code",
            y="benefit_npv",
            color="Trạng thái",
            hover_name="project_name",
            hover_data=[
                "field",
                "cost_total",
            ],
            color_discrete_map={
                "Được chọn": PINK,
                "Không chọn": "#E8E1E4",
            },
        )

        fig_selected = style_plotly(
            fig_selected,
            title=(
                "Dự án được chọn trong danh mục tối ưu"
            ),
            x_title="Mã dự án",
            y_title="NPV lợi ích (tỷ VND)",
            height=500,
        )

        st.plotly_chart(
            fig_selected,
            use_container_width=True,
        )

        budget_df = pd.DataFrame({
            "Nhóm": [
                "Ngân sách 5 năm",
                "Ngân sách năm 1–2",
            ],
            "Đã dùng": [
                base["total_cost"],
                base["early_cost"],
            ],
            "Hạn mức": [
                total_budget,
                early_budget,
            ],
        })

        budget_df["Còn lại"] = (
            budget_df["Hạn mức"]
            - budget_df["Đã dùng"]
        )

        fig_budget = px.bar(
            budget_df.melt(
                id_vars="Nhóm",
                value_vars=[
                    "Đã dùng",
                    "Còn lại",
                ],
                var_name="Trạng thái",
                value_name="Ngân sách",
            ),
            x="Nhóm",
            y="Ngân sách",
            color="Trạng thái",
            barmode="stack",
            text_auto=",.0f",
            color_discrete_sequence=[
                PINK,
                "#E8E1E4",
            ],
        )

        fig_budget = style_plotly(
            fig_budget,
            title=(
                "Mức sử dụng hai hạn mức ngân sách"
            ),
            x_title="Hạn mức",
            y_title="Tỷ VND",
        )

        st.plotly_chart(
            fig_budget,
            use_container_width=True,
        )

        st.markdown(
            "#### Kiểm tra ràng buộc"
        )

        st.dataframe(
            base[
                "constraint_check"
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Tải danh mục tối ưu cơ sở",
            data=csv_bytes(
                selected_display
            ),
            file_name=(
                "bai05_541_danh_muc_co_so.csv"
            ),
            mime="text/csv",
        )

with tabs[4]:
    st.subheader(
        "Câu 5.4.2 — Nới ngân sách lên 100.000 tỷ VND"
    )

    if not expanded["success"]:
        st.error(
            expanded["status"]
        )
    else:
        e1, e2, e3, e4 = st.columns(4)

        e1.metric(
            "Z* cơ sở",
            f"{base['objective']:,.0f}",
        )
        e2.metric(
            "Z* ngân sách mở rộng",
            f"{expanded['objective']:,.0f}",
        )
        e3.metric(
            "Thay đổi Z*",
            (
                f"{expanded['objective'] - base['objective']:,.0f}"
            ),
        )
        e4.metric(
            "Chi phí năm 1–2",
            f"{expanded['early_cost']:,.0f}/"
            f"{early_budget:,.0f}",
        )

        st.dataframe(
            result[
                "expanded_changes"
            ],
            use_container_width=True,
            hide_index=True,
        )

        scenario_compare = result[
            "scenario_table"
        ]

        fig_scenario = px.bar(
            scenario_compare.dropna(
                subset=[
                    "Giá trị mục tiêu"
                ]
            ),
            x="Kịch bản",
            y="Giá trị mục tiêu",
            color="Kịch bản",
            text_auto=",.0f",
            color_discrete_sequence=(
                PASTEL_SEQUENCE
            ),
        )

        fig_scenario.update_layout(
            showlegend=False
        )

        fig_scenario = style_plotly(
            fig_scenario,
            title=(
                "So sánh giá trị mục tiêu giữa các kịch bản"
            ),
            x_title="Kịch bản",
            y_title="Giá trị mục tiêu",
            height=530,
        )

        fig_scenario.update_xaxes(
            tickangle=-15
        )

        st.plotly_chart(
            fig_scenario,
            use_container_width=True,
        )

        if (
            set(
                expanded[
                    "selected_codes"
                ]
            )
            == set(
                base[
                    "selected_codes"
                ]
            )
        ):
            st.warning(
                "Danh mục không thay đổi dù ngân sách 5 năm tăng. "
                "Nguyên nhân là ràng buộc ngân sách năm 1–2 vẫn giữ ở "
                f"{early_budget:,.0f} tỷ VND và đang là nút thắt chính."
            )
        else:
            added = (
                set(
                    expanded[
                        "selected_codes"
                    ]
                )
                - set(
                    base[
                        "selected_codes"
                    ]
                )
            )

            removed = (
                set(
                    base[
                        "selected_codes"
                    ]
                )
                - set(
                    expanded[
                        "selected_codes"
                    ]
                )
            )

            st.success(
                f"Dự án được thêm: {sorted(added)}; "
                f"dự án bị loại: {sorted(removed)}."
            )

with tabs[5]:
    st.subheader(
        "Câu 5.4.3 — Bắt buộc cả P1 và P2 để bảo đảm redundancy"
    )

    if not redundancy[
        "success"
    ]:
        st.error(
            "Kịch bản không khả thi: "
            + redundancy["status"]
        )
    else:
        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Khả thi?",
            "Có",
        )
        r2.metric(
            "Z* redundancy",
            f"{redundancy['objective']:,.0f}",
        )
        r3.metric(
            "Thay đổi Z*",
            (
                f"{redundancy['objective'] - base['objective']:,.0f}"
            ),
        )
        r4.metric(
            "Chi phí năm 1–2",
            (
                f"{redundancy['early_cost']:,.0f}/"
                f"{early_budget:,.0f}"
            ),
        )

        st.dataframe(
            result[
                "redundancy_changes"
            ],
            use_container_width=True,
            hide_index=True,
        )

        compare_red = pd.DataFrame({
            "Kịch bản": [
                "Cơ sở",
                "Bắt buộc P1 và P2",
            ],
            "Z*": [
                base["objective"],
                redundancy[
                    "objective"
                ],
            ],
            "Chi phí 5 năm": [
                base["total_cost"],
                redundancy[
                    "total_cost"
                ],
            ],
        })

        fig_red = px.bar(
            compare_red,
            x="Kịch bản",
            y="Z*",
            color="Kịch bản",
            text_auto=",.0f",
            color_discrete_sequence=[
                PINK,
                MINT,
            ],
        )

        fig_red.update_layout(
            showlegend=False
        )

        fig_red = style_plotly(
            fig_red,
            title=(
                "Đánh đổi giữa redundancy và tổng NPV"
            ),
            x_title="Kịch bản",
            y_title="Z* (tỷ VND)",
        )

        st.plotly_chart(
            fig_red,
            use_container_width=True,
        )

        st.info(
            "Để kiểm tra redundancy, mô hình gỡ ràng buộc loại trừ "
            "y₁+y₂≤1 và thay bằng y₁=y₂=1. "
            "Kịch bản có thể khả thi nhưng tạo chi phí cơ hội vì hai dự án "
            "cùng chiếm ngân sách giải ngân sớm."
        )

with tabs[6]:
    st.subheader(
        "Câu 5.4.4 — Tối đa hóa lợi ích kỳ vọng có rủi ro"
    )

    if not risk["success"]:
        st.error(
            risk["status"]
        )
    else:
        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "E[Z] tối ưu",
            f"{risk['objective']:,.0f}",
        )
        q2.metric(
            "NPV danh nghĩa",
            f"{risk['nominal_benefit']:,.0f}",
        )
        q3.metric(
            "Lợi ích kỳ vọng danh mục",
            f"{risk['expected_benefit']:,.0f}",
        )
        q4.metric(
            "Số dự án",
            str(
                len(
                    risk[
                        "selected_codes"
                    ]
                )
            ),
        )

        st.dataframe(
            result[
                "risk_changes"
            ],
            use_container_width=True,
            hide_index=True,
        )

        risk_projects = project_data.copy()

        risk_projects[
            "NPV danh nghĩa"
        ] = risk_projects[
            "benefit_npv"
        ]

        risk_projects[
            "Lợi ích kỳ vọng"
        ] = risk_projects[
            "expected_benefit"
        ]

        risk_long = risk_projects.melt(
            id_vars=[
                "code",
                "project_name",
            ],
            value_vars=[
                "NPV danh nghĩa",
                "Lợi ích kỳ vọng",
            ],
            var_name="Loại lợi ích",
            value_name="Giá trị",
        )

        fig_risk = px.bar(
            risk_long,
            x="code",
            y="Giá trị",
            color="Loại lợi ích",
            barmode="group",
            hover_name="project_name",
            color_discrete_sequence=[
                PINK,
                MINT,
            ],
        )

        fig_risk = style_plotly(
            fig_risk,
            title=(
                "NPV danh nghĩa và lợi ích kỳ vọng sau điều chỉnh rủi ro"
            ),
            x_title="Mã dự án",
            y_title="Tỷ VND",
            height=520,
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True,
        )

        selected_risk = risk[
            "selected_df"
        ][
            [
                "code",
                "project_name",
                "field",
                "completion_probability",
                "benefit_npv",
                "expected_benefit",
            ]
        ].copy()

        st.dataframe(
            selected_risk.round(3),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Tải danh mục điều chỉnh rủi ro",
            data=csv_bytes(
                selected_risk
            ),
            file_name=(
                "bai05_544_danh_muc_rui_ro.csv"
            ),
            mime="text/csv",
        )

with tabs[7]:
    st.subheader(
        "5.5 — Thảo luận chính sách"
    )

    p15_selected = (
        "P15"
        in base.get(
            "selected_codes",
            [],
        )
    )

    p14_cost = np.nan

    if (
        base["success"]
        and without_p14[
            "success"
        ]
    ):
        p14_cost = (
            without_p14[
                "objective"
            ]
            - base[
                "objective"
            ]
        )

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(
            """
            <div style="
                background:#FFF1F6;
                border:1px solid #F0D5DF;
                border-radius:16px;
                padding:18px;
                min-height:360px;
            ">
                <h4 style="color:#503743;">
                    5.5a — Trường hợp P15 Open Data
                </h4>
            """,
            unsafe_allow_html=True,
        )

        if p15_selected:
            st.write(
                "Với đúng dữ liệu và ràng buộc hiện tại, "
                "**P15 thực tế được chọn**, nên giả định "
                "“mô hình bỏ qua P15” không xảy ra."
            )

            st.write(
                f"Nếu cưỡng chế loại P15, Z* giảm "
                f"**{base['objective'] - exclude_p15['objective']:,.0f}**. "
                "Điều này xác nhận P15 tạo giá trị tốt nhờ chi phí thấp "
                "và tỷ suất NPV/chi phí cao."
            )
        else:
            st.write(
                "P15 bị bỏ qua dù có tỷ suất cao vì mô hình tối ưu "
                "toàn danh mục, không xếp hạng riêng lẻ. "
                "Giới hạn số dự án, ngân sách sớm và quan hệ tiên quyết "
                "có thể khiến dự án tỷ suất cao vẫn bị loại."
            )

        st.write(
            "Về chính sách, dữ liệu mở còn có lợi ích nền tảng và ngoại tác "
            "mà NPV riêng lẻ có thể chưa đo hết."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with s2:
        st.markdown(
            """
            <div style="
                background:#F7F0FC;
                border:1px solid #E4D5F1;
                border-radius:16px;
                padding:18px;
                min-height:360px;
            ">
                <h4 style="color:#503743;">
                    5.5b — P14 an ninh mạng bắt buộc
                </h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"Bỏ yêu cầu bắt buộc P14 làm Z* tăng "
            f"**{p14_cost:,.0f} tỷ VND** trong mô hình."
        )

        st.write(
            "Khoản chênh lệch này là chi phí cơ hội định lượng của "
            "an ninh mạng, nhưng không phải bằng chứng nên bỏ P14. "
            "SOC quốc gia giảm rủi ro hệ thống, bảo vệ dữ liệu và "
            "tăng độ tin cậy của toàn bộ danh mục."
        )

        st.write(
            "Một mô hình chỉ tối đa NPV sẽ đánh giá thấp lợi ích phòng ngừa "
            "sự cố hiếm nhưng thiệt hại rất lớn."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with s3:
        st.markdown(
            """
            <div style="
                background:#EEF8F7;
                border:1px solid #D1E9E6;
                border-radius:16px;
                padding:18px;
                min-height:360px;
            ">
                <h4 style="color:#503743;">
                    5.5c — Cộng hưởng P8 và P13
                </h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            "Hiệu ứng cộng hưởng được mô hình hóa bằng biến nhị phân "
            "z₈,₁₃, bằng 1 khi cả P8 và P13 được chọn."
        )

        st.write(
            f"Với bonus **{synergy_bonus:,.0f} tỷ VND**, "
            f"kịch bản cộng hưởng có Z* = "
            f"**{synergy['objective']:,.0f}** và "
            f"{'chọn' if {'P8','P13'}.issubset(set(synergy['selected_codes'])) else 'không chọn đồng thời'} "
            "P8–P13."
        )

        st.write(
            "Cách này phản ánh AI cần hạ tầng tính toán và chip, "
            "trong khi hệ sinh thái bán dẫn hưởng lợi từ nhu cầu AI."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "#### Bảng so sánh toàn bộ kịch bản"
    )

    st.dataframe(
        result[
            "scenario_table"
        ].round(3),
        use_container_width=True,
        hide_index=True,
    )

    p15_compare = pd.DataFrame({
        "Kịch bản": [
            "Cơ sở",
            "Bắt buộc có P15",
            "Loại P15",
        ],
        "Z*": [
            base[
                "objective"
            ],
            force_p15[
                "objective"
            ],
            exclude_p15[
                "objective"
            ],
        ],
    })

    fig_p15 = px.bar(
        p15_compare,
        x="Kịch bản",
        y="Z*",
        color="Kịch bản",
        text_auto=",.0f",
        color_discrete_sequence=[
            PINK,
            MINT,
            LAVENDER,
        ],
    )

    fig_p15.update_layout(
        showlegend=False
    )

    fig_p15 = style_plotly(
        fig_p15,
        title=(
            "Kiểm định giả định chính sách về P15"
        ),
        x_title="Kịch bản",
        y_title="Z* (tỷ VND)",
    )

    st.plotly_chart(
        fig_p15,
        use_container_width=True,
    )

    p14_compare = pd.DataFrame({
        "Kịch bản": [
            "P14 bắt buộc",
            "P14 không bắt buộc",
        ],
        "Z*": [
            base["objective"],
            without_p14[
                "objective"
            ],
        ],
    })

    fig_p14 = px.bar(
        p14_compare,
        x="Kịch bản",
        y="Z*",
        color="Kịch bản",
        text_auto=",.0f",
        color_discrete_sequence=[
            ROSE,
            MINT,
        ],
    )

    fig_p14.update_layout(
        showlegend=False
    )

    fig_p14 = style_plotly(
        fig_p14,
        title=(
            "Chi phí cơ hội mô phỏng của ràng buộc P14"
        ),
        x_title="Chính sách an ninh mạng",
        y_title="Z* (tỷ VND)",
    )

    st.plotly_chart(
        fig_p14,
        use_container_width=True,
    )

    st.markdown(
        "#### Danh mục thay đổi khi bổ sung cộng hưởng P8–P13"
    )

    st.dataframe(
        result[
            "synergy_changes"
        ],
        use_container_width=True,
        hide_index=True,
    )

with tabs[8]:
    st.subheader(
        "Tác nhân AI phân tích kết quả Bài 5"
    )

    configured = gemini_is_configured()

    if configured:
        st.success(
            "Gemini API đã được cấu hình."
        )
    else:
        st.warning(
            "Chưa tìm thấy GEMINI_API_KEY trong "
            ".streamlit/secrets.toml."
        )

    if base["success"]:
        result_summary = f"""
BÀI 5 — MIP LỰA CHỌN DỰ ÁN CHUYỂN ĐỔI SỐ

Kết quả cơ sở:
- Trạng thái: {base['status']}
- Dự án chọn: {base['selected_codes']}
- Z*: {base['objective']:.2f}
- Tổng chi phí: {base['total_cost']:.2f}
- Chi phí năm 1-2: {base['early_cost']:.2f}
- NPV/Chi phí: {base['npv_per_cost']:.4f}

Ngân sách mở rộng:
- Dự án chọn: {expanded.get('selected_codes')}
- Z*: {expanded.get('objective')}

Redundancy P1-P2:
- Khả thi: {redundancy['success']}
- Dự án chọn: {redundancy.get('selected_codes')}
- Z*: {redundancy.get('objective')}

Rủi ro:
- Dự án chọn: {risk.get('selected_codes')}
- E[Z]: {risk.get('objective')}
- NPV danh nghĩa: {risk.get('nominal_benefit')}

P15:
- Được chọn trong cơ sở: {p15_selected}
- Z khi loại P15: {exclude_p15.get('objective')}

P14:
- Z khi không bắt buộc P14: {without_p14.get('objective')}
- Chi phí cơ hội P14: {p14_cost}

Cộng hưởng:
- Bonus P8-P13: {synergy_bonus}
- Dự án chọn: {synergy.get('selected_codes')}
- Giá trị mục tiêu: {synergy.get('objective')}
"""
    else:
        result_summary = (
            "Mô hình cơ sở chưa giải thành công."
        )

    policy_questions = """
1. Vì sao P15 được chọn hoặc bị loại và kết quả có hợp lý về chính sách không?
2. Ràng buộc P14 làm giảm Z* bao nhiêu và có nên duy trì không?
3. Nới ngân sách 5 năm nhưng giữ ngân sách năm 1-2 cho thấy nút thắt nào?
4. Bắt buộc cả P1 và P2 tạo ra đánh đổi redundancy–hiệu quả ra sao?
5. Điều chỉnh rủi ro làm danh mục thay đổi như thế nào?
6. Cộng hưởng P8–P13 nên được lượng hóa và quản trị thế nào?
"""

    with st.expander(
        "Xem dữ liệu sẽ gửi cho Gemini"
    ):
        st.text_area(
            "Tóm tắt kết quả",
            value=result_summary.strip(),
            height=420,
            disabled=True,
        )

    analyze_clicked = st.button(
        "✨ Phân tích kết quả bằng Gemini",
        disabled=(
            not configured
            or not base["success"]
        ),
        use_container_width=True,
        key="gemini_bai05",
    )

    if analyze_clicked:
        with st.spinner(
            "Gemini đang phân tích kết quả Bài 5..."
        ):
            try:
                analysis = analyze_result(
                    exercise_name=(
                        "Bài 5 — MIP lựa chọn 15 dự án "
                        "chuyển đổi số quốc gia"
                    ),
                    model_name=(
                        "Generalized knapsack MIP bằng PuLP/CBC "
                        "với ngân sách đa năm, precedence, exclusion, "
                        "risk adjustment và synergy"
                    ),
                    parameters={
                        "Ngân sách 5 năm":
                            total_budget,
                        "Ngân sách năm 1-2":
                            early_budget,
                        "Số dự án tối thiểu":
                            int(min_projects),
                        "Số dự án tối đa":
                            int(max_projects),
                        "Ngân sách mở rộng":
                            expanded_budget,
                        "Bonus P8-P13":
                            synergy_bonus,
                    },
                    result_summary=(
                        result_summary.strip()
                    ),
                    policy_questions=(
                        policy_questions.strip()
                    ),
                )

                st.session_state[
                    "bai05_gemini_analysis"
                ] = analysis

            except GeminiAgentError as error:
                st.error(str(error))

    saved_analysis = st.session_state.get(
        "bai05_gemini_analysis"
    )

    if saved_analysis:
        st.markdown(
            """
            <div style="
                background:#FFF1F6;
                border:1px solid #F0D5DF;
                border-left:5px solid #D989A5;
                border-radius:16px;
                padding:18px 20px;
                margin-top:16px;
            ">
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            saved_analysis
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇️ Tải phân tích Gemini Bài 5",
            data=saved_analysis.encode(
                "utf-8"
            ),
            file_name=(
                "bai05_phan_tich_gemini.md"
            ),
            mime="text/markdown",
            use_container_width=True,
        )
