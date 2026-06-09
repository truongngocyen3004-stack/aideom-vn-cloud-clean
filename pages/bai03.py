from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.bai03_model import (
    DEFAULT_RAW_WEIGHTS,
    DISPLAY_CRITERIA,
    GROWTH_WEIGHTS,
    INCLUSIVE_WEIGHTS,
    calculate_priority,
    compare_policy_orientations,
    load_sector_data,
    mining_diagnostics,
    ai_weight_sensitivity,
)
from services.ai_agent import (
    GeminiAgentError,
    analyze_result,
    gemini_is_configured,
)
from ui.theme import page_header


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "vietnam_sectors_2024.csv"

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
    "#E7B8A0",
    "#BFD7B5",
    "#D8B4C8",
    "#B7C7E8",
]

PASTEL_HEATMAP = [
    [0.00, "#FFF7FA"],
    [0.20, "#FBE4EC"],
    [0.40, "#EEDCF5"],
    [0.60, "#D8E9F3"],
    [0.80, "#BFE3DD"],
    [1.00, "#7DBFB4"],
]


def style_plotly(
    fig: go.Figure,
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 450,
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
            "r": 30,
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


def format_top3(
    ranking: pd.DataFrame,
) -> str:
    return " → ".join(
        ranking.head(3)[
            "sector_name_vi"
        ].tolist()
    )


page_header(
    "Bài 3 — Chỉ số ưu tiên ngành Priorityᵢ cho 10 ngành Việt Nam",
    "Chuẩn hóa min-max, xếp hạng đa tiêu chí, kiểm định độ nhạy trọng số AI và so sánh định hướng tăng trưởng với định hướng bao trùm.",
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
        <b>Mô hình:</b>
        Priorityᵢ = a₁Growthᵢ + a₂Productivityᵢ + a₃Spilloverᵢ
        + a₄Exportᵢ + a₅Employmentᵢ + a₆AIReadinessᵢ
        + a₇Safetyᵢ.
        <br>
        <b>Lưu ý:</b> Safety là rủi ro tự động hóa đã đảo chiều,
        nên điểm cao nghĩa là rủi ro thấp.
    </div>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(
        f"Không tìm thấy file dữ liệu: {DATA_PATH}"
    )
    st.stop()

sector_data = load_sector_data(
    DATA_PATH
)

with st.expander(
    "⚙️ Thiết lập bộ trọng số",
    expanded=True,
):
    st.caption(
        "Bộ trọng số gốc trong đề có tổng 1,10. "
        "Website tự chuẩn hóa tỷ lệ về tổng bằng 1; "
        "việc này không làm thay đổi thứ hạng."
    )

    w1, w2, w3, w4 = st.columns(4)
    w5, w6, w7, w8 = st.columns(4)

    with w1:
        weight_growth = st.slider(
            "a₁ — Tăng trưởng",
            0.00,
            0.50,
            float(
                DEFAULT_RAW_WEIGHTS[
                    "Growth_norm"
                ]
            ),
            0.01,
        )

    with w2:
        weight_productivity = st.slider(
            "a₂ — Năng suất",
            0.00,
            0.50,
            float(
                DEFAULT_RAW_WEIGHTS[
                    "Productivity_norm"
                ]
            ),
            0.01,
        )

    with w3:
        weight_spillover = st.slider(
            "a₃ — Lan tỏa",
            0.00,
            0.50,
            float(
                DEFAULT_RAW_WEIGHTS[
                    "Spillover_norm"
                ]
            ),
            0.01,
        )

    with w4:
        weight_export = st.slider(
            "a₄ — Xuất khẩu",
            0.00,
            0.50,
            float(
                DEFAULT_RAW_WEIGHTS[
                    "Export_norm"
                ]
            ),
            0.01,
        )

    with w5:
        weight_employment = st.slider(
            "a₅ — Việc làm",
            0.00,
            0.50,
            float(
                DEFAULT_RAW_WEIGHTS[
                    "Employment_norm"
                ]
            ),
            0.01,
        )

    with w6:
        weight_ai = st.slider(
            "a₆ — AI Readiness",
            0.00,
            0.50,
            float(
                DEFAULT_RAW_WEIGHTS[
                    "AIReadiness_norm"
                ]
            ),
            0.01,
        )

    with w7:
        weight_safety = st.slider(
            "a₇ — Giảm rủi ro",
            0.00,
            0.50,
            float(
                DEFAULT_RAW_WEIGHTS[
                    "Safety_norm"
                ]
            ),
            0.01,
        )

    raw_weights = {
        "Growth_norm": weight_growth,
        "Productivity_norm": weight_productivity,
        "Spillover_norm": weight_spillover,
        "Export_norm": weight_export,
        "Employment_norm": weight_employment,
        "AIReadiness_norm": weight_ai,
        "Safety_norm": weight_safety,
    }

    raw_sum = float(
        sum(raw_weights.values())
    )

    with w8:
        st.metric(
            "Tổng trọng số gốc",
            f"{raw_sum:.2f}",
            "Tự chuẩn hóa về 1,00",
        )

    run_clicked = st.button(
        "🌸 Chạy toàn bộ mô hình Bài 3",
        use_container_width=True,
        type="primary",
    )

signature = tuple(
    raw_weights.values()
)

if (
    run_clicked
    or "bai03_result" not in st.session_state
    or st.session_state.get(
        "bai03_signature"
    ) != signature
):
    with st.spinner(
        "Đang chuẩn hóa dữ liệu, xếp hạng, phân tích độ nhạy và so sánh chính sách..."
    ):
        default_result = (
            calculate_priority(
                sector_data,
                raw_weights,
            )
        )

        sensitivity_result = (
            ai_weight_sensitivity(
                sector_data,
                base_weights=raw_weights,
            )
        )

        orientation_result = (
            compare_policy_orientations(
                sector_data
            )
        )

        mining_result = mining_diagnostics(
            default_result["ranking"]
        )

        st.session_state[
            "bai03_result"
        ] = {
            "default": default_result,
            "sensitivity":
                sensitivity_result,
            "orientations":
                orientation_result,
            "mining": mining_result,
        }

        st.session_state[
            "bai03_signature"
        ] = signature

result = st.session_state[
    "bai03_result"
]

default_result = result["default"]
ranking = default_result["ranking"]
normalized = default_result["normalized"]
sensitivity = result["sensitivity"]
orientations = result["orientations"]
mining = result["mining"]

tabs = st.tabs([
    "3.1 — Bối cảnh",
    "3.2–3.3 — Mô hình & dữ liệu",
    "3.4.1 — Chuẩn hóa",
    "3.4.2 — Priority & xếp hạng",
    "3.4.3 — Độ nhạy AI",
    "3.4.4 — Hai định hướng",
    "3.5 — Thảo luận chính sách",
    "✨ Phân tích AI",
])

with tabs[0]:
    st.subheader(
        "3.1 — Bối cảnh lựa chọn ngành ưu tiên"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Số ngành",
        f"{len(sector_data)}",
    )

    c2.metric(
        "Số tiêu chí",
        "7",
    )

    c3.metric(
        "AI Readiness cao nhất",
        (
            f"{sector_data['ai_readiness_0_100'].max():.0f}/100"
        ),
    )

    c4.metric(
        "Việc làm lớn nhất",
        (
            f"{sector_data['labor_million'].max():.2f} triệu"
        ),
    )

    bubble = px.scatter(
        sector_data,
        x="growth_rate_2024_pct",
        y="ai_readiness_0_100",
        size="labor_million",
        color="automation_risk_pct",
        hover_name="sector_name_vi",
        text="sector_name_vi",
        color_continuous_scale=[
            "#BFE3DD",
            "#F2D7A7",
            "#D989A5",
        ],
        size_max=52,
    )

    bubble.update_traces(
        textposition="top center"
    )

    bubble = style_plotly(
        bubble,
        title=(
            "Tăng trưởng, mức sẵn sàng AI, việc làm và rủi ro tự động hóa"
        ),
        x_title="Tăng trưởng năm 2024 (%)",
        y_title="AI Readiness (0–100)",
        height=560,
    )

    bubble.update_layout(
        coloraxis_colorbar_title=(
            "Rủi ro<br>TĐH (%)"
        )
    )

    st.plotly_chart(
        bubble,
        use_container_width=True,
    )

    st.info(
        "Một ngành không thể được ưu tiên chỉ vì có một chỉ tiêu cao. "
        "Mô hình phải cân bằng tốc độ tăng trưởng, năng suất, lan tỏa, "
        "xuất khẩu, việc làm, mức sẵn sàng AI và rủi ro tự động hóa."
    )

with tabs[1]:
    st.subheader(
        "3.2 — Mô hình toán học"
    )

    st.latex(
        r"""
        Priority_i =
        a_1 Growth_i +
        a_2 Productivity_i +
        a_3 Spillover_i +
        a_4 Export_i +
        a_5 Employment_i +
        a_6 AIReadiness_i +
        a_7 Safety_i
        """
    )

    st.latex(
        r"""
        x^{norm}_i =
        \frac{x_i-\min(x)}
        {\max(x)-\min(x)}
        """
    )

    st.latex(
        r"""
        Safety_i =
        \frac{\max(Risk)-Risk_i}
        {\max(Risk)-\min(Risk)}
        """
    )

    st.caption(
        "Cách cộng trọng số Safety tương đương về thứ hạng với "
        "việc trừ trọng số của Risk chuẩn hóa trực tiếp."
    )

    st.subheader(
        "3.3 — Dữ liệu 10 ngành Việt Nam năm 2024"
    )

    display_data = sector_data.rename(
        columns={
            "sector_name_vi": "Ngành",
            "growth_rate_2024_pct":
                "Tăng trưởng (%)",
            "labor_productivity_million_VND":
                "Năng suất (triệu VND/LĐ)",
            "spillover_coef_0_1":
                "Lan tỏa (0–1)",
            "export_billion_USD":
                "Xuất khẩu (tỷ USD)",
            "labor_million":
                "Việc làm (triệu LĐ)",
            "ai_readiness_0_100":
                "AI Readiness",
            "automation_risk_pct":
                "Rủi ro TĐH (%)",
        }
    )

    st.dataframe(
        display_data.round(3),
        use_container_width=True,
        hide_index=True,
    )

    weight_table = pd.DataFrame({
        "Tiêu chí": [
            DISPLAY_CRITERIA[key]
            for key in default_result[
                "weights_normalized"
            ]
        ],
        "Trọng số gốc": [
            raw_weights[key]
            for key in default_result[
                "weights_normalized"
            ]
        ],
        "Trọng số sau chuẩn hóa": [
            default_result[
                "weights_normalized"
            ][key]
            for key in default_result[
                "weights_normalized"
            ]
        ],
    })

    fig_weights = px.bar(
        weight_table,
        x="Tiêu chí",
        y=[
            "Trọng số gốc",
            "Trọng số sau chuẩn hóa",
        ],
        barmode="group",
        color_discrete_sequence=[
            ROSE,
            MINT,
        ],
    )

    fig_weights = style_plotly(
        fig_weights,
        title=(
            "Bộ trọng số đang sử dụng"
        ),
        x_title="Tiêu chí",
        y_title="Trọng số",
    )

    st.plotly_chart(
        fig_weights,
        use_container_width=True,
    )

with tabs[2]:
    st.subheader(
        "Câu 3.4.1 — Ma trận chuẩn hóa min-max"
    )

    normalized_display = normalized.rename(
        columns={
            "sector_name_vi": "Ngành",
            **{
                key: value
                for key, value
                in DISPLAY_CRITERIA.items()
            },
        }
    )

    st.dataframe(
        normalized_display.round(4),
        use_container_width=True,
        hide_index=True,
    )

    heat_matrix = (
        normalized_display
        .set_index("Ngành")
    )

    heatmap = go.Figure(
        data=go.Heatmap(
            z=heat_matrix.values,
            x=heat_matrix.columns,
            y=heat_matrix.index,
            colorscale=PASTEL_HEATMAP,
            zmin=0,
            zmax=1,
            text=np.round(
                heat_matrix.values,
                2,
            ),
            texttemplate="%{text}",
            colorbar={
                "title": "Điểm",
            },
            hovertemplate=(
                "Ngành=%{y}<br>"
                "Tiêu chí=%{x}<br>"
                "Điểm=%{z:.3f}"
                "<extra></extra>"
            ),
        )
    )

    heatmap = style_plotly(
        heatmap,
        title=(
            "Heatmap ma trận chuẩn hóa của 10 ngành"
        ),
        x_title="Tiêu chí",
        y_title="Ngành",
        height=610,
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True,
    )

    st.success(
        "Tất cả tiêu chí đã được đưa về thang [0,1]. "
        "Riêng rủi ro tự động hóa đã được đảo chiều thành Safety: "
        "điểm cao nghĩa là rủi ro thấp."
    )

    st.download_button(
        "⬇️ Tải ma trận chuẩn hóa",
        data=csv_bytes(
            normalized_display
        ),
        file_name=(
            "bai03_341_ma_tran_chuan_hoa.csv"
        ),
        mime="text/csv",
    )

with tabs[3]:
    st.subheader(
        "Câu 3.4.2 — Tính Priority và xếp hạng"
    )

    top3 = ranking.head(3)

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Top 1",
        top3.iloc[0][
            "sector_name_vi"
        ],
        f"Priority {top3.iloc[0]['Priority']:.3f}",
    )

    r2.metric(
        "Top 2",
        top3.iloc[1][
            "sector_name_vi"
        ],
        f"Priority {top3.iloc[1]['Priority']:.3f}",
    )

    r3.metric(
        "Top 3",
        top3.iloc[2][
            "sector_name_vi"
        ],
        f"Priority {top3.iloc[2]['Priority']:.3f}",
    )

    r4.metric(
        "Khoảng cách Top 1–2",
        (
            f"{top3.iloc[0]['Priority'] - top3.iloc[1]['Priority']:.4f}"
        ),
    )

    ranking_plot = ranking.sort_values(
        "Priority",
        ascending=True,
    )

    fig_rank = px.bar(
        ranking_plot,
        x="Priority",
        y="sector_name_vi",
        orientation="h",
        color="Rank",
        text="Priority",
        color_continuous_scale=[
            "#F4B8C8",
            "#CDB8E5",
            "#A8D5D1",
        ],
    )

    fig_rank.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig_rank = style_plotly(
        fig_rank,
        title=(
            "Xếp hạng Priority của 10 ngành"
        ),
        x_title="Priority",
        y_title="Ngành",
        height=590,
    )

    fig_rank.update_layout(
        coloraxis_colorbar_title="Hạng"
    )

    st.plotly_chart(
        fig_rank,
        use_container_width=True,
    )

    contribution_columns = {
        "Contribution_Growth_norm":
            "Tăng trưởng",
        "Contribution_Productivity_norm":
            "Năng suất",
        "Contribution_Spillover_norm":
            "Lan tỏa",
        "Contribution_Export_norm":
            "Xuất khẩu",
        "Contribution_Employment_norm":
            "Việc làm",
        "Contribution_AIReadiness_norm":
            "AI Readiness",
        "Contribution_Safety_norm":
            "Giảm rủi ro",
    }

    contributions = ranking[
        [
            "sector_name_vi",
            *contribution_columns.keys(),
        ]
    ].melt(
        id_vars="sector_name_vi",
        var_name="Tiêu chí",
        value_name="Đóng góp",
    )

    contributions["Tiêu chí"] = (
        contributions["Tiêu chí"]
        .map(contribution_columns)
    )

    fig_contributions = px.bar(
        contributions,
        x="sector_name_vi",
        y="Đóng góp",
        color="Tiêu chí",
        barmode="stack",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_contributions = style_plotly(
        fig_contributions,
        title=(
            "Cấu phần tạo nên Priority của từng ngành"
        ),
        x_title="Ngành",
        y_title="Đóng góp vào Priority",
        height=530,
    )

    fig_contributions.update_xaxes(
        tickangle=-25
    )

    st.plotly_chart(
        fig_contributions,
        use_container_width=True,
    )

    display_ranking = ranking[
        [
            "Rank",
            "sector_name_vi",
            "Priority",
            "Growth_norm",
            "Productivity_norm",
            "Spillover_norm",
            "Export_norm",
            "Employment_norm",
            "AIReadiness_norm",
            "Safety_norm",
        ]
    ].rename(
        columns={
            "Rank": "Hạng",
            "sector_name_vi": "Ngành",
            "Growth_norm": "Tăng trưởng",
            "Productivity_norm": "Năng suất",
            "Spillover_norm": "Lan tỏa",
            "Export_norm": "Xuất khẩu",
            "Employment_norm": "Việc làm",
            "AIReadiness_norm": "AI Readiness",
            "Safety_norm": "Giảm rủi ro",
        }
    )

    st.dataframe(
        display_ranking.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Tải bảng xếp hạng mặc định",
        data=csv_bytes(
            display_ranking
        ),
        file_name=(
            "bai03_342_xep_hang_priority.csv"
        ),
        mime="text/csv",
    )

with tabs[4]:
    st.subheader(
        "Câu 3.4.3 — Độ nhạy theo trọng số AI Readiness"
    )

    sensitivity_summary = sensitivity[
        "summary"
    ]
    rank_matrix_long = sensitivity[
        "ranks"
    ]
    score_matrix_long = sensitivity[
        "scores"
    ]

    st.dataframe(
        sensitivity_summary.round(4),
        use_container_width=True,
        hide_index=True,
    )

    rank_pivot = rank_matrix_long.pivot(
        index="Ngành",
        columns="Trọng số AI ban đầu",
        values="Xếp hạng",
    )

    rank_heatmap = go.Figure(
        data=go.Heatmap(
            z=rank_pivot.values,
            x=[
                f"{value:.2f}"
                for value
                in rank_pivot.columns
            ],
            y=rank_pivot.index,
            colorscale=[
                [0.00, "#A8D5D1"],
                [0.50, "#EEDCF5"],
                [1.00, "#F4B8C8"],
            ],
            reversescale=False,
            text=rank_pivot.values.astype(int),
            texttemplate="%{text}",
            colorbar={
                "title": "Hạng",
            },
            hovertemplate=(
                "Ngành=%{y}<br>"
                "a₆ gốc=%{x}<br>"
                "Hạng=%{z:.0f}"
                "<extra></extra>"
            ),
        )
    )

    rank_heatmap = style_plotly(
        rank_heatmap,
        title=(
            "Heatmap thứ hạng khi a₆ thay đổi từ 0,05 đến 0,40"
        ),
        x_title="Trọng số AI Readiness a₆ trước chuẩn hóa",
        y_title="Ngành",
        height=620,
    )

    st.plotly_chart(
        rank_heatmap,
        use_container_width=True,
    )

    fig_sensitivity_line = px.line(
        score_matrix_long,
        x="Trọng số AI ban đầu",
        y="Priority",
        color="Ngành",
        markers=True,
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_sensitivity_line = style_plotly(
        fig_sensitivity_line,
        title=(
            "Điểm Priority thay đổi khi tăng trọng số AI Readiness"
        ),
        x_title="Trọng số AI Readiness a₆ trước chuẩn hóa",
        y_title="Priority",
        height=560,
    )

    st.plotly_chart(
        fig_sensitivity_line,
        use_container_width=True,
    )

    if (
        sensitivity[
            "top3_configurations"
        ] == 1
    ):
        st.success(
            "Top-3 ổn định trong toàn bộ khoảng a₆ = 0,05–0,40. "
            "Kết quả này cho thấy thành phần nhóm ưu tiên khá vững, "
            "dù điểm và vị trí bên trong nhóm có thể thay đổi."
        )
    else:
        st.warning(
            f"Top-3 thay đổi giữa "
            f"{sensitivity['top3_configurations']} cấu hình. "
            "Kết quả phụ thuộc đáng kể vào mức ưu tiên dành cho AI."
        )

    st.download_button(
        "⬇️ Tải bảng độ nhạy AI",
        data=csv_bytes(
            sensitivity_summary
        ),
        file_name=(
            "bai03_343_do_nhay_ai.csv"
        ),
        mime="text/csv",
    )

with tabs[5]:
    st.subheader(
        "Câu 3.4.4 — So sánh hai định hướng chính sách"
    )

    growth_ranking = orientations[
        "growth_ranking"
    ]
    inclusive_ranking = orientations[
        "inclusive_ranking"
    ]
    comparison = orientations[
        "comparison"
    ]

    g1, g2 = st.columns(2)

    with g1:
        st.markdown(
            "#### Định hướng tăng trưởng"
        )

        st.dataframe(
            growth_ranking.head(3)[
                [
                    "Rank",
                    "sector_name_vi",
                    "Priority",
                ]
            ].rename(
                columns={
                    "Rank": "Hạng",
                    "sector_name_vi": "Ngành",
                }
            ).round(4),
            use_container_width=True,
            hide_index=True,
        )

    with g2:
        st.markdown(
            "#### Định hướng bao trùm"
        )

        st.dataframe(
            inclusive_ranking.head(3)[
                [
                    "Rank",
                    "sector_name_vi",
                    "Priority",
                ]
            ].rename(
                columns={
                    "Rank": "Hạng",
                    "sector_name_vi": "Ngành",
                }
            ).round(4),
            use_container_width=True,
            hide_index=True,
        )

    weight_compare = pd.concat(
        [
            orientations[
                "growth_weights"
            ].assign(
                **{
                    "Định hướng":
                        "Tăng trưởng"
                }
            ),
            orientations[
                "inclusive_weights"
            ].assign(
                **{
                    "Định hướng":
                        "Bao trùm"
                }
            ),
        ],
        ignore_index=True,
    )

    fig_weight_compare = px.bar(
        weight_compare,
        x="Tiêu chí",
        y="Trọng số",
        color="Định hướng",
        barmode="group",
        color_discrete_sequence=[
            PINK,
            MINT,
        ],
    )

    fig_weight_compare = style_plotly(
        fig_weight_compare,
        title=(
            "Hai bộ trọng số phản ánh hai mục tiêu chính sách"
        ),
        x_title="Tiêu chí",
        y_title="Trọng số",
    )

    st.plotly_chart(
        fig_weight_compare,
        use_container_width=True,
    )

    rank_long = comparison.melt(
        id_vars="Ngành",
        value_vars=[
            "Rank - Tăng trưởng",
            "Rank - Bao trùm",
        ],
        var_name="Định hướng",
        value_name="Xếp hạng",
    )

    fig_slope = px.line(
        rank_long,
        x="Định hướng",
        y="Xếp hạng",
        color="Ngành",
        markers=True,
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_slope.update_yaxes(
        autorange="reversed",
        dtick=1,
    )

    fig_slope = style_plotly(
        fig_slope,
        title=(
            "Thứ hạng ngành thay đổi khi mục tiêu chính sách thay đổi"
        ),
        x_title="Bộ trọng số",
        y_title="Xếp hạng",
        height=590,
    )

    st.plotly_chart(
        fig_slope,
        use_container_width=True,
    )

    st.dataframe(
        comparison.sort_values(
            "Rank - Tăng trưởng"
        ).round(4),
        use_container_width=True,
        hide_index=True,
    )

    growth_top = format_top3(
        growth_ranking
    )
    inclusive_top = format_top3(
        inclusive_ranking
    )

    if set(
        growth_ranking.head(3)[
            "sector_name_vi"
        ]
    ) == set(
        inclusive_ranking.head(3)[
            "sector_name_vi"
        ]
    ):
        st.info(
            "Hai định hướng tạo ra cùng ba ngành trong nhóm Top-3, "
            "nhưng thứ tự ưu tiên khác nhau:\n\n"
            f"- Tăng trưởng: **{growth_top}**\n"
            f"- Bao trùm: **{inclusive_top}**"
        )
    else:
        st.warning(
            "Thành phần Top-3 thay đổi giữa hai định hướng:\n\n"
            f"- Tăng trưởng: **{growth_top}**\n"
            f"- Bao trùm: **{inclusive_top}**"
        )

    st.download_button(
        "⬇️ Tải bảng so sánh hai định hướng",
        data=csv_bytes(
            comparison
        ),
        file_name=(
            "bai03_344_so_sanh_dinh_huong.csv"
        ),
        mime="text/csv",
    )

with tabs[6]:
    st.subheader(
        "Mục 3.5 — Thảo luận chính sách"
    )

    top3_names = ranking.head(3)[
        "sector_name_vi"
    ].tolist()

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(
            """
            <div style="
                background:#FFF1F6;
                border:1px solid #F0D5DF;
                border-radius:16px;
                padding:18px;
                min-height:310px;
            ">
                <h4 style="color:#503743;">
                    3.5a — Ba ngành ưu tiên
                </h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            "Theo bộ trọng số hiện tại, ba ngành dẫn đầu là:"
        )

        for index, sector in enumerate(
            top3_names,
            start=1,
        ):
            st.markdown(
                f"**{index}. {sector}**"
            )

        st.write(
            "Kết quả phù hợp về mặt định hướng với mục tiêu "
            "thúc đẩy khoa học, công nghệ, đổi mới sáng tạo "
            "và chuyển đổi số; tuy nhiên đây là bằng chứng mô hình, "
            "không thay thế đánh giá pháp lý và kế hoạch ngành."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown(
            """
            <div style="
                background:#F7F0FC;
                border:1px solid #E4D5F1;
                border-radius:16px;
                padding:18px;
                min-height:310px;
            ">
                <h4 style="color:#503743;">
                    3.5b — Trường hợp Khai khoáng
                </h4>
            """,
            unsafe_allow_html=True,
        )

        mining_rank = int(
            ranking.loc[
                ranking[
                    "sector_name_vi"
                ] == "Khai khoáng",
                "Rank",
            ].iloc[0]
        )

        st.write(
            f"Khai khoáng đứng hạng **{mining_rank}**. "
            "Ngành có năng suất chuẩn hóa rất cao nhưng bị kéo xuống "
            "bởi tăng trưởng âm, lan tỏa thấp, việc làm thấp, "
            "AI Readiness trung bình và rủi ro tự động hóa cao."
        )

        st.write(
            "Điều này minh họa lợi ích của MCDM: "
            "không để một tiêu chí đơn lẻ quyết định toàn bộ ưu tiên."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with p3:
        st.markdown(
            """
            <div style="
                background:#EEF8F7;
                border:1px solid #D1E9E6;
                border-radius:16px;
                padding:18px;
                min-height:310px;
            ">
                <h4 style="color:#503743;">
                    3.5c — Ai quyết định trọng số?
                </h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            "Nên dùng mô hình ba lớp: chuyên gia kỹ thuật xây dựng "
            "và kiểm định mô hình; hội đồng chính sách lựa chọn "
            "trọng số chính thức; doanh nghiệp, địa phương và người "
            "lao động tham gia đối thoại, phản biện."
        )

        st.write(
            "Quy trình này cân bằng năng lực chuyên môn, "
            "trách nhiệm công và tính chính danh chính sách."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "#### Minh chứng định lượng cho trường hợp Khai khoáng"
    )

    st.dataframe(
        mining.round(4),
        use_container_width=True,
        hide_index=True,
    )

    mining_plot = mining[
        mining["Chỉ tiêu"].isin(
            [
                "Năng suất chuẩn hóa",
                "Tăng trưởng chuẩn hóa",
                "Lan tỏa chuẩn hóa",
                "Xuất khẩu chuẩn hóa",
                "Việc làm chuẩn hóa",
                "AI Readiness chuẩn hóa",
                "An toàn trước TĐH",
            ]
        )
    ]

    fig_mining = px.bar(
        mining_plot,
        x="Chỉ tiêu",
        y="Giá trị",
        color="Chỉ tiêu",
        text_auto=".2f",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_mining.update_layout(
        showlegend=False
    )

    fig_mining = style_plotly(
        fig_mining,
        title=(
            "Hồ sơ đa tiêu chí của ngành Khai khoáng"
        ),
        x_title="Tiêu chí",
        y_title="Điểm chuẩn hóa",
        height=490,
    )

    st.plotly_chart(
        fig_mining,
        use_container_width=True,
    )

    governance = pd.DataFrame({
        "Chủ thể": [
            "Chuyên gia kỹ thuật",
            "Hội đồng chính sách",
            "Đối thoại công khai",
        ],
        "Vai trò phù hợp": [
            "Thiết kế tiêu chí, dữ liệu, chuẩn hóa và phân tích độ nhạy.",
            "Phê duyệt mục tiêu, cân bằng tăng trưởng, bao trùm và an sinh.",
            "Phản biện tác động xã hội, khả năng thực thi và tính công bằng.",
        ],
        "Rủi ro nếu quyết định một mình": [
            "Quá thiên về chỉ tiêu đo được và bỏ sót giá trị xã hội.",
            "Có thể chịu ảnh hưởng của ưu tiên nhiệm kỳ hoặc lợi ích bộ ngành.",
            "Có thể kéo dài quyết định nếu thiếu khung kỹ thuật rõ ràng.",
        ],
    })

    st.markdown(
        "#### Khung governance đề xuất"
    )

    st.dataframe(
        governance,
        use_container_width=True,
        hide_index=True,
    )

with tabs[7]:
    st.subheader(
        "Tác nhân AI phân tích kết quả Bài 3"
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

    result_summary = f"""
BÀI 3 — CHỈ SỐ ƯU TIÊN NGÀNH

Bộ trọng số gốc:
{raw_weights}

Bộ trọng số sau chuẩn hóa:
{default_result['weights_normalized']}

Top-3 mặc định:
1. {ranking.iloc[0]['sector_name_vi']} — {ranking.iloc[0]['Priority']:.4f}
2. {ranking.iloc[1]['sector_name_vi']} — {ranking.iloc[1]['Priority']:.4f}
3. {ranking.iloc[2]['sector_name_vi']} — {ranking.iloc[2]['Priority']:.4f}

Độ nhạy AI:
- Số cấu hình Top-3 khác nhau: {sensitivity['top3_configurations']}
{sensitivity['summary'].round(4).to_string(index=False)}

Top-3 định hướng tăng trưởng:
{format_top3(orientations['growth_ranking'])}

Top-3 định hướng bao trùm:
{format_top3(orientations['inclusive_ranking'])}

Chẩn đoán Khai khoáng:
{mining.round(4).to_string(index=False)}
"""

    policy_questions = """
1. Ba ngành nào nên được ưu tiên chuyển đổi số và AI trước?
2. Kết quả có phù hợp về định hướng với Nghị quyết 57-NQ/TW không?
3. Vì sao Khai khoáng có năng suất cao nhưng không nằm trong nhóm ưu tiên?
4. Bộ trọng số nên được quyết định theo cơ chế governance nào?
5. Độ nhạy theo trọng số AI cho thấy kết quả vững hay phụ thuộc chính sách?
"""

    with st.expander(
        "Xem dữ liệu sẽ gửi cho Gemini"
    ):
        st.text_area(
            "Tóm tắt kết quả",
            value=result_summary.strip(),
            height=390,
            disabled=True,
        )

    analyze_clicked = st.button(
        "✨ Phân tích kết quả bằng Gemini",
        disabled=not configured,
        use_container_width=True,
        key="gemini_bai03",
    )

    if analyze_clicked:
        with st.spinner(
            "Gemini đang phân tích kết quả Bài 3..."
        ):
            try:
                analysis = analyze_result(
                    exercise_name=(
                        "Bài 3 — Chỉ số ưu tiên ngành "
                        "Priority cho 10 ngành Việt Nam"
                    ),
                    model_name=(
                        "MCDM trọng số tuyến tính, "
                        "chuẩn hóa min-max và sensitivity analysis"
                    ),
                    parameters={
                        "Trọng số gốc":
                            raw_weights,
                        "Tổng trọng số gốc":
                            f"{raw_sum:.2f}",
                        "Khoảng độ nhạy AI":
                            "0.05–0.40, bước 0.05",
                    },
                    result_summary=(
                        result_summary.strip()
                    ),
                    policy_questions=(
                        policy_questions.strip()
                    ),
                )

                st.session_state[
                    "bai03_gemini_analysis"
                ] = analysis

            except GeminiAgentError as error:
                st.error(str(error))

    saved_analysis = st.session_state.get(
        "bai03_gemini_analysis"
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

        st.markdown(saved_analysis)

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇️ Tải phân tích Gemini Bài 3",
            data=saved_analysis.encode(
                "utf-8"
            ),
            file_name=(
                "bai03_phan_tich_gemini.md"
            ),
            mime="text/markdown",
            use_container_width=True,
        )
