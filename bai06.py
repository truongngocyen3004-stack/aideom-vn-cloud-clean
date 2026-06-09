from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.bai06_model import (
    CRITERIA,
    DISPLAY_NAMES,
    EXPERT_WEIGHTS,
    ahp_weights,
    ai_weight_sensitivity,
    build_ahp_matrix,
    compare_rankings,
    correlation_diagnostics,
    entropy_weights,
    load_region_data,
    topsis,
)
from services.ai_agent import (
    GeminiAgentError,
    analyze_result,
    gemini_is_configured,
)
from ui.theme import page_header


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    ROOT
    / "data"
    / "vietnam_regions_2024.csv"
)

MODEL_VERSION = "bai06_v1"

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
    height: int = 470,
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


def format_top3(
    ranking: pd.DataFrame,
) -> str:
    return " → ".join(
        ranking.head(
            3
        )[
            "region_name_vi"
        ].tolist()
    )


page_header(
    "Bài 6 — TOPSIS xếp hạng 6 vùng ưu tiên đầu tư AI",
    "Xây dựng TOPSIS từ đầu, so sánh trọng số chuyên gia với Entropy, kiểm định độ nhạy trọng số AI và mở rộng AHP.",
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
        <b>Hệ số gần gũi TOPSIS:</b>
        Cᵢ* = Sᵢ⁻ / (Sᵢ⁺ + Sᵢ⁻).
        Điểm càng gần 1 thì vùng càng gần phương án lý tưởng.
        <br>
        <b>Tiêu chí:</b>
        7 tiêu chí lợi ích và Gini là tiêu chí chi phí.
    </div>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(
        f"Không tìm thấy file dữ liệu: {DATA_PATH}"
    )
    st.stop()

region_data = load_region_data(
    DATA_PATH
)

with st.expander(
    "⚙️ Thiết lập trọng số chuyên gia",
    expanded=True,
):
    st.caption(
        "Mặc định đúng theo đề: "
        "[0,10; 0,10; 0,15; 0,20; "
        "0,15; 0,15; 0,05; 0,10]."
    )

    first_row = st.columns(4)
    second_row = st.columns(4)

    weight_values = []

    for index, column in enumerate(
        CRITERIA
    ):
        target_column = (
            first_row[index]
            if index < 4
            else second_row[
                index - 4
            ]
        )

        with target_column:
            weight_values.append(
                st.slider(
                    DISPLAY_NAMES[
                        column
                    ],
                    min_value=0.00,
                    max_value=0.50,
                    value=float(
                        EXPERT_WEIGHTS[
                            index
                        ]
                    ),
                    step=0.01,
                    key=(
                        f"bai06_weight_"
                        f"{index}"
                    ),
                )
            )

    raw_weights = np.array(
        weight_values,
        dtype=float,
    )

    weight_sum = float(
        raw_weights.sum()
    )

    st.metric(
        "Tổng trọng số",
        f"{weight_sum:.2f}",
        (
            "Đúng bằng 1"
            if np.isclose(
                weight_sum,
                1.0,
            )
            else "Web tự chuẩn hóa về 1"
        ),
    )

    run_clicked = st.button(
        "🌸 Chạy toàn bộ mô hình Bài 6",
        type="primary",
        use_container_width=True,
    )

signature = (
    MODEL_VERSION,
    *raw_weights.tolist(),
)

signature_changed = (
    st.session_state.get(
        "bai06_signature"
    )
    != signature
)

if (
    run_clicked
    or "bai06_result"
    not in st.session_state
    or signature_changed
):
    st.session_state.pop(
        "bai06_gemini_analysis",
        None,
    )

    with st.spinner(
        "Đang tính TOPSIS, Entropy, độ nhạy và AHP..."
    ):
        expert_result = topsis(
            region_data,
            raw_weights,
        )

        entropy_result = (
            entropy_weights(
                region_data
            )
        )

        entropy_topsis = topsis(
            region_data,
            entropy_result[
                "weights"
            ],
        )

        comparison = (
            compare_rankings(
                expert_result[
                    "ranking"
                ],
                entropy_topsis[
                    "ranking"
                ],
            )
        )

        sensitivity = (
            ai_weight_sensitivity(
                region_data,
                raw_weights,
            )
        )

        pairwise = (
            build_ahp_matrix(
                raw_weights
            )
        )

        ahp_result = (
            ahp_weights(
                pairwise
            )
        )

        ahp_topsis = topsis(
            region_data,
            ahp_result[
                "weights"
            ],
        )

        correlation = (
            correlation_diagnostics(
                region_data
            )
        )

        st.session_state[
            "bai06_result"
        ] = {
            "expert":
                expert_result,
            "entropy":
                entropy_result,
            "entropy_topsis":
                entropy_topsis,
            "comparison":
                comparison,
            "sensitivity":
                sensitivity,
            "ahp":
                ahp_result,
            "ahp_topsis":
                ahp_topsis,
            "correlation":
                correlation,
        }

        st.session_state[
            "bai06_signature"
        ] = signature

result = st.session_state[
    "bai06_result"
]

expert = result["expert"]
entropy = result["entropy"]
entropy_topsis = result[
    "entropy_topsis"
]
comparison = result[
    "comparison"
]
sensitivity = result[
    "sensitivity"
]
ahp = result["ahp"]
ahp_topsis = result[
    "ahp_topsis"
]
correlation = result[
    "correlation"
]

tabs = st.tabs([
    "6.1 — Bối cảnh",
    "6.2 — Quy trình TOPSIS",
    "6.3 — Dữ liệu vùng",
    "6.4.1 — Trọng số chuyên gia",
    "6.4.2 — Entropy",
    "6.4.3 — Độ nhạy AI",
    "6.4.4 — AHP mở rộng",
    "6.5 — Thảo luận chính sách",
    "✨ Phân tích AI",
])

with tabs[0]:
    st.subheader(
        "6.1 — Bối cảnh ưu tiên đầu tư AI theo vùng"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Số vùng",
        "6",
    )
    c2.metric(
        "Số tiêu chí",
        "8",
    )
    c3.metric(
        "AI Readiness cao nhất",
        (
            f"{region_data['ai_readiness_0_100'].max():.0f}/100"
        ),
    )
    c4.metric(
        "Internet cao nhất",
        (
            f"{region_data['internet_penetration_pct'].max():.0f}%"
        ),
    )

    bubble = px.scatter(
        region_data,
        x="ai_readiness_0_100",
        y="digital_index_0_100",
        size="fdi_registered_billion_USD",
        color="gini_coef",
        text="region_code",
        hover_name="region_name_vi",
        hover_data=[
            "grdp_per_capita_million_VND",
            "trained_labor_pct",
            "rd_intensity_pct",
            "internet_penetration_pct",
        ],
        color_continuous_scale=[
            "#A8D5D1",
            "#F2D7A7",
            "#D989A5",
        ],
        size_max=60,
    )

    bubble.update_traces(
        textposition="top center"
    )

    bubble = style_plotly(
        bubble,
        title=(
            "AI Readiness, Digital Index, FDI và Gini của 6 vùng"
        ),
        x_title="AI Readiness",
        y_title="Digital Index",
        height=560,
    )

    bubble.update_layout(
        coloraxis_colorbar_title=(
            "Gini"
        )
    )

    st.plotly_chart(
        bubble,
        use_container_width=True,
    )

    st.info(
        "TOPSIS phù hợp vì mỗi vùng có điểm mạnh và điểm yếu khác nhau. "
        "Đông Nam Bộ và Đồng bằng sông Hồng mạnh về AI, số hóa và FDI; "
        "các vùng còn lại có thể có lợi thế cân bằng xã hội hoặc dư địa phát triển."
    )

with tabs[1]:
    st.subheader(
        "6.2 — Năm bước của phương pháp TOPSIS"
    )

    step_table = pd.DataFrame({
        "Bước": [
            "1",
            "2",
            "3",
            "4",
            "5",
        ],
        "Nội dung": [
            "Chuẩn hóa vector ma trận quyết định",
            "Nhân trọng số để tạo ma trận V",
            "Xác định lý tưởng dương A+ và âm A-",
            "Tính khoảng cách Euclide S+ và S-",
            "Tính C* và xếp hạng giảm dần",
        ],
        "Công thức": [
            "rᵢⱼ = xᵢⱼ / √Σxᵢⱼ²",
            "vᵢⱼ = wⱼrᵢⱼ",
            "A+ tốt nhất; A- xấu nhất",
            "Sᵢ± = √Σ(vᵢⱼ-Aⱼ±)²",
            "Cᵢ* = Sᵢ-/(Sᵢ+ + Sᵢ-)",
        ],
    })

    st.dataframe(
        step_table,
        use_container_width=True,
        hide_index=True,
    )

    st.latex(
        r"""
        r_{ij} =
        \frac{x_{ij}}
        {\sqrt{\sum_i x_{ij}^2}}
        """
    )

    st.latex(
        r"""
        v_{ij} =
        w_j r_{ij}
        """
    )

    st.latex(
        r"""
        C_i^* =
        \frac{S_i^-}
        {S_i^+ + S_i^-}
        """
    )

    st.success(
        "Gini được xử lý là tiêu chí chi phí: "
        "A+ lấy giá trị Gini có trọng số thấp nhất, "
        "A- lấy giá trị cao nhất."
    )

with tabs[2]:
    st.subheader(
        "6.3 — Dữ liệu 6 vùng kinh tế xã hội"
    )

    display_data = region_data.rename(
        columns={
            "region_code":
                "Mã vùng",
            "region_name_vi":
                "Vùng",
            "grdp_per_capita_million_VND":
                "GRDP/người",
            "fdi_registered_billion_USD":
                "FDI",
            "digital_index_0_100":
                "Digital Index",
            "ai_readiness_0_100":
                "AI Readiness",
            "trained_labor_pct":
                "LĐ qua đào tạo",
            "rd_intensity_pct":
                "R&D/GRDP",
            "internet_penetration_pct":
                "Internet",
            "gini_coef":
                "Gini",
        }
    )

    st.dataframe(
        display_data.round(3),
        use_container_width=True,
        hide_index=True,
    )

    normalized = expert[
        "normalized_matrix"
    ].set_index(
        "Vùng"
    )

    heatmap = go.Figure(
        data=go.Heatmap(
            z=normalized.values,
            x=normalized.columns,
            y=normalized.index,
            colorscale=PASTEL_HEATMAP,
            text=np.round(
                normalized.values,
                3,
            ),
            texttemplate="%{text}",
            colorbar={
                "title":
                    "Chuẩn vector"
            },
            hovertemplate=(
                "Vùng=%{y}<br>"
                "Tiêu chí=%{x}<br>"
                "rᵢⱼ=%{z:.4f}"
                "<extra></extra>"
            ),
        )
    )

    heatmap = style_plotly(
        heatmap,
        title=(
            "Ma trận chuẩn hóa vector rᵢⱼ"
        ),
        x_title="Tiêu chí",
        y_title="Vùng",
        height=590,
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Tải dữ liệu 6 vùng",
        data=csv_bytes(
            display_data
        ),
        file_name=(
            "bai06_du_lieu_6_vung.csv"
        ),
        mime="text/csv",
    )

with tabs[3]:
    st.subheader(
        "Câu 6.4.1 — TOPSIS với trọng số chuyên gia"
    )

    ranking = expert[
        "ranking"
    ]

    top3 = ranking.head(
        3
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Top 1",
        top3.iloc[0][
            "region_name_vi"
        ],
        (
            f"C* = "
            f"{top3.iloc[0]['TOPSIS_score']:.4f}"
        ),
    )
    k2.metric(
        "Top 2",
        top3.iloc[1][
            "region_name_vi"
        ],
        (
            f"C* = "
            f"{top3.iloc[1]['TOPSIS_score']:.4f}"
        ),
    )
    k3.metric(
        "Top 3",
        top3.iloc[2][
            "region_name_vi"
        ],
        (
            f"C* = "
            f"{top3.iloc[2]['TOPSIS_score']:.4f}"
        ),
    )
    k4.metric(
        "Khoảng cách Top 1–2",
        (
            f"{top3.iloc[0]['TOPSIS_score'] - top3.iloc[1]['TOPSIS_score']:.4f}"
        ),
    )

    rank_plot = ranking.sort_values(
        "TOPSIS_score",
        ascending=True,
    )

    fig_rank = px.bar(
        rank_plot,
        x="TOPSIS_score",
        y="region_name_vi",
        orientation="h",
        color="Rank",
        text="TOPSIS_score",
        color_continuous_scale=[
            "#F4B8C8",
            "#EEDCF5",
            "#A8D5D1",
        ],
    )

    fig_rank.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside",
    )

    fig_rank = style_plotly(
        fig_rank,
        title=(
            "Xếp hạng TOPSIS theo trọng số chuyên gia"
        ),
        x_title="C*",
        y_title="Vùng",
        height=540,
    )

    fig_rank.update_layout(
        coloraxis_colorbar_title=(
            "Hạng"
        )
    )

    st.plotly_chart(
        fig_rank,
        use_container_width=True,
    )

    distance_long = ranking[
        [
            "region_name_vi",
            "S_plus",
            "S_minus",
        ]
    ].melt(
        id_vars="region_name_vi",
        var_name="Khoảng cách",
        value_name="Giá trị",
    )

    distance_long[
        "Khoảng cách"
    ] = distance_long[
        "Khoảng cách"
    ].replace({
        "S_plus":
            "Đến lý tưởng dương S+",
        "S_minus":
            "Đến lý tưởng âm S-",
    })

    fig_distance = px.bar(
        distance_long,
        x="region_name_vi",
        y="Giá trị",
        color="Khoảng cách",
        barmode="group",
        color_discrete_sequence=[
            PINK,
            MINT,
        ],
    )

    fig_distance = style_plotly(
        fig_distance,
        title=(
            "Khoảng cách đến phương án lý tưởng tốt và xấu"
        ),
        x_title="Vùng",
        y_title="Khoảng cách Euclide",
        height=500,
    )

    fig_distance.update_xaxes(
        tickangle=-18
    )

    st.plotly_chart(
        fig_distance,
        use_container_width=True,
    )

    st.dataframe(
        ranking.rename(
            columns={
                "region_code":
                    "Mã vùng",
                "region_name_vi":
                    "Vùng",
                "TOPSIS_score":
                    "C*",
                "Rank":
                    "Hạng",
            }
        ).round(5),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "#### Lời giải lý tưởng"
    )

    st.dataframe(
        expert[
            "ideal_table"
        ].round(6),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Tải xếp hạng trọng số chuyên gia",
        data=csv_bytes(
            ranking
        ),
        file_name=(
            "bai06_641_topsis_chuyen_gia.csv"
        ),
        mime="text/csv",
    )

with tabs[4]:
    st.subheader(
        "Câu 6.4.2 — Trọng số khách quan Entropy"
    )

    weight_compare = pd.DataFrame({
        "Tiêu chí": [
            DISPLAY_NAMES[
                column
            ]
            for column in CRITERIA
        ],
        "Chuyên gia":
            expert[
                "weights"
            ],
        "Entropy":
            entropy[
                "weights"
            ],
    })

    weight_long = weight_compare.melt(
        id_vars="Tiêu chí",
        var_name="Phương pháp",
        value_name="Trọng số",
    )

    fig_weights = px.bar(
        weight_long,
        x="Tiêu chí",
        y="Trọng số",
        color="Phương pháp",
        barmode="group",
        text_auto=".3f",
        color_discrete_sequence=[
            PINK,
            MINT,
        ],
    )

    fig_weights = style_plotly(
        fig_weights,
        title=(
            "So sánh trọng số chuyên gia và Entropy"
        ),
        x_title="Tiêu chí",
        y_title="Trọng số",
        height=500,
    )

    fig_weights.update_xaxes(
        tickangle=-20
    )

    st.plotly_chart(
        fig_weights,
        use_container_width=True,
    )

    st.dataframe(
        entropy[
            "table"
        ].round(6),
        use_container_width=True,
        hide_index=True,
    )

    rank_long = comparison[
        [
            "region_name_vi",
            "Hạng chuyên gia",
            "Hạng Entropy",
        ]
    ].melt(
        id_vars="region_name_vi",
        var_name="Phương pháp",
        value_name="Hạng",
    )

    fig_slope = px.line(
        rank_long,
        x="Phương pháp",
        y="Hạng",
        color="region_name_vi",
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
            "Thay đổi thứ hạng khi chuyển sang Entropy"
        ),
        x_title="Bộ trọng số",
        y_title="Xếp hạng",
        height=560,
    )

    st.plotly_chart(
        fig_slope,
        use_container_width=True,
    )

    st.dataframe(
        comparison.round(5),
        use_container_width=True,
        hide_index=True,
    )

    largest_change = (
        comparison.iloc[0]
    )

    st.info(
        f"Vùng có thay đổi hạng tuyệt đối lớn nhất là "
        f"**{largest_change['region_name_vi']}**, "
        f"từ hạng {int(largest_change['Hạng chuyên gia'])} "
        f"sang hạng {int(largest_change['Hạng Entropy'])}."
    )

    st.download_button(
        "⬇️ Tải so sánh Entropy",
        data=csv_bytes(
            comparison
        ),
        file_name=(
            "bai06_642_entropy.csv"
        ),
        mime="text/csv",
    )

with tabs[5]:
    st.subheader(
        "Câu 6.4.3 — Độ nhạy trọng số AI Readiness"
    )

    sensitivity_summary = (
        sensitivity[
            "summary"
        ]
    )

    st.dataframe(
        sensitivity_summary,
        use_container_width=True,
        hide_index=True,
    )

    rank_pivot = (
        sensitivity[
            "ranks"
        ]
        .pivot(
            index="Vùng",
            columns="Trọng số AI",
            values="Hạng",
        )
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
            text=rank_pivot.values.astype(
                int
            ),
            texttemplate="%{text}",
            colorbar={
                "title": "Hạng"
            },
            hovertemplate=(
                "Vùng=%{y}<br>"
                "w_AI=%{x}<br>"
                "Hạng=%{z:.0f}"
                "<extra></extra>"
            ),
        )
    )

    rank_heatmap = style_plotly(
        rank_heatmap,
        title=(
            "Heatmap thứ hạng khi w_AI thay đổi 0,10–0,40"
        ),
        x_title="Trọng số AI Readiness",
        y_title="Vùng",
        height=560,
    )

    st.plotly_chart(
        rank_heatmap,
        use_container_width=True,
    )

    fig_score = px.line(
        sensitivity[
            "scores"
        ],
        x="Trọng số AI",
        y="TOPSIS_score",
        color="Vùng",
        markers=True,
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_score = style_plotly(
        fig_score,
        title=(
            "Điểm TOPSIS theo trọng số AI Readiness"
        ),
        x_title="w_AI",
        y_title="C*",
        height=550,
    )

    st.plotly_chart(
        fig_score,
        use_container_width=True,
    )

    if (
        sensitivity[
            "top3_configurations"
        ] == 1
    ):
        st.success(
            "Top-3 ổn định trong toàn bộ khoảng w_AI = 0,10–0,40."
        )
    else:
        st.warning(
            f"Top-3 xuất hiện "
            f"{sensitivity['top3_configurations']} cấu hình khác nhau. "
            "Kết quả phụ thuộc đáng kể vào mức ưu tiên cho AI."
        )

    st.download_button(
        "⬇️ Tải độ nhạy w_AI",
        data=csv_bytes(
            sensitivity_summary
        ),
        file_name=(
            "bai06_643_do_nhay_ai.csv"
        ),
        mime="text/csv",
    )

with tabs[6]:
    st.subheader(
        "Câu 6.4.4 — Mở rộng AHP đơn giản"
    )

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "λmax",
        f"{ahp['lambda_max']:.4f}",
    )
    a2.metric(
        "Consistency Ratio",
        f"{ahp['consistency_ratio']:.4f}",
    )
    a3.metric(
        "Nhất quán?",
        (
            "Có"
            if ahp[
                "is_consistent"
            ]
            else "Chưa đạt"
        ),
    )

    pairwise = ahp[
        "pairwise_matrix"
    ]

    ahp_heatmap = go.Figure(
        data=go.Heatmap(
            z=pairwise.values,
            x=pairwise.columns,
            y=pairwise.index,
            colorscale=PASTEL_HEATMAP,
            text=np.round(
                pairwise.values,
                3,
            ),
            texttemplate="%{text}",
            colorbar={
                "title": "So sánh"
            },
        )
    )

    ahp_heatmap = style_plotly(
        ahp_heatmap,
        title=(
            "Ma trận so sánh cặp AHP theo thang Saaty"
        ),
        x_title="Tiêu chí j",
        y_title="Tiêu chí i",
        height=650,
    )

    st.plotly_chart(
        ahp_heatmap,
        use_container_width=True,
    )

    all_weights = pd.DataFrame({
        "Tiêu chí": [
            DISPLAY_NAMES[
                column
            ]
            for column in CRITERIA
        ],
        "Chuyên gia":
            expert[
                "weights"
            ],
        "Entropy":
            entropy[
                "weights"
            ],
        "AHP":
            ahp[
                "weights"
            ],
    })

    all_weights_long = (
        all_weights.melt(
            id_vars="Tiêu chí",
            var_name="Phương pháp",
            value_name="Trọng số",
        )
    )

    fig_all_weights = px.bar(
        all_weights_long,
        x="Tiêu chí",
        y="Trọng số",
        color="Phương pháp",
        barmode="group",
        color_discrete_sequence=[
            PINK,
            MINT,
            LAVENDER,
        ],
    )

    fig_all_weights = style_plotly(
        fig_all_weights,
        title=(
            "So sánh trọng số chuyên gia, Entropy và AHP"
        ),
        x_title="Tiêu chí",
        y_title="Trọng số",
        height=520,
    )

    fig_all_weights.update_xaxes(
        tickangle=-20
    )

    st.plotly_chart(
        fig_all_weights,
        use_container_width=True,
    )

    ranking_methods = pd.DataFrame({
        "Vùng": region_data[
            "region_name_vi"
        ],
    })

    for method_name, method_result in [
        (
            "Chuyên gia",
            expert,
        ),
        (
            "Entropy",
            entropy_topsis,
        ),
        (
            "AHP",
            ahp_topsis,
        ),
    ]:
        method_ranks = (
            method_result[
                "ranking"
            ]
            .set_index(
                "region_name_vi"
            )["Rank"]
        )

        ranking_methods[
            method_name
        ] = ranking_methods[
            "Vùng"
        ].map(
            method_ranks
        )

    rank_methods_long = (
        ranking_methods.melt(
            id_vars="Vùng",
            var_name="Phương pháp",
            value_name="Hạng",
        )
    )

    fig_rank_methods = px.line(
        rank_methods_long,
        x="Phương pháp",
        y="Hạng",
        color="Vùng",
        markers=True,
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_rank_methods.update_yaxes(
        autorange="reversed",
        dtick=1,
    )

    fig_rank_methods = style_plotly(
        fig_rank_methods,
        title=(
            "So sánh thứ hạng dưới ba cách xác định trọng số"
        ),
        x_title="Phương pháp",
        y_title="Hạng",
        height=570,
    )

    st.plotly_chart(
        fig_rank_methods,
        use_container_width=True,
    )

    st.dataframe(
        ranking_methods,
        use_container_width=True,
        hide_index=True,
    )

    if ahp["is_consistent"]:
        st.success(
            "CR ≤ 0,10: ma trận AHP đạt mức nhất quán chấp nhận được."
        )
    else:
        st.warning(
            "CR > 0,10: cần điều chỉnh lại các so sánh cặp."
        )

with tabs[7]:
    st.subheader(
        "6.5 — Thảo luận chính sách"
    )

    expert_top3 = expert[
        "ranking"
    ].head(
        3
    )

    entropy_change = (
        comparison.iloc[0]
    )

    p1, p2 = st.columns(2)

    with p1:
        st.markdown(
            """
            <div style="
                background:#FFF1F6;
                border:1px solid #F0D5DF;
                border-radius:16px;
                padding:18px;
                min-height:350px;
            ">
                <h4 style="color:#503743;">
                    6.5a — Vùng dẫn đầu
                </h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"Vùng dẫn đầu theo trọng số chuyên gia là "
            f"**{expert_top3.iloc[0]['region_name_vi']}**, "
            f"với C* = "
            f"**{expert_top3.iloc[0]['TOPSIS_score']:.4f}**."
        )

        st.write(
            "Kết quả là căn cứ định lượng quan trọng, nhưng quyết định "
            "đặt trung tâm AI quốc gia còn cần đất đai, điện, dữ liệu, "
            "an ninh, khả năng liên kết viện–trường–doanh nghiệp "
            "và cân bằng vùng."
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
                min-height:350px;
            ">
                <h4 style="color:#503743;">
                    6.5b — Thay đổi khi dùng Entropy
                </h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"Vùng thay đổi hạng lớn nhất là "
            f"**{entropy_change['region_name_vi']}**: "
            f"hạng {int(entropy_change['Hạng chuyên gia'])} "
            f"→ {int(entropy_change['Hạng Entropy'])}."
        )

        st.write(
            "Entropy tăng trọng số cho tiêu chí có độ phân tán lớn giữa "
            "các vùng. Do đó, một vùng mạnh ở tiêu chí phân hóa cao "
            "có thể tăng hạng dù chuyên gia không đặt trọng số lớn cho nó."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    p3, p4 = st.columns(2)

    with p3:
        st.markdown(
            """
            <div style="
                background:#EEF8F7;
                border:1px solid #D1E9E6;
                border-radius:16px;
                padding:18px;
                min-height:380px;
            ">
                <h4 style="color:#503743;">
                    6.5c — Tương quan AI và Internet
                </h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"Hệ số tương quan Pearson giữa AI Readiness và Internet "
            f"là **{correlation['correlation']:.4f}**, "
            f"VIF hai biến xấp xỉ **{correlation['vif']:.2f}**."
        )

        st.write(
            "Nếu tương quan cao, TOPSIS có thể đếm hai lần cùng một "
            "năng lực nền tảng và làm vùng mạnh về số hóa được thưởng quá mức."
        )

        st.write(
            "Cách xử lý: giảm/gộp trọng số, dùng PCA hoặc factor analysis, "
            "loại một tiêu chí trùng lặp, hoặc chạy độ nhạy theo các bộ tiêu chí."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with p4:
        st.markdown(
            """
            <div style="
                background:#FFF8EA;
                border:1px solid #F0E1BA;
                border-radius:16px;
                padding:18px;
                min-height:380px;
            ">
                <h4 style="color:#503743;">
                    6.5d — Chọn 3 trung tâm AI
                </h4>
            """,
            unsafe_allow_html=True,
        )

        for index, row in expert_top3.iterrows():
            st.markdown(
                f"**{int(row['Rank'])}. "
                f"{row['region_name_vi']}** "
                f"(C* = {row['TOPSIS_score']:.4f})"
            )

        st.write(
            "Nên bổ sung tiêu chí địa–chính trị và khả năng phục hồi: "
            "phân bố Bắc–Trung–Nam, an ninh năng lượng, rủi ro thiên tai, "
            "kết nối quốc phòng, khả năng dự phòng dữ liệu và tính đại diện vùng."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    correlation_plot = px.scatter(
        region_data,
        x="internet_penetration_pct",
        y="ai_readiness_0_100",
        text="region_code",
        hover_name="region_name_vi",
        color="region_name_vi",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    x_values = region_data[
        "internet_penetration_pct"
    ].to_numpy(
        dtype=float
    )
    y_values = region_data[
        "ai_readiness_0_100"
    ].to_numpy(
        dtype=float
    )

    slope, intercept = np.polyfit(
        x_values,
        y_values,
        1,
    )

    x_line = np.linspace(
        x_values.min(),
        x_values.max(),
        100,
    )

    correlation_plot.add_trace(
        go.Scatter(
            x=x_line,
            y=(
                slope * x_line
                + intercept
            ),
            mode="lines",
            name="Xu hướng tuyến tính",
            line={
                "color": TEXT,
                "width": 2,
                "dash": "dash",
            },
        )
    )

    correlation_plot.update_traces(
        textposition="top center"
    )

    correlation_plot = style_plotly(
        correlation_plot,
        title=(
            "Mối quan hệ giữa Internet penetration và AI Readiness"
        ),
        x_title="Internet penetration (%)",
        y_title="AI Readiness",
        height=520,
    )

    correlation_plot.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        correlation_plot,
        use_container_width=True,
    )

    stability_table = pd.DataFrame({
        "Phương pháp": [
            "Chuyên gia",
            "Entropy",
            "AHP",
        ],
        "Top-3": [
            format_top3(
                expert[
                    "ranking"
                ]
            ),
            format_top3(
                entropy_topsis[
                    "ranking"
                ]
            ),
            format_top3(
                ahp_topsis[
                    "ranking"
                ]
            ),
        ],
    })

    st.markdown(
        "#### Đối chiếu Top-3 giữa ba phương pháp"
    )

    st.dataframe(
        stability_table,
        use_container_width=True,
        hide_index=True,
    )

with tabs[8]:
    st.subheader(
        "Tác nhân AI phân tích kết quả Bài 6"
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
BÀI 6 — TOPSIS XẾP HẠNG 6 VÙNG ƯU TIÊN ĐẦU TƯ AI

Trọng số chuyên gia sau chuẩn hóa:
{dict(zip([DISPLAY_NAMES[c] for c in CRITERIA], expert['weights'].round(6)))}

Xếp hạng chuyên gia:
{expert['ranking'].round(6).to_string(index=False)}

Trọng số Entropy:
{entropy['table'].round(6).to_string(index=False)}

So sánh hạng:
{comparison.round(6).to_string(index=False)}

Độ nhạy AI:
- Số cấu hình Top-3 khác nhau: {sensitivity['top3_configurations']}
{sensitivity['summary'].to_string(index=False)}

AHP:
- Lambda max: {ahp['lambda_max']:.6f}
- CI: {ahp['consistency_index']:.6f}
- CR: {ahp['consistency_ratio']:.6f}
- Nhất quán: {ahp['is_consistent']}
- Top-3 AHP: {format_top3(ahp_topsis['ranking'])}

Tương quan AI Readiness và Internet:
- Pearson r: {correlation['correlation']:.6f}
- R²: {correlation['r_squared']:.6f}
- VIF: {correlation['vif']:.6f}
"""

    policy_questions = """
1. Vùng nào dẫn đầu theo trọng số chuyên gia và có nên đặt trung tâm AI quốc gia đầu tiên tại đó không?
2. Vùng nào thay đổi hạng lớn nhất khi dùng Entropy và nguyên nhân là gì?
3. Tương quan AI Readiness–Internet ảnh hưởng đến TOPSIS như thế nào?
4. Nên chọn ba vùng nào cho ba trung tâm AI lớn?
5. Có cần thêm tiêu chí địa–chính trị, năng lượng, an ninh và khả năng phục hồi không?
6. Kết quả giữa chuyên gia, Entropy và AHP có đủ vững để ra quyết định không?
"""

    with st.expander(
        "Xem dữ liệu sẽ gửi cho Gemini"
    ):
        st.text_area(
            "Tóm tắt kết quả",
            value=result_summary.strip(),
            height=430,
            disabled=True,
        )

    analyze_clicked = st.button(
        "✨ Phân tích kết quả bằng Gemini",
        disabled=not configured,
        use_container_width=True,
        key="gemini_bai06",
    )

    if analyze_clicked:
        with st.spinner(
            "Gemini đang phân tích kết quả Bài 6..."
        ):
            try:
                analysis = analyze_result(
                    exercise_name=(
                        "Bài 6 — TOPSIS xếp hạng "
                        "6 vùng ưu tiên đầu tư AI"
                    ),
                    model_name=(
                        "TOPSIS từ đầu bằng numpy, "
                        "Entropy weighting, sensitivity analysis "
                        "và AHP đơn giản"
                    ),
                    parameters={
                        "Trọng số chuyên gia":
                            expert[
                                "weights"
                            ].round(
                                6
                            ).tolist(),
                        "Khoảng w_AI":
                            "0.10–0.40",
                        "Gini":
                            "Tiêu chí chi phí",
                    },
                    result_summary=(
                        result_summary.strip()
                    ),
                    policy_questions=(
                        policy_questions.strip()
                    ),
                )

                st.session_state[
                    "bai06_gemini_analysis"
                ] = analysis

            except GeminiAgentError as error:
                st.error(str(error))

    saved_analysis = st.session_state.get(
        "bai06_gemini_analysis"
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
            "⬇️ Tải phân tích Gemini Bài 6",
            data=saved_analysis.encode(
                "utf-8"
            ),
            file_name=(
                "bai06_phan_tich_gemini.md"
            ),
            mime="text/markdown",
            use_container_width=True,
        )
