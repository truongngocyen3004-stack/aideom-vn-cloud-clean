from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.bai04_model import (
    BETA,
    DIGITAL_INDEX_INITIAL,
    ITEM_NAMES,
    ITEMS,
    REGION_NAMES,
    REGIONS,
    run_full_bai04,
)
from services.ai_agent import (
    GeminiAgentError,
    analyze_result,
    gemini_is_configured,
)
from ui.theme import page_header


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
        index=True
    ).encode("utf-8-sig")


def format_number(
    value: float,
) -> str:
    if pd.isna(value):
        return "Không có"
    return f"{value:,.2f}"


page_header(
    "Bài 4 — Quy hoạch tuyến tính phân bổ ngân sách số theo vùng",
    "Phân bổ 50.000 tỷ VND cho 6 vùng và 4 hạng mục nhằm tối đa hóa GDP gain, đồng thời kiểm soát công bằng số và phân quyền vùng.",
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
        <b>Biến quyết định:</b>
        x<sub>r,j</sub> là ngân sách đầu tư cho vùng r và hạng mục j.
        <br>
        <b>Hàm mục tiêu:</b>
        Max Z = Σ<sub>r,j</sub> β<sub>r,j</sub>x<sub>r,j</sub>.
        <br>
        <b>Ràng buộc:</b>
        ngân sách tổng, sàn/trần vùng, sàn nhân lực số và công bằng Digital Index.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander(
    "⚙️ Thiết lập mô hình Bài 4",
    expanded=True,
):
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        total_budget = st.number_input(
            "Ngân sách tổng (tỷ VND)",
            min_value=30000.0,
            max_value=100000.0,
            value=50000.0,
            step=1000.0,
        )

    with p2:
        min_region = st.number_input(
            "Sàn mỗi vùng (tỷ VND)",
            min_value=0.0,
            max_value=15000.0,
            value=5000.0,
            step=500.0,
        )

    with p3:
        max_region = st.number_input(
            "Trần mỗi vùng C3 (tỷ VND)",
            min_value=5000.0,
            max_value=50000.0,
            value=13000.0,
            step=500.0,
        )

    with p4:
        min_h_total = st.number_input(
            "Sàn H toàn quốc (tỷ VND)",
            min_value=0.0,
            max_value=40000.0,
            value=12000.0,
            step=500.0,
        )

    p5, p6 = st.columns(2)

    with p5:
        gamma = st.number_input(
            "γ — Hiệu quả đầu tư D",
            min_value=0.0005,
            max_value=0.0100,
            value=0.0020,
            step=0.0005,
            format="%.4f",
        )

    with p6:
        lam = st.slider(
            "λ — Mức công bằng vùng",
            min_value=0.50,
            max_value=0.95,
            value=0.70,
            step=0.05,
        )

    st.caption(
        "Lưu ý kỹ thuật: với γ = 0,002 và λ = 0,70, trần 12.000 tỷ VND "
        "trong gợi ý đề bài làm cấu hình công bằng C5 không khả thi vì "
        "Tây Nguyên cần tối thiểu khoảng 12.700 tỷ VND đầu tư D. "
        "Web dùng mặc định 13.000 để có nghiệm cơ sở; bạn vẫn có thể "
        "đổi về 12.000 để minh họa trạng thái không khả thi."
    )

    run_clicked = st.button(
        "🌸 Chạy toàn bộ mô hình Bài 4",
        use_container_width=True,
        type="primary",
    )

signature = (
    total_budget,
    min_region,
    max_region,
    min_h_total,
    gamma,
    lam,
)

if (
    run_clicked
    or "bai04_result" not in st.session_state
    or st.session_state.get(
        "bai04_signature"
    ) != signature
):
    try:
        with st.spinner(
            "Đang giải PuLP, CVXPY và các mô hình đối chứng..."
        ):
            st.session_state[
                "bai04_result"
            ] = run_full_bai04(
                total_budget=total_budget,
                min_region=min_region,
                max_region=max_region,
                min_h_total=min_h_total,
                gamma=gamma,
                lam=lam,
            )

            st.session_state[
                "bai04_signature"
            ] = signature

    except ValueError as error:
        st.error(str(error))
        st.stop()

result = st.session_state[
    "bai04_result"
]

data = result["data"]
quick_check = result[
    "quick_feasibility"
]
pulp_result = result["pulp"]
cvxpy_result = result["cvxpy"]
solver_comparison = result[
    "solver_comparison"
]
fairness = result[
    "fairness_comparison"
]
cap_comparison = result[
    "cap_comparison"
]
central_highlands = result[
    "central_highlands"
]

if quick_check["is_warning"]:
    st.warning(
        "Cảnh báo nhanh: để vùng yếu nhất đạt ngưỡng công bằng "
        f"theo Digital Index ban đầu, có thể cần khoảng "
        f"{quick_check['required_d_for_weakest']:,.0f} tỷ VND "
        "đầu tư D, cao hơn trần một vùng. Solver sẽ kiểm tra "
        "đầy đủ với toàn bộ hệ ràng buộc."
    )
else:
    st.success(
        "Kiểm tra nhanh cho thấy ràng buộc công bằng có khả năng khả thi "
        "với gamma, lambda và trần vùng hiện tại."
    )

tabs = st.tabs([
    "4.1 — Bối cảnh",
    "4.2 — Mô hình toán học",
    "4.3 — Dữ liệu β",
    "4.4.1 — PuLP/CBC",
    "4.4.2 — CVXPY",
    "4.4.3 — Heatmap phân bổ",
    "4.4.4 — Chi phí công bằng",
    "4.5 — Thảo luận chính sách",
    "✨ Phân tích AI",
])

with tabs[0]:
    st.subheader(
        "4.1 — Bối cảnh chênh lệch số giữa 6 vùng"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Ngân sách mô phỏng",
        f"{total_budget:,.0f} tỷ VND",
    )
    c2.metric(
        "Số vùng",
        "6",
    )
    c3.metric(
        "Hạng mục đầu tư",
        "4",
    )
    c4.metric(
        "Biến quyết định",
        "24 xᵣⱼ",
    )

    digital_table = data[
        "digital_table"
    ]

    fig_digital = px.bar(
        digital_table.sort_values(
            "Digital Index ban đầu"
        ),
        x="Digital Index ban đầu",
        y="Vùng",
        orientation="h",
        color="Digital Index ban đầu",
        text="Digital Index ban đầu",
        color_continuous_scale=[
            "#F4B8C8",
            "#EEDCF5",
            "#A8D5D1",
        ],
    )

    fig_digital.update_traces(
        textposition="outside"
    )

    fig_digital = style_plotly(
        fig_digital,
        title=(
            "Chênh lệch Digital Index ban đầu giữa 6 vùng"
        ),
        x_title="Digital Index",
        y_title="Vùng",
        height=500,
    )

    st.plotly_chart(
        fig_digital,
        use_container_width=True,
    )

    sankey = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 22,
                    "thickness": 20,
                    "line": {
                        "color": "#FFFFFF",
                        "width": 0.5,
                    },
                    "label": [
                        f"Ngân sách {total_budget:,.0f}",
                        "6 vùng",
                        "4 hạng mục",
                        "Tối đa GDP gain",
                        "Công bằng vùng",
                        "Ma trận phân bổ 6×4",
                    ],
                    "color": [
                        PINK,
                        LAVENDER,
                        MINT,
                        YELLOW,
                        ROSE,
                        BLUE,
                    ],
                },
                link={
                    "source": [
                        0,
                        1,
                        2,
                        2,
                        3,
                        4,
                    ],
                    "target": [
                        1,
                        2,
                        3,
                        4,
                        5,
                        5,
                    ],
                    "value": [
                        50,
                        50,
                        29,
                        21,
                        29,
                        21,
                    ],
                    "color": [
                        "#F6D7E1",
                        "#E8DDF3",
                        "#D9ECE9",
                        "#F9EBCF",
                        "#F6D7E1",
                        "#DDE8F5",
                    ],
                },
            )
        ]
    )

    sankey = style_plotly(
        sankey,
        title=(
            "Luồng ra quyết định từ ngân sách quốc gia đến phân bổ vùng–hạng mục"
        ),
        height=520,
    )

    st.plotly_chart(
        sankey,
        use_container_width=True,
    )

    st.info(
        "Nếu chỉ tối đa hóa hiệu quả biên, vốn có xu hướng tập trung "
        "vào vùng và hạng mục có β cao. Bài 4 bổ sung công bằng vùng "
        "để tránh khoảng cách số tiếp tục mở rộng."
    )

with tabs[1]:
    st.subheader(
        "4.2 — Mô hình toán học đầy đủ"
    )

    st.markdown(
        "**Biến quyết định**"
    )

    st.latex(
        r"""
        x_{r,j} \ge 0
        """
    )

    st.markdown(
        "Trong đó r là vùng và j thuộc {I, D, AI, H}."
    )

    st.markdown(
        "**Hàm mục tiêu**"
    )

    st.latex(
        r"""
        \max Z =
        \sum_{r \in R}
        \sum_{j \in J}
        \beta_{r,j}x_{r,j}
        """
    )

    st.markdown(
        "**C1 — Ngân sách tổng**"
    )

    st.latex(
        r"""
        \sum_r\sum_j x_{r,j}
        \le B
        """
    )

    st.markdown(
        "**C2 và C3 — Sàn và trần vùng**"
    )

    st.latex(
        r"""
        L_r \le
        \sum_j x_{r,j}
        \le U_r
        """
    )

    st.markdown(
        "**C4 — Sàn nhân lực số toàn quốc**"
    )

    st.latex(
        r"""
        \sum_r x_{r,H}
        \ge H_{\min}
        """
    )

    st.markdown(
        "**C5 — Công bằng Digital Index**"
    )

    st.latex(
        r"""
        D_r^{after}
        =
        D_r^0 + \gamma x_{r,D}
        """
    )

    st.latex(
        r"""
        D_r^{after} \le M,
        \qquad
        D_r^{after} \ge \lambda M
        """
    )

    st.success(
        "Biến phụ M tuyến tính hóa điều kiện: Digital Index sau đầu tư "
        "của mọi vùng phải đạt tối thiểu λ lần mức cao nhất."
    )

with tabs[2]:
    st.subheader(
        "4.3 — Hệ số tác động biên β theo vùng–hạng mục"
    )

    beta_matrix = data[
        "beta_matrix"
    ]

    st.dataframe(
        beta_matrix,
        use_container_width=True,
    )

    beta_heatmap = go.Figure(
        data=go.Heatmap(
            z=beta_matrix.values,
            x=beta_matrix.columns,
            y=beta_matrix.index,
            colorscale=PASTEL_HEATMAP,
            text=np.round(
                beta_matrix.values,
                2,
            ),
            texttemplate="%{text}",
            colorbar={
                "title": "β",
            },
            hovertemplate=(
                "Vùng=%{y}<br>"
                "Hạng mục=%{x}<br>"
                "β=%{z:.2f}"
                "<extra></extra>"
            ),
        )
    )

    beta_heatmap = style_plotly(
        beta_heatmap,
        title=(
            "Heatmap hệ số GDP gain βᵣⱼ"
        ),
        x_title="Hạng mục",
        y_title="Vùng",
        height=560,
    )

    st.plotly_chart(
        beta_heatmap,
        use_container_width=True,
    )

    beta_long = data[
        "beta_long"
    ]

    best_beta = beta_long.loc[
        beta_long[
            "β tác động biên"
        ].idxmax()
    ]

    st.success(
        f"Hệ số cao nhất là {best_beta['β tác động biên']:.2f}, "
        f"thuộc hạng mục **{best_beta['Hạng mục']}** tại "
        f"**{best_beta['Vùng']}**."
    )

    st.download_button(
        "⬇️ Tải dữ liệu β Bài 4",
        data=csv_bytes(
            beta_long
        ),
        file_name=(
            "bai04_du_lieu_beta.csv"
        ),
        mime="text/csv",
    )

with tabs[3]:
    st.subheader(
        "Câu 4.4.1 — Giải mô hình bằng PuLP/CBC"
    )

    if not pulp_result[
        "success"
    ]:
        st.error(
            pulp_result["status"]
        )
        st.code(
            "python -m pip install pulp",
            language="bash",
        )
    else:
        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Trạng thái",
            pulp_result["status"],
        )
        k2.metric(
            "Z*",
            format_number(
                pulp_result["objective"]
            ),
        )
        k3.metric(
            "Ngân sách sử dụng",
            (
                f"{pulp_result['total_used']:,.0f} "
                "tỷ VND"
            ),
        )
        k4.metric(
            "Đầu tư H",
            (
                f"{pulp_result['total_h']:,.0f} "
                "tỷ VND"
            ),
        )

        st.markdown(
            "#### Ma trận phân bổ tối ưu 6×4"
        )

        st.dataframe(
            pulp_result[
                "allocation_matrix"
            ].round(2),
            use_container_width=True,
        )

        left, right = st.columns(2)

        with left:
            st.markdown(
                "#### Tổng hợp theo vùng"
            )

            st.dataframe(
                pulp_result[
                    "region_summary"
                ].round(3),
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.markdown(
                "#### Tổng hợp theo hạng mục"
            )

            st.dataframe(
                pulp_result[
                    "item_summary"
                ].round(3),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(
            "#### Kiểm tra ràng buộc"
        )

        st.dataframe(
            pulp_result[
                "constraint_checks"
            ].round(4),
            use_container_width=True,
            hide_index=True,
        )

        if bool(
            pulp_result[
                "constraint_checks"
            ]["Đạt?"].all()
        ):
            st.success(
                "Tất cả ràng buộc C1–C5 đều được thỏa mãn."
            )
        else:
            st.warning(
                "Có ràng buộc cần kiểm tra lại."
            )

        st.markdown(
            "#### Shadow price và slack"
        )

        st.dataframe(
            pulp_result[
                "shadow_table"
            ].round(5),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Tải ma trận phân bổ PuLP",
            data=csv_bytes(
                pulp_result[
                    "allocation_matrix"
                ]
            ),
            file_name=(
                "bai04_441_phan_bo_pulp.csv"
            ),
            mime="text/csv",
        )

with tabs[4]:
    st.subheader(
        "Câu 4.4.2 — Giải lại bằng CVXPY"
    )

    if not cvxpy_result[
        "success"
    ]:
        st.warning(
            cvxpy_result["status"]
        )
        st.code(
            "python -m pip install cvxpy",
            language="bash",
        )
    else:
        v1, v2, v3, v4 = st.columns(4)

        v1.metric(
            "Solver CVXPY",
            cvxpy_result["solver"],
        )
        v2.metric(
            "Z* CVXPY",
            format_number(
                cvxpy_result[
                    "objective"
                ]
            ),
        )
        v3.metric(
            "|Z PuLP − Z CVXPY|",
            format_number(
                solver_comparison[
                    "objective_difference"
                ]
            ),
        )
        v4.metric(
            "Chênh lệch ô lớn nhất",
            format_number(
                solver_comparison[
                    "max_cell_difference"
                ]
            ),
        )

        compare_objective = pd.DataFrame({
            "Phương pháp": [
                "PuLP/CBC",
                cvxpy_result["solver"],
            ],
            "Z*": [
                pulp_result[
                    "objective"
                ],
                cvxpy_result[
                    "objective"
                ],
            ],
        })

        fig_solver = px.bar(
            compare_objective,
            x="Phương pháp",
            y="Z*",
            color="Phương pháp",
            text_auto=".2f",
            color_discrete_sequence=[
                PINK,
                MINT,
            ],
        )

        fig_solver.update_layout(
            showlegend=False
        )

        fig_solver = style_plotly(
            fig_solver,
            title=(
                "So sánh giá trị tối ưu PuLP và CVXPY"
            ),
            x_title="Solver",
            y_title="Z*",
        )

        st.plotly_chart(
            fig_solver,
            use_container_width=True,
        )

        difference_matrix = (
            solver_comparison[
                "difference_matrix"
            ]
        )

        if difference_matrix is not None:
            diff_heatmap = go.Figure(
                data=go.Heatmap(
                    z=difference_matrix.values,
                    x=difference_matrix.columns,
                    y=difference_matrix.index,
                    colorscale=[
                        [0.00, "#A8D5D1"],
                        [0.50, "#FFFFFF"],
                        [1.00, "#D989A5"],
                    ],
                    zmid=0,
                    text=np.round(
                        difference_matrix.values,
                        3,
                    ),
                    texttemplate="%{text}",
                    colorbar={
                        "title": (
                            "PuLP − CVXPY"
                        ),
                    },
                )
            )

            diff_heatmap = style_plotly(
                diff_heatmap,
                title=(
                    "Chênh lệch ma trận nghiệm giữa hai solver"
                ),
                x_title="Hạng mục",
                y_title="Vùng",
                height=560,
            )

            st.plotly_chart(
                diff_heatmap,
                use_container_width=True,
            )

        if solver_comparison[
            "same_objective"
        ]:
            st.success(
                "Hai phương pháp cho cùng giá trị mục tiêu trong sai số số học. "
                "Ma trận phân bổ có thể khác nhẹ nếu bài toán có nhiều nghiệm tối ưu."
            )
        else:
            st.warning(
                "Giá trị mục tiêu giữa hai solver khác đáng kể; "
                "cần kiểm tra solver hoặc độ chính xác."
            )

with tabs[5]:
    st.subheader(
        "Câu 4.4.3 — Heatmap phân bổ tối ưu"
    )

    if not pulp_result[
        "success"
    ]:
        st.error(
            "Cần giải thành công PuLP trước."
        )
    else:
        allocation = pulp_result[
            "allocation_matrix"
        ]

        allocation_heatmap = go.Figure(
            data=go.Heatmap(
                z=allocation.values,
                x=allocation.columns,
                y=allocation.index,
                colorscale=PASTEL_HEATMAP,
                text=np.round(
                    allocation.values,
                    0,
                ),
                texttemplate="%{text:,.0f}",
                colorbar={
                    "title": (
                        "Tỷ VND"
                    ),
                },
                hovertemplate=(
                    "Vùng=%{y}<br>"
                    "Hạng mục=%{x}<br>"
                    "Ngân sách=%{z:,.2f}"
                    "<extra></extra>"
                ),
            )
        )

        allocation_heatmap = style_plotly(
            allocation_heatmap,
            title=(
                "Heatmap ma trận phân bổ ngân sách tối ưu 6×4"
            ),
            x_title="Hạng mục",
            y_title="Vùng",
            height=590,
        )

        st.plotly_chart(
            allocation_heatmap,
            use_container_width=True,
        )

        region_summary = pulp_result[
            "region_summary"
        ].sort_values(
            "Tổng ngân sách, tỷ VND",
            ascending=False,
        )

        fig_region = px.bar(
            region_summary,
            x="Vùng",
            y="Tổng ngân sách, tỷ VND",
            color="Vùng",
            text_auto=",.0f",
            color_discrete_sequence=(
                PASTEL_SEQUENCE
            ),
        )

        fig_region.update_layout(
            showlegend=False
        )

        fig_region = style_plotly(
            fig_region,
            title=(
                "Tổng ngân sách mỗi vùng"
            ),
            x_title="Vùng",
            y_title="Ngân sách (tỷ VND)",
            height=520,
        )

        fig_region.update_xaxes(
            tickangle=-18
        )

        st.plotly_chart(
            fig_region,
            use_container_width=True,
        )

        top_region = region_summary.iloc[0]

        st.success(
            f"Vùng nhận ngân sách nhiều nhất là "
            f"**{top_region['Vùng']}**, với "
            f"**{top_region['Tổng ngân sách, tỷ VND']:,.0f} tỷ VND**."
        )

        st.markdown(
            "#### Hạng mục ưu tiên ở từng vùng"
        )

        st.dataframe(
            pulp_result[
                "preferred_items"
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )

with tabs[6]:
    st.subheader(
        "Câu 4.4.4 — So sánh có và không có công bằng C5"
    )

    full = fairness[
        "with_fairness"
    ]
    no_fair = fairness[
        "without_fairness"
    ]

    if (
        not full["success"]
        or not no_fair["success"]
    ):
        st.error(
            "Không thể so sánh vì ít nhất một mô hình không khả thi."
        )
    else:
        f1, f2, f3, f4 = st.columns(4)

        f1.metric(
            "Z* có công bằng",
            format_number(
                full["objective"]
            ),
        )
        f2.metric(
            "Z* không công bằng",
            format_number(
                no_fair["objective"]
            ),
        )
        f3.metric(
            "Chi phí công bằng",
            (
                f"{fairness['cost_absolute']:,.2f} "
                "GDP gain"
            ),
        )
        f4.metric(
            "Chi phí công bằng (%)",
            f"{fairness['cost_pct']:.3f}%",
        )

        compare_long = (
            fairness[
                "region_comparison"
            ][
                [
                    "Vùng",
                    "Có công bằng - Ngân sách",
                    "Không công bằng - Ngân sách",
                ]
            ]
            .melt(
                id_vars="Vùng",
                var_name="Mô hình",
                value_name="Ngân sách",
            )
        )

        fig_fair_budget = px.bar(
            compare_long,
            x="Vùng",
            y="Ngân sách",
            color="Mô hình",
            barmode="group",
            text_auto=",.0f",
            color_discrete_sequence=[
                MINT,
                ROSE,
            ],
        )

        fig_fair_budget = style_plotly(
            fig_fair_budget,
            title=(
                "Ngân sách theo vùng: có và không có công bằng C5"
            ),
            x_title="Vùng",
            y_title="Ngân sách (tỷ VND)",
            height=530,
        )

        fig_fair_budget.update_xaxes(
            tickangle=-18
        )

        st.plotly_chart(
            fig_fair_budget,
            use_container_width=True,
        )

        digital_long = (
            fairness[
                "region_comparison"
            ][
                [
                    "Vùng",
                    "Có công bằng - Digital Index",
                    "Không công bằng - Digital Index",
                ]
            ]
            .melt(
                id_vars="Vùng",
                var_name="Mô hình",
                value_name="Digital Index",
            )
        )

        fig_fair_digital = px.line(
            digital_long,
            x="Vùng",
            y="Digital Index",
            color="Mô hình",
            markers=True,
            color_discrete_sequence=[
                PINK,
                BLUE,
            ],
        )

        fig_fair_digital = style_plotly(
            fig_fair_digital,
            title=(
                "Digital Index sau đầu tư dưới hai mô hình"
            ),
            x_title="Vùng",
            y_title="Digital Index",
            height=500,
        )

        fig_fair_digital.update_xaxes(
            tickangle=-18
        )

        st.plotly_chart(
            fig_fair_digital,
            use_container_width=True,
        )

        st.dataframe(
            fairness[
                "region_comparison"
            ].round(3),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Chi phí kinh tế của công bằng là phần GDP gain tối ưu bị giảm "
            "khi buộc các vùng yếu phải thu hẹp khoảng cách Digital Index."
        )

with tabs[7]:
    st.subheader(
        "Mục 4.5 — Thảo luận chính sách"
    )

    no_fair = fairness[
        "without_fairness"
    ]

    if (
        pulp_result["success"]
        and no_fair["success"]
    ):
        no_fair_region = (
            no_fair[
                "region_summary"
            ]
            .sort_values(
                "Tổng ngân sách, tỷ VND",
                ascending=False,
            )
        )

        capital_destination = (
            no_fair_region.iloc[0]
        )

        q1, q2, q3 = st.columns(3)

        with q1:
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
                        4.5a — Nếu bỏ công bằng
                    </h4>
                """,
                unsafe_allow_html=True,
            )

            st.write(
                f"Vốn chảy nhiều nhất về **{capital_destination['Vùng']}**, "
                f"đạt {capital_destination['Tổng ngân sách, tỷ VND']:,.0f} tỷ VND. "
                "Nguyên nhân là vùng này có tổ hợp hệ số β cao hơn ở các "
                "hạng mục sinh lợi biên lớn."
            )

            st.write(
                "Trong dài hạn, tập trung vốn như vậy có thể làm khoảng cách "
                "hạ tầng, năng lực số, cơ hội việc làm và khả năng tiếp cận "
                "dịch vụ công giữa các vùng tiếp tục gia tăng."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        with q2:
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
                        4.5b — Chi phí của C3
                    </h4>
                """,
                unsafe_allow_html=True,
            )

            st.write(
                f"Trần vùng C3 làm Z* giảm "
                f"**{cap_comparison['cost_absolute']:,.2f}**, tương đương "
                f"**{cap_comparison['cost_pct']:.3f}%** so với mô hình "
                "không có trần vùng."
            )

            st.write(
                "Nếu tỷ lệ giảm nhỏ, đánh đổi có thể chấp nhận được vì "
                "C3 giảm tập trung ngân sách, tăng quyền tiếp cận nguồn lực "
                "của nhiều vùng và nâng tính chính danh của chính sách."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        with q3:
            st.markdown(
                """
                <div style="
                    background:#EEF8F7;
                    border:1px solid #D1E9E6;
                    border-radius:16px;
                    padding:18px;
                    min-height:350px;
                ">
                    <h4 style="color:#503743;">
                        4.5c — Tây Nguyên
                    </h4>
                """,
                unsafe_allow_html=True,
            )

            ai_beta = BETA[
                ("CH", "AI")
            ]
            h_beta = BETA[
                ("CH", "H")
            ]
            i_beta = BETA[
                ("CH", "I")
            ]

            st.write(
                f"β_AI của Tây Nguyên chỉ **{ai_beta:.2f}**, thấp hơn "
                f"β_H = **{h_beta:.2f}** và β_I = **{i_beta:.2f}**."
            )

            st.write(
                "Do đó, mô hình ưu tiên nhân lực số H và hạ tầng số I "
                "trước AI. Đây là trình tự hợp lý khi năng lực hấp thụ "
                "công nghệ của vùng còn hạn chế."
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "#### Minh chứng câu a — Phân bổ khi bỏ C5"
        )

        fig_no_fair = px.bar(
            no_fair_region,
            x="Vùng",
            y="Tổng ngân sách, tỷ VND",
            color="Vùng",
            text_auto=",.0f",
            color_discrete_sequence=(
                PASTEL_SEQUENCE
            ),
        )

        fig_no_fair.update_layout(
            showlegend=False
        )

        fig_no_fair = style_plotly(
            fig_no_fair,
            title=(
                "Dòng vốn khi bỏ ràng buộc công bằng"
            ),
            x_title="Vùng",
            y_title="Ngân sách (tỷ VND)",
        )

        fig_no_fair.update_xaxes(
            tickangle=-18
        )

        st.plotly_chart(
            fig_no_fair,
            use_container_width=True,
        )

        st.markdown(
            "#### Minh chứng câu b — Có và không có C3"
        )

        cap_long = (
            cap_comparison[
                "region_comparison"
            ]
            .melt(
                id_vars="Vùng",
                value_vars=[
                    "Có C3",
                    "Không C3",
                ],
                var_name="Kịch bản",
                value_name="Ngân sách",
            )
        )

        fig_cap = px.bar(
            cap_long,
            x="Vùng",
            y="Ngân sách",
            color="Kịch bản",
            barmode="group",
            text_auto=",.0f",
            color_discrete_sequence=[
                PINK,
                MINT,
            ],
        )

        fig_cap = style_plotly(
            fig_cap,
            title=(
                "Tác động của trần ngân sách vùng C3"
            ),
            x_title="Vùng",
            y_title="Ngân sách (tỷ VND)",
        )

        fig_cap.update_xaxes(
            tickangle=-18
        )

        st.plotly_chart(
            fig_cap,
            use_container_width=True,
        )

        st.markdown(
            "#### Minh chứng câu c — Tây Nguyên"
        )

        st.dataframe(
            central_highlands.round(3),
            use_container_width=True,
            hide_index=True,
        )

        fig_ch = go.Figure()

        fig_ch.add_trace(
            go.Bar(
                x=central_highlands[
                    "Hạng mục"
                ],
                y=central_highlands[
                    "Ngân sách, tỷ VND"
                ],
                name="Ngân sách",
                marker_color=PINK,
            )
        )

        fig_ch.add_trace(
            go.Scatter(
                x=central_highlands[
                    "Hạng mục"
                ],
                y=central_highlands[
                    "β Tây Nguyên"
                ],
                name="β Tây Nguyên",
                mode="lines+markers",
                yaxis="y2",
                line={
                    "color": MINT,
                    "width": 3,
                },
                marker={
                    "size": 9,
                },
            )
        )

        fig_ch.update_layout(
            yaxis={
                "title":
                    "Ngân sách (tỷ VND)",
            },
            yaxis2={
                "title": "β",
                "overlaying": "y",
                "side": "right",
            },
        )

        fig_ch = style_plotly(
            fig_ch,
            title=(
                "Tây Nguyên: ngân sách tối ưu và hiệu quả biên β"
            ),
            x_title="Hạng mục",
            y_title="Ngân sách (tỷ VND)",
            height=500,
        )

        st.plotly_chart(
            fig_ch,
            use_container_width=True,
        )

with tabs[8]:
    st.subheader(
        "Tác nhân AI phân tích kết quả Bài 4"
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

    if pulp_result["success"]:
        result_summary = f"""
BÀI 4 — LP PHÂN BỔ NGÂN SÁCH SỐ THEO VÙNG

Tham số:
- Ngân sách tổng: {total_budget:,.2f}
- Sàn mỗi vùng: {min_region:,.2f}
- Trần mỗi vùng: {max_region:,.2f}
- Sàn H toàn quốc: {min_h_total:,.2f}
- Gamma: {gamma:.4f}
- Lambda: {lam:.2f}

Kết quả PuLP:
- Trạng thái: {pulp_result['status']}
- Z*: {pulp_result['objective']:.4f}
- Ngân sách sử dụng: {pulp_result['total_used']:.4f}
- Tổng H: {pulp_result['total_h']:.4f}

Ma trận phân bổ:
{pulp_result['allocation_matrix'].round(3).to_string()}

Tổng hợp vùng:
{pulp_result['region_summary'].round(3).to_string(index=False)}

So sánh PuLP/CVXPY:
- CVXPY thành công: {cvxpy_result['success']}
- Chênh lệch Z*: {solver_comparison['objective_difference']}
- Chênh lệch ô lớn nhất: {solver_comparison['max_cell_difference']}

Chi phí công bằng C5:
- Tuyệt đối: {fairness['cost_absolute']}
- Phần trăm: {fairness['cost_pct']}

Chi phí trần vùng C3:
- Tuyệt đối: {cap_comparison['cost_absolute']}
- Phần trăm: {cap_comparison['cost_pct']}

Tây Nguyên:
{central_highlands.round(3).to_string(index=False)}
"""
    else:
        result_summary = (
            "Mô hình PuLP chưa giải thành công."
        )

    policy_questions = """
1. Nếu bỏ ràng buộc công bằng C5, vốn chảy về vùng nào và vì sao?
2. Hậu quả xã hội dài hạn của tập trung vốn theo hiệu quả biên là gì?
3. Trần vùng C3 làm Z* giảm bao nhiêu phần trăm và mức giảm có chấp nhận được không?
4. Tây Nguyên nên đầu tư AI hay ưu tiên H và I trước?
5. Đánh đổi giữa tăng trưởng và công bằng vùng nên được quản trị thế nào?
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
            or not pulp_result[
                "success"
            ]
        ),
        use_container_width=True,
        key="gemini_bai04",
    )

    if analyze_clicked:
        with st.spinner(
            "Gemini đang phân tích kết quả Bài 4..."
        ):
            try:
                analysis = analyze_result(
                    exercise_name=(
                        "Bài 4 — Quy hoạch tuyến tính "
                        "phân bổ ngân sách số theo vùng"
                    ),
                    model_name=(
                        "LP 24 biến, PuLP/CBC, CVXPY, "
                        "công bằng Digital Index và phân quyền vùng"
                    ),
                    parameters={
                        "Ngân sách tổng":
                            total_budget,
                        "Sàn vùng":
                            min_region,
                        "Trần vùng C3":
                            max_region,
                        "Sàn H":
                            min_h_total,
                        "Gamma":
                            gamma,
                        "Lambda":
                            lam,
                    },
                    result_summary=(
                        result_summary.strip()
                    ),
                    policy_questions=(
                        policy_questions.strip()
                    ),
                )

                st.session_state[
                    "bai04_gemini_analysis"
                ] = analysis

            except GeminiAgentError as error:
                st.error(str(error))

    saved_analysis = st.session_state.get(
        "bai04_gemini_analysis"
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
            "⬇️ Tải phân tích Gemini Bài 4",
            data=saved_analysis.encode(
                "utf-8"
            ),
            file_name=(
                "bai04_phan_tich_gemini.md"
            ),
            mime="text/markdown",
            use_container_width=True,
        )
