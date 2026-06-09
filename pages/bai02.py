from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.bai02_model import (
    DEFAULT_IMPACTS,
    DEFAULT_MINIMUMS,
    ITEM_NAMES,
    run_full_bai02,
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


def style_plotly(
    fig: go.Figure,
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 440,
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
            "l": 55,
            "r": 30,
            "t": 70,
            "b": 60,
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


def format_status(
    success: bool,
) -> str:
    return (
        "Tối ưu"
        if success
        else "Không khả thi"
    )


page_header(
    "Bài 2 — Quy hoạch tuyến tính phân bổ ngân sách số",
    "Tối đa hóa tác động GDP kỳ vọng khi phân bổ ngân sách cho hạ tầng số, AI và dữ liệu, nhân lực số và R&D công nghệ.",
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
        <b>Hàm mục tiêu:</b>
        Max Z = 0,85x₁ + 1,20x₂ + 0,95x₃ + 1,35x₄
        <br>
        <b>Ràng buộc chính:</b>
        tổng ngân sách; mức sàn từng hạng mục; và
        tỷ trọng tối thiểu của AI + R&D trong tổng phân bổ.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander(
    "⚙️ Thiết lập bài toán",
    expanded=True,
):
    st.markdown(
        "**Ngân sách và yêu cầu chính sách**"
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        budget = st.number_input(
            "Tổng ngân sách B (nghìn tỷ VND)",
            min_value=10.0,
            max_value=300.0,
            value=100.0,
            step=5.0,
        )

    with p2:
        strategic_share = st.slider(
            "Tỷ trọng tối thiểu AI + R&D",
            min_value=0.10,
            max_value=0.80,
            value=0.35,
            step=0.01,
            format="%.2f",
        )

    with p3:
        priority_human_floor = st.number_input(
            "Sàn nhân lực trong kịch bản ưu tiên x₃",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=5.0,
        )

    st.markdown(
        "**Mức đầu tư tối thiểu**"
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        min_x1 = st.number_input(
            "x₁ — Hạ tầng số",
            min_value=0.0,
            max_value=150.0,
            value=float(
                DEFAULT_MINIMUMS[0]
            ),
            step=1.0,
        )

    with m2:
        min_x2 = st.number_input(
            "x₂ — AI và dữ liệu",
            min_value=0.0,
            max_value=150.0,
            value=float(
                DEFAULT_MINIMUMS[1]
            ),
            step=1.0,
        )

    with m3:
        min_x3 = st.number_input(
            "x₃ — Nhân lực số",
            min_value=0.0,
            max_value=150.0,
            value=float(
                DEFAULT_MINIMUMS[2]
            ),
            step=1.0,
        )

    with m4:
        min_x4 = st.number_input(
            "x₄ — R&D công nghệ",
            min_value=0.0,
            max_value=150.0,
            value=float(
                DEFAULT_MINIMUMS[3]
            ),
            step=1.0,
        )

    st.markdown(
        "**Hệ số tác động GDP kỳ vọng**"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        impact_x1 = st.number_input(
            "Hệ số x₁",
            min_value=0.0,
            max_value=3.0,
            value=float(
                DEFAULT_IMPACTS[0]
            ),
            step=0.05,
        )

    with c2:
        impact_x2 = st.number_input(
            "Hệ số x₂",
            min_value=0.0,
            max_value=3.0,
            value=float(
                DEFAULT_IMPACTS[1]
            ),
            step=0.05,
        )

    with c3:
        impact_x3 = st.number_input(
            "Hệ số x₃",
            min_value=0.0,
            max_value=3.0,
            value=float(
                DEFAULT_IMPACTS[2]
            ),
            step=0.05,
        )

    with c4:
        impact_x4 = st.number_input(
            "Hệ số x₄",
            min_value=0.0,
            max_value=3.0,
            value=float(
                DEFAULT_IMPACTS[3]
            ),
            step=0.05,
        )

    run_clicked = st.button(
        "🌸 Chạy toàn bộ mô hình Bài 2",
        use_container_width=True,
        type="primary",
    )

minimums = np.array(
    [
        min_x1,
        min_x2,
        min_x3,
        min_x4,
    ],
    dtype=float,
)

impacts = np.array(
    [
        impact_x1,
        impact_x2,
        impact_x3,
        impact_x4,
    ],
    dtype=float,
)

signature = (
    budget,
    strategic_share,
    priority_human_floor,
    *minimums.tolist(),
    *impacts.tolist(),
)

if (
    run_clicked
    or "bai02_result" not in st.session_state
    or st.session_state.get(
        "bai02_signature"
    ) != signature
):
    with st.spinner(
        "Đang giải SciPy, PuLP, độ nhạy ngân sách và kịch bản ưu tiên nhân lực..."
    ):
        st.session_state[
            "bai02_result"
        ] = run_full_bai02(
            budget=budget,
            minimums=minimums,
            strategic_share=strategic_share,
            impacts=impacts,
            sensitivity_budgets=(
                100.0,
                120.0,
                140.0,
            ),
            priority_human_floor=(
                priority_human_floor
            ),
        )

        st.session_state[
            "bai02_signature"
        ] = signature

result = st.session_state[
    "bai02_result"
]

scipy_result = result["scipy"]
pulp_result = result["pulp"]
sensitivity = result["sensitivity"]
human_priority = result[
    "human_priority"
]

tabs = st.tabs([
    "📘 Mô hình & ràng buộc",
    "2.4.1 — SciPy",
    "2.4.2 — PuLP & dual",
    "2.4.3 — Độ nhạy ngân sách",
    "2.4.4 — Ưu tiên nhân lực",
    "2.5 — Thảo luận chính sách",
    "✨ Phân tích AI",
])

with tabs[0]:
    st.subheader(
        "Biến quyết định và ý nghĩa"
    )

    model_table = pd.DataFrame({
        "Biến": [
            "x₁",
            "x₂",
            "x₃",
            "x₄",
        ],
        "Hạng mục": ITEM_NAMES,
        "Hệ số tác động": impacts,
        "Mức tối thiểu": minimums,
    })

    st.dataframe(
        model_table.round(3),
        use_container_width=True,
        hide_index=True,
    )

    constraints_table = pd.DataFrame({
        "Ràng buộc": [
            "Ngân sách tổng",
            "Sàn hạ tầng số",
            "Sàn AI và dữ liệu",
            "Sàn nhân lực số",
            "Sàn R&D",
            "Công nghệ chiến lược",
        ],
        "Biểu thức": [
            "x₁ + x₂ + x₃ + x₄ ≤ B",
            f"x₁ ≥ {min_x1:.1f}",
            f"x₂ ≥ {min_x2:.1f}",
            f"x₃ ≥ {min_x3:.1f}",
            f"x₄ ≥ {min_x4:.1f}",
            (
                "x₂ + x₄ ≥ "
                f"{strategic_share:.0%}"
                "(x₁+x₂+x₃+x₄)"
            ),
        ],
    })

    st.dataframe(
        constraints_table,
        use_container_width=True,
        hide_index=True,
    )

    minimum_sum = float(
        minimums.sum()
    )

    strategic_at_minimum = (
        float(
            minimums[1]
            + minimums[3]
        )
        / minimum_sum
        if minimum_sum > 0
        else 0.0
    )

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Tổng các mức sàn",
        f"{minimum_sum:.1f} nghìn tỷ VND",
    )

    q2.metric(
        "Ngân sách còn lại sau mức sàn",
        f"{budget - minimum_sum:.1f}",
    )

    q3.metric(
        "Tỷ trọng AI + R&D tại mức sàn",
        f"{strategic_at_minimum:.1%}",
    )

    if minimum_sum > budget:
        st.error(
            "Tổng các mức đầu tư tối thiểu vượt ngân sách. "
            "Bài toán chắc chắn không khả thi."
        )
    else:
        st.success(
            "Tổng các mức sàn không vượt ngân sách. "
            "Tính khả thi cuối cùng còn phụ thuộc ràng buộc tỷ trọng chiến lược."
        )

with tabs[1]:
    st.subheader(
        "Câu 2.4.1 — Giải bằng scipy.optimize.linprog"
    )

    if not scipy_result["success"]:
        st.error(
            "Bài toán không khả thi với bộ tham số hiện tại. "
            f"Thông báo solver: {scipy_result['status']}"
        )
    else:
        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Trạng thái",
            scipy_result["status"],
        )

        s2.metric(
            "Z* — GDP kỳ vọng",
            f"{scipy_result['z']:.2f}",
        )

        s3.metric(
            "Ngân sách sử dụng",
            (
                f"{scipy_result['total_used']:.2f} "
                "nghìn tỷ VND"
            ),
        )

        s4.metric(
            "Tỷ trọng AI + R&D",
            f"{scipy_result['strategic_ratio']:.1%}",
        )

        allocation = scipy_result[
            "table"
        ]

        allocation_long = allocation[
            [
                "Hạng mục",
                "Mức tối thiểu",
                "Phân bổ tối ưu",
            ]
        ].melt(
            id_vars="Hạng mục",
            var_name="Loại",
            value_name="Ngân sách",
        )

        fig_allocation = px.bar(
            allocation_long,
            x="Hạng mục",
            y="Ngân sách",
            color="Loại",
            barmode="group",
            text_auto=".1f",
            color_discrete_sequence=[
                ROSE,
                PINK,
            ],
        )

        fig_allocation = style_plotly(
            fig_allocation,
            title=(
                "Phân bổ tối ưu và mức đầu tư tối thiểu"
            ),
            x_title="Hạng mục",
            y_title="Ngân sách (nghìn tỷ VND)",
        )

        st.plotly_chart(
            fig_allocation,
            use_container_width=True,
        )

        fig_contribution = px.bar(
            allocation,
            x="Hạng mục",
            y="Đóng góp vào Z",
            color="Hạng mục",
            text_auto=".2f",
            color_discrete_sequence=(
                PASTEL_SEQUENCE
            ),
        )

        fig_contribution.update_layout(
            showlegend=False
        )

        fig_contribution = style_plotly(
            fig_contribution,
            title=(
                "Đóng góp của từng hạng mục vào giá trị mục tiêu"
            ),
            x_title="Hạng mục",
            y_title="Đóng góp vào Z",
        )

        st.plotly_chart(
            fig_contribution,
            use_container_width=True,
        )

        st.dataframe(
            allocation.round(3),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Tải nghiệm tối ưu SciPy",
            data=csv_bytes(
                allocation
            ),
            file_name=(
                "bai02_241_phan_bo_scipy.csv"
            ),
            mime="text/csv",
        )

with tabs[2]:
    st.subheader(
        "Câu 2.4.2 — Giải lại bằng PuLP và giá đối ngẫu"
    )

    if not pulp_result["success"]:
        st.warning(
            pulp_result["status"]
        )

        st.code(
            "python -m pip install pulp",
            language="bash",
        )
    else:
        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Trạng thái PuLP",
            pulp_result["status"],
        )

        p2.metric(
            "Z* PuLP",
            f"{pulp_result['z']:.2f}",
        )

        scipy_pulp_difference = (
            pulp_result["z"]
            - scipy_result["z"]
            if scipy_result["success"]
            else np.nan
        )

        p3.metric(
            "Chênh lệch PuLP − SciPy",
            f"{scipy_pulp_difference:.8f}",
        )

        compare = scipy_result[
            "table"
        ][
            [
                "Hạng mục",
                "Phân bổ tối ưu",
            ]
        ].rename(
            columns={
                "Phân bổ tối ưu":
                    "SciPy"
            }
        )

        compare["PuLP"] = pulp_result[
            "x"
        ]

        compare_long = compare.melt(
            id_vars="Hạng mục",
            var_name="Solver",
            value_name="Phân bổ",
        )

        fig_compare = px.bar(
            compare_long,
            x="Hạng mục",
            y="Phân bổ",
            color="Solver",
            barmode="group",
            text_auto=".2f",
            color_discrete_sequence=[
                PINK,
                MINT,
            ],
        )

        fig_compare = style_plotly(
            fig_compare,
            title=(
                "So sánh nghiệm SciPy và PuLP"
            ),
            x_title="Hạng mục",
            y_title="Ngân sách (nghìn tỷ VND)",
        )

        st.plotly_chart(
            fig_compare,
            use_container_width=True,
        )

        dual_table = pulp_result[
            "dual_table"
        ].copy()

        fig_dual = px.bar(
            dual_table,
            x="Ràng buộc",
            y="Shadow price",
            color="Ràng buộc",
            text_auto=".3f",
            color_discrete_sequence=(
                PASTEL_SEQUENCE
            ),
        )

        fig_dual.update_layout(
            showlegend=False
        )

        fig_dual = style_plotly(
            fig_dual,
            title=(
                "Giá đối ngẫu của các ràng buộc"
            ),
            x_title="Ràng buộc",
            y_title="Shadow price",
            height=480,
        )

        st.plotly_chart(
            fig_dual,
            use_container_width=True,
        )

        st.dataframe(
            dual_table.round(5),
            use_container_width=True,
            hide_index=True,
        )

        budget_dual_rows = dual_table[
            dual_table[
                "Ràng buộc"
            ] == "Ngan_sach_tong"
        ]

        if not budget_dual_rows.empty:
            budget_shadow = float(
                budget_dual_rows.iloc[0][
                    "Shadow price"
                ]
            )

            st.success(
                f"Shadow price của ràng buộc ngân sách tổng là "
                f"{budget_shadow:.3f}. Trong vùng ổn định hiện tại, "
                f"tăng thêm 1 đơn vị ngân sách làm Z* tăng xấp xỉ "
                f"{budget_shadow:.3f} đơn vị."
            )

        st.download_button(
            "⬇️ Tải bảng dual PuLP",
            data=csv_bytes(
                dual_table
            ),
            file_name=(
                "bai02_242_gia_doi_ngau.csv"
            ),
            mime="text/csv",
        )

with tabs[3]:
    st.subheader(
        "Câu 2.4.3 — Phân tích độ nhạy ngân sách"
    )

    st.info(
        "Theo đề bài, ngân sách được so sánh tại ba mức "
        "100, 120 và 140 nghìn tỷ VND."
    )

    valid_sensitivity = (
        sensitivity.dropna(
            subset=["Z*"]
        )
    )

    fig_sensitivity = px.line(
        valid_sensitivity,
        x="Ngân sách B",
        y="Z*",
        markers=True,
        text="Z*",
        color_discrete_sequence=[
            PINK
        ],
    )

    fig_sensitivity.update_traces(
        line={
            "width": 3
        },
        marker={
            "size": 10
        },
        texttemplate="%{text:.2f}",
        textposition="top center",
    )

    fig_sensitivity = style_plotly(
        fig_sensitivity,
        title="Đường cong giá trị tối ưu Z*(B)",
        x_title="Ngân sách B (nghìn tỷ VND)",
        y_title="Z*",
    )

    st.plotly_chart(
        fig_sensitivity,
        use_container_width=True,
    )

    marginal_table = sensitivity.dropna(
        subset=[
            "GDP tăng thêm trên 1 đơn vị B"
        ]
    )

    if not marginal_table.empty:
        fig_marginal = px.bar(
            marginal_table,
            x="Ngân sách B",
            y=(
                "GDP tăng thêm trên 1 đơn vị B"
            ),
            text_auto=".3f",
            color_discrete_sequence=[
                LAVENDER
            ],
        )

        fig_marginal = style_plotly(
            fig_marginal,
            title=(
                "Giá trị biên của ngân sách giữa các kịch bản"
            ),
            x_title="Mức ngân sách mới",
            y_title=(
                "ΔZ*/ΔB"
            ),
        )

        st.plotly_chart(
            fig_marginal,
            use_container_width=True,
        )

    st.dataframe(
        sensitivity.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Tải bảng độ nhạy ngân sách",
        data=csv_bytes(
            sensitivity
        ),
        file_name=(
            "bai02_243_do_nhay_ngan_sach.csv"
        ),
        mime="text/csv",
    )

with tabs[4]:
    st.subheader(
        "Câu 2.4.4 — Kịch bản ưu tiên nhân lực số"
    )

    baseline = human_priority[
        "baseline"
    ]

    priority = human_priority[
        "priority"
    ]

    h1, h2, h3, h4 = st.columns(4)

    h1.metric(
        "Trạng thái cơ sở",
        format_status(
            baseline["success"]
        ),
    )

    h2.metric(
        "Trạng thái ưu tiên",
        format_status(
            priority["success"]
        ),
    )

    if (
        baseline["success"]
        and priority["success"]
    ):
        h3.metric(
            "Z* cơ sở",
            f"{baseline['z']:.2f}",
        )

        h4.metric(
            "Thay đổi Z*",
            (
                f"{human_priority['objective_change']:.2f}"
            ),
        )

        comparison = human_priority[
            "comparison"
        ]

        comparison_long = comparison.melt(
            id_vars="Hạng mục",
            var_name="Kịch bản",
            value_name="Phân bổ",
        )

        fig_human = px.bar(
            comparison_long,
            x="Hạng mục",
            y="Phân bổ",
            color="Kịch bản",
            barmode="group",
            text_auto=".1f",
            color_discrete_sequence=[
                ROSE,
                MINT,
            ],
        )

        fig_human = style_plotly(
            fig_human,
            title=(
                "So sánh phân bổ cơ sở và ưu tiên nhân lực số"
            ),
            x_title="Hạng mục",
            y_title="Ngân sách (nghìn tỷ VND)",
        )

        st.plotly_chart(
            fig_human,
            use_container_width=True,
        )

        objective_compare = pd.DataFrame({
            "Kịch bản": [
                "Cơ sở",
                "Ưu tiên nhân lực",
            ],
            "Z*": [
                baseline["z"],
                priority["z"],
            ],
        })

        fig_objective = px.bar(
            objective_compare,
            x="Kịch bản",
            y="Z*",
            color="Kịch bản",
            text_auto=".2f",
            color_discrete_sequence=[
                PINK,
                LAVENDER,
            ],
        )

        fig_objective.update_layout(
            showlegend=False
        )

        fig_objective = style_plotly(
            fig_objective,
            title=(
                "Chi phí cơ hội của chính sách ưu tiên nhân lực"
            ),
            x_title="Kịch bản",
            y_title="Z*",
        )

        st.plotly_chart(
            fig_objective,
            use_container_width=True,
        )

        st.dataframe(
            comparison.round(3),
            use_container_width=True,
            hide_index=True,
        )

        if (
            human_priority[
                "objective_change"
            ]
            < 0
        ):
            st.warning(
                f"Ràng buộc x₃ ≥ {priority_human_floor:.1f} "
                f"vẫn khả thi nhưng làm Z* giảm "
                f"{abs(human_priority['objective_change']):.2f}. "
                "Đây là chi phí kinh tế mô phỏng của việc ưu tiên "
                "nhân lực số so với phân bổ thuần túy theo hiệu quả biên."
            )
        else:
            st.success(
                "Kịch bản ưu tiên nhân lực không làm giảm Z* "
                "với bộ tham số hiện tại."
            )

        st.download_button(
            "⬇️ Tải bảng so sánh ưu tiên nhân lực",
            data=csv_bytes(
                comparison
            ),
            file_name=(
                "bai02_244_uu_tien_nhan_luc.csv"
            ),
            mime="text/csv",
        )
    else:
        st.error(
            "Kịch bản ưu tiên nhân lực không khả thi "
            "với ngân sách và các mức sàn hiện tại."
        )

with tabs[5]:
    st.subheader(
        "Mục 2.5 — Thảo luận chính sách"
    )

    if scipy_result["success"]:
        scipy_budget_shadow = float(
            scipy_result[
                "constraint_table"
            ].iloc[0][
                "Shadow price của Max theo RHS"
            ]
        )
    else:
        scipy_budget_shadow = np.nan

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown(
            """
            <div style="
                background:#FFF1F6;
                border:1px solid #F0D5DF;
                border-radius:16px;
                padding:18px;
                min-height:285px;
            ">
                <h4 style="color:#503743;">
                    2.5a — Giá trị biên của vốn công
                </h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"Shadow price ước tính của ngân sách tổng là "
            f"**{scipy_budget_shadow:.3f}**. Trong khoảng độ nhạy "
            "đang xét, tăng thêm 1 nghìn tỷ VND làm Z* tăng xấp xỉ "
            f"{scipy_budget_shadow:.3f} đơn vị."
        )

        st.write(
            "Giá trị này là mức lợi ích biên trong mô hình, "
            "không tự động trở thành cận trên chính thức của "
            "chi phí cơ hội vì mô hình chưa phản ánh đầy đủ "
            "thuế, nợ công, độ trễ và rủi ro thực thi."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with d2:
        st.markdown(
            """
            <div style="
                background:#F7F0FC;
                border:1px solid #E4D5F1;
                border-radius:16px;
                padding:18px;
                min-height:285px;
            ">
                <h4 style="color:#503743;">
                    2.5b — Vì sao sàn R&D thấp?
                </h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"R&D có hệ số tác động cao nhất "
            f"({impact_x4:.2f}) nhưng mức sàn chỉ "
            f"{min_x4:.1f}. Hệ số mục tiêu phản ánh lợi ích biên, "
            "còn mức sàn là yêu cầu chính sách tối thiểu."
        )

        st.write(
            "Sàn thấp không có nghĩa R&D kém quan trọng. "
            "Nó cho phép solver linh hoạt phân bổ phần ngân sách "
            "còn lại theo hiệu quả và các ràng buộc khác."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with d3:
        st.markdown(
            """
            <div style="
                background:#EEF8F7;
                border:1px solid #D1E9E6;
                border-radius:16px;
                padding:18px;
                min-height:285px;
            ">
                <h4 style="color:#503743;">
                    2.5c — Tỷ lệ 35% công nghệ chiến lược
                </h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"Mô hình yêu cầu AI + R&D đạt ít nhất "
            f"{strategic_share:.0%} tổng phân bổ. "
            "Về toán học, yêu cầu này khả thi nếu ngân sách "
            "và các mức sàn tương thích."
        )

        st.write(
            "Trong quản lý thực tế, tính khả thi còn phụ thuộc "
            "vào ưu tiên hạ tầng giao thông, an sinh xã hội, "
            "năng lực giải ngân và khả năng hấp thụ công nghệ."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

with tabs[6]:
    st.subheader(
        "Tác nhân AI phân tích kết quả Bài 2"
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

    if scipy_result["success"]:
        allocation_text = (
            scipy_result["table"]
            .round(4)
            .to_string(index=False)
        )

        result_summary = f"""
BÀI 2 — LP PHÂN BỔ NGÂN SÁCH SỐ

Kết quả SciPy:
- Trạng thái: {scipy_result['status']}
- Z*: {scipy_result['z']:.4f}
- Ngân sách sử dụng: {scipy_result['total_used']:.4f}
- Ngân sách chưa sử dụng: {scipy_result['unused_budget']:.4f}
- Tỷ trọng AI + R&D: {scipy_result['strategic_ratio']:.4%}
- Shadow price ngân sách tổng: {scipy_budget_shadow:.4f}

Phân bổ:
{allocation_text}

Độ nhạy ngân sách:
{sensitivity.round(4).to_string(index=False)}

Kịch bản ưu tiên nhân lực:
- Sàn x3: {priority_human_floor:.2f}
- Khả thi: {priority['success']}
- Z* thay đổi: {human_priority['objective_change']:.4f}
"""
    else:
        result_summary = (
            "Bài toán không khả thi với bộ tham số hiện tại."
        )

    policy_questions = """
1. Khi tăng thêm 1 nghìn tỷ VND ngân sách, Z* thay đổi thế nào?
2. Vì sao R&D có hệ số tác động cao nhất nhưng mức sàn thấp nhất?
3. Tỷ trọng tối thiểu 35% cho AI + R&D có ý nghĩa và hạn chế gì?
4. Kịch bản x3 >= 30 tạo ra đánh đổi chính sách nào?
"""

    with st.expander(
        "Xem dữ liệu sẽ gửi cho Gemini"
    ):
        st.text_area(
            "Tóm tắt kết quả",
            value=result_summary.strip(),
            height=360,
            disabled=True,
        )

    analyze_clicked = st.button(
        "✨ Phân tích kết quả bằng Gemini",
        disabled=(
            not configured
            or not scipy_result["success"]
        ),
        use_container_width=True,
        key="gemini_bai02",
    )

    if analyze_clicked:
        with st.spinner(
            "Gemini đang phân tích kết quả Bài 2..."
        ):
            try:
                analysis = analyze_result(
                    exercise_name=(
                        "Bài 2 — Quy hoạch tuyến tính "
                        "phân bổ ngân sách số"
                    ),
                    model_name=(
                        "Linear Programming bằng SciPy "
                        "và PuLP, dual values và sensitivity analysis"
                    ),
                    parameters={
                        "Ngân sách B":
                            f"{budget:.1f} nghìn tỷ VND",
                        "Sàn x1":
                            f"{min_x1:.1f}",
                        "Sàn x2":
                            f"{min_x2:.1f}",
                        "Sàn x3":
                            f"{min_x3:.1f}",
                        "Sàn x4":
                            f"{min_x4:.1f}",
                        "Tỷ trọng AI + R&D":
                            f"{strategic_share:.1%}",
                        "Hệ số tác động":
                            impacts.tolist(),
                    },
                    result_summary=(
                        result_summary.strip()
                    ),
                    policy_questions=(
                        policy_questions.strip()
                    ),
                )

                st.session_state[
                    "bai02_gemini_analysis"
                ] = analysis

            except GeminiAgentError as error:
                st.error(str(error))

    saved_analysis = st.session_state.get(
        "bai02_gemini_analysis"
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
            "⬇️ Tải phân tích Gemini Bài 2",
            data=saved_analysis.encode(
                "utf-8"
            ),
            file_name=(
                "bai02_phan_tich_gemini.md"
            ),
            mime="text/markdown",
            use_container_width=True,
        )
