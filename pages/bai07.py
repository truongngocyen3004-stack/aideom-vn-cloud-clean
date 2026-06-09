from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.bai07_model import (
    EMISSION,
    ParetoConfig,
    REGION_NAMES,
    SECURITY_REDUCTION,
    SECURITY_RISK,
    run_full_bai07,
)
from services.ai_agent import (
    GeminiAgentError,
    analyze_result,
    gemini_is_configured,
)


# ============================================================
# CẤU HÌNH GIAO DIỆN
# ============================================================

MODEL_VERSION = "bai07_v2_full"

BG = "#FFF9FB"
CARD = "#FFF3F7"
CARD_2 = "#FAF4FC"
TEXT = "#49313D"
TEXT_SUB = "#705865"
BORDER = "#E8C9D6"
GRID = "#EEDDE5"

PINK = "#D989A5"
ROSE = "#F3B7CA"
LAVENDER = "#CDB8E5"
MINT = "#9FD3CF"
YELLOW = "#F2D7A7"
BLUE = "#A9C9E8"

PASTEL_SEQUENCE = [
    PINK,
    MINT,
    LAVENDER,
    YELLOW,
    BLUE,
    ROSE,
]


def inject_css() -> None:
    """CSS riêng của Bài 7, không cần sửa app.py hay ui/theme.py."""

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {BG};
            color: {TEXT};
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT} !important;
        }}

        p, li, label, span {{
            color: {TEXT};
        }}

        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
        }}

        .bai07-header {{
            background:
                linear-gradient(
                    135deg,
                    #F7C6D7 0%,
                    #E6D7F3 52%,
                    #D7ECF0 100%
                );
            border: 1px solid {BORDER};
            border-radius: 22px;
            padding: 24px 26px;
            margin-bottom: 18px;
            box-shadow: 0 8px 22px rgba(92, 53, 74, 0.08);
        }}

        .bai07-title {{
            color: {TEXT};
            font-size: 31px;
            font-weight: 800;
            margin-bottom: 8px;
        }}

        .bai07-subtitle {{
            color: {TEXT_SUB};
            font-size: 15px;
            line-height: 1.65;
        }}

        div[data-testid="stMetric"] {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 10px 12px;
        }}

        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricValue"] * {{
            color: {TEXT} !important;
        }}

        button[data-baseweb="tab"] {{
            color: {TEXT} !important;
            font-weight: 650 !important;
        }}

        .stButton > button {{
            background: {PINK} !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
        }}

        .stDownloadButton > button {{
            background: #E6A5BD !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
        }}

        div[data-testid="stAlert"] {{
            border-radius: 14px;
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 12px;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="bai07-header">
            <div class="bai07-title">{title}</div>
            <div class="bai07-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(
    fig: go.Figure,
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 500,
) -> go.Figure:
    """Đồng bộ màu chữ đậm và nền pastel cho toàn bộ biểu đồ."""

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
            "size": 14,
        },
        title_font={
            "family": "Arial",
            "size": 21,
            "color": TEXT,
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title_text="",
        height=height,
        margin={
            "l": 60,
            "r": 35,
            "t": 78,
            "b": 70,
        },
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "font_color": TEXT,
            "font_size": 13,
        },
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor=BORDER,
        tickfont={
            "color": TEXT,
            "size": 13,
        },
        title_font={
            "color": TEXT,
            "size": 14,
        },
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=BORDER,
        tickfont={
            "color": TEXT,
            "size": 13,
        },
        title_font={
            "color": TEXT,
            "size": 14,
        },
    )

    return fig


def show_table(
    dataframe: pd.DataFrame,
    hide_index: bool = True,
) -> None:
    """Hiển thị bảng có chữ đậm, tránh bảng chữ xám khó đọc."""

    styled = (
        dataframe.style
        .set_properties(
            **{
                "color": TEXT,
                "background-color": "#FFFDFE",
                "border-color": BORDER,
                "font-size": "13px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#F7E1EA"),
                        ("color", TEXT),
                        ("font-weight", "700"),
                        ("border-color", BORDER),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("color", TEXT),
                        ("border-color", BORDER),
                    ],
                },
            ]
        )
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=hide_index,
    )


def csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(
        index=False
    ).encode("utf-8-sig")


# ============================================================
# KHỞI TẠO TRANG
# ============================================================

inject_css()

page_header(
    "Bài 7 — Tối ưu đa mục tiêu Pareto bằng NSGA-II",
    (
        "Tạo tập nghiệm đánh đổi giữa tăng trưởng, bao trùm, "
        "môi trường và an ninh dữ liệu; sau đó chọn nghiệm "
        "thỏa hiệp bằng TOPSIS."
    ),
)

st.markdown(
    f"""
    <div style="
        background:{CARD};
        border:1px solid {BORDER};
        border-radius:16px;
        padding:18px 20px;
        margin-bottom:16px;
        color:{TEXT};
        line-height:1.65;
    ">
        <b>24 biến quyết định:</b> ma trận 6 vùng × 4 hạng mục
        I, D, AI và H.<br>
        <b>Bốn mục tiêu:</b> tối đa tăng trưởng; tối thiểu
        bất bình đẳng, phát thải và rủi ro an ninh dữ liệu ròng.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# THAM SỐ
# ============================================================

with st.expander(
    "⚙️ Thiết lập NSGA-II và TOPSIS",
    expanded=True,
):
    c1, c2, c3, c4 = st.columns(4)

    total_budget = c1.number_input(
        "Ngân sách tổng",
        min_value=30000.0,
        max_value=100000.0,
        value=50000.0,
        step=1000.0,
    )

    min_region = c2.number_input(
        "Sàn mỗi vùng",
        min_value=0.0,
        max_value=15000.0,
        value=5000.0,
        step=500.0,
    )

    max_region = c3.number_input(
        "Trần mỗi vùng",
        min_value=5000.0,
        max_value=30000.0,
        value=13000.0,
        step=500.0,
    )

    min_h = c4.number_input(
        "Sàn H toàn quốc",
        min_value=0.0,
        max_value=30000.0,
        value=12000.0,
        step=500.0,
    )

    c5, c6, c7, c8 = st.columns(4)

    min_d = c5.number_input(
        "Sàn D toàn quốc",
        min_value=0.0,
        max_value=30000.0,
        value=8000.0,
        step=500.0,
    )

    pop_size = c6.slider(
        "Population size",
        min_value=40,
        max_value=200,
        value=100,
        step=20,
    )

    n_gen = c7.slider(
        "Số thế hệ",
        min_value=40,
        max_value=300,
        value=200,
        step=20,
    )

    seed = c8.number_input(
        "Seed",
        min_value=1,
        max_value=999,
        value=42,
        step=1,
    )

    st.markdown(
        f"<b style='color:{TEXT};'>Trọng số TOPSIS chọn nghiệm thỏa hiệp</b>",
        unsafe_allow_html=True,
    )

    w1, w2, w3, w4 = st.columns(4)

    wg = w1.slider(
        "Tăng trưởng",
        0.05,
        0.70,
        0.40,
        0.05,
    )

    wi = w2.slider(
        "Bao trùm",
        0.05,
        0.60,
        0.25,
        0.05,
    )

    we = w3.slider(
        "Môi trường",
        0.05,
        0.60,
        0.20,
        0.05,
    )

    ws = w4.slider(
        "An ninh",
        0.05,
        0.60,
        0.15,
        0.05,
    )

    run_clicked = st.button(
        "🌸 Chạy NSGA-II và chọn nghiệm thỏa hiệp",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# KIỂM TRA TÍNH KHẢ THI
# ============================================================

if min_region > max_region:
    st.error(
        "Sàn mỗi vùng không được lớn hơn trần mỗi vùng."
    )
    st.stop()

if 6 * min_region > total_budget:
    st.error(
        "Tổng sàn của 6 vùng đang lớn hơn ngân sách tổng."
    )
    st.stop()

if 6 * max_region < total_budget:
    st.error(
        "Tổng trần của 6 vùng đang nhỏ hơn ngân sách tổng."
    )
    st.stop()

if min_h + min_d > total_budget:
    st.error(
        "Tổng sàn H và D đang lớn hơn ngân sách tổng."
    )
    st.stop()


# ============================================================
# CHẠY MÔ HÌNH
# ============================================================

signature = (
    MODEL_VERSION,
    total_budget,
    min_region,
    max_region,
    min_h,
    min_d,
    pop_size,
    n_gen,
    int(seed),
    wg,
    wi,
    we,
    ws,
)

signature_changed = (
    st.session_state.get(
        "bai07_signature"
    )
    != signature
)

if (
    run_clicked
    or "bai07_result"
    not in st.session_state
    or signature_changed
):
    st.session_state.pop(
        "bai07_gemini_analysis",
        None,
    )

    config = ParetoConfig(
        total_budget=float(total_budget),
        min_region=float(min_region),
        max_region=float(max_region),
        min_h_total=float(min_h),
        min_d_total=float(min_d),
        pop_size=int(pop_size),
        n_gen=int(n_gen),
        seed=int(seed),
    )

    policy_weights = np.array(
        [
            wg,
            wi,
            we,
            ws,
        ],
        dtype=float,
    )

    try:
        with st.spinner(
            "Đang tạo tập nghiệm Pareto và tính TOPSIS..."
        ):
            st.session_state[
                "bai07_result"
            ] = run_full_bai07(
                config=config,
                policy_weights=policy_weights,
            )

            st.session_state[
                "bai07_signature"
            ] = signature

    except Exception as error:
        st.error(
            f"Không chạy được Bài 7: {type(error).__name__}: {error}"
        )
        st.stop()


result = st.session_state[
    "bai07_result"
]

pareto = result[
    "pareto"
]

scored = result[
    "topsis"
][
    "scored"
]

compromise = result[
    "topsis"
][
    "compromise"
]

costs = result[
    "opportunity_cost"
]


# ============================================================
# CÁC TAB
# ============================================================

tabs = st.tabs(
    [
        "7.1 — Bối cảnh",
        "7.2–7.3 — Mô hình & tham số",
        "7.4.1 — NSGA-II",
        "7.4.2 — Pareto trực quan",
        "7.4.3 — TOPSIS",
        "7.4.4 — Chi phí cơ hội",
        "7.5 — Chính sách",
        "✨ Phân tích AI",
    ]
)


# ============================================================
# TAB 7.1
# ============================================================

with tabs[0]:
    st.subheader(
        "7.1 — Bốn mục tiêu chính sách xung đột"
    )

    policy = pd.DataFrame(
        {
            "Mục tiêu": [
                "Tăng trưởng",
                "Bao trùm",
                "Môi trường",
                "An ninh dữ liệu",
            ],
            "Thước đo": [
                "GDP gain",
                "MAD ngân sách vùng",
                "CO₂ gián tiếp",
                "Rủi ro AI trừ giảm rủi ro H",
            ],
            "Hướng tối ưu": [
                "Tối đa",
                "Tối thiểu",
                "Tối thiểu",
                "Tối thiểu",
            ],
        }
    )

    show_table(
        policy
    )

    # Sankey đã sửa màu chữ đậm và màu liên kết pastel,
    # không còn chữ xám khó đọc như phiên bản cũ.
    sankey = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                textfont={
                    "family": "Arial",
                    "color": TEXT,
                    "size": 17,
                },
                node={
                    "pad": 22,
                    "thickness": 27,
                    "line": {
                        "color": "#A9788D",
                        "width": 1.2,
                    },
                    "label": [
                        "Ngân sách số",
                        "Tăng trưởng",
                        "Bao trùm",
                        "Môi trường",
                        "An ninh",
                        "Tập Pareto",
                        "Quyết định chính sách",
                    ],
                    "color": [
                        PINK,
                        MINT,
                        LAVENDER,
                        YELLOW,
                        BLUE,
                        ROSE,
                        PINK,
                    ],
                },
                link={
                    "source": [
                        0,
                        0,
                        0,
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                    ],
                    "target": [
                        1,
                        2,
                        3,
                        4,
                        5,
                        5,
                        5,
                        5,
                        6,
                    ],
                    "value": [
                        25,
                        25,
                        25,
                        25,
                        25,
                        25,
                        25,
                        25,
                        100,
                    ],
                    "color": [
                        "rgba(159,211,207,0.42)",
                        "rgba(205,184,229,0.42)",
                        "rgba(242,215,167,0.42)",
                        "rgba(169,201,232,0.42)",
                        "rgba(159,211,207,0.42)",
                        "rgba(205,184,229,0.42)",
                        "rgba(242,215,167,0.42)",
                        "rgba(169,201,232,0.42)",
                        "rgba(217,137,165,0.42)",
                    ],
                },
            )
        ]
    )

    sankey.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={
            "family": "Arial",
            "color": TEXT,
            "size": 16,
        },
        title={
            "text": "Từ ngân sách số đến tập nghiệm Pareto",
            "x": 0.02,
            "xanchor": "left",
        },
        title_font={
            "color": TEXT,
            "size": 22,
        },
        height=650,
        margin={
            "l": 30,
            "r": 30,
            "t": 80,
            "b": 35,
        },
    )

    st.plotly_chart(
        sankey,
        use_container_width=True,
    )


# ============================================================
# TAB 7.2–7.3
# ============================================================

with tabs[1]:
    st.subheader(
        "7.2 — Mô hình toán học"
    )

    st.latex(
        r"""
        \max f_1(x)
        =
        \sum_r\sum_j
        \beta_{j,r}x_{j,r}
        """
    )

    st.latex(
        r"""
        \min f_2(x)
        =
        \frac{1}{6\bar B}
        \sum_r
        |B_r-\bar B|
        """
    )

    st.latex(
        r"""
        \min f_3(x)
        =
        \sum_r
        e_r
        (x_{I,r}+x_{AI,r})
        """
    )

    st.latex(
        r"""
        \min f_4(x)
        =
        \sum_r
        \rho_r x_{AI,r}
        -
        \sum_r
        \sigma_r x_{H,r}
        """
    )

    params = pd.DataFrame(
        {
            "Vùng": REGION_NAMES,
            "e — phát thải": EMISSION,
            "rho — rủi ro AI": SECURITY_RISK,
            "sigma — giảm rủi ro H": SECURITY_REDUCTION,
        }
    )

    show_table(
        params.round(4)
    )

    st.info(
        "NSGA-II không tạo một nghiệm duy nhất. "
        "Nó tạo tập nghiệm không bị trội, sau đó TOPSIS "
        "được dùng để chọn một phương án thỏa hiệp."
    )


# ============================================================
# TAB 7.4.1
# ============================================================

with tabs[2]:
    st.subheader(
        "7.4.1 — Quần thể Pareto cuối cùng"
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Engine",
        result[
            "engine"
        ],
    )

    k2.metric(
        "Số nghiệm Pareto",
        f"{len(pareto):,}",
    )

    k3.metric(
        "Tăng trưởng cao nhất",
        (
            f"{pareto['growth_gain'].max():,.0f}"
        ),
    )

    k4.metric(
        "Phát thải thấp nhất",
        (
            f"{pareto['emission'].min():,.0f}"
        ),
    )

    show_table(
        pareto.round(4)
    )

    st.download_button(
        "⬇️ Tải tập Pareto",
        data=csv_bytes(
            pareto
        ),
        file_name=(
            "bai07_pareto.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# TAB 7.4.2
# ============================================================

with tabs[3]:
    st.subheader(
        "7.4.2 — Trực quan hóa tập Pareto"
    )

    fig3d = px.scatter_3d(
        pareto,
        x="growth_gain",
        y="inequality",
        z="emission",
        color="security_risk",
        size="AI_total",
        hover_data=[
            "solution_id",
            "H_total",
            "D_total",
        ],
        color_continuous_scale=[
            "#9FD3CF",
            "#F2D7A7",
            "#D989A5",
        ],
    )

    fig3d.update_layout(
        height=700,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={
            "color": TEXT,
            "size": 13,
        },
        title={
            "text":
                "Tập nghiệm Pareto trong không gian ba chiều",
            "x": 0.02,
        },
        title_font={
            "color": TEXT,
            "size": 21,
        },
        scene={
            "xaxis": {
                "title": "Tăng trưởng",
                "backgroundcolor": BG,
                "gridcolor": GRID,
                "tickfont": {
                    "color": TEXT,
                },
            },
            "yaxis": {
                "title": "Bất bình đẳng",
                "backgroundcolor": BG,
                "gridcolor": GRID,
                "tickfont": {
                    "color": TEXT,
                },
            },
            "zaxis": {
                "title": "Phát thải",
                "backgroundcolor": BG,
                "gridcolor": GRID,
                "tickfont": {
                    "color": TEXT,
                },
            },
        },
    )

    st.plotly_chart(
        fig3d,
        use_container_width=True,
    )

    parallel = px.parallel_coordinates(
        pareto,
        dimensions=[
            "growth_gain",
            "inequality",
            "emission",
            "security_risk",
            "I_total",
            "D_total",
            "AI_total",
            "H_total",
        ],
        color="growth_gain",
        color_continuous_scale=[
            "#F4B8C8",
            "#CDB8E5",
            "#9FD3CF",
        ],
    )

    parallel.update_layout(
        height=650,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={
            "color": TEXT,
            "size": 13,
        },
        title={
            "text":
                "Biểu đồ song song của các mục tiêu và cơ cấu vốn",
            "x": 0.02,
        },
        title_font={
            "color": TEXT,
            "size": 21,
        },
    )

    st.plotly_chart(
        parallel,
        use_container_width=True,
    )


# ============================================================
# TAB 7.4.3
# ============================================================

with tabs[4]:
    st.subheader(
        "7.4.3 — Nghiệm thỏa hiệp bằng TOPSIS"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Solution ID",
        int(
            compromise[
                "solution_id"
            ]
        ),
    )

    m2.metric(
        "TOPSIS score",
        (
            f"{compromise['TOPSIS_score']:.4f}"
        ),
    )

    m3.metric(
        "Tăng trưởng",
        (
            f"{compromise['growth_gain']:,.0f}"
        ),
    )

    m4.metric(
        "Bất bình đẳng",
        (
            f"{compromise['inequality']:.4f}"
        ),
    )

    show_table(
        scored.head(
            15
        ).round(
            4
        )
    )

    allocation = result[
        "compromise_allocation"
    ]

    heat = go.Figure(
        data=go.Heatmap(
            z=allocation.values,
            x=allocation.columns,
            y=allocation.index,
            colorscale=[
                [0.00, "#FFF7FA"],
                [0.50, "#E7D8F3"],
                [1.00, "#7DBFB4"],
            ],
            text=np.round(
                allocation.values,
                0,
            ),
            texttemplate="%{text:,.0f}",
            textfont={
                "color": TEXT,
                "size": 13,
            },
            colorbar={
                "title": "Ngân sách",
                "tickfont": {
                    "color": TEXT,
                },
            },
        )
    )

    st.plotly_chart(
        style_plotly(
            heat,
            "Ma trận phân bổ của nghiệm thỏa hiệp",
            "Hạng mục",
            "Vùng",
            590,
        ),
        use_container_width=True,
    )


# ============================================================
# TAB 7.4.4
# ============================================================

with tabs[5]:
    st.subheader(
        "7.4.4 — Chi phí cơ hội của tăng trưởng cực đại"
    )

    opportunity = pd.DataFrame(
        {
            "Chỉ tiêu": [
                "Lợi thế tăng trưởng so với thỏa hiệp",
                "Hy sinh bao trùm",
                "Hy sinh môi trường",
                "Hy sinh an ninh dữ liệu",
            ],
            "Phần trăm": [
                costs[
                    "growth_advantage_pct"
                ],
                costs[
                    "inclusion_sacrifice_pct"
                ],
                costs[
                    "environment_sacrifice_pct"
                ],
                costs[
                    "security_sacrifice_pct"
                ],
            ],
        }
    )

    show_table(
        opportunity.round(
            3
        )
    )

    fig_cost = px.bar(
        opportunity,
        x="Chỉ tiêu",
        y="Phần trăm",
        color="Chỉ tiêu",
        text_auto=".2f",
        color_discrete_sequence=[
            PINK,
            MINT,
            LAVENDER,
            YELLOW,
        ],
    )

    fig_cost.update_layout(
        showlegend=False
    )

    fig_cost.update_traces(
        textfont={
            "color": TEXT,
            "size": 13,
        }
    )

    st.plotly_chart(
        style_plotly(
            fig_cost,
            "Đánh đổi của nghiệm tăng trưởng cao nhất",
            "Mục tiêu",
            "% so với nghiệm thỏa hiệp",
            520,
        ),
        use_container_width=True,
    )


# ============================================================
# TAB 7.5
# ============================================================

with tabs[6]:
    st.subheader(
        "7.5 — Thảo luận chính sách"
    )

    st.markdown(
        f"""
        **a) Nghiệm thỏa hiệp** có tăng trưởng khoảng
        **{compromise['growth_gain']:,.0f}**, bất bình đẳng
        **{compromise['inequality']:.4f}**, phát thải
        **{compromise['emission']:,.0f}** và rủi ro an ninh
        **{compromise['security_risk']:,.0f}**.

        **b) Phù hợp với định hướng xanh:** giảm trọng số tăng trưởng,
        tăng trọng số môi trường; đồng thời đặt trần đầu tư I+AI
        tại vùng có cường độ phát thải cao và tăng đầu tư H, D,
        điện sạch cho trung tâm dữ liệu.

        **c) NSGA-II khác LP đơn mục tiêu:** NSGA-II tạo bản đồ
        các phương án không bị trội, không quyết định thay cho cơ quan
        hoạch định. Trọng số TOPSIS và lựa chọn cuối cùng vẫn là
        quyết định giá trị, thể chế và trách nhiệm giải trình.

        **d) An ninh dữ liệu:** nếu ưu tiên an ninh, cần tăng trọng số
        an ninh trong TOPSIS, tăng đầu tư H và giảm tốc độ mở rộng AI
        tại những vùng có rủi ro cao nhưng năng lực nhân lực còn yếu.
        """
    )


# ============================================================
# TAB AI
# ============================================================

with tabs[7]:
    st.subheader(
        "Tác nhân Gemini phân tích kết quả Bài 7"
    )

    configured = (
        gemini_is_configured()
    )

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
BÀI 7 — NSGA-II PARETO

Thông số:
- Ngân sách: {total_budget}
- Sàn vùng: {min_region}
- Trần vùng: {max_region}
- Sàn H: {min_h}
- Sàn D: {min_d}
- Population size: {pop_size}
- Số thế hệ: {n_gen}
- Trọng số TOPSIS: tăng trưởng={wg}, bao trùm={wi},
  môi trường={we}, an ninh={ws}

Kết quả:
- Engine: {result['engine']}
- Số nghiệm Pareto: {len(pareto)}
- Solution thỏa hiệp: {int(compromise['solution_id'])}
- TOPSIS score: {compromise['TOPSIS_score']:.6f}
- Growth gain: {compromise['growth_gain']:.6f}
- Inequality: {compromise['inequality']:.6f}
- Emission: {compromise['emission']:.6f}
- Security risk: {compromise['security_risk']:.6f}

Chi phí cơ hội:
{opportunity.round(4).to_string(index=False)}

Phân bổ nghiệm thỏa hiệp:
{allocation.round(3).to_string()}
"""

    policy_questions = """
1. Nghiệm thỏa hiệp có cân bằng tốt giữa bốn mục tiêu không?
2. Mục tiêu nào đang được ưu tiên mạnh nhất và mục tiêu nào yếu nhất?
3. Nếu tăng ưu tiên an ninh dữ liệu thì cơ cấu vốn nên thay đổi thế nào?
4. Nghiệm tăng trưởng cực đại phải đánh đổi bao nhiêu về bao trùm,
   môi trường và an ninh?
5. Đề xuất phương án chính sách ngắn hạn và dài hạn cho Việt Nam.
6. Nêu rõ giới hạn của NSGA-II và vai trò của quyết định chính trị.
"""

    with st.expander(
        "Xem nội dung sẽ gửi cho Gemini"
    ):
        st.text_area(
            "Tóm tắt kết quả",
            value=result_summary.strip(),
            height=440,
            disabled=True,
            key="bai07_ai_preview",
        )

    analyze_clicked = st.button(
        "✨ Phân tích kết quả bằng Gemini",
        disabled=not configured,
        use_container_width=True,
        key="gemini_bai07",
    )

    if analyze_clicked:
        try:
            with st.spinner(
                "Gemini đang phân tích Bài 7..."
            ):
                analysis = analyze_result(
                    exercise_name=(
                        "Bài 7 — NSGA-II Pareto"
                    ),
                    model_name=(
                        "NSGA-II 24 biến, 4 mục tiêu "
                        "và TOPSIS chọn nghiệm thỏa hiệp"
                    ),
                    parameters={
                        "total_budget":
                            total_budget,
                        "min_region":
                            min_region,
                        "max_region":
                            max_region,
                        "min_h_total":
                            min_h,
                        "min_d_total":
                            min_d,
                        "population_size":
                            pop_size,
                        "n_generations":
                            n_gen,
                        "TOPSIS_weights": {
                            "growth":
                                wg,
                            "inclusion":
                                wi,
                            "environment":
                                we,
                            "security":
                                ws,
                        },
                    },
                    result_summary=(
                        result_summary.strip()
                    ),
                    policy_questions=(
                        policy_questions.strip()
                    ),
                )

                st.session_state[
                    "bai07_gemini_analysis"
                ] = analysis

        except GeminiAgentError as error:
            st.error(
                str(error)
            )

        except Exception as error:
            st.error(
                f"Lỗi ngoài dự kiến khi gọi Gemini: "
                f"{type(error).__name__}: {error}"
            )

    saved_analysis = (
        st.session_state.get(
            "bai07_gemini_analysis"
        )
    )

    if saved_analysis:
        st.markdown(
            f"""
            <div style="
                background:{CARD_2};
                border:1px solid {BORDER};
                border-left:6px solid {PINK};
                border-radius:16px;
                padding:20px;
                margin-top:16px;
                color:{TEXT};
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
            "⬇️ Tải phân tích Gemini Bài 7",
            data=saved_analysis.encode(
                "utf-8"
            ),
            file_name=(
                "bai07_phan_tich_gemini.md"
            ),
            mime="text/markdown",
            use_container_width=True,
            key="bai07_download_ai",
        )
