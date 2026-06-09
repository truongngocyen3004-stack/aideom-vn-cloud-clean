from __future__ import annotations

from typing import Any

import streamlit as st


# ============================================================
# BẢNG MÀU PASTEL DÙNG CHUNG CHO TOÀN WEBSITE
# ============================================================

BG_MAIN = "#FFF9FB"
BG_CARD = "#FFF3F7"
BG_SOFT = "#FAF4FC"

TEXT_MAIN = "#49313D"
TEXT_SUB = "#705865"
TEXT_LIGHT = "#8A7280"

PINK = "#D989A5"
ROSE = "#F3B7CA"
LAVENDER = "#CDB8E5"
MINT = "#9FD3CF"
PEACH = "#F3CEBD"
YELLOW = "#F2D7A7"
BLUE = "#A9C9E8"

BORDER = "#E8C9D6"
GRID = "#EEDDE5"

PASTEL_COLORS = [
    PINK,
    MINT,
    LAVENDER,
    YELLOW,
    BLUE,
    ROSE,
    PEACH,
]

# Các bí danh giữ tương thích với những page cũ.
BG = BG_MAIN
CARD = BG_CARD
CARD_2 = BG_SOFT
TEXT = TEXT_MAIN
PASTEL_SEQUENCE = PASTEL_COLORS


# ============================================================
# CSS TOÀN CỤC
# ============================================================

def inject_global_css() -> None:
    """
    Chèn CSS pastel dùng chung cho toàn bộ ứng dụng Streamlit.
    Có thể gọi nhiều lần mà không làm hỏng giao diện.
    """

    st.markdown(
        f"""
        <style>
        :root {{
            --aideom-bg: {BG_MAIN};
            --aideom-card: {BG_CARD};
            --aideom-text: {TEXT_MAIN};
            --aideom-subtext: {TEXT_SUB};
            --aideom-border: {BORDER};
            --aideom-pink: {PINK};
        }}

        html, body, [class*="css"] {{
            font-family:
                Inter,
                "Segoe UI",
                Arial,
                sans-serif;
        }}

        .stApp {{
            background: {BG_MAIN};
            color: {TEXT_MAIN};
        }}

        section[data-testid="stSidebar"] {{
            background:
                linear-gradient(
                    180deg,
                    #FCEEF4 0%,
                    #FFF7FA 100%
                );
            border-right: 1px solid {BORDER};
        }}

        section[data-testid="stSidebar"] * {{
            color: {TEXT_MAIN};
        }}

        .block-container {{
            padding-top: 1.35rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT_MAIN} !important;
            letter-spacing: -0.01em;
        }}

        p, li, label, span {{
            color: {TEXT_MAIN};
        }}

        a {{
            color: #A95575 !important;
        }}

        /* Tiêu đề trang */
        .aideom-page-header {{
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
            box-shadow:
                0 8px 22px
                rgba(92, 53, 74, 0.08);
        }}

        .aideom-page-title {{
            color: {TEXT_MAIN};
            font-size: 31px;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 8px;
        }}

        .aideom-page-subtitle {{
            color: {TEXT_SUB};
            font-size: 15px;
            line-height: 1.65;
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 12px 14px;
            box-shadow:
                0 5px 14px
                rgba(92, 53, 74, 0.05);
        }}

        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricValue"] *,
        div[data-testid="stMetricDelta"] * {{
            color: {TEXT_MAIN} !important;
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            color: {TEXT_MAIN} !important;
            font-weight: 650 !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: #A95575 !important;
            border-bottom-color: {PINK} !important;
        }}

        /* Nút */
        .stButton > button {{
            background: {PINK} !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            min-height: 42px;
        }}

        .stButton > button:hover {{
            background: #C97595 !important;
            color: white !important;
            box-shadow:
                0 5px 14px
                rgba(92, 53, 74, 0.16);
        }}

        .stButton > button:disabled {{
            background: #E8CAD5 !important;
            color: #FFFFFF !important;
            opacity: 0.80;
        }}

        .stDownloadButton > button {{
            background: #E6A5BD !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            min-height: 42px;
        }}

        /* Ô nhập */
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div {{
            background: #FFFDFE !important;
            border-color: {BORDER} !important;
            color: {TEXT_MAIN} !important;
        }}

        input, textarea {{
            color: {TEXT_MAIN} !important;
        }}

        /* Slider */
        div[data-testid="stSlider"] * {{
            color: {TEXT_MAIN};
        }}

        /* Bảng */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 12px;
            overflow: hidden;
            background: #FFFDFE;
        }}

        /* Expander */
        details[data-testid="stExpander"] {{
            background: #FFFDFE;
            border: 1px solid {BORDER};
            border-radius: 14px;
        }}

        /* Alert */
        div[data-testid="stAlert"] {{
            border-radius: 14px;
        }}

        /* Code */
        code {{
            color: #62384A !important;
        }}

        pre {{
            background: #FFF3F7 !important;
            border: 1px solid {BORDER} !important;
            border-radius: 12px !important;
        }}

        /* Menu trang */
        nav[data-testid="stSidebarNav"] span {{
            color: {TEXT_MAIN} !important;
        }}

        /* Plotly container */
        div[data-testid="stPlotlyChart"] {{
            border-radius: 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_theme() -> None:
    """
    Hàm mà app.py hiện đang import.

    Đây là bí danh chính thức của inject_global_css(),
    giúp tương thích với app.py và các phiên bản code cũ.
    """

    inject_global_css()


# ============================================================
# TIÊU ĐỀ TRANG
# ============================================================

def page_header(
    title: str,
    subtitle: str = "",
) -> None:
    """
    Hiển thị tiêu đề gradient pastel cho từng bài.
    """

    st.markdown(
        f"""
        <div class="aideom-page-header">
            <div class="aideom-page-title">
                {title}
            </div>
            <div class="aideom-page-subtitle">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# STYLE PLOTLY
# ============================================================

def style_plotly(
    fig: Any,
    title: str = "",
    x_title: str = "",
    y_title: str = "",
    height: int = 500,
) -> Any:
    """
    Đồng bộ nền pastel và chữ đậm cho biểu đồ Plotly.

    Hàm nhận Any để không bắt buộc import plotly trong theme.py.
    """

    fig.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
        },
        paper_bgcolor=BG_MAIN,
        plot_bgcolor=BG_MAIN,
        font={
            "family":
                'Inter, "Segoe UI", Arial, sans-serif',
            "color": TEXT_MAIN,
            "size": 14,
        },
        title_font={
            "family":
                'Inter, "Segoe UI", Arial, sans-serif',
            "size": 21,
            "color": TEXT_MAIN,
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title_text="",
        height=int(height),
        margin={
            "l": 60,
            "r": 35,
            "t": 78,
            "b": 70,
        },
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "font_color": TEXT_MAIN,
            "font_size": 13,
        },
    )

    # Một số loại biểu đồ như Sankey hoặc Polar không có xaxis/yaxis.
    try:
        fig.update_xaxes(
            showgrid=False,
            linecolor=BORDER,
            tickfont={
                "color": TEXT_MAIN,
                "size": 13,
            },
            title_font={
                "color": TEXT_MAIN,
                "size": 14,
            },
        )
    except Exception:
        pass

    try:
        fig.update_yaxes(
            gridcolor=GRID,
            zerolinecolor=BORDER,
            tickfont={
                "color": TEXT_MAIN,
                "size": 13,
            },
            title_font={
                "color": TEXT_MAIN,
                "size": 14,
            },
        )
    except Exception:
        pass

    return fig
