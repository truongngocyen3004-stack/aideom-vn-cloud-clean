from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.bai01_model import (
    calculate_tfp,
    forecast_to_2030,
    forecast_with_mean_tfp,
    growth_accounting,
    load_macro_data,
)
from services.ai_agent import (
    GeminiAgentError,
    analyze_result,
    gemini_is_configured,
)
from ui.theme import PASTEL_COLORS, page_header


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "vietnam_macro_2020_2025.csv"

PINK = "#D989A5"
ROSE = "#F4B8C8"
LAVENDER = "#CDB8E5"
MINT = "#A8D5D1"
YELLOW = "#F2D7A7"
BLUE = "#A9C9E8"
TEXT = "#503743"
GRID = "#EEDFE5"
BG = "#FFF9FB"


def style_plotly(
    fig: go.Figure,
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 430,
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
            "b": 55,
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


def classify_mape(
    mape: float,
) -> str:
    if mape < 5:
        return "Rất tốt"
    if mape < 10:
        return "Tốt"
    if mape < 20:
        return "Có thể chấp nhận"
    return "Cần hiệu chỉnh"


def run_model(
    data: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    theta: float,
    target_d_2030: float,
    target_ai_2030: float,
    target_h_2030: float,
    k_growth_pct: float,
    l_growth_pct: float,
    tfp_growth_pct: float,
) -> dict:
    tfp_table = calculate_tfp(
        data=data,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        theta=theta,
    )

    forecast_result = (
        forecast_with_mean_tfp(
            tfp_table=tfp_table,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            theta=theta,
        )
    )

    accounting_result = (
        growth_accounting(
            tfp_table=tfp_table,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            theta=theta,
        )
    )

    scenario_result = forecast_to_2030(
        tfp_table=tfp_table,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        theta=theta,
        target_d_2030=target_d_2030,
        target_ai_2030=target_ai_2030,
        target_h_2030=target_h_2030,
        k_growth_pct=k_growth_pct,
        l_growth_pct=l_growth_pct,
        tfp_growth_pct=tfp_growth_pct,
    )

    return {
        "input_data": data,
        "tfp": tfp_table,
        "forecast": forecast_result,
        "accounting": accounting_result,
        "scenario": scenario_result,
    }


page_header(
    "Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa",
    "Tính TFP, đánh giá độ phù hợp của mô hình, phân rã tăng trưởng và dự báo GDP Việt Nam đến năm 2030.",
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
        Y<sub>t</sub> = A<sub>t</sub>
        K<sub>t</sub><sup>α</sup>
        L<sub>t</sub><sup>β</sup>
        D<sub>t</sub><sup>γ</sup>
        AI<sub>t</sub><sup>δ</sup>
        H<sub>t</sub><sup>θ</sup>
        &nbsp;&nbsp;với&nbsp;&nbsp;
        α + β + γ + δ + θ = 1.
    </div>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(
        f"Không tìm thấy file dữ liệu: {DATA_PATH}"
    )
    st.stop()

default_data = load_macro_data(
    DATA_PATH
)

with st.expander(
    "⚙️ Thiết lập mô hình và kịch bản",
    expanded=True,
):
    st.markdown(
        "**Hệ số co giãn của hàm sản xuất**"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        alpha = st.slider(
            "α — Vốn K",
            min_value=0.10,
            max_value=0.60,
            value=0.33,
            step=0.01,
        )

    with c2:
        beta = st.slider(
            "β — Lao động L",
            min_value=0.10,
            max_value=0.60,
            value=0.42,
            step=0.01,
        )

    with c3:
        gamma = st.slider(
            "γ — Số hóa D",
            min_value=0.00,
            max_value=0.30,
            value=0.10,
            step=0.01,
        )

    with c4:
        delta = st.slider(
            "δ — AI",
            min_value=0.00,
            max_value=0.30,
            value=0.08,
            step=0.01,
        )

    theta = (
        1.0
        - alpha
        - beta
        - gamma
        - delta
    )

    with c5:
        st.metric(
            "θ — Nhân lực H",
            f"{theta:.2f}",
        )

    if theta < 0:
        st.error(
            "Tổng α + β + γ + δ vượt quá 1. "
            "Hãy giảm ít nhất một hệ số."
        )
        st.stop()

    st.markdown(
        "**Kịch bản đến năm 2030**"
    )

    s1, s2, s3 = st.columns(3)

    with s1:
        target_d_2030 = st.number_input(
            "D năm 2030 (%)",
            min_value=19.5,
            max_value=60.0,
            value=30.0,
            step=0.5,
        )

        k_growth_pct = st.number_input(
            "Tăng trưởng K (%/năm)",
            min_value=0.0,
            max_value=15.0,
            value=6.0,
            step=0.5,
        )

    with s2:
        target_ai_2030 = st.number_input(
            "AI năm 2030 (nghìn DN số)",
            min_value=80.1,
            max_value=250.0,
            value=100.0,
            step=1.0,
        )

        l_growth_pct = st.number_input(
            "Tăng trưởng L (%/năm)",
            min_value=-2.0,
            max_value=10.0,
            value=6.0,
            step=0.5,
        )

    with s3:
        target_h_2030 = st.number_input(
            "H năm 2030 (%)",
            min_value=29.2,
            max_value=70.0,
            value=35.0,
            step=0.5,
        )

        tfp_growth_pct = st.number_input(
            "Tăng trưởng TFP (%/năm)",
            min_value=0.0,
            max_value=5.0,
            value=1.2,
            step=0.1,
        )

    run_clicked = st.button(
        "🌸 Chạy toàn bộ mô hình Bài 1",
        use_container_width=True,
        type="primary",
    )

parameter_signature = (
    alpha,
    beta,
    gamma,
    delta,
    theta,
    target_d_2030,
    target_ai_2030,
    target_h_2030,
    k_growth_pct,
    l_growth_pct,
    tfp_growth_pct,
)

if (
    run_clicked
    or "bai01_result" not in st.session_state
    or st.session_state.get(
        "bai01_signature"
    ) != parameter_signature
):
    with st.spinner(
        "Đang tính TFP, MAPE, phân rã tăng trưởng và dự báo 2030..."
    ):
        st.session_state[
            "bai01_result"
        ] = run_model(
            data=default_data,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            theta=theta,
            target_d_2030=target_d_2030,
            target_ai_2030=target_ai_2030,
            target_h_2030=target_h_2030,
            k_growth_pct=k_growth_pct,
            l_growth_pct=l_growth_pct,
            tfp_growth_pct=tfp_growth_pct,
        )

        st.session_state[
            "bai01_signature"
        ] = parameter_signature

result = st.session_state[
    "bai01_result"
]

tfp_table = result["tfp"]
forecast_result = result["forecast"]
accounting = result["accounting"]
scenario = result["scenario"]

tabs = st.tabs([
    "📘 Dữ liệu & mô hình",
    "1.4.1 — TFP",
    "1.4.2 — Dự báo & MAPE",
    "1.4.3 — Phân rã tăng trưởng",
    "1.4.4 — Kịch bản 2030",
    "1.5 — Thảo luận chính sách",
    "✨ Phân tích AI",
])

with tabs[0]:
    st.subheader(
        "Dữ liệu Việt Nam 2020–2025"
    )

    st.dataframe(
        result["input_data"].round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Tải dữ liệu đầu vào",
        data=csv_bytes(
            result["input_data"]
        ),
        file_name=(
            "bai01_du_lieu_dau_vao.csv"
        ),
        mime="text/csv",
    )

    st.info(
        "Đơn vị: GDP và K tính theo nghìn tỷ VND; "
        "L tính theo triệu lao động; D và H tính theo %; "
        "AI tính theo nghìn doanh nghiệp số."
    )

with tabs[1]:
    st.subheader(
        "Câu 1.4.1 — Năng suất nhân tố tổng hợp TFP"
    )

    tfp_2020 = float(
        tfp_table.iloc[0]["TFP_A_t"]
    )
    tfp_2025 = float(
        tfp_table.iloc[-1]["TFP_A_t"]
    )
    total_tfp_change = (
        tfp_2025
        / tfp_2020
        - 1.0
    ) * 100.0
    average_tfp_growth = float(
        tfp_table[
            "TFP_growth_pct"
        ].dropna().mean()
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "TFP năm 2020",
        f"{tfp_2020:.4f}",
    )
    m2.metric(
        "TFP năm 2025",
        f"{tfp_2025:.4f}",
    )
    m3.metric(
        "Thay đổi 2020–2025",
        f"{total_tfp_change:.2f}%",
    )
    m4.metric(
        "Tăng TFP bình quân",
        f"{average_tfp_growth:.2f}%/năm",
    )

    left, right = st.columns(2)

    with left:
        fig_tfp = px.line(
            tfp_table,
            x="Year",
            y="TFP_A_t",
            markers=True,
            color_discrete_sequence=[
                PINK
            ],
        )

        fig_tfp.update_traces(
            line={
                "width": 3
            },
            marker={
                "size": 9
            },
        )

        fig_tfp = style_plotly(
            fig_tfp,
            title=(
                "Năng suất nhân tố tổng hợp Việt Nam, 2020–2025"
            ),
            x_title="Năm",
            y_title="TFP Aₜ",
        )

        st.plotly_chart(
            fig_tfp,
            use_container_width=True,
        )

    with right:
        tfp_growth_plot = (
            tfp_table.dropna(
                subset=["TFP_growth_pct"]
            )
        )

        fig_tfp_growth = px.bar(
            tfp_growth_plot,
            x="Year",
            y="TFP_growth_pct",
            color_discrete_sequence=[
                LAVENDER
            ],
            text_auto=".2f",
        )

        fig_tfp_growth = style_plotly(
            fig_tfp_growth,
            title=(
                "Tốc độ tăng TFP theo năm"
            ),
            x_title="Năm",
            y_title="Tăng TFP (%)",
        )

        st.plotly_chart(
            fig_tfp_growth,
            use_container_width=True,
        )

    display_tfp = tfp_table[
        [
            "Year",
            "GDP_trillion_VND",
            "TFP_A_t",
            "TFP_growth_pct",
        ]
    ].rename(
        columns={
            "Year": "Năm",
            "GDP_trillion_VND":
                "GDP thực tế",
            "TFP_A_t": "TFP Aₜ",
            "TFP_growth_pct":
                "Tăng TFP (%)",
        }
    )

    st.dataframe(
        display_tfp.round(4),
        use_container_width=True,
        hide_index=True,
    )

    trend_text = (
        "tăng"
        if total_tfp_change > 0
        else "giảm"
    )

    st.success(
        f"TFP {trend_text} {abs(total_tfp_change):.2f}% "
        f"trong giai đoạn 2020–2025. "
        f"Tốc độ thay đổi bình quân đạt {average_tfp_growth:.2f}%/năm."
    )

    st.download_button(
        "⬇️ Tải bảng TFP",
        data=csv_bytes(
            display_tfp
        ),
        file_name="bai01_141_tfp.csv",
        mime="text/csv",
    )

with tabs[2]:
    st.subheader(
        "Câu 1.4.2 — GDP dự báo và MAPE"
    )

    forecast_table = forecast_result[
        "table"
    ]

    f1, f2, f3, f4 = st.columns(4)

    f1.metric(
        "TFP trung bình",
        f"{forecast_result['mean_tfp']:.4f}",
    )
    f2.metric(
        "MAPE",
        f"{forecast_result['mape']:.2f}%",
    )
    f3.metric(
        "Đánh giá",
        classify_mape(
            forecast_result["mape"]
        ),
    )
    f4.metric(
        "Sai số lớn nhất",
        (
            f"{forecast_result['max_ape']:.2f}% "
            f"({forecast_result['max_error_year']})"
        ),
    )

    left, right = st.columns(2)

    with left:
        forecast_long = forecast_table[
            [
                "Year",
                "GDP_trillion_VND",
                "GDP_forecast",
            ]
        ].melt(
            id_vars="Year",
            var_name="Loại",
            value_name="GDP",
        )

        forecast_long["Loại"] = (
            forecast_long["Loại"]
            .replace({
                "GDP_trillion_VND":
                    "GDP thực tế",
                "GDP_forecast":
                    "GDP dự báo",
            })
        )

        fig_forecast = px.line(
            forecast_long,
            x="Year",
            y="GDP",
            color="Loại",
            markers=True,
            color_discrete_sequence=[
                PINK,
                MINT,
            ],
        )

        fig_forecast.update_traces(
            line={
                "width": 3
            },
            marker={
                "size": 8
            },
        )

        fig_forecast = style_plotly(
            fig_forecast,
            title=(
                "So sánh GDP thực tế và GDP dự báo"
            ),
            x_title="Năm",
            y_title="GDP (nghìn tỷ VND)",
        )

        st.plotly_chart(
            fig_forecast,
            use_container_width=True,
        )

    with right:
        fig_ape = px.bar(
            forecast_table,
            x="Year",
            y="APE_pct",
            color_discrete_sequence=[
                ROSE
            ],
            text_auto=".2f",
        )

        fig_ape.add_hline(
            y=forecast_result["mape"],
            line_dash="dash",
            line_color=PINK,
            annotation_text=(
                f"MAPE = "
                f"{forecast_result['mape']:.2f}%"
            ),
        )

        fig_ape = style_plotly(
            fig_ape,
            title=(
                "Sai số phần trăm tuyệt đối theo năm"
            ),
            x_title="Năm",
            y_title="APE (%)",
        )

        st.plotly_chart(
            fig_ape,
            use_container_width=True,
        )

    display_forecast = forecast_table[
        [
            "Year",
            "GDP_trillion_VND",
            "GDP_forecast",
            "Error",
            "APE_pct",
        ]
    ].rename(
        columns={
            "Year": "Năm",
            "GDP_trillion_VND":
                "GDP thực tế",
            "GDP_forecast":
                "GDP dự báo",
            "Error": "Sai số",
            "APE_pct": "APE (%)",
        }
    )

    st.dataframe(
        display_forecast.round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Tải bảng dự báo và MAPE",
        data=csv_bytes(
            display_forecast
        ),
        file_name=(
            "bai01_142_du_bao_mape.csv"
        ),
        mime="text/csv",
    )

with tabs[3]:
    st.subheader(
        "Câu 1.4.3 — Phân rã tăng trưởng"
    )

    annual = accounting["annual"]
    summary = accounting["summary"]

    contribution_mapping = {
        "TFP_contribution_pp": "TFP",
        "K_contribution_pp": "Vốn K",
        "L_contribution_pp": "Lao động L",
        "D_contribution_pp": "Số hóa D",
        "AI_contribution_pp": "AI",
        "H_contribution_pp": "Nhân lực H",
    }

    annual_long = annual[
        [
            "Period",
            *contribution_mapping.keys(),
        ]
    ].melt(
        id_vars="Period",
        var_name="Yếu tố",
        value_name="Đóng góp",
    )

    annual_long["Yếu tố"] = (
        annual_long["Yếu tố"]
        .map(
            contribution_mapping
        )
    )

    fig_stack = px.bar(
        annual_long,
        x="Period",
        y="Đóng góp",
        color="Yếu tố",
        barmode="relative",
        color_discrete_sequence=[
            PINK,
            ROSE,
            LAVENDER,
            MINT,
            YELLOW,
            BLUE,
        ],
    )

    fig_stack.add_scatter(
        x=annual["Period"],
        y=annual[
            "GDP_growth_log_pct"
        ],
        mode="lines+markers",
        name="Tăng GDP thực tế",
        line={
            "color": TEXT,
            "width": 3,
        },
    )

    fig_stack = style_plotly(
        fig_stack,
        title=(
            "Đóng góp của các yếu tố vào tăng trưởng GDP"
        ),
        x_title="Giai đoạn",
        y_title="Đóng góp (điểm % log)",
        height=500,
    )

    st.plotly_chart(
        fig_stack,
        use_container_width=True,
    )

    fig_share = px.bar(
        summary.sort_values(
            "Share_of_growth_pct"
        ),
        x="Share_of_growth_pct",
        y="Factor",
        orientation="h",
        color="Factor",
        color_discrete_sequence=[
            PINK,
            ROSE,
            LAVENDER,
            MINT,
            YELLOW,
            BLUE,
        ],
        text_auto=".1f",
    )

    fig_share = style_plotly(
        fig_share,
        title=(
            "Tỷ trọng đóng góp bình quân giai đoạn 2020–2025"
        ),
        x_title="Tỷ trọng trong tăng trưởng (%)",
        y_title="Yếu tố",
        height=450,
    )

    fig_share.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        fig_share,
        use_container_width=True,
    )

    st.markdown(
        "**Bảng phân rã theo từng giai đoạn**"
    )

    st.dataframe(
        annual.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "**Bảng tổng hợp toàn giai đoạn**"
    )

    st.dataframe(
        summary.round(4),
        use_container_width=True,
        hide_index=True,
    )

    maximum_residual = float(
        annual[
            "Residual_pp"
        ].abs().max()
    )

    st.info(
        f"Sai số kiểm tra lớn nhất của phương trình phân rã "
        f"là {maximum_residual:.8f} điểm %, gần bằng 0."
    )

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            "⬇️ Tải phân rã theo năm",
            data=csv_bytes(
                annual
            ),
            file_name=(
                "bai01_143_phan_ra_theo_nam.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with d2:
        st.download_button(
            "⬇️ Tải tổng hợp đóng góp",
            data=csv_bytes(
                summary
            ),
            file_name=(
                "bai01_143_tong_hop_dong_gop.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

with tabs[4]:
    st.subheader(
        "Câu 1.4.4 — Mô phỏng GDP Việt Nam đến năm 2030"
    )

    scenario_table = scenario["table"]

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "GDP dự báo 2030",
        f"{scenario['gdp_2030']:,.1f} nghìn tỷ VND",
    )
    s2.metric(
        "Tăng so với 2025",
        f"{scenario['increase_pct']:.2f}%",
    )
    s3.metric(
        "CAGR GDP",
        f"{scenario['cagr_pct']:.2f}%/năm",
    )
    s4.metric(
        "TFP năm 2030",
        f"{scenario['tfp_2030']:.4f}",
    )

    fig_gdp_2030 = px.line(
        scenario_table,
        x="Year",
        y="GDP_forecast",
        markers=True,
        color_discrete_sequence=[
            PINK
        ],
    )

    fig_gdp_2030.update_traces(
        line={
            "width": 3
        },
        marker={
            "size": 9
        },
    )

    fig_gdp_2030 = style_plotly(
        fig_gdp_2030,
        title=(
            "Quỹ đạo GDP dự báo 2025–2030"
        ),
        x_title="Năm",
        y_title="GDP (nghìn tỷ VND)",
    )

    st.plotly_chart(
        fig_gdp_2030,
        use_container_width=True,
    )

    index_columns = {
        "K_index_2025_100": "Vốn K",
        "L_index_2025_100": "Lao động L",
        "D_index_2025_100": "Số hóa D",
        "AI_index_2025_100": "AI",
        "H_index_2025_100": "Nhân lực H",
        "TFP_A_t_index_2025_100": "TFP",
    }

    index_long = scenario_table[
        [
            "Year",
            *index_columns.keys(),
        ]
    ].melt(
        id_vars="Year",
        var_name="Yếu tố",
        value_name="Chỉ số",
    )

    index_long["Yếu tố"] = (
        index_long["Yếu tố"]
        .map(
            index_columns
        )
    )

    fig_index = px.line(
        index_long,
        x="Year",
        y="Chỉ số",
        color="Yếu tố",
        markers=True,
        color_discrete_sequence=[
            PINK,
            ROSE,
            LAVENDER,
            MINT,
            YELLOW,
            BLUE,
        ],
    )

    fig_index = style_plotly(
        fig_index,
        title=(
            "Chỉ số hóa các yếu tố đầu vào, năm 2025 = 100"
        ),
        x_title="Năm",
        y_title="Chỉ số 2025 = 100",
        height=500,
    )

    st.plotly_chart(
        fig_index,
        use_container_width=True,
    )

    st.dataframe(
        scenario_table[
            [
                "Year",
                "K",
                "L",
                "D",
                "AI",
                "H",
                "TFP_A_t",
                "GDP_forecast",
            ]
        ].round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Tải bảng dự báo 2025–2030",
        data=csv_bytes(
            scenario_table
        ),
        file_name=(
            "bai01_144_du_bao_2030.csv"
        ),
        mime="text/csv",
    )

with tabs[5]:
    st.subheader(
        "Mục 1.5 — Thảo luận chính sách"
    )

    summary = accounting["summary"]

    new_factors = (
        summary[
            summary["Factor"].isin(
                ["D", "AI", "H"]
            )
        ]
        .sort_values(
            "Average_contribution_pp",
            ascending=False,
        )
    )

    leading_factor = str(
        new_factors.iloc[0]["Factor"]
    )

    leading_value = float(
        new_factors.iloc[0][
            "Average_contribution_pp"
        ]
    )

    policy_col1, policy_col2, policy_col3 = (
        st.columns(3)
    )

    with policy_col1:
        st.markdown(
            """
            <div style="
                background:#FFF1F6;
                border:1px solid #F0D5DF;
                border-radius:16px;
                padding:18px;
                min-height:250px;
            ">
                <h4 style="color:#503743;">1.5a — Xu hướng TFP</h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"TFP thay đổi {total_tfp_change:.2f}% "
            f"trong giai đoạn 2020–2025. "
            "Xu hướng này phản ánh phần tăng trưởng không được "
            "giải thích trực tiếp bởi K, L, D, AI và H."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with policy_col2:
        st.markdown(
            """
            <div style="
                background:#F7F0FC;
                border:1px solid #E4D5F1;
                border-radius:16px;
                padding:18px;
                min-height:250px;
            ">
                <h4 style="color:#503743;">1.5b — D, AI hay H?</h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"Trong ba yếu tố mới, **{leading_factor}** "
            f"có đóng góp bình quân cao nhất, đạt "
            f"{leading_value:.3f} điểm % log mỗi năm."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with policy_col3:
        st.markdown(
            """
            <div style="
                background:#EEF8F7;
                border:1px solid #D1E9E6;
                border-radius:16px;
                padding:18px;
                min-height:250px;
            ">
                <h4 style="color:#503743;">1.5c — Mục tiêu D = 30%</h4>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            f"Kịch bản đặt D năm 2030 ở mức "
            f"{target_d_2030:.1f}% cho GDP dự báo "
            f"{scenario['gdp_2030']:,.1f} nghìn tỷ VND. "
            "Kết quả còn phụ thuộc vào tốc độ tích lũy vốn, "
            "lao động, TFP, AI và chất lượng nhân lực."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.warning(
        "Mô hình Cobb-Douglas giả định hệ số co giãn ổn định, "
        "lợi suất không đổi theo quy mô và chưa phản ánh đầy đủ "
        "độ trễ chính sách, chất lượng thể chế, rủi ro công nghệ "
        "và tác động qua lại giữa AI với nhân lực."
    )

with tabs[6]:
    st.subheader(
        "Tác nhân AI phân tích kết quả Bài 1"
    )

    ai_summary = f"""
BÀI 1 — COBB-DOUGLAS MỞ RỘNG

Hệ số:
alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f},
delta={delta:.2f}, theta={theta:.2f}.

Kết quả chính:
- TFP 2020: {tfp_2020:.4f}
- TFP 2025: {tfp_2025:.4f}
- Thay đổi TFP 2020-2025: {total_tfp_change:.2f}%
- TFP bình quân: {forecast_result['mean_tfp']:.4f}
- MAPE: {forecast_result['mape']:.2f}%
- Năm có APE lớn nhất: {forecast_result['max_error_year']}
- APE lớn nhất: {forecast_result['max_ape']:.2f}%
- GDP dự báo 2030: {scenario['gdp_2030']:.2f} nghìn tỷ VND
- Tăng GDP 2025-2030: {scenario['increase_pct']:.2f}%
- CAGR GDP 2025-2030: {scenario['cagr_pct']:.2f}%
- TFP 2030: {scenario['tfp_2030']:.4f}
- Yếu tố mới đóng góp lớn nhất: {leading_factor}
- Đóng góp bình quân của yếu tố dẫn đầu: {leading_value:.4f} điểm % log/năm
"""

    policy_questions = f"""
1. Xu hướng TFP giai đoạn 2020-2025 nói gì về chất lượng tăng trưởng?
2. Trong D, AI và H, yếu tố nào đóng góp nhiều nhất và vì sao?
3. Với D năm 2030 bằng {target_d_2030:.1f}%, AI bằng
   {target_ai_2030:.1f} nghìn doanh nghiệp số và H bằng
   {target_h_2030:.1f}%, kịch bản có ý nghĩa chính sách gì?
4. Những hạn chế nào khiến kết quả không nên được xem là dự báo chính thức?
"""

    configured = gemini_is_configured()

    if configured:
        st.success(
            "Gemini API đã được cấu hình. "
            "Bạn có thể bấm nút để phân tích kết quả hiện tại."
        )
    else:
        st.warning(
            "Chưa tìm thấy GEMINI_API_KEY trong "
            ".streamlit/secrets.toml. Sau khi thêm khóa, "
            "hãy dừng và chạy lại Streamlit."
        )

    with st.expander(
        "Xem dữ liệu sẽ gửi cho Gemini",
        expanded=False,
    ):
        st.text_area(
            "Tóm tắt kết quả",
            value=ai_summary.strip(),
            height=290,
            disabled=True,
        )

    analyze_clicked = st.button(
        "✨ Phân tích kết quả bằng Gemini",
        disabled=not configured,
        use_container_width=True,
        key="gemini_bai01",
    )

    if analyze_clicked:
        with st.spinner(
            "Gemini đang phân tích kết quả Bài 1..."
        ):
            try:
                analysis = analyze_result(
                    exercise_name=(
                        "Bài 1 — Hàm sản xuất "
                        "Cobb-Douglas mở rộng với AI và số hóa"
                    ),
                    model_name=(
                        "Growth accounting, dự báo bằng TFP "
                        "trung bình và mô phỏng GDP đến 2030"
                    ),
                    parameters={
                        "alpha - Vốn K":
                            f"{alpha:.2f}",
                        "beta - Lao động L":
                            f"{beta:.2f}",
                        "gamma - Số hóa D":
                            f"{gamma:.2f}",
                        "delta - AI":
                            f"{delta:.2f}",
                        "theta - Nhân lực H":
                            f"{theta:.2f}",
                        "D mục tiêu 2030":
                            f"{target_d_2030:.1f}%",
                        "AI mục tiêu 2030":
                            (
                                f"{target_ai_2030:.1f} "
                                "nghìn doanh nghiệp số"
                            ),
                        "H mục tiêu 2030":
                            f"{target_h_2030:.1f}%",
                        "Tăng trưởng K":
                            f"{k_growth_pct:.1f}%/năm",
                        "Tăng trưởng L":
                            f"{l_growth_pct:.1f}%/năm",
                        "Tăng trưởng TFP":
                            f"{tfp_growth_pct:.1f}%/năm",
                    },
                    result_summary=(
                        ai_summary.strip()
                    ),
                    policy_questions=(
                        policy_questions.strip()
                    ),
                )

                st.session_state[
                    "bai01_gemini_analysis"
                ] = analysis

            except GeminiAgentError as error:
                st.error(str(error))

    saved_analysis = st.session_state.get(
        "bai01_gemini_analysis"
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
            "⬇️ Tải phân tích Gemini",
            data=saved_analysis.encode(
                "utf-8"
            ),
            file_name=(
                "bai01_phan_tich_gemini.md"
            ),
            mime="text/markdown",
            use_container_width=True,
        )
