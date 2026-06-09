from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.bai12_model import (
    ITEMS,
    REGIONS,
    SCENARIOS,
    IntegratedConfig,
    run_full_bai12,
)
from services.ai_agent import (
    GeminiAgentError,
    analyze_result,
    gemini_is_configured,
)
from ui.theme import page_header


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "bai12_v1"

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
    MINT,
    LAVENDER,
    YELLOW,
    BLUE,
    ROSE,
    "#D6B5A5",
    "#BFD7B5",
]


def style_plotly(
    fig: go.Figure,
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 500,
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


page_header(
    "Bài 12 — AIDEOM-VN tích hợp 11 mô hình ra quyết định",
    "Dashboard điều hành chính sách số kết nối dự báo tăng trưởng, sẵn sàng vùng, tối ưu phân bổ, lao động, rủi ro và so sánh năm kịch bản đến năm 2030.",
)

st.markdown(
    """
    <div style="
        background:linear-gradient(135deg,#F9C9D8 0%,#E8D7F2 52%,#D8EDF1 100%);
        border:1px solid #EDC6D3;
        border-radius:22px;
        padding:24px 26px;
        margin-bottom:18px;
        color:#503743;
        box-shadow:0 8px 24px rgba(93,55,73,0.08);
    ">
        <div style="font-size:28px;font-weight:800;margin-bottom:8px;">
            VN AIDEOM-VN — Phòng điều hành chính sách số
        </div>
        <div style="font-size:15px;line-height:1.6;">
            Bài 12 không thay thế 11 bài trước mà tổng hợp các chỉ báo cốt lõi
            thành một hệ thống hỗ trợ quyết định: chọn kịch bản, phân bổ ngân sách,
            dự báo GDP, đánh giá việc làm, cảnh báo rủi ro và chuẩn bị bàn giao.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Tham số Bài 12")

    total_budget = st.slider(
        "Tổng ngân sách 2026–2030",
        min_value=120000.0,
        max_value=400000.0,
        value=250000.0,
        step=10000.0,
        format="%.0f tỷ VND",
    )

    scenario_options = {
        f"{code} — {values['name']}":
            code
        for code, values
        in SCENARIOS.items()
    }

    scenario_label = st.selectbox(
        "Kịch bản đang phân tích",
        options=list(
            scenario_options.keys()
        ),
        index=4,
    )

    scenario_code = (
        scenario_options[
            scenario_label
        ]
    )

    st.markdown("---")
    st.caption(
        "Ngưỡng cảnh báo rủi ro"
    )

    cyber_threshold = st.slider(
        "Cyber",
        20.0,
        90.0,
        60.0,
        5.0,
    )

    emission_threshold = st.slider(
        "Phát thải",
        20.0,
        90.0,
        60.0,
        5.0,
    )

    dependency_threshold = st.slider(
        "Phụ thuộc công nghệ",
        20.0,
        90.0,
        60.0,
        5.0,
    )

    macro_threshold = st.slider(
        "Vĩ mô",
        20.0,
        90.0,
        60.0,
        5.0,
    )

    run_clicked = st.button(
        "🌸 Cập nhật dashboard",
        type="primary",
        use_container_width=True,
    )

signature = (
    MODEL_VERSION,
    total_budget,
    scenario_code,
    cyber_threshold,
    emission_threshold,
    dependency_threshold,
    macro_threshold,
)

if (
    run_clicked
    or "bai12_result"
    not in st.session_state
    or st.session_state.get(
        "bai12_signature"
    )
    != signature
):
    st.session_state.pop(
        "bai12_gemini_analysis",
        None,
    )

    config = IntegratedConfig(
        total_budget=float(
            total_budget
        ),
        scenario_code=(
            scenario_code
        ),
        cyber_threshold=float(
            cyber_threshold
        ),
        emission_threshold=float(
            emission_threshold
        ),
        dependency_threshold=float(
            dependency_threshold
        ),
        macro_threshold=float(
            macro_threshold
        ),
    )

    with st.spinner(
        "Đang tích hợp dự báo, phân bổ, lao động, readiness và rủi ro..."
    ):
        st.session_state[
            "bai12_result"
        ] = run_full_bai12(
            config=config,
            project_root=PROJECT_ROOT,
        )

        st.session_state[
            "bai12_signature"
        ] = signature

result = st.session_state[
    "bai12_result"
]

forecast = result[
    "forecast"
]
allocation = result[
    "allocation"
]
region_total = result[
    "region_total"
]
readiness = result[
    "readiness"
]
labor = result[
    "labor"
]
risk = result[
    "risk"
]
scenarios = result[
    "scenarios"
]
scenario = result[
    "scenario"
]
scorecard = result[
    "scorecard"
]

scenario_name = (
    SCENARIOS[
        scenario_code
    ][
        "name"
    ]
)

warning_count = int(
    (
        risk[
            "status"
        ]
        == "Cảnh báo"
    ).sum()
)

st.info(
    f"Đang xem **{scenario_code} — {scenario_name}**. "
    "Cơ cấu ngân sách được xác định từ mục tiêu tăng trưởng, "
    "bao trùm, môi trường và mức ác cảm rủi ro của kịch bản."
)

tabs = st.tabs([
    "🏠 Tổng quan",
    "💰 Phân bổ",
    "📊 Kịch bản",
    "🧭 Sẵn sàng vùng",
    "👥 Lao động",
    "⚠️ Rủi ro",
    "🧩 Tích hợp 11 bài",
    "📦 Bàn giao",
    "✨ Phân tích AI",
])

with tabs[0]:
    st.subheader(
        "Tổng quan điều hành 2025–2030"
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "GDP 2030",
        (
            f"{forecast['gdp_thousand_billion_vnd'].iloc[-1]:,.1f} "
            "nghìn tỷ VND"
        ),
        (
            f"CAGR "
            f"{scenario['cagr_2025_2030_pct']:.2f}%"
        ),
    )

    k2.metric(
        "NetJob",
        (
            f"{labor['net_jobs'].iloc[0]:,.0f} "
            "việc làm mô phỏng"
        ),
        (
            f"Đào tạo phủ "
            f"{labor['training_coverage_pct'].iloc[0]:.1f}%"
        ),
    )

    k3.metric(
        "Readiness bình quân",
        (
            f"{readiness['composite_readiness'].mean():.1f}/100"
        ),
        (
            f"Top 1: "
            f"{readiness.iloc[0]['region']}"
        ),
    )

    k4.metric(
        "Rủi ro tổng hợp",
        (
            f"{risk['score'].mean():.1f}/100"
        ),
        (
            f"{warning_count} cảnh báo"
        ),
        delta_color=(
            "inverse"
        ),
    )

    left, right = st.columns(
        [1.35, 1.0]
    )

    with left:
        fig_gdp = px.line(
            forecast,
            x="year",
            y=(
                "gdp_thousand_billion_vnd"
            ),
            markers=True,
            color_discrete_sequence=[
                PINK
            ],
        )

        fig_gdp.update_traces(
            line={
                "width": 4,
            },
            marker={
                "size": 9,
            },
        )

        st.plotly_chart(
            style_plotly(
                fig_gdp,
                "Quỹ đạo GDP 2025–2030",
                "Năm",
                "GDP — nghìn tỷ VND",
                height=470,
            ),
            use_container_width=True,
        )

    with right:
        item_total = (
            allocation.groupby(
                "item",
                as_index=False,
            )["allocation"]
            .sum()
        )

        fig_structure = px.pie(
            item_total,
            names="item",
            values="allocation",
            hole=0.48,
            color_discrete_sequence=(
                PASTEL_SEQUENCE
            ),
        )

        fig_structure.update_layout(
            title=(
                "Cơ cấu chính sách"
            ),
            height=470,
            paper_bgcolor=BG,
            font_color=TEXT,
            legend_title_text="",
        )

        st.plotly_chart(
            fig_structure,
            use_container_width=True,
        )

    scenario_rank = int(
        scenario[
            "balanced_rank"
        ]
    )

    if scenario_rank == 1:
        st.success(
            f"{scenario_code} đang đứng hạng 1 theo điểm cân bằng tổng hợp."
        )
    else:
        best = scenarios.iloc[0]
        st.warning(
            f"{scenario_code} đang xếp hạng {scenario_rank}. "
            f"Kịch bản cân bằng tốt nhất hiện là "
            f"{best['scenario_code']} — {best['scenario_name']}."
        )

with tabs[1]:
    st.subheader(
        "Phân bổ ngân sách theo vùng và hạng mục"
    )

    allocation_matrix = (
        allocation.pivot(
            index="region",
            columns="item",
            values="allocation",
        )
        .reindex(
            index=REGIONS,
            columns=ITEMS,
        )
    )

    heat = go.Figure(
        data=go.Heatmap(
            z=allocation_matrix.values,
            x=allocation_matrix.columns,
            y=allocation_matrix.index,
            colorscale=[
                [0.00, "#FFF7FA"],
                [0.30, "#F5DAE5"],
                [0.60, "#E4D8F0"],
                [1.00, "#85C8BF"],
            ],
            text=allocation_matrix.values,
            texttemplate="%{text:,.0f}",
            colorbar={
                "title": "Tỷ VND",
            },
            hovertemplate=(
                "Vùng=%{y}<br>"
                "Hạng mục=%{x}<br>"
                "Ngân sách=%{z:,.1f}"
                "<extra></extra>"
            ),
        )
    )

    st.plotly_chart(
        style_plotly(
            heat,
            "Ma trận phân bổ vùng × hạng mục",
            "Hạng mục",
            "Vùng",
            height=600,
        ),
        use_container_width=True,
    )

    allocation_long = (
        allocation.copy()
    )

    fig_region = px.bar(
        allocation_long,
        x="region",
        y="allocation",
        color="item",
        barmode="stack",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_region.update_xaxes(
        tickangle=-20
    )

    st.plotly_chart(
        style_plotly(
            fig_region,
            "Tổng vốn theo vùng và cơ cấu hạng mục",
            "Vùng",
            "Tỷ VND",
            height=540,
        ),
        use_container_width=True,
    )

    st.dataframe(
        allocation_matrix.round(2),
        use_container_width=True,
    )

    d1, d2 = st.columns(2)

    d1.download_button(
        "⬇️ Tải phân bổ dạng dài",
        data=csv_bytes(
            allocation
        ),
        file_name=(
            "bai12_phan_bo_dang_dai.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    d2.download_button(
        "⬇️ Tải ma trận phân bổ",
        data=allocation_matrix.to_csv(
            encoding="utf-8-sig"
        ).encode(
            "utf-8-sig"
        ),
        file_name=(
            "bai12_ma_tran_phan_bo.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

with tabs[2]:
    st.subheader(
        "So sánh năm kịch bản chính sách"
    )

    st.dataframe(
        scenarios.round(4),
        use_container_width=True,
        hide_index=True,
    )

    scenario_long = (
        scenarios.melt(
            id_vars=[
                "scenario_code",
                "scenario_name",
            ],
            value_vars=[
                "gdp_2030",
                "net_jobs",
                "mean_readiness",
                "risk_mean",
            ],
            var_name="Chỉ tiêu",
            value_name="Giá trị",
        )
    )

    fig_scenario = px.bar(
        scenario_long,
        x="scenario_code",
        y="Giá trị",
        color="Chỉ tiêu",
        facet_col="Chỉ tiêu",
        facet_col_wrap=2,
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_scenario.update_yaxes(
        matches=None,
    )

    fig_scenario.for_each_annotation(
        lambda annotation: annotation.update(
            text=annotation.text.split(
                "="
            )[-1]
        )
    )

    st.plotly_chart(
        style_plotly(
            fig_scenario,
            "GDP, việc làm, readiness và rủi ro theo kịch bản",
            "Kịch bản",
            "Giá trị",
            height=760,
        ),
        use_container_width=True,
    )

    radar_columns = [
        "gdp_2030",
        "net_jobs",
        "mean_readiness",
        "risk_mean",
    ]

    radar = scenarios[
        [
            "scenario_code",
            *radar_columns,
        ]
    ].copy()

    for column in radar_columns:
        low = float(
            radar[column].min()
        )
        high = float(
            radar[column].max()
        )

        if high == low:
            radar[
                f"{column}_norm"
            ] = 1.0
        elif column == "risk_mean":
            radar[
                f"{column}_norm"
            ] = (
                high
                - radar[column]
            ) / (
                high
                - low
            )
        else:
            radar[
                f"{column}_norm"
            ] = (
                radar[column]
                - low
            ) / (
                high
                - low
            )

    radar_fig = go.Figure()

    radar_labels = [
        "GDP 2030",
        "NetJob",
        "Readiness",
        "Rủi ro thấp",
    ]

    for position, row in radar.iterrows():
        values = [
            row[
                f"{column}_norm"
            ]
            for column
            in radar_columns
        ]

        radar_fig.add_trace(
            go.Scatterpolar(
                r=values + [
                    values[0]
                ],
                theta=radar_labels
                + [
                    radar_labels[0]
                ],
                fill="toself",
                name=row[
                    "scenario_code"
                ],
                opacity=0.55,
                line={
                    "color":
                        PASTEL_SEQUENCE[
                            position
                            % len(
                                PASTEL_SEQUENCE
                            )
                        ]
                },
            )
        )

    radar_fig.update_layout(
        title=(
            "Hồ sơ cân bằng của năm kịch bản"
        ),
        polar={
            "radialaxis": {
                "visible": True,
                "range": [
                    0,
                    1,
                ],
            }
        },
        paper_bgcolor=BG,
        font_color=TEXT,
        height=600,
    )

    st.plotly_chart(
        radar_fig,
        use_container_width=True,
    )

with tabs[3]:
    st.subheader(
        "Chỉ số sẵn sàng và ưu tiên vùng"
    )

    st.dataframe(
        readiness[
            [
                "rank",
                "region",
                "region_budget",
                "digital_readiness",
                "inclusive_readiness",
                "green_readiness",
                "composite_readiness",
            ]
        ].round(3),
        use_container_width=True,
        hide_index=True,
    )

    readiness_long = readiness.melt(
        id_vars=[
            "region",
            "rank",
        ],
        value_vars=[
            "digital_readiness",
            "inclusive_readiness",
            "green_readiness",
            "composite_readiness",
        ],
        var_name="Chỉ số",
        value_name="Điểm",
    )

    fig_readiness = px.bar(
        readiness_long,
        x="region",
        y="Điểm",
        color="Chỉ số",
        barmode="group",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_readiness.update_xaxes(
        tickangle=-20
    )

    st.plotly_chart(
        style_plotly(
            fig_readiness,
            "Sẵn sàng số, bao trùm, xanh và tổng hợp",
            "Vùng",
            "Điểm",
            height=570,
        ),
        use_container_width=True,
    )

    fig_priority = px.scatter(
        readiness,
        x="composite_readiness",
        y="region_budget",
        size="grdp_per_capita",
        color="emission_intensity",
        text="region",
        color_continuous_scale=[
            "#A8D5D1",
            "#F2D7A7",
            "#D989A5",
        ],
        hover_data=[
            "digital_index",
            "ai_readiness",
            "trained_labor_pct",
            "gini",
        ],
    )

    fig_priority.update_traces(
        textposition="top center"
    )

    st.plotly_chart(
        style_plotly(
            fig_priority,
            "Readiness và mức ngân sách được phân bổ",
            "Điểm readiness tổng hợp",
            "Ngân sách vùng",
            height=560,
        ),
        use_container_width=True,
    )

with tabs[4]:
    st.subheader(
        "Thị trường lao động và năng lực đào tạo lại"
    )

    l1, l2, l3, l4 = st.columns(4)

    l1.metric(
        "Việc làm mới",
        f"{labor['new_ai_jobs'].iloc[0]:,.0f}",
    )
    l2.metric(
        "Việc làm nâng cấp",
        f"{labor['upgraded_jobs'].iloc[0]:,.0f}",
    )
    l3.metric(
        "Việc làm dịch chuyển",
        f"{labor['displaced_jobs'].iloc[0]:,.0f}",
    )
    l4.metric(
        "NetJob",
        f"{labor['net_jobs'].iloc[0]:,.0f}",
    )

    labor_chart = pd.DataFrame({
        "Chỉ tiêu": [
            "Việc làm mới",
            "Nâng cấp kỹ năng",
            "Dịch chuyển",
            "Năng lực đào tạo",
            "NetJob",
        ],
        "Giá trị": [
            labor[
                "new_ai_jobs"
            ].iloc[0],
            labor[
                "upgraded_jobs"
            ].iloc[0],
            labor[
                "displaced_jobs"
            ].iloc[0],
            labor[
                "training_capacity"
            ].iloc[0],
            labor[
                "net_jobs"
            ].iloc[0],
        ],
    })

    fig_labor = px.bar(
        labor_chart,
        x="Chỉ tiêu",
        y="Giá trị",
        color="Chỉ tiêu",
        text_auto=",.0f",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_labor.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        style_plotly(
            fig_labor,
            "Tác động lao động của cơ cấu chính sách",
            "Chỉ tiêu",
            "Việc làm mô phỏng",
            height=500,
        ),
        use_container_width=True,
    )

    coverage = float(
        labor[
            "training_coverage_pct"
        ].iloc[0]
    )

    if coverage >= 100:
        st.success(
            f"Năng lực đào tạo lại bao phủ {coverage:.1f}% số việc làm bị dịch chuyển."
        )
    else:
        st.warning(
            f"Năng lực đào tạo mới bao phủ {coverage:.1f}%. "
            "Cần tăng tỷ trọng Nhân lực số hoặc giảm tốc độ AI hóa."
        )

with tabs[5]:
    st.subheader(
        "Cảnh báo rủi ro theo ngưỡng điều hành"
    )

    st.dataframe(
        risk.round(3),
        use_container_width=True,
        hide_index=True,
    )

    fig_risk = px.bar(
        risk,
        x="risk_type",
        y="score",
        color="status",
        text="score",
        color_discrete_map={
            "Trong ngưỡng":
                MINT,
            "Cảnh báo":
                ROSE,
        },
    )

    fig_risk.add_scatter(
        x=risk[
            "risk_type"
        ],
        y=risk[
            "threshold"
        ],
        mode="lines+markers",
        name="Ngưỡng",
        line={
            "color": TEXT,
            "dash": "dash",
        },
    )

    fig_risk.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        selector={
            "type": "bar",
        },
    )

    st.plotly_chart(
        style_plotly(
            fig_risk,
            "Rủi ro và ngưỡng cảnh báo",
            "Loại rủi ro",
            "Điểm rủi ro",
            height=520,
        ),
        use_container_width=True,
    )

    for _, row in risk.iterrows():
        if row["status"] == "Cảnh báo":
            st.error(
                f"{row['risk_type']}: "
                f"{row['score']:.1f} vượt ngưỡng "
                f"{row['threshold']:.1f}."
            )

    st.markdown(
        """
        **Nguyên tắc phản ứng:**

        - Cyber cao: tăng SOC, chuẩn bảo mật, đầu tư H và giảm tốc độ triển khai AI thiếu kiểm soát.
        - Phát thải cao: gắn trung tâm dữ liệu với điện sạch và tiêu chuẩn hiệu suất năng lượng.
        - Phụ thuộc công nghệ cao: tăng R&D nội địa, dữ liệu mở, đào tạo và đa dạng nhà cung cấp.
        - Vĩ mô cao: chia giai đoạn đầu tư, giữ ngân sách dự phòng và dùng stochastic/robust planning.
        """
    )

with tabs[6]:
    st.subheader(
        "Bảng điều khiển tích hợp kết quả Bài 1–11"
    )

    st.markdown(
        "#### Scorecard chính sách"
    )

    st.dataframe(
        scorecard.round(4),
        use_container_width=True,
        hide_index=True,
    )

    score_plot = scorecard.copy()

    fig_score = px.bar(
        score_plot,
        x="Bài",
        y="Chỉ số tích hợp",
        color="Mô hình",
        text_auto=".2f",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_score.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        style_plotly(
            fig_score,
            "Chỉ số đại diện được bàn giao từ 11 mô hình",
            "Bài",
            "Giá trị chỉ số",
            height=500,
        ),
        use_container_width=True,
    )

    st.markdown(
        "#### Kiểm tra file và khả năng import"
    )

    audit = result[
        "integration_audit"
    ]

    st.dataframe(
        audit,
        use_container_width=True,
        hide_index=True,
    )

    missing = audit[
        (
            audit[
                "Core file"
            ]
            == "Thiếu"
        )
        | (
            audit[
                "Page file"
            ]
            == "Thiếu"
        )
    ]

    if missing.empty:
        st.success(
            "Đã phát hiện đủ core và page của Bài 1–11."
        )
    else:
        st.warning(
            f"Còn {len(missing)} bài thiếu core hoặc page theo quy ước tên file baiXX."
        )

    st.caption(
        "Một số bài cũ có thể dùng tên module khác hoặc chỉ nằm trong app.py; "
        "bảng audit giúp kiểm tra kỹ thuật trước khi đóng gói, không phủ nhận kết quả đã có."
    )

with tabs[7]:
    st.subheader(
        "Bàn giao, kiểm thử và xuất dữ liệu"
    )

    handoff_json = json.dumps(
        result[
            "handoff"
        ],
        ensure_ascii=False,
        indent=2,
    )

    st.markdown(
        """
        **Bộ bàn giao cuối kỳ nên có:**

        1. `app.py`, thư mục `pages`, `core`, `services`, `ui`, `data`.
        2. `requirements.txt` và `.streamlit/secrets.toml.example`.
        3. Bản Word/PDF mô tả chức năng, cách sử dụng và giải thích kết quả.
        4. Ảnh chụp 12 menu và kết quả tác nhân AI.
        5. File ZIP/RAR theo đúng tên `MSV_Họ và tên`.
        """
    )

    st.code(
        """
python -m compileall -q core pages services ui
python -m pytest -q
python -m streamlit run app.py
""".strip(),
        language="bash",
    )

    d1, d2, d3 = st.columns(3)

    d1.download_button(
        "⬇️ Tải forecast GDP",
        data=csv_bytes(
            forecast
        ),
        file_name=(
            "bai12_forecast_gdp.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    d2.download_button(
        "⬇️ Tải bảng kịch bản",
        data=csv_bytes(
            scenarios
        ),
        file_name=(
            "bai12_so_sanh_kich_ban.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    d3.download_button(
        "⬇️ Tải gói handoff JSON",
        data=handoff_json.encode(
            "utf-8"
        ),
        file_name=(
            "bai12_handoff.json"
        ),
        mime=(
            "application/json"
        ),
        use_container_width=True,
    )

    with st.expander(
        "Xem cấu trúc handoff JSON"
    ):
        st.code(
            handoff_json,
            language="json",
        )

with tabs[8]:
    st.subheader(
        "Tác nhân AI tổng hợp Bài 12"
    )

    configured = (
        gemini_is_configured()
    )

    if configured:
        st.success(
            "Gemini API đã được cấu hình."
        )
    else:
        st.info(
            "Phần AI của cả 12 bài sẽ được chuẩn hóa đồng loạt ở bước cuối."
        )

    result_summary = f"""
BÀI 12 — DASHBOARD AIDEOM-VN TÍCH HỢP

Kịch bản:
- Code: {scenario_code}
- Tên: {scenario_name}
- Hạng cân bằng: {scenario['balanced_rank']}
- Điểm cân bằng: {scenario['balanced_score']:.6f}

KPI:
- GDP 2030: {forecast['gdp_thousand_billion_vnd'].iloc[-1]:.6f}
- CAGR GDP: {scenario['cagr_2025_2030_pct']:.6f}%
- NetJob: {labor['net_jobs'].iloc[0]:.6f}
- Training coverage: {labor['training_coverage_pct'].iloc[0]:.6f}%
- Readiness bình quân: {readiness['composite_readiness'].mean():.6f}
- Rủi ro bình quân: {risk['score'].mean():.6f}
- Số cảnh báo: {warning_count}

Rủi ro:
{risk.round(4).to_string(index=False)}

Top vùng:
{readiness[['rank','region','composite_readiness','region_budget']].head(6).round(4).to_string(index=False)}

So sánh kịch bản:
{scenarios.round(4).to_string(index=False)}
"""

    policy_questions = """
1. Kịch bản đang chọn có cân bằng tốt giữa GDP, việc làm, readiness và rủi ro không?
2. Vùng nào nên được ưu tiên và vùng nào cần cơ chế bù đắp?
3. Cơ cấu Nhân lực số–Số hóa–AI–Vốn vật chất có hợp lý không?
4. Rủi ro nào cần hành động ngay và biện pháp giảm thiểu là gì?
5. Dashboard Bài 12 tổng hợp các bài trước như thế nào và còn hạn chế dữ liệu gì?
6. Đề xuất quyết định chính sách 2026–2030 theo ba giai đoạn.
"""

    with st.expander(
        "Xem nội dung sẽ gửi cho Gemini"
    ):
        st.text_area(
            "Tóm tắt tích hợp",
            value=result_summary.strip(),
            height=450,
            disabled=True,
        )

    analyze_clicked = st.button(
        "✨ Phân tích dashboard bằng Gemini",
        disabled=not configured,
        use_container_width=True,
        key="gemini_bai12",
    )

    if analyze_clicked:
        try:
            with st.spinner(
                "Gemini đang tổng hợp toàn bộ dashboard..."
            ):
                st.session_state[
                    "bai12_gemini_analysis"
                ] = analyze_result(
                    exercise_name=(
                        "Bài 12 — AIDEOM-VN "
                        "tích hợp 11 mô hình"
                    ),
                    model_name=(
                        "Integrated policy dashboard: "
                        "forecasting, allocation, readiness, "
                        "labor, risk and scenario comparison"
                    ),
                    parameters={
                        "scenario_code":
                            scenario_code,
                        "total_budget":
                            total_budget,
                        "risk_thresholds": {
                            "cyber":
                                cyber_threshold,
                            "emission":
                                emission_threshold,
                            "dependency":
                                dependency_threshold,
                            "macro":
                                macro_threshold,
                        },
                    },
                    result_summary=(
                        result_summary.strip()
                    ),
                    policy_questions=(
                        policy_questions.strip()
                    ),
                )
        except GeminiAgentError as error:
            st.error(str(error))

    saved_analysis = (
        st.session_state.get(
            "bai12_gemini_analysis"
        )
    )

    if saved_analysis:
        st.markdown(
            saved_analysis
        )

        st.download_button(
            "⬇️ Tải phân tích Gemini Bài 12",
            data=saved_analysis.encode(
                "utf-8"
            ),
            file_name=(
                "bai12_phan_tich_gemini.md"
            ),
            mime="text/markdown",
            use_container_width=True,
        )
