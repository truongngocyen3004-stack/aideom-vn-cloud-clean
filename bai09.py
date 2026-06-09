from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.bai09_model import (
    load_labor_data,
    minimum_training_threshold,
    run_full_bai09,
    solve_labor_lp,
)
from services.ai_agent import (
    GeminiAgentError,
    analyze_result,
    gemini_is_configured,
)
from ui.theme import page_header


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "bai09_labor_parameters.csv"
MODEL_VERSION = "bai09_v1"

PINK = "#D989A5"
ROSE = "#F4B8C8"
LAVENDER = "#CDB8E5"
MINT = "#A8D5D1"
YELLOW = "#F2D7A7"
BLUE = "#A9C9E8"
TEXT = "#503743"
GRID = "#EEDFE5"
BG = "#FFF9FB"


def style_plotly(fig, title, x_title="", y_title="", height=500):
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font={"family": "Arial", "color": TEXT, "size": 13},
        title_font={"size": 19, "color": TEXT},
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title_text="",
        height=height,
        margin={"l": 60, "r": 35, "t": 72, "b": 65},
    )
    fig.update_xaxes(showgrid=False, linecolor="#DCCBD3")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#DCCBD3")
    return fig


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


page_header(
    "Bài 9 — Tác động AI tới thị trường lao động Việt Nam",
    "Phân bổ ngân sách AI và đào tạo lại cho 8 ngành nhằm tối đa hóa NetJob, kiểm soát dịch chuyển lao động và bảo đảm năng lực đào tạo.",
)

st.markdown(
    """
    <div style="background:#FFF1F6;border:1px solid #F0D5DF;
    border-radius:16px;padding:18px 20px;margin-bottom:16px;color:#503743;">
    <b>NetJob:</b> việc làm mới do AI + việc làm được nâng cấp
    − việc làm bị dịch chuyển do tự động hóa.
    <br><b>Ràng buộc bảo vệ:</b> NetJobᵢ ≥ 0 và
    DisplacedJobᵢ ≤ RetrainingCapacityᵢ.
    </div>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(f"Không tìm thấy dữ liệu: {DATA_PATH}")
    st.stop()

data = load_labor_data(DATA_PATH)

with st.expander("⚙️ Thiết lập mô hình lao động", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    total_budget = c1.number_input("Ngân sách (tỷ VND)", 10000.0, 60000.0, 30000.0, 1000.0)
    max_sector_share = c2.slider("Trần ngân sách/ngành", 0.15, 0.50, 0.28, 0.01)
    min_ai_share = c3.slider("Sàn tỷ trọng AI", 0.10, 0.60, 0.25, 0.01)
    min_h_share = c4.slider("Sàn tỷ trọng H", 0.10, 0.70, 0.30, 0.01)

    c5, c6, c7 = st.columns(3)
    vulnerable_share = c5.slider("Sàn H nhóm dễ tổn thương", 0.05, 0.50, 0.20, 0.01)
    manufacturing_h = c6.number_input("Sàn H chế biến chế tạo", 0.0, 10000.0, 2000.0, 250.0)
    digital_complement = c7.slider("Bổ trợ D trong đầu tư AI", 0.00, 1.00, 0.40, 0.05)

    run_clicked = st.button(
        "🌸 Tối ưu phân bổ AI và đào tạo lại",
        type="primary",
        use_container_width=True,
    )

signature = (
    MODEL_VERSION,
    total_budget,
    max_sector_share,
    min_ai_share,
    min_h_share,
    vulnerable_share,
    manufacturing_h,
    digital_complement,
)

if (
    run_clicked
    or "bai09_result" not in st.session_state
    or st.session_state.get("bai09_signature") != signature
):
    st.session_state.pop("bai09_gemini_analysis", None)

    try:
        with st.spinner("Đang giải LP lao động và kịch bản an sinh..."):
            base = solve_labor_lp(
                data,
                total_budget=total_budget,
                digital_complement=digital_complement,
                max_sector_share=max_sector_share,
                min_ai_share=min_ai_share,
                min_h_share=min_h_share,
                min_vulnerable_h_share=vulnerable_share,
                min_manufacturing_h=manufacturing_h,
                add_5pct_cap=False,
            )
            social = solve_labor_lp(
                data,
                total_budget=total_budget,
                digital_complement=digital_complement,
                max_sector_share=max_sector_share,
                min_ai_share=min_ai_share,
                min_h_share=min_h_share,
                min_vulnerable_h_share=vulnerable_share,
                min_manufacturing_h=manufacturing_h,
                add_5pct_cap=True,
            )

            full = run_full_bai09(DATA_PATH, total_budget=total_budget)
            full["base"] = base
            full["social_cap"] = social

            st.session_state["bai09_result"] = full
            st.session_state["bai09_signature"] = signature
    except Exception as error:
        st.error(f"Không chạy được Bài 9: {error}")
        st.stop()

result = st.session_state["bai09_result"]
base = result["base"]
social = result["social_cap"]
curve = result["sensitivity"]

tabs = st.tabs([
    "9.1 — Bối cảnh",
    "9.2 — Mô hình toán học",
    "9.3 — Dữ liệu 8 ngành",
    "9.4.1 — Nghiệm tối ưu",
    "9.4.2 — Đào tạo lại",
    "9.4.3 — Độ nhạy ngân sách",
    "9.4.4 — An sinh 5%",
    "9.5 — Chính sách",
    "✨ Phân tích AI",
])

with tabs[0]:
    st.subheader("9.1 — AI vừa tạo việc làm, vừa làm dịch chuyển lao động")
    context = data[["sector", "labor_million", "risk_pct"]].copy()
    fig_context = px.scatter(
        context,
        x="labor_million",
        y="risk_pct",
        size="labor_million",
        color="risk_pct",
        text="sector",
        color_continuous_scale=["#A8D5D1", "#F2D7A7", "#D989A5"],
    )
    fig_context.update_traces(textposition="top center")
    st.plotly_chart(
        style_plotly(fig_context, "Quy mô lao động và rủi ro tự động hóa", "Lao động (triệu)", "Rủi ro (%)", 560),
        use_container_width=True,
    )
    st.info(
        "Ngành đông lao động không nhất thiết có rủi ro cao nhất, nhưng nếu tự động hóa "
        "xảy ra nhanh thì số người cần đào tạo lại có thể rất lớn."
    )

with tabs[1]:
    st.subheader("9.2 — Mô hình toán học")
    st.latex(r"NetJob_i=NewJob_i+UpgradeJob_i-DisplacedJob_i")
    st.latex(r"NewJob_i=a_{1i}x_{AI,i}+a_{2i}\lambda_Dx_{AI,i}")
    st.latex(r"UpgradeJob_i=b_{1i}x_{H,i}")
    st.latex(r"DisplacedJob_i=c_{1i}x_{AI,i}risk_i")
    st.latex(r"RetrainingCapacity_i=d_{1i}x_{H,i}")
    st.latex(r"\max \sum_i NetJob_i")
    st.latex(r"\sum_i(x_{AI,i}+x_{H,i})\le 30000")
    st.latex(r"NetJob_i\ge0,\quad DisplacedJob_i\le RetrainingCapacity_i")

with tabs[2]:
    st.subheader("9.3 — Tham số 8 ngành")
    st.dataframe(data.round(4), use_container_width=True, hide_index=True)

    heat_columns = [
        "a1_new_ai_job_per_billion",
        "a2_new_digital_job_per_billion",
        "b1_upgrade_job_per_billion",
        "c1_displace_job_per_billion",
        "d1_retrain_capacity_per_billion",
    ]
    heat = px.imshow(
        data.set_index("sector")[heat_columns],
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale=["#FFF7FA", "#EEDCF5", "#7DBFB4"],
    )
    heat.update_layout(height=560, paper_bgcolor=BG, font_color=TEXT)
    st.plotly_chart(heat, use_container_width=True)

with tabs[3]:
    st.subheader("9.4.1 — Phân bổ tối ưu và NetJob")
    if not base["success"]:
        st.error(base["status"])
    else:
        summary = base["summary"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tổng NetJob", f"{summary['objective_total_netjob']:,.0f}")
        k2.metric("Đầu tư AI", f"{summary['total_x_AI']:,.0f}")
        k3.metric("Đầu tư H", f"{summary['total_x_H']:,.0f}")
        k4.metric("Áp lực tự động hóa max", f"{summary['max_pressure']:.3f}")

        result_df = base["result_df"]
        alloc_long = result_df.melt(
            id_vars="sector",
            value_vars=["x_AI", "x_H"],
            var_name="Hạng mục",
            value_name="Ngân sách",
        )
        fig_alloc = px.bar(
            alloc_long,
            x="sector",
            y="Ngân sách",
            color="Hạng mục",
            barmode="group",
            color_discrete_sequence=[PINK, MINT],
        )
        fig_alloc.update_xaxes(tickangle=-25)
        st.plotly_chart(
            style_plotly(fig_alloc, "Phân bổ AI và đào tạo lại theo ngành", "Ngành", "Tỷ VND", 540),
            use_container_width=True,
        )

        jobs_long = result_df.melt(
            id_vars="sector",
            value_vars=["NewJob_AI", "UpgradeJob", "DisplacedJob", "NetJob"],
            var_name="Thành phần",
            value_name="Việc làm",
        )
        fig_jobs = px.bar(
            jobs_long,
            x="sector",
            y="Việc làm",
            color="Thành phần",
            barmode="group",
            color_discrete_sequence=[PINK, MINT, ROSE, LAVENDER],
        )
        fig_jobs.update_xaxes(tickangle=-25)
        st.plotly_chart(
            style_plotly(fig_jobs, "Cấu phần việc làm theo ngành", "Ngành", "Số việc làm mô phỏng", 560),
            use_container_width=True,
        )

        st.dataframe(result_df.round(4), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Tải kết quả Bài 9",
            csv_bytes(result_df),
            "bai09_ket_qua.csv",
            "text/csv",
        )

with tabs[4]:
    st.subheader("9.4.2 — Ngưỡng đào tạo lại tối thiểu")
    sector_name = st.selectbox(
        "Chọn ngành",
        data["sector"].tolist(),
        index=1,
        key="bai09_threshold_sector",
    )
    sector_id = int(data.loc[data["sector"] == sector_name, "sector_id"].iloc[0])
    ai_test = st.number_input(
        "Đầu tư AI giả định cho ngành (tỷ VND)",
        0.0,
        15000.0,
        4000.0,
        250.0,
        key="bai09_ai_test",
    )

    threshold = minimum_training_threshold(
        data,
        sector_id=sector_id,
        ai_investment=ai_test,
        digital_complement=digital_complement,
    )

    t1, t2, t3 = st.columns(3)
    t1.metric("H để NetJob ≥ 0", f"{threshold['min_h_for_netjob']:,.2f}")
    t2.metric("H để đủ capacity", f"{threshold['min_h_for_capacity']:,.2f}")
    t3.metric("Ngưỡng H bắt buộc", f"{threshold['required_h']:,.2f}")

    threshold_df = pd.DataFrame({
        "Điều kiện": ["NetJob không âm", "Đủ năng lực đào tạo", "Ngưỡng chính sách"],
        "Đầu tư H tối thiểu": [
            threshold["min_h_for_netjob"],
            threshold["min_h_for_capacity"],
            threshold["required_h"],
        ],
    })
    fig_threshold = px.bar(
        threshold_df,
        x="Điều kiện",
        y="Đầu tư H tối thiểu",
        color="Điều kiện",
        text_auto=".2f",
        color_discrete_sequence=[PINK, MINT, LAVENDER],
    )
    fig_threshold.update_layout(showlegend=False)
    st.plotly_chart(
        style_plotly(fig_threshold, f"Ngưỡng đào tạo lại — {sector_name}", "Điều kiện", "Tỷ VND"),
        use_container_width=True,
    )

with tabs[5]:
    st.subheader("9.4.3 — Phân tích nhạy cảm ngân sách")
    st.dataframe(curve.round(3), use_container_width=True, hide_index=True)

    fig_curve = px.line(
        curve,
        x="budget",
        y="total_netjob",
        markers=True,
        color_discrete_sequence=[PINK],
    )
    st.plotly_chart(
        style_plotly(fig_curve, "Đường phản ứng NetJob theo ngân sách", "Ngân sách", "Tổng NetJob"),
        use_container_width=True,
    )

    curve_alloc = curve.melt(
        id_vars="budget",
        value_vars=["x_AI", "x_H"],
        var_name="Hạng mục",
        value_name="Ngân sách phân bổ",
    )
    fig_curve_alloc = px.line(
        curve_alloc,
        x="budget",
        y="Ngân sách phân bổ",
        color="Hạng mục",
        markers=True,
        color_discrete_sequence=[PINK, MINT],
    )
    st.plotly_chart(
        style_plotly(fig_curve_alloc, "Cơ cấu AI–H theo quy mô ngân sách", "Ngân sách tổng", "Phân bổ"),
        use_container_width=True,
    )

with tabs[6]:
    st.subheader("9.4.4 — Trần an sinh: displaced không quá 5% lao động ngành")
    compare = result["comparison"].copy()
    compare.iloc[0] = {
        "Kịch bản": "Không trần 5%",
        "Trạng thái": base["status"],
        "Tổng NetJob": base.get("objective", np.nan),
        "Tổng displaced": base["summary"]["total_displaced"] if base.get("success") else np.nan,
        "x_AI": base["summary"]["total_x_AI"] if base.get("success") else np.nan,
        "x_H": base["summary"]["total_x_H"] if base.get("success") else np.nan,
    }
    compare.iloc[1] = {
        "Kịch bản": "Có trần 5%",
        "Trạng thái": social["status"],
        "Tổng NetJob": social.get("objective", np.nan),
        "Tổng displaced": social["summary"]["total_displaced"] if social.get("success") else np.nan,
        "x_AI": social["summary"]["total_x_AI"] if social.get("success") else np.nan,
        "x_H": social["summary"]["total_x_H"] if social.get("success") else np.nan,
    }

    st.dataframe(compare.round(3), use_container_width=True, hide_index=True)

    fig_social = px.bar(
        compare,
        x="Kịch bản",
        y="Tổng NetJob",
        color="Kịch bản",
        text_auto=",.0f",
        color_discrete_sequence=[PINK, MINT],
    )
    fig_social.update_layout(showlegend=False)
    st.plotly_chart(
        style_plotly(fig_social, "Chi phí của hàng rào an sinh 5%", "Kịch bản", "Tổng NetJob"),
        use_container_width=True,
    )

    if social["success"]:
        st.success("Mô hình vẫn khả thi khi thêm trần an sinh 5%.")
    else:
        st.warning(
            "Kịch bản 5% không khả thi với tham số hiện tại; cần tăng H, giảm AI hoặc nới trần."
        )

with tabs[7]:
    st.subheader("9.5 — Thảo luận chính sách")
    if base["success"]:
        df = base["result_df"]
        top_h = df.sort_values("x_H", ascending=False).iloc[0]
        top_ai = df.sort_values("x_AI", ascending=False).iloc[0]
        max_pressure = df.sort_values("AutomationPressure", ascending=False).iloc[0]

        st.markdown(
            f"""
            **a) Ngành cần đào tạo lại nhiều nhất:** **{top_h['sector']}**,
            nhận khoảng **{top_h['x_H']:,.0f} tỷ VND**. Kết quả phản ánh quy mô
            lao động, rủi ro tự động hóa và hiệu quả đào tạo.

            **b) Ngành nhận đầu tư AI lớn nhất:** **{top_ai['sector']}**,
            khoảng **{top_ai['x_AI']:,.0f} tỷ VND**. Đầu tư AI cần đi kèm sàn H
            để tránh tự động hóa nhanh hơn khả năng chuyển đổi lao động.

            **c) Ngành có áp lực tự động hóa cao nhất:** **{max_pressure['sector']}**,
            với tỷ số displaced/retraining capacity là
            **{max_pressure['AutomationPressure']:.3f}**.

            **d) Chính sách:** ưu tiên đào tạo quy mô lớn cho chế biến chế tạo,
            bán lẻ, logistics và nông nghiệp; đồng thời phát triển kỹ năng AI,
            dữ liệu và vận hành hệ thống cho nhóm ngành tạo việc làm mới.
            """
        )

with tabs[8]:
    st.subheader("Tác nhân AI — Bài 9")
    configured = gemini_is_configured()
    if not configured:
        st.info("Phần AI sẽ được chuẩn hóa đồng loạt sau khi hoàn thành Bài 12.")

    if base["success"]:
        summary = f"""
Bài 9:
- Tổng NetJob: {base['summary']['objective_total_netjob']:.4f}
- x_AI: {base['summary']['total_x_AI']:.4f}
- x_H: {base['summary']['total_x_H']:.4f}
- Displaced: {base['summary']['total_displaced']:.4f}
- Max pressure: {base['summary']['max_pressure']:.6f}
{base['result_df'].round(4).to_string(index=False)}
"""
    else:
        summary = f"Bài 9 không tối ưu: {base['status']}"

    if st.button(
        "✨ Phân tích kết quả bằng Gemini",
        disabled=not configured,
        use_container_width=True,
        key="gemini_bai09",
    ):
        try:
            with st.spinner("Gemini đang phân tích Bài 9..."):
                st.session_state["bai09_gemini_analysis"] = analyze_result(
                    exercise_name="Bài 9 — AI và thị trường lao động",
                    model_name="LP phân bổ AI và đào tạo lại theo 8 ngành",
                    parameters={"budget": total_budget, "social_cap": "5%"},
                    result_summary=summary,
                    policy_questions="Ngành cần đào tạo lại, ngưỡng H, an sinh xã hội và chiến lược việc làm.",
                )
        except GeminiAgentError as error:
            st.error(str(error))

    if st.session_state.get("bai09_gemini_analysis"):
        st.markdown(st.session_state["bai09_gemini_analysis"])
