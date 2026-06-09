from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core.bai08_model import (
    DynamicConfig,
    run_full_bai08,
)
from services.ai_agent import (
    GeminiAgentError,
    analyze_result,
    gemini_is_configured,
)
from ui.theme import page_header


MODEL_VERSION = "bai08_v1"
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
    "Bài 8 — Tối ưu động phân bổ liên thời gian 2026–2035",
    "Tìm quỹ đạo tiêu dùng và đầu tư K, D, AI, H tối đa hóa phúc lợi chiết khấu, đồng thời phân tích đầu tư sớm/muộn và tầm nhìn dài hạn.",
)

st.markdown(
    """
    <div style="background:#FFF1F6;border:1px solid #F0D5DF;
    border-radius:16px;padding:18px 20px;margin-bottom:16px;color:#503743;">
    <b>Trạng thái động:</b> K, D, AI, H và TFP A thay đổi qua 10 năm.
    <br><b>Mục tiêu:</b> tối đa hóa Σρᵗlog(Cₜ) với ràng buộc nguồn lực
    và sàn đầu tư số, AI, nhân lực.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("⚙️ Thiết lập tối ưu động", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    rho = c1.slider("Hệ số chiết khấu ρ", 0.85, 0.99, 0.97, 0.01)
    short_rho = c2.slider("Kịch bản ngắn hạn", 0.80, 0.95, 0.90, 0.01)
    maxiter = c3.slider("Số vòng lặp tối đa", 100, 700, 350, 50)
    min_c = c4.slider("Sàn tiêu dùng", 0.45, 0.75, 0.55, 0.01)

    c5, c6, c7, c8 = st.columns(4)
    max_inv = c5.slider("Trần tổng đầu tư", 0.25, 0.48, 0.42, 0.01)
    min_h = c6.slider("Sàn tỷ lệ đầu tư H", 0.01, 0.10, 0.03, 0.01)
    min_dai = c7.slider("Sàn D+AI", 0.02, 0.15, 0.04, 0.01)
    brain_drain = c8.slider("Chảy máu chất xám μ", 0.00, 0.08, 0.02, 0.005)

    run_clicked = st.button(
        "🌸 Tối ưu quỹ đạo 2026–2035",
        type="primary",
        use_container_width=True,
    )

signature = (
    MODEL_VERSION,
    rho,
    short_rho,
    maxiter,
    min_c,
    max_inv,
    min_h,
    min_dai,
    brain_drain,
)

if (
    run_clicked
    or "bai08_result" not in st.session_state
    or st.session_state.get("bai08_signature") != signature
):
    st.session_state.pop("bai08_gemini_analysis", None)

    config = DynamicConfig(
        rho=rho,
        min_consumption_share=min_c,
        max_total_investment_share=max_inv,
        min_H_investment_share=min_h,
        min_DAI_investment_share=min_dai,
        brain_drain=brain_drain,
    )

    try:
        with st.spinner("Đang giải SLSQP và benchmark Bellman-style..."):
            st.session_state["bai08_result"] = run_full_bai08(
                config=config,
                short_term_rho=short_rho,
                maxiter=int(maxiter),
            )
            st.session_state["bai08_signature"] = signature
    except Exception as error:
        st.error(f"Không chạy được Bài 8: {error}")
        st.stop()

result = st.session_state["bai08_result"]
optimal = result["optimal"]
short_term = result["short_term"]
path = optimal["path"]
bellman = result["bellman"]

tabs = st.tabs([
    "8.1 — Bối cảnh",
    "8.2 — Mô hình động",
    "8.3.1 — Quỹ đạo tối ưu",
    "8.3.2 — Đầu tư liên thời gian",
    "8.3.3 — Bellman benchmark",
    "8.4 — Chính sách",
    "✨ Phân tích AI",
])

with tabs[0]:
    st.subheader("8.1 — Tăng trưởng dài hạn là bài toán về thời điểm")
    context = pd.DataFrame({
        "Mục tiêu": [
            "Tăng trưởng và thu nhập 2030–2045",
            "Tích lũy hạ tầng số",
            "Phát triển AI",
            "Nâng vốn nhân lực số",
        ],
        "Vấn đề liên thời gian": [
            "Đầu tư hôm nay làm giảm tiêu dùng hiện tại nhưng tăng sản lượng tương lai",
            "D khấu hao nhanh nên cần duy trì đầu tư",
            "AI khấu hao công nghệ cao và cần năng lực hấp thụ",
            "H tích lũy chậm, chịu chảy máu chất xám",
        ],
    })
    st.dataframe(context, use_container_width=True, hide_index=True)

    baseline = pd.DataFrame({
        "Biến": ["K0", "D0", "AI0", "H0", "L0", "ρ"],
        "Giá trị": [27500, 20.3, 86.0, 30.0, 53.9, rho],
    })
    fig_base = px.bar(
        baseline,
        x="Biến",
        y="Giá trị",
        color="Biến",
        text_auto=".2f",
        color_discrete_sequence=[PINK, MINT, LAVENDER, YELLOW, BLUE, ROSE],
    )
    fig_base.update_layout(showlegend=False)
    st.plotly_chart(
        style_plotly(fig_base, "Điều kiện ban đầu của mô hình", "Biến trạng thái", "Giá trị"),
        use_container_width=True,
    )

with tabs[1]:
    st.subheader("8.2 — Mô hình toán học")
    st.latex(r"Y_t=A_tK_t^{0.33}L_t^{0.42}D_t^{0.10}AI_t^{0.08}H_t^{0.07}")
    st.latex(r"K_{t+1}=(1-\delta_K)K_t+I_{K,t}")
    st.latex(r"D_{t+1}=(1-\delta_D)D_t+s_DI_{D,t}")
    st.latex(r"AI_{t+1}=(1-\delta_{AI})AI_t+s_{AI}I_{AI,t}")
    st.latex(r"H_{t+1}=H_t+\theta_Hs_HI_{H,t}-\mu H_t")
    st.latex(r"\max \sum_{t=0}^{9}\rho^t\log(C_t)")
    st.info(
        "Quyết định gồm 5 tỷ lệ mỗi năm: tiêu dùng, đầu tư K, D, AI và H. "
        "Toàn bài có 50 biến quyết định."
    )

with tabs[2]:
    st.subheader("8.3.1 — Quỹ đạo trạng thái và sản lượng")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Solver", "SLSQP")
    k2.metric("Thành công?", "Có" if optimal["success"] else "Cảnh báo")
    k3.metric("Phúc lợi", f"{optimal['welfare']:.4f}")
    k4.metric("GDP năm cuối", f"{path['Y'].iloc[-1]:,.2f}")

    state_long = path.melt(
        id_vars="year",
        value_vars=["K", "D", "AI", "H", "A_TFP"],
        var_name="Trạng thái",
        value_name="Giá trị",
    )
    fig_state = px.line(
        state_long,
        x="year",
        y="Giá trị",
        color="Trạng thái",
        markers=True,
        color_discrete_sequence=[PINK, MINT, LAVENDER, YELLOW, BLUE],
    )
    st.plotly_chart(
        style_plotly(fig_state, "Quỹ đạo K, D, AI, H và TFP", "Năm", "Chỉ số/trạng thái", 560),
        use_container_width=True,
    )

    output_long = path.melt(
        id_vars="year",
        value_vars=["Y", "C", "Total_investment"],
        var_name="Chỉ tiêu",
        value_name="Giá trị",
    )
    fig_output = px.line(
        output_long,
        x="year",
        y="Giá trị",
        color="Chỉ tiêu",
        markers=True,
        color_discrete_sequence=[PINK, MINT, LAVENDER],
    )
    st.plotly_chart(
        style_plotly(fig_output, "Sản lượng, tiêu dùng và đầu tư", "Năm", "Giá trị"),
        use_container_width=True,
    )

    st.dataframe(path.round(5), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Tải quỹ đạo tối ưu", csv_bytes(path), "bai08_quy_dao.csv", "text/csv")

with tabs[3]:
    st.subheader("8.3.2 — Cơ cấu đầu tư theo thời gian")
    investment_long = path.melt(
        id_vars="year",
        value_vars=["I_K", "I_D", "I_AI", "I_H"],
        var_name="Hạng mục",
        value_name="Đầu tư",
    )
    fig_inv = px.area(
        investment_long,
        x="year",
        y="Đầu tư",
        color="Hạng mục",
        color_discrete_sequence=[PINK, MINT, LAVENDER, YELLOW],
    )
    st.plotly_chart(
        style_plotly(fig_inv, "Cơ cấu đầu tư liên thời gian", "Năm", "Đầu tư", 540),
        use_container_width=True,
    )

    share_long = path.melt(
        id_vars="year",
        value_vars=["IK_share", "ID_share", "IAI_share", "IH_share"],
        var_name="Tỷ lệ",
        value_name="Giá trị",
    )
    fig_share = px.line(
        share_long,
        x="year",
        y="Giá trị",
        color="Tỷ lệ",
        markers=True,
        color_discrete_sequence=[PINK, MINT, LAVENDER, YELLOW],
    )
    st.plotly_chart(
        style_plotly(fig_share, "Tỷ lệ đầu tư tối ưu theo năm", "Năm", "Tỷ lệ"),
        use_container_width=True,
    )

    st.success(
        f"Mẫu phân bổ: **{optimal['loading_pattern']}**. "
        f"Tỷ lệ AI/H trung bình = **{optimal['ai_h_ratio_mean']:.3f}**, "
        f"hệ số biến thiên = **{optimal['ai_h_ratio_cv']:.3f}**."
    )

with tabs[4]:
    st.subheader("8.3.3 — Benchmark Bellman-style dạng lưới")
    table = bellman["table"]
    best = bellman["best"]

    b1, b2, b3 = st.columns(3)
    b1.metric("Tỷ lệ tiết kiệm tốt nhất", f"{best['saving_share']:.2%}")
    b2.metric("Định hướng AI", f"{best['ai_orientation']:.2%}")
    b3.metric("Phúc lợi benchmark", f"{best['welfare']:.4f}")

    fig_grid = px.scatter(
        table,
        x="saving_share",
        y="ai_orientation",
        color="welfare",
        size="GDP_2035",
        color_continuous_scale=["#F4B8C8", "#EEDCF5", "#A8D5D1"],
        hover_data=["C_total", "GDP_2035"],
    )
    st.plotly_chart(
        style_plotly(fig_grid, "Không gian chính sách rời rạc Bellman-style", "Tỷ lệ tiết kiệm", "Định hướng AI", 550),
        use_container_width=True,
    )
    st.caption(
        "Đây là benchmark lưới chính sách tỷ lệ cố định để đối chiếu với nghiệm SLSQP 50 biến, "
        "không thay thế hoàn toàn Bellman đa trạng thái."
    )

with tabs[5]:
    st.subheader("8.4 — Thảo luận chính sách")
    compare = result["discount_comparison"]
    st.dataframe(compare.round(4), use_container_width=True, hide_index=True)

    fig_compare = px.bar(
        compare,
        x="Kịch bản",
        y=["Đầu tư AI tổng", "Đầu tư H tổng", "Tiêu dùng tổng"],
        barmode="group",
        color_discrete_sequence=[PINK, MINT, LAVENDER],
    )
    st.plotly_chart(
        style_plotly(fig_compare, "Ảnh hưởng của hệ số chiết khấu", "Kịch bản", "Tổng giai đoạn"),
        use_container_width=True,
    )

    st.markdown(
        f"""
        **a) Front-loaded hay back-loaded?** Kết quả hiện tại là
        **{optimal['loading_pattern']}**. Đầu tư sớm tạo lợi ích tích lũy qua K, D, AI,
        H và TFP; đầu tư muộn bảo vệ tiêu dùng hiện tại nhưng mất thời gian khuếch đại.

        **b) AI đi trước hay đồng thời với H?** Tỷ lệ AI/H trung bình là
        **{optimal['ai_h_ratio_mean']:.3f}**. Chính sách hợp lý là đầu tư đồng thời,
        vì AI thiếu nhân lực sẽ làm giảm năng lực hấp thụ và tăng rủi ro chuyển đổi.

        **c) ρ thấp hơn:** kịch bản ρ={short_rho:.2f} đặt trọng lượng thấp hơn cho
        lợi ích xa trong tương lai, thường làm tăng ưu tiên tiêu dùng hiện tại và giảm
        đầu tư dài hạn. Đây là một cơ chế giải thích hiện tượng dưới đầu tư R&D.
        """
    )

with tabs[6]:
    st.subheader("Tác nhân AI — Bài 8")
    configured = gemini_is_configured()
    if not configured:
        st.info("Phần AI sẽ được chuẩn hóa đồng loạt sau khi hoàn thành Bài 12.")

    summary = f"""
Bài 8:
- Thành công: {optimal['success']}
- Phúc lợi: {optimal['welfare']:.6f}
- Mẫu phân bổ: {optimal['loading_pattern']}
- GDP cuối kỳ: {path['Y'].iloc[-1]:.6f}
- AI/H trung bình: {optimal['ai_h_ratio_mean']:.6f}
- So sánh chiết khấu:
{result['discount_comparison'].round(5).to_string(index=False)}
"""

    if st.button(
        "✨ Phân tích kết quả bằng Gemini",
        disabled=not configured,
        use_container_width=True,
        key="gemini_bai08",
    ):
        try:
            with st.spinner("Gemini đang phân tích Bài 8..."):
                st.session_state["bai08_gemini_analysis"] = analyze_result(
                    exercise_name="Bài 8 — Tối ưu động 2026–2035",
                    model_name="SLSQP nonlinear dynamic optimization và Bellman-style benchmark",
                    parameters={"rho": rho, "rho_short": short_rho},
                    result_summary=summary,
                    policy_questions="Đầu tư sớm/muộn, AI-H, chiết khấu dài hạn và dưới đầu tư R&D.",
                )
        except GeminiAgentError as error:
            st.error(str(error))

    if st.session_state.get("bai08_gemini_analysis"):
        st.markdown(st.session_state["bai08_gemini_analysis"])
