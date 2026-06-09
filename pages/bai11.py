from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.bai11_model import (
    ACTION_ALLOCATIONS,
    ACTION_NAMES,
    ACTUAL_VIETNAM_2026_STATE,
    GYM_AVAILABLE,
    RLConfig,
    action_table,
    run_full_bai11,
    state_to_label,
    train_dqn_optional,
)
from services.ai_agent import (
    GeminiAgentError,
    analyze_result,
    gemini_is_configured,
)
from ui.theme import page_header


MODEL_VERSION = "bai11_v1"

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
    "Bài 11 — Q-learning cho chính sách kinh tế thích nghi",
    "Mô hình hóa nền kinh tế Việt Nam thành MDP 81 trạng thái, huấn luyện Q-table qua nhiều episode, so sánh với chính sách cố định và mở rộng DQN.",
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
        <b>Trạng thái:</b>
        tăng trưởng GDP, chỉ số số hóa, năng lực AI và rủi ro thất nghiệp,
        mỗi yếu tố có 3 mức nên có 3⁴ = 81 trạng thái.
        <br>
        <b>Hành động:</b>
        5 cấu trúc phân bổ ngân sách K–D–AI–H.
        <br>
        <b>Reward:</b>
        tăng trưởng trừ thất nghiệp, rủi ro an ninh mạng và phát thải.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander(
    "⚙️ Thiết lập Q-learning",
    expanded=True,
):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        n_episodes = st.slider(
            "Số episode huấn luyện",
            min_value=1_000,
            max_value=20_000,
            value=10_000,
            step=1_000,
        )

    with c2:
        learning_rate = st.slider(
            "Learning rate α",
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01,
        )

    with c3:
        discount_factor = st.slider(
            "Discount γ",
            min_value=0.70,
            max_value=0.99,
            value=0.95,
            step=0.01,
        )

    with c4:
        shock_probability = st.slider(
            "Xác suất cú sốc",
            min_value=0.00,
            max_value=0.40,
            value=0.12,
            step=0.01,
        )

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        annual_budget = st.number_input(
            "Ngân sách mỗi năm",
            min_value=500.0,
            max_value=5000.0,
            value=1000.0,
            step=100.0,
        )

    with c6:
        epsilon_end = st.slider(
            "Epsilon cuối",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
        )

    with c7:
        evaluation_episodes = st.slider(
            "Episode đánh giá",
            min_value=50,
            max_value=500,
            value=300,
            step=50,
        )

    with c8:
        seed = st.number_input(
            "Seed",
            min_value=1,
            max_value=999,
            value=42,
            step=1,
        )

    st.markdown("**Trọng số reward**")

    w1, w2, w3, w4 = st.columns(4)

    reward_growth = w1.slider(
        "Tăng trưởng",
        0.05,
        0.80,
        0.40,
        0.05,
    )

    reward_unemployment = w2.slider(
        "Thất nghiệp",
        0.05,
        0.60,
        0.25,
        0.05,
    )

    reward_cyber = w3.slider(
        "Cyber risk",
        0.05,
        0.60,
        0.20,
        0.05,
    )

    reward_emission = w4.slider(
        "Phát thải",
        0.05,
        0.60,
        0.15,
        0.05,
    )

    run_clicked = st.button(
        "🌸 Huấn luyện Q-learning Bài 11",
        type="primary",
        use_container_width=True,
    )

signature = (
    MODEL_VERSION,
    n_episodes,
    learning_rate,
    discount_factor,
    shock_probability,
    annual_budget,
    epsilon_end,
    evaluation_episodes,
    int(seed),
    reward_growth,
    reward_unemployment,
    reward_cyber,
    reward_emission,
)

if (
    run_clicked
    or "bai11_result"
    not in st.session_state
    or st.session_state.get(
        "bai11_signature"
    ) != signature
):
    st.session_state.pop(
        "bai11_gemini_analysis",
        None,
    )
    st.session_state.pop(
        "bai11_dqn_result",
        None,
    )

    config = RLConfig(
        n_episodes=int(
            n_episodes
        ),
        learning_rate=float(
            learning_rate
        ),
        discount_factor=float(
            discount_factor
        ),
        epsilon_start=1.0,
        epsilon_end=float(
            epsilon_end
        ),
        episode_length=10,
        annual_budget=float(
            annual_budget
        ),
        shock_probability=float(
            shock_probability
        ),
        reward_growth_weight=float(
            reward_growth
        ),
        reward_unemployment_weight=float(
            reward_unemployment
        ),
        reward_cyber_weight=float(
            reward_cyber
        ),
        reward_emission_weight=float(
            reward_emission
        ),
        seed=int(seed),
        evaluation_episodes=int(
            evaluation_episodes
        ),
    )

    with st.spinner(
        "Đang huấn luyện Q-table và đánh giá các chính sách..."
    ):
        st.session_state[
            "bai11_result"
        ] = run_full_bai11(
            config
        )

        st.session_state[
            "bai11_signature"
        ] = signature

result = st.session_state[
    "bai11_result"
]

if not result.get(
    "success",
    False,
):
    st.error(
        result.get(
            "status",
            "Không chạy được Bài 11.",
        )
    )

    st.code(
        "python -m pip install gymnasium",
        language="bash",
    )

    st.dataframe(
        result.get(
            "environment_check",
            pd.DataFrame(),
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.stop()

config = result[
    "config"
]
training = result[
    "training"
]
comparison = result[
    "comparison"
]
policy_map = result[
    "policy_map"
]
test_states = result[
    "test_states"
]

tabs = st.tabs([
    "11.1 — Bối cảnh",
    "11.2 — MDP",
    "11.3.1 — Environment",
    "11.3.2 — Q-learning",
    "11.3.3 — Policy π*",
    "11.3.4 — So sánh",
    "11.3.5 — DQN",
    "11.4 — Chính sách",
    "✨ Phân tích AI",
])

with tabs[0]:
    st.subheader(
        "11.1 — Từ chính sách cố định đến chính sách thích nghi"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Số trạng thái",
        "81",
        "3⁴",
    )
    c2.metric(
        "Số hành động",
        "5",
        "a0–a4",
    )
    c3.metric(
        "Một episode",
        "10 năm",
        "T = 10",
    )
    c4.metric(
        "Training",
        f"{config.n_episodes:,}",
        "episodes",
    )

    labels = [
        "Trạng thái kinh tế sₜ",
        "GDP growth",
        "Digital index",
        "AI capacity",
        "Unemployment risk",
        "Chính sách aₜ",
        "Phân bổ K/D/AI/H",
        "Môi trường kinh tế",
        "Reward Rₜ",
        "Trạng thái sₜ₊₁",
        "Cập nhật Q(s,a)",
    ]

    sankey = go.Figure(
        data=[
            go.Sankey(
                node={
                    "label": labels,
                    "pad": 18,
                    "thickness": 18,
                    "color": [
                        PINK,
                        MINT,
                        LAVENDER,
                        YELLOW,
                        BLUE,
                        ROSE,
                        PINK,
                        MINT,
                        LAVENDER,
                        YELLOW,
                        BLUE,
                    ],
                },
                link={
                    "source": [
                        0,
                        0,
                        0,
                        0,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                    ],
                    "target": [
                        1,
                        2,
                        3,
                        4,
                        6,
                        7,
                        8,
                        9,
                        10,
                        5,
                    ],
                    "value": [
                        20,
                        20,
                        20,
                        20,
                        80,
                        80,
                        80,
                        80,
                        80,
                        40,
                    ],
                },
            )
        ]
    )

    st.plotly_chart(
        style_plotly(
            sankey,
            "Vòng lặp MDP: quan sát → hành động → reward → trạng thái mới",
            height=570,
        ),
        use_container_width=True,
    )

    st.warning(
        "RL chỉ minh họa kỹ thuật thích nghi. "
        "Agent không thay thế trách nhiệm chính trị, giám sát xã hội "
        "hay đánh giá tác động phân phối."
    )

with tabs[1]:
    st.subheader(
        "11.2 — Trạng thái, hành động, reward và transition"
    )

    st.markdown(
        "**Không gian trạng thái**"
    )

    state_table = pd.DataFrame({
        "Yếu tố": [
            "GDP growth",
            "Digital index",
            "AI capacity",
            "Unemployment risk",
        ],
        "Mức": [
            "low / medium / high",
            "low / medium / high",
            "low / medium / high",
            "low / medium / high",
        ],
        "Ý nghĩa": [
            "Sức khỏe tăng trưởng",
            "Nền tảng số",
            "Năng lực AI",
            "Rủi ro dịch chuyển lao động",
        ],
    })

    st.dataframe(
        state_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "**Năm hành động phân bổ**"
    )

    actions = action_table()

    st.dataframe(
        actions,
        use_container_width=True,
        hide_index=True,
    )

    action_long = actions.melt(
        id_vars=[
            "action_id",
            "action",
        ],
        value_vars=[
            "K_share",
            "D_share",
            "AI_share",
            "H_share",
        ],
        var_name="Hạng mục",
        value_name="Tỷ trọng",
    )

    fig_actions = px.bar(
        action_long,
        x="action",
        y="Tỷ trọng",
        color="Hạng mục",
        barmode="stack",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_actions.update_xaxes(
        tickangle=-18
    )

    st.plotly_chart(
        style_plotly(
            fig_actions,
            "Cấu trúc năm hành động chính sách",
            "Hành động",
            "Tỷ trọng ngân sách",
            height=540,
        ),
        use_container_width=True,
    )

    st.markdown(
        "**Reward**"
    )

    st.latex(
        r"""
        R_t =
        w_g Growth_t
        - w_u \Delta U_t
        - w_c CyberRisk_t
        - w_e Emission_t
        """
    )

    reward_table = pd.DataFrame({
        "Thành phần": [
            "Tăng trưởng",
            "Thất nghiệp",
            "Cyber risk",
            "Phát thải",
        ],
        "Trọng số chuẩn hóa":
            config.reward_weights,
        "Dấu trong reward": [
            "+",
            "−",
            "−",
            "−",
        ],
    })

    st.dataframe(
        reward_table.round(4),
        use_container_width=True,
        hide_index=True,
    )

with tabs[2]:
    st.subheader(
        "Câu 11.3.1 — Kiểm tra môi trường Gymnasium"
    )

    st.dataframe(
        result[
            "environment_check"
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.code(
        """
env = VietnamEconomyEnv()
state, info = env.reset(seed=42)
next_state, reward, terminated, truncated, info = env.step(action)
""".strip(),
        language="python",
    )

    st.success(
        "Environment có action_space = Discrete(5), "
        "observation_space = MultiDiscrete([3,3,3,3]) "
        "và mỗi episode kết thúc sau 10 bước."
    )

with tabs[3]:
    st.subheader(
        "Câu 11.3.2 — Huấn luyện Q-learning"
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Reward 500 episode đầu",
        f"{training['reward_first_500']:.4f}",
    )
    k2.metric(
        "Reward 500 episode cuối",
        f"{training['reward_last_500']:.4f}",
    )
    k3.metric(
        "Cải thiện",
        f"{training['improvement']:.4f}",
    )
    k4.metric(
        "Epsilon cuối",
        f"{training['epsilons'][-1]:.3f}",
    )

    learning_curve = training[
        "learning_curve"
    ]

    fig_learning = go.Figure()

    fig_learning.add_trace(
        go.Scatter(
            x=learning_curve[
                "episode"
            ],
            y=learning_curve[
                "reward"
            ],
            name="Reward từng episode",
            mode="lines",
            opacity=0.20,
            line={
                "color": ROSE,
                "width": 1,
            },
        )
    )

    fig_learning.add_trace(
        go.Scatter(
            x=learning_curve[
                "episode"
            ],
            y=learning_curve[
                "reward_rolling"
            ],
            name="Reward trung bình trượt",
            mode="lines",
            line={
                "color": PINK,
                "width": 3,
            },
        )
    )

    st.plotly_chart(
        style_plotly(
            fig_learning,
            "Learning curve của Q-learning",
            "Episode",
            "Reward tích lũy",
            height=550,
        ),
        use_container_width=True,
    )

    fig_epsilon = px.line(
        learning_curve,
        x="episode",
        y="epsilon",
        color_discrete_sequence=[
            MINT
        ],
    )

    st.plotly_chart(
        style_plotly(
            fig_epsilon,
            "Epsilon giảm từ 1,0 xuống mức cuối",
            "Episode",
            "Epsilon",
            height=430,
        ),
        use_container_width=True,
    )

    action_training = training[
        "action_table"
    ].copy()

    fig_action_count = px.bar(
        action_training,
        x="action",
        y="training_count",
        color="action",
        text_auto=",.0f",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_action_count.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        style_plotly(
            fig_action_count,
            "Tần suất hành động trong quá trình huấn luyện",
            "Hành động",
            "Số lần",
            height=470,
        ),
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Tải learning curve",
        data=csv_bytes(
            learning_curve
        ),
        file_name=(
            "bai11_1132_learning_curve.csv"
        ),
        mime="text/csv",
    )

with tabs[4]:
    st.subheader(
        "Câu 11.3.3 — Chính sách tối ưu π*(s)"
    )

    st.markdown(
        "#### Năm trạng thái kiểm tra"
    )

    st.dataframe(
        test_states.round(5),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "#### Policy map của 81 trạng thái"
    )

    st.dataframe(
        policy_map.round(5),
        use_container_width=True,
        hide_index=True,
    )

    action_distribution = (
        policy_map.groupby(
            "best_action",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "Số trạng thái"
            }
        )
    )

    fig_policy_distribution = px.pie(
        action_distribution,
        names="best_action",
        values="Số trạng thái",
        hole=0.45,
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_policy_distribution.update_layout(
        title="Cơ cấu hành động tối ưu trên 81 trạng thái",
        height=500,
        paper_bgcolor=BG,
        font_color=TEXT,
    )

    st.plotly_chart(
        fig_policy_distribution,
        use_container_width=True,
    )

    heat_data = (
        policy_map[
            policy_map[
                "Unemployment"
            ]
            == "Trung bình"
        ]
        .pivot_table(
            index="GDP",
            columns=[
                "Digital",
                "AI",
            ],
            values="best_action_id",
            aggfunc="first",
        )
    )

    fig_policy_heat = go.Figure(
        data=go.Heatmap(
            z=heat_data.values,
            x=[
                f"D={column[0]}, AI={column[1]}"
                for column in heat_data.columns
            ],
            y=heat_data.index,
            colorscale=[
                [0.00, "#F4B8C8"],
                [0.25, "#A8D5D1"],
                [0.50, "#CDB8E5"],
                [0.75, "#F2D7A7"],
                [1.00, "#A9C9E8"],
            ],
            zmin=0,
            zmax=4,
            text=heat_data.values,
            texttemplate="a%{text}",
            colorbar={
                "title": "Action ID",
            },
        )
    )

    st.plotly_chart(
        style_plotly(
            fig_policy_heat,
            "Policy map khi rủi ro thất nghiệp ở mức trung bình",
            "Tổ hợp D–AI",
            "GDP growth",
            height=500,
        ),
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Tải policy map",
        data=csv_bytes(
            policy_map
        ),
        file_name=(
            "bai11_1133_policy_map.csv"
        ),
        mime="text/csv",
    )

with tabs[5]:
    st.subheader(
        "Câu 11.3.4 — So sánh π* với ba chính sách rule-based"
    )

    summary = comparison[
        "summary"
    ]

    st.dataframe(
        summary.round(5),
        use_container_width=True,
        hide_index=True,
    )

    fig_box = px.box(
        comparison[
            "compare_df"
        ],
        x="policy_label",
        y="total_reward",
        color="policy_label",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_box.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        style_plotly(
            fig_box,
            "Phân phối reward tích lũy của các chính sách",
            "Chính sách",
            "Reward 10 năm",
            height=540,
        ),
        use_container_width=True,
    )

    fig_summary = px.bar(
        summary,
        x="policy_label",
        y="mean_reward",
        error_y="std_reward",
        color="policy_label",
        text_auto=".4f",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    fig_summary.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        style_plotly(
            fig_summary,
            "Reward trung bình của π* và rule-based policies",
            "Chính sách",
            "Mean reward",
            height=500,
        ),
        use_container_width=True,
    )

    q_trace = comparison[
        "trace_df"
    ][
        comparison[
            "trace_df"
        ][
            "policy_label"
        ]
        == "π* Q-learning"
    ]

    st.markdown(
        "#### Một quỹ đạo minh họa của π*"
    )

    st.dataframe(
        q_trace[
            [
                "t",
                "state_before",
                "action_label",
                "reward",
                "state_after",
                "gdp_growth_pct",
                "D",
                "AI",
                "H",
                "U",
                "cyber_risk",
                "emission",
                "shock",
            ]
        ].round(4),
        use_container_width=True,
        hide_index=True,
    )

    trace_long = q_trace.melt(
        id_vars="t",
        value_vars=[
            "reward",
            "gdp_growth_pct",
            "U",
            "cyber_risk",
            "emission",
        ],
        var_name="Chỉ tiêu",
        value_name="Giá trị",
    )

    fig_trace = px.line(
        trace_long,
        x="t",
        y="Giá trị",
        color="Chỉ tiêu",
        markers=True,
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    st.plotly_chart(
        style_plotly(
            fig_trace,
            "Quỹ đạo π*: reward, tăng trưởng, thất nghiệp và rủi ro",
            "Năm",
            "Giá trị",
            height=540,
        ),
        use_container_width=True,
    )

with tabs[6]:
    st.subheader(
        "Câu 11.3.5 — Mở rộng Deep Q-Network"
    )

    st.warning(
        "DQN là phần mở rộng. Huấn luyện trực tiếp trên web có thể chậm "
        "và phụ thuộc khả năng cài PyTorch trên Python đang sử dụng. "
        "Q-learning tabular vẫn là kết quả chính, minh bạch hơn với 81 trạng thái."
    )

    d1, d2, d3 = st.columns(3)

    total_timesteps = d1.slider(
        "DQN timesteps",
        min_value=500,
        max_value=10_000,
        value=2_000,
        step=500,
    )

    dqn_learning_rate = d2.select_slider(
        "DQN learning rate",
        options=[
            1e-4,
            5e-4,
            1e-3,
            5e-3,
        ],
        value=1e-3,
    )

    dqn_gamma = d3.slider(
        "DQN gamma",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
    )

    st.code(
        """
model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs={"net_arch": [64, 64]},
    learning_rate=1e-3,
    gamma=0.95,
)
""".strip(),
        language="python",
    )

    train_dqn_clicked = st.button(
        "🧠 Huấn luyện DQN mở rộng",
        use_container_width=True,
        key="bai11_train_dqn",
    )

    if train_dqn_clicked:
        with st.spinner(
            "Đang huấn luyện DQN..."
        ):
            st.session_state[
                "bai11_dqn_result"
            ] = train_dqn_optional(
                config=config,
                total_timesteps=int(
                    total_timesteps
                ),
                learning_rate=float(
                    dqn_learning_rate
                ),
                gamma=float(
                    dqn_gamma
                ),
            )

    dqn_result = st.session_state.get(
        "bai11_dqn_result"
    )

    if dqn_result:
        if dqn_result.get(
            "success",
            False,
        ):
            q_mean = float(
                summary.loc[
                    summary[
                        "policy_label"
                    ]
                    == "π* Q-learning",
                    "mean_reward",
                ].iloc[0]
            )

            dqn_mean = dqn_result[
                "mean_reward"
            ]

            x1, x2, x3 = st.columns(3)

            x1.metric(
                "Q-learning mean reward",
                f"{q_mean:.4f}",
            )
            x2.metric(
                "DQN mean reward",
                f"{dqn_mean:.4f}",
            )
            x3.metric(
                "DQN − Q",
                f"{dqn_mean - q_mean:.4f}",
            )

            dqn_compare = pd.DataFrame({
                "Mô hình": [
                    "Q-learning",
                    "DQN",
                ],
                "Mean reward": [
                    q_mean,
                    dqn_mean,
                ],
                "Std reward": [
                    float(
                        summary.loc[
                            summary[
                                "policy_label"
                            ]
                            == "π* Q-learning",
                            "std_reward",
                        ].iloc[0]
                    ),
                    dqn_result[
                        "std_reward"
                    ],
                ],
            })

            fig_dqn = px.bar(
                dqn_compare,
                x="Mô hình",
                y="Mean reward",
                error_y="Std reward",
                color="Mô hình",
                text_auto=".4f",
                color_discrete_sequence=[
                    PINK,
                    MINT,
                ],
            )

            fig_dqn.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                style_plotly(
                    fig_dqn,
                    "So sánh Q-learning và DQN",
                    "Mô hình",
                    "Mean reward",
                ),
                use_container_width=True,
            )

            if dqn_mean > q_mean:
                st.success(
                    "DQN cho reward trung bình cao hơn trong lần chạy hiện tại."
                )
            else:
                st.info(
                    "DQN chưa cải thiện so với Q-learning trong lần chạy hiện tại. "
                    "Điều này hợp lý khi không gian chỉ có 81 trạng thái và thời gian training ngắn."
                )
        else:
            st.error(
                dqn_result.get(
                    "status",
                    "DQN không chạy được.",
                )
            )

            if dqn_result.get(
                "error"
            ):
                st.code(
                    dqn_result[
                        "error"
                    ]
                )

            st.code(
                "python -m pip install stable-baselines3 torch",
                language="bash",
            )

with tabs[7]:
    st.subheader(
        "11.4 — Thảo luận và hàm ý chính sách"
    )

    best_policy = summary.iloc[0]

    st.markdown(
        f"""
        **a) Chính sách thích nghi có tốt hơn không?** Trong lần đánh giá hiện tại,
        chính sách có reward trung bình cao nhất là
        **{best_policy['policy_label']}**, đạt
        **{best_policy['mean_reward']:.4f}**. Nếu π* đứng đầu, kết quả minh họa
        lợi ích của việc điều chỉnh chính sách theo trạng thái kinh tế thay vì
        giữ một cấu trúc ngân sách cố định.

        **b) Reward function là một lựa chọn giá trị.** Tăng trọng số GDP có thể
        khiến agent ưu tiên AI hoặc vốn vật chất dù thất nghiệp, phát thải và
        cyber risk tăng. Tăng trọng số an sinh sẽ làm chính sách thiên về H và
        cấu trúc bao trùm.

        **c) Tabular hay DQN?** Với chỉ 81 trạng thái, Q-table dễ kiểm tra,
        giải thích và giám sát hơn. DQN có lợi khi trạng thái liên tục hoặc
        không gian rất lớn, nhưng kém minh bạch và đòi hỏi kiểm định mạnh hơn.

        **d) Giới hạn chính sách:** môi trường là mô phỏng đơn giản, transition
        và reward không phải quy luật kinh tế đã được ước lượng nhân quả. Kết quả
        chỉ dùng để học kỹ thuật và khám phá kịch bản, không dùng để tự động hóa
        hoạch định chính sách.
        """
    )

    st.markdown(
        "#### Trạng thái Việt Nam 2026"
    )

    vn_state = test_states.iloc[0]

    st.success(
        f"{state_to_label(ACTUAL_VIETNAM_2026_STATE)} → "
        f"**{vn_state['Hành động agent chọn']}**."
    )

    reward_sensitivity = pd.DataFrame({
        "Cấu hình": [
            "Hiện tại",
            "Ưu tiên tăng trưởng",
            "Ưu tiên an sinh",
            "Ưu tiên xanh và an ninh",
        ],
        "Growth": [
            config.reward_weights[0],
            0.60,
            0.25,
            0.25,
        ],
        "Unemployment": [
            config.reward_weights[1],
            0.15,
            0.45,
            0.20,
        ],
        "Cyber": [
            config.reward_weights[2],
            0.15,
            0.15,
            0.30,
        ],
        "Emission": [
            config.reward_weights[3],
            0.10,
            0.15,
            0.25,
        ],
    })

    reward_long = reward_sensitivity.melt(
        id_vars="Cấu hình",
        var_name="Thành phần",
        value_name="Trọng số",
    )

    fig_reward = px.bar(
        reward_long,
        x="Cấu hình",
        y="Trọng số",
        color="Thành phần",
        barmode="stack",
        color_discrete_sequence=(
            PASTEL_SEQUENCE
        ),
    )

    st.plotly_chart(
        style_plotly(
            fig_reward,
            "Reward function phản ánh các hệ giá trị chính sách khác nhau",
            "Cấu hình",
            "Trọng số",
            height=500,
        ),
        use_container_width=True,
    )

with tabs[8]:
    st.subheader(
        "Tác nhân AI phân tích kết quả Bài 11"
    )

    configured = gemini_is_configured()

    if configured:
        st.success(
            "Gemini API đã được cấu hình."
        )
    else:
        st.info(
            "Phần AI sẽ được chuẩn hóa đồng loạt sau khi hoàn thành Bài 12."
        )

    result_summary = f"""
BÀI 11 — Q-LEARNING CHÍNH SÁCH KINH TẾ THÍCH NGHI

Thông số:
- Episodes: {config.n_episodes}
- Alpha: {config.learning_rate}
- Gamma: {config.discount_factor}
- Epsilon: {config.epsilon_start} -> {config.epsilon_end}
- Shock probability: {config.shock_probability}
- Reward weights: {config.reward_weights.tolist()}

Training:
- Reward 500 đầu: {training['reward_first_500']:.6f}
- Reward 500 cuối: {training['reward_last_500']:.6f}
- Cải thiện: {training['improvement']:.6f}

Năm trạng thái:
{test_states.round(5).to_string(index=False)}

So sánh chính sách:
{summary.round(6).to_string(index=False)}

Phân bố hành động policy:
{policy_map.groupby('best_action').size().to_string()}
"""

    policy_questions = """
1. Q-learning có thực sự vượt ba chính sách rule-based không?
2. Hành động agent chọn cho Việt Nam 2026 có hợp lý không?
3. Reward hiện tại có thiên lệch quá mức về tăng trưởng hay không?
4. Khi nào nên dùng DQN thay tabular Q-learning?
5. Những giới hạn thể chế và đạo đức nào ngăn việc tự động hóa hoạch định chính sách?
"""

    with st.expander(
        "Xem dữ liệu sẽ gửi cho Gemini"
    ):
        st.text_area(
            "Tóm tắt kết quả",
            value=result_summary.strip(),
            height=440,
            disabled=True,
        )

    analyze_clicked = st.button(
        "✨ Phân tích kết quả bằng Gemini",
        disabled=not configured,
        use_container_width=True,
        key="gemini_bai11",
    )

    if analyze_clicked:
        try:
            with st.spinner(
                "Gemini đang phân tích Bài 11..."
            ):
                st.session_state[
                    "bai11_gemini_analysis"
                ] = analyze_result(
                    exercise_name=(
                        "Bài 11 — Q-learning cho "
                        "chính sách kinh tế thích nghi"
                    ),
                    model_name=(
                        "Gymnasium MDP 81 trạng thái, "
                        "5 hành động, tabular Q-learning và DQN mở rộng"
                    ),
                    parameters={
                        "episodes":
                            config.n_episodes,
                        "alpha":
                            config.learning_rate,
                        "gamma":
                            config.discount_factor,
                        "epsilon_end":
                            config.epsilon_end,
                        "shock_probability":
                            config.shock_probability,
                        "reward_weights":
                            config.reward_weights.tolist(),
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

    saved_analysis = st.session_state.get(
        "bai11_gemini_analysis"
    )

    if saved_analysis:
        st.markdown(
            saved_analysis
        )

        st.download_button(
            "⬇️ Tải phân tích Gemini Bài 11",
            data=saved_analysis.encode(
                "utf-8"
            ),
            file_name=(
                "bai11_phan_tich_gemini.md"
            ),
            mime="text/markdown",
            use_container_width=True,
        )
