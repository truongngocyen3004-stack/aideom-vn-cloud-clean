from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Year",
    "GDP_trillion_VND",
    "K",
    "L",
    "D",
    "AI",
    "H",
]


def load_macro_data(csv_path: str | Path) -> pd.DataFrame:
    """Đọc và kiểm tra dữ liệu vĩ mô Việt Nam 2020-2025."""

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}"
        )

    df = pd.read_csv(path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "File dữ liệu thiếu các cột: "
            + ", ".join(missing)
        )

    df = df[REQUIRED_COLUMNS].copy()

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    df = (
        df.sort_values("Year")
        .reset_index(drop=True)
    )

    if df["Year"].duplicated().any():
        raise ValueError(
            "Cột Year có năm bị trùng."
        )

    if (
        df[
            [
                "GDP_trillion_VND",
                "K",
                "L",
                "D",
                "AI",
                "H",
            ]
        ]
        <= 0
    ).any().any():
        raise ValueError(
            "GDP và các yếu tố đầu vào phải lớn hơn 0."
        )

    return df


def validate_elasticities(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    theta: float,
    tolerance: float = 1e-8,
) -> None:
    """Kiểm tra các hệ số Cobb-Douglas."""

    coefficients = np.array(
        [alpha, beta, gamma, delta, theta],
        dtype=float,
    )

    if np.any(coefficients < 0):
        raise ValueError(
            "Các hệ số co giãn không được âm."
        )

    if not np.isclose(
        coefficients.sum(),
        1.0,
        atol=tolerance,
    ):
        raise ValueError(
            "Tổng alpha + beta + gamma + delta + theta "
            f"phải bằng 1. Tổng hiện tại = {coefficients.sum():.6f}."
        )


def calculate_tfp(
    data: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    theta: float,
) -> pd.DataFrame:
    """Câu 1.4.1: Tính TFP A_t bằng cách giải ngược hàm sản xuất."""

    validate_elasticities(
        alpha,
        beta,
        gamma,
        delta,
        theta,
    )

    df = data.copy()

    denominator = (
        df["K"] ** alpha
        * df["L"] ** beta
        * df["D"] ** gamma
        * df["AI"] ** delta
        * df["H"] ** theta
    )

    df["TFP_A_t"] = (
        df["GDP_trillion_VND"]
        / denominator
    )

    df["TFP_growth_pct"] = (
        df["TFP_A_t"]
        .pct_change()
        * 100.0
    )

    return df


def forecast_with_mean_tfp(
    tfp_table: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    theta: float,
) -> dict[str, Any]:
    """Câu 1.4.2: Dùng TFP trung bình để dự báo GDP và tính MAPE."""

    mean_tfp = float(
        tfp_table["TFP_A_t"].mean()
    )

    forecast = tfp_table.copy()

    forecast["GDP_forecast"] = (
        mean_tfp
        * forecast["K"] ** alpha
        * forecast["L"] ** beta
        * forecast["D"] ** gamma
        * forecast["AI"] ** delta
        * forecast["H"] ** theta
    )

    forecast["Error"] = (
        forecast["GDP_forecast"]
        - forecast["GDP_trillion_VND"]
    )

    forecast["APE_pct"] = (
        np.abs(
            forecast["Error"]
            / forecast["GDP_trillion_VND"]
        )
        * 100.0
    )

    mape = float(
        forecast["APE_pct"].mean()
    )

    max_error_index = (
        forecast["APE_pct"].idxmax()
    )

    max_error_year = int(
        forecast.loc[
            max_error_index,
            "Year",
        ]
    )

    max_ape = float(
        forecast.loc[
            max_error_index,
            "APE_pct",
        ]
    )

    return {
        "mean_tfp": mean_tfp,
        "mape": mape,
        "max_error_year": max_error_year,
        "max_ape": max_ape,
        "table": forecast,
    }


def growth_accounting(
    tfp_table: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    theta: float,
) -> dict[str, pd.DataFrame]:
    """Câu 1.4.3: Phân rã tăng trưởng bằng sai phân logarit."""

    df = tfp_table.copy()

    log_columns = {
        "GDP": "GDP_trillion_VND",
        "TFP": "TFP_A_t",
        "K": "K",
        "L": "L",
        "D": "D",
        "AI": "AI",
        "H": "H",
    }

    log_diff = pd.DataFrame(
        {
            key: np.log(
                df[source]
            ).diff() * 100.0
            for key, source in log_columns.items()
        }
    )

    annual = pd.DataFrame({
        "Period": (
            df["Year"].shift(1).astype("Int64").astype(str)
            + "-"
            + df["Year"].astype(int).astype(str)
        ),
        "GDP_growth_log_pct": log_diff["GDP"],
        "TFP_contribution_pp": log_diff["TFP"],
        "K_contribution_pp": alpha * log_diff["K"],
        "L_contribution_pp": beta * log_diff["L"],
        "D_contribution_pp": gamma * log_diff["D"],
        "AI_contribution_pp": delta * log_diff["AI"],
        "H_contribution_pp": theta * log_diff["H"],
    })

    annual = (
        annual.iloc[1:]
        .reset_index(drop=True)
    )

    contribution_columns = [
        "TFP_contribution_pp",
        "K_contribution_pp",
        "L_contribution_pp",
        "D_contribution_pp",
        "AI_contribution_pp",
        "H_contribution_pp",
    ]

    annual["Explained_growth_pp"] = (
        annual[contribution_columns].sum(
            axis=1
        )
    )

    annual["Residual_pp"] = (
        annual["GDP_growth_log_pct"]
        - annual["Explained_growth_pp"]
    )

    average_values = (
        annual[contribution_columns]
        .mean()
    )

    average_gdp_growth = float(
        annual[
            "GDP_growth_log_pct"
        ].mean()
    )

    summary = pd.DataFrame({
        "Factor": [
            "TFP",
            "K",
            "L",
            "D",
            "AI",
            "H",
        ],
        "Average_contribution_pp": [
            average_values[
                "TFP_contribution_pp"
            ],
            average_values[
                "K_contribution_pp"
            ],
            average_values[
                "L_contribution_pp"
            ],
            average_values[
                "D_contribution_pp"
            ],
            average_values[
                "AI_contribution_pp"
            ],
            average_values[
                "H_contribution_pp"
            ],
        ],
    })

    if not np.isclose(
        average_gdp_growth,
        0.0,
    ):
        summary["Share_of_growth_pct"] = (
            summary[
                "Average_contribution_pp"
            ]
            / average_gdp_growth
            * 100.0
        )
    else:
        summary["Share_of_growth_pct"] = np.nan

    return {
        "annual": annual,
        "summary": summary,
    }


def forecast_to_2030(
    tfp_table: pd.DataFrame,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    theta: float,
    target_d_2030: float = 30.0,
    target_ai_2030: float = 100.0,
    target_h_2030: float = 35.0,
    k_growth_pct: float = 6.0,
    l_growth_pct: float = 6.0,
    tfp_growth_pct: float = 1.2,
    start_year: int = 2025,
    end_year: int = 2030,
) -> dict[str, Any]:
    """Câu 1.4.4: Mô phỏng quỹ đạo GDP từ 2025 đến 2030."""

    if end_year <= start_year:
        raise ValueError(
            "end_year phải lớn hơn start_year."
        )

    base_rows = tfp_table.loc[
        tfp_table["Year"] == start_year
    ]

    if base_rows.empty:
        raise ValueError(
            f"Không có dữ liệu năm {start_year}."
        )

    base = base_rows.iloc[0]

    years = np.arange(
        start_year,
        end_year + 1,
    )

    number_of_steps = (
        end_year - start_year
    )

    d_path = np.linspace(
        float(base["D"]),
        float(target_d_2030),
        number_of_steps + 1,
    )

    ai_path = np.linspace(
        float(base["AI"]),
        float(target_ai_2030),
        number_of_steps + 1,
    )

    h_path = np.linspace(
        float(base["H"]),
        float(target_h_2030),
        number_of_steps + 1,
    )

    step_index = np.arange(
        number_of_steps + 1
    )

    k_path = (
        float(base["K"])
        * (
            1.0
            + k_growth_pct / 100.0
        ) ** step_index
    )

    l_path = (
        float(base["L"])
        * (
            1.0
            + l_growth_pct / 100.0
        ) ** step_index
    )

    tfp_path = (
        float(base["TFP_A_t"])
        * (
            1.0
            + tfp_growth_pct / 100.0
        ) ** step_index
    )

    gdp_path = (
        tfp_path
        * k_path ** alpha
        * l_path ** beta
        * d_path ** gamma
        * ai_path ** delta
        * h_path ** theta
    )

    forecast_table = pd.DataFrame({
        "Year": years,
        "K": k_path,
        "L": l_path,
        "D": d_path,
        "AI": ai_path,
        "H": h_path,
        "TFP_A_t": tfp_path,
        "GDP_forecast": gdp_path,
    })

    for factor in [
        "K",
        "L",
        "D",
        "AI",
        "H",
        "TFP_A_t",
        "GDP_forecast",
    ]:
        forecast_table[
            f"{factor}_index_2025_100"
        ] = (
            forecast_table[factor]
            / forecast_table[factor].iloc[0]
            * 100.0
        )

    gdp_2025 = float(
        forecast_table[
            "GDP_forecast"
        ].iloc[0]
    )

    gdp_2030 = float(
        forecast_table[
            "GDP_forecast"
        ].iloc[-1]
    )

    cagr = (
        (
            gdp_2030
            / gdp_2025
        ) ** (
            1.0 / number_of_steps
        )
        - 1.0
    ) * 100.0

    return {
        "table": forecast_table,
        "gdp_2025": gdp_2025,
        "gdp_2030": gdp_2030,
        "increase_pct": (
            gdp_2030
            / gdp_2025
            - 1.0
        ) * 100.0,
        "cagr_pct": float(cagr),
        "tfp_2030": float(
            forecast_table[
                "TFP_A_t"
            ].iloc[-1]
        ),
    }


def run_full_bai01(
    csv_path: str | Path,
    alpha: float = 0.33,
    beta: float = 0.42,
    gamma: float = 0.10,
    delta: float = 0.08,
    theta: float = 0.07,
    target_d_2030: float = 30.0,
    target_ai_2030: float = 100.0,
    target_h_2030: float = 35.0,
    k_growth_pct: float = 6.0,
    l_growth_pct: float = 6.0,
    tfp_growth_pct: float = 1.2,
) -> dict[str, Any]:
    """Chạy toàn bộ yêu cầu Bài 1 trong một hàm."""

    data = load_macro_data(
        csv_path
    )

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

    scenario_result = (
        forecast_to_2030(
            tfp_table=tfp_table,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            theta=theta,
            target_d_2030=(
                target_d_2030
            ),
            target_ai_2030=(
                target_ai_2030
            ),
            target_h_2030=(
                target_h_2030
            ),
            k_growth_pct=(
                k_growth_pct
            ),
            l_growth_pct=(
                l_growth_pct
            ),
            tfp_growth_pct=(
                tfp_growth_pct
            ),
        )
    )

    return {
        "input_data": data,
        "tfp": tfp_table,
        "forecast": forecast_result,
        "accounting": accounting_result,
        "scenario_2030": scenario_result,
    }
