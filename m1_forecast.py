"""M1 - Dự báo kinh tế 2026-2030 bằng Cobb-Douglas mở rộng."""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE = {
    "year": 2025,
    "Y": 12847.6,
    "K": 25900.0,
    "L": 53.4,
    "D": 19.5,
    "AI": 80.1,
    "H": 29.2,
    "A": 34.9136,
}

ELASTICITIES = {
    "K": 0.33,
    "L": 0.42,
    "D": 0.10,
    "AI": 0.08,
    "H": 0.07,
}


def cobb_douglas(
    A: float,
    K: float,
    L: float,
    D: float,
    AI: float,
    H: float,
) -> float:
    """Tính GDP theo hàm Cobb-Douglas mở rộng."""

    return float(
        A
        * K ** ELASTICITIES["K"]
        * L ** ELASTICITIES["L"]
        * D ** ELASTICITIES["D"]
        * AI ** ELASTICITIES["AI"]
        * H ** ELASTICITIES["H"]
    )


def forecast_economy(
    total_budget: float,
    shares: np.ndarray,
    start_year: int = 2026,
    end_year: int = 2030,
) -> pd.DataFrame:
    """
    Dự báo kinh tế theo một cơ cấu phân bổ chính sách.

    Parameters
    ----------
    total_budget:
        Tổng ngân sách chính sách 2026-2030, đơn vị tỷ VND.
    shares:
        Tỷ trọng [K, D, AI, H], tổng bằng một.
    """

    shares = np.asarray(
        shares,
        dtype=float,
    )

    if shares.shape != (4,):
        raise ValueError(
            "shares phải gồm bốn tỷ trọng [K, D, AI, H]."
        )

    if not np.isclose(
        shares.sum(),
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Tổng tỷ trọng phân bổ phải bằng 1."
        )

    if total_budget <= 0:
        raise ValueError(
            "Tổng ngân sách phải lớn hơn 0."
        )

    years = np.arange(
        start_year,
        end_year + 1,
    )

    annual_budget_trillion = (
        total_budget
        / len(years)
        / 1000.0
    )

    K = float(BASE["K"])
    L = float(BASE["L"])
    D = float(BASE["D"])
    AI = float(BASE["AI"])
    H = float(BASE["H"])
    A = float(BASE["A"])

    rows = [{
        "Năm": int(BASE["year"]),
        "GDP": float(BASE["Y"]),
        "K": K,
        "L": L,
        "D": D,
        "AI": AI,
        "H": H,
        "TFP": A,
        "Tăng trưởng GDP (%)": np.nan,
    }]

    previous_y = float(BASE["Y"])

    for year in years:
        policy_k = (
            annual_budget_trillion
            * shares[0]
        )
        policy_d = (
            annual_budget_trillion
            * shares[1]
        )
        policy_ai = (
            annual_budget_trillion
            * shares[2]
        )
        policy_h = (
            annual_budget_trillion
            * shares[3]
        )

        # Xu hướng cơ sở cộng với tác động của gói chính sách.
        K = (
            K * 1.050
            + 4.0 * policy_k
        )
        L = (
            L * 1.004
            + 0.0008 * policy_h
        )
        D = (
            D
            + 1.25
            + 0.070 * policy_d
        )
        AI = (
            AI
            + 4.5
            + 0.40 * policy_ai
        )
        H = (
            H
            + 0.65
            + 0.085 * policy_h
        )

        tfp_growth = (
            0.010
            + 0.00020 * D
            + 0.00010 * AI
            + 0.00025 * H
        )

        A = A * (
            1.0 + tfp_growth
        )

        y = cobb_douglas(
            A=A,
            K=K,
            L=L,
            D=D,
            AI=AI,
            H=H,
        )

        growth = (
            y / previous_y
            - 1.0
        ) * 100.0

        rows.append({
            "Năm": int(year),
            "GDP": y,
            "K": K,
            "L": L,
            "D": D,
            "AI": AI,
            "H": H,
            "TFP": A,
            "Tăng trưởng GDP (%)": growth,
        })

        previous_y = y

    return pd.DataFrame(rows)
