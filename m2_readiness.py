"""M2 - Đánh giá sẵn sàng số và AI của sáu vùng."""

from __future__ import annotations

import numpy as np
import pandas as pd


REGION_DATA = pd.DataFrame({
    "Vùng": [
        "Trung du miền núi phía Bắc",
        "Đồng bằng sông Hồng",
        "Bắc Trung Bộ + DH Trung Bộ",
        "Tây Nguyên",
        "Đông Nam Bộ",
        "Đồng bằng sông Cửu Long",
    ],
    "Digital Index 2025": [38.0, 78.0, 55.0, 32.0, 82.0, 48.0],
    "AI Readiness 2025": [22.0, 68.0, 40.0, 18.0, 75.0, 30.0],
    "Lao động đào tạo 2025": [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
    "Internet 2025": [72.0, 92.0, 84.0, 68.0, 94.0, 78.0],
    "Hệ số hấp thụ": [0.85, 1.15, 0.95, 0.78, 1.20, 0.88],
})


def _minmax(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())

    if np.isclose(
        maximum,
        minimum,
    ):
        return pd.Series(
            np.ones(len(series)),
            index=series.index,
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


def assess_regional_readiness(
    total_budget: float,
    shares: np.ndarray,
) -> pd.DataFrame:
    """
    Dự báo chỉ số sẵn sàng số của sáu vùng đến năm 2030.
    """

    shares = np.asarray(
        shares,
        dtype=float,
    )

    df = REGION_DATA.copy()

    budget_trillion = (
        total_budget / 1000.0
    )

    digital_gap = (
        100.0
        - df["Digital Index 2025"]
    )

    ai_gap = (
        100.0
        - df["AI Readiness 2025"]
    )

    human_gap = (
        60.0
        - df["Lao động đào tạo 2025"]
    ).clip(lower=0.0)

    digital_weights = (
        digital_gap
        * df["Hệ số hấp thụ"]
    )
    digital_weights = (
        digital_weights
        / digital_weights.sum()
    )

    ai_weights = (
        (
            0.55
            * df["AI Readiness 2025"]
            + 0.45
            * df["Hệ số hấp thụ"]
            * 50.0
        )
    )
    ai_weights = (
        ai_weights
        / ai_weights.sum()
    )

    human_weights = (
        human_gap
        * np.sqrt(
            df["Hệ số hấp thụ"]
        )
    )
    human_weights = (
        human_weights
        / human_weights.sum()
    )

    digital_increment = (
        5.0
        + 0.90
        * budget_trillion
        * shares[1]
        * digital_weights
    )

    ai_increment = (
        4.0
        + 1.20
        * budget_trillion
        * shares[2]
        * ai_weights
    )

    human_increment = (
        2.5
        + 0.55
        * budget_trillion
        * shares[3]
        * human_weights
    )

    internet_increment = (
        2.0
        + 0.25
        * digital_increment
    )

    df["Digital Index 2030"] = np.clip(
        df["Digital Index 2025"]
        + digital_increment,
        0.0,
        100.0,
    )

    df["AI Readiness 2030"] = np.clip(
        df["AI Readiness 2025"]
        + ai_increment,
        0.0,
        100.0,
    )

    df["Lao động đào tạo 2030"] = np.clip(
        df["Lao động đào tạo 2025"]
        + human_increment,
        0.0,
        100.0,
    )

    df["Internet 2030"] = np.clip(
        df["Internet 2025"]
        + internet_increment,
        0.0,
        100.0,
    )

    df["Điểm sẵn sàng 2030"] = (
        0.30
        * _minmax(
            df["Digital Index 2030"]
        )
        + 0.30
        * _minmax(
            df["AI Readiness 2030"]
        )
        + 0.20
        * _minmax(
            df["Lao động đào tạo 2030"]
        )
        + 0.20
        * _minmax(
            df["Internet 2030"]
        )
    )

    df["Xếp hạng"] = (
        df["Điểm sẵn sàng 2030"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return (
        df.sort_values(
            ["Xếp hạng", "Vùng"]
        )
        .reset_index(drop=True)
    )
