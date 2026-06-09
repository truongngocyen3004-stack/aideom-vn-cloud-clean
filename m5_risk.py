"""M5 - Đánh giá rủi ro cyber, môi trường, phụ thuộc và xã hội."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assess_risks(
    shares: np.ndarray,
    regional_allocation: pd.DataFrame,
    labor_summary: dict[str, float],
    cyber_threshold: float = 60.0,
    emission_threshold: float = 60.0,
    dependency_threshold: float = 60.0,
) -> tuple[pd.DataFrame, list[str], float]:
    """
    Tính các chỉ số rủi ro trên thang 0-100 và tạo cảnh báo.
    """

    shares = np.asarray(
        shares,
        dtype=float,
    )

    k, d, ai, h = shares

    cyber = np.clip(
        18.0
        + 105.0
        * (
            0.72 * ai
            + 0.16 * d
            - 0.38 * h
        ),
        0.0,
        100.0,
    )

    emissions = np.clip(
        15.0
        + 95.0
        * (
            0.78 * k
            + 0.24 * ai
            - 0.20 * d
            - 0.12 * h
        ),
        0.0,
        100.0,
    )

    dependency = np.clip(
        12.0
        + 100.0
        * (
            0.58 * ai
            + 0.24 * k
            - 0.30 * h
        ),
        0.0,
        100.0,
    )

    region_totals = (
        regional_allocation[
            "Tổng ngân sách"
        ]
        .to_numpy(dtype=float)
    )

    regional_disparity = np.clip(
        100.0
        * np.std(region_totals)
        / max(
            np.mean(region_totals),
            1e-9,
        ),
        0.0,
        100.0,
    )

    displaced = float(
        labor_summary[
            "displaced_jobs"
        ]
    )

    retraining_gap = float(
        labor_summary[
            "retraining_gap"
        ]
    )

    labor_risk = np.clip(
        100.0
        * retraining_gap
        / max(
            displaced,
            1.0,
        ),
        0.0,
        100.0,
    )

    table = pd.DataFrame({
        "Rủi ro": [
            "An ninh mạng",
            "Phát thải",
            "Phụ thuộc công nghệ",
            "Chênh lệch vùng",
            "Khoảng trống đào tạo lại",
        ],
        "Điểm": [
            cyber,
            emissions,
            dependency,
            regional_disparity,
            labor_risk,
        ],
        "Ngưỡng cảnh báo": [
            cyber_threshold,
            emission_threshold,
            dependency_threshold,
            35.0,
            20.0,
        ],
    })

    table["Trạng thái"] = np.where(
        table["Điểm"]
        > table["Ngưỡng cảnh báo"],
        "Cảnh báo",
        "Trong giới hạn",
    )

    alerts: list[str] = []

    for _, row in table.iterrows():
        if row["Trạng thái"] == "Cảnh báo":
            alerts.append(
                f'{row["Rủi ro"]}: {row["Điểm"]:.1f} '
                f'> ngưỡng {row["Ngưỡng cảnh báo"]:.1f}.'
            )

    overall_risk = float(
        0.25 * cyber
        + 0.20 * emissions
        + 0.20 * dependency
        + 0.15 * regional_disparity
        + 0.20 * labor_risk
    )

    return table, alerts, overall_risk
