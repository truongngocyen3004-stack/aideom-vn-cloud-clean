"""M4 - Mô phỏng việc làm ròng theo tám ngành."""

from __future__ import annotations

import numpy as np
import pandas as pd


LABOR_DATA = pd.DataFrame({
    "Ngành": [
        "Nông-Lâm-Thủy sản",
        "CN chế biến chế tạo",
        "Xây dựng",
        "Bán buôn-bán lẻ",
        "Tài chính-Ngân hàng",
        "Logistics-Vận tải",
        "CNTT-Truyền thông",
        "Giáo dục-Đào tạo",
    ],
    "Lao động (triệu)": [
        13.20, 11.50, 4.80, 7.80,
        0.55, 1.95, 0.62, 2.15,
    ],
    "Risk": [
        0.18, 0.42, 0.25, 0.38,
        0.52, 0.35, 0.28, 0.22,
    ],
    "a1": [
        8.5, 32.5, 12.8, 22.4,
        45.8, 28.5, 62.5, 18.5,
    ],
    "b1": [
        45.0, 28.0, 35.0, 32.0,
        22.0, 30.0, 20.0, 55.0,
    ],
    "c1": [
        5.2, 62.4, 18.5, 48.2,
        72.5, 42.8, 32.5, 12.5,
    ],
    "d1": [
        50.0, 32.0, 42.0, 38.0,
        26.0, 36.0, 24.0, 62.0,
    ],
    "AI hấp thụ": [
        0.35, 0.90, 0.45, 0.60,
        0.80, 0.70, 1.00, 0.40,
    ],
})


def simulate_labor_market(
    total_budget: float,
    shares: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Mô phỏng việc làm mới, việc làm nâng cấp và việc làm bị thay thế.
    """

    shares = np.asarray(
        shares,
        dtype=float,
    )

    df = LABOR_DATA.copy()

    ai_budget = (
        total_budget
        * shares[2]
    )
    human_budget = (
        total_budget
        * shares[3]
    )

    ai_weights = (
        df["AI hấp thụ"]
        * (
            0.40
            + df["a1"]
            / df["a1"].max()
        )
    )
    ai_weights = (
        ai_weights
        / ai_weights.sum()
    )

    human_weights = (
        df["Lao động (triệu)"]
        * (
            0.60
            + df["Risk"]
        )
    )
    human_weights = (
        human_weights
        / human_weights.sum()
    )

    df["Đầu tư AI"] = (
        ai_budget
        * ai_weights
    )

    df["Đầu tư nhân lực"] = (
        human_budget
        * human_weights
    )

    # Hệ số 0,10 quy đổi gói chính sách toàn kỳ về quy mô việc làm.
    scale = 0.10

    df["Việc làm mới"] = (
        scale
        * df["a1"]
        * df["Đầu tư AI"]
    )

    df["Việc làm nâng cấp"] = (
        scale
        * df["b1"]
        * df["Đầu tư nhân lực"]
    )

    df["Việc làm bị thay thế"] = (
        scale
        * df["c1"]
        * df["Risk"]
        * df["Đầu tư AI"]
    )

    df["Năng lực đào tạo lại"] = (
        scale
        * df["d1"]
        * df["Đầu tư nhân lực"]
    )

    df["Đã đào tạo lại"] = np.minimum(
        df["Việc làm bị thay thế"],
        df["Năng lực đào tạo lại"],
    )

    df["Khoảng trống đào tạo lại"] = np.maximum(
        df["Việc làm bị thay thế"]
        - df["Năng lực đào tạo lại"],
        0.0,
    )

    df["NetJob"] = (
        df["Việc làm mới"]
        + df["Việc làm nâng cấp"]
        - df["Việc làm bị thay thế"]
    )

    summary = {
        "new_jobs": float(
            df["Việc làm mới"].sum()
        ),
        "upgraded_jobs": float(
            df["Việc làm nâng cấp"].sum()
        ),
        "displaced_jobs": float(
            df["Việc làm bị thay thế"].sum()
        ),
        "retraining_capacity": float(
            df["Năng lực đào tạo lại"].sum()
        ),
        "retraining_gap": float(
            df["Khoảng trống đào tạo lại"].sum()
        ),
        "net_jobs": float(
            df["NetJob"].sum()
        ),
    }

    return df, summary
