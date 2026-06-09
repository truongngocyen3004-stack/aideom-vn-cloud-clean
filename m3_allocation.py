"""M3 - Phân bổ ngân sách theo vùng và hạng mục."""

from __future__ import annotations

import numpy as np
import pandas as pd


REGIONS = [
    "Trung du miền núi phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ + DH Trung Bộ",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long",
]

ITEMS = [
    "Vốn vật chất",
    "Số hóa",
    "AI",
    "Nhân lực số",
]

BASE_WEIGHTS = np.array([
    [1.40, 1.25, 0.55, 1.30],
    [0.80, 0.95, 1.40, 0.90],
    [1.10, 1.05, 0.85, 1.15],
    [1.50, 1.35, 0.45, 1.40],
    [0.70, 0.85, 1.55, 0.80],
    [1.20, 1.20, 0.65, 1.25],
], dtype=float)


def allocate_budget(
    total_budget: float,
    shares: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phân bổ ngân sách theo 6 vùng x 4 hạng mục.

    Mỗi cột được phân bổ theo trọng số nhu cầu/hấp thụ và luôn
    có tổng đúng bằng ngân sách của hạng mục tương ứng.
    """

    shares = np.asarray(
        shares,
        dtype=float,
    )

    if total_budget <= 0:
        raise ValueError(
            "Tổng ngân sách phải lớn hơn 0."
        )

    if shares.shape != (4,):
        raise ValueError(
            "shares phải có bốn phần tử."
        )

    if not np.isclose(
        shares.sum(),
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Tổng tỷ trọng phải bằng 1."
        )

    column_weights = (
        BASE_WEIGHTS
        / BASE_WEIGHTS.sum(
            axis=0,
            keepdims=True,
        )
    )

    category_budgets = (
        total_budget
        * shares
    )

    matrix = (
        column_weights
        * category_budgets
    )

    allocation = pd.DataFrame(
        matrix,
        index=REGIONS,
        columns=ITEMS,
    )

    allocation[
        "Tổng ngân sách"
    ] = allocation.sum(
        axis=1
    )

    allocation = (
        allocation.reset_index()
        .rename(
            columns={
                "index": "Vùng"
            }
        )
    )

    category_summary = pd.DataFrame({
        "Hạng mục": ITEMS,
        "Tỷ trọng": shares,
        "Ngân sách": category_budgets,
    })

    return (
        allocation,
        category_summary,
    )
