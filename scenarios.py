"""Định nghĩa năm kịch bản chính sách của AIDEOM-VN."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ITEMS = ("K", "D", "AI", "H")
ITEM_NAMES = {
    "K": "Vốn vật chất",
    "D": "Số hóa",
    "AI": "Trí tuệ nhân tạo",
    "H": "Nhân lực số",
}

SCENARIOS = {
    "S1": {
        "name": "Truyền thống",
        "description": "Ưu tiên vốn vật chất, FDI, hạ tầng truyền thống và xuất khẩu.",
        "shares": np.array([0.70, 0.10, 0.10, 0.10], dtype=float),
    },
    "S2": {
        "name": "Số hóa nhanh",
        "description": "Tăng đầu tư chính phủ số, doanh nghiệp số và thanh toán số.",
        "shares": np.array([0.25, 0.45, 0.15, 0.15], dtype=float),
    },
    "S3": {
        "name": "AI dẫn dắt",
        "description": "Ưu tiên AI, dữ liệu lớn, bán dẫn và trung tâm dữ liệu.",
        "shares": np.array([0.20, 0.20, 0.45, 0.15], dtype=float),
    },
    "S4": {
        "name": "Bao trùm số",
        "description": "Ưu tiên vùng yếu, SME, giáo dục số và đào tạo lại lao động.",
        "shares": np.array([0.30, 0.20, 0.10, 0.40], dtype=float),
    },
}


@lru_cache(maxsize=1)
def optimize_balanced_shares() -> tuple[float, float, float, float]:
    """Tìm cơ cấu S5 bằng tối ưu hóa đa tiêu chí dạng rút gọn."""

    def objective(x: np.ndarray) -> float:
        k, d, ai, h = x

        # Lợi ích có tính lợi suất giảm dần để tránh nghiệm góc.
        growth = (
            0.90 * np.log1p(4.0 * k)
            + 1.15 * np.log1p(4.0 * d)
            + 1.35 * np.log1p(4.0 * ai)
            + 1.10 * np.log1p(4.0 * h)
        )

        inclusion = (
            0.45 * np.sqrt(max(d, 0.0))
            + 0.70 * np.sqrt(max(h, 0.0))
        )

        cyber_risk = max(
            0.0,
            0.70 * ai
            + 0.15 * d
            - 0.45 * h,
        )

        emissions = max(
            0.0,
            0.80 * k
            + 0.25 * ai
            - 0.20 * d
            - 0.10 * h,
        )

        dependency = max(
            0.0,
            0.55 * ai
            + 0.20 * k
            - 0.30 * h,
        )

        diversification = -np.sum(
            (x - 0.25) ** 2
        )

        score = (
            0.45 * growth
            + 0.25 * inclusion
            - 0.12 * cyber_risk
            - 0.10 * emissions
            - 0.08 * dependency
            + 0.12 * diversification
        )

        return -float(score)

    constraints = {
        "type": "eq",
        "fun": lambda x: float(np.sum(x) - 1.0),
    }

    bounds = [
        (0.15, 0.45),  # K
        (0.15, 0.45),  # D
        (0.10, 0.40),  # AI
        (0.15, 0.45),  # H
    ]

    result = minimize(
        objective,
        x0=np.array([0.27, 0.25, 0.23, 0.25]),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not result.success:
        # Phương án dự phòng vẫn cân bằng và tổng bằng một.
        return (0.27, 0.25, 0.22, 0.26)

    x = np.maximum(result.x, 0.0)
    x = x / x.sum()
    return tuple(float(value) for value in x)


def get_scenario_shares(code: str) -> np.ndarray:
    """Trả về vector tỷ trọng [K, D, AI, H] của một kịch bản."""

    normalized_code = code.strip().upper()

    if normalized_code == "S5":
        return np.asarray(
            optimize_balanced_shares(),
            dtype=float,
        )

    if normalized_code not in SCENARIOS:
        raise ValueError(
            f"Kịch bản không hợp lệ: {code}. "
            "Chọn một trong S1, S2, S3, S4, S5."
        )

    return SCENARIOS[
        normalized_code
    ]["shares"].copy()


def scenario_name(code: str) -> str:
    """Tên đầy đủ của kịch bản."""

    code = code.upper()

    if code == "S5":
        return "Tối ưu cân bằng"

    return str(
        SCENARIOS[code]["name"]
    )


def scenario_description(code: str) -> str:
    """Mô tả ngắn của kịch bản."""

    code = code.upper()

    if code == "S5":
        return (
            "Cơ cấu được xác định tự động bằng mô hình tối ưu cân bằng "
            "giữa tăng trưởng, bao trùm, môi trường và an ninh."
        )

    return str(
        SCENARIOS[code]["description"]
    )


def scenario_catalog() -> pd.DataFrame:
    """Tạo bảng danh mục năm kịch bản."""

    rows = []

    for code in ("S1", "S2", "S3", "S4", "S5"):
        shares = get_scenario_shares(code)

        rows.append({
            "Mã": code,
            "Kịch bản": scenario_name(code),
            "Mô tả": scenario_description(code),
            "K (%)": shares[0] * 100,
            "D (%)": shares[1] * 100,
            "AI (%)": shares[2] * 100,
            "H (%)": shares[3] * 100,
        })

    return pd.DataFrame(rows)
