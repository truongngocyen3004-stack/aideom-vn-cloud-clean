from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CRITERIA = [
    "growth_rate_2024_pct",
    "labor_productivity_million_VND",
    "spillover_coef_0_1",
    "export_billion_USD",
    "labor_million",
    "ai_readiness_0_100",
    "automation_risk_pct",
]

GOOD_CRITERIA = CRITERIA[:-1]
RISK_CRITERION = "automation_risk_pct"

NORMALIZED_COLUMNS = {
    "growth_rate_2024_pct": "Growth_norm",
    "labor_productivity_million_VND": "Productivity_norm",
    "spillover_coef_0_1": "Spillover_norm",
    "export_billion_USD": "Export_norm",
    "labor_million": "Employment_norm",
    "ai_readiness_0_100": "AIReadiness_norm",
    "automation_risk_pct": "Safety_norm",
}

DISPLAY_CRITERIA = {
    "Growth_norm": "Tăng trưởng",
    "Productivity_norm": "Năng suất",
    "Spillover_norm": "Lan tỏa",
    "Export_norm": "Xuất khẩu",
    "Employment_norm": "Việc làm",
    "AIReadiness_norm": "AI Readiness",
    "Safety_norm": "An toàn trước TĐH",
}

DEFAULT_RAW_WEIGHTS = {
    "Growth_norm": 0.15,
    "Productivity_norm": 0.15,
    "Spillover_norm": 0.20,
    "Export_norm": 0.15,
    "Employment_norm": 0.10,
    "AIReadiness_norm": 0.20,
    "Safety_norm": 0.15,
}

GROWTH_WEIGHTS = {
    "Growth_norm": 0.25,
    "Productivity_norm": 0.25,
    "Spillover_norm": 0.10,
    "Export_norm": 0.25,
    "Employment_norm": 0.05,
    "AIReadiness_norm": 0.10,
    "Safety_norm": 0.00,
}

INCLUSIVE_WEIGHTS = {
    "Growth_norm": 0.10,
    "Productivity_norm": 0.10,
    "Spillover_norm": 0.25,
    "Export_norm": 0.05,
    "Employment_norm": 0.25,
    "AIReadiness_norm": 0.10,
    "Safety_norm": 0.15,
}


def load_sector_data(csv_path: str | Path) -> pd.DataFrame:
    """Đọc và kiểm tra dữ liệu 10 ngành Việt Nam."""

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}"
        )

    df = pd.read_csv(path)

    required = ["sector_name_vi", *CRITERIA]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "File dữ liệu thiếu các cột: "
            + ", ".join(missing)
        )

    df = df[required].copy()

    for column in CRITERIA:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if df["sector_name_vi"].duplicated().any():
        raise ValueError("Tên ngành bị trùng.")

    if len(df) != 10:
        raise ValueError(
            f"Đề bài yêu cầu 10 ngành, file hiện có {len(df)} ngành."
        )

    return df.reset_index(drop=True)


def minmax_good(series: pd.Series) -> pd.Series:
    """Chuẩn hóa tiêu chí lợi ích về [0, 1]."""

    low = float(series.min())
    high = float(series.max())

    if np.isclose(high, low):
        return pd.Series(
            np.ones(len(series)),
            index=series.index,
            dtype=float,
        )

    return (series - low) / (high - low)


def minmax_bad_reversed(series: pd.Series) -> pd.Series:
    """
    Đảo chiều tiêu chí chi phí/rủi ro.

    Điểm càng cao nghĩa là rủi ro tự động hóa càng thấp.
    """

    low = float(series.min())
    high = float(series.max())

    if np.isclose(high, low):
        return pd.Series(
            np.ones(len(series)),
            index=series.index,
            dtype=float,
        )

    return (high - series) / (high - low)


def normalize_sector_matrix(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Câu 3.4.1: chuẩn hóa min-max đủ 7 tiêu chí."""

    normalized = pd.DataFrame({
        "sector_name_vi": data["sector_name_vi"],
    })

    for column in GOOD_CRITERIA:
        normalized[
            NORMALIZED_COLUMNS[column]
        ] = minmax_good(data[column])

    normalized[
        NORMALIZED_COLUMNS[RISK_CRITERION]
    ] = minmax_bad_reversed(
        data[RISK_CRITERION]
    )

    return normalized


def normalize_weights(
    weights: dict[str, float],
) -> dict[str, float]:
    """
    Chuẩn hóa trọng số về tổng bằng 1.

    Bộ trọng số mặc định trong đề có tổng 1,10; chia đều theo tổng
    không làm thay đổi thứ hạng, nhưng đưa Priority về thang dễ đọc.
    """

    expected = set(DISPLAY_CRITERIA.keys())

    if set(weights.keys()) != expected:
        missing = expected.difference(weights.keys())
        extra = set(weights.keys()).difference(expected)
        raise ValueError(
            f"Sai tên trọng số. Thiếu={sorted(missing)}, thừa={sorted(extra)}"
        )

    values = np.array(
        list(weights.values()),
        dtype=float,
    )

    if np.any(values < 0):
        raise ValueError("Trọng số không được âm.")

    total = float(values.sum())

    if total <= 0:
        raise ValueError("Tổng trọng số phải lớn hơn 0.")

    return {
        key: float(value / total)
        for key, value in weights.items()
    }


def calculate_priority(
    data: pd.DataFrame,
    weights: dict[str, float] = DEFAULT_RAW_WEIGHTS,
) -> dict[str, Any]:
    """Câu 3.4.2: tính Priority và xếp hạng 10 ngành."""

    normalized = normalize_sector_matrix(data)
    normalized_weights = normalize_weights(weights)

    result = data.copy()

    for column in DISPLAY_CRITERIA:
        result[column] = normalized[column]

    contribution_columns = []

    for column, weight in normalized_weights.items():
        contribution_column = f"Contribution_{column}"
        result[contribution_column] = result[column] * weight
        contribution_columns.append(contribution_column)

    result["Priority"] = result[contribution_columns].sum(axis=1)

    result["Rank"] = (
        result["Priority"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    result = result.sort_values(
        ["Priority", "sector_name_vi"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return {
        "ranking": result,
        "normalized": normalized,
        "weights_raw": dict(weights),
        "weights_normalized": normalized_weights,
        "weight_sum_raw": float(sum(weights.values())),
        "contribution_columns": contribution_columns,
    }


def ai_weight_sensitivity(
    data: pd.DataFrame,
    base_weights: dict[str, float] = DEFAULT_RAW_WEIGHTS,
    ai_values: Iterable[float] = np.arange(0.05, 0.401, 0.05),
) -> dict[str, pd.DataFrame]:
    """
    Câu 3.4.3:
    thay a6 từ 0,05 đến 0,40; các trọng số còn lại giữ nguyên
    rồi chuẩn hóa lại tổng về 1.
    """

    summary_rows = []
    rank_rows = []
    score_rows = []

    for ai_weight in ai_values:
        temp_weights = dict(base_weights)
        temp_weights["AIReadiness_norm"] = float(ai_weight)

        result = calculate_priority(
            data=data,
            weights=temp_weights,
        )

        ranking = result["ranking"]
        top3 = ranking.head(3)["sector_name_vi"].tolist()

        summary_rows.append({
            "Trọng số AI ban đầu": round(float(ai_weight), 2),
            "Trọng số AI sau chuẩn hóa": result[
                "weights_normalized"
            ]["AIReadiness_norm"],
            "Top 1": top3[0],
            "Top 2": top3[1],
            "Top 3": top3[2],
            "Top-3": " | ".join(top3),
        })

        for _, row in ranking.iterrows():
            rank_rows.append({
                "Trọng số AI ban đầu": round(float(ai_weight), 2),
                "Trọng số AI sau chuẩn hóa": result[
                    "weights_normalized"
                ]["AIReadiness_norm"],
                "Ngành": row["sector_name_vi"],
                "Xếp hạng": int(row["Rank"]),
            })

            score_rows.append({
                "Trọng số AI ban đầu": round(float(ai_weight), 2),
                "Trọng số AI sau chuẩn hóa": result[
                    "weights_normalized"
                ]["AIReadiness_norm"],
                "Ngành": row["sector_name_vi"],
                "Priority": float(row["Priority"]),
            })

    summary = pd.DataFrame(summary_rows)
    ranks = pd.DataFrame(rank_rows)
    scores = pd.DataFrame(score_rows)

    return {
        "summary": summary,
        "ranks": ranks,
        "scores": scores,
        "top3_configurations": int(summary["Top-3"].nunique()),
    }


def compare_policy_orientations(
    data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Câu 3.4.4: so sánh tăng trưởng và bao trùm."""

    growth = calculate_priority(
        data,
        GROWTH_WEIGHTS,
    )["ranking"]

    inclusive = calculate_priority(
        data,
        INCLUSIVE_WEIGHTS,
    )["ranking"]

    comparison = pd.DataFrame({
        "Ngành": data["sector_name_vi"],
    })

    comparison = comparison.merge(
        growth[
            ["sector_name_vi", "Priority", "Rank"]
        ],
        left_on="Ngành",
        right_on="sector_name_vi",
        how="left",
    ).drop(columns=["sector_name_vi"])

    comparison = comparison.rename(
        columns={
            "Priority": "Priority - Tăng trưởng",
            "Rank": "Rank - Tăng trưởng",
        }
    )

    comparison = comparison.merge(
        inclusive[
            ["sector_name_vi", "Priority", "Rank"]
        ],
        left_on="Ngành",
        right_on="sector_name_vi",
        how="left",
    ).drop(columns=["sector_name_vi"])

    comparison = comparison.rename(
        columns={
            "Priority": "Priority - Bao trùm",
            "Rank": "Rank - Bao trùm",
        }
    )

    comparison["Thay đổi hạng (Bao trùm - Tăng trưởng)"] = (
        comparison["Rank - Bao trùm"]
        - comparison["Rank - Tăng trưởng"]
    )

    return {
        "growth_ranking": growth,
        "inclusive_ranking": inclusive,
        "comparison": comparison,
        "growth_weights": pd.DataFrame({
            "Tiêu chí": [
                DISPLAY_CRITERIA[key]
                for key in GROWTH_WEIGHTS
            ],
            "Trọng số": list(GROWTH_WEIGHTS.values()),
        }),
        "inclusive_weights": pd.DataFrame({
            "Tiêu chí": [
                DISPLAY_CRITERIA[key]
                for key in INCLUSIVE_WEIGHTS
            ],
            "Trọng số": list(INCLUSIVE_WEIGHTS.values()),
        }),
    }


def mining_diagnostics(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Tách hồ sơ của Khai khoáng để giải thích câu 3.5b."""

    mining = ranking.loc[
        ranking["sector_name_vi"] == "Khai khoáng"
    ]

    if mining.empty:
        return pd.DataFrame()

    row = mining.iloc[0]

    return pd.DataFrame({
        "Chỉ tiêu": [
            "Năng suất chuẩn hóa",
            "Tăng trưởng chuẩn hóa",
            "Lan tỏa chuẩn hóa",
            "Xuất khẩu chuẩn hóa",
            "Việc làm chuẩn hóa",
            "AI Readiness chuẩn hóa",
            "An toàn trước TĐH",
            "Priority",
            "Xếp hạng",
        ],
        "Giá trị": [
            row["Productivity_norm"],
            row["Growth_norm"],
            row["Spillover_norm"],
            row["Export_norm"],
            row["Employment_norm"],
            row["AIReadiness_norm"],
            row["Safety_norm"],
            row["Priority"],
            row["Rank"],
        ],
    })


def run_full_bai03(
    csv_path: str | Path,
    weights: dict[str, float] = DEFAULT_RAW_WEIGHTS,
) -> dict[str, Any]:
    """Chạy toàn bộ yêu cầu 3.4.1-3.4.4."""

    data = load_sector_data(csv_path)
    default_result = calculate_priority(data, weights)
    sensitivity = ai_weight_sensitivity(
        data,
        base_weights=weights,
    )
    orientations = compare_policy_orientations(data)
    mining = mining_diagnostics(
        default_result["ranking"]
    )

    return {
        "data": data,
        "default": default_result,
        "sensitivity": sensitivity,
        "orientations": orientations,
        "mining": mining,
    }
