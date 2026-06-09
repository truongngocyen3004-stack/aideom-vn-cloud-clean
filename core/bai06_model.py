from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CRITERIA = [
    "grdp_per_capita_million_VND",
    "fdi_registered_billion_USD",
    "digital_index_0_100",
    "ai_readiness_0_100",
    "trained_labor_pct",
    "rd_intensity_pct",
    "internet_penetration_pct",
    "gini_coef",
]

DISPLAY_NAMES = {
    "grdp_per_capita_million_VND":
        "GRDP/người",
    "fdi_registered_billion_USD":
        "FDI",
    "digital_index_0_100":
        "Digital Index",
    "ai_readiness_0_100":
        "AI Readiness",
    "trained_labor_pct":
        "LĐ qua đào tạo",
    "rd_intensity_pct":
        "R&D/GRDP",
    "internet_penetration_pct":
        "Internet",
    "gini_coef":
        "Gini",
}

IS_BENEFIT = np.array(
    [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ],
    dtype=bool,
)

EXPERT_WEIGHTS = np.array(
    [
        0.10,
        0.10,
        0.15,
        0.20,
        0.15,
        0.15,
        0.05,
        0.10,
    ],
    dtype=float,
)

AI_CRITERION_INDEX = 3


def load_region_data(
    csv_path: str | Path,
) -> pd.DataFrame:
    """Đọc và kiểm tra dữ liệu 6 vùng."""

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}"
        )

    data = pd.read_csv(path)

    required = [
        "region_code",
        "region_name_vi",
        *CRITERIA,
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "File dữ liệu thiếu các cột: "
            + ", ".join(missing)
        )

    data = data[required].copy()

    for column in CRITERIA:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if len(data) != 6:
        raise ValueError(
            f"Đề yêu cầu 6 vùng, dữ liệu hiện có {len(data)} vùng."
        )

    if data["region_code"].duplicated().any():
        raise ValueError(
            "Mã vùng bị trùng."
        )

    if (data[CRITERIA] <= 0).any().any():
        raise ValueError(
            "TOPSIS và Entropy yêu cầu dữ liệu đầu vào dương."
        )

    return data.reset_index(
        drop=True
    )


def validate_weights(
    weights: np.ndarray,
) -> np.ndarray:
    """Kiểm tra và chuẩn hóa trọng số về tổng bằng 1."""

    values = np.asarray(
        weights,
        dtype=float,
    )

    if values.shape != (
        len(CRITERIA),
    ):
        raise ValueError(
            f"Cần đúng {len(CRITERIA)} trọng số."
        )

    if np.any(values < 0):
        raise ValueError(
            "Trọng số không được âm."
        )

    total = float(
        values.sum()
    )

    if total <= 0:
        raise ValueError(
            "Tổng trọng số phải lớn hơn 0."
        )

    return values / total


def vector_normalize(
    matrix: np.ndarray,
) -> np.ndarray:
    """Chuẩn hóa vector theo TOPSIS."""

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    denominator = np.sqrt(
        np.square(matrix).sum(
            axis=0
        )
    )

    if np.any(
        np.isclose(
            denominator,
            0.0,
        )
    ):
        raise ValueError(
            "Có tiêu chí có chuẩn vector bằng 0."
        )

    return matrix / denominator


def topsis(
    data: pd.DataFrame,
    weights: np.ndarray = EXPERT_WEIGHTS,
    is_benefit: np.ndarray = IS_BENEFIT,
) -> dict[str, Any]:
    """Cài đặt TOPSIS từ đầu bằng numpy."""

    normalized_weights = (
        validate_weights(
            weights
        )
    )

    benefit_flags = np.asarray(
        is_benefit,
        dtype=bool,
    )

    if benefit_flags.shape != (
        len(CRITERIA),
    ):
        raise ValueError(
            "is_benefit không đúng kích thước."
        )

    matrix = data[
        CRITERIA
    ].to_numpy(
        dtype=float
    )

    normalized_matrix = (
        vector_normalize(
            matrix
        )
    )

    weighted_matrix = (
        normalized_matrix
        * normalized_weights
    )

    ideal_positive = np.where(
        benefit_flags,
        weighted_matrix.max(
            axis=0
        ),
        weighted_matrix.min(
            axis=0
        ),
    )

    ideal_negative = np.where(
        benefit_flags,
        weighted_matrix.min(
            axis=0
        ),
        weighted_matrix.max(
            axis=0
        ),
    )

    distance_positive = np.sqrt(
        np.square(
            weighted_matrix
            - ideal_positive
        ).sum(
            axis=1
        )
    )

    distance_negative = np.sqrt(
        np.square(
            weighted_matrix
            - ideal_negative
        ).sum(
            axis=1
        )
    )

    denominator = (
        distance_positive
        + distance_negative
    )

    score = np.divide(
        distance_negative,
        denominator,
        out=np.zeros_like(
            distance_negative
        ),
        where=(
            denominator > 0
        ),
    )

    ranking = data[
        [
            "region_code",
            "region_name_vi",
        ]
    ].copy()

    ranking["S_plus"] = (
        distance_positive
    )

    ranking["S_minus"] = (
        distance_negative
    )

    ranking["TOPSIS_score"] = (
        score
    )

    ranking["Rank"] = (
        ranking[
            "TOPSIS_score"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    ranking = ranking.sort_values(
        [
            "TOPSIS_score",
            "region_name_vi",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    normalized_df = pd.DataFrame(
        normalized_matrix,
        columns=[
            DISPLAY_NAMES[column]
            for column in CRITERIA
        ],
    )

    normalized_df.insert(
        0,
        "Vùng",
        data[
            "region_name_vi"
        ].values,
    )

    weighted_df = pd.DataFrame(
        weighted_matrix,
        columns=[
            DISPLAY_NAMES[column]
            for column in CRITERIA
        ],
    )

    weighted_df.insert(
        0,
        "Vùng",
        data[
            "region_name_vi"
        ].values,
    )

    ideal_table = pd.DataFrame({
        "Tiêu chí": [
            DISPLAY_NAMES[column]
            for column in CRITERIA
        ],
        "Loại": [
            (
                "Lợi ích"
                if flag
                else "Chi phí"
            )
            for flag in benefit_flags
        ],
        "Trọng số": (
            normalized_weights
        ),
        "Lý tưởng dương A+": (
            ideal_positive
        ),
        "Lý tưởng âm A-": (
            ideal_negative
        ),
    })

    return {
        "weights":
            normalized_weights,
        "normalized_matrix":
            normalized_df,
        "weighted_matrix":
            weighted_df,
        "ideal_positive":
            ideal_positive,
        "ideal_negative":
            ideal_negative,
        "ideal_table":
            ideal_table,
        "ranking":
            ranking,
    }


def entropy_weights(
    data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Câu 6.4.2: tính trọng số khách quan bằng Entropy.

    Entropy đo mức độ phân tán thông tin của tiêu chí.
    Hướng lợi ích/chi phí được xử lý ở bước TOPSIS,
    không làm thay đổi công thức trọng số Entropy.
    """

    matrix = data[
        CRITERIA
    ].to_numpy(
        dtype=float
    )

    column_sum = matrix.sum(
        axis=0
    )

    proportion = np.divide(
        matrix,
        column_sum,
        out=np.zeros_like(
            matrix
        ),
        where=(
            column_sum > 0
        ),
    )

    n_alternatives = (
        matrix.shape[0]
    )

    k_value = (
        1.0
        / np.log(
            n_alternatives
        )
    )

    entropy = (
        -k_value
        * np.sum(
            proportion
            * np.log(
                proportion
                + 1e-12
            ),
            axis=0,
        )
    )

    divergence = (
        1.0
        - entropy
    )

    if np.isclose(
        divergence.sum(),
        0.0,
    ):
        weights = np.ones(
            len(CRITERIA)
        ) / len(CRITERIA)
    else:
        weights = (
            divergence
            / divergence.sum()
        )

    table = pd.DataFrame({
        "Tiêu chí": [
            DISPLAY_NAMES[column]
            for column in CRITERIA
        ],
        "Entropy E_j":
            entropy,
        "Mức phân biệt d_j":
            divergence,
        "Trọng số Entropy":
            weights,
    })

    return {
        "weights": weights,
        "proportion_matrix":
            pd.DataFrame(
                proportion,
                columns=[
                    DISPLAY_NAMES[
                        column
                    ]
                    for column
                    in CRITERIA
                ],
                index=data[
                    "region_name_vi"
                ],
            ),
        "table": table,
    }


def compare_rankings(
    expert_ranking: pd.DataFrame,
    entropy_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """So sánh thay đổi thứ hạng giữa hai bộ trọng số."""

    expert = expert_ranking[
        [
            "region_code",
            "region_name_vi",
            "TOPSIS_score",
            "Rank",
        ]
    ].rename(
        columns={
            "TOPSIS_score":
                "Điểm chuyên gia",
            "Rank":
                "Hạng chuyên gia",
        }
    )

    entropy = entropy_ranking[
        [
            "region_code",
            "TOPSIS_score",
            "Rank",
        ]
    ].rename(
        columns={
            "TOPSIS_score":
                "Điểm Entropy",
            "Rank":
                "Hạng Entropy",
        }
    )

    comparison = expert.merge(
        entropy,
        on="region_code",
        how="inner",
    )

    comparison[
        "Thay đổi hạng"
    ] = (
        comparison[
            "Hạng Entropy"
        ]
        - comparison[
            "Hạng chuyên gia"
        ]
    )

    comparison[
        "|Thay đổi hạng|"
    ] = comparison[
        "Thay đổi hạng"
    ].abs()

    return comparison.sort_values(
        [
            "|Thay đổi hạng|",
            "Hạng chuyên gia",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


def rescale_ai_weight(
    base_weights: np.ndarray,
    ai_weight: float,
) -> np.ndarray:
    """
    Đặt trọng số AI ở một mức mới và co giãn tỷ lệ
    các trọng số còn lại để tổng vẫn bằng 1.
    """

    if not 0 <= ai_weight < 1:
        raise ValueError(
            "Trọng số AI phải nằm trong [0,1)."
        )

    base = validate_weights(
        base_weights
    )

    other_indices = [
        index
        for index in range(
            len(base)
        )
        if index
        != AI_CRITERION_INDEX
    ]

    other_sum = float(
        base[
            other_indices
        ].sum()
    )

    adjusted = np.zeros_like(
        base
    )

    adjusted[
        AI_CRITERION_INDEX
    ] = ai_weight

    adjusted[
        other_indices
    ] = (
        base[
            other_indices
        ]
        / other_sum
        * (
            1.0
            - ai_weight
        )
    )

    return adjusted


def ai_weight_sensitivity(
    data: pd.DataFrame,
    base_weights: np.ndarray = EXPERT_WEIGHTS,
    ai_values: Iterable[float] = np.arange(
        0.10,
        0.401,
        0.05,
    ),
) -> dict[str, Any]:
    """Câu 6.4.3: phân tích độ nhạy trọng số AI."""

    summary_rows = []
    rank_rows = []
    score_rows = []

    for ai_weight in ai_values:
        weights = rescale_ai_weight(
            base_weights,
            float(ai_weight),
        )

        result = topsis(
            data,
            weights,
        )

        ranking = result[
            "ranking"
        ]

        top3 = ranking.head(
            3
        )[
            "region_name_vi"
        ].tolist()

        summary_rows.append({
            "Trọng số AI":
                float(ai_weight),
            "Top 1":
                top3[0],
            "Top 2":
                top3[1],
            "Top 3":
                top3[2],
            "Cấu hình Top-3":
                " | ".join(
                    top3
                ),
        })

        for _, row in ranking.iterrows():
            rank_rows.append({
                "Trọng số AI":
                    float(
                        ai_weight
                    ),
                "Vùng":
                    row[
                        "region_name_vi"
                    ],
                "Hạng":
                    int(
                        row[
                            "Rank"
                        ]
                    ),
            })

            score_rows.append({
                "Trọng số AI":
                    float(
                        ai_weight
                    ),
                "Vùng":
                    row[
                        "region_name_vi"
                    ],
                "TOPSIS_score":
                    float(
                        row[
                            "TOPSIS_score"
                        ]
                    ),
            })

    summary = pd.DataFrame(
        summary_rows
    )

    ranks = pd.DataFrame(
        rank_rows
    )

    scores = pd.DataFrame(
        score_rows
    )

    top3_sets = summary[
        "Cấu hình Top-3"
    ].nunique()

    return {
        "summary":
            summary,
        "ranks":
            ranks,
        "scores":
            scores,
        "top3_configurations":
            int(
                top3_sets
            ),
    }


def ratio_to_saaty(
    ratio: float,
) -> float:
    """
    Chuyển tỷ số ưu tiên thành thang Saaty 1-9 gần nhất.
    """

    if ratio <= 0:
        raise ValueError(
            "Tỷ số AHP phải dương."
        )

    if ratio >= 1:
        return float(
            min(
                9,
                max(
                    1,
                    int(
                        round(
                            ratio
                        )
                    ),
                ),
            )
        )

    reciprocal = ratio_to_saaty(
        1.0 / ratio
    )

    return (
        1.0
        / reciprocal
    )


def build_ahp_matrix(
    seed_weights: np.ndarray = EXPERT_WEIGHTS,
) -> np.ndarray:
    """
    Xây dựng ma trận so sánh cặp AHP đơn giản
    từ mức ưu tiên chuyên gia, làm tròn theo thang Saaty.
    """

    seed = validate_weights(
        seed_weights
    )

    size = len(seed)

    matrix = np.ones(
        (
            size,
            size,
        ),
        dtype=float,
    )

    for row in range(size):
        for column in range(
            row + 1,
            size,
        ):
            value = ratio_to_saaty(
                seed[row]
                / seed[column]
            )

            matrix[
                row,
                column,
            ] = value

            matrix[
                column,
                row,
            ] = 1.0 / value

    return matrix


def ahp_weights(
    pairwise_matrix: np.ndarray,
) -> dict[str, Any]:
    """Tính vector trọng số AHP và tỷ lệ nhất quán CR."""

    matrix = np.asarray(
        pairwise_matrix,
        dtype=float,
    )

    size = len(CRITERIA)

    if matrix.shape != (
        size,
        size,
    ):
        raise ValueError(
            "Ma trận AHP không đúng kích thước."
        )

    if np.any(
        matrix <= 0
    ):
        raise ValueError(
            "Các phần tử AHP phải dương."
        )

    eigenvalues, eigenvectors = (
        np.linalg.eig(
            matrix
        )
    )

    max_index = int(
        np.argmax(
            eigenvalues.real
        )
    )

    lambda_max = float(
        eigenvalues[
            max_index
        ].real
    )

    weights = np.abs(
        eigenvectors[
            :,
            max_index,
        ].real
    )

    weights = (
        weights
        / weights.sum()
    )

    consistency_index = (
        (
            lambda_max
            - size
        )
        / (
            size
            - 1
        )
    )

    random_index = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }[size]

    consistency_ratio = (
        consistency_index
        / random_index
        if random_index > 0
        else 0.0
    )

    weight_table = pd.DataFrame({
        "Tiêu chí": [
            DISPLAY_NAMES[
                column
            ]
            for column
            in CRITERIA
        ],
        "Trọng số AHP":
            weights,
    })

    pairwise_df = pd.DataFrame(
        matrix,
        columns=[
            DISPLAY_NAMES[
                column
            ]
            for column
            in CRITERIA
        ],
        index=[
            DISPLAY_NAMES[
                column
            ]
            for column
            in CRITERIA
        ],
    )

    return {
        "weights":
            weights,
        "lambda_max":
            lambda_max,
        "consistency_index":
            float(
                consistency_index
            ),
        "consistency_ratio":
            float(
                consistency_ratio
            ),
        "is_consistent":
            bool(
                consistency_ratio
                <= 0.10
            ),
        "weight_table":
            weight_table,
        "pairwise_matrix":
            pairwise_df,
    }


def correlation_diagnostics(
    data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Kiểm tra tương quan giữa AI Readiness và Internet penetration.
    """

    correlation = float(
        data[
            [
                "ai_readiness_0_100",
                "internet_penetration_pct",
            ]
        ]
        .corr()
        .iloc[
            0,
            1,
        ]
    )

    r_squared = (
        correlation ** 2
    )

    vif = (
        1.0
        / (
            1.0
            - r_squared
        )
        if r_squared < 1
        else np.inf
    )

    return {
        "correlation":
            correlation,
        "r_squared":
            float(
                r_squared
            ),
        "vif":
            float(
                vif
            ),
    }


def run_full_bai06(
    csv_path: str | Path,
    expert_weights: np.ndarray = EXPERT_WEIGHTS,
) -> dict[str, Any]:
    """Chạy đầy đủ câu 6.4.1–6.4.4."""

    data = load_region_data(
        csv_path
    )

    expert = topsis(
        data,
        expert_weights,
    )

    entropy = entropy_weights(
        data
    )

    entropy_topsis = topsis(
        data,
        entropy["weights"],
    )

    comparison = compare_rankings(
        expert[
            "ranking"
        ],
        entropy_topsis[
            "ranking"
        ],
    )

    sensitivity = (
        ai_weight_sensitivity(
            data,
            expert_weights,
        )
    )

    pairwise_matrix = (
        build_ahp_matrix(
            expert_weights
        )
    )

    ahp = ahp_weights(
        pairwise_matrix
    )

    ahp_topsis = topsis(
        data,
        ahp["weights"],
    )

    correlation = (
        correlation_diagnostics(
            data
        )
    )

    return {
        "data": data,
        "expert": expert,
        "entropy":
            entropy,
        "entropy_topsis":
            entropy_topsis,
        "comparison":
            comparison,
        "sensitivity":
            sensitivity,
        "ahp": ahp,
        "ahp_topsis":
            ahp_topsis,
        "correlation":
            correlation,
    }
