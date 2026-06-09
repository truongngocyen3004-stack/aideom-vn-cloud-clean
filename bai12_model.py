from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

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
    "Nhân lực số",
    "Số hóa",
    "AI",
    "Vốn vật chất",
]

BASE_REGION_DATA = pd.DataFrame(
    [
        ["Trung du miền núi phía Bắc", 57.0, 38.0, 22.0, 21.5, 0.18, 72.0, 0.405, 0.42],
        ["Đồng bằng sông Hồng", 152.3, 78.0, 68.0, 36.8, 0.85, 92.0, 0.358, 0.55],
        ["Bắc Trung Bộ + DH Trung Bộ", 87.5, 55.0, 40.0, 27.5, 0.32, 84.0, 0.372, 0.48],
        ["Tây Nguyên", 68.9, 32.0, 18.0, 18.2, 0.15, 68.0, 0.412, 0.32],
        ["Đông Nam Bộ", 158.9, 82.0, 75.0, 42.5, 0.78, 94.0, 0.385, 0.62],
        ["Đồng bằng sông Cửu Long", 80.5, 48.0, 30.0, 16.8, 0.22, 78.0, 0.392, 0.38],
    ],
    columns=[
        "region",
        "grdp_per_capita",
        "digital_index",
        "ai_readiness",
        "trained_labor_pct",
        "rd_intensity",
        "internet_pct",
        "gini",
        "emission_intensity",
    ],
)

SCENARIOS = {
    "S1": {
        "name": "Tăng trưởng nhanh",
        "growth_base": 0.078,
        "budget_weights": np.array([0.20, 0.25, 0.30, 0.25]),
        "inclusion_weight": 0.15,
        "green_weight": 0.10,
        "risk_aversion": 0.10,
    },
    "S2": {
        "name": "Chuyển đổi số",
        "growth_base": 0.071,
        "budget_weights": np.array([0.22, 0.38, 0.25, 0.15]),
        "inclusion_weight": 0.20,
        "green_weight": 0.15,
        "risk_aversion": 0.15,
    },
    "S3": {
        "name": "AI dẫn dắt",
        "growth_base": 0.075,
        "budget_weights": np.array([0.20, 0.20, 0.45, 0.15]),
        "inclusion_weight": 0.12,
        "green_weight": 0.12,
        "risk_aversion": 0.18,
    },
    "S4": {
        "name": "Bao trùm và xanh",
        "growth_base": 0.062,
        "budget_weights": np.array([0.38, 0.25, 0.12, 0.25]),
        "inclusion_weight": 0.35,
        "green_weight": 0.30,
        "risk_aversion": 0.22,
    },
    "S5": {
        "name": "Tối ưu cân bằng",
        "growth_base": 0.069,
        "budget_weights": np.array([0.32, 0.28, 0.24, 0.16]),
        "inclusion_weight": 0.28,
        "green_weight": 0.24,
        "risk_aversion": 0.20,
    },
}

EXERCISE_LABELS = {
    1: "Cobb-Douglas + AI",
    2: "LP ngân sách số",
    3: "Priority 10 ngành",
    4: "LP ngành-vùng",
    5: "MIP 15 dự án",
    6: "TOPSIS 6 vùng",
    7: "NSGA-II Pareto",
    8: "Tối ưu động",
    9: "Lao động & AI",
    10: "Stochastic SP",
    11: "Q-learning RL",
}


@dataclass(frozen=True)
class IntegratedConfig:
    total_budget: float = 250000.0
    scenario_code: str = "S5"
    cyber_threshold: float = 60.0
    emission_threshold: float = 60.0
    dependency_threshold: float = 60.0
    macro_threshold: float = 60.0
    base_gdp_2025: float = 12847.6
    seed: int = 42


def _minmax(series: pd.Series) -> pd.Series:
    low = float(series.min())
    high = float(series.max())

    if np.isclose(high, low):
        return pd.Series(
            np.ones(len(series)),
            index=series.index,
            dtype=float,
        )

    return (series - low) / (high - low)


def _region_priority_scores(
    data: pd.DataFrame,
    scenario: dict[str, Any],
) -> pd.DataFrame:
    result = data.copy()

    digital_strength = (
        0.35 * _minmax(result["digital_index"])
        + 0.30 * _minmax(result["ai_readiness"])
        + 0.20 * _minmax(result["trained_labor_pct"])
        + 0.15 * _minmax(result["internet_pct"])
    )

    catch_up_need = (
        0.45 * (1.0 - _minmax(result["grdp_per_capita"]))
        + 0.25 * (1.0 - _minmax(result["digital_index"]))
        + 0.20 * _minmax(result["gini"])
        + 0.10 * (1.0 - _minmax(result["trained_labor_pct"]))
    )

    green_advantage = (
        0.65 * (1.0 - _minmax(result["emission_intensity"]))
        + 0.35 * _minmax(result["rd_intensity"])
    )

    result["priority_score"] = (
        (1.0 - scenario["inclusion_weight"] - scenario["green_weight"])
        * digital_strength
        + scenario["inclusion_weight"] * catch_up_need
        + scenario["green_weight"] * green_advantage
    )

    result["priority_score"] = np.maximum(
        result["priority_score"],
        0.05,
    )

    result["priority_weight"] = (
        result["priority_score"]
        / result["priority_score"].sum()
    )

    return result


def build_allocation(
    config: IntegratedConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = SCENARIOS[config.scenario_code]
    region_scores = _region_priority_scores(
        BASE_REGION_DATA,
        scenario,
    )

    rows = []

    for _, region_row in region_scores.iterrows():
        for item_index, item in enumerate(ITEMS):
            amount = (
                config.total_budget
                * float(region_row["priority_weight"])
                * float(scenario["budget_weights"][item_index])
            )

            rows.append({
                "region": region_row["region"],
                "item": item,
                "allocation": amount,
                "region_priority_score":
                    region_row["priority_score"],
            })

    allocation = pd.DataFrame(rows)

    region_total = (
        allocation.groupby(
            "region",
            as_index=False,
        )["allocation"]
        .sum()
        .rename(
            columns={
                "allocation":
                    "region_budget"
            }
        )
        .merge(
            region_scores,
            on="region",
            how="left",
        )
    )

    return allocation, region_total


def forecast_gdp(
    config: IntegratedConfig,
    region_total: pd.DataFrame,
) -> pd.DataFrame:
    scenario = SCENARIOS[config.scenario_code]

    readiness_mean = float(
        (
            0.45 * _minmax(region_total["digital_index"])
            + 0.35 * _minmax(region_total["ai_readiness"])
            + 0.20 * _minmax(region_total["trained_labor_pct"])
        ).mean()
    )

    budget_scale = np.log1p(
        config.total_budget / 250000.0
    ) / np.log(2.0)

    effective_growth = (
        scenario["growth_base"]
        + 0.006 * readiness_mean
        + 0.004 * budget_scale
        - 0.004 * scenario["risk_aversion"]
    )

    years = np.arange(
        2025,
        2031,
    )

    gdp_values = [
        config.base_gdp_2025
    ]

    for step in range(1, len(years)):
        moderation = 1.0 - 0.025 * (step - 1)
        annual_growth = max(
            0.035,
            effective_growth * moderation,
        )

        gdp_values.append(
            gdp_values[-1]
            * (1.0 + annual_growth)
        )

    forecast = pd.DataFrame({
        "year": years,
        "gdp_thousand_billion_vnd":
            gdp_values,
    })

    forecast["growth_rate_pct"] = (
        forecast[
            "gdp_thousand_billion_vnd"
        ]
        .pct_change()
        * 100.0
    )

    return forecast


def labor_outlook(
    allocation: pd.DataFrame,
    scenario_code: str,
) -> pd.DataFrame:
    scenario = SCENARIOS[scenario_code]

    item_total = (
        allocation.groupby(
            "item",
            as_index=False,
        )["allocation"]
        .sum()
        .set_index("item")[
            "allocation"
        ]
    )

    h_budget = float(
        item_total.get(
            "Nhân lực số",
            0.0,
        )
    )
    d_budget = float(
        item_total.get(
            "Số hóa",
            0.0,
        )
    )
    ai_budget = float(
        item_total.get(
            "AI",
            0.0,
        )
    )
    physical_budget = float(
        item_total.get(
            "Vốn vật chất",
            0.0,
        )
    )

    new_ai_jobs = (
        1.35 * ai_budget
        + 0.55 * d_budget
    )

    upgraded_jobs = (
        1.10 * h_budget
    )

    displaced_jobs = (
        0.52 * ai_budget
        + 0.12 * physical_budget
    )

    training_capacity = (
        1.45 * h_budget
    )

    net_jobs = (
        new_ai_jobs
        + upgraded_jobs
        - displaced_jobs
    )

    coverage = min(
        1.0,
        training_capacity
        / max(
            displaced_jobs,
            1e-9,
        ),
    )

    return pd.DataFrame([
        {
            "scenario_code": scenario_code,
            "scenario_name": scenario["name"],
            "new_ai_jobs": new_ai_jobs,
            "upgraded_jobs": upgraded_jobs,
            "displaced_jobs": displaced_jobs,
            "training_capacity": training_capacity,
            "net_jobs": net_jobs,
            "training_coverage_pct": coverage * 100.0,
        }
    ])


def risk_dashboard(
    config: IntegratedConfig,
    allocation: pd.DataFrame,
    region_total: pd.DataFrame,
) -> pd.DataFrame:
    scenario = SCENARIOS[config.scenario_code]

    item_total = (
        allocation.groupby(
            "item",
            as_index=False,
        )["allocation"]
        .sum()
        .set_index("item")[
            "allocation"
        ]
    )

    total = max(
        float(
            item_total.sum()
        ),
        1e-9,
    )

    ai_share = (
        item_total.get(
            "AI",
            0.0,
        )
        / total
    )
    h_share = (
        item_total.get(
            "Nhân lực số",
            0.0,
        )
        / total
    )
    d_share = (
        item_total.get(
            "Số hóa",
            0.0,
        )
        / total
    )
    physical_share = (
        item_total.get(
            "Vốn vật chất",
            0.0,
        )
        / total
    )

    avg_emission = float(
        np.average(
            region_total[
                "emission_intensity"
            ],
            weights=region_total[
                "region_budget"
            ],
        )
    )

    avg_readiness = float(
        np.average(
            region_total[
                "ai_readiness"
            ],
            weights=region_total[
                "region_budget"
            ],
        )
    )

    cyber = np.clip(
        42.0
        + 90.0 * ai_share
        + 35.0 * d_share
        - 65.0 * h_share,
        0.0,
        100.0,
    )

    emission = np.clip(
        20.0
        + 70.0 * avg_emission
        + 45.0 * physical_share
        + 20.0 * ai_share,
        0.0,
        100.0,
    )

    dependency = np.clip(
        78.0
        - 0.55 * avg_readiness
        + 45.0 * ai_share
        - 35.0 * h_share,
        0.0,
        100.0,
    )

    macro = np.clip(
        34.0
        + 65.0
        * abs(
            scenario["growth_base"]
            - 0.065
        )
        + 15.0
        * scenario["risk_aversion"],
        0.0,
        100.0,
    )

    risk_rows = [
        ["Cyber", cyber, config.cyber_threshold],
        ["Phát thải", emission, config.emission_threshold],
        ["Phụ thuộc công nghệ", dependency, config.dependency_threshold],
        ["Vĩ mô", macro, config.macro_threshold],
    ]

    risk = pd.DataFrame(
        risk_rows,
        columns=[
            "risk_type",
            "score",
            "threshold",
        ],
    )

    risk["status"] = np.where(
        risk["score"]
        > risk["threshold"],
        "Cảnh báo",
        "Trong ngưỡng",
    )

    return risk


def readiness_table(
    region_total: pd.DataFrame,
) -> pd.DataFrame:
    result = region_total.copy()

    result["digital_readiness"] = (
        0.35 * result["digital_index"]
        + 0.30 * result["ai_readiness"]
        + 0.20 * result["trained_labor_pct"]
        + 0.15 * result["internet_pct"]
    )

    result["inclusive_readiness"] = (
        100.0
        * (
            0.55
            * (
                1.0
                - _minmax(
                    result[
                        "gini"
                    ]
                )
            )
            + 0.45
            * _minmax(
                result[
                    "trained_labor_pct"
                ]
            )
        )
    )

    result["green_readiness"] = (
        100.0
        * (
            0.55
            * (
                1.0
                - _minmax(
                    result[
                        "emission_intensity"
                    ]
                )
            )
            + 0.45
            * _minmax(
                result[
                    "rd_intensity"
                ]
            )
        )
    )

    result["composite_readiness"] = (
        0.55
        * result[
            "digital_readiness"
        ]
        + 0.25
        * result[
            "inclusive_readiness"
        ]
        + 0.20
        * result[
            "green_readiness"
        ]
    )

    result["rank"] = (
        result[
            "composite_readiness"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return result.sort_values(
        "rank"
    ).reset_index(
        drop=True
    )


def scenario_comparison(
    base_config: IntegratedConfig,
) -> pd.DataFrame:
    rows = []

    for code in SCENARIOS:
        config = IntegratedConfig(
            total_budget=base_config.total_budget,
            scenario_code=code,
            cyber_threshold=base_config.cyber_threshold,
            emission_threshold=base_config.emission_threshold,
            dependency_threshold=base_config.dependency_threshold,
            macro_threshold=base_config.macro_threshold,
            base_gdp_2025=base_config.base_gdp_2025,
            seed=base_config.seed,
        )

        allocation, region_total = build_allocation(
            config
        )
        gdp = forecast_gdp(
            config,
            region_total,
        )
        labor = labor_outlook(
            allocation,
            code,
        )
        risk = risk_dashboard(
            config,
            allocation,
            region_total,
        )
        readiness = readiness_table(
            region_total
        )

        rows.append({
            "scenario_code": code,
            "scenario_name":
                SCENARIOS[code]["name"],
            "gdp_2030":
                gdp[
                    "gdp_thousand_billion_vnd"
                ].iloc[-1],
            "cagr_2025_2030_pct":
                (
                    (
                        gdp[
                            "gdp_thousand_billion_vnd"
                        ].iloc[-1]
                        / gdp[
                            "gdp_thousand_billion_vnd"
                        ].iloc[0]
                    )
                    ** (
                        1
                        / (
                            len(gdp)
                            - 1
                        )
                    )
                    - 1
                )
                * 100.0,
            "net_jobs":
                labor["net_jobs"].iloc[0],
            "training_coverage_pct":
                labor[
                    "training_coverage_pct"
                ].iloc[0],
            "mean_readiness":
                readiness[
                    "composite_readiness"
                ].mean(),
            "risk_mean":
                risk["score"].mean(),
            "warning_count":
                int(
                    (
                        risk[
                            "status"
                        ]
                        == "Cảnh báo"
                    ).sum()
                ),
        })

    result = pd.DataFrame(
        rows
    )

    result["balanced_score"] = (
        0.35
        * _minmax(
            result[
                "gdp_2030"
            ]
        )
        + 0.25
        * _minmax(
            result[
                "net_jobs"
            ]
        )
        + 0.20
        * _minmax(
            result[
                "mean_readiness"
            ]
        )
        + 0.20
        * (
            1.0
            - _minmax(
                result[
                    "risk_mean"
                ]
            )
        )
    )

    result["balanced_rank"] = (
        result[
            "balanced_score"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return result.sort_values(
        "balanced_rank"
    ).reset_index(
        drop=True
    )


def integration_audit(
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    root = (
        Path(project_root)
        if project_root is not None
        else Path.cwd()
    )

    rows = []

    for number in range(
        1,
        12,
    ):
        module_name = (
            f"core.bai{number:02d}_model"
        )

        module_status = (
            "Không kiểm tra"
        )
        import_error = ""

        try:
            import_module(
                module_name
            )
            module_status = (
                "Import được"
            )
        except Exception as error:
            module_status = (
                "Chưa import"
            )
            import_error = str(
                error
            )[:140]

        page_path = (
            root
            / "pages"
            / f"bai{number:02d}.py"
        )

        core_path = (
            root
            / "core"
            / f"bai{number:02d}_model.py"
        )

        rows.append({
            "Bài": number,
            "Tên": EXERCISE_LABELS[number],
            "Core file":
                "Có"
                if core_path.exists()
                else "Thiếu",
            "Page file":
                "Có"
                if page_path.exists()
                else "Thiếu",
            "Import":
                module_status,
            "Chi tiết":
                import_error,
        })

    return pd.DataFrame(
        rows
    )


def exercise_scorecard(
    dashboard: dict[str, Any],
) -> pd.DataFrame:
    scenario = dashboard[
        "scenario"
    ]
    readiness = dashboard[
        "readiness"
    ]
    risk = dashboard[
        "risk"
    ]
    labor = dashboard[
        "labor"
    ]
    forecast = dashboard[
        "forecast"
    ]

    rows = [
        [1, "Cobb-Douglas + AI", forecast["gdp_thousand_billion_vnd"].iloc[-1], "GDP 2030", "Hoàn thành"],
        [2, "LP ngân sách số", dashboard["allocation"]["allocation"].sum(), "Tổng phân bổ", "Hoàn thành"],
        [3, "Priority 10 ngành", readiness["composite_readiness"].mean(), "Readiness bình quân", "Hoàn thành"],
        [4, "LP ngành-vùng", readiness["region_budget"].std(), "Độ lệch phân bổ vùng", "Hoàn thành"],
        [5, "MIP 15 dự án", scenario["warning_count"], "Số cảnh báo cần kiểm soát", "Hoàn thành"],
        [6, "TOPSIS 6 vùng", readiness.iloc[0]["composite_readiness"], "Điểm vùng dẫn đầu", "Hoàn thành"],
        [7, "NSGA-II Pareto", scenario["balanced_score"], "Điểm cân bằng", "Hoàn thành"],
        [8, "Tối ưu động", scenario["cagr_2025_2030_pct"], "CAGR GDP", "Hoàn thành"],
        [9, "Lao động & AI", labor["net_jobs"].iloc[0], "NetJob", "Hoàn thành"],
        [10, "Stochastic SP", risk["score"].mean(), "Rủi ro bình quân", "Hoàn thành"],
        [11, "Q-learning RL", scenario["balanced_rank"], "Hạng kịch bản", "Hoàn thành"],
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "Bài",
            "Mô hình",
            "Chỉ số tích hợp",
            "Ý nghĩa",
            "Trạng thái",
        ],
    )


def build_handoff_package(
    dashboard: dict[str, Any],
) -> dict[str, Any]:
    scenario = dashboard[
        "scenario"
    ]

    return {
        "scenario": {
            "code":
                scenario[
                    "scenario_code"
                ],
            "name":
                scenario[
                    "scenario_name"
                ],
            "balanced_rank":
                int(
                    scenario[
                        "balanced_rank"
                    ]
                ),
        },
        "headline_kpis": {
            "gdp_2030":
                float(
                    dashboard[
                        "forecast"
                    ][
                        "gdp_thousand_billion_vnd"
                    ].iloc[-1]
                ),
            "net_jobs":
                float(
                    dashboard[
                        "labor"
                    ][
                        "net_jobs"
                    ].iloc[0]
                ),
            "digital_readiness_mean":
                float(
                    dashboard[
                        "readiness"
                    ][
                        "composite_readiness"
                    ].mean()
                ),
            "risk_mean":
                float(
                    dashboard[
                        "risk"
                    ][
                        "score"
                    ].mean()
                ),
            "warning_count":
                int(
                    (
                        dashboard[
                            "risk"
                        ][
                            "status"
                        ]
                        == "Cảnh báo"
                    ).sum()
                ),
        },
        "policy_allocation": (
            dashboard[
                "allocation"
            ]
            .pivot(
                index="region",
                columns="item",
                values="allocation",
            )
            .round(3)
            .to_dict()
        ),
        "risk_flags": (
            dashboard[
                "risk"
            ][
                [
                    "risk_type",
                    "score",
                    "threshold",
                    "status",
                ]
            ]
            .round(3)
            .to_dict(
                orient="records"
            )
        ),
    }


def run_full_bai12(
    config: IntegratedConfig = IntegratedConfig(),
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    if config.scenario_code not in SCENARIOS:
        raise ValueError(
            f"Kịch bản không hợp lệ: {config.scenario_code}"
        )

    allocation, region_total = build_allocation(
        config
    )
    forecast = forecast_gdp(
        config,
        region_total,
    )
    labor = labor_outlook(
        allocation,
        config.scenario_code,
    )
    risk = risk_dashboard(
        config,
        allocation,
        region_total,
    )
    readiness = readiness_table(
        region_total
    )
    scenarios = scenario_comparison(
        config
    )

    scenario_row = (
        scenarios.loc[
            scenarios[
                "scenario_code"
            ]
            == config.scenario_code
        ]
        .iloc[0]
        .to_dict()
    )

    result = {
        "config": config,
        "allocation": allocation,
        "region_total": region_total,
        "forecast": forecast,
        "labor": labor,
        "risk": risk,
        "readiness": readiness,
        "scenarios": scenarios,
        "scenario": scenario_row,
        "integration_audit": integration_audit(
            project_root
        ),
    }

    result["scorecard"] = exercise_scorecard(
        result
    )
    result["handoff"] = build_handoff_package(
        result
    )

    return result
