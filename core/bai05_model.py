from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "code",
    "project_name",
    "field",
    "cost_total",
    "benefit_npv",
    "cost_year_1_2",
    "cost_year_3_5",
    "completion_probability",
]


def load_project_data(
    csv_path: str | Path,
) -> pd.DataFrame:
    """Đọc và kiểm tra danh mục 15 dự án."""

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}"
        )

    data = pd.read_csv(path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "File dữ liệu thiếu các cột: "
            + ", ".join(missing)
        )

    data = data.copy()

    numeric_columns = [
        "cost_total",
        "benefit_npv",
        "cost_year_1_2",
        "cost_year_3_5",
        "completion_probability",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if len(data) != 15:
        raise ValueError(
            f"Đề yêu cầu 15 dự án, dữ liệu hiện có {len(data)}."
        )

    if data["code"].duplicated().any():
        raise ValueError(
            "Mã dự án bị trùng."
        )

    if not np.allclose(
        data["cost_total"],
        (
            data["cost_year_1_2"]
            + data["cost_year_3_5"]
        ),
    ):
        raise ValueError(
            "Chi phí 5 năm không bằng tổng chi phí hai giai đoạn."
        )

    if (
        (
            data["completion_probability"]
            <= 0
        )
        | (
            data["completion_probability"]
            > 1
        )
    ).any():
        raise ValueError(
            "Xác suất hoàn thành phải nằm trong (0,1]."
        )

    data["benefit_cost_ratio"] = (
        data["benefit_npv"]
        / data["cost_total"]
    )

    data["expected_benefit"] = (
        data["benefit_npv"]
        * data[
            "completion_probability"
        ]
    )

    return data.reset_index(
        drop=True
    )


def validate_parameters(
    total_budget: float,
    early_budget: float,
    min_projects: int,
    max_projects: int,
) -> None:
    if total_budget <= 0:
        raise ValueError(
            "Ngân sách 5 năm phải lớn hơn 0."
        )

    if early_budget <= 0:
        raise ValueError(
            "Ngân sách năm 1-2 phải lớn hơn 0."
        )

    if early_budget > total_budget:
        raise ValueError(
            "Ngân sách năm 1-2 không nên vượt ngân sách 5 năm."
        )

    if not 1 <= min_projects <= 15:
        raise ValueError(
            "Số dự án tối thiểu phải trong [1,15]."
        )

    if not 1 <= max_projects <= 15:
        raise ValueError(
            "Số dự án tối đa phải trong [1,15]."
        )

    if min_projects > max_projects:
        raise ValueError(
            "Số dự án tối thiểu không được lớn hơn tối đa."
        )


def solve_project_mip(
    data: pd.DataFrame,
    total_budget: float = 80000.0,
    early_budget: float = 40000.0,
    min_projects: int = 7,
    max_projects: int = 11,
    force_p14: bool = True,
    data_center_exclusion: bool = True,
    force_both_data_centers: bool = False,
    risk_adjusted: bool = False,
    force_p15: bool | None = None,
    synergy_bonus: float = 0.0,
) -> dict[str, Any]:
    """
    Giải MIP lựa chọn dự án bằng PuLP/CBC.

    synergy_bonus bổ sung lợi ích khi cả P8 và P13 được chọn.
    """

    validate_parameters(
        total_budget=total_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
    )

    try:
        import pulp
    except ImportError as error:
        return {
            "success": False,
            "status": (
                "Chưa cài PuLP. Chạy: "
                "python -m pip install pulp"
            ),
            "error": str(error),
        }

    project_codes = (
        data["code"].tolist()
    )

    indexed = data.set_index(
        "code"
    )

    model = pulp.LpProblem(
        "VN_Project_Selection",
        pulp.LpMaximize,
    )

    y = pulp.LpVariable.dicts(
        "select",
        project_codes,
        cat="Binary",
    )

    z_synergy = pulp.LpVariable(
        "synergy_P8_P13",
        cat="Binary",
    )

    objective_values = {}

    for code in project_codes:
        if risk_adjusted:
            objective_values[code] = float(
                indexed.loc[
                    code,
                    "expected_benefit",
                ]
            )
        else:
            objective_values[code] = float(
                indexed.loc[
                    code,
                    "benefit_npv",
                ]
            )

    model += (
        pulp.lpSum(
            objective_values[code]
            * y[code]
            for code in project_codes
        )
        + float(synergy_bonus)
        * z_synergy,
        "Portfolio_value",
    )

    model += (
        pulp.lpSum(
            float(
                indexed.loc[
                    code,
                    "cost_total",
                ]
            )
            * y[code]
            for code in project_codes
        )
        <= float(total_budget),
        "C1_Total_budget",
    )

    model += (
        pulp.lpSum(
            float(
                indexed.loc[
                    code,
                    "cost_year_1_2",
                ]
            )
            * y[code]
            for code in project_codes
        )
        <= float(early_budget),
        "C2_Early_budget",
    )

    if data_center_exclusion:
        model += (
            y["P1"] + y["P2"] <= 1,
            "C3_Data_center_exclusion",
        )

    if force_both_data_centers:
        model += (
            y["P1"] == 1,
            "C3a_Force_P1",
        )

        model += (
            y["P2"] == 1,
            "C3b_Force_P2",
        )

    model += (
        y["P8"] <= y["P12"],
        "C4_AI_requires_training",
    )

    model += (
        y["P13"] <= y["P12"],
        "C5_Semiconductor_requires_training",
    )

    model += (
        y["P4"] + y["P5"] >= 1,
        "C6a_Digital_government",
    )

    if force_p14:
        model += (
            y["P14"] == 1,
            "C6b_Cybersecurity_mandatory",
        )

    model += (
        pulp.lpSum(
            y[code]
            for code in project_codes
        )
        >= int(min_projects),
        "C7a_Min_projects",
    )

    model += (
        pulp.lpSum(
            y[code]
            for code in project_codes
        )
        <= int(max_projects),
        "C7b_Max_projects",
    )

    if force_p15 is True:
        model += (
            y["P15"] == 1,
            "Policy_Force_P15",
        )
    elif force_p15 is False:
        model += (
            y["P15"] == 0,
            "Policy_Exclude_P15",
        )

    # Linear hóa z = y8*y13.
    model += (
        z_synergy <= y["P8"],
        "Syn_1",
    )
    model += (
        z_synergy <= y["P13"],
        "Syn_2",
    )
    model += (
        z_synergy
        >= y["P8"] + y["P13"] - 1,
        "Syn_3",
    )

    solver = pulp.PULP_CBC_CMD(
        msg=False
    )

    status_code = model.solve(
        solver
    )

    status = pulp.LpStatus.get(
        status_code,
        str(status_code),
    )

    if status != "Optimal":
        return {
            "success": False,
            "status": status,
            "model": model,
        }

    selected_codes = [
        code
        for code in project_codes
        if float(
            y[code].value()
            or 0.0
        ) > 0.5
    ]

    selected_df = data[
        data["code"].isin(
            selected_codes
        )
    ].copy()

    selected_df["selected"] = 1

    full_df = data.copy()

    full_df["selected"] = (
        full_df["code"]
        .isin(selected_codes)
        .astype(int)
    )

    full_df["objective_value"] = (
        full_df[
            "expected_benefit"
        ]
        if risk_adjusted
        else full_df[
            "benefit_npv"
        ]
    )

    total_cost = float(
        selected_df[
            "cost_total"
        ].sum()
    )

    early_cost = float(
        selected_df[
            "cost_year_1_2"
        ].sum()
    )

    nominal_benefit = float(
        selected_df[
            "benefit_npv"
        ].sum()
    )

    expected_benefit = float(
        selected_df[
            "expected_benefit"
        ].sum()
    )

    synergy_realized = (
        float(synergy_bonus)
        if (
            "P8" in selected_codes
            and "P13" in selected_codes
        )
        else 0.0
    )

    objective = float(
        pulp.value(
            model.objective
        )
    )

    constraints = []

    constraints.append({
        "Ràng buộc":
            "C1 Ngân sách 5 năm",
        "Giá trị":
            total_cost,
        "Điều kiện":
            f"≤ {total_budget:,.0f}",
        "Đạt?":
            total_cost
            <= total_budget + 1e-6,
    })

    constraints.append({
        "Ràng buộc":
            "C2 Ngân sách năm 1-2",
        "Giá trị":
            early_cost,
        "Điều kiện":
            f"≤ {early_budget:,.0f}",
        "Đạt?":
            early_cost
            <= early_budget + 1e-6,
    })

    constraints.append({
        "Ràng buộc":
            "C3 Trung tâm dữ liệu",
        "Giá trị":
            int(
                "P1" in selected_codes
            )
            + int(
                "P2" in selected_codes
            ),
        "Điều kiện":
            (
                "= 2"
                if force_both_data_centers
                else (
                    "≤ 1"
                    if data_center_exclusion
                    else "Không áp dụng"
                )
            ),
        "Đạt?":
            (
                (
                    "P1" in selected_codes
                    and "P2" in selected_codes
                )
                if force_both_data_centers
                else (
                    not (
                        "P1" in selected_codes
                        and "P2" in selected_codes
                    )
                    if data_center_exclusion
                    else True
                )
            ),
    })

    constraints.append({
        "Ràng buộc":
            "C4 P8 cần P12",
        "Giá trị":
            (
                f"P8={int('P8' in selected_codes)}, "
                f"P12={int('P12' in selected_codes)}"
            ),
        "Điều kiện":
            "y8 ≤ y12",
        "Đạt?":
            (
                "P8" not in selected_codes
                or "P12" in selected_codes
            ),
    })

    constraints.append({
        "Ràng buộc":
            "C5 P13 cần P12",
        "Giá trị":
            (
                f"P13={int('P13' in selected_codes)}, "
                f"P12={int('P12' in selected_codes)}"
            ),
        "Điều kiện":
            "y13 ≤ y12",
        "Đạt?":
            (
                "P13" not in selected_codes
                or "P12" in selected_codes
            ),
    })

    constraints.append({
        "Ràng buộc":
            "C6 Chính phủ số",
        "Giá trị":
            int(
                "P4" in selected_codes
            )
            + int(
                "P5" in selected_codes
            ),
        "Điều kiện":
            "≥ 1",
        "Đạt?":
            bool(
                {
                    "P4",
                    "P5",
                }
                & set(
                    selected_codes
                )
            ),
    })

    constraints.append({
        "Ràng buộc":
            "C6 P14 bắt buộc",
        "Giá trị":
            int(
                "P14" in selected_codes
            ),
        "Điều kiện":
            (
                "= 1"
                if force_p14
                else "Không bắt buộc"
            ),
        "Đạt?":
            (
                "P14" in selected_codes
                if force_p14
                else True
            ),
    })

    constraints.append({
        "Ràng buộc":
            "C7 Số dự án",
        "Giá trị":
            len(selected_codes),
        "Điều kiện":
            f"{min_projects} ≤ n ≤ {max_projects}",
        "Đạt?":
            min_projects
            <= len(selected_codes)
            <= max_projects,
    })

    return {
        "success": True,
        "status": status,
        "objective": objective,
        "risk_adjusted": risk_adjusted,
        "selected_codes": selected_codes,
        "selected_df": (
            selected_df
            .sort_values(
                "code"
            )
            .reset_index(
                drop=True
            )
        ),
        "full_df": full_df,
        "total_cost": total_cost,
        "early_cost": early_cost,
        "nominal_benefit":
            nominal_benefit,
        "expected_benefit":
            expected_benefit,
        "npv_per_cost": (
            nominal_benefit
            / total_cost
            if total_cost > 0
            else np.nan
        ),
        "objective_per_cost": (
            objective
            / total_cost
            if total_cost > 0
            else np.nan
        ),
        "synergy_realized":
            synergy_realized,
        "constraint_check":
            pd.DataFrame(
                constraints
            ),
        "model": model,
    }


def compare_scenarios(
    scenarios: dict[
        str,
        dict[str, Any],
    ],
) -> pd.DataFrame:
    """Tạo bảng KPI so sánh nhiều kịch bản."""

    rows = []

    for name, result in scenarios.items():
        if not result[
            "success"
        ]:
            rows.append({
                "Kịch bản": name,
                "Trạng thái":
                    result["status"],
                "Số dự án": np.nan,
                "Chi phí 5 năm":
                    np.nan,
                "Chi phí năm 1-2":
                    np.nan,
                "NPV danh nghĩa":
                    np.nan,
                "Lợi ích kỳ vọng":
                    np.nan,
                "Giá trị mục tiêu":
                    np.nan,
                "NPV/Chi phí":
                    np.nan,
            })

            continue

        rows.append({
            "Kịch bản": name,
            "Trạng thái":
                result["status"],
            "Số dự án":
                len(
                    result[
                        "selected_codes"
                    ]
                ),
            "Chi phí 5 năm":
                result["total_cost"],
            "Chi phí năm 1-2":
                result["early_cost"],
            "NPV danh nghĩa":
                result[
                    "nominal_benefit"
                ],
            "Lợi ích kỳ vọng":
                result[
                    "expected_benefit"
                ],
            "Giá trị mục tiêu":
                result["objective"],
            "NPV/Chi phí":
                result["npv_per_cost"],
        })

    return pd.DataFrame(
        rows
    )


def project_change_table(
    data: pd.DataFrame,
    base: dict[str, Any],
    alternative: dict[str, Any],
    alternative_name: str,
) -> pd.DataFrame:
    """So sánh trạng thái chọn dự án giữa hai kịch bản."""

    base_set = set(
        base.get(
            "selected_codes",
            [],
        )
    )

    alt_set = set(
        alternative.get(
            "selected_codes",
            [],
        )
    )

    table = data[
        [
            "code",
            "project_name",
            "field",
            "cost_total",
            "benefit_npv",
        ]
    ].copy()

    table["Cơ sở"] = (
        table["code"]
        .isin(base_set)
        .map({
            True: "Chọn",
            False: "Không",
        })
    )

    table[
        alternative_name
    ] = (
        table["code"]
        .isin(alt_set)
        .map({
            True: "Chọn",
            False: "Không",
        })
    )

    table["Thay đổi"] = np.select(
        [
            (
                table["code"]
                .isin(
                    alt_set
                    - base_set
                )
            ),
            (
                table["code"]
                .isin(
                    base_set
                    - alt_set
                )
            ),
        ],
        [
            "Được thêm",
            "Bị loại",
        ],
        default="Không đổi",
    )

    return table


def run_full_bai05(
    csv_path: str | Path,
    total_budget: float = 80000.0,
    early_budget: float = 40000.0,
    min_projects: int = 7,
    max_projects: int = 11,
    expanded_budget: float = 100000.0,
    synergy_bonus: float = 5000.0,
) -> dict[str, Any]:
    """Chạy đầy đủ 5.4.1–5.4.4 và các đối chứng 5.5."""

    data = load_project_data(
        csv_path
    )

    base = solve_project_mip(
        data=data,
        total_budget=total_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
    )

    expanded = solve_project_mip(
        data=data,
        total_budget=expanded_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
    )

    redundancy = solve_project_mip(
        data=data,
        total_budget=total_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
        data_center_exclusion=False,
        force_both_data_centers=True,
    )

    risk = solve_project_mip(
        data=data,
        total_budget=total_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
        risk_adjusted=True,
    )

    without_p14 = solve_project_mip(
        data=data,
        total_budget=total_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
        force_p14=False,
    )

    force_p15 = solve_project_mip(
        data=data,
        total_budget=total_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
        force_p15=True,
    )

    exclude_p15 = solve_project_mip(
        data=data,
        total_budget=total_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
        force_p15=False,
    )

    synergy = solve_project_mip(
        data=data,
        total_budget=total_budget,
        early_budget=early_budget,
        min_projects=min_projects,
        max_projects=max_projects,
        synergy_bonus=synergy_bonus,
    )

    scenarios = {
        "Cơ sở":
            base,
        "Ngân sách 100.000":
            expanded,
        "Bắt buộc P1 và P2":
            redundancy,
        "Điều chỉnh rủi ro":
            risk,
        "Không bắt buộc P14":
            without_p14,
        "Cộng hưởng P8–P13":
            synergy,
    }

    return {
        "data": data,
        "base": base,
        "expanded": expanded,
        "redundancy":
            redundancy,
        "risk": risk,
        "without_p14":
            without_p14,
        "force_p15":
            force_p15,
        "exclude_p15":
            exclude_p15,
        "synergy": synergy,
        "scenario_table":
            compare_scenarios(
                scenarios
            ),
        "expanded_changes":
            project_change_table(
                data,
                base,
                expanded,
                "Ngân sách 100.000",
            ),
        "redundancy_changes":
            project_change_table(
                data,
                base,
                redundancy,
                "Bắt buộc P1 và P2",
            ),
        "risk_changes":
            project_change_table(
                data,
                base,
                risk,
                "Điều chỉnh rủi ro",
            ),
        "synergy_changes":
            project_change_table(
                data,
                base,
                synergy,
                "Cộng hưởng P8–P13",
            ),
    }
