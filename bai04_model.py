from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REGIONS = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
ITEMS = ["I", "D", "AI", "H"]

REGION_NAMES = {
    "NMM": "Trung du và miền núi phía Bắc",
    "RRD": "Đồng bằng sông Hồng",
    "NCC": "Bắc Trung Bộ và Duyên hải miền Trung",
    "CH": "Tây Nguyên",
    "SE": "Đông Nam Bộ",
    "MD": "Đồng bằng sông Cửu Long",
}

ITEM_NAMES = {
    "I": "Hạ tầng số",
    "D": "CĐS doanh nghiệp",
    "AI": "Ứng dụng AI",
    "H": "Nhân lực số",
}

BETA = {
    ("NMM", "I"): 1.15, ("NMM", "D"): 0.85,
    ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,

    ("RRD", "I"): 0.95, ("RRD", "D"): 1.25,
    ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,

    ("NCC", "I"): 1.05, ("NCC", "D"): 0.95,
    ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,

    ("CH", "I"): 1.20, ("CH", "D"): 0.75,
    ("CH", "AI"): 0.45, ("CH", "H"): 1.35,

    ("SE", "I"): 0.90, ("SE", "D"): 1.30,
    ("SE", "AI"): 1.55, ("SE", "H"): 1.00,

    ("MD", "I"): 1.10, ("MD", "D"): 0.85,
    ("MD", "AI"): 0.65, ("MD", "H"): 1.25,
}

DIGITAL_INDEX_INITIAL = {
    "NMM": 38.0,
    "RRD": 78.0,
    "NCC": 55.0,
    "CH": 32.0,
    "SE": 82.0,
    "MD": 48.0,
}


def get_model_data() -> dict[str, Any]:
    """Trả dữ liệu vùng, hạng mục, beta và Digital Index ban đầu."""

    beta_matrix = pd.DataFrame(
        [
            [
                BETA[(region, item)]
                for item in ITEMS
            ]
            for region in REGIONS
        ],
        index=[
            REGION_NAMES[region]
            for region in REGIONS
        ],
        columns=[
            ITEM_NAMES[item]
            for item in ITEMS
        ],
    )

    beta_long_rows = []

    for region in REGIONS:
        for item in ITEMS:
            beta_long_rows.append({
                "Mã vùng": region,
                "Vùng": REGION_NAMES[region],
                "Mã hạng mục": item,
                "Hạng mục": ITEM_NAMES[item],
                "β tác động biên": BETA[(region, item)],
                "Digital Index ban đầu": (
                    DIGITAL_INDEX_INITIAL[region]
                ),
            })

    digital_table = pd.DataFrame({
        "Mã vùng": REGIONS,
        "Vùng": [
            REGION_NAMES[region]
            for region in REGIONS
        ],
        "Digital Index ban đầu": [
            DIGITAL_INDEX_INITIAL[region]
            for region in REGIONS
        ],
    })

    return {
        "beta_matrix": beta_matrix,
        "beta_long": pd.DataFrame(
            beta_long_rows
        ),
        "digital_table": digital_table,
    }


def validate_parameters(
    total_budget: float,
    min_region: float,
    max_region: float,
    min_h_total: float,
    gamma: float,
    lam: float,
) -> None:
    """Kiểm tra tham số trước khi giải."""

    if total_budget <= 0:
        raise ValueError(
            "Ngân sách tổng phải lớn hơn 0."
        )

    if min_region < 0:
        raise ValueError(
            "Sàn vùng không được âm."
        )

    if max_region < min_region:
        raise ValueError(
            "Trần vùng phải lớn hơn hoặc bằng sàn vùng."
        )

    if min_h_total < 0:
        raise ValueError(
            "Sàn nhân lực số không được âm."
        )

    if gamma <= 0:
        raise ValueError(
            "Gamma phải lớn hơn 0."
        )

    if not 0 < lam <= 1:
        raise ValueError(
            "Lambda phải nằm trong (0, 1]."
        )

    if len(REGIONS) * min_region > total_budget:
        raise ValueError(
            "Tổng sàn của 6 vùng vượt ngân sách tổng."
        )

    if min_h_total > total_budget:
        raise ValueError(
            "Sàn nhân lực số vượt ngân sách tổng."
        )


def quick_feasibility_check(
    max_region: float,
    gamma: float,
    lam: float,
) -> dict[str, float | bool]:
    """
    Kiểm tra sơ bộ khả năng kéo vùng yếu nhất lên ngưỡng công bằng.

    Đây là cảnh báo nhanh, không thay thế kết quả solver.
    """

    d_max_initial = max(
        DIGITAL_INDEX_INITIAL.values()
    )

    d_min_initial = min(
        DIGITAL_INDEX_INITIAL.values()
    )

    required_d_for_weakest = max(
        0.0,
        (
            lam * d_max_initial
            - d_min_initial
        ) / gamma,
    )

    suggested_lambda = (
        d_min_initial
        + gamma * max_region
    ) / d_max_initial

    return {
        "is_warning": (
            required_d_for_weakest
            > max_region
        ),
        "required_d_for_weakest":
            float(required_d_for_weakest),
        "suggested_lambda":
            float(suggested_lambda),
        "d_max_initial":
            float(d_max_initial),
        "d_min_initial":
            float(d_min_initial),
    }


def _empty_result(
    status: str,
    solver: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "status": status,
        "solver": solver,
        "objective": np.nan,
        "allocation_matrix": None,
        "allocation_long": None,
        "region_summary": None,
        "item_summary": None,
        "shadow_table": None,
        "fairness_table": None,
        "constraint_checks": None,
    }


def _build_output_tables(
    allocation_matrix: pd.DataFrame,
    objective: float,
    total_budget: float,
    min_region: float,
    max_region: float,
    min_h_total: float,
    gamma: float,
    lam: float,
    enforce_fairness: bool,
    enforce_region_cap: bool,
    shadow_table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Xây dựng toàn bộ bảng đầu ra dùng chung."""

    allocation_matrix = (
        allocation_matrix
        .clip(lower=0.0)
        .astype(float)
    )

    allocation_long = (
        allocation_matrix
        .reset_index()
        .rename(
            columns={
                "index": "Vùng"
            }
        )
        .melt(
            id_vars="Vùng",
            var_name="Hạng mục",
            value_name=(
                "Ngân sách phân bổ, tỷ VND"
            ),
        )
    )

    region_totals = (
        allocation_matrix.sum(axis=1)
    )

    item_totals = (
        allocation_matrix.sum(axis=0)
    )

    digital_after = []

    for region in REGIONS:
        region_name = REGION_NAMES[region]

        digital_after.append(
            DIGITAL_INDEX_INITIAL[region]
            + gamma
            * allocation_matrix.loc[
                region_name,
                ITEM_NAMES["D"],
            ]
        )

    max_digital_after = float(
        max(digital_after)
    )

    fairness_threshold = (
        lam * max_digital_after
    )

    region_summary = pd.DataFrame({
        "Mã vùng": REGIONS,
        "Vùng": [
            REGION_NAMES[region]
            for region in REGIONS
        ],
        "Tổng ngân sách, tỷ VND": [
            region_totals[
                REGION_NAMES[region]
            ]
            for region in REGIONS
        ],
        "Digital Index ban đầu": [
            DIGITAL_INDEX_INITIAL[region]
            for region in REGIONS
        ],
        "Đầu tư D, tỷ VND": [
            allocation_matrix.loc[
                REGION_NAMES[region],
                ITEM_NAMES["D"],
            ]
            for region in REGIONS
        ],
        "Digital Index sau đầu tư":
            digital_after,
    })

    total_used = float(
        region_summary[
            "Tổng ngân sách, tỷ VND"
        ].sum()
    )

    region_summary[
        "Tỷ trọng ngân sách, %"
    ] = (
        region_summary[
            "Tổng ngân sách, tỷ VND"
        ]
        / total_used
        * 100.0
        if total_used > 0
        else np.nan
    )

    region_summary[
        "Ngưỡng công bằng λ·max"
    ] = fairness_threshold

    region_summary[
        "Đạt công bằng?"
    ] = (
        region_summary[
            "Digital Index sau đầu tư"
        ]
        >= fairness_threshold - 1e-5
    )

    item_summary = pd.DataFrame({
        "Hạng mục":
            item_totals.index,
        "Tổng ngân sách, tỷ VND":
            item_totals.values,
    })

    item_summary["Tỷ trọng, %"] = (
        item_summary[
            "Tổng ngân sách, tỷ VND"
        ]
        / total_used
        * 100.0
        if total_used > 0
        else np.nan
    )

    checks = []

    checks.append({
        "Nhóm ràng buộc":
            "C1 Ngân sách tổng",
        "Giá trị kiểm tra":
            total_used,
        "Ngưỡng":
            f"≤ {total_budget:,.2f}",
        "Đạt?":
            total_used
            <= total_budget + 1e-5,
    })

    total_h = float(
        allocation_matrix[
            ITEM_NAMES["H"]
        ].sum()
    )

    checks.append({
        "Nhóm ràng buộc":
            "C4 Sàn nhân lực số",
        "Giá trị kiểm tra":
            total_h,
        "Ngưỡng":
            f"≥ {min_h_total:,.2f}",
        "Đạt?":
            total_h
            >= min_h_total - 1e-5,
    })

    for region in REGIONS:
        region_name = REGION_NAMES[region]

        region_value = float(
            region_totals[region_name]
        )

        checks.append({
            "Nhóm ràng buộc":
                f"C2 Sàn vùng - {region_name}",
            "Giá trị kiểm tra":
                region_value,
            "Ngưỡng":
                f"≥ {min_region:,.2f}",
            "Đạt?":
                region_value
                >= min_region - 1e-5,
        })

        if enforce_region_cap:
            checks.append({
                "Nhóm ràng buộc":
                    f"C3 Trần vùng - {region_name}",
                "Giá trị kiểm tra":
                    region_value,
                "Ngưỡng":
                    f"≤ {max_region:,.2f}",
                "Đạt?":
                    region_value
                    <= max_region + 1e-5,
            })

    if enforce_fairness:
        for _, row in region_summary.iterrows():
            checks.append({
                "Nhóm ràng buộc":
                    f"C5 Công bằng - {row['Vùng']}",
                "Giá trị kiểm tra":
                    float(
                        row[
                            "Digital Index sau đầu tư"
                        ]
                    ),
                "Ngưỡng":
                    f"≥ {fairness_threshold:.4f}",
                "Đạt?":
                    bool(
                        row[
                            "Đạt công bằng?"
                        ]
                    ),
            })

    preferred_rows = []

    for region_name in allocation_matrix.index:
        preferred_item = (
            allocation_matrix.loc[
                region_name
            ].idxmax()
        )

        preferred_rows.append({
            "Vùng": region_name,
            "Hạng mục ưu tiên":
                preferred_item,
            "Ngân sách ưu tiên, tỷ VND":
                float(
                    allocation_matrix.loc[
                        region_name,
                        preferred_item,
                    ]
                ),
        })

    return {
        "success": True,
        "status": "Optimal",
        "objective": float(objective),
        "allocation_matrix":
            allocation_matrix,
        "allocation_long":
            allocation_long,
        "region_summary":
            region_summary,
        "item_summary":
            item_summary,
        "shadow_table":
            shadow_table,
        "fairness_table":
            region_summary[
                [
                    "Vùng",
                    "Digital Index ban đầu",
                    "Đầu tư D, tỷ VND",
                    "Digital Index sau đầu tư",
                    "Ngưỡng công bằng λ·max",
                    "Đạt công bằng?",
                ]
            ].copy(),
        "constraint_checks":
            pd.DataFrame(checks),
        "preferred_items":
            pd.DataFrame(
                preferred_rows
            ),
        "total_used":
            total_used,
        "unused_budget":
            float(
                total_budget
                - total_used
            ),
        "total_h":
            total_h,
        "max_digital_after":
            max_digital_after,
        "fairness_threshold":
            fairness_threshold,
    }


def solve_pulp_model(
    total_budget: float = 50000.0,
    min_region: float = 5000.0,
    max_region: float = 13000.0,
    min_h_total: float = 12000.0,
    gamma: float = 0.002,
    lam: float = 0.70,
    enforce_fairness: bool = True,
    enforce_region_cap: bool = True,
) -> dict[str, Any]:
    """Câu 4.4.1: giải mô hình bằng PuLP/CBC."""

    validate_parameters(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
    )

    try:
        import pulp
    except ImportError as error:
        result = _empty_result(
            status=(
                "Chưa cài PuLP. "
                "Chạy: python -m pip install pulp"
            ),
            solver="PuLP/CBC",
        )
        result["error"] = str(error)
        return result

    model = pulp.LpProblem(
        "VN_Digital_Budget_Region_LP",
        pulp.LpMaximize,
    )

    x = pulp.LpVariable.dicts(
        "x",
        (REGIONS, ITEMS),
        lowBound=0,
        cat="Continuous",
    )

    model += (
        pulp.lpSum(
            BETA[(region, item)]
            * x[region][item]
            for region in REGIONS
            for item in ITEMS
        ),
        "GDP_gain",
    )

    model += (
        pulp.lpSum(
            x[region][item]
            for region in REGIONS
            for item in ITEMS
        )
        <= total_budget,
        "C1_Total_budget",
    )

    for region in REGIONS:
        model += (
            pulp.lpSum(
                x[region][item]
                for item in ITEMS
            )
            >= min_region,
            f"C2_Min_region_{region}",
        )

        if enforce_region_cap:
            model += (
                pulp.lpSum(
                    x[region][item]
                    for item in ITEMS
                )
                <= max_region,
                f"C3_Max_region_{region}",
            )

    model += (
        pulp.lpSum(
            x[region]["H"]
            for region in REGIONS
        )
        >= min_h_total,
        "C4_Min_total_H",
    )

    if enforce_fairness:
        # Tuyến tính hóa trực tiếp điều kiện:
        # min(D_after) >= lambda * max(D_after)
        #
        # Tương đương với:
        # D_after[r] >= lambda * D_after[s]
        # với mọi cặp vùng r, s.
        #
        # Dạng cặp đôi này ổn định số hơn khi giải bằng CBC,
        # tránh việc biến max phụ làm solver báo Infeasible sai.
        for region in REGIONS:
            for reference_region in REGIONS:
                model += (
                    DIGITAL_INDEX_INITIAL[region]
                    + gamma * x[region]["D"]
                    >= lam
                    * (
                        DIGITAL_INDEX_INITIAL[
                            reference_region
                        ]
                        + gamma
                        * x[
                            reference_region
                        ]["D"]
                    ),
                    (
                        "C5_Fairness_"
                        f"{region}_vs_"
                        f"{reference_region}"
                    ),
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
        return _empty_result(
            status=status,
            solver="PuLP/CBC",
        )

    allocation_matrix = pd.DataFrame(
        [
            [
                float(
                    pulp.value(
                        x[region][item]
                    )
                    or 0.0
                )
                for item in ITEMS
            ]
            for region in REGIONS
        ],
        index=[
            REGION_NAMES[region]
            for region in REGIONS
        ],
        columns=[
            ITEM_NAMES[item]
            for item in ITEMS
        ],
    )

    shadow_rows = []

    for name, constraint in (
        model.constraints.items()
    ):
        shadow_rows.append({
            "Ràng buộc": name,
            "Shadow price": (
                float(constraint.pi)
                if constraint.pi
                is not None
                else np.nan
            ),
            "Slack": (
                float(constraint.slack)
                if constraint.slack
                is not None
                else np.nan
            ),
        })

    result = _build_output_tables(
        allocation_matrix=
            allocation_matrix,
        objective=float(
            pulp.value(
                model.objective
            )
        ),
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=
            enforce_fairness,
        enforce_region_cap=
            enforce_region_cap,
        shadow_table=pd.DataFrame(
            shadow_rows
        ),
    )

    result["status"] = status
    result["solver"] = "PuLP/CBC"
    result["model"] = model

    return result


def solve_cvxpy_model(
    total_budget: float = 50000.0,
    min_region: float = 5000.0,
    max_region: float = 13000.0,
    min_h_total: float = 12000.0,
    gamma: float = 0.002,
    lam: float = 0.70,
    enforce_fairness: bool = True,
    enforce_region_cap: bool = True,
) -> dict[str, Any]:
    """Câu 4.4.2: giải lại mô hình bằng CVXPY."""

    validate_parameters(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
    )

    try:
        import cvxpy as cp
    except ImportError as error:
        result = _empty_result(
            status=(
                "Chưa cài CVXPY. "
                "Chạy: python -m pip install cvxpy"
            ),
            solver="CVXPY",
        )
        result["error"] = str(error)
        return result

    beta_matrix = np.array(
        [
            [
                BETA[(region, item)]
                for item in ITEMS
            ]
            for region in REGIONS
        ],
        dtype=float,
    )

    d0_vector = np.array(
        [
            DIGITAL_INDEX_INITIAL[
                region
            ]
            for region in REGIONS
        ],
        dtype=float,
    )

    x = cp.Variable(
        (
            len(REGIONS),
            len(ITEMS),
        ),
        nonneg=True,
    )

    objective = cp.Maximize(
        cp.sum(
            cp.multiply(
                beta_matrix,
                x,
            )
        )
    )

    constraints = [
        cp.sum(x) <= total_budget,
        cp.sum(x[:, 3]) >= min_h_total,
    ]

    for region_index in range(
        len(REGIONS)
    ):
        constraints.append(
            cp.sum(
                x[region_index, :]
            )
            >= min_region
        )

        if enforce_region_cap:
            constraints.append(
                cp.sum(
                    x[region_index, :]
                )
                <= max_region
            )

    if enforce_fairness:
        digital_after = (
            d0_vector
            + gamma * x[:, 1]
        )

        # Điều kiện công bằng theo từng cặp vùng:
        # D_after[r] >= lambda * D_after[s]
        # với mọi r, s.
        for region_index in range(
            len(REGIONS)
        ):
            for reference_index in range(
                len(REGIONS)
            ):
                constraints.append(
                    digital_after[
                        region_index
                    ]
                    >= lam
                    * digital_after[
                        reference_index
                    ]
                )

    problem = cp.Problem(
        objective,
        constraints,
    )

    solver_used = None
    installed_solvers = set(
        cp.installed_solvers()
    )

    for solver_name in [
        "CLARABEL",
        "SCIPY",
        "ECOS",
        "SCS",
    ]:
        if solver_name not in (
            installed_solvers
        ):
            continue

        try:
            problem.solve(
                solver=solver_name
            )
            solver_used = solver_name

            if problem.status in {
                "optimal",
                "optimal_inaccurate",
            }:
                break
        except Exception:
            continue

    if problem.status not in {
        "optimal",
        "optimal_inaccurate",
    }:
        return _empty_result(
            status=str(
                problem.status
            ),
            solver=(
                f"CVXPY/{solver_used}"
            ),
        )

    allocation_matrix = pd.DataFrame(
        np.asarray(
            x.value,
            dtype=float,
        ),
        index=[
            REGION_NAMES[region]
            for region in REGIONS
        ],
        columns=[
            ITEM_NAMES[item]
            for item in ITEMS
        ],
    )

    result = _build_output_tables(
        allocation_matrix=
            allocation_matrix,
        objective=float(
            problem.value
        ),
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=
            enforce_fairness,
        enforce_region_cap=
            enforce_region_cap,
        shadow_table=None,
    )

    result["status"] = str(
        problem.status
    )

    result["solver"] = (
        f"CVXPY/{solver_used}"
    )

    return result


def compare_solver_results(
    pulp_result: dict[str, Any],
    cvxpy_result: dict[str, Any],
) -> dict[str, Any]:
    """So sánh mục tiêu và ma trận nghiệm của PuLP với CVXPY."""

    if (
        not pulp_result["success"]
        or not cvxpy_result["success"]
    ):
        return {
            "comparable": False,
            "objective_difference":
                np.nan,
            "max_cell_difference":
                np.nan,
            "same_objective":
                False,
            "same_allocation":
                False,
            "difference_matrix":
                None,
        }

    difference_matrix = (
        pulp_result[
            "allocation_matrix"
        ]
        - cvxpy_result[
            "allocation_matrix"
        ]
    )

    objective_difference = abs(
        pulp_result["objective"]
        - cvxpy_result["objective"]
    )

    max_cell_difference = float(
        difference_matrix.abs()
        .to_numpy()
        .max()
    )

    return {
        "comparable": True,
        "objective_difference":
            float(
                objective_difference
            ),
        "max_cell_difference":
            max_cell_difference,
        "same_objective":
            bool(
                objective_difference
                <= 1e-3
            ),
        "same_allocation":
            bool(
                max_cell_difference
                <= 1e-2
            ),
        "difference_matrix":
            difference_matrix,
    }


def compare_fairness(
    total_budget: float = 50000.0,
    min_region: float = 5000.0,
    max_region: float = 13000.0,
    min_h_total: float = 12000.0,
    gamma: float = 0.002,
    lam: float = 0.70,
) -> dict[str, Any]:
    """Câu 4.4.4: so sánh có và không có ràng buộc công bằng C5."""

    with_fairness = solve_pulp_model(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=True,
        enforce_region_cap=True,
    )

    without_fairness = solve_pulp_model(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=False,
        enforce_region_cap=True,
    )

    if (
        not with_fairness["success"]
        or not without_fairness[
            "success"
        ]
    ):
        return {
            "with_fairness":
                with_fairness,
            "without_fairness":
                without_fairness,
            "cost_absolute": np.nan,
            "cost_pct": np.nan,
            "region_comparison":
                pd.DataFrame(),
        }

    cost_absolute = (
        without_fairness["objective"]
        - with_fairness["objective"]
    )

    cost_pct = (
        cost_absolute
        / without_fairness[
            "objective"
        ]
        * 100.0
        if without_fairness[
            "objective"
        ]
        else np.nan
    )

    region_comparison = (
        with_fairness[
            "region_summary"
        ][
            [
                "Vùng",
                "Tổng ngân sách, tỷ VND",
                "Digital Index sau đầu tư",
            ]
        ]
        .rename(
            columns={
                "Tổng ngân sách, tỷ VND":
                    "Có công bằng - Ngân sách",
                "Digital Index sau đầu tư":
                    "Có công bằng - Digital Index",
            }
        )
        .merge(
            without_fairness[
                "region_summary"
            ][
                [
                    "Vùng",
                    "Tổng ngân sách, tỷ VND",
                    "Digital Index sau đầu tư",
                ]
            ].rename(
                columns={
                    "Tổng ngân sách, tỷ VND":
                        "Không công bằng - Ngân sách",
                    "Digital Index sau đầu tư":
                        "Không công bằng - Digital Index",
                }
            ),
            on="Vùng",
            how="inner",
        )
    )

    region_comparison[
        "Chênh lệch ngân sách"
    ] = (
        region_comparison[
            "Có công bằng - Ngân sách"
        ]
        - region_comparison[
            "Không công bằng - Ngân sách"
        ]
    )

    return {
        "with_fairness":
            with_fairness,
        "without_fairness":
            without_fairness,
        "cost_absolute":
            float(cost_absolute),
        "cost_pct":
            float(cost_pct),
        "region_comparison":
            region_comparison,
    }


def compare_region_cap(
    total_budget: float = 50000.0,
    min_region: float = 5000.0,
    max_region: float = 13000.0,
    min_h_total: float = 12000.0,
    gamma: float = 0.002,
    lam: float = 0.70,
) -> dict[str, Any]:
    """Mục 4.5b: đo chi phí của ràng buộc trần vùng C3."""

    with_cap = solve_pulp_model(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=True,
        enforce_region_cap=True,
    )

    without_cap = solve_pulp_model(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=True,
        enforce_region_cap=False,
    )

    if (
        not with_cap["success"]
        or not without_cap[
            "success"
        ]
    ):
        return {
            "with_cap": with_cap,
            "without_cap":
                without_cap,
            "cost_absolute": np.nan,
            "cost_pct": np.nan,
            "region_comparison":
                pd.DataFrame(),
        }

    cost_absolute = (
        without_cap["objective"]
        - with_cap["objective"]
    )

    cost_pct = (
        cost_absolute
        / without_cap["objective"]
        * 100.0
        if without_cap["objective"]
        else np.nan
    )

    comparison = (
        with_cap[
            "region_summary"
        ][
            [
                "Vùng",
                "Tổng ngân sách, tỷ VND",
            ]
        ]
        .rename(
            columns={
                "Tổng ngân sách, tỷ VND":
                    "Có C3",
            }
        )
        .merge(
            without_cap[
                "region_summary"
            ][
                [
                    "Vùng",
                    "Tổng ngân sách, tỷ VND",
                ]
            ].rename(
                columns={
                    "Tổng ngân sách, tỷ VND":
                        "Không C3",
                }
            ),
            on="Vùng",
            how="inner",
        )
    )

    comparison[
        "Chênh lệch"
    ] = (
        comparison["Có C3"]
        - comparison["Không C3"]
    )

    return {
        "with_cap": with_cap,
        "without_cap":
            without_cap,
        "cost_absolute":
            float(cost_absolute),
        "cost_pct":
            float(cost_pct),
        "region_comparison":
            comparison,
    }


def central_highlands_diagnostics(
    full_result: dict[str, Any],
) -> pd.DataFrame:
    """Mục 4.5c: hồ sơ đầu tư của Tây Nguyên."""

    if not full_result["success"]:
        return pd.DataFrame()

    region_name = REGION_NAMES["CH"]

    rows = []

    for item in ITEMS:
        allocation = float(
            full_result[
                "allocation_matrix"
            ].loc[
                region_name,
                ITEM_NAMES[item],
            ]
        )

        beta_value = BETA[
            ("CH", item)
        ]

        rows.append({
            "Hạng mục":
                ITEM_NAMES[item],
            "Mã":
                item,
            "Ngân sách, tỷ VND":
                allocation,
            "β Tây Nguyên":
                beta_value,
            "GDP gain kỳ vọng":
                allocation
                * beta_value,
        })

    return pd.DataFrame(rows)


def run_full_bai04(
    total_budget: float = 50000.0,
    min_region: float = 5000.0,
    max_region: float = 13000.0,
    min_h_total: float = 12000.0,
    gamma: float = 0.002,
    lam: float = 0.70,
) -> dict[str, Any]:
    """Chạy toàn bộ yêu cầu 4.4.1-4.4.4 và phân tích 4.5."""

    pulp_result = solve_pulp_model(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=True,
        enforce_region_cap=True,
    )

    cvxpy_result = solve_cvxpy_model(
        total_budget=total_budget,
        min_region=min_region,
        max_region=max_region,
        min_h_total=min_h_total,
        gamma=gamma,
        lam=lam,
        enforce_fairness=True,
        enforce_region_cap=True,
    )

    solver_comparison = (
        compare_solver_results(
            pulp_result,
            cvxpy_result,
        )
    )

    fairness_comparison = (
        compare_fairness(
            total_budget=total_budget,
            min_region=min_region,
            max_region=max_region,
            min_h_total=min_h_total,
            gamma=gamma,
            lam=lam,
        )
    )

    cap_comparison = (
        compare_region_cap(
            total_budget=total_budget,
            min_region=min_region,
            max_region=max_region,
            min_h_total=min_h_total,
            gamma=gamma,
            lam=lam,
        )
    )

    central_highlands = (
        central_highlands_diagnostics(
            pulp_result
        )
    )

    return {
        "data": get_model_data(),
        "quick_feasibility":
            quick_feasibility_check(
                max_region=max_region,
                gamma=gamma,
                lam=lam,
            ),
        "pulp": pulp_result,
        "cvxpy": cvxpy_result,
        "solver_comparison":
            solver_comparison,
        "fairness_comparison":
            fairness_comparison,
        "cap_comparison":
            cap_comparison,
        "central_highlands":
            central_highlands,
    }
