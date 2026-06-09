from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linprog


ITEM_CODES = ["x1", "x2", "x3", "x4"]

ITEM_NAMES = [
    "Hạ tầng số",
    "AI và dữ liệu",
    "Nhân lực số",
    "R&D công nghệ",
]

DEFAULT_IMPACTS = np.array(
    [0.85, 1.20, 0.95, 1.35],
    dtype=float,
)

DEFAULT_MINIMUMS = np.array(
    [25.0, 15.0, 20.0, 10.0],
    dtype=float,
)


def validate_inputs(
    budget: float,
    minimums: np.ndarray,
    strategic_share: float,
    impacts: np.ndarray,
) -> None:
    minimums = np.asarray(minimums, dtype=float)
    impacts = np.asarray(impacts, dtype=float)

    if budget <= 0:
        raise ValueError("Ngân sách tổng phải lớn hơn 0.")

    if minimums.shape != (4,):
        raise ValueError("minimums phải gồm bốn giá trị x1-x4.")

    if impacts.shape != (4,):
        raise ValueError("impacts phải gồm bốn hệ số mục tiêu.")

    if np.any(minimums < 0):
        raise ValueError("Các mức đầu tư tối thiểu không được âm.")

    if np.any(impacts < 0):
        raise ValueError("Các hệ số tác động không được âm.")

    if not 0 <= strategic_share <= 1:
        raise ValueError("Tỷ trọng AI + R&D phải nằm trong [0, 1].")


def build_scipy_problem(
    budget: float,
    minimums: np.ndarray,
    strategic_share: float,
) -> tuple[np.ndarray, np.ndarray]:
    s = float(strategic_share)

    a_ub = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [s, s - 1.0, s, s - 1.0],
        ],
        dtype=float,
    )

    b_ub = np.array(
        [
            float(budget),
            -float(minimums[0]),
            -float(minimums[1]),
            -float(minimums[2]),
            -float(minimums[3]),
            0.0,
        ],
        dtype=float,
    )

    return a_ub, b_ub


def allocation_table(
    x: np.ndarray,
    impacts: np.ndarray,
    minimums: np.ndarray,
) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    impacts = np.asarray(impacts, dtype=float)
    minimums = np.asarray(minimums, dtype=float)

    return pd.DataFrame({
        "Mã biến": ITEM_CODES,
        "Hạng mục": ITEM_NAMES,
        "Hệ số tác động": impacts,
        "Mức tối thiểu": minimums,
        "Phân bổ tối ưu": x,
        "Phần vượt mức sàn": x - minimums,
        "Đóng góp vào Z": x * impacts,
    })


def solve_with_scipy(
    budget: float = 100.0,
    minimums: np.ndarray = DEFAULT_MINIMUMS,
    strategic_share: float = 0.35,
    impacts: np.ndarray = DEFAULT_IMPACTS,
) -> dict[str, Any]:
    minimums = np.asarray(minimums, dtype=float)
    impacts = np.asarray(impacts, dtype=float)

    validate_inputs(
        budget=budget,
        minimums=minimums,
        strategic_share=strategic_share,
        impacts=impacts,
    )

    a_ub, b_ub = build_scipy_problem(
        budget=budget,
        minimums=minimums,
        strategic_share=strategic_share,
    )

    result = linprog(
        c=-impacts,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] * 4,
        method="highs",
    )

    if not result.success:
        return {
            "success": False,
            "status": str(result.message),
            "raw": result,
        }

    x = np.asarray(result.x, dtype=float)
    total_used = float(x.sum())
    strategic_amount = float(x[1] + x[3])
    strategic_ratio = (
        strategic_amount / total_used
        if total_used > 0
        else np.nan
    )

    constraint_names = [
        "Ngân sách tổng",
        "Sàn hạ tầng số",
        "Sàn AI và dữ liệu",
        "Sàn nhân lực số",
        "Sàn R&D công nghệ",
        "Tỷ trọng AI + R&D",
    ]

    residuals = np.asarray(
        result.ineqlin.residual,
        dtype=float,
    )

    marginals = np.asarray(
        result.ineqlin.marginals,
        dtype=float,
    )

    scipy_duals = pd.DataFrame({
        "Ràng buộc": constraint_names,
        "Slack theo dạng scipy": residuals,
        "Marginal của bài toán Min": marginals,
    })

    scipy_duals[
        "Shadow price của Max theo RHS"
    ] = -scipy_duals[
        "Marginal của bài toán Min"
    ]

    return {
        "success": True,
        "status": "Tối ưu",
        "x": x,
        "z": float(-result.fun),
        "total_used": total_used,
        "unused_budget": float(budget - total_used),
        "strategic_amount": strategic_amount,
        "strategic_ratio": float(strategic_ratio),
        "table": allocation_table(
            x=x,
            impacts=impacts,
            minimums=minimums,
        ),
        "constraint_table": scipy_duals,
        "raw": result,
    }


def solve_with_pulp(
    budget: float = 100.0,
    minimums: np.ndarray = DEFAULT_MINIMUMS,
    strategic_share: float = 0.35,
    impacts: np.ndarray = DEFAULT_IMPACTS,
) -> dict[str, Any]:
    minimums = np.asarray(minimums, dtype=float)
    impacts = np.asarray(impacts, dtype=float)

    validate_inputs(
        budget=budget,
        minimums=minimums,
        strategic_share=strategic_share,
        impacts=impacts,
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

    model = pulp.LpProblem(
        "Bai2_LP_NganSachSo",
        pulp.LpMaximize,
    )

    variables = [
        pulp.LpVariable(
            code,
            lowBound=0,
            cat="Continuous",
        )
        for code in ITEM_CODES
    ]

    model += (
        pulp.lpSum(
            impacts[index]
            * variables[index]
            for index in range(4)
        ),
        "GDP_ky_vong",
    )

    model += (
        pulp.lpSum(variables)
        <= float(budget),
        "Ngan_sach_tong",
    )

    for index, code in enumerate(ITEM_CODES):
        model += (
            variables[index]
            >= float(minimums[index]),
            f"San_{code}",
        )

    model += (
        variables[1]
        + variables[3]
        >= float(strategic_share)
        * pulp.lpSum(variables),
        "Ty_trong_AI_RD",
    )

    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = model.solve(solver)

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

    x = np.array(
        [
            float(variable.value())
            for variable in variables
        ],
        dtype=float,
    )

    relation_map = {
        -1: "<=",
        0: "=",
        1: ">=",
    }

    dual_rows = []

    for name, constraint in model.constraints.items():
        dual_value = getattr(
            constraint,
            "pi",
            np.nan,
        )

        slack_value = getattr(
            constraint,
            "slack",
            np.nan,
        )

        dual_rows.append({
            "Ràng buộc": name,
            "Quan hệ": relation_map.get(
                constraint.sense,
                str(constraint.sense),
            ),
            "Shadow price": (
                float(dual_value)
                if dual_value is not None
                else np.nan
            ),
            "Slack": (
                float(slack_value)
                if slack_value is not None
                else np.nan
            ),
        })

    dual_table = pd.DataFrame(dual_rows)
    total_used = float(x.sum())

    return {
        "success": True,
        "status": status,
        "x": x,
        "z": float(pulp.value(model.objective)),
        "total_used": total_used,
        "unused_budget": float(budget - total_used),
        "strategic_amount": float(x[1] + x[3]),
        "strategic_ratio": float(
            (x[1] + x[3]) / total_used
            if total_used > 0
            else np.nan
        ),
        "table": allocation_table(
            x=x,
            impacts=impacts,
            minimums=minimums,
        ),
        "dual_table": dual_table,
        "model": model,
    }


def sensitivity_analysis(
    budgets: Iterable[float] = (
        100.0,
        120.0,
        140.0,
    ),
    minimums: np.ndarray = DEFAULT_MINIMUMS,
    strategic_share: float = 0.35,
    impacts: np.ndarray = DEFAULT_IMPACTS,
) -> pd.DataFrame:
    rows = []
    previous_budget = None
    previous_z = None

    for budget in budgets:
        result = solve_with_scipy(
            budget=float(budget),
            minimums=minimums,
            strategic_share=strategic_share,
            impacts=impacts,
        )

        if result["success"]:
            z_value = float(result["z"])

            marginal_gain = (
                np.nan
                if previous_z is None
                else (
                    z_value - previous_z
                ) / (
                    float(budget)
                    - float(previous_budget)
                )
            )

            rows.append({
                "Ngân sách B": float(budget),
                "Z*": z_value,
                "Trạng thái": "Tối ưu",
                "GDP tăng thêm trên 1 đơn vị B":
                    marginal_gain,
                "Tổng ngân sách sử dụng":
                    result["total_used"],
            })

            previous_budget = float(budget)
            previous_z = z_value

        else:
            rows.append({
                "Ngân sách B": float(budget),
                "Z*": np.nan,
                "Trạng thái": result["status"],
                "GDP tăng thêm trên 1 đơn vị B":
                    np.nan,
                "Tổng ngân sách sử dụng":
                    np.nan,
            })

    return pd.DataFrame(rows)


def human_capital_priority_scenario(
    budget: float = 100.0,
    minimums: np.ndarray = DEFAULT_MINIMUMS,
    priority_human_floor: float = 30.0,
    strategic_share: float = 0.35,
    impacts: np.ndarray = DEFAULT_IMPACTS,
) -> dict[str, Any]:
    minimums = np.asarray(minimums, dtype=float)

    priority_minimums = minimums.copy()
    priority_minimums[2] = max(
        float(priority_human_floor),
        float(priority_minimums[2]),
    )

    baseline = solve_with_scipy(
        budget=budget,
        minimums=minimums,
        strategic_share=strategic_share,
        impacts=impacts,
    )

    priority = solve_with_scipy(
        budget=budget,
        minimums=priority_minimums,
        strategic_share=strategic_share,
        impacts=impacts,
    )

    comparison_rows = []

    if baseline["success"]:
        for index, item in enumerate(ITEM_NAMES):
            comparison_rows.append({
                "Hạng mục": item,
                "Cơ sở": baseline["x"][index],
                "Ưu tiên nhân lực": (
                    priority["x"][index]
                    if priority["success"]
                    else np.nan
                ),
            })

    comparison = pd.DataFrame(comparison_rows)

    objective_change = (
        float(
            priority["z"]
            - baseline["z"]
        )
        if (
            baseline["success"]
            and priority["success"]
        )
        else np.nan
    )

    return {
        "baseline": baseline,
        "priority": priority,
        "priority_minimums":
            priority_minimums,
        "comparison": comparison,
        "objective_change":
            objective_change,
    }


def run_full_bai02(
    budget: float = 100.0,
    minimums: np.ndarray = DEFAULT_MINIMUMS,
    strategic_share: float = 0.35,
    impacts: np.ndarray = DEFAULT_IMPACTS,
    sensitivity_budgets: Iterable[float] = (
        100.0,
        120.0,
        140.0,
    ),
    priority_human_floor: float = 30.0,
) -> dict[str, Any]:
    scipy_result = solve_with_scipy(
        budget=budget,
        minimums=minimums,
        strategic_share=strategic_share,
        impacts=impacts,
    )

    pulp_result = solve_with_pulp(
        budget=budget,
        minimums=minimums,
        strategic_share=strategic_share,
        impacts=impacts,
    )

    sensitivity = sensitivity_analysis(
        budgets=sensitivity_budgets,
        minimums=minimums,
        strategic_share=strategic_share,
        impacts=impacts,
    )

    human_priority = human_capital_priority_scenario(
        budget=budget,
        minimums=minimums,
        priority_human_floor=priority_human_floor,
        strategic_share=strategic_share,
        impacts=impacts,
    )

    return {
        "scipy": scipy_result,
        "pulp": pulp_result,
        "sensitivity": sensitivity,
        "human_priority": human_priority,
    }
