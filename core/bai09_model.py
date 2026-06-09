from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_labor_data(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")

    data = pd.read_csv(path)
    required = [
        "sector_id",
        "sector",
        "labor_million",
        "risk_pct",
        "a1_new_ai_job_per_billion",
        "a2_new_digital_job_per_billion",
        "b1_upgrade_job_per_billion",
        "c1_displace_job_per_billion",
        "d1_retrain_capacity_per_billion",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Thiếu cột: " + ", ".join(missing))
    if len(data) != 8:
        raise ValueError("Bài 9 yêu cầu đúng 8 ngành.")
    return data.copy()


def compute_jobs(
    data: pd.DataFrame,
    x_ai: np.ndarray,
    x_h: np.ndarray,
    digital_complement: float = 0.40,
) -> pd.DataFrame:
    result = data.copy()
    result["x_AI"] = np.asarray(x_ai, dtype=float)
    result["x_H"] = np.asarray(x_h, dtype=float)

    result["NewJob_AI"] = (
        result["a1_new_ai_job_per_billion"] * result["x_AI"]
        + result["a2_new_digital_job_per_billion"]
        * digital_complement
        * result["x_AI"]
    )
    result["UpgradeJob"] = (
        result["b1_upgrade_job_per_billion"] * result["x_H"]
    )
    result["DisplacedJob"] = (
        result["c1_displace_job_per_billion"]
        * result["x_AI"]
        * result["risk_pct"]
        / 100.0
    )
    result["RetrainingCapacity"] = (
        result["d1_retrain_capacity_per_billion"] * result["x_H"]
    )
    result["NetJob"] = (
        result["NewJob_AI"]
        + result["UpgradeJob"]
        - result["DisplacedJob"]
    )
    result["AutomationPressure"] = (
        result["DisplacedJob"]
        / np.maximum(result["RetrainingCapacity"], 1e-9)
    )
    result["labor_jobs"] = result["labor_million"] * 1_000_000
    return result


def solve_labor_lp(
    data: pd.DataFrame,
    total_budget: float = 30000.0,
    digital_complement: float = 0.40,
    max_sector_share: float = 0.28,
    min_ai_share: float = 0.25,
    min_h_share: float = 0.30,
    min_vulnerable_h_share: float = 0.20,
    min_manufacturing_h: float = 2000.0,
    add_5pct_cap: bool = False,
) -> dict[str, Any]:
    try:
        import pulp
    except ImportError as error:
        return {
            "success": False,
            "status": "Chưa cài PuLP.",
            "error": str(error),
        }

    if total_budget <= 0:
        raise ValueError("Ngân sách phải lớn hơn 0.")

    ids = data["sector_id"].astype(int).tolist()
    rows = data.set_index("sector_id").to_dict("index")

    model = pulp.LpProblem("AI_Labor_Vietnam", pulp.LpMaximize)
    x_ai = pulp.LpVariable.dicts("x_AI", ids, lowBound=0)
    x_h = pulp.LpVariable.dicts("x_H", ids, lowBound=0)

    new_job = {}
    upgrade = {}
    displaced = {}
    retrain = {}
    netjob = {}

    for i in ids:
        row = rows[i]
        new_job[i] = (
            row["a1_new_ai_job_per_billion"] * x_ai[i]
            + row["a2_new_digital_job_per_billion"]
            * digital_complement
            * x_ai[i]
        )
        upgrade[i] = row["b1_upgrade_job_per_billion"] * x_h[i]
        displaced[i] = (
            row["c1_displace_job_per_billion"]
            * row["risk_pct"]
            / 100.0
            * x_ai[i]
        )
        retrain[i] = row["d1_retrain_capacity_per_billion"] * x_h[i]
        netjob[i] = new_job[i] + upgrade[i] - displaced[i]

    model += pulp.lpSum(netjob[i] for i in ids), "Total_NetJob"

    model += (
        pulp.lpSum(x_ai[i] + x_h[i] for i in ids) <= total_budget,
        "C1_Total_budget",
    )

    for i in ids:
        model += netjob[i] >= 0, f"C2_NetJob_nonnegative_{i}"
        model += displaced[i] <= retrain[i], f"C3_Retrain_capacity_{i}"
        model += (
            x_ai[i] + x_h[i] <= max_sector_share * total_budget,
            f"P1_Sector_cap_{i}",
        )

        if add_5pct_cap:
            labor_jobs = rows[i]["labor_million"] * 1_000_000
            model += displaced[i] <= 0.05 * labor_jobs, f"C4_Social_cap_{i}"

    model += (
        pulp.lpSum(x_ai[i] for i in ids) >= min_ai_share * total_budget,
        "P2_Min_AI_share",
    )
    model += (
        pulp.lpSum(x_h[i] for i in ids) >= min_h_share * total_budget,
        "P3_Min_H_share",
    )

    vulnerable_ids = [1, 2, 4, 6]
    model += (
        pulp.lpSum(x_h[i] for i in vulnerable_ids)
        >= min_vulnerable_h_share * total_budget,
        "P4_Vulnerable_training",
    )

    if 2 in ids and min_manufacturing_h > 0:
        model += x_h[2] >= min_manufacturing_h, "P5_Manufacturing_training"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = model.solve(solver)
    status = pulp.LpStatus.get(status_code, str(status_code))

    if status != "Optimal":
        return {
            "success": False,
            "status": status,
            "objective": np.nan,
            "result_df": pd.DataFrame(),
        }

    x_ai_value = np.array([float(x_ai[i].value() or 0.0) for i in ids])
    x_h_value = np.array([float(x_h[i].value() or 0.0) for i in ids])

    result_df = compute_jobs(
        data=data,
        x_ai=x_ai_value,
        x_h=x_h_value,
        digital_complement=digital_complement,
    )

    constraints = []
    for name, constraint in model.constraints.items():
        constraints.append({
            "Ràng buộc": name,
            "Slack": float(constraint.slack) if constraint.slack is not None else np.nan,
            "Shadow price": float(constraint.pi) if constraint.pi is not None else np.nan,
            "Binding?": bool(
                constraint.slack is not None
                and abs(float(constraint.slack)) <= 1e-5
            ),
        })

    summary = {
        "status": status,
        "objective_total_netjob": float(pulp.value(model.objective)),
        "total_budget_used": float(result_df["x_AI"].sum() + result_df["x_H"].sum()),
        "total_x_AI": float(result_df["x_AI"].sum()),
        "total_x_H": float(result_df["x_H"].sum()),
        "total_new_job": float(result_df["NewJob_AI"].sum()),
        "total_upgrade_job": float(result_df["UpgradeJob"].sum()),
        "total_displaced": float(result_df["DisplacedJob"].sum()),
        "total_retrain_capacity": float(result_df["RetrainingCapacity"].sum()),
        "min_netjob": float(result_df["NetJob"].min()),
        "max_pressure": float(result_df["AutomationPressure"].max()),
    }

    return {
        "success": True,
        "status": status,
        "objective": summary["objective_total_netjob"],
        "result_df": result_df,
        "summary": summary,
        "constraints_df": pd.DataFrame(constraints),
        "model": model,
    }


def minimum_training_threshold(
    data: pd.DataFrame,
    sector_id: int,
    ai_investment: float,
    digital_complement: float = 0.40,
) -> dict[str, float]:
    row = data.loc[data["sector_id"] == sector_id].iloc[0]

    new_per_ai = (
        row["a1_new_ai_job_per_billion"]
        + digital_complement * row["a2_new_digital_job_per_billion"]
    )
    displaced_per_ai = (
        row["c1_displace_job_per_billion"]
        * row["risk_pct"]
        / 100.0
    )

    min_h_for_netjob = max(
        0.0,
        (displaced_per_ai - new_per_ai)
        * ai_investment
        / row["b1_upgrade_job_per_billion"],
    )
    min_h_for_capacity = (
        displaced_per_ai
        * ai_investment
        / row["d1_retrain_capacity_per_billion"]
    )

    return {
        "min_h_for_netjob": float(min_h_for_netjob),
        "min_h_for_capacity": float(min_h_for_capacity),
        "required_h": float(max(min_h_for_netjob, min_h_for_capacity)),
    }


def sensitivity_budget_curve(
    data: pd.DataFrame,
    budgets: list[float] | None = None,
    add_5pct_cap: bool = False,
) -> pd.DataFrame:
    if budgets is None:
        budgets = [15000, 20000, 25000, 30000, 35000, 40000]

    rows = []
    for budget in budgets:
        result = solve_labor_lp(
            data=data,
            total_budget=float(budget),
            add_5pct_cap=add_5pct_cap,
            min_manufacturing_h=min(2000.0, 0.08 * float(budget)),
        )
        rows.append({
            "budget": float(budget),
            "status": result["status"],
            "total_netjob": result.get("objective", np.nan),
            "x_AI": (
                result["summary"]["total_x_AI"]
                if result.get("success")
                else np.nan
            ),
            "x_H": (
                result["summary"]["total_x_H"]
                if result.get("success")
                else np.nan
            ),
        })
    return pd.DataFrame(rows)


def run_full_bai09(
    csv_path: str | Path,
    total_budget: float = 30000.0,
) -> dict[str, Any]:
    data = load_labor_data(csv_path)
    base = solve_labor_lp(data, total_budget=total_budget, add_5pct_cap=False)
    social = solve_labor_lp(data, total_budget=total_budget, add_5pct_cap=True)

    manufacturing_ai = 4000.0
    threshold = minimum_training_threshold(
        data=data,
        sector_id=2,
        ai_investment=manufacturing_ai,
    )

    curve = sensitivity_budget_curve(data)

    compare = pd.DataFrame([
        {
            "Kịch bản": "Không trần 5%",
            "Trạng thái": base["status"],
            "Tổng NetJob": base.get("objective", np.nan),
            "Tổng displaced": (
                base["summary"]["total_displaced"]
                if base.get("success")
                else np.nan
            ),
            "x_AI": (
                base["summary"]["total_x_AI"]
                if base.get("success")
                else np.nan
            ),
            "x_H": (
                base["summary"]["total_x_H"]
                if base.get("success")
                else np.nan
            ),
        },
        {
            "Kịch bản": "Có trần 5%",
            "Trạng thái": social["status"],
            "Tổng NetJob": social.get("objective", np.nan),
            "Tổng displaced": (
                social["summary"]["total_displaced"]
                if social.get("success")
                else np.nan
            ),
            "x_AI": (
                social["summary"]["total_x_AI"]
                if social.get("success")
                else np.nan
            ),
            "x_H": (
                social["summary"]["total_x_H"]
                if social.get("success")
                else np.nan
            ),
        },
    ])

    return {
        "data": data,
        "base": base,
        "social_cap": social,
        "manufacturing_threshold": threshold,
        "sensitivity": curve,
        "comparison": compare,
    }
