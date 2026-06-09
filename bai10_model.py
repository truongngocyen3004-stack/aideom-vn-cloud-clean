from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ITEMS = ["K", "D", "AI", "H"]
ITEM_NAMES = {
    "K": "Vốn/hạ tầng truyền thống",
    "D": "Chuyển đổi số",
    "AI": "AI và dữ liệu",
    "H": "Nhân lực số",
}
FIRST_STAGE_BETA = {
    "K": 0.95,
    "D": 1.10,
    "AI": 1.20,
    "H": 1.05,
}


def load_scenarios(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu kịch bản: {path}")

    data = pd.read_csv(path)
    required = [
        "scenario_code",
        "scenario_name",
        "probability",
        "beta_K",
        "beta_D",
        "beta_AI",
        "beta_H",
        "recourse_budget",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Thiếu cột: " + ", ".join(missing))

    probability_sum = float(data["probability"].sum())
    if not np.isclose(probability_sum, 1.0):
        data["probability"] = data["probability"] / probability_sum

    return data.reset_index(drop=True)


def _scenario_beta(row: pd.Series) -> dict[str, float]:
    return {
        "K": float(row["beta_K"]),
        "D": float(row["beta_D"]),
        "AI": float(row["beta_AI"]),
        "H": float(row["beta_H"]),
    }


def solve_stochastic_program(
    scenarios: pd.DataFrame,
    first_stage_budget: float = 30000.0,
    min_h_first_stage: float = 5000.0,
    min_dai_first_stage: float = 8000.0,
    adjustment_penalty: float = 0.05,
    fixed_x: dict[str, float] | None = None,
) -> dict[str, Any]:
    try:
        import pulp
    except ImportError as error:
        return {
            "success": False,
            "status": "Chưa cài PuLP.",
            "error": str(error),
        }

    model = pulp.LpProblem("Two_Stage_Stochastic_Vietnam", pulp.LpMaximize)

    x = {
        j: pulp.LpVariable(f"x_{j}", lowBound=0)
        for j in ITEMS
    }

    scenario_codes = scenarios["scenario_code"].tolist()
    y = {
        (s, j): pulp.LpVariable(f"y_{s}_{j}", lowBound=0)
        for s in scenario_codes
        for j in ITEMS
    }

    if fixed_x is None:
        model += pulp.lpSum(x[j] for j in ITEMS) <= first_stage_budget, "First_stage_budget"
        model += x["H"] >= min_h_first_stage, "First_stage_H_floor"
        model += x["D"] + x["AI"] >= min_dai_first_stage, "First_stage_DAI_floor"
    else:
        for j in ITEMS:
            model += x[j] == float(fixed_x[j]), f"Fix_x_{j}"

    expected_recourse = []
    for _, row in scenarios.iterrows():
        s = row["scenario_code"]
        beta_s = _scenario_beta(row)
        model += (
            pulp.lpSum(y[(s, j)] for j in ITEMS)
            <= float(row["recourse_budget"]),
            f"Recourse_budget_{s}",
        )
        model += y[(s, "H")] >= 0.12 * float(row["recourse_budget"]), f"Recourse_H_floor_{s}"
        model += (
            y[(s, "D")] + y[(s, "AI")]
            >= 0.25 * float(row["recourse_budget"]),
            f"Recourse_DAI_floor_{s}",
        )

        scenario_value = pulp.lpSum(
            beta_s[j] * y[(s, j)] - adjustment_penalty * y[(s, j)]
            for j in ITEMS
        )
        expected_recourse.append(float(row["probability"]) * scenario_value)

    first_stage_value = pulp.lpSum(
        FIRST_STAGE_BETA[j] * x[j]
        for j in ITEMS
    )

    model += first_stage_value + pulp.lpSum(expected_recourse), "Expected_total_value"

    status_code = model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus.get(status_code, str(status_code))

    if status != "Optimal":
        return {
            "success": False,
            "status": status,
            "objective": np.nan,
        }

    x_value = {j: float(x[j].value() or 0.0) for j in ITEMS}
    y_value = {
        s: {j: float(y[(s, j)].value() or 0.0) for j in ITEMS}
        for s in scenario_codes
    }

    scenario_rows = []
    for _, row in scenarios.iterrows():
        s = row["scenario_code"]
        beta_s = _scenario_beta(row)
        first_value = sum(FIRST_STAGE_BETA[j] * x_value[j] for j in ITEMS)
        recourse_value = sum(
            (beta_s[j] - adjustment_penalty) * y_value[s][j]
            for j in ITEMS
        )
        scenario_rows.append({
            "scenario_code": s,
            "scenario_name": row["scenario_name"],
            "probability": float(row["probability"]),
            "first_stage_value": first_value,
            "recourse_value": recourse_value,
            "total_value_if_s": first_value + recourse_value,
            **{f"x_{j}": x_value[j] for j in ITEMS},
            **{f"y_{j}": y_value[s][j] for j in ITEMS},
        })

    return {
        "success": True,
        "status": status,
        "objective": float(pulp.value(model.objective)),
        "x": x_value,
        "y": y_value,
        "scenario_table": pd.DataFrame(scenario_rows),
        "model": model,
    }


def solve_single_scenario(
    row: pd.Series,
    first_stage_budget: float = 30000.0,
    min_h_first_stage: float = 5000.0,
    min_dai_first_stage: float = 8000.0,
    adjustment_penalty: float = 0.05,
) -> dict[str, Any]:
    try:
        import pulp
    except ImportError as error:
        return {"success": False, "status": "Chưa cài PuLP.", "error": str(error)}

    beta_s = _scenario_beta(row)
    model = pulp.LpProblem(f"Deterministic_{row['scenario_code']}", pulp.LpMaximize)

    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0) for j in ITEMS}
    y = {j: pulp.LpVariable(f"y_{j}", lowBound=0) for j in ITEMS}

    model += pulp.lpSum(x[j] for j in ITEMS) <= first_stage_budget
    model += x["H"] >= min_h_first_stage
    model += x["D"] + x["AI"] >= min_dai_first_stage

    model += pulp.lpSum(y[j] for j in ITEMS) <= float(row["recourse_budget"])
    model += y["H"] >= 0.12 * float(row["recourse_budget"])
    model += y["D"] + y["AI"] >= 0.25 * float(row["recourse_budget"])

    model += (
        pulp.lpSum(FIRST_STAGE_BETA[j] * x[j] for j in ITEMS)
        + pulp.lpSum((beta_s[j] - adjustment_penalty) * y[j] for j in ITEMS)
    )

    status_code = model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus.get(status_code, str(status_code))

    if status != "Optimal":
        return {"success": False, "status": status, "objective": np.nan}

    return {
        "success": True,
        "status": status,
        "objective": float(pulp.value(model.objective)),
        "x": {j: float(x[j].value() or 0.0) for j in ITEMS},
        "y": {j: float(y[j].value() or 0.0) for j in ITEMS},
    }


def solve_expected_value_model(
    scenarios: pd.DataFrame,
    first_stage_budget: float = 30000.0,
    min_h_first_stage: float = 5000.0,
    min_dai_first_stage: float = 8000.0,
    adjustment_penalty: float = 0.05,
) -> dict[str, Any]:
    mean_row = {
        "scenario_code": "EV",
        "scenario_name": "Kịch bản kỳ vọng",
        "probability": 1.0,
        "recourse_budget": float(
            np.average(
                scenarios["recourse_budget"],
                weights=scenarios["probability"],
            )
        ),
    }

    for item in ITEMS:
        mean_row[f"beta_{item}"] = float(
            np.average(
                scenarios[f"beta_{item}"],
                weights=scenarios["probability"],
            )
        )

    result = solve_single_scenario(
        pd.Series(mean_row),
        first_stage_budget=first_stage_budget,
        min_h_first_stage=min_h_first_stage,
        min_dai_first_stage=min_dai_first_stage,
        adjustment_penalty=adjustment_penalty,
    )
    result["mean_row"] = mean_row
    return result


def solve_robust_regret(
    scenarios: pd.DataFrame,
    scenario_optimum: dict[str, float],
    first_stage_budget: float = 30000.0,
    min_h_first_stage: float = 5000.0,
    min_dai_first_stage: float = 8000.0,
    adjustment_penalty: float = 0.05,
) -> dict[str, Any]:
    try:
        import pulp
    except ImportError as error:
        return {"success": False, "status": "Chưa cài PuLP.", "error": str(error)}

    model = pulp.LpProblem("Minimax_Regret_Vietnam", pulp.LpMinimize)
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0) for j in ITEMS}
    M = pulp.LpVariable("Maximum_regret", lowBound=0)

    y = {
        (s, j): pulp.LpVariable(f"y_{s}_{j}", lowBound=0)
        for s in scenarios["scenario_code"]
        for j in ITEMS
    }

    model += pulp.lpSum(x[j] for j in ITEMS) <= first_stage_budget
    model += x["H"] >= min_h_first_stage
    model += x["D"] + x["AI"] >= min_dai_first_stage

    for _, row in scenarios.iterrows():
        s = row["scenario_code"]
        beta_s = _scenario_beta(row)
        model += (
            pulp.lpSum(y[(s, j)] for j in ITEMS)
            <= float(row["recourse_budget"])
        )
        model += y[(s, "H")] >= 0.12 * float(row["recourse_budget"])
        model += (
            y[(s, "D")] + y[(s, "AI")]
            >= 0.25 * float(row["recourse_budget"])
        )

        achieved = (
            pulp.lpSum(FIRST_STAGE_BETA[j] * x[j] for j in ITEMS)
            + pulp.lpSum(
                (beta_s[j] - adjustment_penalty) * y[(s, j)]
                for j in ITEMS
            )
        )
        model += float(scenario_optimum[s]) - achieved <= M

    model += M

    status_code = model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus.get(status_code, str(status_code))

    if status != "Optimal":
        return {"success": False, "status": status}

    x_value = {j: float(x[j].value() or 0.0) for j in ITEMS}
    rows = []

    for _, row in scenarios.iterrows():
        s = row["scenario_code"]
        beta_s = _scenario_beta(row)
        first_value = sum(FIRST_STAGE_BETA[j] * x_value[j] for j in ITEMS)
        recourse_value = sum(
            (beta_s[j] - adjustment_penalty)
            * float(y[(s, j)].value() or 0.0)
            for j in ITEMS
        )
        achieved = first_value + recourse_value
        rows.append({
            "scenario_code": s,
            "scenario_name": row["scenario_name"],
            "scenario_optimum": scenario_optimum[s],
            "robust_value": achieved,
            "regret": scenario_optimum[s] - achieved,
        })

    regret_table = pd.DataFrame(rows)

    return {
        "success": True,
        "status": status,
        "maximum_regret": float(M.value() or 0.0),
        "x": x_value,
        "regret_table": regret_table,
        "expected_value": float(
            np.average(
                regret_table["robust_value"],
                weights=scenarios["probability"],
            )
        ),
    }


def run_full_bai10(
    csv_path: str | Path,
    first_stage_budget: float = 30000.0,
    adjustment_penalty: float = 0.05,
) -> dict[str, Any]:
    scenarios = load_scenarios(csv_path)

    sp = solve_stochastic_program(
        scenarios=scenarios,
        first_stage_budget=first_stage_budget,
        adjustment_penalty=adjustment_penalty,
    )

    if not sp.get("success", False):
        return {
            "scenarios": scenarios,
            "sp": sp,
            "deterministic": pd.DataFrame(),
            "ev": {
                "success": False,
                "status": sp.get("status", "Không chạy được"),
            },
            "eev": {
                "success": False,
                "status": sp.get("status", "Không chạy được"),
            },
            "metrics": pd.DataFrame(),
            "VSS": np.nan,
            "EVPI": np.nan,
            "WS": np.nan,
            "RP": np.nan,
            "robust": {
                "success": False,
                "status": sp.get("status", "Không chạy được"),
            },
            "x_compare": pd.DataFrame(),
        }

    deterministic_results = {}
    deterministic_rows = []

    for _, row in scenarios.iterrows():
        result = solve_single_scenario(
            row,
            first_stage_budget=first_stage_budget,
            adjustment_penalty=adjustment_penalty,
        )
        deterministic_results[row["scenario_code"]] = result
        deterministic_rows.append({
            "scenario_code": row["scenario_code"],
            "scenario_name": row["scenario_name"],
            "probability": row["probability"],
            "objective": result.get("objective", np.nan),
            **{f"x_{j}": result.get("x", {}).get(j, np.nan) for j in ITEMS},
            **{f"y_{j}": result.get("y", {}).get(j, np.nan) for j in ITEMS},
        })

    deterministic_table = pd.DataFrame(deterministic_rows)

    ev = solve_expected_value_model(
        scenarios=scenarios,
        first_stage_budget=first_stage_budget,
        adjustment_penalty=adjustment_penalty,
    )

    eev = solve_stochastic_program(
        scenarios=scenarios,
        first_stage_budget=first_stage_budget,
        adjustment_penalty=adjustment_penalty,
        fixed_x=ev["x"],
    )

    scenario_optimum = {
        code: result["objective"]
        for code, result in deterministic_results.items()
    }

    WS = float(
        sum(
            float(row["probability"])
            * scenario_optimum[row["scenario_code"]]
            for _, row in scenarios.iterrows()
        )
    )
    RP = float(sp["objective"])
    EEV = float(eev["objective"])
    VSS = RP - EEV
    EVPI = WS - RP

    robust = solve_robust_regret(
        scenarios=scenarios,
        scenario_optimum=scenario_optimum,
        first_stage_budget=first_stage_budget,
        adjustment_penalty=adjustment_penalty,
    )

    metrics = pd.DataFrame([
        ["RP — Stochastic solution", RP, "Lời giải xét toàn bộ phân phối kịch bản"],
        ["EV — Expected value model", float(ev["objective"]), "Lời giải theo kịch bản trung bình"],
        ["EEV — EV evaluated stochastically", EEV, "Giá trị thực khi dùng x_EV trong bất định"],
        ["WS — Perfect information", WS, "Biết trước hoàn hảo kịch bản"],
        ["VSS = RP - EEV", VSS, "Giá trị của mô hình ngẫu nhiên"],
        ["EVPI = WS - RP", EVPI, "Giá trị tối đa của thông tin hoàn hảo"],
    ], columns=["Chỉ tiêu", "Giá trị", "Ý nghĩa"])

    x_compare = pd.DataFrame({
        "Hạng mục": [ITEM_NAMES[j] for j in ITEMS],
        "SP": [sp["x"][j] for j in ITEMS],
        "EV": [ev["x"][j] for j in ITEMS],
        "Robust": [robust["x"][j] for j in ITEMS],
    })

    return {
        "scenarios": scenarios,
        "sp": sp,
        "deterministic": deterministic_table,
        "ev": ev,
        "eev": eev,
        "metrics": metrics,
        "VSS": VSS,
        "EVPI": EVPI,
        "WS": WS,
        "RP": RP,
        "robust": robust,
        "x_compare": x_compare,
    }
