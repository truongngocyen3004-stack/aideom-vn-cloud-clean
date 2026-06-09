from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


REGION_CODES = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
REGION_NAMES = [
    "Trung du miền núi phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ + DH Trung Bộ",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long",
]
ITEM_CODES = ["I", "D", "AI", "H"]
ITEM_NAMES = ["Hạ tầng số", "Dữ liệu/CĐS", "Trí tuệ nhân tạo", "Nhân lực số"]

BETA = np.array([
    [1.15, 0.85, 0.55, 1.30],
    [0.95, 1.25, 1.40, 1.05],
    [1.05, 0.95, 0.85, 1.15],
    [1.20, 0.75, 0.45, 1.35],
    [0.90, 1.30, 1.55, 1.00],
    [1.10, 0.85, 0.65, 1.25],
], dtype=float)

EMISSION = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38], dtype=float)
SECURITY_RISK = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22], dtype=float)
SECURITY_REDUCTION = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30], dtype=float)

DEFAULT_POLICY_WEIGHTS = np.array([0.40, 0.25, 0.20, 0.15], dtype=float)


@dataclass(frozen=True)
class ParetoConfig:
    total_budget: float = 50000.0
    min_region: float = 5000.0
    max_region: float = 13000.0
    min_h_total: float = 12000.0
    min_d_total: float = 8000.0
    pop_size: int = 100
    n_gen: int = 200
    seed: int = 42


def mean_absolute_deviation(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mean_value = float(values.mean())
    if np.isclose(mean_value, 0.0):
        return 0.0
    return float(np.mean(np.abs(values - mean_value)) / mean_value)


def evaluate_allocation(x_flat: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x_flat, dtype=float).reshape(6, 4)
    region_total = x.sum(axis=1)

    growth = float(np.sum(BETA * x))
    inequality = mean_absolute_deviation(region_total)
    emission = float(np.sum(EMISSION * (x[:, 0] + x[:, 2])))
    security = float(np.sum(SECURITY_RISK * x[:, 2] - SECURITY_REDUCTION * x[:, 3]))

    return {
        "growth_gain": growth,
        "inequality": inequality,
        "emission": emission,
        "security_risk": security,
        "region_total": region_total,
        "item_total": x.sum(axis=0),
        "allocation_matrix": x,
    }


def constraint_vector(x_flat: np.ndarray, config: ParetoConfig) -> np.ndarray:
    x = np.asarray(x_flat, dtype=float).reshape(6, 4)
    region_total = x.sum(axis=1)

    constraints = [
        x.sum() - config.total_budget,
        *list(config.min_region - region_total),
        *list(region_total - config.max_region),
        config.min_h_total - x[:, 3].sum(),
        config.min_d_total - x[:, 1].sum(),
    ]
    return np.asarray(constraints, dtype=float)


def _run_pymoo(config: ParetoConfig) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
    except ImportError as error:
        raise ImportError(
            "Chưa cài pymoo. Chạy: python -m pip install pymoo"
        ) from error

    class VietnamDigitalProblem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(
                n_var=24,
                n_obj=4,
                n_ieq_constr=15,
                xl=np.zeros(24),
                xu=np.ones(24) * config.max_region,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            metrics = evaluate_allocation(x)
            out["F"] = np.array([
                -metrics["growth_gain"],
                metrics["inequality"],
                metrics["emission"],
                metrics["security_risk"],
            ])
            out["G"] = constraint_vector(x, config)

    algorithm = NSGA2(
        pop_size=int(config.pop_size),
        eliminate_duplicates=True,
    )

    result = minimize(
        VietnamDigitalProblem(),
        algorithm,
        termination=("n_gen", int(config.n_gen)),
        seed=int(config.seed),
        verbose=False,
    )

    if result.X is None or result.F is None:
        raise RuntimeError(
            "NSGA-II không trả về nghiệm khả thi. "
            "Hãy tăng số thế hệ hoặc nới ràng buộc."
        )

    return np.asarray(result.X), np.asarray(result.F), "pymoo NSGA-II"


def _sample_feasible(config: ParetoConfig, n_samples: int = 4000) -> np.ndarray:
    """Fallback khi chưa có pymoo: lấy mẫu khả thi có định hướng."""

    rng = np.random.default_rng(config.seed)
    samples = []

    for _ in range(n_samples):
        region_extra = rng.dirichlet(np.ones(6))
        region_budget = (
            np.full(6, config.min_region)
            + region_extra
            * (config.total_budget - 6 * config.min_region)
        )
        region_budget = np.minimum(region_budget, config.max_region)

        remaining = config.total_budget - region_budget.sum()
        if remaining > 1e-6:
            slack = config.max_region - region_budget
            if slack.sum() > 0:
                region_budget += remaining * slack / slack.sum()

        x = np.zeros((6, 4))
        for r in range(6):
            shares = rng.dirichlet(np.array([1.2, 1.2, 1.0, 1.4]))
            x[r] = region_budget[r] * shares

        if x[:, 3].sum() < config.min_h_total:
            deficit = config.min_h_total - x[:, 3].sum()
            donor = np.argmax(x[:, 0] + x[:, 2])
            amount = min(deficit, x[donor, 0] + x[donor, 2])
            move_i = min(amount, x[donor, 0])
            x[donor, 0] -= move_i
            x[donor, 3] += move_i
            amount -= move_i
            if amount > 0:
                x[donor, 2] -= amount
                x[donor, 3] += amount

        if x[:, 1].sum() < config.min_d_total:
            deficit = config.min_d_total - x[:, 1].sum()
            donor = np.argmax(x[:, 0] + x[:, 2])
            amount = min(deficit, x[donor, 0] + x[donor, 2])
            move_i = min(amount, x[donor, 0])
            x[donor, 0] -= move_i
            x[donor, 1] += move_i
            amount -= move_i
            if amount > 0:
                x[donor, 2] -= amount
                x[donor, 1] += amount

        if np.all(constraint_vector(x.ravel(), config) <= 1e-6):
            samples.append(x.ravel())

    if not samples:
        raise RuntimeError("Không tạo được nghiệm khả thi trong fallback.")

    return np.asarray(samples)


def _non_dominated_mask(cost_matrix: np.ndarray) -> np.ndarray:
    n = cost_matrix.shape[0]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        dominated_by_any = np.any(
            np.all(cost_matrix <= cost_matrix[i], axis=1)
            & np.any(cost_matrix < cost_matrix[i], axis=1)
        )
        if dominated_by_any:
            keep[i] = False

    return keep


def _run_fallback(config: ParetoConfig) -> tuple[np.ndarray, np.ndarray, str]:
    X = _sample_feasible(config)
    metrics = [evaluate_allocation(row) for row in X]
    F = np.array([
        [-m["growth_gain"], m["inequality"], m["emission"], m["security_risk"]]
        for m in metrics
    ])
    mask = _non_dominated_mask(F)
    X_nd = X[mask]
    F_nd = F[mask]

    order = np.argsort(F_nd[:, 0])
    max_keep = min(len(order), max(config.pop_size, 40))
    order = order[:max_keep]
    return X_nd[order], F_nd[order], "Fallback Pareto sampling"


def build_pareto_table(X: np.ndarray, F: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, (x_flat, f) in enumerate(zip(X, F), start=1):
        metrics = evaluate_allocation(x_flat)
        row = {
            "solution_id": idx,
            "growth_gain": metrics["growth_gain"],
            "inequality": metrics["inequality"],
            "emission": metrics["emission"],
            "security_risk": metrics["security_risk"],
            "total_budget_used": float(np.sum(x_flat)),
            "I_total": metrics["item_total"][0],
            "D_total": metrics["item_total"][1],
            "AI_total": metrics["item_total"][2],
            "H_total": metrics["item_total"][3],
        }
        for r, region in enumerate(REGION_CODES):
            row[f"{region}_total"] = metrics["region_total"][r]
        rows.append(row)
    return pd.DataFrame(rows)


def topsis_compromise(
    pareto_df: pd.DataFrame,
    policy_weights: np.ndarray = DEFAULT_POLICY_WEIGHTS,
) -> dict[str, Any]:
    weights = np.asarray(policy_weights, dtype=float)
    if weights.shape != (4,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Cần 4 trọng số không âm và có tổng dương.")
    weights = weights / weights.sum()

    criteria = pareto_df[
        ["growth_gain", "inequality", "emission", "security_risk"]
    ].to_numpy(dtype=float)

    normalized = np.zeros_like(criteria)
    for j in range(criteria.shape[1]):
        col = criteria[:, j]
        low, high = float(col.min()), float(col.max())
        if np.isclose(high, low):
            normalized[:, j] = 1.0
        elif j == 0:
            normalized[:, j] = (col - low) / (high - low)
        else:
            normalized[:, j] = (high - col) / (high - low)

    weighted = normalized * weights
    ideal = weighted.max(axis=0)
    anti = weighted.min(axis=0)
    d_pos = np.sqrt(np.square(weighted - ideal).sum(axis=1))
    d_neg = np.sqrt(np.square(weighted - anti).sum(axis=1))
    score = d_neg / np.maximum(d_pos + d_neg, 1e-12)

    scored = pareto_df.copy()
    scored["TOPSIS_score"] = score
    scored["TOPSIS_rank"] = (
        scored["TOPSIS_score"].rank(ascending=False, method="min").astype(int)
    )
    scored = scored.sort_values("TOPSIS_score", ascending=False).reset_index(drop=True)

    return {
        "scored": scored,
        "compromise": scored.iloc[0].to_dict(),
        "weights": weights,
    }


def opportunity_cost(pareto_df: pd.DataFrame, compromise: dict[str, Any]) -> dict[str, float]:
    growth_best = pareto_df.loc[pareto_df["growth_gain"].idxmax()]

    def pct_worse(cost_value: float, benchmark: float) -> float:
        denominator = abs(benchmark) if abs(benchmark) > 1e-12 else 1.0
        return (cost_value - benchmark) / denominator * 100.0

    return {
        "growth_advantage_pct": (
            (growth_best["growth_gain"] - compromise["growth_gain"])
            / max(abs(compromise["growth_gain"]), 1e-12)
            * 100.0
        ),
        "inclusion_sacrifice_pct": pct_worse(
            growth_best["inequality"], compromise["inequality"]
        ),
        "environment_sacrifice_pct": pct_worse(
            growth_best["emission"], compromise["emission"]
        ),
        "security_sacrifice_pct": pct_worse(
            growth_best["security_risk"], compromise["security_risk"]
        ),
        "growth_solution_id": int(growth_best["solution_id"]),
    }


def allocation_from_solution(
    X: np.ndarray,
    solution_id: int,
) -> pd.DataFrame:
    idx = int(solution_id) - 1
    matrix = X[idx].reshape(6, 4)
    return pd.DataFrame(matrix, index=REGION_NAMES, columns=ITEM_NAMES)


def run_full_bai07(
    config: ParetoConfig = ParetoConfig(),
    policy_weights: np.ndarray = DEFAULT_POLICY_WEIGHTS,
) -> dict[str, Any]:
    try:
        X, F, engine = _run_pymoo(config)
    except ImportError:
        X, F, engine = _run_fallback(config)

    pareto_df = build_pareto_table(X, F)
    topsis = topsis_compromise(pareto_df, policy_weights)
    compromise = topsis["compromise"]
    costs = opportunity_cost(pareto_df, compromise)

    compromise_allocation = allocation_from_solution(
        X, int(compromise["solution_id"])
    )

    growth_solution_id = costs["growth_solution_id"]
    growth_allocation = allocation_from_solution(X, growth_solution_id)

    return {
        "engine": engine,
        "config": config,
        "X": X,
        "F": F,
        "pareto": pareto_df,
        "topsis": topsis,
        "opportunity_cost": costs,
        "compromise_allocation": compromise_allocation,
        "growth_allocation": growth_allocation,
    }
