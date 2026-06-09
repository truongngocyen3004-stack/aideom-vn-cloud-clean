from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class DynamicConfig:
    start_year: int = 2026
    horizon: int = 10
    rho: float = 0.97
    gamma_utility: float = 1.0

    alpha_K: float = 0.33
    alpha_L: float = 0.42
    alpha_D: float = 0.10
    alpha_AI: float = 0.08
    alpha_H: float = 0.07

    delta_K: float = 0.05
    delta_D: float = 0.12
    delta_AI: float = 0.15
    theta_H: float = 0.80
    brain_drain: float = 0.02

    phi_D: float = 0.003
    phi_AI: float = 0.002
    phi_H: float = 0.004

    labor_0: float = 53.9
    labor_growth: float = 0.004

    K0: float = 27500.0
    D0: float = 20.3
    AI0: float = 86.0
    H0: float = 30.0
    A0: float = 1.0

    scale_K: float = 1.0
    scale_D: float = 0.004
    scale_AI: float = 0.006
    scale_H: float = 0.003

    min_consumption_share: float = 0.55
    max_total_investment_share: float = 0.42
    max_single_investment_share: float = 0.20
    min_H_investment_share: float = 0.03
    min_DAI_investment_share: float = 0.04


def years(config: DynamicConfig) -> list[int]:
    return list(range(config.start_year, config.start_year + config.horizon))


def utility(consumption: np.ndarray | float, gamma: float = 1.0):
    value = np.maximum(consumption, 1e-9)
    if abs(gamma - 1.0) < 1e-9:
        return np.log(value)
    return (np.power(value, 1 - gamma) - 1) / (1 - gamma)


def production(A, K, L, D, AI, H, config: DynamicConfig):
    return (
        A
        * np.power(K, config.alpha_K)
        * np.power(L, config.alpha_L)
        * np.power(D, config.alpha_D)
        * np.power(AI, config.alpha_AI)
        * np.power(H, config.alpha_H)
    )


def unpack_decision(z: np.ndarray, horizon: int):
    z = np.asarray(z, dtype=float)
    return (
        z[0:horizon],
        z[horizon:2*horizon],
        z[2*horizon:3*horizon],
        z[3*horizon:4*horizon],
        z[4*horizon:5*horizon],
    )


def default_decision(config: DynamicConfig) -> np.ndarray:
    T = config.horizon
    shares = [
        np.full(T, 0.60),
        np.full(T, 0.18),
        np.full(T, 0.07),
        np.full(T, 0.06),
        np.full(T, 0.06),
    ]
    return np.concatenate(shares)


def simulate_path(
    z: np.ndarray,
    config: DynamicConfig,
    shock_year: int | None = None,
    shock_pct: float = 0.0,
) -> dict[str, Any]:
    T = config.horizon
    year_list = years(config)
    C_share, IK_share, ID_share, IAI_share, IH_share = unpack_decision(z, T)

    K = np.zeros(T + 1)
    D = np.zeros(T + 1)
    AI = np.zeros(T + 1)
    H = np.zeros(T + 1)
    A = np.zeros(T + 1)
    L = np.zeros(T)

    K[0], D[0], AI[0], H[0], A[0] = (
        config.K0, config.D0, config.AI0, config.H0, config.A0
    )

    rows = []
    welfare = 0.0
    penalty = 0.0

    for t in range(T):
        L[t] = config.labor_0 * ((1 + config.labor_growth) ** t)
        y_plan = production(A[t], K[t], L[t], D[t], AI[t], H[t], config)
        y_actual = y_plan

        if shock_year is not None and year_list[t] == shock_year:
            y_actual *= 1 - shock_pct

        C = C_share[t] * y_actual
        IK = IK_share[t] * y_actual
        ID = ID_share[t] * y_actual
        IAI = IAI_share[t] * y_actual
        IH = IH_share[t] * y_actual
        total_investment = IK + ID + IAI + IH

        resource_gap = C + total_investment - y_actual
        if resource_gap > 0:
            penalty += 1e5 * resource_gap ** 2

        if C <= 0:
            penalty += 1e9

        K[t + 1] = (1 - config.delta_K) * K[t] + config.scale_K * IK
        D[t + 1] = (1 - config.delta_D) * D[t] + config.scale_D * ID
        AI[t + 1] = (1 - config.delta_AI) * AI[t] + config.scale_AI * IAI
        H[t + 1] = (
            H[t]
            + config.theta_H * config.scale_H * IH
            - config.brain_drain * H[t]
        )
        A[t + 1] = A[t] * (
            1
            + config.phi_D * D[t] / 100
            + config.phi_AI * AI[t] / 100
            + config.phi_H * H[t] / 100
        )

        welfare_term = (config.rho ** t) * float(
            utility(C, config.gamma_utility)
        )
        welfare += welfare_term

        rows.append({
            "year": year_list[t],
            "K": K[t],
            "D": D[t],
            "AI": AI[t],
            "H": H[t],
            "A_TFP": A[t],
            "L": L[t],
            "Y_plan": y_plan,
            "Y": y_actual,
            "C": C,
            "I_K": IK,
            "I_D": ID,
            "I_AI": IAI,
            "I_H": IH,
            "Total_investment": total_investment,
            "C_share": C_share[t],
            "IK_share": IK_share[t],
            "ID_share": ID_share[t],
            "IAI_share": IAI_share[t],
            "IH_share": IH_share[t],
            "Welfare_term": welfare_term,
        })

    path = pd.DataFrame(rows)

    terminal = pd.DataFrame([{
        "year": year_list[-1] + 1,
        "K": K[-1],
        "D": D[-1],
        "AI": AI[-1],
        "H": H[-1],
        "A_TFP": A[-1],
    }])

    return {
        "path": path,
        "terminal": terminal,
        "welfare": welfare,
        "penalty": penalty,
        "objective_penalized": welfare - penalty,
    }


def build_constraints(config: DynamicConfig):
    T = config.horizon

    def resources(z):
        C, IK, ID, IAI, IH = unpack_decision(z, T)
        return 1.0 - (C + IK + ID + IAI + IH)

    def investment_cap(z):
        _, IK, ID, IAI, IH = unpack_decision(z, T)
        return config.max_total_investment_share - (IK + ID + IAI + IH)

    def consumption_floor(z):
        C, *_ = unpack_decision(z, T)
        return C - config.min_consumption_share

    def h_floor(z):
        *_, IH = unpack_decision(z, T)
        return IH - config.min_H_investment_share

    def digital_ai_floor(z):
        _, _, ID, IAI, _ = unpack_decision(z, T)
        return ID + IAI - config.min_DAI_investment_share

    return [
        {"type": "ineq", "fun": resources},
        {"type": "ineq", "fun": investment_cap},
        {"type": "ineq", "fun": consumption_floor},
        {"type": "ineq", "fun": h_floor},
        {"type": "ineq", "fun": digital_ai_floor},
    ]


def objective(z: np.ndarray, config: DynamicConfig) -> float:
    result = simulate_path(z, config)
    return -result["objective_penalized"]


def solve_dynamic(
    config: DynamicConfig = DynamicConfig(),
    maxiter: int = 350,
) -> dict[str, Any]:
    T = config.horizon
    x0 = default_decision(config)

    lower = np.concatenate([
        np.full(T, config.min_consumption_share),
        np.zeros(T),
        np.zeros(T),
        np.zeros(T),
        np.full(T, config.min_H_investment_share),
    ])

    upper = np.concatenate([
        np.full(T, 0.85),
        np.full(T, config.max_single_investment_share),
        np.full(T, config.max_single_investment_share),
        np.full(T, config.max_single_investment_share),
        np.full(T, config.max_single_investment_share),
    ])

    result = minimize(
        objective,
        x0=x0,
        args=(config,),
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=build_constraints(config),
        options={
            "maxiter": int(maxiter),
            "ftol": 1e-9,
            "disp": False,
        },
    )

    simulation = simulate_path(result.x, config)

    path = simulation["path"].copy()
    midpoint = max(1, T // 2)
    early_investment = path.iloc[:midpoint]["Total_investment"].sum()
    late_investment = path.iloc[midpoint:]["Total_investment"].sum()
    loading_pattern = (
        "Front-loaded"
        if early_investment > late_investment
        else "Back-loaded"
    )

    ratio = path["I_AI"] / np.maximum(path["I_H"], 1e-9)
    ratio_cv = float(ratio.std() / max(abs(ratio.mean()), 1e-9))

    return {
        "success": bool(result.success),
        "status": str(result.message),
        "raw": result,
        "decision": result.x,
        "path": path,
        "terminal": simulation["terminal"],
        "welfare": float(simulation["welfare"]),
        "loading_pattern": loading_pattern,
        "early_investment": float(early_investment),
        "late_investment": float(late_investment),
        "ai_h_ratio_mean": float(ratio.mean()),
        "ai_h_ratio_cv": ratio_cv,
    }


def compare_discount_rates(
    base_config: DynamicConfig = DynamicConfig(),
    short_term_rho: float = 0.90,
    maxiter: int = 350,
) -> dict[str, Any]:
    long_term = solve_dynamic(base_config, maxiter=maxiter)
    short_config = replace(base_config, rho=float(short_term_rho))
    short_term = solve_dynamic(short_config, maxiter=maxiter)

    summary = pd.DataFrame([
        {
            "Kịch bản": f"rho={base_config.rho:.2f}",
            "Phúc lợi": long_term["welfare"],
            "GDP 2035": long_term["path"]["Y"].iloc[-1],
            "Đầu tư AI tổng": long_term["path"]["I_AI"].sum(),
            "Đầu tư H tổng": long_term["path"]["I_H"].sum(),
            "Tiêu dùng tổng": long_term["path"]["C"].sum(),
            "Mẫu phân bổ": long_term["loading_pattern"],
        },
        {
            "Kịch bản": f"rho={short_term_rho:.2f}",
            "Phúc lợi": short_term["welfare"],
            "GDP 2035": short_term["path"]["Y"].iloc[-1],
            "Đầu tư AI tổng": short_term["path"]["I_AI"].sum(),
            "Đầu tư H tổng": short_term["path"]["I_H"].sum(),
            "Tiêu dùng tổng": short_term["path"]["C"].sum(),
            "Mẫu phân bổ": short_term["loading_pattern"],
        },
    ])

    return {
        "long_term": long_term,
        "short_term": short_term,
        "summary": summary,
    }


def bellman_style_benchmark(
    config: DynamicConfig = DynamicConfig(),
    saving_grid: np.ndarray | None = None,
    ai_h_grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Benchmark Bellman-style đơn giản:
    duyệt các chính sách tỷ lệ cố định và chọn phúc lợi lớn nhất.
    Đây là đối chứng lưới rời rạc, không thay thế SLSQP 50 biến.
    """

    if saving_grid is None:
        saving_grid = np.linspace(0.20, 0.40, 6)
    if ai_h_grid is None:
        ai_h_grid = np.linspace(0.35, 0.65, 7)

    candidates = []
    T = config.horizon

    for saving in saving_grid:
        for ai_share in ai_h_grid:
            c_share = 1.0 - saving
            ih = max(config.min_H_investment_share, saving * (1 - ai_share) * 0.45)
            iai = saving * ai_share * 0.35
            id_share = max(
                config.min_DAI_investment_share - iai,
                saving * 0.15,
            )
            ik = saving - ih - iai - id_share

            if ik < 0:
                continue

            z = np.concatenate([
                np.full(T, c_share),
                np.full(T, ik),
                np.full(T, id_share),
                np.full(T, iai),
                np.full(T, ih),
            ])

            sim = simulate_path(z, config)
            if sim["penalty"] <= 1e-8:
                candidates.append({
                    "saving_share": saving,
                    "ai_orientation": ai_share,
                    "welfare": sim["welfare"],
                    "GDP_2035": sim["path"]["Y"].iloc[-1],
                    "C_total": sim["path"]["C"].sum(),
                    "decision": z,
                    "path": sim["path"],
                })

    if not candidates:
        raise RuntimeError("Không có chính sách lưới khả thi.")

    table = pd.DataFrame([
        {k: v for k, v in row.items() if k not in {"decision", "path"}}
        for row in candidates
    ]).sort_values("welfare", ascending=False).reset_index(drop=True)

    best_idx = int(table.index[0])
    best_key = (
        table.iloc[0]["saving_share"],
        table.iloc[0]["ai_orientation"],
    )
    best = next(
        row for row in candidates
        if np.isclose(row["saving_share"], best_key[0])
        and np.isclose(row["ai_orientation"], best_key[1])
    )

    return {
        "table": table,
        "best": best,
    }


def run_full_bai08(
    config: DynamicConfig = DynamicConfig(),
    short_term_rho: float = 0.90,
    maxiter: int = 350,
) -> dict[str, Any]:
    comparison = compare_discount_rates(
        base_config=config,
        short_term_rho=short_term_rho,
        maxiter=maxiter,
    )
    bellman = bellman_style_benchmark(config)

    return {
        "config": config,
        "optimal": comparison["long_term"],
        "short_term": comparison["short_term"],
        "discount_comparison": comparison["summary"],
        "bellman": bellman,
    }
