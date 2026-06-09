from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except Exception:
    gym = None
    spaces = None
    GYM_AVAILABLE = False


LEVEL_NAMES = {
    0: "Thấp",
    1: "Trung bình",
    2: "Cao",
}

STATE_FACTOR_NAMES = [
    "GDP growth",
    "Digital index",
    "AI capacity",
    "Unemployment risk",
]

ACTION_NAMES = {
    0: "a0 — Truyền thống",
    1: "a1 — Cân bằng",
    2: "a2 — Số hóa nhanh",
    3: "a3 — AI dẫn dắt",
    4: "a4 — Bao trùm",
}

# Thứ tự: K, D, AI, H
ACTION_ALLOCATIONS = {
    0: np.array([0.70, 0.10, 0.10, 0.10], dtype=float),
    1: np.array([0.40, 0.25, 0.15, 0.20], dtype=float),
    2: np.array([0.25, 0.45, 0.15, 0.15], dtype=float),
    3: np.array([0.20, 0.20, 0.45, 0.15], dtype=float),
    4: np.array([0.30, 0.20, 0.10, 0.40], dtype=float),
}

ACTUAL_VIETNAM_2026_STATE = np.array(
    [1, 1, 0, 1],
    dtype=np.int64,
)

TEST_STATES = {
    "Việt Nam 2026": np.array([1, 1, 0, 1], dtype=np.int64),
    "Tăng trưởng thấp, số hóa thấp, AI thấp, thất nghiệp cao":
        np.array([0, 0, 0, 2], dtype=np.int64),
    "Tăng trưởng cao, số hóa cao, AI cao, thất nghiệp thấp":
        np.array([2, 2, 2, 0], dtype=np.int64),
    "Tăng trưởng trung bình, số hóa cao, AI trung bình, thất nghiệp cao":
        np.array([1, 2, 1, 2], dtype=np.int64),
    "Tăng trưởng thấp, số hóa trung bình, AI cao, thất nghiệp trung bình":
        np.array([0, 1, 2, 1], dtype=np.int64),
}


@dataclass(frozen=True)
class RLConfig:
    n_episodes: int = 10_000
    learning_rate: float = 0.10
    discount_factor: float = 0.95
    epsilon_start: float = 1.00
    epsilon_end: float = 0.05
    episode_length: int = 10
    annual_budget: float = 1000.0
    shock_probability: float = 0.12
    reward_growth_weight: float = 0.40
    reward_unemployment_weight: float = 0.25
    reward_cyber_weight: float = 0.20
    reward_emission_weight: float = 0.15
    seed: int = 42
    evaluation_episodes: int = 300

    @property
    def reward_weights(self) -> np.ndarray:
        values = np.array(
            [
                self.reward_growth_weight,
                self.reward_unemployment_weight,
                self.reward_cyber_weight,
                self.reward_emission_weight,
            ],
            dtype=float,
        )
        total = values.sum()
        if total <= 0:
            raise ValueError("Tổng trọng số reward phải lớn hơn 0.")
        return values / total


def state_to_index(state: np.ndarray | list[int] | tuple[int, ...]) -> int:
    g, d, ai, u = [int(value) for value in state]
    return int(np.ravel_multi_index((g, d, ai, u), (3, 3, 3, 3)))


def index_to_state(index: int) -> np.ndarray:
    return np.array(
        np.unravel_index(int(index), (3, 3, 3, 3)),
        dtype=np.int64,
    )


def state_to_label(state: np.ndarray | list[int]) -> str:
    values = [int(value) for value in state]
    return (
        f"GDP={LEVEL_NAMES[values[0]]}; "
        f"D={LEVEL_NAMES[values[1]]}; "
        f"AI={LEVEL_NAMES[values[2]]}; "
        f"U={LEVEL_NAMES[values[3]]}"
    )


def action_table() -> pd.DataFrame:
    rows = []
    for action_id, allocation in ACTION_ALLOCATIONS.items():
        rows.append({
            "action_id": action_id,
            "action": ACTION_NAMES[action_id],
            "K_share": allocation[0],
            "D_share": allocation[1],
            "AI_share": allocation[2],
            "H_share": allocation[3],
        })
    return pd.DataFrame(rows)


if GYM_AVAILABLE:
    class VietnamEconomyEnv(gym.Env):
        """
        Môi trường MDP 81 trạng thái và 5 hành động.

        State = [GDP growth, Digital index, AI capacity, Unemployment risk],
        mỗi thành phần có 3 mức: 0, 1, 2.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            config: RLConfig = RLConfig(),
            start_state: np.ndarray | None = None,
            stochastic: bool = True,
        ) -> None:
            super().__init__()

            self.config = config
            self.action_space = spaces.Discrete(5)
            self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
            self.start_state = (
                None
                if start_state is None
                else np.asarray(start_state, dtype=np.int64)
            )
            self.stochastic = bool(stochastic)
            self.rng = np.random.default_rng(config.seed)

            self.alpha_K = 0.33
            self.alpha_L = 0.42
            self.alpha_D = 0.10
            self.alpha_AI = 0.08
            self.alpha_H = 0.07

            self.reset(seed=config.seed)

        def _production(self) -> float:
            return float(
                self.A
                * self.K ** self.alpha_K
                * self.L ** self.alpha_L
                * max(self.D, 1e-6) ** self.alpha_D
                * max(self.AI, 1e-6) ** self.alpha_AI
                * max(self.H, 1e-6) ** self.alpha_H
            )

        @staticmethod
        def _discretize_growth(value: float) -> int:
            if value < 4.8:
                return 0
            if value < 6.8:
                return 1
            return 2

        @staticmethod
        def _discretize_digital(value: float) -> int:
            if value < 24.0:
                return 0
            if value < 34.0:
                return 1
            return 2

        @staticmethod
        def _discretize_ai(value: float) -> int:
            if value < 95.0:
                return 0
            if value < 125.0:
                return 1
            return 2

        @staticmethod
        def _discretize_unemployment(value: float) -> int:
            if value < 3.5:
                return 0
            if value < 5.0:
                return 1
            return 2

        def _continuous_from_state(self, state: np.ndarray) -> None:
            g, d, ai, u = [int(value) for value in state]
            self.K = [25000.0, 27500.0, 31000.0][g]
            self.D = [18.0, 27.0, 39.0][d]
            self.AI = [80.0, 110.0, 145.0][ai]
            self.H = [25.0, 31.0, 38.0][d]
            self.U = [3.0, 4.3, 5.8][u]
            self.A = 1.0
            self.L = 54.0
            self.current_growth = [3.8, 5.8, 7.6][g]

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None,
        ) -> tuple[np.ndarray, dict]:
            super().reset(seed=seed)

            if seed is not None:
                self.rng = np.random.default_rng(seed)

            self.t = 0

            options = options or {}
            initial_state = options.get("initial_state")

            if initial_state is not None:
                state = np.asarray(initial_state, dtype=np.int64)
            elif self.start_state is not None:
                state = self.start_state.copy()
            else:
                state = ACTUAL_VIETNAM_2026_STATE.copy()

            self.state = state
            self._continuous_from_state(state)
            self.prev_output = self._production()

            return self.state.copy(), {
                "state_label": state_to_label(self.state),
            }

        def step(
            self,
            action: int,
        ) -> tuple[np.ndarray, float, bool, bool, dict]:
            action = int(action)
            allocation = ACTION_ALLOCATIONS[action]
            K_share, D_share, AI_share, H_share = allocation
            budget = self.config.annual_budget

            shock = 0.0
            if self.stochastic and self.rng.random() < self.config.shock_probability:
                shock = float(self.rng.uniform(0.02, 0.09))

            self.K = 0.95 * self.K + budget * K_share
            self.D = 0.90 * self.D + 0.012 * budget * D_share
            self.AI = 0.87 * self.AI + 0.020 * budget * AI_share
            self.H = 0.98 * self.H + 0.010 * budget * H_share
            self.L *= 1.004

            self.A *= (
                1.0
                + 0.0015 * D_share
                + 0.0022 * AI_share
                + 0.0018 * H_share
            )

            output = self._production() * (1.0 - shock)
            growth_pct = (
                (output / max(self.prev_output, 1e-9) - 1.0)
                * 100.0
            )

            # Nhân lực giúp giảm U; AI mạnh nhưng thiếu H làm U tăng.
            mismatch = max(0.0, AI_share - 1.15 * H_share)
            self.U = float(
                np.clip(
                    self.U
                    + 0.90 * mismatch
                    - 0.55 * H_share
                    - 0.20 * D_share
                    + 2.5 * shock,
                    2.0,
                    8.0,
                )
            )

            cyber_risk = float(
                np.clip(
                    0.12
                    + 0.95 * AI_share
                    + 0.40 * D_share
                    - 0.65 * H_share,
                    0.0,
                    1.0,
                )
            )

            emission = float(
                np.clip(
                    0.10
                    + 0.70 * K_share
                    + 0.45 * AI_share
                    - 0.25 * D_share
                    - 0.20 * H_share,
                    0.0,
                    1.0,
                )
            )

            unemployment_change = self.U - [3.0, 4.3, 5.8][int(self.state[3])]
            growth_score = float(np.clip(growth_pct / 10.0, -1.0, 1.5))
            unemployment_penalty = float(np.clip(unemployment_change / 3.0, -1.0, 1.0))

            w = self.config.reward_weights
            reward = float(
                w[0] * growth_score
                - w[1] * unemployment_penalty
                - w[2] * cyber_risk
                - w[3] * emission
            )

            next_state = np.array(
                [
                    self._discretize_growth(growth_pct),
                    self._discretize_digital(self.D),
                    self._discretize_ai(self.AI),
                    self._discretize_unemployment(self.U),
                ],
                dtype=np.int64,
            )

            self.state = next_state
            self.prev_output = output
            self.current_growth = growth_pct
            self.t += 1

            terminated = self.t >= self.config.episode_length
            truncated = False

            info = {
                "t": self.t,
                "action": action,
                "action_label": ACTION_NAMES[action],
                "allocation_K": K_share,
                "allocation_D": D_share,
                "allocation_AI": AI_share,
                "allocation_H": H_share,
                "gdp_growth_pct": growth_pct,
                "D": self.D,
                "AI": self.AI,
                "H": self.H,
                "U": self.U,
                "cyber_risk": cyber_risk,
                "emission": emission,
                "shock": shock,
                "output": output,
            }

            return (
                next_state.copy(),
                reward,
                terminated,
                truncated,
                info,
            )

else:
    class VietnamEconomyEnv:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Chưa cài gymnasium. Chạy: python -m pip install gymnasium"
            )


def epsilon_schedule(config: RLConfig) -> np.ndarray:
    return np.linspace(
        config.epsilon_start,
        config.epsilon_end,
        config.n_episodes,
        dtype=float,
    )


def train_q_learning(
    config: RLConfig = RLConfig(),
) -> dict[str, Any]:
    if not GYM_AVAILABLE:
        return {
            "success": False,
            "status": "Chưa cài gymnasium.",
        }

    rng = np.random.default_rng(config.seed)
    env = VietnamEconomyEnv(config=config, stochastic=True)

    q_table = np.zeros((81, 5), dtype=float)
    rewards = np.zeros(config.n_episodes, dtype=float)
    epsilons = epsilon_schedule(config)
    state_visits = np.zeros(81, dtype=int)
    action_counts = np.zeros(5, dtype=int)

    for episode in range(config.n_episodes):
        state, _ = env.reset(seed=config.seed + episode)
        state_idx = state_to_index(state)
        total_reward = 0.0

        for _ in range(config.episode_length):
            state_visits[state_idx] += 1

            if rng.random() < epsilons[episode]:
                action = int(rng.integers(0, 5))
            else:
                action = int(np.argmax(q_table[state_idx]))

            action_counts[action] += 1

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_idx = state_to_index(next_state)

            td_target = reward + config.discount_factor * np.max(q_table[next_idx])
            td_error = td_target - q_table[state_idx, action]
            q_table[state_idx, action] += config.learning_rate * td_error

            total_reward += reward
            state_idx = next_idx

            if terminated or truncated:
                break

        rewards[episode] = total_reward

    policy = np.argmax(q_table, axis=1).astype(int)

    learning_curve = pd.DataFrame({
        "episode": np.arange(1, config.n_episodes + 1),
        "reward": rewards,
        "epsilon": epsilons,
    })

    rolling_window = max(50, min(500, config.n_episodes // 20))
    learning_curve["reward_rolling"] = (
        learning_curve["reward"]
        .rolling(rolling_window, min_periods=1)
        .mean()
    )

    visit_table = pd.DataFrame({
        "state_index": np.arange(81),
        "state_label": [
            state_to_label(index_to_state(index))
            for index in range(81)
        ],
        "visits": state_visits,
        "best_action_id": policy,
        "best_action": [
            ACTION_NAMES[action]
            for action in policy
        ],
        "max_Q": q_table.max(axis=1),
    })

    action_table_df = action_table()
    action_table_df["training_count"] = [
        action_counts[action]
        for action in range(5)
    ]

    return {
        "success": True,
        "status": "Completed",
        "q_table": q_table,
        "policy": policy,
        "rewards": rewards,
        "epsilons": epsilons,
        "learning_curve": learning_curve,
        "visit_table": visit_table,
        "action_table": action_table_df,
        "reward_first_500": float(rewards[: min(500, len(rewards))].mean()),
        "reward_last_500": float(rewards[-min(500, len(rewards)):].mean()),
        "improvement": float(
            rewards[-min(500, len(rewards)):].mean()
            - rewards[: min(500, len(rewards))].mean()
        ),
    }


def policy_action(
    policy: np.ndarray,
    state: np.ndarray,
) -> int:
    return int(policy[state_to_index(state)])


def extract_test_state_policy(
    policy: np.ndarray,
    q_table: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for name, state in TEST_STATES.items():
        idx = state_to_index(state)
        action = int(policy[idx])

        rows.append({
            "Trạng thái": name,
            "Vector trạng thái": str(tuple(int(value) for value in state)),
            "Mô tả": state_to_label(state),
            "Hành động agent chọn": ACTION_NAMES[action],
            "Q tốt nhất": float(q_table[idx, action]),
        })

    return pd.DataFrame(rows)


def simulate_policy(
    policy_fn: Callable[[np.ndarray, np.random.Generator], int],
    config: RLConfig,
    n_episodes: int | None = None,
    policy_label: str = "Policy",
    initial_state: np.ndarray | None = None,
    seed_offset: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not GYM_AVAILABLE:
        raise ImportError("Chưa cài gymnasium.")

    episode_count = (
        config.evaluation_episodes
        if n_episodes is None
        else int(n_episodes)
    )

    episode_rows = []
    trace_rows = []

    for episode in range(episode_count):
        episode_seed = config.seed + seed_offset + episode
        rng = np.random.default_rng(episode_seed)

        env = VietnamEconomyEnv(
            config=config,
            start_state=initial_state,
            stochastic=True,
        )

        state, _ = env.reset(seed=episode_seed)
        total_reward = 0.0

        for step in range(config.episode_length):
            state_before = state.copy()
            action = int(policy_fn(state_before, rng))

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if episode == 0:
                trace_rows.append({
                    "policy_label": policy_label,
                    "episode": episode,
                    "t": step + 1,
                    "state_before": state_to_label(state_before),
                    "action_id": action,
                    "action_label": ACTION_NAMES[action],
                    "reward": reward,
                    "state_after": state_to_label(state),
                    **info,
                })

            if terminated or truncated:
                break

        episode_rows.append({
            "policy_label": policy_label,
            "episode": episode,
            "total_reward": total_reward,
        })

    return pd.DataFrame(episode_rows), pd.DataFrame(trace_rows)


def compare_policies(
    policy: np.ndarray,
    config: RLConfig,
) -> dict[str, Any]:
    policies = {
        "π* Q-learning":
            lambda state, rng: policy_action(policy, state),
        "Rule-based: luôn a1":
            lambda state, rng: 1,
        "Rule-based: luôn a3":
            lambda state, rng: 3,
        "Random":
            lambda state, rng: int(rng.integers(0, 5)),
    }

    episode_frames = []
    trace_frames = []

    for position, (label, policy_fn) in enumerate(policies.items()):
        episodes, trace = simulate_policy(
            policy_fn=policy_fn,
            config=config,
            policy_label=label,
            seed_offset=30_000 + position * 10_000,
        )
        episode_frames.append(episodes)
        trace_frames.append(trace)

    compare_df = pd.concat(episode_frames, ignore_index=True)
    trace_df = pd.concat(trace_frames, ignore_index=True)

    summary = (
        compare_df.groupby("policy_label", as_index=False)
        .agg(
            mean_reward=("total_reward", "mean"),
            std_reward=("total_reward", "std"),
            min_reward=("total_reward", "min"),
            max_reward=("total_reward", "max"),
        )
        .sort_values("mean_reward", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "compare_df": compare_df,
        "summary": summary,
        "trace_df": trace_df,
    }


def policy_map_table(
    policy: np.ndarray,
    q_table: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for state_index in range(81):
        state = index_to_state(state_index)
        action = int(policy[state_index])

        rows.append({
            "state_index": state_index,
            "GDP": LEVEL_NAMES[int(state[0])],
            "Digital": LEVEL_NAMES[int(state[1])],
            "AI": LEVEL_NAMES[int(state[2])],
            "Unemployment": LEVEL_NAMES[int(state[3])],
            "state_label": state_to_label(state),
            "best_action_id": action,
            "best_action": ACTION_NAMES[action],
            "max_Q": float(q_table[state_index, action]),
        })

    return pd.DataFrame(rows)


def environment_check(
    config: RLConfig = RLConfig(),
) -> pd.DataFrame:
    if not GYM_AVAILABLE:
        return pd.DataFrame([
            {
                "Kiểm tra": "gymnasium",
                "Kết quả": "Chưa cài",
                "Chi tiết": "python -m pip install gymnasium",
            }
        ])

    env = VietnamEconomyEnv(config=config, stochastic=False)
    state, info = env.reset(seed=config.seed)
    next_state, reward, terminated, truncated, step_info = env.step(1)

    return pd.DataFrame([
        {
            "Kiểm tra": "action_space",
            "Kết quả": str(env.action_space),
            "Chi tiết": "5 hành động",
        },
        {
            "Kiểm tra": "observation_space",
            "Kết quả": str(env.observation_space),
            "Chi tiết": "3⁴ = 81 trạng thái",
        },
        {
            "Kiểm tra": "reset",
            "Kết quả": str(tuple(int(value) for value in state)),
            "Chi tiết": info["state_label"],
        },
        {
            "Kiểm tra": "step",
            "Kết quả": str(tuple(int(value) for value in next_state)),
            "Chi tiết": (
                f"reward={reward:.4f}; "
                f"terminated={terminated}; truncated={truncated}"
            ),
        },
    ])


def train_dqn_optional(
    config: RLConfig,
    total_timesteps: int = 2_000,
    learning_rate: float = 1e-3,
    gamma: float = 0.95,
) -> dict[str, Any]:
    if not GYM_AVAILABLE:
        return {
            "success": False,
            "status": "Chưa cài gymnasium.",
        }

    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.monitor import Monitor
    except Exception as error:
        return {
            "success": False,
            "status": (
                "Chưa cài stable-baselines3 hoặc torch không tương thích."
            ),
            "error": str(error),
        }

    env = Monitor(
        VietnamEconomyEnv(
            config=config,
            stochastic=True,
        )
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=float(learning_rate),
        buffer_size=max(5_000, int(total_timesteps) * 2),
        learning_starts=min(200, max(50, int(total_timesteps) // 10)),
        batch_size=32,
        gamma=float(gamma),
        exploration_fraction=0.40,
        exploration_final_eps=0.05,
        policy_kwargs={
            "net_arch": [64, 64],
        },
        verbose=0,
        seed=config.seed,
    )

    model.learn(
        total_timesteps=int(total_timesteps),
        progress_bar=False,
    )

    def dqn_policy(state: np.ndarray, rng: np.random.Generator) -> int:
        action, _ = model.predict(state, deterministic=True)
        return int(action)

    episodes, trace = simulate_policy(
        policy_fn=dqn_policy,
        config=config,
        n_episodes=min(200, config.evaluation_episodes),
        policy_label="DQN",
        seed_offset=90_000,
    )

    return {
        "success": True,
        "status": "Completed",
        "model": model,
        "episodes": episodes,
        "trace": trace,
        "mean_reward": float(episodes["total_reward"].mean()),
        "std_reward": float(episodes["total_reward"].std()),
        "total_timesteps": int(total_timesteps),
    }


def run_full_bai11(
    config: RLConfig = RLConfig(),
) -> dict[str, Any]:
    training = train_q_learning(config)

    if not training.get("success", False):
        return {
            "success": False,
            "status": training.get("status", "Không chạy được."),
            "environment_check": environment_check(config),
        }

    test_states = extract_test_state_policy(
        training["policy"],
        training["q_table"],
    )

    comparison = compare_policies(
        training["policy"],
        config,
    )

    policy_map = policy_map_table(
        training["policy"],
        training["q_table"],
    )

    return {
        "success": True,
        "status": "Completed",
        "config": config,
        "environment_check": environment_check(config),
        "training": training,
        "test_states": test_states,
        "comparison": comparison,
        "policy_map": policy_map,
    }
