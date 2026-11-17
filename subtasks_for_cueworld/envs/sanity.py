from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SanityEnv(gym.Env):
    """
    能動的推論が動くかを見る最小のバンドイット環境。

    - 行動の1つだけ次の記号が決まり、ほかは全記号の一様乱数。
    - reward_mode がオンのときだけ、乱数行動の1つに報酬 1 が付く。
    - 学習するエージェントは置かない。呼び出し側が reset / step する。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        max_steps: int,
        n_actions: int = 3,
        n_symbols: int = 4,
        reward_mode: bool = False,
        symbol_noise: float = 0.0,
        obs_noise: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if n_actions < 3:
            raise ValueError(f"n_actions must be >= 3, got {n_actions}")
        if n_symbols < 2:
            raise ValueError(f"n_symbols must be >= 2, got {n_symbols}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if not 0.0 <= symbol_noise <= 1.0:
            raise ValueError(f"symbol_noise must be in [0, 1], got {symbol_noise}")
        if obs_noise < 0.0:
            raise ValueError(f"obs_noise must be >= 0, got {obs_noise}")

        self.n_actions = int(n_actions)
        self.n_symbols = int(n_symbols)
        self.max_steps = int(max_steps)
        self.reward_mode = bool(reward_mode)
        self.symbol_noise = float(symbol_noise)
        self.obs_noise = float(obs_noise)

        self._assign_rng = np.random.default_rng(seed)
        roles = self._assign_rng.choice(self.n_actions, size=2, replace=False)
        self.deterministic_action = int(roles[0])
        self.reward_action = int(roles[1])
        self.deterministic_symbol = int(self._assign_rng.integers(0, self.n_symbols))

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_symbols,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._episode_started = False
        self._yoke_observations: Optional[np.ndarray] = None
        self._yoke_index = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._episode_started = True
        self._yoke_index = 0
        self._yoke_observations = None

        if options is not None and "yoke_observations" in options:
            tape = np.asarray(options["yoke_observations"], dtype=np.float32)
            expected = (self.max_steps, self.n_symbols)
            if tape.shape != expected:
                raise ValueError(
                    f"yoke_observations shape must be {expected}, got {tape.shape}"
                )
            self._yoke_observations = tape.copy()

        obs = np.zeros(self.n_symbols, dtype=np.float32)
        return obs, self._base_info(symbol=None)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._episode_started:
            raise RuntimeError("Call reset before step")
        if self._step_count >= self.max_steps:
            raise RuntimeError("Episode already truncated; call reset")

        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action out of range: {action}")

        if self._yoke_observations is not None:
            obs = self._yoke_observations[self._yoke_index].copy()
            symbol: Optional[int] = None
            self._yoke_index += 1
        else:
            symbol = self._sample_symbol(action)
            obs = self._symbol_to_obs(symbol)

        reward = self._reward_for(action)
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps
        if truncated:
            self._episode_started = False

        info = self._base_info(symbol=symbol)
        return obs, float(reward), terminated, truncated, info

    def _sample_symbol(self, action: int) -> int:
        if action == self.deterministic_action:
            symbol = self.deterministic_symbol
        else:
            symbol = int(self._rng.integers(0, self.n_symbols))

        if self.symbol_noise > 0.0 and self._rng.random() < self.symbol_noise:
            symbol = int(self._rng.integers(0, self.n_symbols))
        return symbol

    def _symbol_to_obs(self, symbol: int) -> np.ndarray:
        obs = np.zeros(self.n_symbols, dtype=np.float32)
        obs[symbol] = 1.0
        if self.obs_noise > 0.0:
            obs = obs + self._rng.normal(
                0.0, self.obs_noise, size=self.n_symbols
            ).astype(np.float32)
        return obs.astype(np.float32, copy=False)

    def _reward_for(self, action: int) -> float:
        if self.reward_mode and action == self.reward_action:
            return 1.0
        return 0.0

    def _base_info(self, *, symbol: Optional[int]) -> dict[str, Any]:
        return {
            "deterministic_action": self.deterministic_action,
            "reward_action": self.reward_action,
            "deterministic_symbol": self.deterministic_symbol,
            "reward_mode": self.reward_mode,
            "yoke": self._yoke_observations is not None,
            "symbol": symbol,
        }
