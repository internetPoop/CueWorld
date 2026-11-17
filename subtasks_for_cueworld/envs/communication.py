from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CommunicationEnv(gym.Env):
    """
    幼児と親が毎 step 信号を出し合うコミュニケーション環境。

    - 行動は何もしない 1 種 + 信号 4 種。
    - 意味のある塊は、一人が続けて出した信号 2 つ。相手がその 2 step とも
      沈黙して受け取ったときだけ届く。
    - 有効なトリガが届いた次の 2 step で、親は決まった返事を返す。
    - reward は常に 0。学習するエージェントは置かない。
    """

    metadata = {"render_modes": []}

    TRIGGER_TABLE = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    ]
    REPLY_TABLE = [
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
    ]

    N_SYMBOLS = 4
    MAX_ENERGY = 4
    N_ACTIONS = 5  # 0=noop, 1-4=symbols 0-3
    OBS_DIM = 6  # energy/4, who, onehot x4

    def __init__(
        self,
        *,
        max_steps: int,
        n_meaningful: int = 1,
        obs_noise: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if not 1 <= n_meaningful <= len(self.TRIGGER_TABLE):
            raise ValueError(
                f"n_meaningful must be in [1, {len(self.TRIGGER_TABLE)}], "
                f"got {n_meaningful}"
            )
        if obs_noise < 0.0:
            raise ValueError(f"obs_noise must be >= 0, got {obs_noise}")

        self.max_steps = int(max_steps)
        self.n_meaningful = int(n_meaningful)
        self.obs_noise = float(obs_noise)

        self.meaningful_pairs: dict[tuple[int, int], int] = {
            self.TRIGGER_TABLE[i]: i for i in range(self.n_meaningful)
        }
        self.reply_pairs: dict[int, tuple[int, int]] = {
            i: self.REPLY_TABLE[i] for i in range(self.n_meaningful)
        }

        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._episode_started = False
        self._infant_energy = self.MAX_ENERGY
        self._parent_energy = self.MAX_ENERGY
        self._recv_buffer: list[int] = []
        self._reply_queue: list[int] = []
        self._trigger_count = 0
        self._reply_heard_count = 0
        self._reply_step = 0
        self._reply_infant_silent = False
        self._yoke_parent_actions: Optional[np.ndarray] = None
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
        self._infant_energy = self.MAX_ENERGY
        self._parent_energy = self.MAX_ENERGY
        self._recv_buffer = []
        self._reply_queue = []
        self._trigger_count = 0
        self._reply_heard_count = 0
        self._reply_step = 0
        self._reply_infant_silent = False
        self._yoke_index = 0
        self._yoke_parent_actions = None

        if options is not None and "yoke_parent_actions" in options:
            tape = np.asarray(options["yoke_parent_actions"])
            expected = (self.max_steps,)
            if tape.shape != expected:
                raise ValueError(
                    f"yoke_parent_actions shape must be {expected}, got {tape.shape}"
                )
            if tape.dtype.kind not in ("i", "u"):
                tape = tape.astype(np.int64)
            if np.any(tape < 0) or np.any(tape >= self.N_ACTIONS):
                raise ValueError(
                    f"yoke_parent_actions values must be in [0, {self.N_ACTIONS - 1}]"
                )
            self._yoke_parent_actions = tape.astype(np.int64, copy=True)

        obs = np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        return obs, self._base_info(
            infant_action=0,
            parent_action=0,
            trigger_id=None,
            reply_step=0,
        )

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

        is_yoke = self._yoke_parent_actions is not None

        # 1. Resolve infant action under energy constraint
        infant_action = action
        if infant_action != 0 and self._infant_energy == 0:
            infant_action = 0

        # 2. Infant energy update
        if infant_action == 0:
            self._infant_energy = min(self.MAX_ENERGY, self._infant_energy + 1)
            infant_symbol: Optional[int] = None
        else:
            self._infant_energy -= 1
            infant_symbol = infant_action - 1

        # 3-4. Parent action
        in_reply = len(self._reply_queue) > 0 and not is_yoke
        trigger_id: Optional[int] = None
        reply_step = 0

        if is_yoke:
            parent_action = int(self._yoke_parent_actions[self._yoke_index])
            self._yoke_index += 1
            # Yoke tape is replayed regardless of parent energy.
            if parent_action == 0:
                parent_symbol: Optional[int] = None
            else:
                parent_symbol = parent_action - 1
        elif in_reply:
            parent_symbol = self._reply_queue.pop(0)
            parent_action = parent_symbol + 1
            reply_step = 1 if len(self._reply_queue) == 1 else 2
            self._reply_step = reply_step
            if reply_step == 1:
                self._reply_infant_silent = infant_symbol is None
            elif reply_step == 2:
                if self._reply_infant_silent and infant_symbol is None:
                    self._reply_heard_count += 1
                self._reply_infant_silent = False
                self._reply_step = 0
        elif infant_symbol is not None:
            # Parent listens while infant signals
            parent_action = 0
            parent_symbol = None
        else:
            # Infant silent: parent emits random signal (or noop if energy 0)
            if self._parent_energy == 0:
                parent_action = 0
                parent_symbol = None
            else:
                parent_symbol = int(self._rng.integers(0, self.N_SYMBOLS))
                parent_action = parent_symbol + 1

        # 5. Parent energy update
        if parent_action == 0:
            self._parent_energy = min(self.MAX_ENERGY, self._parent_energy + 1)
        else:
            # Reply / yoke may emit even when energy is already 0; clamp at 0.
            self._parent_energy = max(0, self._parent_energy - 1)

        # 6. Receive buffer update (not during yoke, not during reply emission)
        if not is_yoke and not in_reply:
            if infant_symbol is not None:
                self._recv_buffer.append(infant_symbol)
                if len(self._recv_buffer) > 2:
                    self._recv_buffer = self._recv_buffer[-2:]
                if len(self._recv_buffer) == 2:
                    pair = (self._recv_buffer[0], self._recv_buffer[1])
                    if pair in self.meaningful_pairs:
                        tid = self.meaningful_pairs[pair]
                        reply = self.reply_pairs[tid]
                        self._reply_queue = [reply[0], reply[1]]
                        self._recv_buffer = []
                        self._trigger_count += 1
                        trigger_id = tid
                    else:
                        # Sliding window: drop oldest
                        self._recv_buffer = self._recv_buffer[1:]
            else:
                self._recv_buffer = []

        # 7. Build observation
        obs = self._build_obs(infant_symbol, parent_symbol)

        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps
        if truncated:
            self._episode_started = False

        info = self._base_info(
            infant_action=infant_action,
            parent_action=parent_action,
            trigger_id=trigger_id,
            reply_step=reply_step,
        )
        return obs, 0.0, terminated, truncated, info

    def _build_obs(
        self,
        infant_symbol: Optional[int],
        parent_symbol: Optional[int],
    ) -> np.ndarray:
        energy_norm = float(self._infant_energy) / float(self.MAX_ENERGY)
        onehot = np.zeros(self.N_SYMBOLS, dtype=np.float32)

        if infant_symbol is not None:
            who = 1.0
            onehot[infant_symbol] = 1.0
        elif parent_symbol is not None:
            who = -1.0
            onehot[parent_symbol] = 1.0
        else:
            who = 0.0

        if self.obs_noise > 0.0:
            onehot = onehot + self._rng.normal(
                0.0, self.obs_noise, size=self.N_SYMBOLS
            ).astype(np.float32)

        obs = np.empty(self.OBS_DIM, dtype=np.float32)
        obs[0] = energy_norm
        obs[1] = who
        obs[2:] = onehot
        return obs

    def _base_info(
        self,
        *,
        infant_action: int,
        parent_action: int,
        trigger_id: Optional[int],
        reply_step: int,
    ) -> dict[str, Any]:
        return {
            "infant_action": infant_action,
            "parent_action": parent_action,
            "infant_energy": self._infant_energy,
            "parent_energy": self._parent_energy,
            "trigger_id": trigger_id,
            "reply_step": reply_step,
            "yoke": self._yoke_parent_actions is not None,
            "trigger_count": self._trigger_count,
            "reply_heard_count": self._reply_heard_count,
        }
