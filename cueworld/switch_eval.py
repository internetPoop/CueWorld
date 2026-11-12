"""
バッチ E: 切替頻度の短い観測（物理なし）。

test_protocol_manager_intent.py と同じ層で信号辞書を直接入れ、
典型 T ぶん ProtocolManager.update を回す。点灯率は短い観測では 100%（一定信号）。
"""
from __future__ import annotations

import json
import os
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from cueworld.protocol_manager import HOUSEHOLD_INTERVAL_PRODUCTION, ProtocolManager
from cueworld.protocols.base import BaseProtocol

# 典型 T（成功1回, step）。JA は幼児始動・指差しの実測。家事は未計測なので giving 相当。
TYPICAL_STEPS = {
    "joint_attention": 130,
    "object_giving": 175,
    "dyadic": 105,
    "household_solo": 175,  # 仮置き（giving 相当）。実測は観測のあと。
    "observance": 200,  # 家事間隔の既定に合わせた見守り観測の長さ
}
# 上限 T（リトライなし1周）。典型の 2 倍を仮置き。dwell json があれば上書き可。
UPPER_STEPS = {name: steps * 2 for name, steps in TYPICAL_STEPS.items()}
UPPER_STEPS["observance"] = 400

DEFAULT_HOUSEHOLD_INTERVAL = 200
# 本番ランは 6000。コンストラクタ既定・短い観測は 200 のまま。
PRODUCTION_HOUSEHOLD_INTERVAL = HOUSEHOLD_INTERVAL_PRODUCTION
DEFAULT_N_TRIALS = 40
DEFAULT_SEED = 0


class _MockObject:
    def __init__(self, oid: int, concept_id: str, is_platform: bool = False):
        self.id = oid
        self.concept_id = concept_id
        self.is_platform = is_platform
        self.name = concept_id


class MinimalSwitchEnv(gym.Env):
    """PyBullet 不要の最小 env。update() に信号を直接渡すので get_normalized_signals は持たない。"""

    action_num = 17
    eval_mode = False
    active_signals = None
    target_obj_id = None

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = (
            gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32),
            gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32),
        )
        self.step_count = 0
        self.parent = {"body": 1}
        self.infant = {"body": 2, "held_obj": None}
        self.all_objects = {
            101: _MockObject(101, "red_cube"),
            102: _MockObject(102, "green_cube"),
            201: _MockObject(201, "red_board", is_platform=True),
        }
        self.active_objects = [101, 102]
        self.trackable_targets = [101, 102, self.parent["body"]]
        self.reach_radius = 0.4
        self._in_reach = set()
        self.voice_volume_threshold = 0.25
        self.household_task_candidates = [
            {"board_id": 201, "item_ids": [101]},
        ]

    def object_in_agent_reach(self, agent, oid):
        return oid in self._in_reach

    def reset(self, **kwargs):
        self.step_count = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        return np.zeros(1, dtype=np.float32), [0.0, 0.0], False, False, {}


class StubProtocol(BaseProtocol):
    """_manage_state / _determine_action だけ潰す。維持スコアは基底の式。"""

    def __init__(
        self,
        name: str,
        env: Any,
        *,
        typical_steps: Optional[int] = None,
        persistence: float = 0.0,
        solo: bool = False,
        level: int = 0,
        request_type: bool = False,
        response_type: bool = False,
        has_target: bool = False,
        cue_type: Optional[str] = None,
    ):
        super().__init__(name, env)
        self.typical_steps = typical_steps
        self.persistence = persistence
        self.solo = solo
        self.level = level
        self.request_type = request_type
        self.response_type = response_type
        self.has_target = has_target
        self.cue_type = cue_type
        self._fixed_persistence = persistence if typical_steps is None else None

    def _manage_state(self, infant_signals: Dict[str, Any]) -> None:
        return

    def _determine_action(self) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)

    def _calculate_persistence(self, infant_signals: Dict[str, Any]):
        if self._fixed_persistence is not None:
            self.persistence = self._fixed_persistence
            return
        super()._calculate_persistence(infant_signals)

    def infant_is_engaged(self, infant_signals: Dict[str, Any]) -> bool:
        if self.has_target:
            return self._infant_attending_current_object(infant_signals)
        if self.state in ("imitating", "acknowledging"):
            return True
        return bool(getattr(self, "_infant_has_responded", False))


class StubHouseholdProtocol(StubProtocol):
    def __init__(self, env: Any):
        super().__init__(
            "HouseholdSoloProtocol",
            env,
            typical_steps=None,
            persistence=5.0,
            solo=True,
        )

    def _get_task_candidates(self):
        return list(getattr(self.env, "household_task_candidates", []) or [])


def install_stubs(manager: ProtocolManager) -> None:
    env = manager.env
    stubs = {
        "observance": StubProtocol(
            "ObservanceProtocol", env, persistence=1.0, solo=True
        ),
        "joint_attention": StubProtocol(
            "JointAttentionProtocol",
            env,
            typical_steps=TYPICAL_STEPS["joint_attention"],
            level=2,
            request_type=True,
            response_type=True,
            has_target=True,
            cue_type="pointing",
        ),
        "object_giving": StubProtocol(
            "ObjectGivingProtocol",
            env,
            typical_steps=TYPICAL_STEPS["object_giving"],
            level=1,
            request_type=True,
            response_type=True,
            has_target=True,
        ),
        "dyadic": StubProtocol(
            "DyadicProtocol",
            env,
            typical_steps=TYPICAL_STEPS["dyadic"],
            level=0,
            request_type=True,
            response_type=True,
        ),
        "household_solo": StubHouseholdProtocol(env),
    }
    manager.protocols = stubs
    manager.current_protocol = stubs["observance"]
    manager.current_protocol_name = "observance"
    manager.response_protocol = ["joint_attention", "dyadic", "object_giving"]
    manager.request_protocol = ["joint_attention", "dyadic", "object_giving"]


def make_manager(
    *,
    household_interval_steps: int = DEFAULT_HOUSEHOLD_INTERVAL,
    check_interval: int = 10,
) -> Tuple[MinimalSwitchEnv, ProtocolManager]:
    env = MinimalSwitchEnv()
    manager = ProtocolManager(
        env,
        check_interval=check_interval,
        household_interval_steps=household_interval_steps,
        decay_rate=0.5,
    )
    install_stubs(manager)
    return env, manager


def zeros(manager: ProtocolManager) -> Dict[str, Any]:
    signals = {key: 0.0 for key in manager.signal_keys}
    n_targets = len(getattr(manager.env, "trackable_targets", []) or [])
    signals["gesture_targets"] = np.zeros(max(n_targets, 3), dtype=float)
    signals["gaze_targets"] = np.zeros(max(n_targets, 3), dtype=float)
    signals["words"] = {}
    signals["looking_at_parent_eye"] = 0.0
    signals["touching"] = 0.0
    signals["voice_best_distance"] = 10.0
    return signals


def pointing_at(manager: ProtocolManager, target_idx: int = 0) -> Dict[str, Any]:
    s = zeros(manager)
    s["point_intensity"] = 1.0
    s["hand_target_stability"] = 0.8
    s["gesture_targets"][target_idx] = 1.0
    return s


def reach_at(manager: ProtocolManager, target_idx: int = 0) -> Dict[str, Any]:
    s = zeros(manager)
    s["open_intensity"] = 1.0
    s["hand_target_stability"] = 0.8
    s["gesture_targets"][target_idx] = 1.0
    s["holding"] = 0.0
    return s


def gaze_object(manager: ProtocolManager, target_idx: int = 0) -> Dict[str, Any]:
    s = zeros(manager)
    s["gaze_target_stability"] = 1.0
    s["gaze_targets"][target_idx] = 1.0
    return s


def gaze_and_reach(
    manager: ProtocolManager,
    *,
    gaze_idx: int = 0,
    reach_idx: int = 1,
) -> Dict[str, Any]:
    """対象への視線 + リーチ（同じ物 / 別物は idx で分ける）。"""
    s = reach_at(manager, reach_idx)
    s["gaze_target_stability"] = 1.0
    s["gaze_targets"][gaze_idx] = 1.0
    return s


def parent_eye(manager: ProtocolManager) -> Dict[str, Any]:
    s = zeros(manager)
    s["looking_at_parent_eye"] = 0.7
    return s


def word_at(manager: ProtocolManager, concept_id: str = "red_cube") -> Dict[str, Any]:
    s = zeros(manager)
    s["voice_target_stability"] = 1.0
    s["voice_volume"] = 1.0
    s["voice_best_distance"] = 0.1
    s["words"] = {concept_id: 0.1}
    return s


def set_warm(manager: ProtocolManager) -> None:
    n = manager.base_try_count
    manager.current_try_count = n
    manager.level_try_counts = [n] * len(manager.level_try_counts)
    manager.level_scores = [0.5] * len(manager.level_scores)
    manager.infant_initiative_rate = 0.0


def set_cold(manager: ProtocolManager) -> None:
    manager.current_try_count = 0
    manager.level_try_counts = [0] * len(manager.level_try_counts)
    manager.level_scores = [0.0] * len(manager.level_scores)
    manager.infant_initiative_rate = 0.0


def _prob_map(choices: List[str], probs: Optional[np.ndarray]) -> Dict[str, float]:
    if not choices or probs is None:
        return {}
    return {str(name): float(p) for name, p in zip(choices, probs)}


def _switch_prob(pmap: Dict[str, float], stay_key: str) -> float:
    return max(0.0, 1.0 - float(pmap.get(stay_key, 0.0)))


@dataclass
class TrialResult:
    switched: bool = False
    first_switch_step: Optional[int] = None
    dest: Optional[str] = None
    path: str = ""
    check_probs: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class RowResult:
    label: str
    protocol: str
    signal: str
    warm: bool
    steps: int
    n_trials: int
    path: str
    mean_p_switch: float
    first_check_p_switch: float
    empirical_switch_rate: float
    mean_first_switch_step: Optional[float]
    dest_counts: Dict[str, int]
    mean_probs: Dict[str, float]
    first_check_probs: Dict[str, float]


def run_row(
    *,
    protocol_name: str,
    signals_fn: Callable[[ProtocolManager], Dict[str, Any]],
    steps: int,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = DEFAULT_SEED,
    warm: bool = False,
    elapsed_offset: int = 0,
    retry_count: int = 0,
    household_due: bool = False,
    household_interval_steps: int = DEFAULT_HOUSEHOLD_INTERVAL,
    label: str = "",
    signal_name: str = "",
    switch_context: Optional[Dict[str, Any]] = None,
) -> RowResult:
    dest_counts: Dict[str, int] = {}
    switch_flags: List[int] = []
    first_steps: List[int] = []
    all_first_probs: List[Dict[str, float]] = []
    all_mean_probs: List[Dict[str, float]] = []

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        switched = False
        first_switch_step = None
        dest = None
        path = "observance" if protocol_name == "observance" else "response"
        check_probs: List[Dict[str, float]] = []

        with redirect_stdout(StringIO()):
            env, manager = make_manager(household_interval_steps=household_interval_steps)
            if warm:
                set_warm(manager)
            else:
                set_cold(manager)

            env.step_count = 0
            manager.last_household_end_step = 0
            manager.check_counter = 0
            context = dict(switch_context or {"infant_initiative": False})
            manager.switch_protocol(protocol_name, context)

            proto = manager.current_protocol
            if elapsed_offset > 0:
                proto.start_step = env.step_count - elapsed_offset
                manager.current_protocol_start_step = proto.start_step
            proto.retry_count = retry_count

            if household_due:
                env.step_count = household_interval_steps
                manager.last_household_end_step = 0

            signals = signals_fn(manager)
            proto._calculate_persistence(signals)

            for _ in range(steps):
                env.step_count += 1
                would_check = (manager.check_counter + 1) >= manager.check_interval
                if would_check and not switched:
                    if manager.current_protocol_name == "observance":
                        r_choices, r_probs = manager._response_probs(signals)
                        r_map = _prob_map(r_choices, r_probs)
                        i_choices, i_probs = manager._initiative_probs(signals)
                        i_map = _prob_map(i_choices, i_probs)
                        p_infant = _switch_prob(r_map, "maintain")
                        p_parent = _switch_prob(i_map, "observance")
                        merged = dict(r_map)
                        merged.update({f"init:{k}": v for k, v in i_map.items()})
                        merged["p_infant_switch"] = p_infant
                        merged["p_parent_switch"] = p_parent
                        merged["p_household_due"] = (
                            1.0 if manager._can_start_household() else 0.0
                        )
                        check_probs.append(merged)
                        path = "observance"
                    else:
                        choices, probs = manager._response_probs(signals)
                        pmap = _prob_map(choices, probs)
                        pmap["p_switch"] = _switch_prob(pmap, "maintain")
                        check_probs.append(pmap)
                        path = "response"

                prev_name = manager.current_protocol_name
                np.random.seed(int(rng.integers(0, 2**31 - 1)))
                manager.update(signals)
                if not switched and manager.current_protocol_name != prev_name:
                    switched = True
                    first_switch_step = env.step_count
                    dest = manager.current_protocol_name

        switch_flags.append(int(switched))
        if first_switch_step is not None:
            first_steps.append(first_switch_step)
        if dest:
            dest_counts[dest] = dest_counts.get(dest, 0) + 1
        if check_probs:
            all_first_probs.append(check_probs[0])
            keys = set()
            for p in check_probs:
                keys.update(p)
            mean_p = {k: float(np.mean([p.get(k, 0.0) for p in check_probs])) for k in keys}
            all_mean_probs.append(mean_p)

    def _avg_maps(maps: List[Dict[str, float]]) -> Dict[str, float]:
        if not maps:
            return {}
        keys = set()
        for m in maps:
            keys.update(m)
        return {k: float(np.mean([m.get(k, 0.0) for m in maps])) for k in sorted(keys)}

    mean_probs = _avg_maps(all_mean_probs)
    first_probs = _avg_maps(all_first_probs)
    if household_due:
        mean_p_switch = float(mean_probs.get("p_household_due", 0.0))
        first_p_switch = float(first_probs.get("p_household_due", 0.0))
        path = "household_clock"
    elif protocol_name == "observance":
        # 見守り中は幼児訴えが先。親始動の P は「訴えが無いときの条件付き」。
        mean_p_switch = float(mean_probs.get("p_infant_switch", 0.0))
        first_p_switch = float(first_probs.get("p_infant_switch", 0.0))
        if mean_p_switch < 1e-9:
            mean_p_switch = float(mean_probs.get("p_parent_switch", 0.0))
            first_p_switch = float(first_probs.get("p_parent_switch", 0.0))
        path = "observance"
    else:
        mean_p_switch = float(mean_probs.get("p_switch", 0.0))
        first_p_switch = float(first_probs.get("p_switch", 0.0))
        path = "response"

    return RowResult(
        label=label,
        protocol=protocol_name,
        signal=signal_name,
        warm=warm,
        steps=steps,
        n_trials=n_trials,
        path=path,
        mean_p_switch=mean_p_switch,
        first_check_p_switch=first_p_switch,
        empirical_switch_rate=float(np.mean(switch_flags)) if switch_flags else 0.0,
        mean_first_switch_step=(float(np.mean(first_steps)) if first_steps else None),
        dest_counts=dest_counts,
        mean_probs=mean_probs,
        first_check_probs=first_probs,
    )


def load_dwell_overrides(path: Optional[str] = None) -> Dict[str, int]:
    """measure_protocol_state_dwell.py の json を明示したときだけ典型 T を上書きする。"""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    totals: Dict[str, List[int]] = {}
    for run in data.get("runs", []):
        name = run.get("protocol")
        total = run.get("total_steps")
        if name and isinstance(total, int):
            totals.setdefault(name, []).append(total)
    return {name: int(round(float(np.median(vals)))) for name, vals in totals.items()}


def apply_dwell_overrides(overrides: Dict[str, int]) -> None:
    for name, steps in overrides.items():
        if name in TYPICAL_STEPS:
            TYPICAL_STEPS[name] = int(steps)
            UPPER_STEPS[name] = int(steps) * 2


def observation_specs(
    *,
    household_interval_steps: int = DEFAULT_HOUSEHOLD_INTERVAL,
) -> List[Dict[str, Any]]:
    """乗車・手がかりの短い観測の行。弱い腕・くれ中・二項中は足さない。"""
    ja_t = TYPICAL_STEPS["joint_attention"]
    house_t = TYPICAL_STEPS["household_solo"]
    obs_t = household_interval_steps
    ja_upper = UPPER_STEPS["joint_attention"]
    house_upper = UPPER_STEPS["household_solo"]
    obs_upper = UPPER_STEPS["observance"]
    ja_ctx = {"infant_initiative": False, "fixed_target": 101}

    def _both(base: Dict[str, Any], typical: int, upper: int) -> List[Dict[str, Any]]:
        rows = []
        seen = set()
        for length_name, steps in (("typical", typical), ("upper", upper)):
            if steps in seen:
                continue
            seen.add(steps)
            row = dict(base)
            row["steps"] = steps
            row["label"] = f"{base['label']} [{length_name} T={steps}]"
            rows.append(row)
        return rows

    specs: List[Dict[str, Any]] = []
    specs += _both(
        {
            "label": "observance / zeros / cold",
            "protocol_name": "observance",
            "signals_fn": zeros,
            "signal_name": "zeros",
            "warm": False,
        },
        obs_t,
        obs_upper,
    )
    specs += _both(
        {
            "label": "observance / zeros / warm",
            "protocol_name": "observance",
            "signals_fn": zeros,
            "signal_name": "zeros",
            "warm": True,
        },
        obs_t,
        obs_upper,
    )
    specs += _both(
        {
            "label": "observance / gaze / cold",
            "protocol_name": "observance",
            "signals_fn": gaze_object,
            "signal_name": "gaze",
            "warm": False,
        },
        obs_t,
        obs_upper,
    )
    specs += _both(
        {
            "label": "observance / gaze / warm",
            "protocol_name": "observance",
            "signals_fn": gaze_object,
            "signal_name": "gaze",
            "warm": True,
        },
        obs_t,
        obs_upper,
    )
    specs += _both(
        {
            "label": "observance / parent_eye / cold",
            "protocol_name": "observance",
            "signals_fn": parent_eye,
            "signal_name": "parent_eye",
            "warm": False,
        },
        obs_t,
        obs_upper,
    )
    specs += _both(
        {
            "label": "observance / parent_eye / warm",
            "protocol_name": "observance",
            "signals_fn": parent_eye,
            "signal_name": "parent_eye",
            "warm": True,
        },
        obs_t,
        obs_upper,
    )
    specs += _both(
        {
            "label": "observance / pointing",
            "protocol_name": "observance",
            "signals_fn": pointing_at,
            "signal_name": "pointing",
            "warm": False,
        },
        obs_t,
        obs_upper,
    )
    specs += _both(
        {
            "label": "household / pointing",
            "protocol_name": "household_solo",
            "signals_fn": pointing_at,
            "signal_name": "pointing",
            "warm": False,
        },
        house_t,
        house_upper,
    )
    specs += _both(
        {
            "label": "household / zeros",
            "protocol_name": "household_solo",
            "signals_fn": zeros,
            "signal_name": "zeros",
            "warm": False,
        },
        house_t,
        house_upper,
    )
    specs += _both(
        {
            "label": "JA not-engaged / reach",
            "protocol_name": "joint_attention",
            "signals_fn": reach_at,
            "signal_name": "reach",
            "warm": False,
            "retry_count": 0,
            "switch_context": ja_ctx,
        },
        ja_t,
        ja_upper,
    )
    specs += _both(
        {
            "label": "JA engaged / other reach",
            "protocol_name": "joint_attention",
            "signals_fn": lambda m: gaze_and_reach(m, gaze_idx=0, reach_idx=1),
            "signal_name": "gaze+reach_other",
            "warm": False,
            "retry_count": 0,
            "switch_context": ja_ctx,
        },
        ja_t,
        ja_upper,
    )
    specs += _both(
        {
            "label": "JA engaged / same reach",
            "protocol_name": "joint_attention",
            "signals_fn": lambda m: gaze_and_reach(m, gaze_idx=0, reach_idx=0),
            "signal_name": "gaze+reach_same",
            "warm": False,
            "retry_count": 0,
            "switch_context": ja_ctx,
        },
        ja_t,
        ja_upper,
    )
    specs += _both(
        {
            "label": "JA retry / reach",
            "protocol_name": "joint_attention",
            "signals_fn": reach_at,
            "signal_name": "reach",
            "warm": False,
            "retry_count": 1,
            "switch_context": ja_ctx,
        },
        ja_t,
        ja_upper,
    )
    specs += _both(
        {
            "label": "observance / household due / zeros",
            "protocol_name": "observance",
            "signals_fn": zeros,
            "signal_name": "zeros",
            "warm": False,
            "household_due": True,
        },
        household_interval_steps,
        household_interval_steps,
    )
    return specs


def run_all_rows(
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = DEFAULT_SEED,
    household_interval_steps: int = DEFAULT_HOUSEHOLD_INTERVAL,
    dwell_json: Optional[str] = None,
) -> List[RowResult]:
    overrides = load_dwell_overrides(dwell_json)
    if overrides:
        apply_dwell_overrides(overrides)
    results = []
    for spec in observation_specs(household_interval_steps=household_interval_steps):
        results.append(
            run_row(
                protocol_name=spec["protocol_name"],
                signals_fn=spec["signals_fn"],
                steps=spec["steps"],
                n_trials=n_trials,
                seed=seed,
                warm=spec.get("warm", False),
                elapsed_offset=spec.get("elapsed_offset", 0),
                retry_count=spec.get("retry_count", 0),
                household_due=spec.get("household_due", False),
                household_interval_steps=household_interval_steps,
                label=spec["label"],
                signal_name=spec["signal_name"],
                switch_context=spec.get("switch_context"),
            )
        )
    return results


def format_table(results: List[RowResult]) -> str:
    lines = [
        "| row | warm | steps | P(infant) | P(parent) | P(hh due) | P(switch) | empirical | first switch | dest |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        dest = ",".join(f"{k}:{v}" for k, v in sorted(r.dest_counts.items())) or "-"
        first_step = f"{r.mean_first_switch_step:.1f}" if r.mean_first_switch_step is not None else "-"
        fp = r.first_check_probs
        p_infant = fp.get("p_infant_switch", fp.get("p_switch", 0.0))
        p_parent = fp.get("p_parent_switch", 0.0)
        p_hh = fp.get("p_household_due", 0.0)
        lines.append(
            f"| {r.label} | {int(r.warm)} | {r.steps} | "
            f"{p_infant:.3f} | {p_parent:.3f} | {p_hh:.3f} | "
            f"{r.first_check_p_switch:.3f} | {r.empirical_switch_rate:.3f} | "
            f"{first_step} | {dest} |"
        )
    return "\n".join(lines)


def results_to_jsonable(results: List[RowResult]) -> List[Dict[str, Any]]:
    out = []
    for r in results:
        row = asdict(r)
        out.append(row)
    return out
