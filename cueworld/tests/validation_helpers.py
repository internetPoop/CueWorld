"""環境妥当性テスト用の共有ヘルパ。

特権の動き（gaze / point / grasp / phoneme / raise）を実環境の step に載せる。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.intent import compute_intent_flags
from cueworld.language.vocabulary import CONCEPT_TO_PHONEMES
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager


def make_signal_env(**kwargs) -> InfantSignalDetector:
    defaults = dict(
        gui=False,
        num_agents=2,
        num_objects=2,
        object_counts={"RedCube": 1, "BlueCube": 1},
        max_steps=5000,
        action_repeat=1,
        arena_half_extents=(2.5, 2.0),
        placement_method="grid",
    )
    defaults.update(kwargs)
    base = BaseEnv(**defaults)
    return InfantSignalDetector(ParentTools(base))


def make_protocol_env(*, dense_reward: bool = False, **kwargs) -> ProtocolManager:
    signal_env = make_signal_env(**kwargs)
    return ProtocolManager(signal_env, dense_reward=dense_reward)


def normalize_infant_action(env, infant_action: np.ndarray) -> np.ndarray:
    for name, step_size in env.infant_joint_mappings.items():
        if name in ("volume", "pitch"):
            continue
        try:
            idx = env.action_map[name]
            infant_action[idx] = infant_action[idx] / (
                step_size * getattr(env, "action_repeat", 1)
            )
        except KeyError:
            pass
    return np.clip(infant_action, -1.0, 1.0)


def zero_actions(env) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros(env.action_space[0].shape, dtype=np.float32),
        np.zeros(env.action_space[1].shape, dtype=np.float32),
    )


def place_object_in_front(
    env,
    oid: int,
    *,
    distance: float,
    z: float = 0.06,
) -> None:
    """幼児の胴体前方へ物を置く。"""
    infant_pos, fwd = env._agent_torso_pose(env.infant)
    new_pos = [
        float(infant_pos[0] + fwd[0] * distance),
        float(infant_pos[1] + fwd[1] * distance),
        float(z),
    ]
    p.resetBasePositionAndOrientation(
        oid, new_pos, [0, 0, 0, 1], physicsClientId=env.client_id
    )


def first_active_oid(env, class_name: Optional[str] = None) -> int:
    for oid in list(env.active_objects):
        if class_name is None:
            return int(oid)
        if env.all_objects[oid].__class__.__name__ == class_name:
            return int(oid)
    raise RuntimeError(f"no active object for class={class_name}")


def build_word_phoneme_queue(
    concept_id: str,
    *,
    duration: int = 12,
    silence_frames: int = 16,
    phoneme_coords_override: Optional[Sequence[np.ndarray]] = None,
) -> Deque[Any]:
    """音素ラベルまたは座標ベクトルのキュー。silence は None。"""
    queue: Deque[Any] = deque()
    if phoneme_coords_override is not None:
        for coords in phoneme_coords_override:
            for _ in range(int(duration)):
                queue.append(np.asarray(coords, dtype=np.float32))
    else:
        phonemes = CONCEPT_TO_PHONEMES.get(concept_id, [])
        for phoneme in phonemes:
            for _ in range(int(duration)):
                queue.append(phoneme)
    for _ in range(int(silence_frames)):
        queue.append(None)
    return queue


def apply_queued_phoneme(env, infant_action: np.ndarray, speech_queue: Deque[Any]) -> None:
    if len(speech_queue) == 0:
        return
    item = speech_queue.popleft()
    if item is None:
        if "volume" in env.action_map:
            infant_action[env.action_map["volume"]] = -1.0
        return
    if isinstance(item, str):
        mapping = getattr(env, "phoneme_mapping", {})
        if item not in mapping:
            return
        env.apply_phoneme_to_action(infant_action, mapping[item], volume=1.0)
        return
    env.apply_phoneme_to_action(infant_action, np.asarray(item, dtype=np.float32), volume=1.0)


def drive_joint_targets(env, agent, action: np.ndarray, targets: Dict[str, float]) -> None:
    """関節を目標角へ寄せる差分を action に書く。"""
    for name, target in targets.items():
        if name not in env.action_map:
            continue
        cur = float(env.get_joint_angle(agent, name))
        action[env.action_map[name]] = float(target) - cur


def infant_raise_action(env, action: Optional[np.ndarray] = None) -> np.ndarray:
    """目線帯に手が入る掲げ姿勢へ駆動する。"""
    if action is None:
        action = np.zeros(env.action_space[0].shape, dtype=np.float32)
    # 実測で hand_z≈0.63 になる姿勢（band_lo≈0.37）
    drive_joint_targets(
        env,
        env.infant,
        action,
        {
            "shoulder_yaw_joint": -1.5,
            "shoulder_pitch_joint": -2.9,
            "elbow_joint": 0.0,
            "roll_wrist_joint": 0.0,
            "pitch_wrist_joint": 0.0,
        },
    )
    env.open_hand(env.infant, action=action)
    return action


def infant_point_action(env, oid: int, action: Optional[np.ndarray] = None) -> np.ndarray:
    if action is None:
        action = np.zeros(env.action_space[0].shape, dtype=np.float32)
    env.point_object(env.infant, oid, action=action)
    env.gaze_object(env.infant, oid, yaw_link="head", action=action)
    return action


def infant_reach_action(
    env,
    oid: int,
    action: Optional[np.ndarray] = None,
    *,
    reaching_phase: bool = False,
) -> Tuple[np.ndarray, bool]:
    if action is None:
        action = np.zeros(env.action_space[0].shape, dtype=np.float32)
    if not reaching_phase:
        yaw, _pitch = env.gaze_object(env.infant, oid, yaw_link="torso", action=action)
        if abs(yaw) < math.radians(5.0):
            reaching_phase = True
    else:
        env.hand_reach_object(env.infant, oid, action=action)
        env.gaze_object(env.infant, oid, yaw_link="head", action=action)
        env.open_hand(env.infant, action=action)
    return action, reaching_phase


def infant_gaze_action(env, oid: int, action: Optional[np.ndarray] = None) -> np.ndarray:
    if action is None:
        action = np.zeros(env.action_space[0].shape, dtype=np.float32)
    env.gaze_object(env.infant, oid, yaw_link="head", action=action)
    return action


def step_with_parent(
    env,
    infant_action: np.ndarray,
    parent_action: Optional[np.ndarray] = None,
) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
    if parent_action is None:
        parent_action = np.zeros(env.action_space[1].shape, dtype=np.float32)
    obs, reward, done, truncated, info = env.step([infant_action, parent_action])
    next_parent = info.get("parent_action")
    if next_parent is not None:
        info = dict(info)
        info["_next_parent_action"] = np.asarray(next_parent, dtype=np.float32)
    return obs, reward, done, truncated, info


def signals_of(info: Dict[str, Any], env) -> Dict[str, Any]:
    if isinstance(info, dict) and info.get("infant_signals") is not None:
        return info["infant_signals"]
    if hasattr(env, "get_normalized_signals"):
        return env.get_normalized_signals()
    return {}


def intent_flags_from_body(env, info: Dict[str, Any]) -> Dict[str, float]:
    signals = signals_of(info, env)
    return compute_intent_flags(signals)


def pm_intents_after_strength(pm: ProtocolManager, signals: Dict[str, Any]) -> Dict[str, float]:
    """到達圏ゲート込みの意図フラグ（くじの前）。"""
    pm._calculate_signal_strength(signals)
    meta = getattr(pm, "_last_intent_meta", {}) or {}
    intents = dict(meta.get("intents") or {})
    return {k: float(intents.get(k, 0.0)) for k in ("pointing", "reach", "weak_arm", "word", "dyadic")}


def phoneme_vertex(env, label: str) -> np.ndarray:
    return np.asarray(env.phoneme_mapping[label], dtype=np.float32)


def neighbor_phoneme_label(label: str) -> str:
    """立方体で 1 ビット違いの隣音素。"""
    i = int(label[1:])
    return f"p{i ^ 1}"  # 最下位ビットを反転


def lerp_phoneme(vertex: np.ndarray, other: np.ndarray, t: float) -> np.ndarray:
    """t=1 で vertex、t=0 で other。"""
    t = float(t)
    return np.asarray(other, dtype=np.float32) * (1.0 - t) + np.asarray(vertex, dtype=np.float32) * t
