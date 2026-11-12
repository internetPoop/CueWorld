"""層 3: 成功経路のプロトコル遷移。正しい動き vs 動きなし。

固定プロトコルを走らせ、正しい動きでは session_success、
動きなしでは成功にならないことを確認する。
"""

from __future__ import annotations

import sys
import unittest
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_DEMO = Path(__file__).resolve().parents[1] / "demo"
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager
from cueworld.tests.validation_helpers import (
    apply_queued_phoneme,
    build_word_phoneme_queue,
    first_active_oid,
    infant_gaze_action,
    infant_raise_action,
    make_protocol_env,
    normalize_infant_action,
    place_object_in_front,
)
from demo_catalog_common import (
    HOUSEHOLD_OFF,
    SLIDE_ARENA,
    find_object,
    ja_engage_action,
    slide_object_placements,
    step_parent,
)


def _run_until_done(
    env: ProtocolManager,
    infant_policy: Callable,
    *,
    max_steps: int = 500,
) -> Dict[str, Any]:
    parent = np.zeros(env.action_space[1].shape, dtype=np.float32)
    states: List[str] = []
    info: Dict[str, Any] = {}
    ctx: Dict[str, Any] = {}
    for t in range(max_steps):
        proto = env.current_protocol
        state = getattr(proto, "state", None) if proto else None
        if state is not None and (not states or states[-1] != state):
            states.append(str(state))
        action = infant_policy(env, state, t, ctx)
        action = normalize_infant_action(env, action)
        _obs, _r, done, _tr, info = env.step([action, parent])
        parent = np.asarray(info.get("parent_action", parent), dtype=np.float32)
        proto = env.current_protocol
        if done or (proto is not None and proto.is_finished):
            break
    proto = env.current_protocol
    return {
        "session_success": bool(getattr(proto, "session_success", False)),
        "state": getattr(proto, "state", None),
        "is_finished": bool(getattr(proto, "is_finished", False)),
        "states": states,
        "steps": t,
        "infant_spoke": bool(getattr(proto, "infant_spoke", False)),
        "word_match": list(getattr(proto, "word_match", []) or []),
    }


def _zero_policy(env, state, t, ctx):
    return np.zeros(env.action_space[0].shape, dtype=np.float32)


class TestProtocolSuccessPaths(unittest.TestCase):
    def _assert_success(self, result: Dict[str, Any], label: str):
        self.assertTrue(
            result["session_success"],
            msg=f"{label}: expected success, got {result}",
        )

    def _assert_no_success(self, result: Dict[str, Any], label: str):
        self.assertFalse(
            result["session_success"],
            msg=f"{label}: expected no success, got {result}",
        )

    def test_ja_looking_correct_vs_still(self):
        for correct in (True, False):
            env = make_protocol_env()
            try:
                env.reset(seed=0)
                oid = first_active_oid(env, "RedCube")
                place_object_in_front(env, oid, distance=1.2)
                env.set_fixed_protocol(
                    "joint_attention",
                    {
                        "fixed_target": oid,
                        "cue_type": "looking",
                        "intent_type": "informative",
                        "infant_initiative": False,
                    },
                )

                def policy(e, state, t, ctx, _oid=oid, _correct=correct):
                    a = np.zeros(e.action_space[0].shape, dtype=np.float32)
                    if _correct and state in (
                        "looking",
                        "pointing",
                        "post_shared_attention",
                        "approach",
                    ):
                        infant_gaze_action(e, _oid, a)
                    return a

                result = _run_until_done(env, policy, max_steps=250)
                if correct:
                    self._assert_success(result, "ja_looking/correct")
                    self.assertIn("post_shared_attention", result["states"])
                else:
                    self._assert_no_success(result, "ja_looking/still")
                    # 動きなしは見るだけ→指差しへ上がってから失敗してよい
                    self.assertTrue(
                        result["state"] == "failure"
                        or "pointing" in result["states"]
                    )
            finally:
                env.close()

    def test_ja_pointing_correct_vs_still(self):
        for correct in (True, False):
            env = make_protocol_env()
            try:
                env.reset(seed=1)
                oid = first_active_oid(env, "RedCube")
                place_object_in_front(env, oid, distance=1.2)
                env.set_fixed_protocol(
                    "joint_attention",
                    {
                        "fixed_target": oid,
                        "cue_type": "pointing",
                        "intent_type": "informative",
                        "infant_initiative": False,
                    },
                )

                def policy(e, state, t, ctx, _oid=oid, _correct=correct):
                    a = np.zeros(e.action_space[0].shape, dtype=np.float32)
                    if _correct and state in (
                        "pointing",
                        "post_shared_attention",
                        "approach",
                    ):
                        infant_gaze_action(e, _oid, a)
                    return a

                result = _run_until_done(env, policy, max_steps=250)
                if correct:
                    self._assert_success(result, "ja_pointing/correct")
                else:
                    self._assert_no_success(result, "ja_pointing/still")
            finally:
                env.close()

    def test_ja_shake_correct_vs_still(self):
        """掲げて見せるは cinematic 配置 + ja_engage_action で成功する。"""
        for correct in (True, False):
            placements = slide_object_placements()
            base = BaseEnv(
                gui=False,
                num_agents=2,
                num_objects=len(placements),
                object_counts={},
                max_steps=400,
                action_repeat=1,
                arena_half_extents=SLIDE_ARENA,
                arena_rows=3,
                arena_cols=4,
                fixed_placements=placements,
                placement_method="grid",
                use_cinematic_model=True,
            )
            env = ProtocolManager(
                InfantSignalDetector(ParentTools(base)),
                household_interval_steps=HOUSEHOLD_OFF,
                rejection=False,
            )
            try:
                _, info = env.reset(seed=42)
                red = find_object(env, "RedCube")
                env.set_fixed_protocol(
                    "joint_attention",
                    {
                        "fixed_target": red,
                        "cue_type": "shake",
                        "intent_type": "informative",
                        "infant_initiative": False,
                    },
                )
                speech_queue: deque = deque()
                speech_started = [False]
                states: List[str] = []
                for t in range(300):
                    proto = env.current_protocol
                    state = getattr(proto, "state", None)
                    if state is not None and (not states or states[-1] != state):
                        states.append(str(state))
                    if correct:
                        action = ja_engage_action(
                            env, red, speech_queue, speech_started, False
                        )
                    else:
                        action = np.zeros(env.action_space[0].shape, dtype=np.float32)
                    _, _, _, _, info, _ = step_parent(env, info, action)
                    if proto.is_finished or state in (
                        "completed",
                        "failure",
                        "post_shared_attention",
                    ):
                        if state == "post_shared_attention" and correct:
                            break
                        if proto.is_finished:
                            break
                proto = env.current_protocol
                result = {
                    "session_success": bool(proto.session_success),
                    "state": proto.state,
                    "states": states,
                }
                if correct:
                    self._assert_success(result, "ja_shake/correct")
                else:
                    self._assert_no_success(result, "ja_shake/still")
            finally:
                env.close()

    def test_giving_correct_vs_still(self):
        for correct in (True, False):
            env = make_protocol_env()
            try:
                env.reset(seed=2)
                oid = first_active_oid(env, "RedCube")
                place_object_in_front(env, oid, distance=1.5)
                env.set_fixed_protocol(
                    "object_giving",
                    {"fixed_target": oid, "infant_initiative": False},
                )

                def policy(e, state, t, ctx, _oid=oid, _correct=correct):
                    a = np.zeros(e.action_space[0].shape, dtype=np.float32)
                    if not _correct:
                        return a
                    if state == "waiting_infant":
                        e.grasp_object(
                            e.infant,
                            _oid,
                            action=a,
                            approach_dist=float(e.grasp_approach_dist),
                        )
                    elif state in ("fetching", "approach", "placing", "showing", "reaction"):
                        infant_gaze_action(e, _oid, a)
                    return a

                result = _run_until_done(env, policy, max_steps=450)
                if correct:
                    self._assert_success(result, "giving/correct")
                    self.assertIn("waiting_infant", result["states"])
                else:
                    self._assert_no_success(result, "giving/still")
            finally:
                env.close()

    def test_dyadic_arm_raise_correct_vs_still(self):
        for correct in (True, False):
            env = make_protocol_env()
            try:
                env.reset(seed=3)
                env.set_fixed_protocol(
                    "dyadic",
                    {
                        "infant_initiative": False,
                        "parent_channels": ["arm_raise"],
                    },
                )

                def policy(e, state, t, ctx, _correct=correct):
                    a = np.zeros(e.action_space[0].shape, dtype=np.float32)
                    if _correct and state in (
                        "performing_parent_cue",
                        "waiting_babble",
                        "waiting_infant",
                    ):
                        infant_raise_action(e, a)
                    return a

                result = _run_until_done(env, policy, max_steps=250)
                if correct:
                    self._assert_success(result, "dyadic_arm/correct")
                else:
                    self._assert_no_success(result, "dyadic_arm/still")
            finally:
                env.close()

    def test_dyadic_hand_open_close_correct_vs_still(self):
        for correct in (True, False):
            env = make_protocol_env()
            try:
                env.reset(seed=4)
                env.set_fixed_protocol(
                    "dyadic",
                    {
                        "infant_initiative": False,
                        "parent_channels": ["hand_open_close"],
                    },
                )

                def policy(e, state, t, ctx, _correct=correct):
                    a = np.zeros(e.action_space[0].shape, dtype=np.float32)
                    if _correct and state in (
                        "performing_parent_cue",
                        "waiting_babble",
                        "waiting_infant",
                    ):
                        # 掲げを維持したまま指だけ開閉（gesture=True）
                        infant_raise_action(e, a)
                        phase = (t % 30) / 30.0
                        e.animate_hand_open_close(
                            e.infant, action=a, phase=phase, gesture=True
                        )
                    return a

                result = _run_until_done(env, policy, max_steps=300)
                if correct:
                    self._assert_success(result, "dyadic_hand/correct")
                else:
                    self._assert_no_success(result, "dyadic_hand/still")
            finally:
                env.close()

    def test_dyadic_voice_correct_vs_still(self):
        """幼児始動の声: 開始時の喃語チャンクでエコーが始まる。"""
        babble = ["p0", "p1", "p0", "p1"]
        # 正しい動き: context に喃語を渡す（デモと同じ起動）
        env = make_protocol_env()
        try:
            env.reset(seed=5)
            env.set_fixed_protocol(
                "dyadic",
                {"infant_initiative": True, "babbling_chunk": list(babble)},
            )
            result = _run_until_done(env, _zero_policy, max_steps=200)
            self._assert_success(result, "dyadic_voice/correct")
            self.assertIn("echoing", result["states"])
        finally:
            env.close()

        # 動きなし: 喃語も渡さずゼロ行動 → 成功しない
        env = make_protocol_env()
        try:
            env.reset(seed=5)
            env.set_fixed_protocol("dyadic", {"infant_initiative": True})
            result = _run_until_done(env, _zero_policy, max_steps=200)
            self._assert_no_success(result, "dyadic_voice/still")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
