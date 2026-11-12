"""層 4: 切り分け条件の報酬ゲート。

密な報酬オン・成功で閉じるオンで、特権スクリプトが成功と成形項を出し、
動きなし対照は成功しないことを確認する。
"""

from __future__ import annotations

import unittest
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from cueworld.protocols.dyadic import DENSE_SUCCESS_BONUS as DYADIC_BONUS
from cueworld.protocols.joint_attention import DENSE_SUCCESS_BONUS as JA_BONUS
from cueworld.protocols.object_giving import DENSE_GRASP_BONUS
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


def _accumulate_dense(env, infant_policy, *, max_steps: int = 400) -> Dict[str, Any]:
    parent = np.zeros(env.action_space[1].shape, dtype=np.float32)
    dense_sum = 0.0
    shaping_sum = 0.0
    bonus_hits = 0
    dense_steps: List[float] = []
    ctx: Dict[str, Any] = {}
    for t in range(max_steps):
        proto = env.current_protocol
        state = getattr(proto, "state", None) if proto else None
        action = infant_policy(env, state, t, ctx)
        action = normalize_infant_action(env, action)
        _obs, reward, done, _tr, info = env.step([action, parent])
        parent = np.asarray(info.get("parent_action", parent), dtype=np.float32)
        dense = float(info.get("dense_reward", 0.0) or 0.0)
        dense_steps.append(dense)
        dense_sum += dense
        # 成功ボーナスは定数と比較
        if abs(dense - JA_BONUS) < 1e-6 or abs(dense - DENSE_GRASP_BONUS) < 1e-6 or abs(dense - DYADIC_BONUS) < 1e-6:
            bonus_hits += 1
        else:
            shaping_sum += dense
        if done or (proto is not None and proto.is_finished):
            break
    proto = env.current_protocol
    return {
        "session_success": bool(getattr(proto, "session_success", False)),
        "state": getattr(proto, "state", None),
        "dense_sum": dense_sum,
        "shaping_sum": shaping_sum,
        "bonus_hits": bonus_hits,
        "reward0_last": float(reward[0]) if reward is not None else 0.0,
        "dense_steps": dense_steps,
        "steps": t,
    }


class TestRewardGate(unittest.TestCase):
    def test_ja_looking_dense_gate(self):
        for correct in (True, False):
            env = make_protocol_env(dense_reward=True)
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
                        "close_on_success": True,
                    },
                )

                def policy(e, state, t, ctx, _oid=oid, _correct=correct):
                    a = np.zeros(e.action_space[0].shape, dtype=np.float32)
                    if _correct and state in ("looking", "pointing", "approach"):
                        infant_gaze_action(e, _oid, a)
                    return a

                result = _accumulate_dense(env, policy, max_steps=250)
                if correct:
                    self.assertTrue(result["session_success"], msg=str(result))
                    self.assertGreater(result["shaping_sum"], 0.0, msg=str(result))
                    self.assertEqual(result["bonus_hits"], 1, msg=str(result))
                else:
                    self.assertFalse(result["session_success"], msg=str(result))
            finally:
                env.close()

    def test_giving_dense_gate(self):
        for correct in (True, False):
            env = make_protocol_env(dense_reward=True)
            try:
                env.reset(seed=1)
                oid = first_active_oid(env, "RedCube")
                place_object_in_front(env, oid, distance=1.5)
                env.set_fixed_protocol(
                    "object_giving",
                    {
                        "fixed_target": oid,
                        "infant_initiative": False,
                        "close_on_grasp": True,
                    },
                )
                if env.current_protocol is not None:
                    env.current_protocol.parent_utterance = False

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
                    elif state in ("fetching", "approach", "placing", "showing"):
                        infant_gaze_action(e, _oid, a)
                    return a

                result = _accumulate_dense(env, policy, max_steps=450)
                if correct:
                    self.assertTrue(result["session_success"], msg=str(result))
                    self.assertGreater(result["shaping_sum"], 0.0, msg=str(result))
                    self.assertEqual(result["bonus_hits"], 1, msg=str(result))
                else:
                    self.assertFalse(result["session_success"], msg=str(result))
            finally:
                env.close()

    def test_dyadic_raise_dense_gate(self):
        for correct in (True, False):
            env = make_protocol_env(dense_reward=True)
            try:
                env.reset(seed=2)
                env.set_fixed_protocol(
                    "dyadic",
                    {
                        "infant_initiative": False,
                        "parent_channels": ["arm_raise"],
                        "close_on_raise": True,
                    },
                )

                def policy(e, state, t, ctx, _correct=correct):
                    a = np.zeros(e.action_space[0].shape, dtype=np.float32)
                    if _correct and state in ("performing_parent_cue", "waiting_babble"):
                        infant_raise_action(e, a)
                    return a

                result = _accumulate_dense(env, policy, max_steps=250)
                if correct:
                    self.assertTrue(result["session_success"], msg=str(result))
                    self.assertGreater(result["shaping_sum"], 0.0, msg=str(result))
                    self.assertEqual(result["bonus_hits"], 1, msg=str(result))
                else:
                    self.assertFalse(result["session_success"], msg=str(result))
            finally:
                env.close()

    def test_voice_dense_gate(self):
        """見本語を言うと判定 step の報酬が正。沈黙はほぼ 0。"""
        env = make_protocol_env(dense_reward=True)
        try:
            env.reset(seed=3)
            env.set_fixed_protocol(
                "voice_echo",
                {"use_geta": False, "target_concept": "red_cube", "max_rounds": 3},
            )
            queue = deque()

            def policy(e, state, t, ctx, _queue=queue):
                a = np.zeros(e.action_space[0].shape, dtype=np.float32)
                if state == "waiting":
                    if not _queue:
                        _queue.extend(
                            build_word_phoneme_queue(
                                "red_cube", duration=12, silence_frames=20
                            )
                        )
                    apply_queued_phoneme(e, a, _queue)
                return a

            result = _accumulate_dense(env, policy, max_steps=400)
            positive = [d for d in result["dense_steps"] if d > 0.0]
            self.assertTrue(
                len(positive) > 0 and max(positive) >= 1.0 - 1e-6,
                msg=f"voice/correct: dense_sum={result['dense_sum']} positive={positive[:5]}",
            )
            speak_sum = result["dense_sum"]
        finally:
            env.close()

        env = make_protocol_env(dense_reward=True)
        try:
            env.reset(seed=3)
            env.set_fixed_protocol(
                "voice_echo",
                {"use_geta": False, "target_concept": "red_cube", "max_rounds": 3},
            )

            def silent(e, state, t, ctx):
                return np.zeros(e.action_space[0].shape, dtype=np.float32)

            result = _accumulate_dense(env, silent, max_steps=400)
            self.assertLess(result["dense_sum"], speak_sum, msg=str(result))
            self.assertLess(result["dense_sum"], 1.0, msg=str(result))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
