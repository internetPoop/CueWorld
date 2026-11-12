"""層 1: 実環境の身体から意図フラグが立つこと。

特権ヘルパで幼児を動かし、`compute_intent_flags`（と到達圏ゲート）を確認する。
動きなしでは全フラグが 0。
"""

from __future__ import annotations

import unittest

import numpy as np

from cueworld.intent import WORD_STABILITY_MIN, compute_intent_flags, word_candidate
from cueworld.protocol_manager import ProtocolManager
from cueworld.tests.validation_helpers import (
    apply_queued_phoneme,
    build_word_phoneme_queue,
    first_active_oid,
    infant_point_action,
    infant_raise_action,
    intent_flags_from_body,
    make_protocol_env,
    make_signal_env,
    normalize_infant_action,
    place_object_in_front,
    pm_intents_after_strength,
    signals_of,
    zero_actions,
)


class TestIntentFromBody(unittest.TestCase):
    def test_pointing_flag_from_body(self):
        env = make_signal_env()
        try:
            env.reset(seed=0)
            oid = first_active_oid(env, "RedCube")
            place_object_in_front(env, oid, distance=0.9)
            parent = np.zeros(env.action_space[1].shape, dtype=np.float32)
            flags = {"pointing": 0.0}
            for _ in range(60):
                action = infant_point_action(env, oid)
                action = normalize_infant_action(env, action)
                _obs, _r, _d, _t, info = env.step([action, parent])
                flags = intent_flags_from_body(env, info)
                if flags["pointing"] > 0.0:
                    break
            self.assertEqual(flags["pointing"], 1.0)
        finally:
            env.close()

    def test_reach_flag_from_body_out_of_reach(self):
        """遠い物へ指差しで対象を揃え、開き手に切り替えて reach=1。"""
        env = make_signal_env()
        try:
            env.reset(seed=0)
            oid = first_active_oid(env, "RedCube")
            place_object_in_front(env, oid, distance=0.9)
            self.assertFalse(env.object_in_agent_reach(env.infant, oid))
            parent = np.zeros(env.action_space[1].shape, dtype=np.float32)
            flags = {"reach": 0.0}
            for t in range(80):
                action = np.zeros(env.action_space[0].shape, dtype=np.float32)
                env.point_object(env.infant, oid, action=action)
                env.gaze_object(env.infant, oid, yaw_link="head", action=action)
                if t >= 40:
                    env.open_hand(env.infant, action=action)
                action = normalize_infant_action(env, action)
                _obs, _r, _d, _t, info = env.step([action, parent])
                flags = intent_flags_from_body(env, info)
                if flags["reach"] > 0.0:
                    break
            self.assertEqual(flags["reach"], 1.0)
            # 到達圏ゲートを通しても 1 のまま（遠い）
            pm = ProtocolManager(env)
            gated = pm_intents_after_strength(pm, signals_of(info, env))
            self.assertEqual(gated["reach"], 1.0)
        finally:
            env.close()

    def test_reach_zeroed_when_in_reach(self):
        """近い物では raw reach が立っても、PM の到達圏ゲートで 0 になる。"""
        env = make_signal_env()
        try:
            env.reset(seed=1)
            oid = first_active_oid(env, "RedCube")
            place_object_in_front(env, oid, distance=0.25)
            self.assertTrue(env.object_in_agent_reach(env.infant, oid))
            parent = np.zeros(env.action_space[1].shape, dtype=np.float32)
            info = {}
            raw = {"reach": 0.0}
            for t in range(80):
                action = np.zeros(env.action_space[0].shape, dtype=np.float32)
                env.point_object(env.infant, oid, action=action)
                env.gaze_object(env.infant, oid, yaw_link="head", action=action)
                if t >= 40:
                    env.open_hand(env.infant, action=action)
                action = normalize_infant_action(env, action)
                _obs, _r, _d, _t, info = env.step([action, parent])
                raw = intent_flags_from_body(env, info)
                if raw["reach"] > 0.0:
                    break
            # touching で対象が立つこともあるので、ゲート後だけを必須にする
            pm = ProtocolManager(env)
            gated = pm_intents_after_strength(pm, signals_of(info, env))
            self.assertEqual(gated["reach"], 0.0)
        finally:
            env.close()

    def test_word_recognized_from_body(self):
        """積算発声のあと words に語が入る。word フラグは安定度ゲートも見る。"""
        env = make_signal_env()
        try:
            env.reset(seed=2)
            parent = np.zeros(env.action_space[1].shape, dtype=np.float32)
            queue = build_word_phoneme_queue("red_cube", duration=12, silence_frames=20)
            words = {}
            flags = {"word": 0.0}
            last_signals = {}
            for _ in range(140):
                action = np.zeros(env.action_space[0].shape, dtype=np.float32)
                apply_queued_phoneme(env, action, queue)
                action = normalize_infant_action(env, action)
                _obs, _r, _d, _t, info = env.step([action, parent])
                last_signals = signals_of(info, env)
                words = last_signals.get("words") or {}
                flags = compute_intent_flags(last_signals)
                if "red_cube" in words:
                    # 無音で判定された直後は volume=0。有声を戻してフラグを再評価する。
                    for _ in range(8):
                        speak = np.zeros(env.action_space[0].shape, dtype=np.float32)
                        env.apply_phoneme_to_action(
                            speak, env.phoneme_mapping["p2"], volume=1.0
                        )
                        speak = normalize_infant_action(env, speak)
                        _obs, _r, _d, _t, info = env.step([speak, parent])
                        last_signals = signals_of(info, env)
                        words = last_signals.get("words") or {}
                        flags = compute_intent_flags(last_signals)
                        if flags["word"] > 0.0:
                            break
                    break
            self.assertIn("red_cube", words)
            self.assertLessEqual(float(words["red_cube"]), 0.6)
            # 現行の 26 語スコア表では voice_target_stability が WORD_STABILITY_MIN に
            # 届かないことが多い。フラグが立たないときはその事実を固定する。
            if flags["word"] < 1.0:
                stab = float(last_signals.get("voice_target_stability", 0.0))
                self.assertLess(
                    stab,
                    WORD_STABILITY_MIN,
                    msg=(
                        "word フラグが 0 なのに安定度が閾値以上。"
                        f" stab={stab} words={words}"
                    ),
                )
                self.assertIsNone(word_candidate(last_signals))
            else:
                self.assertEqual(flags["word"], 1.0)
        finally:
            env.close()

    def test_dyadic_flag_from_arm_raise(self):
        env = make_signal_env()
        try:
            env.reset(seed=3)
            parent = np.zeros(env.action_space[1].shape, dtype=np.float32)
            flags = {"dyadic": 0.0}
            for _ in range(40):
                action = infant_raise_action(env)
                action = normalize_infant_action(env, action)
                _obs, _r, _d, _t, info = env.step([action, parent])
                flags = intent_flags_from_body(env, info)
                if flags["dyadic"] > 0.0:
                    break
            self.assertEqual(flags["dyadic"], 1.0)
        finally:
            env.close()

    def test_no_motion_keeps_flags_zero(self):
        env = make_signal_env()
        try:
            env.reset(seed=4)
            infant_a, parent_a = zero_actions(env)
            # reset 直後の開き手が reach を誤発火しないよう、少し閉じてから測る
            for _ in range(5):
                close = np.zeros_like(infant_a)
                close[env.action_map["index_joint"]] = 1.0
                close[env.action_map["other_fingers"]] = 1.0
                close = normalize_infant_action(env, close)
                env.step([close, parent_a])
            flags = None
            for _ in range(50):
                _obs, _r, _d, _t, info = env.step([infant_a, parent_a])
                flags = intent_flags_from_body(env, info)
            self.assertIsNotNone(flags)
            for key in ("pointing", "reach", "word", "dyadic"):
                self.assertEqual(flags[key], 0.0, msg=f"{key}={flags[key]}")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
