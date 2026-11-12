"""ProtocolManager バッチ §6-D2: 意図レイヤの穴を固定する。"""

import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from cueworld.protocol_manager import ProtocolManager


class _MockObject:
    def __init__(self, oid: int, concept_id: str, is_platform: bool = False):
        self.id = oid
        self.concept_id = concept_id
        self.is_platform = is_platform


class _MockEnv(gym.Env):
    action_num = 16
    eval_mode = False
    active_signals = None
    target_obj_id = None

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = (
            gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32),
            gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32),
        )
        self.step_count = 0
        self.parent = {"body": 1, "held_obj": None}
        self.infant = {"body": 2, "held_obj": None}
        self.all_objects = {
            101: _MockObject(101, "red_cube"),
            102: _MockObject(102, "green_cube"),
        }
        self.active_objects = [101, 102]
        self.trackable_targets = [101, 102, self.parent["body"]]
        self.reach_radius = 0.4
        self._in_reach = set()
        self.voice_volume_threshold = 0.25

    def object_in_agent_reach(self, agent, oid):
        return oid in self._in_reach

    def reset(self, **kwargs):
        self.step_count = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        return np.zeros(1, dtype=np.float32), [0.0, 0.0], False, False, {}


class TestProtocolManagerIntent(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.manager = ProtocolManager(
            self.env,
            fixed_protocol="observance",
            fixed_context={},
            decay_rate=0.5,
        )
        self.ja_idx = self.manager.response_protocol.index("joint_attention")
        self.giving_idx = self.manager.response_protocol.index("object_giving")
        self.dyadic_idx = self.manager.response_protocol.index("dyadic")

    def _zeros(self):
        signals = {key: 0.0 for key in self.manager.signal_keys}
        signals["gesture_targets"] = np.zeros(3, dtype=float)
        signals["gaze_targets"] = np.zeros(3, dtype=float)
        signals["words"] = {}
        signals["looking_at_parent_eye"] = 0.0
        signals["touching"] = 0.0
        signals["voice_best_distance"] = 10.0
        return signals

    def _pointing_at(self, target_idx: int = 0):
        s = self._zeros()
        s["point_intensity"] = 1.0
        s["hand_target_stability"] = 0.8
        s["gesture_targets"][target_idx] = 1.0
        return s

    def _reach_at(self, target_idx: int = 0):
        s = self._zeros()
        s["open_intensity"] = 1.0
        s["hand_target_stability"] = 0.8
        s["gesture_targets"][target_idx] = 1.0
        s["holding"] = 0.0
        return s

    def test_gaze_alone_does_not_raise_ja(self):
        base = self.manager._calculate_signal_strength(self._zeros())
        gazed = self._zeros()
        gazed["gaze_target_stability"] = 1.0
        gazed["gaze_targets"][0] = 1.0
        score = self.manager._calculate_signal_strength(gazed)
        self.assertAlmostEqual(float(score[self.ja_idx]), float(base[self.ja_idx]), places=5)

    def test_arm_ext_alone_does_not_raise_giving(self):
        base = self.manager._calculate_signal_strength(self._zeros())
        armed = self._zeros()
        armed["arm_ext"] = 1.0
        score = self.manager._calculate_signal_strength(armed)
        self.assertAlmostEqual(float(score[self.giving_idx]), float(base[self.giving_idx]), places=5)

    def test_complete_pointing_raises_ja(self):
        base = self.manager._calculate_signal_strength(self._zeros())[self.ja_idx]
        score = self.manager._calculate_signal_strength(self._pointing_at())[self.ja_idx]
        # pointing 2.0 + weak_arm 0.3（完成形でも弱い腕は常時足す）
        self.assertAlmostEqual(float(score - base), 2.3, places=5)
        self.assertEqual(self.manager._last_intent_meta["hand_target_id"], 101)

    @patch("cueworld.protocol_manager.np.random.choice")
    def test_complete_pointing_switches_with_infant_initiative(self, mock_choice):
        mock_choice.return_value = "joint_attention"
        self.manager.current_protocol.persistence = 0.0
        should_switch, name, context = self.manager.check_for_response(self._pointing_at())
        self.assertTrue(should_switch)
        self.assertEqual(name, "joint_attention")
        self.assertTrue(context["infant_initiative"])
        self.assertEqual(context.get("intent_target_id"), 101)

    def test_parent_cues_object_and_dyadic(self):
        from cueworld.intent import parent_object_cue, parent_dyadic_cue

        looking = self._zeros()
        looking["gaze_target_stability"] = 0.7
        self.assertEqual(parent_object_cue(looking), 1.0)
        self.assertEqual(parent_dyadic_cue(looking), 0.0)

        touching = self._zeros()
        touching["touching"] = 1.0
        self.assertEqual(parent_object_cue(touching), 1.0)

        holding = self._zeros()
        holding["holding"] = 1.0
        self.assertEqual(parent_object_cue(holding), 1.0)

        eye = self._zeros()
        eye["looking_at_parent_eye"] = 0.7
        self.assertEqual(parent_dyadic_cue(eye), 1.0)
        self.assertEqual(parent_object_cue(eye), 0.0)

    @patch("cueworld.protocol_manager.np.random.choice")
    def test_parent_cue_initiative_flag_is_false(self, mock_choice):
        mock_choice.return_value = "joint_attention"
        self.manager.current_try_count = self.manager.base_try_count
        self.manager.level_try_counts = [self.manager.base_try_count] * len(
            self.manager.level_try_counts
        )
        self.manager.level_scores = [0.5] * len(self.manager.level_scores)
        signals = self._zeros()
        signals["gaze_target_stability"] = 1.0
        signals["gaze_targets"][0] = 1.0
        should_switch, name, context = self.manager.check_for_initiative(signals)
        self.assertTrue(should_switch)
        self.assertEqual(name, "joint_attention")
        self.assertFalse(context["infant_initiative"])
        self.assertFalse(self.manager.current_infant_initiative)

    def test_reachable_reach_zeros_giving(self):
        self.env._in_reach.add(101)
        score = self.manager._calculate_signal_strength(self._reach_at())[self.giving_idx]
        self.assertAlmostEqual(float(score), 0.0, places=5)

    def test_parent_holding_reachable_object_raises_giving(self):
        """親が持っている物は、床 XY が届く範囲でもくれに乗る（バッチ F）。"""
        self.env._in_reach.add(101)
        self.env.parent["held_obj"] = 101
        score = self.manager._calculate_signal_strength(self._reach_at())[self.giving_idx]
        self.assertAlmostEqual(float(score), 2.3, places=5)

    def test_floor_reachable_without_parent_hold_zeros_giving(self):
        """親が持っておらず床に置いて届く物は、今どおりくれにしない。"""
        self.env._in_reach.add(101)
        self.env.parent["held_obj"] = None
        score = self.manager._calculate_signal_strength(self._reach_at())[self.giving_idx]
        self.assertAlmostEqual(float(score), 0.0, places=5)

    def test_out_of_reach_reach_raises_giving(self):
        score = self.manager._calculate_signal_strength(self._reach_at())[self.giving_idx]
        # reach 2.0 + weak_arm 0.3
        self.assertAlmostEqual(float(score), 2.3, places=5)

    def test_pointing_plus_word_does_not_add_word_to_giving(self):
        signals = self._pointing_at()
        signals["voice_target_stability"] = 1.0
        signals["voice_volume"] = 1.0
        signals["voice_best_distance"] = 0.1
        signals["words"] = {"red_cube": 0.1, "green_cube": 2.0}
        score = self.manager._calculate_signal_strength(signals)
        # 単語はくれに足さない。弱い腕の薄い寄与 0.3 だけ残る。
        self.assertAlmostEqual(float(score[self.giving_idx]), 0.3, places=5)
        self.assertEqual(self.manager._last_intent_meta["intents"]["word"], 0.0)
        self.assertGreater(float(score[self.ja_idx]), 0.0)

    def test_dyadic_survives_when_ja_giving_zeroed_by_penalty(self):
        signals = self._zeros()
        signals["hand_height_margin"] = 1.0
        signals["babbling_ratio"] = 0.0
        signals["voice_volume"] = 0.0
        signals["voice_target_stability"] = 1.0

        strengths = self.manager._calculate_signal_strength(signals)
        self.assertGreater(float(strengths[self.dyadic_idx]), 0.0)

        # ペナルティで JA/giving が既に 0 のとき、二項ゲートは dyadic を落とさない
        self.manager.protocol_penalties["joint_attention"] = 10.0
        self.manager.protocol_penalties["object_giving"] = 10.0
        with patch("cueworld.protocol_manager.np.random.choice", return_value="dyadic"):
            self.manager.current_protocol.persistence = 0.0
            should_switch, name, context = self.manager.check_for_response(signals)
        self.assertTrue(should_switch)
        self.assertEqual(name, "dyadic")
        self.assertTrue(context["infant_initiative"])

    def test_weight_matrix_is_intent_shaped(self):
        self.assertEqual(
            self.manager.weights_for_strength_calc.shape,
            (len(self.manager.response_protocol), len(self.manager.intent_keys)),
        )

    def _set_cold(self):
        self.manager.current_try_count = 0
        self.manager.level_try_counts = [0] * len(self.manager.level_try_counts)
        self.manager.level_scores = [0.0] * len(self.manager.level_scores)
        self.manager.infant_initiative_rate = 0.0

    def _set_warm(self):
        n = self.manager.base_try_count
        self.manager.current_try_count = n
        self.manager.level_try_counts = [n] * len(self.manager.level_try_counts)
        self.manager.level_scores = [0.5] * len(self.manager.level_scores)
        self.manager.infant_initiative_rate = 0.0

    def _parent_start_p(self, signals):
        choices, probs = self.manager._initiative_probs(signals)
        self.assertTrue(choices)
        self.assertIsNotNone(probs)
        return 1.0 - float(probs[0])

    def test_two_stage_zeros_warm_matches_cold(self):
        zeros = self._zeros()
        self._set_cold()
        p_cold = self._parent_start_p(zeros)
        self._set_warm()
        p_warm = self._parent_start_p(zeros)
        self.assertAlmostEqual(p_cold, p_warm, places=5)
        self.assertAlmostEqual(p_cold, 0.0056, delta=0.001)

    def test_two_stage_gaze_is_about_six_times_zeros_when_warm(self):
        gazed = self._zeros()
        gazed["gaze_target_stability"] = 1.0
        gazed["gaze_targets"][0] = 1.0
        self._set_warm()
        p_zeros = self._parent_start_p(self._zeros())
        p_gaze = self._parent_start_p(gazed)
        self.assertAlmostEqual(p_gaze, 0.033, places=2)
        self.assertGreater(p_gaze / p_zeros, 5.0)
        self.assertLess(p_gaze / p_zeros, 7.0)

    def test_two_stage_cold_gaze_matches_zeros(self):
        gazed = self._zeros()
        gazed["gaze_target_stability"] = 1.0
        gazed["gaze_targets"][0] = 1.0
        self._set_cold()
        p_zeros = self._parent_start_p(self._zeros())
        p_gaze = self._parent_start_p(gazed)
        self.assertAlmostEqual(p_zeros, p_gaze, places=5)

    def test_two_stage_parent_eye_raises_even_when_cold(self):
        eye = self._zeros()
        eye["looking_at_parent_eye"] = 0.7
        self._set_cold()
        p_zeros = self._parent_start_p(self._zeros())
        p_eye = self._parent_start_p(eye)
        self.assertAlmostEqual(p_eye, 0.033, places=2)
        self.assertGreater(p_eye, p_zeros * 5.0)

    def test_two_stage_candidate_count_does_not_change_start_p(self):
        self._set_warm()
        zeros = self._zeros()
        gazed = self._zeros()
        gazed["gaze_target_stability"] = 1.0
        gazed["gaze_targets"][0] = 1.0
        full = list(self.manager.request_protocol)
        p_zeros_full = self._parent_start_p(zeros)
        p_gaze_full = self._parent_start_p(gazed)
        self.manager.request_protocol = ["joint_attention"]
        p_zeros_one = self._parent_start_p(zeros)
        p_gaze_one = self._parent_start_p(gazed)
        self.manager.request_protocol = full
        self.assertAlmostEqual(p_zeros_full, p_zeros_one, places=5)
        self.assertAlmostEqual(p_gaze_full, p_gaze_one, places=5)


if __name__ == "__main__":
    unittest.main()
