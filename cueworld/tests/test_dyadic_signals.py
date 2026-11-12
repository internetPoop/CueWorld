"""模倣用シグナル（周期性・顕著性）のユニットテスト。"""

import math
import unittest
from collections import deque

import numpy as np

from cueworld.infant_signal import InfantSignalDetector, HAND_HEIGHT_TOL, HAND_HEIGHT_WINDOW
from cueworld.protocols.dyadic import (
    ARM_RAISE_HIT_RATE_MIN,
    DyadicProtocol,
    VOICE_CHANNEL,
)


class _MinimalEnv:
    easy_voice = True
    step_count = 0
    action_repeat = 1
    client_id = 0
    infant_voice_window = deque()
    voice_mapping = {"volume": 0, "pitch": 1, "v_1": 2, "v_2": 3, "v_3": 4}
    phoneme_index = 2

    infant = {
        "index_joint": 0,
        "thumb_joint": 1,
        "head_tilt_joint": 2,
        "body": 0,
    }

    def get_joint_angle(self, agent, joint_key):
        return self._angles[joint_key]

    def _get_hand_shape(self, agent):
        return self._hand_pose, 0.0, 0.0


class TestImitationSignals(unittest.TestCase):
    def setUp(self):
        self.env = _MinimalEnv()
        self.env._angles = {
            "index_joint": 0.0,
            "thumb_joint": 0.0,
            "head_tilt_joint": 0.0,
        }
        self.env._hand_pose = "closed"
        self.detector = InfantSignalDetector(self.env, history_length=30)
        self.detector.phoneme_history = deque(["p0"] * 10, maxlen=15)
        self.detector.voice_volume_history = deque([0.8] * 5, maxlen=100)

    def _append_step(self, index_angle, thumb_angle, pose):
        self.env._angles["index_joint"] = index_angle
        self.env._angles["thumb_joint"] = thumb_angle
        self.env._hand_pose = pose
        self.detector.hand_pose_labels.append(pose)
        self.detector.index_joint_history.append(index_angle)
        self.detector.thumb_joint_history.append(thumb_angle)

    def test_hand_open_close_periodicity_detects_rhythmic_open_close(self):
        poses = ["closed", "open", "closed", "open", "closed", "open", "closed", "open"]
        angles = [0.2, 1.5, 0.2, 1.5, 0.2, 1.5, 0.2, 1.5]
        for pose, angle in zip(poses, angles):
            self._append_step(angle, 0.0, pose)

        score = self.detector.compute_hand_open_close_periodicity()
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_hand_open_close_periodicity_same_for_phase_shift(self):
        poses_a = ["closed", "open", "closed", "open", "closed", "open", "closed", "open"]
        angles_a = [0.2, 1.5, 0.2, 1.5, 0.2, 1.5, 0.2, 1.5]
        for pose, angle in zip(poses_a, angles_a):
            self._append_step(angle, 0.0, pose)
        score_a = self.detector.compute_hand_open_close_periodicity()

        self.detector.hand_pose_labels.clear()
        self.detector.index_joint_history.clear()
        poses_b = ["open", "closed", "open", "closed", "open", "closed", "open", "closed"]
        angles_b = [1.5, 0.2, 1.5, 0.2, 1.5, 0.2, 1.5, 0.2]
        for pose, angle in zip(poses_b, angles_b):
            self._append_step(angle, 0.0, pose)
        score_b = self.detector.compute_hand_open_close_periodicity()

        self.assertAlmostEqual(score_a, score_b, places=5)

    def test_hand_open_close_periodicity_zero_when_static_open(self):
        for _ in range(8):
            self._append_step(1.5, -0.2, "open")

        score = self.detector.compute_hand_open_close_periodicity()
        self.assertEqual(score, 0.0)

    def test_get_salient_channels_independent_gates(self):
        signals = {
            "hand_open_close_periodicity": 0.8,
            "hand_height_margin": ARM_RAISE_HIT_RATE_MIN,
            "babbling_salience": 0.0,
        }

        ranked = DyadicProtocol.get_salient_channels(signals)
        names = [name for name, _ in ranked]
        self.assertEqual(names, ["arm_raise", "hand_open_close"])
        for name, salience in ranked:
            self.assertGreaterEqual(salience, DyadicProtocol.channel_tau(name))
            self.assertIn(name, [VOICE_CHANNEL, "arm_raise", "hand_open_close"])

    def test_get_salient_channels_open_close_zero_when_raise_off(self):
        signals = {
            "hand_open_close_periodicity": 0.8,
            "hand_height_margin": 0.7,
            "babbling_salience": 0.0,
        }
        salience = DyadicProtocol.compute_channel_salience(signals)
        self.assertEqual(salience["hand_open_close"], 0.0)
        self.assertAlmostEqual(salience["arm_raise"], 0.7)
        ranked = DyadicProtocol.get_salient_channels(signals)
        names = [name for name, _ in ranked]
        self.assertNotIn("arm_raise", names)
        self.assertNotIn("hand_open_close", names)
        self.assertNotIn("arm_wave", names)
        self.assertNotIn("head_nod", names)

    def test_get_salient_channels_includes_arm_raise_when_hit_rate_on(self):
        signals = {
            "hand_open_close_periodicity": 0.8,
            "hand_height_margin": ARM_RAISE_HIT_RATE_MIN,
            "babbling_salience": 0.0,
        }
        ranked = DyadicProtocol.get_salient_channels(signals)
        names = [name for name, _ in ranked]
        self.assertIn("arm_raise", names)
        self.assertIn("hand_open_close", names)

    def test_compute_hand_height_margin_hit_rate(self):
        self.detector.infant_eye_height = 1.0
        band_lo = 1.0 - HAND_HEIGHT_TOL  # z > 0.6 が帯内
        in_band = band_lo + 0.01
        out_band = band_lo - 0.1

        for z in [in_band] * (HAND_HEIGHT_WINDOW - 1):
            self.detector.hand_world_z.append(z)
        self.assertEqual(self.detector.compute_hand_height_margin(), 0.0)

        self.detector.hand_world_z.clear()
        for z in [in_band] * 8 + [out_band] * 2:
            self.detector.hand_world_z.append(z)
        self.assertAlmostEqual(self.detector.compute_hand_height_margin(), 0.8, places=5)

        self.detector.hand_world_z.clear()
        for z in [in_band] * HAND_HEIGHT_WINDOW:
            self.detector.hand_world_z.append(z)
        self.assertEqual(self.detector.compute_hand_height_margin(), 1.0)

        self.detector.hand_world_z.clear()
        for z in [out_band] * HAND_HEIGHT_WINDOW:
            self.detector.hand_world_z.append(z)
        self.assertEqual(self.detector.compute_hand_height_margin(), 0.0)

    def test_get_phoneme_returns_unknown_for_ambiguous_easy_voice(self):
        avg_voice = np.array([0.8, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        phoneme, scores = self.detector.get_phoneme(avg_voice)

        self.assertEqual(phoneme, "unknown")
        self.assertIn("p0", scores)
        self.assertIn("p1", scores)
        self.assertLess(abs(scores["p0"] - scores["p1"]), self.detector.phoneme_ambiguity_margin)

    def test_get_word_adopts_cheapest_when_raw_is_under_three(self):
        self.detector.word_dictionary = {
            "word_a": ["p0", "p1"],
            "word_b": ["p0", "p2"],
        }
        self.detector.phoneme_history = deque(["p0", "p1"], maxlen=15)
        phoneme_scores_history = [
            {"p0": 0.05, "p1": 0.10},
            {"p0": 0.05, "p1": 0.09},
        ]

        word, scores = self.detector.get_word(phoneme_scores_history)

        self.assertEqual(word, "word_a")
        self.assertIn("word_a", scores)
        self.assertIn("word_b", scores)
        self.assertLess(
            self.detector.last_word_recognition_metrics["best_raw_score"],
            3.0,
        )

    def test_get_word_accepts_clear_sequence_with_good_margin(self):
        self.detector.word_dictionary = {
            "word_a": ["p0", "p1"],
            "word_b": ["p2", "p3"],
        }
        self.detector.phoneme_history = deque(["p0", "p1"], maxlen=15)
        phoneme_scores_history = [
            {"p0": 0.02, "p1": 0.60},
            {"p0": 0.03, "p1": 0.65},
        ]

        word, _scores = self.detector.get_word(phoneme_scores_history)

        self.assertEqual(word, "word_a")
        self.assertEqual(self.detector.last_word_recognition_metrics["length"], 2.0)

    def test_no_utterance_keeps_distance_at_zero(self):
        signals, words = self.detector.get_voice_signals()
        self.assertEqual(words, {})
        self.assertEqual(signals["voice_best_distance"], 0.0)
        self.assertEqual(signals["voice_target_stability"], 0.0)
        self.assertFalse(signals["voice_unadopted_attempt"])
        self.assertEqual(signals["voice_segment_seq"], 0)

    def test_rejected_word_is_not_published_as_a_word(self):
        from cueworld.intent import word_candidate

        self.detector.voice_recognizer.last_word_recognition_metrics = {"length": 5.0}
        self.detector._commit_word_observation(
            None, {"red_cube": 0.05, "green_cube": 0.2, "blue_cube": 0.4}
        )
        signals, words = self.detector.get_voice_signals()
        self.assertEqual(words, {})
        self.assertTrue(math.isinf(signals["voice_best_distance"]))
        self.assertEqual(signals["voice_target_stability"], 0.0)
        self.assertTrue(signals["voice_unadopted_attempt"])
        self.assertEqual(signals["voice_segment_seq"], 1)
        appeal = dict(signals)
        appeal["words"] = words
        appeal["voice_volume"] = 1.0
        self.assertIsNone(word_candidate(appeal))
        # 不採用だけでは babbling_chunk はまだ立たない（沈黙閉じで立てる）。
        # babbling_salience が 1 のときだけ声チャネルが立つ。
        salience = DyadicProtocol.compute_channel_salience({
            "babbling_salience": 1.0,
        })
        self.assertAlmostEqual(salience[VOICE_CHANNEL], 1.0)
        salience_off = DyadicProtocol.compute_channel_salience({
            "babbling_salience": 0.0,
        })
        self.assertAlmostEqual(salience_off[VOICE_CHANNEL], 0.0)

    def test_short_rejection_does_not_raise_unadopted_attempt(self):
        self.detector.voice_recognizer.last_word_recognition_metrics = {"length": 2.0}
        self.detector._commit_word_observation(None, {"red_cube": 0.05})
        signals, words = self.detector.get_voice_signals()
        self.assertEqual(words, {})
        self.assertFalse(signals["voice_unadopted_attempt"])
        self.assertTrue(math.isinf(signals["voice_best_distance"]))
        self.assertEqual(signals["voice_target_stability"], 0.0)

    def test_adopted_word_stays_available_for_appeal(self):
        from cueworld.intent import word_candidate

        self.detector.voice_recognizer.last_word_recognition_metrics = {"length": 10.0}
        self.detector._commit_word_observation(
            "red_cube", {"red_cube": 0.1, "green_cube": 8.0}
        )
        signals, words = self.detector.get_voice_signals()
        self.assertEqual(min(words, key=words.get), "red_cube")
        self.assertAlmostEqual(signals["voice_best_distance"], 0.1)
        self.assertFalse(signals["voice_unadopted_attempt"])
        appeal = dict(signals)
        appeal["words"] = words
        appeal["voice_volume"] = 1.0
        self.assertEqual(word_candidate(appeal), "red_cube")


if __name__ == "__main__":
    unittest.main()
