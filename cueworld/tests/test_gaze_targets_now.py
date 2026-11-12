"""視線の瞬間ベクトル gaze_targets_now と 30 step 平均 gaze_targets の公開。"""

import unittest
from collections import deque

import numpy as np

from cueworld.infant_signal import InfantSignalDetector


class _MinimalEnv:
    easy_voice = True
    step_count = 0
    action_repeat = 1
    client_id = 0
    infant_voice_window = deque()
    voice_mapping = {"volume": 0, "pitch": 1, "v_1": 2, "v_2": 3, "v_3": 4}
    phoneme_index = 2
    trackable_targets = [101, 102, 1]
    infant = {"body": 2}
    parent = {"body": 1}


class TestGazeTargetsNow(unittest.TestCase):
    def setUp(self):
        self.env = _MinimalEnv()
        self.detector = InfantSignalDetector(self.env, history_length=30)

    def test_now_is_last_vector_and_gaze_targets_is_mean(self):
        first = np.array([1.0, 0.0, 0.0], dtype=float)
        last = np.array([0.0, 1.0, 0.0], dtype=float)
        self.detector.gaze_targets.append(first)
        self.detector.gaze_targets.append(last)
        signals = self.detector.get_normalized_signals()
        np.testing.assert_array_equal(signals["gaze_targets_now"], last)
        np.testing.assert_allclose(signals["gaze_targets"], (first + last) / 2.0)

    def test_ignored_targets_zero_only_the_average(self):
        first = np.array([1.0, 0.0, 0.0], dtype=float)
        last = np.array([0.0, 1.0, 0.0], dtype=float)
        self.detector.gaze_targets.append(first)
        self.detector.gaze_targets.append(last)
        signals = self.detector.get_normalized_signals(ignored_targets=[102])
        np.testing.assert_array_equal(signals["gaze_targets_now"], last)
        expected_avg = np.array([0.5, 0.0, 0.0], dtype=float)
        np.testing.assert_allclose(signals["gaze_targets"], expected_avg)

    def test_empty_history_omits_now(self):
        signals = self.detector.get_normalized_signals()
        self.assertNotIn("gaze_targets_now", signals)

    def test_zero_vector_is_published(self):
        zeros = np.zeros(3, dtype=float)
        self.detector.gaze_targets.append(zeros)
        signals = self.detector.get_normalized_signals()
        np.testing.assert_array_equal(signals["gaze_targets_now"], zeros)


if __name__ == "__main__":
    unittest.main()
