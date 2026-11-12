"""手形 intensity（classify_hand_shape → infant_signal）のユニットテスト。"""

import math
import unittest
from collections import deque

import numpy as np

from cueworld.base import (
    HAND_SHAPE_BOUNDARY,
    INDEX_CLOSED,
    THUMB_CLOSED,
    classify_hand_shape,
)
from cueworld.infant_signal import InfantSignalDetector
from cueworld.intent import HAND_TARGET_MASS_MIN, reach_is_complete


class TestClassifyHandShape(unittest.TestCase):
    def test_ideal_open_is_one(self):
        state, grip, intensity = classify_hand_shape(0.0, 0.0)
        self.assertEqual(state, "open")
        self.assertAlmostEqual(grip, 0.0)
        self.assertAlmostEqual(intensity, 1.0)

    def test_ideal_pointing_is_one(self):
        state, grip, intensity = classify_hand_shape(0.0, THUMB_CLOSED)
        self.assertEqual(state, "pointing")
        self.assertAlmostEqual(grip, 0.0)
        self.assertAlmostEqual(intensity, 1.0)

    def test_ideal_closed_is_one(self):
        state, grip, intensity = classify_hand_shape(INDEX_CLOSED, THUMB_CLOSED)
        self.assertEqual(state, "closed")
        self.assertAlmostEqual(grip, THUMB_CLOSED)
        self.assertAlmostEqual(intensity, 1.0)

    def test_open_near_boundary_still_positive(self):
        eps = 1e-3
        index = HAND_SHAPE_BOUNDARY - eps
        thumb = -HAND_SHAPE_BOUNDARY + eps
        state, _, intensity = classify_hand_shape(index, thumb)
        self.assertEqual(state, "open")
        self.assertGreater(intensity, 0.0)
        self.assertLess(intensity, 0.1)

    def test_other_is_zero_intensity(self):
        state, grip, intensity = classify_hand_shape(
            HAND_SHAPE_BOUNDARY + 0.1,
            -HAND_SHAPE_BOUNDARY + 0.1,
        )
        self.assertEqual(state, "other")
        self.assertAlmostEqual(grip, 0.0)
        self.assertAlmostEqual(intensity, 0.0)


class _HandShapeMockEnv:
    easy_voice = True
    step_count = 0
    action_repeat = 1
    client_id = 0
    infant_voice_window = deque()
    voice_mapping = {"volume": 0, "pitch": 1}
    trackable_targets = [101]

    infant = {"body": 0}

    def _get_hand_shape(self, agent):
        return self._hand_shape


class TestDetectorHandShapeIntensity(unittest.TestCase):
    def setUp(self):
        self.env = _HandShapeMockEnv()
        self.detector = InfantSignalDetector(self.env, history_length=30)

    def _signals_for_pose(self, pose_type, intensity):
        self.env._hand_shape = (pose_type, 0.0, intensity)
        self.detector.hand_poses.append((pose_type, intensity))
        self.detector.grab_flags.append(False)
        self.detector.touch_flags.append(False)
        self.detector.hand_targets.append(np.zeros(1, dtype=float))
        return self.detector.get_gesture_signals()[0]

    def test_closed_does_not_set_open_or_point(self):
        signals = self._signals_for_pose("closed", 0.8)
        self.assertNotIn("fist_intensity", signals)
        self.assertAlmostEqual(signals["open_intensity"], 0.0)
        self.assertAlmostEqual(signals["point_intensity"], 0.0)

    def test_open_intensity_at_reset_pose(self):
        signals = self._signals_for_pose("open", 1.0)
        self.assertAlmostEqual(signals["open_intensity"], 1.0)
        self.assertAlmostEqual(signals["point_intensity"], 0.0)


class TestReachIntentWithOpenIntensity(unittest.TestCase):
    def test_reach_complete_with_ideal_open_hand(self):
        gesture_targets = np.zeros(3, dtype=float)
        gesture_targets[0] = HAND_TARGET_MASS_MIN
        signals = {
            "open_intensity": 1.0,
            "holding": 0.0,
            "hand_target_stability": 0.6,
            "gesture_targets": gesture_targets,
        }
        self.assertTrue(reach_is_complete(signals))


if __name__ == "__main__":
    unittest.main()
