"""円錐レイ確度・意図質量しきい値のユニットテスト。"""

import unittest
from collections import deque
from unittest.mock import patch

import numpy as np

from cueworld.infant_signal import (
    CONE_MIN_ACCURACY,
    CONE_RING_SPECS,
    InfantSignalDetector,
    cone_ray_accuracy,
)
from cueworld.intent import (
    HAND_TARGET_MASS_MIN,
    hand_target_is_clear,
    pointing_is_complete,
)


class _ConeMockEnv:
    easy_voice = True
    step_count = 0
    action_repeat = 1
    client_id = 0
    infant_voice_window = deque()
    voice_mapping = {"volume": 0, "pitch": 1}
    planeId = 0
    trackable_targets = [101, 102, 999]
    all_objects = {101: object(), 102: object()}
    parent = {
        "body": 999,
        "held_obj": None,
        "face_links": (10, 11, 12, 13),
        "hold_gaze_links": (20, 21, 22),
        "eyes_link": 10,
    }
    infant = {
        "body": 888,
        "held_obj": None,
        "face_links": (30, 31, 32, 33),
        "hold_gaze_links": (40, 41, 42),
        "eyes_link": 30,
    }


def _num_cone_rays() -> int:
    return 1 + sum(r["num_rays"] for r in CONE_RING_SPECS)


def _ray_results_for_hits(hit_by_index):
    """hit_by_index[i] = (body_id, link_id) or None."""
    out = []
    for i in range(_num_cone_rays()):
        hit = hit_by_index.get(i)
        if hit is None:
            out.append((-1, -1, -1.0, [0, 0, 0], [0, 0, 0]))
        else:
            out.append((hit[0], hit[1], 0.5, [0, 0, 0], [0, 0, 0]))
    return out


class TestConeRayAccuracy(unittest.TestCase):
    def test_center_is_one(self):
        self.assertAlmostEqual(cone_ray_accuracy(0.0, 8.0), 1.0)

    def test_three_degree_ring(self):
        self.assertAlmostEqual(cone_ray_accuracy(3.0, 8.0), 0.625)

    def test_eight_degree_ring_at_floor(self):
        self.assertAlmostEqual(cone_ray_accuracy(8.0, 8.0), CONE_MIN_ACCURACY)

    def test_custom_min_accuracy(self):
        self.assertAlmostEqual(cone_ray_accuracy(8.0, 8.0, min_accuracy=0.1), 0.1)

    def test_classify_discards_zero_accuracy(self):
        env = _ConeMockEnv()
        detector = InfantSignalDetector(env)
        self.assertIsNone(detector._classify_gaze_hit(999, 10, 0.0))

    def test_classify_parent_face_at_min_accuracy(self):
        env = _ConeMockEnv()
        detector = InfantSignalDetector(env)
        out = detector._classify_gaze_hit(999, 10, CONE_MIN_ACCURACY)
        self.assertEqual(out, (999, CONE_MIN_ACCURACY))


class TestHandTargetMassThreshold(unittest.TestCase):
    def _signals(self, mass: float):
        return {
            "hand_target_stability": 1.0,
            "gesture_targets": np.array([mass, 0.0, 0.0], dtype=float),
            "point_intensity": 1.0,
            "holding": 0.0,
        }

    def test_mass_at_threshold_passes(self):
        self.assertTrue(hand_target_is_clear(self._signals(HAND_TARGET_MASS_MIN)))

    def test_mass_below_threshold_fails(self):
        self.assertFalse(hand_target_is_clear(self._signals(0.15)))

    def test_custom_mass_min(self):
        s = self._signals(0.3)
        self.assertFalse(hand_target_is_clear(s, mass_min=0.35))
        self.assertTrue(hand_target_is_clear(s, mass_min=0.25))

    def test_pointing_complete_with_eight_degree_mass(self):
        s = self._signals(CONE_MIN_ACCURACY)
        self.assertTrue(pointing_is_complete(s))


class TestConeRaycastMocked(unittest.TestCase):
    def setUp(self):
        self.env = _ConeMockEnv()
        self.detector = InfantSignalDetector(self.env)
        self.ray_from = np.array([0.0, 0.0, 1.0], dtype=float)
        self.forward = np.array([1.0, 0.0, 0.0], dtype=float)

    @patch("cueworld.infant_signal.p.rayTestBatch")
    def test_center_hit_only(self, mock_batch):
        hits = {0: (101, -1)}
        mock_batch.return_value = _ray_results_for_hits(hits)
        vec = self.detector._get_targets_by_cone_raycast(
            self.env.infant, self.ray_from, self.forward
        )
        self.assertAlmostEqual(vec[0], 1.0)
        self.assertAlmostEqual(vec[1], 0.0)

    @patch("cueworld.infant_signal.p.rayTestBatch")
    def test_eight_degree_hit_only(self, mock_batch):
        # 8° リングは index 5..12
        hits = {i: (102, -1) for i in range(5, 13)}
        mock_batch.return_value = _ray_results_for_hits(hits)
        vec = self.detector._get_targets_by_cone_raycast(
            self.env.infant, self.ray_from, self.forward
        )
        self.assertAlmostEqual(vec[1], CONE_MIN_ACCURACY)
        self.assertAlmostEqual(vec[0], 0.0)

    @patch("cueworld.infant_signal.p.rayTestBatch")
    def test_center_beats_eight_degree_on_same_object(self, mock_batch):
        hits = {0: (101, -1), 5: (101, -1)}
        mock_batch.return_value = _ray_results_for_hits(hits)
        vec = self.detector._get_targets_by_cone_raycast(
            self.env.infant, self.ray_from, self.forward
        )
        self.assertAlmostEqual(vec[0], 1.0)

    @patch("cueworld.infant_signal.p.rayTestBatch")
    def test_floor_ignored(self, mock_batch):
        hits = {0: (0, -1)}
        mock_batch.return_value = _ray_results_for_hits(hits)
        vec = self.detector._get_targets_by_cone_raycast(
            self.env.infant, self.ray_from, self.forward
        )
        self.assertEqual(float(np.sum(vec)), 0.0)

    @patch("cueworld.infant_signal.p.rayTestBatch")
    def test_parent_face_eight_degree_on_gaze_vector(self, mock_batch):
        hits = {5: (999, 10)}
        mock_batch.return_value = _ray_results_for_hits(hits)
        vec, hit_eye, hit_face = self.detector._get_targets_by_cone_raycast(
            self.env.infant, self.ray_from, self.forward, parent_eye=True
        )
        self.assertTrue(hit_eye)
        self.assertTrue(hit_face)
        self.assertAlmostEqual(vec[2], CONE_MIN_ACCURACY)


if __name__ == "__main__":
    unittest.main()
