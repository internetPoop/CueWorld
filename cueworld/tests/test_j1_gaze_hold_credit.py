"""J1: 把持中視線クレジット振替・2段拒否のユニットテスト。"""

import unittest
from collections import deque
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from cueworld.infant_signal import InfantSignalDetector
from cueworld.protocols.base import (
    BaseProtocol,
    REJECTION_TURN_AWAY_TAU,
)


class _J1ProtoEnv(gym.Env):
    action_num = 17

    def __init__(self, parent=None):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32)
        self.parent = parent or {"body": 999, "held_obj": 101}

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


class _J1MockEnv:
    easy_voice = True
    step_count = 0
    action_repeat = 1
    client_id = 0
    infant_voice_window = deque()
    voice_mapping = {"volume": 0, "pitch": 1}
    action_map = {
        "torso_joint": 0,
        "head_pan_joint": 1,
        "head_tilt_joint": 2,
    }
    trackable_targets = [101, 102, 999]  # obj_a, obj_b, parent_body
    all_objects = {101: object(), 102: object()}

    parent = {
        "body": 999,
        "held_obj": 101,
        "face_links": (10, 11, 12, 13),
        "hold_gaze_links": (20, 21, 22),
        "eyes_link": 10,
    }
    infant = {
        "body": 888,
        "held_obj": 102,
        "face_links": (30, 31, 32, 33),
        "hold_gaze_links": (40, 41, 42),
        "eyes_link": 30,
    }


class TestJ1GazeCredit(unittest.TestCase):
    def setUp(self):
        self.env = _J1MockEnv()
        self.detector = InfantSignalDetector(self.env, history_length=30)

    def test_classify_parent_hold_link_to_held_obj(self):
        out = self.detector._classify_gaze_hit(999, 20, 0.8)
        self.assertEqual(out, (101, 0.8))

    def test_classify_parent_face_to_parent_body(self):
        out = self.detector._classify_gaze_hit(999, 10, 0.7)
        self.assertEqual(out, (999, 0.7))

    def test_classify_parent_shoulder_discarded(self):
        out = self.detector._classify_gaze_hit(999, 99, 0.9)
        self.assertIsNone(out)

    def test_classify_infant_hold_link_to_held_obj(self):
        out = self.detector._classify_gaze_hit(888, 40, 0.75)
        self.assertEqual(out, (102, 0.75))

    def test_classify_infant_non_hold_discarded(self):
        out = self.detector._classify_gaze_hit(888, 30, 0.6)
        self.assertIsNone(out)

    def test_classify_direct_object_hit(self):
        out = self.detector._classify_gaze_hit(101, -1, 0.95)
        self.assertEqual(out, (101, 0.95))

    def test_head_turn_away_sustained_horizontal(self):
        for _ in range(20):
            self.detector.infant_horizontal_turn_actions.append(0.9)
            self.detector.infant_head_tilt_actions.append(0.0)
        score = self.detector.get_head_turn_away_intensity()
        self.assertGreater(score, REJECTION_TURN_AWAY_TAU * 0.5)

    def test_head_turn_away_oscillation_low(self):
        for i in range(20):
            self.detector.infant_horizontal_turn_actions.append(0.8 if i % 2 == 0 else -0.8)
            self.detector.infant_head_tilt_actions.append(0.0)
        score = self.detector.get_head_turn_away_intensity()
        self.assertLess(score, REJECTION_TURN_AWAY_TAU)


class TestJ1DetectRejection(unittest.TestCase):
    def _proto(self):
        env = _J1ProtoEnv()
        proto = BaseProtocol("Test", env, rejection_check=True)
        proto.target_object = SimpleNamespace(id=101)
        return proto

    def test_gaze_shift_only_not_rejection(self):
        proto = self._proto()
        signals = {
            "avoidance_intensity": 0.0,
            "head_turn_away_intensity": 0.0,
            "gaze_shift_dict": {101: -1.0, 999: -0.5},
            "gaze_late_mass": {101: 0.0, 999: 0.0},
            "negative_emotion_surge": 0.0,
        }
        self.assertFalse(proto.detect_rejection(signals))

    def test_gaze_shift_with_avoidance_is_not_rejection(self):
        proto = self._proto()
        signals = {
            "avoidance_intensity": 1.0,
            "head_turn_away_intensity": 0.0,
            "gaze_shift_dict": {101: -1.0, 999: -0.5},
            "gaze_late_mass": {101: 0.0, 999: 0.0},
            "negative_emotion_surge": 0.0,
        }
        self.assertFalse(proto.detect_rejection(signals))

    def test_gaze_shift_with_head_turn_is_rejection(self):
        proto = self._proto()
        signals = {
            "avoidance_intensity": 0.0,
            "head_turn_away_intensity": REJECTION_TURN_AWAY_TAU + 0.1,
            "gaze_shift_dict": {101: -1.0, 999: -0.5},
            "gaze_late_mass": {101: 0.0, 999: 0.0},
            "negative_emotion_surge": 0.0,
        }
        self.assertTrue(proto.detect_rejection(signals))

    def test_not_engaged_head_turn_is_rejection(self):
        proto = self._proto()
        signals = {
            "avoidance_intensity": 0.0,
            "head_turn_away_intensity": REJECTION_TURN_AWAY_TAU + 0.2,
            "gaze_shift_dict": {101: 0.0, 999: 0.0},
            "gaze_late_mass": {101: 0.0, 999: 0.0},
            "negative_emotion_surge": 0.0,
        }
        self.assertTrue(proto.detect_rejection(signals))


if __name__ == "__main__":
    unittest.main()
