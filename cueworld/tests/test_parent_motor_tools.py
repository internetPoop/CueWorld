"""親モータ模倣ツール（wave_arm / nod_head / animate_hand_open_close）のユニットテスト。"""

import math
import unittest

import gymnasium as gym
import numpy as np

from cueworld.parent_tools import ParentTools


class _MockBaseEnv(gym.Env):
    action_map = {
        "head_tilt_joint": 2,
        "shoulder_yaw_joint": 3,
        "shoulder_pitch_joint": 4,
        "elbow_joint": 5,
        "roll_wrist_joint": 6,
        "pitch_wrist_joint": 7,
        "index_joint": 8,
        "other_fingers": 9,
    }
    action_dim = 16

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32)
        self.parent = {"body": 1, "little_finger_joint": 0}
        self._angles = {
            "head_tilt_joint": 0.0,
            "shoulder_yaw_joint": 0.0,
            "shoulder_pitch_joint": 0.0,
            "elbow_joint": 0.0,
            "roll_wrist_joint": 0.0,
            "pitch_wrist_joint": 0.0,
            "index_joint": 0.0,
            "little_finger_joint": 0.0,
        }

    def get_joint_angle(self, agent, joint_key):
        return self._angles[joint_key]

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


class TestParentMotorTools(unittest.TestCase):
    def setUp(self):
        self.tools = ParentTools(_MockBaseEnv())

    def test_wave_arm_sets_shoulder_pitch_delta(self):
        action = np.zeros(self.tools.env.action_dim, dtype=np.float32)
        self.tools.wave_arm(self.tools.env.parent, action=action, phase=0.25)

        pitch_idx = self.tools.env.action_map["shoulder_pitch_joint"]
        expected = (-0.8 + 0.35 * math.sin(0.5 * math.pi)) - 0.0
        self.assertAlmostEqual(float(action[pitch_idx]), expected, places=5)
        self.assertNotEqual(float(action[self.tools.env.action_map["elbow_joint"]]), 0.0)

    def test_nod_head_adds_head_tilt_delta(self):
        action = np.zeros(self.tools.env.action_dim, dtype=np.float32)
        self.tools.nod_head(self.tools.env.parent, action=action, phase=0.0)

        tilt_idx = self.tools.env.action_map["head_tilt_joint"]
        self.assertAlmostEqual(float(action[tilt_idx]), 0.0, places=5)

        action2 = np.zeros(self.tools.env.action_dim, dtype=np.float32)
        self.tools.nod_head(self.tools.env.parent, action=action2, phase=0.25)
        self.assertAlmostEqual(
            float(action2[tilt_idx]),
            0.25 * math.sin(0.5 * math.pi),
            places=5,
        )

    def test_animate_hand_open_close_cycles_fingers(self):
        action_open = np.zeros(self.tools.env.action_dim, dtype=np.float32)
        self.tools.animate_hand_open_close(self.tools.env.parent, action=action_open, phase=0.0)
        index_idx = self.tools.env.action_map["index_joint"]
        self.assertAlmostEqual(float(action_open[index_idx]), -1.2, places=5)

        action_closed = np.zeros(self.tools.env.action_dim, dtype=np.float32)
        self.tools.animate_hand_open_close(self.tools.env.parent, action=action_closed, phase=0.5)
        self.assertAlmostEqual(float(action_closed[index_idx]), 1.2, places=5)

    def test_motor_channels_additive_on_shared_action(self):
        action = np.zeros(self.tools.env.action_dim, dtype=np.float32)
        self.tools.wave_arm(self.tools.env.parent, action=action, phase=0.0)
        self.tools.animate_hand_open_close(self.tools.env.parent, action=action, phase=0.0)

        pitch_idx = self.tools.env.action_map["shoulder_pitch_joint"]
        self.assertNotEqual(float(action[pitch_idx]), 0.0)
        self.assertNotEqual(float(action[self.tools.env.action_map["index_joint"]]), 0.0)

    def test_wave_arm_gesture_overlays_shoulder_pitch(self):
        action = np.zeros(self.tools.env.action_dim, dtype=np.float32)
        pitch_idx = self.tools.env.action_map["shoulder_pitch_joint"]
        # IK 差分が既にある想定
        action[pitch_idx] = -0.5
        self.tools.wave_arm(
            self.tools.env.parent, action=action, phase=0.25, gesture=True
        )
        expected = -0.5 + 0.35 * math.sin(0.5 * math.pi)
        self.assertAlmostEqual(float(action[pitch_idx]), expected, places=5)
        # gesture 時は yaw / elbow を触らない
        self.assertEqual(float(action[self.tools.env.action_map["shoulder_yaw_joint"]]), 0.0)
        self.assertEqual(float(action[self.tools.env.action_map["elbow_joint"]]), 0.0)

    def test_animate_hand_open_close_gesture_only_moves_fingers(self):
        action = np.zeros(self.tools.env.action_dim, dtype=np.float32)
        self.tools.animate_hand_open_close(
            self.tools.env.parent, action=action, phase=0.5, gesture=True
        )
        self.assertAlmostEqual(
            float(action[self.tools.env.action_map["index_joint"]]), 1.2, places=5
        )
        self.assertEqual(
            float(action[self.tools.env.action_map["shoulder_pitch_joint"]]), 0.0
        )


if __name__ == "__main__":
    unittest.main()
