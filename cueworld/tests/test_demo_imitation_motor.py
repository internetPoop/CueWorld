"""demo_imitation_motor.py のヘルパーとシナリオ設定のユニットテスト。"""

import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

_DEMO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "demo", "demo_imitation_motor.py")
)
_spec = importlib.util.spec_from_file_location("demo_imitation_motor", _DEMO_PATH)
_demo = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["demo_imitation_motor"] = _demo
_spec.loader.exec_module(_demo)


class _MockEnv:
    action_repeat = 1
    action_map = {
        "head_pan_joint": 0,
        "head_tilt_joint": 1,
        "shoulder_pitch_joint": 2,
        "volume": 3,
        "v_1": 4,
        "v_2": 5,
        "v_3": 6,
    }
    infant_joint_mappings = {
        "head_pan_joint": 0.1,
        "head_tilt_joint": 0.1,
        "shoulder_pitch_joint": 0.1,
    }
    phoneme_mapping = {"p0": np.array([-1.0, -1.0, -1.0], dtype=np.float32)}

    def __init__(self):
        self.infant = {"body": 1}
        self.parent = {"body": 2}
        self.action_space = (MagicMock(shape=(9,)), MagicMock(shape=(9,)))
        self.motor_calls = []

    def gaze_agent(self, agent_from, agent_to, yaw_link="head", action=None):
        return 0.0, 0.0

    def wave_arm(self, agent, action=None, phase=0.0):
        self.motor_calls.append(("wave_arm", phase))

    def nod_head(self, agent, action=None, phase=0.0):
        self.motor_calls.append(("nod_head", phase))

    def animate_hand_open_close(self, agent, action=None, phase=0.0):
        self.motor_calls.append(("animate_hand_open_close", phase))


class TestDemoImitationMotorHelpers(unittest.TestCase):
    def test_motor_phase_wraps(self):
        self.assertAlmostEqual(_demo.motor_phase(0), 0.0)
        self.assertAlmostEqual(_demo.motor_phase(15), 0.5)
        self.assertAlmostEqual(_demo.motor_phase(30), 0.0)

    def test_scenario_context_infant_initiated(self):
        ctx = _demo.scenario_context("arm_wave")
        self.assertTrue(ctx["infant_initiative"])

    def test_scenario_context_parent_cue(self):
        ctx = _demo.scenario_context("parent_cue")
        self.assertFalse(ctx["infant_initiative"])
        self.assertEqual(ctx["parent_channels"], ["voice", "arm_raise"])

    def test_build_infant_action_hand_open_close_after_warmup(self):
        env = _MockEnv()
        action = _demo.build_infant_action(env, _demo.WARMUP_STEPS, "hand_open_close")
        self.assertEqual(env.motor_calls[-1][0], "animate_hand_open_close")
        self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))

    def test_build_infant_action_arm_wave(self):
        env = _MockEnv()
        _demo.build_infant_action(env, _demo.WARMUP_STEPS + 5, "arm_wave")
        self.assertEqual(env.motor_calls[-1][0], "wave_arm")

    def test_build_infant_action_voice_arm_wave_sets_volume(self):
        env = _MockEnv()
        action = _demo.build_infant_action(env, _demo.WARMUP_STEPS + 1, "voice_arm_wave")
        self.assertEqual(env.motor_calls[-1][0], "wave_arm")
        self.assertEqual(action[env.action_map["volume"]], 1.0)

    def test_build_infant_action_parent_cue_waits_for_response_state(self):
        env = _MockEnv()
        env.current_protocol = MagicMock(state="performing_parent_cue")
        _demo.build_infant_action(env, _demo.WARMUP_STEPS + 10, "parent_cue")
        self.assertEqual(env.motor_calls, [])

        env.current_protocol.state = "waiting_babble"
        _demo.build_infant_action(env, _demo.WARMUP_STEPS + 10, "parent_cue")
        self.assertEqual(env.motor_calls[-1][0], "wave_arm")


if __name__ == "__main__":
    unittest.main()
