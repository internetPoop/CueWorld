"""二項の掲げ切り分けテスト（dyadic_raise_isolation.md）のスイッチ確認。"""

import unittest
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np

from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager
from cueworld.protocols.dyadic import (
    ARM_RAISE_HIT_RATE_MIN,
    DENSE_SUCCESS_BONUS,
    DyadicProtocol,
)


class _MockEnv(gym.Env):
    action_num = 17
    action_map = {
        "head_pan_joint": 1,
        "head_tilt_joint": 2,
        "shoulder_pitch_joint": 4,
        "v_1": 14,
        "v_2": 15,
        "v_3": 16,
        "volume": 12,
    }
    phoneme_mapping = {
        "p0": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        "p1": np.array([-1.0, -1.0, 1.0], dtype=np.float32),
    }
    client_id = 0

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32)
        self.parent = {"body": 1}
        self.infant = {"body": 2, "eyes_link": 0, "hand_link": 1}
        self.infant_eye_height = 1.0
        self._hand_z = 0.2  # 帯の下端 0.6 まで不足 0.4 m
        self.motor_calls = []

    def _ensure_infant_eye_height(self):
        return float(self.infant_eye_height)

    def _get_hand_world_z(self, agent):
        return float(self._hand_z)

    def clear_babbling_chunk(self):
        pass

    def gaze_agent(self, agent_from, agent_to, yaw_link="head", action=None):
        return 0.0, 0.0

    def reset_arm_pose(self, agent, action=None):
        pass

    def wave_arm(self, agent, action=None, phase=0.0, gesture=False):
        self.motor_calls.append(("wave_arm", phase, gesture))

    def animate_hand_open_close(self, agent, action=None, phase=0.0, gesture=False):
        self.motor_calls.append(("animate_hand_open_close", phase, gesture))

    def hand_reach_location(self, agent, target_pos, action=None, ik_every=0, oid=-1):
        return action if action is not None else {}

    def palm_reach_error(self, agent, target_pos):
        return 0.0

    def _agent_torso_pose(self, agent):
        return np.array([0.0, 0.0, 0.5], dtype=np.float32), np.array(
            [1.0, 0.0, 0.0], dtype=np.float32
        )

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def _motor_signals(raise_on=True, face=1.0, babbling_chunk=None):
    return {
        "hand_height_margin": ARM_RAISE_HIT_RATE_MIN if raise_on else 0.0,
        "hand_open_close_periodicity": 0.0,
        "babbling_salience": 1.0 if babbling_chunk else 0.0,
        "babbling_chunk": babbling_chunk,
        "instant_parent_face_hit": float(face),
    }


def _make_base(**kwargs):
    defaults = dict(
        gui=False,
        num_agents=2,
        num_objects=2,
        object_counts={"RedCube": 1, "BlueCube": 1},
        max_steps=500,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        placement_method="grid",
    )
    defaults.update(kwargs)
    return BaseEnv(**defaults)


def _zero_actions(base: BaseEnv):
    return (
        np.zeros(base.action_space[0].shape, dtype=np.float32),
        np.zeros(base.action_space[1].shape, dtype=np.float32),
    )


class TestCloseOnRaise(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.protocol = DyadicProtocol(self.env, rejection_check=False)
        self.protocol._compute_gesture_pos = lambda: np.array(
            [0.35, 0.0, 0.8], dtype=np.float32
        )

    def test_close_on_raise_finishes_on_hit(self):
        self.protocol.start(
            {
                "parent_channels": ["arm_raise"],
                "close_on_raise": True,
            }
        )
        self.assertEqual(self.protocol.state, "performing_parent_cue")
        self.protocol._manage_state(_motor_signals(raise_on=True))
        self.assertEqual(self.protocol.state, "completed")
        self.assertTrue(self.protocol.is_finished)
        self.assertTrue(self.protocol.session_success)
        self.assertFalse(self.protocol.motor_active)

    def test_default_opens_motor_lane(self):
        self.protocol.start({"parent_channels": ["arm_raise"]})
        self.assertFalse(self.protocol.close_on_raise)
        self.protocol._manage_state(_motor_signals(raise_on=True))
        self.assertTrue(self.protocol.motor_active)
        self.assertTrue(self.protocol.session_success)
        self.assertFalse(self.protocol.is_finished)

    def test_close_on_raise_fails_when_cue_ends(self):
        self.protocol.start(
            {
                "parent_channels": ["arm_raise"],
                "close_on_raise": True,
            }
        )
        # 見本の duration を終えた扱いにする（到達済みの mock でタイマだけ進める）
        self.protocol.motor_timers["arm_raise"] = 90
        self.protocol._manage_state(_motor_signals(raise_on=False))
        self.assertEqual(self.protocol.state, "failure")
        self.assertTrue(self.protocol.is_finished)
        self.assertFalse(self.protocol.session_success)

    def test_close_on_raise_ignores_babble(self):
        self.protocol.start(
            {
                "parent_channels": ["arm_raise"],
                "close_on_raise": True,
            }
        )
        self.protocol._manage_state(
            _motor_signals(raise_on=False, babbling_chunk=["p0", "p1", "p0"])
        )
        self.assertEqual(self.protocol.state, "performing_parent_cue")
        self.assertFalse(self.protocol.voice_active)
        self.assertNotEqual(self.protocol.state, "echoing")
        self.assertFalse(self.protocol.session_success)


class TestDenseRewardFunction(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.protocol = DyadicProtocol(self.env, rejection_check=False)
        self.protocol._compute_gesture_pos = lambda: np.array(
            [0.35, 0.0, 0.8], dtype=np.float32
        )

    def test_inactive_before_cue(self):
        # 幼児始動で見本に入らない
        self.protocol.start({"infant_initiative": True})
        self.assertEqual(self.protocol.dense_reward_function({}), 0.0)

    def test_height_decrease_positive_increase_negative(self):
        self.protocol.start({"parent_channels": ["arm_raise"]})
        self.assertTrue(self.protocol._dense_active)
        self.assertAlmostEqual(self.protocol._dense_prev_h, 0.4, places=5)
        # 初回は prev 記録済みなので 0
        self.assertEqual(self.protocol.dense_reward_function({}), 0.0)

        # 手を 0.2 m 上げる → h: 0.4 → 0.2 → +0.5
        self.env._hand_z = 0.4
        r = self.protocol.dense_reward_function({})
        self.assertAlmostEqual(r, 0.5, places=5)

        # 帯へ入れる → h: 0.2 → 0 → +0.5
        self.env._hand_z = 0.6
        r = self.protocol.dense_reward_function({})
        self.assertAlmostEqual(r, 0.5, places=5)

        # 帯の中でさらに上げる → 0
        self.env._hand_z = 0.9
        r = self.protocol.dense_reward_function({})
        self.assertAlmostEqual(r, 0.0, places=5)

        # 帯から下ろす → 負
        self.env._hand_z = 0.2
        r = self.protocol.dense_reward_function({})
        self.assertAlmostEqual(r, -1.0, places=5)

    def test_success_bonus_then_zero(self):
        self.protocol.start(
            {
                "parent_channels": ["arm_raise"],
                "close_on_raise": True,
            }
        )
        self.protocol.dense_reward_function({})  # 変化なし
        self.protocol._manage_state(_motor_signals(raise_on=True))
        self.assertTrue(self.protocol.is_finished)
        self.assertTrue(self.protocol._dense_success_pending)
        self.assertAlmostEqual(
            self.protocol.dense_reward_function({}), DENSE_SUCCESS_BONUS
        )
        self.assertEqual(self.protocol.dense_reward_function({}), 0.0)

    def test_cue_end_without_success_zeros_dense(self):
        self.protocol.start(
            {
                "parent_channels": ["arm_raise"],
                "close_on_raise": True,
            }
        )
        self.protocol.dense_reward_function({})
        self.protocol.motor_timers["arm_raise"] = 90
        self.protocol._manage_state(_motor_signals(raise_on=False))
        self.assertTrue(self.protocol._dense_done)
        self.assertFalse(self.protocol._dense_success_pending)
        self.assertEqual(self.protocol.dense_reward_function({}), 0.0)


class TestProtocolManagerDyadicDense(unittest.TestCase):
    def test_dense_reward_strips_preference_and_injects(self):
        base = _make_base(
            num_objects=1,
            object_counts={"RedCube": 1},
            layout_freeze="full",
            infant_action_names=[
                "torso_joint",
                "shoulder_pitch_joint",
                "elbow_joint",
            ],
            infant_obs_keys=["image", "joint_angles"],
        )
        env = ProtocolManager(
            InfantSignalDetector(ParentTools(base)),
            dense_reward=True,
            household_interval_steps=10**9,
            rejection=False,
        )
        try:
            env.reset(seed=0)
            env.set_fixed_protocol(
                "dyadic",
                {
                    "parent_channels": ["arm_raise"],
                    "close_on_raise": True,
                },
            )
            proto = env.current_protocol
            self.assertIsNotNone(proto)
            proto.dense_reward_function = MagicMock(return_value=1.5)
            isd = env.env
            while not hasattr(isd, "_compute_preference_reward"):
                isd = isd.env
            isd._compute_preference_reward = MagicMock(
                return_value=(0.7, {"hold": 0.0, "look": 0.7, "total": 0.7})
            )
            _, r, _, _, info = env.step(_zero_actions(base))
            self.assertAlmostEqual(info["dense_reward"], 1.5)
            self.assertEqual(info["preference_reward"]["total"], 0.0)
            self.assertAlmostEqual(float(r[0]), 1.5)
        finally:
            base.close()


if __name__ == "__main__":
    unittest.main()
