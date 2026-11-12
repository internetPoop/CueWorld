"""バッチ F: 開始時の手放し前段。"""

import unittest

import gymnasium as gym
import numpy as np

from cueworld.protocols.base import RELEASE_TIMEOUT_STEPS, BaseProtocol
from cueworld.protocols.dyadic import DyadicProtocol
from cueworld.protocols.joint_attention import JointAttentionProtocol
from cueworld.protocols.object_giving import ObjectGivingProtocol


class _MockObject:
    def __init__(self, oid: int, concept_id: str):
        self.id = oid
        self.concept_id = concept_id
        self.name = concept_id


class _MockEnv(gym.Env):
    action_num = 17
    action_map = {
        "index_joint": 8,
        "other_fingers": 9,
        "volume": 12,
        "v_1": 14,
        "v_2": 15,
        "v_3": 16,
        "head_pan_joint": 1,
        "head_tilt_joint": 2,
    }
    phoneme_mapping = {"p0": np.array([-1.0, -1.0, -1.0], dtype=np.float32)}
    client_id = 0
    stand_dist = 1.0
    place_dist = 0.5
    grasp_approach_dist = 0.2

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = (
            gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32),
            gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32),
        )
        self.parent = {"body": 1, "held_obj": None}
        self.infant = {"body": 2, "held_obj": None}
        self.all_objects = {
            101: _MockObject(101, "red_cube"),
            102: _MockObject(102, "green_cube"),
        }
        self.active_objects = [101, 102]
        self.trackable_targets = [101, 102, self.parent["body"]]
        self.open_hand_calls = 0
        self.drop_on_open = False

    def open_hand(self, agent, action=None):
        self.open_hand_calls += 1
        if self.drop_on_open:
            agent["held_obj"] = None
        if action is None:
            action = np.zeros(self.action_num, dtype=np.float32)
        action[self.action_map["index_joint"]] = -3.14
        action[self.action_map["other_fingers"]] = -3.14
        return action

    def reset_arm_pose(self, agent, action=None):
        return action

    def gaze_object(self, agent, oid, yaw_link="head", action=None, option=None):
        return 0.0, 0.0

    def gaze_agent(self, agent_from, agent_to, yaw_link="head", action=None):
        return 0.0, 0.0

    def point_object(self, agent, oid, action=None, ik_every=0):
        return action

    def grasp_object(self, agent, oid, action=None, approach_dist=0.2):
        agent["held_obj"] = oid
        return action

    def grip_object(self, agent, action=None):
        return action

    def approach_infant(self, agent, infant, action=None, approach_dist=1.0):
        return 0.0, 0.5

    def apply_phoneme_to_action(self, action, coords, volume=0.5):
        return action


def _empty_signals():
    return {
        "gaze_targets": np.zeros(3, dtype=float),
        "gesture_targets": np.zeros(3, dtype=float),
        "words": {},
        "voice_volume": 0.0,
    }


class TestPreRelease(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()

    def test_giving_other_object_enters_pre_release(self):
        proto = ObjectGivingProtocol(self.env, rejection_check=False)
        self.env.parent["held_obj"] = 102
        proto.start({"fixed_target": 101})
        action, _, success, _ = proto(_empty_signals(), {})
        self.assertTrue(proto._in_pre_release)
        self.assertEqual(proto.state, "init")
        self.assertGreater(self.env.open_hand_calls, 0)
        self.assertAlmostEqual(
            float(action[self.env.action_map["index_joint"]]), -3.14, places=4
        )
        self.assertIsNone(success)

    def test_giving_other_object_proceeds_after_drop(self):
        proto = ObjectGivingProtocol(self.env, rejection_check=False)
        self.env.parent["held_obj"] = 102
        proto.start({"fixed_target": 101})
        proto(_empty_signals(), {})
        self.assertTrue(proto._in_pre_release)
        self.env.parent["held_obj"] = None
        proto(_empty_signals(), {})
        self.assertFalse(proto._in_pre_release)
        self.assertEqual(proto.state, "fetching")

    def test_giving_same_object_skips_pre_release(self):
        proto = ObjectGivingProtocol(self.env, rejection_check=False)
        self.env.parent["held_obj"] = 101
        proto.start({"fixed_target": 101})
        proto(_empty_signals(), {})
        self.assertFalse(proto._in_pre_release)
        self.assertEqual(proto.state, "fetching")
        self.assertEqual(self.env.open_hand_calls, 0)
        proto(_empty_signals(), {})
        self.assertEqual(proto.state, "approach")

    def test_pointing_ja_releases_any_held(self):
        proto = JointAttentionProtocol(
            self.env, cue_type="pointing", rejection_check=False
        )
        self.env.parent["held_obj"] = 101
        proto.start({"fixed_target": 101})
        proto(_empty_signals(), {})
        self.assertTrue(proto._in_pre_release)
        self.assertEqual(proto.state, "init")

    def test_shake_ja_keeps_same_object(self):
        proto = JointAttentionProtocol(
            self.env, cue_type="shake", rejection_check=False
        )
        self.env.parent["held_obj"] = 101
        proto.start({"fixed_target": 101})
        proto(_empty_signals(), {})
        self.assertFalse(proto._in_pre_release)
        self.assertEqual(proto.state, "fetching")
        proto(_empty_signals(), {})
        self.assertEqual(proto.state, "approach")

    def test_dyadic_releases_any_held(self):
        proto = DyadicProtocol(self.env, rejection_check=False)
        self.env.parent["held_obj"] = 101
        proto.start({"parent_channels": ["voice"]})
        proto(_empty_signals(), {})
        self.assertTrue(proto._in_pre_release)
        self.assertGreater(self.env.open_hand_calls, 0)

    def test_timeout_fails_without_blocking_target(self):
        proto = ObjectGivingProtocol(self.env, rejection_check=False)
        self.env.parent["held_obj"] = 102
        proto.start({"fixed_target": 101})
        for _ in range(RELEASE_TIMEOUT_STEPS + 1):
            proto(_empty_signals(), {})
        self.assertEqual(proto.state, proto.failure_state)
        self.assertTrue(proto.is_finished)
        self.assertTrue(proto._pre_release_failed)
        self.assertFalse(proto.blocks_target_on_failure())
        self.assertEqual(self.env.parent["held_obj"], 102)

    def test_base_default_does_not_keep(self):
        proto = BaseProtocol("Dummy", self.env, rejection_check=False)
        self.assertFalse(proto.keeps_held_object(101))
        self.assertTrue(proto.blocks_target_on_failure())
