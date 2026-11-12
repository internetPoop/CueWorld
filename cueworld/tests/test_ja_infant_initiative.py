"""JointAttentionProtocol の J2 幼児始動フローのユニットテスト。"""

import unittest

import gymnasium as gym
import numpy as np

from cueworld.protocols.joint_attention import JointAttentionProtocol
from cueworld.protocols.base import ATTENTION_HOLD_STEPS, ATTENTION_WAIT_STEPS


class _MockObject:
    def __init__(self, oid: int, concept_id: str):
        self.id = oid
        self.concept_id = concept_id
        self.name = concept_id


class _MockEnv(gym.Env):
    action_num = 17
    action_map = {
        "head_pan_joint": 1,
        "head_tilt_joint": 2,
        "volume": 12,
        "v_1": 14,
        "v_2": 15,
        "v_3": 16,
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
        self.reset_arm_called = False
        self.gaze_object_calls = []
        self.gaze_agent_calls = []
        self.point_calls = []

    @property
    def unwrapped(self):
        return self

    def reset_arm_pose(self, agent, action=None):
        self.reset_arm_called = True

    def gaze_object(self, agent, oid, yaw_link="head", action=None, option=None):
        self.gaze_object_calls.append((agent["body"], oid))
        return 0.0, 0.0

    def gaze_agent(self, agent_from, agent_to, yaw_link="head", action=None):
        self.gaze_agent_calls.append((agent_from["body"], agent_to["body"]))
        return 0.0, 0.0

    def point_object(self, agent, oid, action=None, ik_every=0):
        self.point_calls.append((agent["body"], oid))
        return action

    def approach_infant_for_ja(
        self, agent, infant, oid, action=None, thresh=None, approach_dist=1.0, min_angle_deg=30.0
    ):
        return 0.0, 3.0

    def approach_infant(self, agent, infant, action=None, approach_dist=1.0):
        return 0.0, 0.5

    def compute_show_position(self, parent, infant, show_dist=0.5):
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def palm_reach_error(self, agent, target_pos):
        return 0.0

    def grasp_object(self, agent, oid, action=None, approach_dist=0.2):
        agent["held_obj"] = oid

    def grip_object(self, agent, action=None):
        return action

    def shake_object(self, agent, action=None, target_pos=None, shake_amp=0.5):
        return action

    def apply_phoneme_to_action(self, action, coords, volume=0.5):
        if "volume" in self.action_map:
            action[self.action_map["volume"]] = volume
        return action


def _signals_for_target(target_id: int, *, point=False, stable=True):
    gaze_targets = np.array([0.0, 0.0, 0.0], dtype=float)
    gesture_targets = np.array([0.0, 0.0, 0.0], dtype=float)
    if target_id == 101:
        gaze_targets[0] = 1.0
        gesture_targets[0] = 1.0
    elif target_id == 102:
        gaze_targets[1] = 1.0
        gesture_targets[1] = 1.0
    return {
        "gaze_targets": gaze_targets,
        "gaze_targets_now": gaze_targets.copy(),
        "gesture_targets": gesture_targets,
        "point_intensity": 0.2 if point else 0.0,
        "hand_target_stability": 0.8 if stable else 0.3,
        "voice_best_distance": 1.0,
        "words": {},
        "looking_at_parent_eye": 0.0,
    }


class TestJointAttentionInfantInitiative(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.protocol = JointAttentionProtocol(
            self.env, cue_type="looking", rejection_check=False
        )

    def test_parent_initiated_hold_enters_post_shared(self):
        self.protocol.start({"fixed_target": 101, "intent_type": "informative"})
        self.protocol.state = "looking"
        self.protocol.attention_tracker.hold_steps = 3

        self.protocol._manage_state(_signals_for_target(102))
        self.assertEqual(self.protocol.state, "looking")
        self.assertGreater(len(self.protocol.utterance_queue), 0)

        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "post_shared_attention")
        self.assertTrue(self.protocol.ja_success)

    def test_infant_initiated_looking_queues_utterance(self):
        self.protocol.start(
            {
                "fixed_target": 101,
                "intent_type": "informative",
                "infant_initiative": True,
            }
        )
        self.protocol.state = "looking"
        self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "looking")
        self.assertGreater(len(self.protocol.utterance_queue), 0)

    def test_infant_initiated_hold_completes_without_speech(self):
        self.protocol.start(
            {
                "fixed_target": 101,
                "intent_type": "informative",
                "infant_initiative": True,
            }
        )
        self.protocol.state = "looking"
        self.protocol._cue_started = True
        self.protocol.attention_tracker.hold_steps = 3
        self.protocol.parent_utterance = False

        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "post_shared_attention")

        self.protocol.voice_wait_timer = 0
        for _ in range(ATTENTION_WAIT_STEPS):
            self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "completed")

    def test_infant_initiated_cue_utterance_only_once(self):
        self.protocol.start(
            {
                "fixed_target": 101,
                "intent_type": "informative",
                "infant_initiative": True,
            }
        )
        self.protocol.state = "looking"
        self.protocol._manage_state(_signals_for_target(102))
        first_len = len(self.protocol.utterance_queue)
        self.assertTrue(self.protocol._cue_started)
        self.assertGreater(first_len, 0)

        self.protocol._manage_state(_signals_for_target(102))
        self.assertEqual(len(self.protocol.utterance_queue), first_len)

    def test_select_target_prefers_reliable_pointing(self):
        signals = _signals_for_target(101, point=True, stable=True)
        signals["gaze_targets"] = np.array([0.0, 1.0, 0.0], dtype=float)
        self.protocol.start(
            {
                "infant_initiative": True,
                "infant_signals": signals,
            }
        )
        self.assertEqual(self.protocol.target_object.id, 101)

    def test_select_target_falls_back_to_gaze_when_pointing_unreliable(self):
        signals = _signals_for_target(101, point=True, stable=False)
        signals["gaze_targets"] = np.array([0.0, 1.0, 0.0], dtype=float)
        self.protocol.start(
            {
                "infant_initiative": True,
                "infant_signals": signals,
            }
        )
        self.assertEqual(self.protocol.target_object.id, 102)

    def test_select_target_prefers_weak_arm_gesture(self):
        """§6-D2: 指差し形がなくても弱い腕（向きがはっきり）なら手の先。"""
        signals = _signals_for_target(101, point=False, stable=True)
        signals["point_intensity"] = 0.0
        signals["holding"] = 0.0
        signals["gaze_targets"] = np.array([0.0, 1.0, 0.0], dtype=float)
        self.protocol.start(
            {
                "infant_initiative": True,
                "infant_signals": signals,
            }
        )
        self.assertEqual(self.protocol.target_object.id, 101)

    def test_get_infant_target_always_uses_gaze(self):
        """§6-D2: 成功判定は視線。揃った指差し中でも gesture に切り替えない。"""
        self.protocol.start(
            {
                "fixed_target": 101,
                "infant_initiative": True,
            }
        )
        signals = _signals_for_target(101, point=True, stable=True)
        signals["gaze_targets"] = np.array([1.0, 0.0, 0.0], dtype=float)
        signals["gaze_targets_now"] = np.array([0.0, 1.0, 0.0], dtype=float)
        self.assertEqual(self.protocol.get_infant_target(signals), 102)

    def test_get_infant_target_returns_none_for_zero_mass(self):
        self.protocol.start({"fixed_target": 101})
        self.assertIsNone(
            self.protocol.get_infant_target(
                {
                    "gaze_targets": np.array([1.0, 0.0, 0.0], dtype=float),
                    "gaze_targets_now": np.zeros(3, dtype=float),
                    "gesture_targets": np.zeros(3, dtype=float),
                    "point_intensity": 0.0,
                    "hand_target_stability": 0.0,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
