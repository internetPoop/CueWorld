"""バッチ F: 除外リスト（blocked_pairs / success_streak）。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from cueworld.protocol_manager import SUCCESS_STREAK_MAX, ProtocolManager


class _MockObject:
    def __init__(self, oid: int, concept_id: str, is_platform: bool = False):
        self.id = oid
        self.concept_id = concept_id
        self.is_platform = is_platform


class _MockEnv(gym.Env):
    action_num = 16
    eval_mode = False
    active_signals = None
    target_obj_id = None

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = (
            gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32),
            gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32),
        )
        self.step_count = 0
        self.parent = {"body": 1, "held_obj": None}
        self.infant = {"body": 2, "held_obj": None}
        self.all_objects = {
            101: _MockObject(101, "red_cube"),
            102: _MockObject(102, "green_cube"),
        }
        self.active_objects = [101, 102]
        self.trackable_targets = [101, 102, self.parent["body"]]
        self.household_task_candidates = []
        self.voice_volume_threshold = 0.25

    def object_in_agent_reach(self, agent, oid):
        return False

    def reset(self, **kwargs):
        self.step_count = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        return np.zeros(1, dtype=np.float32), [0.0, 0.0], False, False, {}


class _FakeProtocol:
    def __init__(self, name, *, solo=False, persistence=0.0, cue_type=None):
        self.name = name
        self.solo = solo
        self.persistence = persistence
        self.has_target = False
        self.state = "running"
        self.success_state = "completed"
        self.failure_state = "failure"
        self.is_finished = False
        self.target_object = None
        self.current_item_id = None
        self.infant_initiative = False
        self.cue_type = cue_type
        self.active_channels = []
        self.salient_channels = []
        self.call_success_flag = None
        self._block_on_failure = True

    def start(self, context):
        self.is_finished = False
        self.state = "running"

    def stop(self):
        pass

    def blocks_target_on_failure(self):
        return self._block_on_failure

    def __call__(self, infant_signals, parent_info):
        return (
            np.zeros(16, dtype=np.float32),
            0.0,
            self.call_success_flag,
            0.0,
        )


class TestTargetBlockList(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.manager = ProtocolManager(
            self.env,
            fixed_protocol="observance",
            fixed_context={},
            check_interval=1,
            household_interval_steps=5,
            decay_rate=0.5,
        )
        self.observance = _FakeProtocol("ObservanceProtocol", solo=True, persistence=1.0)
        self.joint_attention = _FakeProtocol(
            "JointAttentionProtocol", persistence=5.0, cue_type="shake"
        )
        self.object_giving = _FakeProtocol("ObjectGivingProtocol", persistence=5.0)
        self.dyadic = _FakeProtocol("DyadicProtocol", persistence=5.0)
        self.household = _FakeProtocol("HouseholdSoloProtocol", solo=True, persistence=5.0)
        self.manager.protocols = {
            "observance": self.observance,
            "joint_attention": self.joint_attention,
            "object_giving": self.object_giving,
            "dyadic": self.dyadic,
            "household_solo": self.household,
        }
        self.manager.current_protocol = self.observance
        self.manager.current_protocol_name = "observance"
        self.manager.protocol_history.clear()

    def _zeros(self):
        signals = {key: 0.0 for key in self.manager.signal_keys}
        signals["gesture_targets"] = np.zeros(3, dtype=float)
        signals["gaze_targets"] = np.zeros(3, dtype=float)
        signals["words"] = {}
        signals["looking_at_parent_eye"] = 0.0
        signals["touching"] = 0.0
        signals["voice_best_distance"] = 10.0
        return signals

    def _pointing_at(self, target_idx: int = 0):
        s = self._zeros()
        s["point_intensity"] = 1.0
        s["hand_target_stability"] = 0.8
        s["gesture_targets"][target_idx] = 1.0
        return s

    def _reach_at(self, target_idx: int = 0):
        s = self._zeros()
        s["open_intensity"] = 1.0
        s["hand_target_stability"] = 0.8
        s["gesture_targets"][target_idx] = 1.0
        s["holding"] = 0.0
        return s

    def _use(self, proto, name, *, target_id=101):
        self.manager.fixed_protocol = None
        self.manager.current_protocol = proto
        self.manager.current_protocol_name = name
        self.manager.current_protocol_start_step = 0
        proto.is_finished = False
        proto.call_success_flag = None
        proto.target_object = SimpleNamespace(id=target_id) if target_id is not None else None
        proto.state = "running"

    def test_failure_adds_pair(self):
        self._use(self.joint_attention, "joint_attention")
        self.joint_attention.call_success_flag = False
        self.joint_attention.is_finished = True
        self.joint_attention.state = "failure"
        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            self.manager.update(self._zeros())
        self.assertIn(("joint_attention", 101), self.manager.blocked_pairs)

    def test_interrupt_does_not_add(self):
        self._use(self.joint_attention, "joint_attention")
        with patch.object(
            self.manager,
            "check_for_response",
            return_value=(True, "dyadic", {"infant_initiative": True}),
        ):
            self.manager.update(self._zeros())
        self.assertNotIn(("joint_attention", 101), self.manager.blocked_pairs)
        self.assertEqual(self.manager.blocked_pairs, set())

    def test_pre_release_failure_does_not_add(self):
        self._use(self.object_giving, "object_giving")
        self.object_giving._block_on_failure = False
        self.object_giving.call_success_flag = False
        self.object_giving.is_finished = True
        self.object_giving.state = "failure"
        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            self.manager.update(self._zeros())
        self.assertNotIn(("object_giving", 101), self.manager.blocked_pairs)

    def test_streak_three_adds_pair(self):
        for _ in range(SUCCESS_STREAK_MAX):
            self._use(self.joint_attention, "joint_attention")
            self.joint_attention.call_success_flag = True
            self.joint_attention.is_finished = True
            self.joint_attention.state = "completed"
            with patch.object(
                self.manager, "check_for_response", return_value=(False, "", {})
            ):
                self.manager.update(self._zeros())
        self.assertIn(("joint_attention", 101), self.manager.blocked_pairs)

    def test_different_target_resets_streak(self):
        for target in (101, 101, 102, 102):
            self._use(self.joint_attention, "joint_attention", target_id=target)
            self.joint_attention.call_success_flag = True
            self.joint_attention.is_finished = True
            self.joint_attention.state = "completed"
            with patch.object(
                self.manager, "check_for_response", return_value=(False, "", {})
            ):
                self.manager.update(self._zeros())
        self.assertNotIn(("joint_attention", 101), self.manager.blocked_pairs)
        self.assertNotIn(("joint_attention", 102), self.manager.blocked_pairs)
        self.assertEqual(self.manager.success_streak["joint_attention"], (102, 2))

    def test_success_clears_nonempty_list(self):
        self.manager.blocked_pairs.add(("joint_attention", 101))
        self.manager.success_streak["joint_attention"] = (101, 2)
        self._use(self.object_giving, "object_giving", target_id=102)
        self.object_giving.call_success_flag = True
        self.object_giving.is_finished = True
        self.object_giving.state = "completed"
        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            self.manager.update(self._zeros())
        self.assertEqual(self.manager.blocked_pairs, set())
        self.assertEqual(self.manager.success_streak, {})

    def test_household_success_clears_list(self):
        self.manager.blocked_pairs.add(("joint_attention", 101))
        self.manager.success_streak["joint_attention"] = (101, 1)
        self._use(self.household, "household_solo", target_id=None)
        self.household.current_item_id = 101
        self.household.call_success_flag = True
        self.household.is_finished = True
        self.household.state = "completed"
        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            self.manager.update(self._zeros())
        self.assertEqual(self.manager.blocked_pairs, set())
        self.assertEqual(self.manager.success_streak, {})

    def test_empty_list_success_only_advances_streak(self):
        self._use(self.joint_attention, "joint_attention")
        self.joint_attention.call_success_flag = True
        self.joint_attention.is_finished = True
        self.joint_attention.state = "completed"
        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            self.manager.update(self._zeros())
        self.assertEqual(self.manager.blocked_pairs, set())
        self.assertEqual(self.manager.success_streak["joint_attention"], (101, 1))

    @patch("cueworld.protocol_manager.np.random.choice")
    def test_blocked_play_drops_to_maintain(self, mock_choice):
        mock_choice.return_value = "maintain"
        self.manager.blocked_pairs.add(("joint_attention", 101))
        self.manager.current_protocol.persistence = 0.0
        self.manager.check_for_response(self._pointing_at(0))
        choices = list(mock_choice.call_args[0][0])
        self.assertNotIn("joint_attention", choices)
        self.assertIn("maintain", choices)

    @patch("cueworld.protocol_manager.np.random.choice")
    def test_blocked_play_goes_to_other_play(self, mock_choice):
        self.manager.blocked_pairs.add(("joint_attention", 101))
        self.manager.current_protocol.persistence = 0.0

        def _choose(choices, p=None):
            names = list(choices)
            self.assertNotIn("joint_attention", names)
            self.assertIn("object_giving", names)
            return "object_giving"

        mock_choice.side_effect = _choose
        switched, name, ctx = self.manager.check_for_response(self._reach_at(0))
        self.assertTrue(switched)
        self.assertEqual(name, "object_giving")
        self.assertEqual(ctx.get("intent_target_id"), 101)

    @patch("cueworld.protocol_manager.np.random.choice")
    def test_none_target_is_not_dropped(self, mock_choice):
        mock_choice.return_value = "observance"
        self.manager.blocked_pairs.add(("object_giving", 101))
        self.manager.fixed_protocol = None
        self.manager.current_protocol = self.observance
        self.manager.current_protocol_name = "observance"
        self.manager.check_for_initiative(self._zeros())
        choices = list(mock_choice.call_args[0][0])
        self.assertIn("object_giving", choices)

    def test_reset_clears_list(self):
        self.manager.blocked_pairs.add(("joint_attention", 101))
        self.manager.success_streak["joint_attention"] = (101, 2)
        self.manager.reset()
        self.assertEqual(self.manager.blocked_pairs, set())
        self.assertEqual(self.manager.success_streak, {})
