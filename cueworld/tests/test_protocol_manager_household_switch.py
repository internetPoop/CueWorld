"""ProtocolManager の household 時計切替テスト。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from cueworld.protocol_manager import ProtocolManager


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
        self.household_task_candidates = []
        self.all_objects = {}

    def reset(self, **kwargs):
        self.step_count = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        return np.zeros(1, dtype=np.float32), [0.0, 0.0], False, False, {}


class _FakeProtocol:
    def __init__(
        self,
        name,
        *,
        solo=False,
        request_type=False,
        response_type=False,
        level=0,
        persistence=0.0,
        urgency=False,
    ):
        self.name = name
        self.solo = solo
        self.request_type = request_type
        self.response_type = response_type
        self.level = level
        self.persistence = persistence
        self.urgency = urgency
        self.has_target = False
        self.state = "running"
        self.failure_state = "failure"
        self.is_finished = False
        self.target_object = None
        self.start_calls = []
        self.call_success_flag = None
        self.call_reward = 0.0
        self.call_proximity = 0.0

    def start(self, context):
        self.start_calls.append(dict(context))
        self.is_finished = False
        self.state = "running"

    def stop(self):
        pass

    def __call__(self, infant_signals, parent_info):
        return (
            np.zeros(16, dtype=np.float32),
            self.call_proximity,
            self.call_success_flag,
            self.call_reward,
        )


class _FakeHouseholdProtocol(_FakeProtocol):
    def __init__(self, env):
        super().__init__("HouseholdSoloProtocol", solo=True, persistence=5.0)
        self.env = env

    def _get_task_candidates(self):
        return list(getattr(self.env, "household_task_candidates", []) or [])

    def start(self, context):
        super().start(context)
        candidates = self._get_task_candidates()
        if not candidates:
            self.state = "completed"
            self.is_finished = True
            return
        board_id = candidates[0]["board_id"]
        board_obj = self.env.all_objects[board_id]
        _ = board_obj.name


class TestProtocolManagerHouseholdSwitch(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.manager = ProtocolManager(
            self.env,
            fixed_protocol="observance",
            fixed_context={},
            check_interval=1,
            household_interval_steps=5,
        )

        self.observance = _FakeProtocol("ObservanceProtocol", solo=True, persistence=1.0)
        self.dyadic = _FakeProtocol(
            "DyadicProtocol",
            request_type=True,
            response_type=True,
            level=0,
            persistence=5.0,
        )
        self.joint_attention = _FakeProtocol(
            "JointAttentionProtocol",
            request_type=True,
            response_type=True,
            level=2,
            persistence=5.0,
        )
        self.object_giving = _FakeProtocol(
            "ObjectGivingProtocol",
            request_type=True,
            response_type=True,
            level=1,
            persistence=5.0,
        )
        self.household = _FakeHouseholdProtocol(self.env)

        self.manager.protocols = {
            "observance": self.observance,
            "joint_attention": self.joint_attention,
            "dyadic": self.dyadic,
            "object_giving": self.object_giving,
            "household_solo": self.household,
        }
        self.manager.current_protocol = self.observance
        self.manager.current_protocol_name = "observance"
        self.manager.response_protocol = ["joint_attention", "dyadic", "object_giving"]
        self.manager.request_protocol = ["joint_attention", "dyadic", "object_giving"]

    def _signals(self):
        return {key: 0.0 for key in self.manager.signal_keys}

    def _set_household_candidate(self, *, board=True):
        self.env.household_task_candidates = []
        self.env.all_objects = {}
        if board:
            self.env.household_task_candidates = [
                {"task_type": "collect_red", "board_id": 10, "item_ids": [1]}
            ]
            self.env.all_objects = {10: SimpleNamespace(name="RedBoard")}

    def test_observance_before_due_does_not_start_household(self):
        self.env.step_count = 4
        self._set_household_candidate()
        self.manager.fixed_protocol = None

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})), \
             patch.object(self.manager, "check_for_initiative", return_value=(False, "", {})):
            self.manager.update(self._signals())

        self.assertEqual(self.manager.current_protocol_name, "observance")

    def test_observance_due_starts_household_before_initiative(self):
        self.env.step_count = 5
        self._set_household_candidate()
        self.manager.fixed_protocol = None

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})), \
             patch.object(self.manager, "check_for_initiative") as mock_initiative:
            self.manager.update(self._signals())

        self.assertEqual(self.manager.current_protocol_name, "household_solo")
        self.assertFalse(self.manager.current_infant_initiative)
        mock_initiative.assert_not_called()

    def test_response_wins_over_due_household(self):
        self.env.step_count = 5
        self._set_household_candidate()
        self.manager.fixed_protocol = None

        with patch.object(
            self.manager,
            "check_for_response",
            return_value=(True, "dyadic", {"infant_initiative": True}),
        ), patch.object(self.manager, "check_for_initiative") as mock_initiative:
            self.manager.update(self._signals())

        self.assertEqual(self.manager.current_protocol_name, "dyadic")
        mock_initiative.assert_not_called()

    def test_due_household_does_not_interrupt_non_observance(self):
        self.env.step_count = 5
        self._set_household_candidate()
        self.manager.fixed_protocol = None
        self.manager.current_protocol = self.dyadic
        self.manager.current_protocol_name = "dyadic"

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})), \
             patch.object(self.manager, "check_for_initiative") as mock_initiative:
            self.manager.update(self._signals())

        self.assertEqual(self.manager.current_protocol_name, "dyadic")
        mock_initiative.assert_not_called()

    def test_finished_interaction_goes_directly_to_household_when_due(self):
        self.env.step_count = 5
        self._set_household_candidate()
        self.manager.fixed_protocol = None
        self.manager.current_protocol = self.dyadic
        self.manager.current_protocol_name = "dyadic"
        self.dyadic.is_finished = True

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            self.manager.update(self._signals())

        self.assertEqual(self.manager.current_protocol_name, "household_solo")

    def test_household_completion_updates_clock_and_returns_to_observance(self):
        self.env.step_count = 9
        self._set_household_candidate()
        self.manager.fixed_protocol = None
        self.manager.current_protocol = self.household
        self.manager.current_protocol_name = "household_solo"
        self.household.is_finished = True

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            self.manager.update(self._signals())

        self.assertEqual(self.manager.last_household_end_step, 9)
        self.assertEqual(self.manager.current_protocol_name, "observance")

    def test_due_without_candidates_keeps_observing_and_deadline_remains(self):
        self.env.step_count = 5
        self.env.household_task_candidates = []
        self.env.all_objects = {}
        self.manager.fixed_protocol = None

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})), \
             patch.object(self.manager, "check_for_initiative", return_value=(False, "", {})):
            self.manager.update(self._signals())

        self.assertEqual(self.manager.current_protocol_name, "observance")
        self.assertTrue(self.manager._household_is_due())

    def test_due_without_board_keeps_observing(self):
        self.env.step_count = 5
        self.env.household_task_candidates = []
        self.env.all_objects = {}
        self.manager.fixed_protocol = None

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})), \
             patch.object(self.manager, "check_for_initiative", return_value=(False, "", {})):
            self.manager.update(self._signals())

        self.assertEqual(self.manager.current_protocol_name, "observance")

    def test_household_does_not_update_learning_ema(self):
        self.manager.current_try_count = 7
        self.manager.protocol_try_counts["household_solo"] = 3
        self.manager.infant_initiative_rate = 0.25
        self.manager.level_scores = [0.2, 0.4, 0.6]

        self.manager.save_protocol_history("household_solo", True)

        self.assertEqual(self.manager.current_try_count, 7)
        self.assertEqual(self.manager.protocol_try_counts["household_solo"], 3)
        self.assertEqual(self.manager.infant_initiative_rate, 0.25)
        self.assertEqual(self.manager.level_scores, [0.2, 0.4, 0.6])

    def test_household_still_checks_response(self):
        self.env.step_count = 5
        self._set_household_candidate()
        self.manager.fixed_protocol = None
        self.manager.current_protocol = self.household
        self.manager.current_protocol_name = "household_solo"

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})) as mock_response, \
             patch.object(self.manager, "check_for_initiative") as mock_initiative:
            self.manager.update(self._signals())

        mock_response.assert_called_once()
        mock_initiative.assert_not_called()

    def test_household_never_calls_initiative(self):
        self.env.step_count = 5
        self._set_household_candidate()
        self.manager.fixed_protocol = None
        self.manager.current_protocol = self.household
        self.manager.current_protocol_name = "household_solo"

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})), \
             patch.object(self.manager, "check_for_initiative") as mock_initiative:
            self.manager.update(self._signals())

        mock_initiative.assert_not_called()


if __name__ == "__main__":
    unittest.main()
