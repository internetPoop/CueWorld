"""ProtocolManager の関わり履歴バッファ（バッチ C）テスト。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from cueworld.protocol_manager import (
    PROTOCOL_HISTORY_INFO_LEN,
    PROTOCOL_HISTORY_MAXLEN,
    ProtocolManager,
)


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
        cue_type=None,
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
        self.infant_initiative = False
        self.cue_type = cue_type
        self.active_channels = []
        self.salient_channels = []
        self.start_calls = []
        self.call_success_flag = None
        self.call_reward = 0.0
        self.call_proximity = 0.0
        self.session_success = False
        self._pm_leave_done = False

    def start(self, context):
        self.start_calls.append(dict(context))
        self.is_finished = False
        self.state = "running"
        self.infant_initiative = context.get("infant_initiative", False)
        self.session_success = False
        self._pm_leave_done = False

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


class TestProtocolManagerHistory(unittest.TestCase):
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
            cue_type="shake",
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
        self.manager.protocol_history.clear()

    def _signals(self):
        return {key: 0.0 for key in self.manager.signal_keys}

    def _use_joint_attention(self, *, start_step=10, infant_initiative=False):
        self.manager.fixed_protocol = None
        self.manager.current_protocol = self.joint_attention
        self.manager.current_protocol_name = "joint_attention"
        self.manager.current_protocol_start_step = start_step
        self.joint_attention.infant_initiative = infant_initiative
        self.joint_attention.is_finished = False
        self.joint_attention.call_success_flag = None
        self.joint_attention.target_object = SimpleNamespace(id=101)

    def test_natural_success_records_row(self):
        self.env.step_count = 20
        self._use_joint_attention(start_step=10, infant_initiative=True)
        self.joint_attention.call_success_flag = True
        self.joint_attention.session_success = True
        self.joint_attention.is_finished = True

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            self.manager.update(self._signals())

        self.assertEqual(len(self.manager.protocol_history), 1)
        row = self.manager.protocol_history[0]
        self.assertEqual(row["name"], "joint_attention")
        self.assertTrue(row["success"])
        self.assertEqual(row["end_reason"], "success")
        self.assertTrue(row["infant_initiative"])
        self.assertEqual(row["target_id"], 101)
        self.assertEqual(row["kind"], "shake")
        self.assertEqual(row["start_step"], 10)
        self.assertEqual(row["end_step"], 20)
        self.assertEqual(row["length"], 10)

    def test_natural_failure_records_row(self):
        self.env.step_count = 15
        self._use_joint_attention(start_step=5)
        self.joint_attention.call_success_flag = False
        self.joint_attention.session_success = False
        self.joint_attention.is_finished = True
        self.joint_attention.state = "failure"

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            self.manager.update(self._signals())

        self.assertEqual(len(self.manager.protocol_history), 1)
        row = self.manager.protocol_history[0]
        self.assertFalse(row["success"])
        self.assertEqual(row["end_reason"], "failure")
        self.assertGreater(self.manager.protocol_penalties["joint_attention"], 0.0)

    def test_interrupt_records_old_protocol_initiative(self):
        self.env.step_count = 25
        self._use_joint_attention(start_step=10, infant_initiative=False)
        self.joint_attention.session_success = False
        self.manager.current_infant_initiative = False

        def _interrupt_to_dyadic(_signals):
            self.manager.current_infant_initiative = True
            return True, "dyadic", {"infant_initiative": True}

        with patch.object(
            self.manager,
            "check_for_response",
            side_effect=_interrupt_to_dyadic,
        ):
            self.manager.update(self._signals())

        self.assertEqual(self.manager.current_protocol_name, "dyadic")
        self.assertTrue(self.manager.current_infant_initiative)
        self.assertEqual(len(self.manager.protocol_history), 1)
        row = self.manager.protocol_history[0]
        self.assertEqual(row["name"], "joint_attention")
        self.assertFalse(row["success"])
        self.assertEqual(row["end_reason"], "interrupt")
        self.assertFalse(row["infant_initiative"])
        self.assertEqual(row["target_id"], 101)
        self.assertEqual(row["kind"], "shake")
        self.assertEqual(row["start_step"], 10)
        self.assertEqual(row["end_step"], 25)
        self.assertEqual(row["length"], 15)

    def test_interrupt_after_session_success_records_success(self):
        self.env.step_count = 30
        self._use_joint_attention(start_step=10, infant_initiative=True)
        self.joint_attention.session_success = True
        self.manager.current_infant_initiative = True
        before_penalty = self.manager.protocol_penalties["joint_attention"]

        def _interrupt_to_dyadic(_signals):
            self.manager.current_infant_initiative = True
            return True, "dyadic", {"infant_initiative": True}

        with patch.object(
            self.manager,
            "check_for_response",
            side_effect=_interrupt_to_dyadic,
        ):
            self.manager.update(self._signals())

        row = self.manager.protocol_history[0]
        self.assertEqual(row["name"], "joint_attention")
        self.assertTrue(row["success"])
        self.assertEqual(row["end_reason"], "interrupt")
        self.assertTrue(row["infant_initiative"])
        self.assertEqual(
            self.manager.protocol_penalties["joint_attention"], before_penalty
        )

    def test_household_finish_does_not_append(self):
        self.env.step_count = 9
        self.manager.fixed_protocol = None
        self.manager.current_protocol = self.household
        self.manager.current_protocol_name = "household_solo"
        self.household.is_finished = True
        self.household.call_success_flag = True

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            self.manager.update(self._signals())

        self.assertEqual(len(self.manager.protocol_history), 0)

    def test_observance_switch_does_not_append(self):
        self.env.step_count = 3
        self.manager.fixed_protocol = None
        self.manager.current_protocol = self.observance
        self.manager.current_protocol_name = "observance"

        with patch.object(
            self.manager,
            "check_for_response",
            return_value=(True, "dyadic", {"infant_initiative": True}),
        ):
            self.manager.update(self._signals())

        self.assertEqual(len(self.manager.protocol_history), 0)

    def test_giving_kind_is_none(self):
        self.env.step_count = 12
        self.manager.fixed_protocol = None
        self.manager.current_protocol = self.object_giving
        self.manager.current_protocol_name = "object_giving"
        self.manager.current_protocol_start_step = 4
        self.object_giving.call_success_flag = True
        self.object_giving.session_success = True
        self.object_giving.is_finished = True
        self.object_giving.target_object = SimpleNamespace(id=7)

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            self.manager.update(self._signals())

        row = self.manager.protocol_history[0]
        self.assertEqual(row["name"], "object_giving")
        self.assertIsNone(row["kind"])
        self.assertEqual(row["target_id"], 7)

    def test_dyadic_kind_uses_active_channels(self):
        self.env.step_count = 8
        self.manager.fixed_protocol = None
        self.manager.current_protocol = self.dyadic
        self.manager.current_protocol_name = "dyadic"
        self.manager.current_protocol_start_step = 1
        self.dyadic.active_channels = ["voice", "arm_raise"]
        self.dyadic.salient_channels = ["voice"]
        self.dyadic.call_success_flag = True
        self.dyadic.session_success = True
        self.dyadic.is_finished = True

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            self.manager.update(self._signals())

        row = self.manager.protocol_history[0]
        self.assertEqual(row["kind"], ["voice", "arm_raise"])
        self.assertIsNone(row["target_id"])

    def test_reset_keeps_buffer_and_does_not_append_in_progress(self):
        self._use_joint_attention(start_step=10, infant_initiative=False)
        self.manager.protocol_history.append({
            "name": "dyadic",
            "success": True,
            "infant_initiative": False,
            "target_id": None,
            "start_step": 0,
            "end_step": 4,
            "length": 4,
            "kind": ["voice"],
            "end_reason": "success",
        })
        before = list(self.manager.protocol_history)

        self.manager.reset()

        self.assertEqual(list(self.manager.protocol_history), before)

    def test_maxlen_drops_oldest(self):
        self.manager.current_protocol = self.joint_attention
        self.manager.current_protocol_name = "joint_attention"
        self.joint_attention.infant_initiative = False
        for i in range(PROTOCOL_HISTORY_MAXLEN + 1):
            self.env.step_count = i + 1
            self.manager.current_protocol_start_step = i
            self.manager.save_protocol_history("joint_attention", True)

        self.assertEqual(len(self.manager.protocol_history), PROTOCOL_HISTORY_MAXLEN)
        self.assertEqual(self.manager.protocol_history[0]["end_step"], 2)

    def test_step_info_includes_recent_when_flag_on(self):
        self.manager.current_protocol = self.joint_attention
        self.manager.current_protocol_name = "joint_attention"
        self.joint_attention.infant_initiative = False
        for i in range(10):
            self.env.step_count = i + 1
            self.manager.current_protocol_start_step = i
            self.manager.save_protocol_history("joint_attention", True)

        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            _, _, _, _, info = self.manager.step(np.zeros(16, dtype=np.float32))
        self.assertIn("protocol_history", info)
        self.assertEqual(len(info["protocol_history"]), PROTOCOL_HISTORY_INFO_LEN)
        self.assertEqual(info["protocol_history"][-1]["end_step"], 10)
        info["protocol_history"][0]["name"] = "mutated"
        self.assertNotEqual(self.manager.protocol_history[-PROTOCOL_HISTORY_INFO_LEN]["name"], "mutated")

    def test_step_info_omits_history_when_flag_off(self):
        self.manager.log_protocol_history = False
        self.manager.current_protocol = self.joint_attention
        self.manager.current_protocol_name = "joint_attention"
        self.manager.save_protocol_history("joint_attention", True)
        with patch.object(self.manager, "check_for_response", return_value=(False, "", {})):
            _, _, _, _, info = self.manager.step(np.zeros(16, dtype=np.float32))
        self.assertNotIn("protocol_history", info)


if __name__ == "__main__":
    unittest.main()
