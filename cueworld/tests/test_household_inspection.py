"""HouseholdSoloProtocol Phase 4（inspection / hesitation）のテスト。"""

import unittest
from unittest.mock import MagicMock, ANY

import numpy as np

from cueworld.base import BaseEnv
from cueworld.parent_tools import ParentTools
from cueworld.infant_signal import InfantSignalDetector
from cueworld.protocol_manager import ProtocolManager
from cueworld.protocols.household_solo import HouseholdSoloProtocol


def _make_wrapped_env():
    yh = 1.2
    fixed = [{"class": "Red_Board", "x": 0.0, "y": yh - 0.05, "z": 0.5, "yaw": 90.0}]
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=3,
        object_counts={"RedCube": 1, "RedSphere": 1},
        max_steps=500,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        arena_rows=4,
        arena_cols=5,
        fixed_placements=fixed,
        placement_method="grid",
        use_cinematic_model=True,
    )
    return ProtocolManager(InfantSignalDetector(ParentTools(base)), demo_mode=False)


def _no_inspection_context(**extra):
    ctx = {
        "fixed_task": "collect_red",
        "inspect_item_prob": 0.0,
        "hesitate_prob": 0.0,
        "inspect_board_prob": 0.0,
    }
    ctx.update(extra)
    return ctx


def _get_base_env(env):
    current = env
    while hasattr(current, "env"):
        if isinstance(current, BaseEnv):
            return current
        current = current.env
    return current


def _run_until_finished(env, max_steps=500):
    proto = env.current_protocol
    base = _get_base_env(env)
    infant_action = np.zeros(base.action_space[0].shape, dtype=np.float32)
    parent_action = np.zeros(base.action_space[1].shape, dtype=np.float32)
    step = 0
    while not proto.is_finished and step < max_steps:
        _, _, _, _, info = env.step((infant_action, parent_action))
        if "parent_action" in info:
            parent_action = info["parent_action"]
        step += 1
    return step, proto.is_finished


class TestHouseholdInspection(unittest.TestCase):
    def test_inspecting_item_transitions_to_moving_after_duration(self):
        proto = HouseholdSoloProtocol(MagicMock())
        proto.inspection_params = {
            "inspect_item_duration": 5,
            "hesitate_duration": 5,
            "inspect_board_duration": 5,
            "inspect_item_prob": 1.0,
            "hesitate_prob": 0.0,
            "inspect_board_prob": 0.0,
        }
        proto.current_item_id = 1
        proto.global_intent = "collect_red"
        proto.env.all_objects = {1: MagicMock(name="RedCube")}
        proto.env.all_objects[1].name = "RedCube"
        proto._start_inspecting_item()
        self.assertEqual(proto.state, "inspecting_item")

        for _ in range(4):
            proto._manage_state({})
        self.assertEqual(proto.state, "inspecting_item")

        proto._manage_state({})
        self.assertEqual(proto.state, "moving_to_item")
        self.assertEqual(proto.state_timer, 0)

    def test_inspecting_board_after_grasp(self):
        proto = HouseholdSoloProtocol(MagicMock())
        proto.inspection_params = {
            "inspect_item_duration": 5,
            "hesitate_duration": 5,
            "inspect_board_duration": 3,
            "inspect_item_prob": 0.0,
            "hesitate_prob": 0.0,
            "inspect_board_prob": 1.0,
        }
        proto.current_item_id = 1
        proto.target_board_id = 2
        proto.global_intent = "collect_red"
        proto.env.parent = {"held_obj": 1}
        proto.env.all_objects = {
            1: MagicMock(name="RedCube"),
            2: MagicMock(name="RedBoard"),
        }
        proto.env.all_objects[1].name = "RedCube"
        proto.env.all_objects[2].name = "RedBoard"
        proto.state = "grasping_item"
        proto.state_timer = 0

        proto._manage_state({})
        self.assertEqual(proto.state, "inspecting_board")
        self.assertEqual(len(proto.inspection_history), 1)
        self.assertEqual(proto.inspection_history[0]["kind"], "inspecting_board")

        for _ in range(3):
            proto._manage_state({})
        self.assertEqual(proto.state, "moving_to_board")

    def test_determine_action_uses_gaze_not_approach_during_inspection(self):
        proto = HouseholdSoloProtocol(MagicMock())
        proto.action_dim = 8
        proto.env.action_num = 8
        proto.env.reset_arm_pose = MagicMock()
        proto.env.gaze_object = MagicMock()
        proto.env.approach_object = MagicMock()
        proto.current_item_id = 3
        proto.state = "inspecting_item"

        action = proto._determine_action()
        proto.env.gaze_object.assert_called_once()
        proto.env.approach_object.assert_not_called()
        self.assertEqual(len(action), 8)

    def test_start_with_forced_item_inspection(self):
        env = _make_wrapped_env()
        env.reset(seed=42)
        env.set_fixed_protocol(
            "household_solo",
            _no_inspection_context(inspect_item_prob=1.0),
        )
        proto = env.current_protocol
        self.assertIsInstance(proto, HouseholdSoloProtocol)
        self.assertEqual(proto.state, "inspecting_item")
        self.assertEqual(proto.global_intent, "collect_red")
        self.assertEqual(len(proto.inspection_history), 1)
        env.close()

    def test_inspection_extends_completion_time(self):
        # AR=1 では 1 env.step あたりの実効移動が旧 AR=4 より小さい。
        # 長時間の item+board 強制 inspection は完走が不安定なため、
        # item inspection のみ・短め duration で「完走が延びる」ことを見る。
        steps_without = []
        steps_with = []
        for _ in range(2):
            env = _make_wrapped_env()
            env.reset(seed=7)
            env.set_fixed_protocol("household_solo", _no_inspection_context())
            proto = env.current_protocol
            intent = proto.global_intent
            step, finished = _run_until_finished(env, max_steps=2500)
            steps_without.append(step)
            self.assertTrue(finished, "protocol should complete without inspection")
            self.assertEqual(proto.global_intent, intent)
            env.close()

        for _ in range(2):
            env = _make_wrapped_env()
            env.reset(seed=7)
            env.set_fixed_protocol(
                "household_solo",
                _no_inspection_context(
                    inspect_item_prob=1.0,
                    inspect_board_prob=0.0,
                    inspect_item_duration=15,
                ),
            )
            proto = env.current_protocol
            intent = proto.global_intent
            step, finished = _run_until_finished(env, max_steps=2500)
            steps_with.append(step)
            self.assertTrue(finished, "protocol should complete with inspection")
            self.assertEqual(proto.global_intent, intent)
            self.assertGreater(len(proto.inspection_history), 0)
            env.close()

        self.assertGreater(min(steps_with), min(steps_without))

    def test_hesitating_alternates_gaze_targets(self):
        proto = HouseholdSoloProtocol(MagicMock())
        proto.action_dim = 8
        proto.env.action_num = 8
        proto.env.reset_arm_pose = MagicMock()
        proto.env.gaze_object = MagicMock()
        proto.candidate_items = [1, 2]
        proto.current_item_id = 1
        proto._remaining_item_candidates = lambda: [1, 2]
        proto.hesitation_alt_item_id = 2
        proto.state = "hesitating"
        proto.state_timer = 5
        proto._determine_action()
        proto.env.gaze_object.assert_called_with(
            proto.env.parent, 1, yaw_link="head", action=ANY
        )

        proto.state_timer = 15
        proto._determine_action()
        proto.env.gaze_object.assert_called_with(
            proto.env.parent, 2, yaw_link="head", action=ANY
        )


class TestHouseholdBatchAContract(unittest.TestCase):
    def test_start_defaults_no_play_target_and_inspection_off(self):
        env = _make_wrapped_env()
        env.reset(seed=42)
        env.set_fixed_protocol("household_solo", {"fixed_task": "collect_red"})
        proto = env.current_protocol
        self.assertIsInstance(proto, HouseholdSoloProtocol)
        self.assertFalse(proto.has_target)
        self.assertIsNone(proto.target_object)
        self.assertEqual(float(proto.inspection_params["inspect_item_prob"]), 0.0)
        self.assertEqual(float(proto.inspection_params["inspect_board_prob"]), 0.0)
        self.assertEqual(float(proto.inspection_params["hesitate_prob"]), 0.0)
        self.assertNotIn(
            proto.state,
            {"inspecting_item", "hesitating", "inspecting_board"},
        )
        env.close()

    def test_start_leaves_target_object_none(self):
        env = _make_wrapped_env()
        env.reset(seed=42)
        env.set_fixed_protocol("household_solo", {"fixed_task": "collect_red"})
        proto = env.current_protocol
        self.assertIsNone(proto.target_object)
        self.assertIsNotNone(proto.target_board_id)
        self.assertIsNotNone(proto.current_item_id)
        env.close()

    def test_household_finish_does_not_pollute_blocked_pairs(self):
        env = _make_wrapped_env()
        env.reset(seed=42)
        env.set_fixed_protocol("household_solo", {"fixed_task": "collect_red"})
        before = set(env.blocked_pairs)
        _, finished = _run_until_finished(env, max_steps=2500)
        self.assertTrue(finished, "household_solo should complete without inspection")
        self.assertEqual(set(env.blocked_pairs), before)
        env.close()


if __name__ == "__main__":
    unittest.main()
