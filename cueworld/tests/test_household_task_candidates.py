"""HouseholdSoloProtocol のタスク候補（Phase 2 完了基準）テスト。"""

import unittest

import numpy as np

from cueworld.base import BaseEnv
from cueworld.parent_tools import ParentTools
from cueworld.infant_signal import InfantSignalDetector
from cueworld.protocol_manager import ProtocolManager
from cueworld.protocols.household_solo import HouseholdSoloProtocol


def _get_base_env(env):
    current = env
    while hasattr(current, "env"):
        if isinstance(current, BaseEnv):
            return current
        current = current.env
    return current


def _make_wrapped_env(num_objects, object_counts, fixed_placements):
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=num_objects,
        object_counts=object_counts,
        max_steps=100,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        arena_rows=4,
        arena_cols=5,
        fixed_placements=fixed_placements,
        placement_method="grid",
        use_cinematic_model=True,
    )
    return ProtocolManager(InfantSignalDetector(ParentTools(base)), demo_mode=False)


class TestHouseholdTaskCandidates(unittest.TestCase):
    YH = 1.2
    ALL_BOARDS = [
        {"class": "Red_Board", "x": -1.2, "y": YH - 0.05, "z": 0.5, "yaw": 90.0},
        {"class": "Blue_Board", "x": -0.4, "y": YH - 0.05, "z": 0.5, "yaw": 90.0},
        {"class": "Green_Board", "x": 0.4, "y": YH - 0.05, "z": 0.5, "yaw": 90.0},
        {"class": "Yellow_Board", "x": 1.2, "y": YH - 0.05, "z": 0.5, "yaw": 90.0},
    ]

    def _candidate_types(self, env):
        base = _get_base_env(env)
        return {c["task_type"] for c in base.household_task_candidates}

    def test_missing_item_or_board_excluded_from_candidates(self):
        """完了基準2: 対応 item / board が無い種別は候補に入らない。"""
        env = _make_wrapped_env(
            num_objects=4,
            object_counts={"RedCube": 1},
            fixed_placements=[self.ALL_BOARDS[0]],
        )
        env.reset(seed=0)
        types = self._candidate_types(env)
        self.assertEqual(types, {"collect_red"})
        env.close()

        env = _make_wrapped_env(
            num_objects=4,
            object_counts={"BlueSphere": 1},
            fixed_placements=[self.ALL_BOARDS[1]],
        )
        env.reset(seed=0)
        types = self._candidate_types(env)
        self.assertEqual(types, {"collect_blue"})
        env.close()

        env = _make_wrapped_env(
            num_objects=4,
            object_counts={"GreenCylinder": 1},
            fixed_placements=[self.ALL_BOARDS[2]],
        )
        env.reset(seed=0)
        types = self._candidate_types(env)
        self.assertEqual(types, {"collect_green"})
        env.close()

        # board と item のカテゴリ不一致 → 候補なし
        env = _make_wrapped_env(
            num_objects=4,
            object_counts={"RedCube": 1},
            fixed_placements=[self.ALL_BOARDS[2]],
        )
        env.reset(seed=0)
        types = self._candidate_types(env)
        self.assertEqual(types, set())
        env.close()

    def test_zero_candidates_protocol_completes_immediately(self):
        """完了基準3: 成立候補 0 件のときプロトコルが即終了する。"""
        env = _make_wrapped_env(
            num_objects=3,
            object_counts={},
            fixed_placements=self.ALL_BOARDS,
        )
        env.reset(seed=0)
        base = _get_base_env(env)
        self.assertEqual(base.household_task_candidates, [])

        env.set_fixed_protocol("household_solo", {})
        proto = env.current_protocol
        self.assertIsInstance(proto, HouseholdSoloProtocol)
        self.assertTrue(proto.is_finished)
        self.assertEqual(proto.state, "completed")
        self.assertIsNone(proto.global_intent)

        infant_action = np.zeros(base.action_space[0].shape, dtype=np.float32)
        parent_action = np.zeros(base.action_space[1].shape, dtype=np.float32)
        env.step((infant_action, parent_action))
        self.assertEqual(proto.state, "completed")
        self.assertTrue(proto.is_finished)
        env.close()

    def test_fixed_task_skipped_when_not_feasible(self):
        env = _make_wrapped_env(
            num_objects=4,
            object_counts={"RedCube": 1},
            fixed_placements=[self.ALL_BOARDS[0]],
        )
        env.reset(seed=0)
        env.set_fixed_protocol("household_solo", {"fixed_task": "collect_blue"})
        proto = env.current_protocol
        self.assertTrue(proto.is_finished)
        self.assertEqual(proto.state, "completed")
        env.close()


if __name__ == "__main__":
    unittest.main()
