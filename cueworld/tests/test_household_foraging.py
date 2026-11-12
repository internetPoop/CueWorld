"""HouseholdSoloProtocol Phase 3（foraging 的サブゴール選択）のテスト。"""

import random
import unittest
from unittest.mock import MagicMock

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
        max_steps=100,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        arena_rows=4,
        arena_cols=5,
        fixed_placements=fixed,
        placement_method="grid",
        use_cinematic_model=True,
    )
    return ProtocolManager(InfantSignalDetector(ParentTools(base)), demo_mode=False)


class TestHouseholdForaging(unittest.TestCase):
    def test_select_next_item_uses_softmax_not_always_nearest(self):
        """確率選択により、近くない方も選ばれうる。"""
        proto = HouseholdSoloProtocol(MagicMock())
        proto.env.parent = {"body": 0}
        distances = {1: 0.5, 2: 2.0}
        proto.env._dist_body_to = lambda agent, pos: distances.get(pos[0], 1.0)
        proto.env._obj_pos = lambda oid: [oid, 0.0, 0.0]
        proto.env.all_objects = {
            1: MagicMock(name="RedCube", __class__=type("RedCube", (), {})),
            2: MagicMock(name="RedSphere", __class__=type("RedSphere", (), {})),
        }
        proto.env.all_objects[1].name = "RedCube"
        proto.env.all_objects[2].name = "RedSphere"

        rng = random.Random(0)
        picks = []
        for _ in range(100):
            chosen, reason = proto.select_next_item([1, 2], "committed", rng=rng)
            picks.append(chosen)
            self.assertIn(chosen, (1, 2))
            self.assertEqual(reason["style"], "committed")
            self.assertEqual(len(reason["probabilities"]), 2)

        self.assertIn(1, picks)
        self.assertIn(2, picks)

    def test_opportunistic_style_favors_near_item_more(self):
        proto = HouseholdSoloProtocol(MagicMock())
        proto.env.parent = {"body": 0}
        proto.env._dist_body_to = lambda agent, pos: pos[0]
        proto.env._obj_pos = lambda oid: [oid, 0.0, 0.0]
        proto.env.all_objects = {
            1: MagicMock(name="Near"),
            2: MagicMock(name="Far"),
        }
        proto.env.all_objects[1].name = "Near"
        proto.env.all_objects[1].__class__.__name__ = "RedCube"
        proto.env.all_objects[2].name = "Far"
        proto.env.all_objects[2].__class__.__name__ = "RedSphere"

        rng = random.Random(1)
        committed_near = 0
        opportunistic_near = 0
        for _ in range(200):
            if proto.select_next_item([1, 2], "committed", rng=rng)[0] == 1:
                committed_near += 1
        rng = random.Random(1)
        for _ in range(200):
            if proto.select_next_item([1, 2], "opportunistic", rng=rng)[0] == 1:
                opportunistic_near += 1
        self.assertGreater(opportunistic_near, committed_near)

    def test_first_pick_varies_across_runs_same_seed(self):
        """同一シード・同一配置でも、複数回の開始で選ばれる item が揺れる。"""
        first_picks = set()
        intents = set()
        for _ in range(50):
            env = _make_wrapped_env()
            env.reset(seed=42)
            env.set_fixed_protocol("household_solo", {"fixed_task": "collect_red"})
            proto = env.current_protocol
            self.assertIsInstance(proto, HouseholdSoloProtocol)
            intents.add(proto.global_intent)
            first_picks.add(proto.current_item_id)
            self.assertEqual(len(proto.item_selection_history), 1)
            env.close()

        self.assertEqual(intents, {"collect_red"})
        self.assertGreater(len(first_picks), 1)

    def test_enter_moving_to_item_reselects_on_grasp_retry(self):
        env = _make_wrapped_env()
        env.reset(seed=42)
        env.set_fixed_protocol(
            "household_solo",
            {
                "fixed_task": "collect_red",
                "inspect_item_prob": 0.0,
                "hesitate_prob": 0.0,
                "inspect_board_prob": 0.0,
            },
        )
        proto = env.current_protocol
        first_pick = proto.current_item_id
        proto.state = "grasping_item"
        proto.state_timer = 101
        proto._manage_state({})
        self.assertEqual(proto.state, "moving_to_item")
        self.assertIsNotNone(proto.current_item_id)
        self.assertEqual(len(proto.item_selection_history), 2)
        env.close()


if __name__ == "__main__":
    unittest.main()
