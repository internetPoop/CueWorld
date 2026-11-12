"""graspable_obj / touching 正本（base → infant_signal）のテスト。"""

import unittest

import numpy as np

from cueworld.base import BaseEnv
from cueworld.parent_tools import ParentTools
from cueworld.infant_signal import InfantSignalDetector


def _make_base(object_counts=None, fixed_placements=None, num_objects=1):
    return BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=num_objects,
        object_counts=object_counts if object_counts is not None else {"RedCube": 1},
        max_steps=500,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        placement_method="grid",
        fixed_placements=fixed_placements or [],
    )


def _make_signal_env(object_counts=None, fixed_placements=None, num_objects=1):
    base = _make_base(object_counts, fixed_placements, num_objects)
    return base, InfantSignalDetector(ParentTools(base))


class TestGraspDistances(unittest.TestCase):
    """Board の bbox が grasp_distance を膨らませない。"""

    BOARD = {"class": "Red_Board", "x": 0.0, "y": 1.15, "z": 0.5, "yaw": 90.0}

    def test_toy_only_grasp_distance(self):
        base = _make_base(object_counts={"RedCube": 1})
        base.reset(seed=0)
        try:
            self.assertLess(base.grasp_distance, 0.35)
            self.assertGreater(base.grasp_contact_distance, 0.0)
            self.assertAlmostEqual(
                base.grasp_distance,
                base.grasp_contact_distance + base.grasp_margin,
                places=5,
            )
        finally:
            base.close()

    def test_board_does_not_inflate_grasp_distance(self):
        base = _make_base(
            object_counts={"RedCube": 1},
            fixed_placements=[self.BOARD],
            num_objects=2,
        )
        base.reset(seed=0)
        try:
            self.assertLess(base.grasp_distance, 0.35)
            board_ids = [
                bid
                for bid, obj in base.all_objects.items()
                if getattr(obj, "is_platform", False)
            ]
            self.assertTrue(board_ids)
            self.assertFalse(base.all_objects[board_ids[0]].is_graspable)
        finally:
            base.close()


class TestGraspableAgentKeys(unittest.TestCase):
    def test_reset_clears_graspable_keys(self):
        base = _make_base()
        base.reset(seed=1)
        try:
            self.assertIsNone(base.infant["graspable_obj"])
            self.assertIsNone(base.infant["graspable_link"])
            self.assertIsNone(base.parent["graspable_obj"])
            self.assertIsNone(base.parent["graspable_link"])
        finally:
            base.close()

    def test_board_not_graspable_candidate(self):
        base = _make_base(
            object_counts={"RedCube": 1},
            fixed_placements=[TestGraspDistances.BOARD],
            num_objects=2,
        )
        base.reset(seed=0)
        try:
            board_id = next(
                bid
                for bid, obj in base.all_objects.items()
                if getattr(obj, "is_platform", False)
            )
            infant_action = np.zeros(base.action_space[0].shape, dtype=np.float32)
            parent_action = np.zeros(base.action_space[1].shape, dtype=np.float32)
            for _ in range(5):
                base.step((infant_action, parent_action))
            for agent in (base.infant, base.parent):
                oid = agent.get("graspable_obj")
                if oid is not None:
                    self.assertNotEqual(oid, board_id)
                    self.assertTrue(base.all_objects[oid].is_graspable)
        finally:
            base.close()


class TestInfantSignalHoldingTouching(unittest.TestCase):
    def test_holding_and_touching_after_parent_grasp(self):
        """親が掴んだあと graspable_obj / holding / touching が整合する。"""
        base, env = _make_signal_env()
        base.reset(seed=42)
        try:
            target_oid = next(iter(base.active_objects))
            infant_action = np.zeros(base.action_space[0].shape, dtype=np.float32)
            grasped = False
            for _ in range(120):
                pa = np.zeros(base.action_space[1].shape, dtype=np.float32)
                pa = env.env.grasp_object(
                    base.parent, target_oid, action=pa, approach_dist=0.05
                )
                _, _, _, _, info = env.step((infant_action, pa))
                if base.parent.get("held_obj") == target_oid:
                    grasped = True
                    break
            self.assertTrue(grasped, "前提: 親の把持成功が必要")
            self.assertEqual(base.parent.get("graspable_obj"), target_oid)
            self.assertEqual(base.parent.get("held_obj"), target_oid)
            # infant_signal は幼児のみ。親の graspable は base 側の整合確認。
            detector = env
            status = detector.get_hand_status(base.infant)
            grab_flag = status[2]
            touch_flag = status[4]
            self.assertFalse(grab_flag)
            # 幼児は物から離れている想定。近くに無ければ touching も 0。
            if base.infant.get("graspable_obj") is None:
                self.assertFalse(touch_flag)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
