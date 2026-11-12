"""実シーンでの円錐注意（視線対象ベクトル）のテスト。"""

import math
import unittest

import numpy as np
import pybullet as p

from cueworld.base import BaseEnv
from cueworld.infant_signal import CONE_MIN_ACCURACY, InfantSignalDetector
from cueworld.parent_tools import ParentTools


def _make_env(num_objects=2):
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=num_objects,
        object_counts={"RedCube": 1, "BlueCube": max(0, num_objects - 1)},
        max_steps=500,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        placement_method="grid",
        debug_info=False,
    )
    env = ParentTools(base)
    env.reset(seed=42)
    return base, env


def _eye_pose(base):
    return base._agent_eye_pose(base.infant)


def _place_object_at_eye_offset(base, oid, dist, angle_deg, z_offset=0.0):
    """目の位置・前方から水平角 angle_deg・距離 dist に物体を置く。"""
    eye_pos, fwd = _eye_pose(base)
    fwd_xy = np.array([fwd[0], fwd[1]], dtype=np.float64)
    norm = np.linalg.norm(fwd_xy)
    if norm < 1e-8:
        fwd_xy = np.array([1.0, 0.0], dtype=np.float64)
    else:
        fwd_xy /= norm
    perp = np.array([-fwd_xy[1], fwd_xy[0]], dtype=np.float64)
    rad = math.radians(float(angle_deg))
    dir_xy = math.cos(rad) * fwd_xy + math.sin(rad) * perp
    obj_pos = [
        float(eye_pos[0]) + float(dir_xy[0]) * dist,
        float(eye_pos[1]) + float(dir_xy[1]) * dist,
        float(eye_pos[2]) + z_offset,
    ]
    p.resetBasePositionAndOrientation(
        oid, obj_pos, [0, 0, 0, 1], physicsClientId=base.client_id
    )
    return obj_pos


def _gaze_vector(base):
    detector = InfantSignalDetector(base)
    return detector.get_gaze_target()


def _argmax_object_id(base, vec):
    if vec is None or len(vec) == 0:
        return None
    idx = int(np.argmax(vec))
    if float(vec[idx]) <= 0.0:
        return None
    targets = base.trackable_targets
    if idx < 0 or idx >= len(targets):
        return None
    return targets[idx]


class TestConeAttentionScene(unittest.TestCase):
    def setUp(self):
        self.base, self.env = _make_env(num_objects=2)
        self.oids = list(self.base.all_objects.keys())
        self.oid_a = self.oids[0]
        self.oid_b = self.oids[1] if len(self.oids) > 1 else None

    def tearDown(self):
        self.env.close()

    def test_on_axis_target_dominates(self):
        _place_object_at_eye_offset(self.base, self.oid_a, dist=0.8, angle_deg=0.0)
        vec, *_ = _gaze_vector(self.base)
        idx_a = self.base.trackable_targets.index(self.oid_a)
        self.assertGreater(float(vec[idx_a]), 0.5)
        self.assertEqual(_argmax_object_id(self.base, vec), self.oid_a)

    def test_distractor_outside_cone_ignored(self):
        _place_object_at_eye_offset(self.base, self.oid_a, dist=0.8, angle_deg=0.0)
        if self.oid_b is not None:
            _place_object_at_eye_offset(self.base, self.oid_b, dist=0.75, angle_deg=20.0)
        vec, *_ = _gaze_vector(self.base)
        idx_a = self.base.trackable_targets.index(self.oid_a)
        self.assertEqual(_argmax_object_id(self.base, vec), self.oid_a)
        if self.oid_b is not None:
            idx_b = self.base.trackable_targets.index(self.oid_b)
            self.assertAlmostEqual(float(vec[idx_b]), 0.0)

    def test_distractor_near_cone_edge_target_still_wins(self):
        _place_object_at_eye_offset(self.base, self.oid_a, dist=0.8, angle_deg=0.0)
        if self.oid_b is None:
            self.skipTest("need two objects")
        _place_object_at_eye_offset(self.base, self.oid_b, dist=0.75, angle_deg=9.0)
        vec, *_ = _gaze_vector(self.base)
        idx_a = self.base.trackable_targets.index(self.oid_a)
        idx_b = self.base.trackable_targets.index(self.oid_b)
        self.assertEqual(_argmax_object_id(self.base, vec), self.oid_a)
        self.assertGreater(float(vec[idx_a]), float(vec[idx_b]))
        if float(vec[idx_b]) > 0.0:
            self.assertGreaterEqual(float(vec[idx_a]), CONE_MIN_ACCURACY)

    def test_target_outside_cone_no_attention(self):
        _place_object_at_eye_offset(self.base, self.oid_a, dist=0.8, angle_deg=25.0)
        vec, *_ = _gaze_vector(self.base)
        idx_a = self.base.trackable_targets.index(self.oid_a)
        self.assertAlmostEqual(float(vec[idx_a]), 0.0)
        self.assertIsNone(_argmax_object_id(self.base, vec))

    def test_no_objects_in_view_near_zero_vector(self):
        # 視線前方に物を置かない（既存配置を遠ざける）
        for oid in self.oids:
            _place_object_at_eye_offset(self.base, oid, dist=0.5, angle_deg=40.0)
        vec, *_ = _gaze_vector(self.base)
        for oid in self.oids:
            idx = self.base.trackable_targets.index(oid)
            self.assertLessEqual(float(vec[idx]), CONE_MIN_ACCURACY)


if __name__ == "__main__":
    unittest.main()
