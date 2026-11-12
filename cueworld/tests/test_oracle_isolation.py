"""オラクル観測（oracle_isolation.md）のキー・形・無し値・並びの確認。"""

import unittest

import numpy as np
import pybullet as p

from cueworld.base import BaseEnv
from cueworld.objects.items import COLOR_SHAKE_EFFECT, SHAPE_SHAKE_SOUND


ORACLE_KEYS = ["oracle_objects", "oracle_parent", "oracle_cues"]
ABSENT_XY = (-2.0, -2.0)
ABSENT_DIST = -1.0


def _make_base(**kwargs):
    defaults = dict(
        gui=False,
        num_agents=2,
        num_objects=2,
        object_counts={"RedCube": 1, "BlueCube": 1},
        max_steps=500,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        placement_method="grid",
    )
    defaults.update(kwargs)
    return BaseEnv(**defaults)


def _zero_actions(base: BaseEnv):
    return (
        np.zeros(base.action_space[0].shape, dtype=np.float32),
        np.zeros(base.action_space[1].shape, dtype=np.float32),
    )


def _face_away(base: BaseEnv):
    """幼児を親・物から背けて、カメラ前方に何も写らない向きにする。"""
    # torso を大きく回して後ろ向きにする（キネマティック）
    torso = base.infant.get("torso_joint")
    if torso is not None:
        p.resetJointState(
            base.infant_body, torso, np.pi, physicsClientId=base.client_id
        )


def _look_at_world_point(base: BaseEnv, target_xyz):
    """幼児の腰と首を、カメラが target を捉える向きにする。"""
    eye, _ = base._agent_eye_pose(base.infant)
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target_xyz, dtype=np.float32)
    v = target - eye
    yaw = float(np.arctan2(v[1], v[0]))
    horiz = float(np.sqrt(v[0] ** 2 + v[1] ** 2))
    # head_tilt: 下を見るとき正（URDF）。仰角 atan2(vz, horiz) の符号を反転
    tilt = float(-np.arctan2(v[2], max(horiz, 1e-6)))
    torso = base.infant.get("torso_joint")
    pan = base.infant.get("head_pan_joint")
    tilt_j = base.infant.get("head_tilt_joint")
    if torso is not None:
        p.resetJointState(base.infant_body, torso, yaw, physicsClientId=base.client_id)
    if pan is not None:
        p.resetJointState(base.infant_body, pan, 0.0, physicsClientId=base.client_id)
    if tilt_j is not None:
        p.resetJointState(
            base.infant_body, tilt_j, float(np.clip(tilt, -1.2, 1.2)),
            physicsClientId=base.client_id,
        )


class TestOracleKeysAbsentByDefault(unittest.TestCase):
    def test_default_obs_has_no_oracle_keys(self):
        base = _make_base()
        try:
            spaces = base.observation_space[0].spaces
            for k in ORACLE_KEYS:
                self.assertNotIn(k, spaces)
            obs, _ = base.reset(seed=0)
            for k in ORACLE_KEYS:
                self.assertNotIn(k, obs[0])
            obs2, _, _, _, _ = base.step(_zero_actions(base))
            for k in ORACLE_KEYS:
                self.assertNotIn(k, obs2[0])
        finally:
            base.close()


class TestOracleShapes(unittest.TestCase):
    def test_keys_shapes_and_slot_count(self):
        keys = ORACLE_KEYS + ["joint_angles"]
        object_counts = {"RedCube": 1, "BlueCube": 1}
        base = _make_base(
            infant_obs_keys=keys,
            object_counts=object_counts,
            num_objects=2,
        )
        try:
            expected_slots = sum(object_counts.values())
            self.assertEqual(base._oracle_slot_count, expected_slots)

            spaces = base.observation_space[0].spaces
            self.assertEqual(set(spaces.keys()), set(keys))
            self.assertEqual(spaces["oracle_objects"].shape, (expected_slots, 7))
            self.assertEqual(spaces["oracle_parent"].shape, (6,))
            self.assertEqual(spaces["oracle_cues"].shape, (7,))

            obs, _ = base.reset(seed=0)
            self.assertEqual(set(obs[0].keys()), set(keys))
            self.assertEqual(obs[0]["oracle_objects"].shape, (expected_slots, 7))
            self.assertEqual(obs[0]["oracle_parent"].shape, (6,))
            self.assertEqual(obs[0]["oracle_cues"].shape, (7,))
            self.assertEqual(obs[0]["oracle_objects"].dtype, np.float32)
        finally:
            base.close()

    def test_slot_count_includes_fixed_placements(self):
        object_counts = {"RedCube": 1}
        fixed = [
            {"class": "RedBoard", "x": 0.0, "y": -1.0, "yaw": 0.0},
        ]
        keys = ["oracle_objects"]
        base = _make_base(
            infant_obs_keys=keys,
            object_counts=object_counts,
            num_objects=1,
            fixed_placements=fixed,
            arena_half_extents=(2.0, 2.0),
            arena_rows=4,
            arena_cols=4,
        )
        try:
            self.assertEqual(base._oracle_slot_count, 2)
            self.assertEqual(
                base.observation_space[0]["oracle_objects"].shape, (2, 7)
            )
            obs, _ = base.reset(seed=0)
            self.assertEqual(obs[0]["oracle_objects"].shape, (2, 7))
        finally:
            base.close()


class TestOracleAbsentValues(unittest.TestCase):
    def test_objects_absent_when_facing_away(self):
        keys = ["oracle_objects"]
        base = _make_base(infant_obs_keys=keys, object_counts={"RedCube": 2}, num_objects=2)
        try:
            obs, _ = base.reset(seed=0)
            _face_away(base)
            obs2, _, _, _, _ = base.step(_zero_actions(base))
            objs = obs2[0]["oracle_objects"]
            for row in objs:
                np.testing.assert_allclose(row[0], ABSENT_XY[0])
                np.testing.assert_allclose(row[1], ABSENT_XY[1])
                np.testing.assert_allclose(row[2], ABSENT_DIST)
                np.testing.assert_allclose(row[3:], 0.0)
        finally:
            base.close()

    def test_parent_absent_when_invisible(self):
        keys = ["oracle_parent", "oracle_cues"]
        base = _make_base(
            infant_obs_keys=keys,
            parent_invisible_mode=True,
            object_counts={"RedCube": 1},
            num_objects=1,
        )
        try:
            obs, _ = base.reset(seed=0)
            parent = obs[0]["oracle_parent"]
            np.testing.assert_allclose(parent[0], ABSENT_XY[0])
            np.testing.assert_allclose(parent[1], ABSENT_XY[1])
            np.testing.assert_allclose(parent[2], ABSENT_DIST)
            np.testing.assert_allclose(parent[3], ABSENT_XY[0])
            np.testing.assert_allclose(parent[4], ABSENT_XY[1])
            np.testing.assert_allclose(parent[5], ABSENT_DIST)
            cues = obs[0]["oracle_cues"]
            np.testing.assert_allclose(cues[0], ABSENT_XY[0])
            np.testing.assert_allclose(cues[1], ABSENT_XY[1])
            np.testing.assert_allclose(cues[6], 0.0)
        finally:
            base.close()


class TestOracleVisibleContent(unittest.TestCase):
    def test_distance_positive_when_visible(self):
        keys = ["oracle_objects"]
        base = _make_base(
            infant_obs_keys=keys,
            object_counts={"RedCube": 1},
            num_objects=1,
            layout_freeze="full",
        )
        try:
            base.reset(seed=0)
            oid = base.active_objects[0]
            _look_at_world_point(base, base._obj_pos(oid))
            obs, _, _, _, _ = base.step(_zero_actions(base))
            objs = obs[0]["oracle_objects"]
            visible_rows = [
                row for row in objs if not np.isclose(row[2], ABSENT_DIST)
            ]
            self.assertGreaterEqual(len(visible_rows), 1)
            for row in visible_rows:
                self.assertGreater(float(row[2]), 0.0)
                self.assertLessEqual(float(row[2]), float(base.view_far) + 1.0)
        finally:
            base.close()

    def test_left_to_right_order(self):
        keys = ["oracle_objects"]
        base = _make_base(
            infant_obs_keys=keys,
            object_counts={"RedCube": 1, "BlueCube": 1},
            num_objects=2,
            layout_freeze="full",
        )
        try:
            base.reset(seed=0)
            centers = [base._obj_pos(oid) for oid in base.active_objects]
            mid = np.mean(np.stack(centers, axis=0), axis=0)
            _look_at_world_point(base, mid)
            obs, _, _, _, _ = base.step(_zero_actions(base))
            objs = obs[0]["oracle_objects"]
            visible = [
                row for row in objs if not np.isclose(row[2], ABSENT_DIST)
            ]
            if len(visible) < 2:
                self.skipTest("両方が同時に写らなかった（配置依存）")
            # 枠 0 が枠 1 より左（画像 x が小さい）
            self.assertLessEqual(float(visible[0][0]), float(visible[1][0]) + 1e-5)
        finally:
            base.close()

    def test_color_shape_match_shake_vectors(self):
        keys = ["oracle_objects"]
        base = _make_base(
            infant_obs_keys=keys,
            object_counts={"RedCube": 1},
            num_objects=1,
            layout_freeze="full",
        )
        try:
            base.reset(seed=0)
            oid = base.active_objects[0]
            obj = base.all_objects[oid]
            _look_at_world_point(base, base._obj_pos(oid))
            obs, _, _, _, _ = base.step(_zero_actions(base))
            objs = obs[0]["oracle_objects"]
            visible = [
                row for row in objs if not np.isclose(row[2], ABSENT_DIST)
            ]
            self.assertGreaterEqual(len(visible), 1)
            row = visible[0]
            expected_c = COLOR_SHAKE_EFFECT[obj.color]
            expected_s = SHAPE_SHAKE_SOUND[obj.shape_type]
            np.testing.assert_allclose(row[3:5], expected_c, atol=1e-5)
            np.testing.assert_allclose(row[5:7], expected_s, atol=1e-5)
        finally:
            base.close()


class TestWorldToNdc(unittest.TestCase):
    def test_behind_camera_returns_absent(self):
        # 単位カメラ: 原点、+Z 前方、+Y 上（簡易）は使わず、PyBullet 行列で背後点を確認
        eye = [0.0, 0.0, 0.0]
        target = [1.0, 0.0, 0.0]
        up = [0.0, 0.0, 1.0]
        vm = p.computeViewMatrix(eye, target, up)
        pm = p.computeProjectionMatrixFOV(
            fov=90.0, aspect=1.0, nearVal=0.01, farVal=5.0
        )
        behind = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        xy = BaseEnv._world_to_ndc(behind, vm, pm)
        self.assertEqual(xy, ABSENT_XY)

    def test_ahead_is_finite(self):
        eye = [0.0, 0.0, 0.0]
        target = [1.0, 0.0, 0.0]
        up = [0.0, 0.0, 1.0]
        vm = p.computeViewMatrix(eye, target, up)
        pm = p.computeProjectionMatrixFOV(
            fov=90.0, aspect=1.0, nearVal=0.01, farVal=5.0
        )
        ahead = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        xy = BaseEnv._world_to_ndc(ahead, vm, pm)
        self.assertNotEqual(xy, ABSENT_XY)
        self.assertTrue(np.isfinite(xy[0]) and np.isfinite(xy[1]))


if __name__ == "__main__":
    unittest.main()
