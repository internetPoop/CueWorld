"""幼児観測拡充（sleepiness / joint_angles / tactile / holding）のテスト。"""

import math
import unittest

import numpy as np

from cueworld.base import BaseEnv, _INFANT_JOINT_OBS_SLOTS_FULL
from cueworld.objects.items import (
    BlueCone,
    BlueCube,
    BlueCylinder,
    BlueSphere,
    GreenCone,
    GreenCube,
    GreenCylinder,
    GreenSphere,
    RedCone,
    RedCube,
    RedCylinder,
    RedSphere,
    YellowCone,
    YellowCube,
    YellowCylinder,
    YellowSphere,
    tactile_vec_for_concept,
)


def _make_base(**kwargs):
  defaults = dict(
      gui=False,
      num_agents=2,
      num_objects=1,
      object_counts={"RedCube": 1},
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


def _get_obs(base: BaseEnv, is_terminal: bool = False):
    return base._get_obs(
        parent_voice_on=base.parent_voice,
        infant_voice_on=base.infant_voice,
        is_terminal=is_terminal,
    )


PRIMITIVE_CLASSES = [
    RedCube, RedSphere, RedCylinder, RedCone,
    BlueCube, BlueSphere, BlueCylinder, BlueCone,
    GreenCube, GreenSphere, GreenCylinder, GreenCone,
    YellowCube, YellowSphere, YellowCylinder, YellowCone,
]


class TestInfantObsSpace(unittest.TestCase):
    def test_keys_present_and_shapes(self):
        base = _make_base()
        try:
            obs_space = base.observation_space[0]
            for key in ("sleepiness", "joint_angles", "tactile", "holding", "fatigue", "fatigue_lock"):
                self.assertIn(key, obs_space.spaces)
            self.assertEqual(obs_space["sleepiness"].shape, (1,))
            # enable_emotion=False（既定）: lip 無し → joint 12、fatigue 9
            self.assertEqual(obs_space["joint_angles"].shape, (12,))
            self.assertEqual(obs_space["tactile"].shape, (2,))
            self.assertEqual(obs_space["holding"].shape, (1,))
            self.assertEqual(obs_space["fatigue"].shape, (9,))
            self.assertEqual(obs_space["fatigue_lock"].shape, (9,))
            self.assertFalse(hasattr(base, "hunger"))

            obs, _ = base.reset(seed=0)
            infant_obs = obs[0]
            for key in ("sleepiness", "joint_angles", "tactile", "holding", "fatigue", "fatigue_lock"):
                self.assertIn(key, infant_obs)
            self.assertEqual(infant_obs["sleepiness"].shape, (1,))
            self.assertEqual(infant_obs["joint_angles"].shape, (12,))
            self.assertEqual(infant_obs["tactile"].shape, (2,))
            self.assertEqual(infant_obs["holding"].shape, (1,))
            self.assertEqual(infant_obs["fatigue"].shape, (9,))
            self.assertEqual(infant_obs["fatigue_lock"].shape, (9,))
            self.assertTrue(np.allclose(infant_obs["fatigue"], 0.0))
            self.assertTrue(np.allclose(infant_obs["fatigue_lock"], 0.0))
        finally:
            base.close()

    def test_infant_action_names_reduces_joint_dim(self):
        base = _make_base(infant_action_names=["head_tilt_joint", "torso_joint"])
        try:
            self.assertEqual(base.observation_space[0]["joint_angles"].shape, (3,))
            obs, _ = base.reset(seed=0)
            self.assertEqual(obs[0]["joint_angles"].shape, (3,))
        finally:
            base.close()


class TestSleepinessObs(unittest.TestCase):
    def test_sleepiness_progress_and_terminal(self):
        base = _make_base()
        try:
            base.sleepiness_steps = 5
            obs, _ = base.reset(seed=0)
            self.assertEqual(obs[0]["sleepiness"][0], 0.0)

            for step in range(1, 5):
                obs, _, done, _, _ = base.step(_zero_actions(base))
                self.assertAlmostEqual(obs[0]["sleepiness"][0], step / 5.0)
                self.assertFalse(done)

            obs, _, done, _, _ = base.step(_zero_actions(base))
            self.assertAlmostEqual(obs[0]["sleepiness"][0], 1.0)
            self.assertTrue(done)
            self.assertTrue(obs[0]["is_terminal"])
        finally:
            base.close()


class TestTactileVecCatalog(unittest.TestCase):
    def test_sixteen_distinct_unit_vectors(self):
        vectors = []
        for cls in PRIMITIVE_CLASSES:
            obj = cls()
            vec = np.array(obj.tactile_vec, dtype=np.float64)
            self.assertAlmostEqual(np.linalg.norm(vec), 1.0, places=5)
            vectors.append(vec)
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                self.assertGreater(np.linalg.norm(vectors[i] - vectors[j]), 1e-3)

    def test_same_shape_closer_than_different_shape(self):
        red_cube = np.array(RedCube().tactile_vec)
        blue_cube = np.array(BlueCube().tactile_vec)
        red_sphere = np.array(RedSphere().tactile_vec)

        def angle_dist(a, b):
            dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
            return math.acos(dot)

        same_shape = angle_dist(red_cube, blue_cube)
        diff_shape = angle_dist(red_cube, red_sphere)
        self.assertLess(same_shape, diff_shape)

    def test_matches_helper(self):
        obj = GreenCylinder()
        expected = tactile_vec_for_concept("green", "cylinder")
        self.assertEqual(obj.tactile_vec, expected)


class TestTactileHoldingObs(unittest.TestCase):
    def test_zero_when_not_touching_or_holding(self):
        base = _make_base()
        try:
            base.reset(seed=0)
            obs = _get_obs(base)
            np.testing.assert_array_equal(obs["tactile"], [0.0, 0.0])
            self.assertEqual(obs["holding"][0], 0.0)
        finally:
            base.close()

    def test_tactile_from_graspable_obj(self):
        base = _make_base()
        try:
            base.reset(seed=0)
            oid = next(iter(base.active_objects))
            expected = np.array(base.all_objects[oid].tactile_vec, dtype=np.float32)
            base.infant["graspable_obj"] = oid
            obs = _get_obs(base)
            np.testing.assert_allclose(obs["tactile"], expected)
            self.assertEqual(obs["holding"][0], 0.0)
        finally:
            base.close()

    def test_holding_flag(self):
        base = _make_base()
        try:
            base.reset(seed=0)
            oid = next(iter(base.active_objects))
            base.infant["held_obj"] = oid
            base.infant["graspable_obj"] = oid
            obs = _get_obs(base)
            self.assertEqual(obs["holding"][0], 1.0)
        finally:
            base.close()


class TestVoiceVolumeObs(unittest.TestCase):
    def test_volume_state_clips_at_zero(self):
        base = _make_base()
        try:
            base.reset(seed=0)
            vol_idx = base.voice_mapping["volume"]
            base.infant_voice[vol_idx] = 0.2
            infant_action = np.zeros(base.action_num, dtype=np.float32)
            infant_action[base.action_map["volume"]] = -1.0
            for _ in range(10):
                base.step((infant_action, _zero_actions(base)[1]))
            self.assertGreaterEqual(base.infant_voice[vol_idx], 0.0)
            self.assertLess(base.infant_voice[vol_idx], 0.25)
        finally:
            base.close()

    def test_voice_threshold_boundary(self):
        base = _make_base()
        try:
            base.reset(seed=0)
            vol_idx = base.voice_mapping["volume"]
            base.infant_voice.fill(0.0)
            base.parent_voice.fill(0.0)
            base.infant_voice[vol_idx] = 0.24
            base.parent_voice[vol_idx] = 0.24
            obs = _get_obs(base)
            np.testing.assert_array_equal(obs["infant_voice"], np.zeros(base.voice_dim))
            np.testing.assert_array_equal(obs["parent_voice"], np.zeros(base.voice_dim))
            np.testing.assert_array_equal(obs["parent_position"], [0.0, 0.0])

            base.infant_voice[vol_idx] = 0.25
            base.parent_voice[vol_idx] = 0.25
            base.parent_voice[vol_idx + 1] = 0.5
            obs = _get_obs(base)
            self.assertGreater(obs["infant_voice"][vol_idx], 0.0)
            self.assertGreater(obs["parent_voice"][vol_idx], 0.0)
            self.assertNotEqual(obs["parent_position"][0], 0.0)
        finally:
            base.close()

    def test_custom_voice_threshold(self):
        base = _make_base(voice_volume_threshold=0.5)
        try:
            base.reset(seed=0)
            vol_idx = base.voice_mapping["volume"]
            base.infant_voice.fill(0.0)
            base.infant_voice[vol_idx] = 0.4
            obs = _get_obs(base)
            np.testing.assert_array_equal(obs["infant_voice"], np.zeros(base.voice_dim))

            base.infant_voice[vol_idx] = 0.5
            obs = _get_obs(base)
            self.assertGreater(obs["infant_voice"][vol_idx], 0.0)
        finally:
            base.close()

    def test_observation_space_volume_low_is_zero(self):
        base = _make_base()
        try:
            vol_idx = base.voice_mapping["volume"]
            obs_space = base.observation_space[0]
            self.assertEqual(obs_space["infant_voice"].low[vol_idx], 0.0)
            self.assertEqual(obs_space["parent_voice"].low[vol_idx], 0.0)
        finally:
            base.close()


class TestObjectRestZ(unittest.TestCase):
    def test_grid_placement_uses_bbox_half_z(self):
        base = _make_base(
            object_counts={"RedCube": 1, "RedCone": 1},
            num_objects=2,
        )
        try:
            base.reset(seed=0)
            z_by_shape = {}
            import pybullet as p
            for oid in base.active_objects:
                obj = base.all_objects[oid]
                pos, _ = p.getBasePositionAndOrientation(
                    oid, physicsClientId=base.client_id
                )
                z_by_shape[obj.shape_type] = pos[2]
            self.assertAlmostEqual(z_by_shape["cube"], 0.06, places=4)
            self.assertAlmostEqual(z_by_shape["cone"], 0.065, places=4)
        finally:
            base.close()

    def test_fall_clamps_to_rest_z(self):
        base = _make_base(fall_per_step=0.22)
        try:
            base.reset(seed=0)
            oid = next(iter(base.active_objects))
            obj = base.all_objects[oid]
            rest_z = base._object_rest_z(obj)
            import pybullet as p
            p.resetBasePositionAndOrientation(
                oid, [0.0, 0.0, 0.5], [0, 0, 0, 1], physicsClientId=base.client_id
            )
            base._apply_simple_physics()
            pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=base.client_id)
            self.assertGreaterEqual(pos[2], rest_z - 1e-6)
            self.assertLessEqual(pos[2], 0.5)
        finally:
            base.close()

    def test_fall_lands_within_three_steps_for_giving_drop(self):
        base = _make_base(fall_per_step=0.05, object_counts={"RedCube": 1})
        try:
            base.reset(seed=0)
            oid = next(iter(base.active_objects))
            obj = base.all_objects[oid]
            rest_z = base._object_rest_z(obj)
            import pybullet as p
            drop_height = rest_z + 0.09
            p.resetBasePositionAndOrientation(
                oid, [0.0, 0.0, drop_height], [0, 0, 0, 1], physicsClientId=base.client_id
            )
            steps = 0
            while steps < 3:
                base._apply_simple_physics()
                steps += 1
                pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=base.client_id)
                if pos[2] <= rest_z + 1e-6:
                    break
            pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=base.client_id)
            self.assertLessEqual(pos[2], rest_z + 1e-6)
            self.assertLessEqual(steps, 3)
        finally:
            base.close()


class TestJointAnglesObs(unittest.TestCase):
    def test_full_twelve_dims_and_wrapped_unit_circle(self):
        # enable_emotion=False（既定）: lip 無しの 12 次元
        base = _make_base()
        try:
            base.reset(seed=0)
            obs = _get_obs(base)
            self.assertEqual(obs["joint_angles"].shape, (12,))
            # torso: indices 0,1 ; roll_wrist: indices 7,8
            torso_sin, torso_cos = obs["joint_angles"][0], obs["joint_angles"][1]
            wrist_sin, wrist_cos = obs["joint_angles"][7], obs["joint_angles"][8]
            self.assertAlmostEqual(torso_sin ** 2 + torso_cos ** 2, 1.0, places=4)
            self.assertAlmostEqual(wrist_sin ** 2 + wrist_cos ** 2, 1.0, places=4)
            # reset 直後は関節 0 rad → (0, 1)
            self.assertAlmostEqual(torso_sin, 0.0, places=4)
            self.assertAlmostEqual(torso_cos, 1.0, places=4)
        finally:
            base.close()

    def test_slot_order_matches_contract(self):
        # 既定は lip を除いた列。フル列（lip 込み）は enable_emotion=True。
        base = _make_base()
        try:
            expected = tuple(s for s in _INFANT_JOINT_OBS_SLOTS_FULL if s != "lip")
            self.assertEqual(tuple(base._infant_joint_obs_slots), expected)
        finally:
            base.close()


if __name__ == "__main__":
    unittest.main()
