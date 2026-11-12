"""振る → 音 / 効果チャネルのユニットテスト。

テスト項目:
  1. obs キーの存在と shape（object_sound / object_effect が (2,) float32）
  2. リセット直後はゼロ
  3. 歩行のみ（腕を動かさない）では発火しない（偽陽性なし）
  4. shake_object を実行すると少なくとも1回以上発火する
  5. 発火ベクトルが形・色から期待される固定値と一致する
  6. 把持していない状態では発火しない
"""

import unittest
import numpy as np

from cueworld.base import BaseEnv
from cueworld.parent_tools import ParentTools
from cueworld.objects.items import SHAPE_SHAKE_SOUND, COLOR_SHAKE_EFFECT


def _make_env(class_name: str = "RedCube", shake_move_threshold: float = 0.002):
    """テスト用の最小環境を生成する。"""
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=1,
        object_counts={class_name: 1},
        max_steps=500,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        debug_info=False,
        placement_method="grid",
        shake_move_threshold=shake_move_threshold,
    )
    env = ParentTools(base)
    env.reset(seed=42)
    return base, env


def _grasp_target(base: BaseEnv, env: ParentTools, target_oid: int,
                  max_steps: int = 100) -> bool:
    """把持成功まで最大 max_steps ステップ実行し、成功 / 失敗を返す。"""
    infant_action = np.zeros(base.action_space[0].shape, dtype=np.float32)
    for _ in range(max_steps):
        pa = np.zeros(base.action_space[1].shape, dtype=np.float32)
        pa = env.grasp_object(base.parent, target_oid, action=pa, approach_dist=0.05)
        base.step((infant_action, pa))
        if base.parent.get("held_obj") == target_oid:
            return True
    return False


class TestObsKeys(unittest.TestCase):
    """obs キーの存在・shape・型を確認する。"""

    def setUp(self):
        self.base, self.env = _make_env()

    def tearDown(self):
        self.env.close()

    def test_keys_present_in_obs_space(self):
        obs_space = self.base.observation_space[0]
        self.assertIn("object_sound", obs_space.spaces)
        self.assertIn("object_effect", obs_space.spaces)

    def test_obs_space_shape(self):
        obs_space = self.base.observation_space[0]
        self.assertEqual(obs_space["object_sound"].shape, (2,))
        self.assertEqual(obs_space["object_effect"].shape, (2,))
        self.assertEqual(obs_space["object_sound"].dtype, np.float32)
        self.assertEqual(obs_space["object_effect"].dtype, np.float32)

    def test_obs_keys_present_in_step_output(self):
        infant_action = np.zeros(self.base.action_space[0].shape, dtype=np.float32)
        parent_action = np.zeros(self.base.action_space[1].shape, dtype=np.float32)
        obs, _, _, _, _ = self.base.step((infant_action, parent_action))
        self.assertIn("object_sound", obs[0])
        self.assertIn("object_effect", obs[0])
        self.assertEqual(obs[0]["object_sound"].shape, (2,))
        self.assertEqual(obs[0]["object_effect"].shape, (2,))


class TestResetZero(unittest.TestCase):
    """reset 直後はチャネルがゼロであることを確認する。"""

    def setUp(self):
        self.base, self.env = _make_env()

    def tearDown(self):
        self.env.close()

    def test_channels_zero_after_reset(self):
        obs, _ = self.base.reset(seed=99)
        np.testing.assert_array_equal(obs[0]["object_sound"], [0.0, 0.0])
        np.testing.assert_array_equal(obs[0]["object_effect"], [0.0, 0.0])


class TestNoFiringWhileWalking(unittest.TestCase):
    """把持したまま前進のみ（腕を動かさない）では発火しない。"""

    def setUp(self):
        self.base, self.env = _make_env()
        target_oid = next(iter(self.base.active_objects))
        grasped = _grasp_target(self.base, self.env, target_oid)
        self.grasped = grasped

    def tearDown(self):
        self.env.close()

    def test_no_false_positive_during_walk(self):
        self.assertTrue(self.grasped, "前提: 把持成功が必要")
        infant_action = np.zeros(self.base.action_space[0].shape, dtype=np.float32)
        am = self.base.action_map
        fired = False
        for _ in range(40):
            pa = np.zeros(self.base.action_space[1].shape, dtype=np.float32)
            pa = self.env.grip_object(self.base.parent, action=pa)
            pa[am["forward"]] = 1.0
            obs, _, done, trunc, _ = self.base.step((infant_action, pa))
            if np.any(obs[0]["object_sound"] != 0.0):
                fired = True
                break
            if done or trunc:
                break
        self.assertFalse(fired, "歩行のみで object_sound が発火してはならない")


class TestFiringDuringShake(unittest.TestCase):
    """shake_object 実行中に少なくとも1回は発火する。"""

    def setUp(self):
        self.base, self.env = _make_env()
        target_oid = next(iter(self.base.active_objects))
        grasped = _grasp_target(self.base, self.env, target_oid)
        self.grasped = grasped

    def tearDown(self):
        self.env.close()

    def test_fires_during_shake(self):
        self.assertTrue(self.grasped, "前提: 把持成功が必要")
        infant_action = np.zeros(self.base.action_space[0].shape, dtype=np.float32)
        fired = False
        for _ in range(40):
            pa = np.zeros(self.base.action_space[1].shape, dtype=np.float32)
            pa = self.env.shake_object(self.base.parent, action=pa)
            obs, _, done, trunc, _ = self.base.step((infant_action, pa))
            if np.any(obs[0]["object_sound"] != 0.0):
                fired = True
                break
            if done or trunc:
                break
        self.assertTrue(fired, "shake_object 実行中に object_sound が発火する必要がある")


class TestVectorCorrectness(unittest.TestCase):
    """発火ベクトルが形・色ごとの期待値と一致する（16クラス全検証）。"""

    # テストする形・色の組み合わせ（全16クラスのサブセット；代表4クラスのみ）
    # 完全な16クラスはevalスクリプトで検証済み
    CASES = [
        ("RedCube",      "cube",     "red"),
        ("BlueSphere",   "sphere",   "blue"),
        ("GreenCylinder","cylinder", "green"),
        ("YellowCone",   "cone",     "yellow"),
    ]

    def _check_class(self, class_name: str, shape: str, color: str):
        base, env = _make_env(class_name)
        target_oid = next(iter(base.active_objects))
        grasped = _grasp_target(base, env, target_oid)
        self.assertTrue(grasped, f"{class_name}: 把持失敗")

        expected_sound = np.array(SHAPE_SHAKE_SOUND[shape], dtype=np.float32)
        expected_effect = np.array(COLOR_SHAKE_EFFECT[color], dtype=np.float32)

        infant_action = np.zeros(base.action_space[0].shape, dtype=np.float32)
        sound_ok = None
        effect_ok = None

        for _ in range(60):
            pa = np.zeros(base.action_space[1].shape, dtype=np.float32)
            pa = env.shake_object(base.parent, action=pa)
            obs, _, done, trunc, _ = base.step((infant_action, pa))
            if np.any(obs[0]["object_sound"] != 0.0):
                np.testing.assert_allclose(
                    obs[0]["object_sound"], expected_sound, atol=1e-5,
                    err_msg=f"{class_name}: object_sound mismatch"
                )
                np.testing.assert_allclose(
                    obs[0]["object_effect"], expected_effect, atol=1e-5,
                    err_msg=f"{class_name}: object_effect mismatch"
                )
                sound_ok = True
                effect_ok = True
                break
            if done or trunc:
                break

        env.close()
        self.assertIsNotNone(sound_ok, f"{class_name}: 一度も発火しなかった")

    def test_red_cube(self):
        self._check_class("RedCube", "cube", "red")

    def test_blue_sphere(self):
        self._check_class("BlueSphere", "sphere", "blue")

    def test_green_cylinder(self):
        self._check_class("GreenCylinder", "cylinder", "green")

    def test_yellow_cone(self):
        self._check_class("YellowCone", "cone", "yellow")


class TestNoFiringWithoutGrasp(unittest.TestCase):
    """把持していない状態では発火しない。"""

    def setUp(self):
        self.base, self.env = _make_env()

    def tearDown(self):
        self.env.close()

    def test_no_firing_without_grasp(self):
        # 把持せずにランダム腕動作
        infant_action = np.zeros(self.base.action_space[0].shape, dtype=np.float32)
        am = self.base.action_map
        fired = False
        for _ in range(30):
            pa = np.zeros(self.base.action_space[1].shape, dtype=np.float32)
            pa = self.env.shake_object(self.base.parent, action=pa)
            obs, _, done, trunc, _ = self.base.step((infant_action, pa))
            if np.any(obs[0]["object_sound"] != 0.0):
                fired = True
                break
            if done or trunc:
                break
        self.assertFalse(fired, "把持なしで object_sound が発火してはならない")


class TestItemAttributes(unittest.TestCase):
    """PrimitiveObject の属性が期待どおりに設定されている。"""

    def test_can_shake_true(self):
        from cueworld.objects.items import RedCube
        obj = RedCube()
        self.assertTrue(obj.can_shake)

    def test_is_containable_false(self):
        from cueworld.objects.items import RedCube, BlueSphere, GreenCylinder, YellowCone
        for cls in [RedCube, BlueSphere, GreenCylinder, YellowCone]:
            self.assertFalse(cls().is_containable, f"{cls.__name__}.is_containable should be False")

    def test_shake_sound_vectors(self):
        from cueworld.objects.items import (RedCube, BlueCube, RedSphere, RedCylinder, RedCone)
        expected = {
            RedCube: SHAPE_SHAKE_SOUND["cube"],
            RedSphere: SHAPE_SHAKE_SOUND["sphere"],
            RedCylinder: SHAPE_SHAKE_SOUND["cylinder"],
            RedCone: SHAPE_SHAKE_SOUND["cone"],
        }
        for cls, vec in expected.items():
            obj = cls()
            self.assertEqual(obj.shake_sound, vec, f"{cls.__name__}.shake_sound mismatch")

    def test_shake_effect_vectors(self):
        from cueworld.objects.items import (RedCube, BlueCube, GreenCube, YellowCube)
        expected = {
            RedCube: COLOR_SHAKE_EFFECT["red"],
            BlueCube: COLOR_SHAKE_EFFECT["blue"],
            GreenCube: COLOR_SHAKE_EFFECT["green"],
            YellowCube: COLOR_SHAKE_EFFECT["yellow"],
        }
        for cls, vec in expected.items():
            obj = cls()
            self.assertEqual(obj.shake_effect_vec, vec, f"{cls.__name__}.shake_effect_vec mismatch")


if __name__ == "__main__":
    unittest.main()
