"""選好報酬（InfantSignalDetector）のテスト。"""

import unittest
from unittest.mock import patch

import numpy as np

from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector


def _make_env(active_signals=None, **kwargs):
    defaults = dict(
        gui=False,
        num_agents=2,
        num_objects=1,
        object_counts={"RedCube": 1},
        max_steps=500,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        placement_method="grid",
        parent_preference=1.2,
    )
    defaults.update(kwargs)
    base = BaseEnv(**defaults)
    return InfantSignalDetector(base, active_signals=active_signals)


def _zero_actions(env):
    base = env.env
    return (
        np.zeros(base.action_space[0].shape, dtype=np.float32),
        np.zeros(base.action_space[1].shape, dtype=np.float32),
    )


class TestPreferenceReward(unittest.TestCase):
    def test_no_hold_no_look_is_zero(self):
        env = _make_env()
        try:
            env.reset(seed=0)
            reward, breakdown = env._compute_preference_reward()
            self.assertEqual(reward, 0.0)
            self.assertEqual(breakdown["total"], 0.0)
        finally:
            env.env.close()

    def test_hold_reward_uses_object_preference(self):
        env = _make_env()
        try:
            env.reset(seed=0)
            oid = next(iter(env.env.active_objects))
            env.env.infant["held_obj"] = oid
            reward, breakdown = env._compute_preference_reward()
            self.assertAlmostEqual(reward, 1.0)
            self.assertAlmostEqual(breakdown["hold"], 1.0)
            self.assertAlmostEqual(breakdown["look"], 0.0)
        finally:
            env.env.close()

    def test_look_reward_uses_parent_preference(self):
        env = _make_env()
        try:
            with patch.object(env, "get_gaze_target", return_value=(np.zeros(2), True, True)):
                env.reset(seed=0)
                env.update()
                reward, breakdown = env._compute_preference_reward()
            self.assertAlmostEqual(reward, 1.2)
            self.assertAlmostEqual(breakdown["look"], 1.2)
        finally:
            env.env.close()

    def test_hold_and_look_sum(self):
        env = _make_env()
        try:
            env.reset(seed=0)
            oid = next(iter(env.env.active_objects))
            env.env.infant["held_obj"] = oid
            env._instant_parent_eye_hit = True
            reward, breakdown = env._compute_preference_reward()
            self.assertAlmostEqual(reward, 2.2)
            self.assertAlmostEqual(breakdown["total"], 2.2)
        finally:
            env.env.close()

    def test_gaze_modality_off_zeros_even_when_holding(self):
        env = _make_env(active_signals=["point_intensity"])
        try:
            env.reset(seed=0)
            oid = next(iter(env.env.active_objects))
            env.env.infant["held_obj"] = oid
            env._instant_parent_eye_hit = True
            reward, breakdown = env._compute_preference_reward()
            self.assertEqual(reward, 0.0)
            self.assertEqual(breakdown["total"], 0.0)
        finally:
            env.env.close()

    def test_step_adds_preference_to_reward(self):
        env = _make_env()
        try:
            with patch.object(env, "get_gaze_target", return_value=(np.zeros(2), True, True)):
                env.reset(seed=0)
                _, reward, _, _, info = env.step(_zero_actions(env))
            self.assertAlmostEqual(reward[0], 1.2)
            self.assertAlmostEqual(info["preference_reward"]["total"], 1.2)
        finally:
            env.env.close()


if __name__ == "__main__":
    unittest.main()
