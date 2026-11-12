"""幼児 fatigue のテスト。"""

import unittest

import numpy as np

from cueworld.base import BaseEnv, _INFANT_FATIGUE_CHANNELS


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


def _infant_action(base: BaseEnv, **kwargs) -> np.ndarray:
    action = np.zeros(base.action_num, dtype=np.float32)
    for key, value in kwargs.items():
        action[base.action_map[key]] = float(value)
    return action


class TestFatigueDisabled(unittest.TestCase):
    def test_disabled_does_not_lock_elbow(self):
        base = _make_base(enable_fatigue=False)
        try:
            base.reset(seed=0)
            elbow_idx = base._fatigue_channel_index["elbow_joint"]
            for _ in range(40):
                infant_action = _infant_action(base, elbow_joint=1.0)
                base.step((infant_action, _zero_actions(base)[1]))
            self.assertFalse(base._infant_fatigue_lock[elbow_idx])
            angle = base.get_joint_angle(base.infant, "elbow_joint")
            self.assertGreater(abs(angle), 0.1)
        finally:
            base.close()


class TestFatigueEnabled(unittest.TestCase):
    def test_elbow_locks_and_retracts_toward_zero(self):
        base = _make_base(
            enable_fatigue=True,
            fatigue_high=5.0,
            fatigue_low=1.0,
            fatigue_recovery=0.1,
        )
        try:
            base.reset(seed=0)
            elbow_idx = base._fatigue_channel_index["elbow_joint"]
            locked = False
            for _ in range(80):
                infant_action = _infant_action(base, elbow_joint=1.0)
                base.step((infant_action, _zero_actions(base)[1]))
                if base._infant_fatigue_lock[elbow_idx]:
                    locked = True
                    break
            self.assertTrue(locked)

            prev_angle = abs(base.get_joint_angle(base.infant, "elbow_joint"))
            infant_action = _infant_action(base, elbow_joint=1.0)
            base.step((infant_action, _zero_actions(base)[1]))
            new_angle = abs(base.get_joint_angle(base.infant, "elbow_joint"))
            self.assertLess(new_angle, prev_angle)
        finally:
            base.close()

    def test_torso_has_no_fatigue_channel(self):
        self.assertNotIn("torso_joint", _INFANT_FATIGUE_CHANNELS)

    def test_volume_fatigue_channel_exists(self):
        self.assertIn("volume", _INFANT_FATIGUE_CHANNELS)


if __name__ == "__main__":
    unittest.main()
