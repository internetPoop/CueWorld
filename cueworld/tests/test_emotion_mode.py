"""感情モード（enable_emotion）の次元と sharing 拒否のテスト。"""

import unittest

import numpy as np

from cueworld.base import BaseEnv, _INFANT_FATIGUE_CHANNELS, _INFANT_JOINT_OBS_SLOTS_FULL
from cueworld.protocols.joint_attention import JointAttentionProtocol
from cueworld.tests.test_ja_infant_initiative import _MockEnv


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


class TestEmotionModeOff(unittest.TestCase):
    """enable_emotion=False（既定）: lip / pitch を次元ごと除外。"""

    def test_action_and_voice_dims(self):
        base = _make_base()
        try:
            self.assertFalse(base.enable_emotion)
            self.assertEqual(base.action_num, 15)
            self.assertEqual(base.action_space[0].shape, (15,))
            self.assertEqual(base.action_space[1].shape, (15,))
            self.assertNotIn("lip", base.action_map)
            self.assertNotIn("pitch", base.action_map)
            self.assertEqual(base.voice_dim, 4)
            self.assertNotIn("pitch", base.voice_mapping)
            self.assertEqual(base.phoneme_index, 1)
            self.assertEqual(
                base.voice_mapping,
                {"volume": 0, "v_1": 1, "v_2": 2, "v_3": 3},
            )
            self.assertNotIn("lip", base.infant_joint_mappings)
            self.assertNotIn("pitch", base.infant_joint_mappings)
            self.assertNotIn("lip", base.parent_joint_mappings)
            self.assertNotIn("pitch", base.parent_joint_mappings)
        finally:
            base.close()

    def test_joint_and_fatigue_obs_dims(self):
        base = _make_base()
        try:
            expected_slots = tuple(s for s in _INFANT_JOINT_OBS_SLOTS_FULL if s != "lip")
            self.assertEqual(tuple(base._infant_joint_obs_slots), expected_slots)
            self.assertEqual(base.observation_space[0]["joint_angles"].shape, (12,))
            self.assertEqual(base._fatigue_dim, 9)
            self.assertNotIn("lip", base._effective_fatigue_channels)
            self.assertEqual(base.observation_space[0]["fatigue"].shape, (9,))
            self.assertEqual(base.observation_space[0]["fatigue_lock"].shape, (9,))

            obs, _ = base.reset(seed=0)
            infant_obs = obs[0]
            self.assertEqual(infant_obs["joint_angles"].shape, (12,))
            self.assertEqual(infant_obs["infant_voice"].shape, (4,))
            self.assertEqual(infant_obs["parent_voice"].shape, (4,))
            self.assertEqual(infant_obs["fatigue"].shape, (9,))
        finally:
            base.close()

    def test_sharing_raises(self):
        env = _MockEnv()
        env.enable_emotion = False
        protocol = JointAttentionProtocol(
            env, cue_type="pointing", rejection_check=False
        )
        with self.assertRaises(ValueError) as ctx:
            protocol.start({"fixed_target": 101, "intent_type": "sharing"})
        self.assertIn("enable_emotion", str(ctx.exception))


class TestEmotionModeOn(unittest.TestCase):
    """enable_emotion=True: 従来どおり lip / pitch を含む。"""

    def test_action_and_voice_dims(self):
        base = _make_base(enable_emotion=True)
        try:
            self.assertTrue(base.enable_emotion)
            self.assertEqual(base.action_num, 17)
            self.assertEqual(base.action_space[0].shape, (17,))
            self.assertIn("lip", base.action_map)
            self.assertIn("pitch", base.action_map)
            self.assertEqual(base.voice_dim, 5)
            self.assertEqual(base.phoneme_index, 2)
            self.assertEqual(
                base.voice_mapping,
                {"volume": 0, "pitch": 1, "v_1": 2, "v_2": 3, "v_3": 4},
            )
            self.assertIn("lip", base.infant_joint_mappings)
            self.assertIn("pitch", base.infant_joint_mappings)
        finally:
            base.close()

    def test_joint_and_fatigue_obs_dims(self):
        base = _make_base(enable_emotion=True)
        try:
            self.assertEqual(
                tuple(base._infant_joint_obs_slots), _INFANT_JOINT_OBS_SLOTS_FULL
            )
            self.assertEqual(base.observation_space[0]["joint_angles"].shape, (13,))
            self.assertEqual(base._fatigue_dim, 10)
            self.assertEqual(
                base._effective_fatigue_channels, _INFANT_FATIGUE_CHANNELS
            )
            self.assertEqual(base.observation_space[0]["fatigue"].shape, (10,))

            obs, _ = base.reset(seed=0)
            infant_obs = obs[0]
            self.assertEqual(infant_obs["joint_angles"].shape, (13,))
            self.assertEqual(infant_obs["infant_voice"].shape, (5,))
            self.assertEqual(infant_obs["fatigue"].shape, (10,))
        finally:
            base.close()

    def test_sharing_allowed(self):
        env = _MockEnv()
        env.enable_emotion = True
        protocol = JointAttentionProtocol(
            env, cue_type="pointing", rejection_check=False
        )
        protocol.start({"fixed_target": 101, "intent_type": "sharing"})
        self.assertEqual(protocol.intent_type, "sharing")


if __name__ == "__main__":
    unittest.main()
