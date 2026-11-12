"""ProtocolManager バッチ D / §6-D2: 視線は訴えにしない／自発性 EMA／意図レイヤ。"""

import unittest

import gymnasium as gym
import numpy as np

from cueworld.protocol_manager import ProtocolManager


class _MockEnv(gym.Env):
    action_num = 16
    eval_mode = False
    active_signals = None
    target_obj_id = None
    all_objects = {}
    active_objects = []
    trackable_targets = []
    parent = {"body": 1}
    infant = {"body": 2}
    voice_volume_threshold = 0.25

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = (
            gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32),
            gym.spaces.Box(-1.0, 1.0, shape=(16,), dtype=np.float32),
        )
        self.step_count = 0

    def object_in_agent_reach(self, agent, oid):
        return False

    def reset(self, **kwargs):
        self.step_count = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        return np.zeros(1, dtype=np.float32), [0.0, 0.0], False, False, {}


class TestProtocolManagerSpontaneity(unittest.TestCase):
    def setUp(self):
        self.manager = ProtocolManager(
            _MockEnv(),
            fixed_protocol="observance",
            fixed_context={},
            decay_rate=0.5,
        )
        self.ja_idx = self.manager.response_protocol.index("joint_attention")
        self.giving_idx = self.manager.response_protocol.index("object_giving")

    def _zeros(self):
        signals = {key: 0.0 for key in self.manager.signal_keys}
        signals["gesture_targets"] = np.zeros(1, dtype=float)
        signals["gaze_targets"] = np.zeros(1, dtype=float)
        signals["words"] = {}
        signals["looking_at_parent_eye"] = 0.0
        signals["touching"] = 0.0
        signals["voice_best_distance"] = 10.0
        return signals

    def test_gaze_is_not_an_intent_key(self):
        self.assertNotIn("gaze_target_stability", self.manager.intent_keys)

    def test_gaze_alone_does_not_change_ja_or_giving_strength(self):
        base = self.manager._calculate_signal_strength(self._zeros())
        gaze_only = self._zeros()
        gaze_only["gaze_target_stability"] = 1.0
        gazed = self.manager._calculate_signal_strength(gaze_only)

        self.assertAlmostEqual(float(gazed[self.ja_idx]), float(base[self.ja_idx]), places=5)
        self.assertAlmostEqual(float(gazed[self.giving_idx]), float(base[self.giving_idx]), places=5)

    def test_complete_pointing_raises_ja_not_via_raw_voice(self):
        base = self.manager._calculate_signal_strength(self._zeros())[self.ja_idx]
        # 単語だけでは JA に乗らない（§6-D2: 単語はくれ側）
        voiced = self._zeros()
        voiced["voice_target_stability"] = 1.0
        voiced["voice_volume"] = 1.0
        voiced["voice_best_distance"] = 0.1
        voiced["words"] = {"red_cube": 0.1}
        score = self.manager._calculate_signal_strength(voiced)[self.ja_idx]
        self.assertAlmostEqual(float(score), float(base), places=5)

    def test_initiative_rate_rises_only_on_infant_initiated_success(self):
        # 自発性は離れるプロトコル自身の infant_initiative を見る。
        ja = self.manager.protocols["joint_attention"]
        self.manager.infant_initiative_rate = 0.0
        ja.infant_initiative = True
        self.manager.save_protocol_history("joint_attention", True)
        self.assertAlmostEqual(self.manager.infant_initiative_rate, 0.5, places=5)

        ja.infant_initiative = False
        self.manager.save_protocol_history("joint_attention", True)
        self.assertAlmostEqual(self.manager.infant_initiative_rate, 0.25, places=5)

        ja.infant_initiative = True
        self.manager.save_protocol_history("joint_attention", False)
        self.assertAlmostEqual(self.manager.infant_initiative_rate, 0.125, places=5)


if __name__ == "__main__":
    unittest.main()
