"""ProtocolManager の dyadic 連携のユニットテスト（§6-D2 意図レイヤ）。"""

import unittest
from unittest.mock import patch

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

    def object_in_agent_reach(self, agent, oid):
        return False

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), [0.0, 0.0], False, False, {}


class TestProtocolManagerDyadic(unittest.TestCase):
    def setUp(self):
        self.manager = ProtocolManager(_MockEnv(), fixed_protocol="observance", fixed_context={})

    def _zeros(self):
        signals = {key: 0.0 for key in self.manager.signal_keys}
        signals["gesture_targets"] = np.zeros(1, dtype=float)
        signals["gaze_targets"] = np.zeros(1, dtype=float)
        signals["words"] = {}
        signals["looking_at_parent_eye"] = 0.0
        signals["touching"] = 0.0
        signals["voice_best_distance"] = 10.0
        return signals

    def test_intent_keys_include_dyadic(self):
        self.assertIn("dyadic", self.manager.intent_keys)
        self.assertIn("hand_height_margin", self.manager.signal_keys)

    def test_dyadic_strength_responds_to_arm_raise_salience(self):
        base_signals = self._zeros()
        base_strength = self.manager._calculate_signal_strength(base_signals)
        dyadic_idx = self.manager.response_protocol.index("dyadic")
        base_score = base_strength[dyadic_idx]

        raise_signals = dict(base_signals)
        raise_signals["hand_height_margin"] = 1.0
        raise_score = self.manager._calculate_signal_strength(raise_signals)[dyadic_idx]

        self.assertGreater(raise_score, base_score)
        self.assertAlmostEqual(float(raise_score - base_score), 2.0, places=5)

    def test_dyadic_strength_ignores_wave_and_nod(self):
        base_signals = self._zeros()
        dyadic_idx = self.manager.response_protocol.index("dyadic")
        base_score = self.manager._calculate_signal_strength(base_signals)[dyadic_idx]

        wave_signals = dict(base_signals)
        wave_signals["hand_movement_size"] = 1.0
        wave_signals["hand_movement_regularity"] = 1.0
        wave_signals["head_nod_periodicity"] = 1.0
        wave_score = self.manager._calculate_signal_strength(wave_signals)[dyadic_idx]

        self.assertAlmostEqual(float(wave_score), float(base_score), places=5)

    @patch("cueworld.protocol_manager.np.random.choice")
    def test_check_for_response_passes_babbling_chunk_for_dyadic(self, mock_choice):
        mock_choice.return_value = "dyadic"
        self.manager.current_protocol.persistence = 0.0

        infant_signals = self._zeros()
        infant_signals.update(
            {
                "babbling_chunk": ["p0", "p1", "p0"],
                "babbling_salience": 1.0,
            }
        )

        should_switch, protocol_name, context = self.manager.check_for_response(infant_signals)

        self.assertTrue(should_switch)
        self.assertEqual(protocol_name, "dyadic")
        self.assertEqual(context["babbling_chunk"], ["p0", "p1", "p0"])
        self.assertTrue(context["infant_initiative"])
        self.assertIn("infant_signals", context)
        self.assertNotIn("salient_channels", context)

    @patch("cueworld.protocol_manager.np.random.choice")
    def test_check_for_response_dyadic_without_babbling_omits_chunk(self, mock_choice):
        mock_choice.return_value = "dyadic"
        self.manager.current_protocol.persistence = 0.0

        infant_signals = self._zeros()
        infant_signals["hand_height_margin"] = 1.0

        should_switch, protocol_name, context = self.manager.check_for_response(infant_signals)

        self.assertTrue(should_switch)
        self.assertEqual(protocol_name, "dyadic")
        self.assertNotIn("babbling_chunk", context)

    def test_weight_matrix_shape_matches_intent_keys(self):
        n_protocols = len(self.manager.response_protocol)
        n_intents = len(self.manager.intent_keys)
        self.assertEqual(self.manager.weights_for_strength_calc.shape, (n_protocols, n_intents))


if __name__ == "__main__":
    unittest.main()
