"""T2: 失敗系 handle_timeout とエラー対策 fail_on_error_timeout の分離テスト。"""

import unittest
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np

from cueworld.protocols.base import BaseProtocol
from cueworld.protocols.object_giving import ObjectGivingProtocol


class _TinyEnv(gym.Env):
    action_num = 17
    action_map = {"volume": 12}
    client_id = 0

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32)
        self.parent = {"body": 1, "held_obj": None}
        self.infant = {"body": 2, "held_obj": None}
        self.place_dist = 0.4
        self.stand_dist = 0.8
        self.reach_radius = 0.45
        self.grasp_approach_dist = 0.5

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


class TestTimeoutKinds(unittest.TestCase):
    def test_fail_on_error_timeout_does_not_touch_retry_count(self):
        env = _TinyEnv()
        proto = BaseProtocol("Dummy", env, rejection_check=False)
        proto.retry_count = 0
        proto.state = "fetching"
        proto.state_timer = 999
        proto.fail_on_error_timeout()
        self.assertEqual(proto.state, proto.failure_state)
        self.assertTrue(proto.is_finished)
        self.assertEqual(proto.retry_count, 0)

    def test_handle_timeout_ladder_for_failure_path(self):
        env = _TinyEnv()
        proto = BaseProtocol("Dummy", env, change_target=True, rejection_check=False)
        self.assertEqual(proto.handle_timeout(max_retries=3), "RETRY")
        self.assertEqual(proto.handle_timeout(max_retries=3), "ESCALATE")
        self.assertEqual(proto.handle_timeout(max_retries=3), "CHANGE_TARGET")
        self.assertEqual(proto.handle_timeout(max_retries=3), "FAILED")

    def test_handle_timeout_max_retries_1_is_retry_then_failed(self):
        env = _TinyEnv()
        proto = BaseProtocol("Dummy", env, change_target=False, rejection_check=False)
        self.assertEqual(proto.handle_timeout(max_retries=1), "RETRY")
        self.assertEqual(proto.handle_timeout(max_retries=1), "FAILED")

    def test_giving_fetch_error_timeout_skips_handle_timeout(self):
        env = _TinyEnv()
        proto = ObjectGivingProtocol(env, rejection_check=False)
        proto.language_manager = MagicMock()
        proto.target_object = MagicMock()
        proto.target_object.id = 7
        env.parent["held_obj"] = None
        proto.state = "fetching"
        proto.state_timer = proto.fetch_timeout_steps
        proto.retry_count = 0
        # state_timer は _manage_state 冒頭で +1 されるので、境界ちょうどで発火させる
        proto.state_timer = proto.fetch_timeout_steps
        infant_signals = {
            "avoidance_intensity": 0.0,
            "gaze_shift_dict": {},
            "negative_emotion_surge": 0.0,
        }
        proto._manage_state(infant_signals)
        self.assertEqual(proto.state, proto.failure_state)
        self.assertTrue(proto.is_finished)
        self.assertEqual(proto.retry_count, 0)

    def test_giving_waiting_infant_uses_handle_timeout(self):
        env = _TinyEnv()
        proto = ObjectGivingProtocol(env, rejection_check=False)
        proto.language_manager = MagicMock()
        proto.language_manager.get_concept_for_object.return_value = "red"
        proto.queue_utterance = MagicMock()
        proto.target_object = MagicMock()
        proto.target_object.id = 7
        env.infant["held_obj"] = None
        proto.state = "waiting_infant"
        proto.state_timer = proto.waiting_infant_timeout_steps
        proto.retry_count = 0
        infant_signals = {
            "avoidance_intensity": 0.0,
            "gaze_shift_dict": {},
            "negative_emotion_surge": 0.0,
        }
        proto._manage_state(infant_signals)
        # RETRY: 発話再キュー、waiting_infant のまま
        self.assertEqual(proto.state, "waiting_infant")
        self.assertEqual(proto.retry_count, 1)
        self.assertFalse(proto.is_finished)
        proto.queue_utterance.assert_called()


if __name__ == "__main__":
    unittest.main()
