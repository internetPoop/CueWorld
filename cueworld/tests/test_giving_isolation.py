"""受け渡しの切り分けテスト（giving_isolation.md）のスイッチ確認。"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager
from cueworld.protocols.object_giving import DENSE_GRASP_BONUS, ObjectGivingProtocol
from cueworld.tests.test_ja_infant_initiative import _MockEnv


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


def _signals_empty():
    return {
        "gaze_targets": np.zeros(3, dtype=float),
        "gaze_targets_now": np.zeros(3, dtype=float),
        "gesture_targets": np.zeros(3, dtype=float),
        "point_intensity": 0.0,
        "hand_target_stability": 0.0,
        "voice_best_distance": 1.0,
        "words": {},
        "looking_at_parent_eye": 0.0,
        "gaze_shift_dict": {},
        "head_turn_away_intensity": 0.0,
        "negative_emotion_surge": 0.0,
    }


class TestCloseOnGrasp(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.env.reach_radius = 0.45
        self.protocol = ObjectGivingProtocol(self.env, rejection_check=False)

    def test_close_on_grasp_finishes_immediately(self):
        self.protocol.start({"fixed_target": 101, "close_on_grasp": True})
        self.protocol.parent_utterance = False
        self.protocol.state = "waiting_infant"
        self.protocol.state_timer = 0
        self.env.infant["held_obj"] = 101
        self.protocol._manage_state(_signals_empty())
        self.assertEqual(self.protocol.state, "completed")
        self.assertTrue(self.protocol.is_finished)
        self.assertTrue(self.protocol.session_success)
        self.assertNotEqual(self.protocol.state, "reaction")

    def test_default_enters_reaction(self):
        self.protocol.start({"fixed_target": 101})
        self.protocol.parent_utterance = False
        self.protocol.state = "waiting_infant"
        self.protocol.state_timer = 0
        self.env.infant["held_obj"] = 101
        self.protocol._manage_state(_signals_empty())
        self.assertFalse(self.protocol.close_on_grasp)
        self.assertEqual(self.protocol.state, "reaction")
        self.assertFalse(self.protocol.is_finished)
        self.assertTrue(self.protocol.session_success)


class TestGivingTimeoutSuppression(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.env.reach_radius = 0.45
        self.protocol = ObjectGivingProtocol(self.env, rejection_check=False)

    def test_close_on_grasp_limits_retries(self):
        self.protocol.start({"fixed_target": 101, "close_on_grasp": True})
        self.protocol.parent_utterance = False
        self.protocol.state = "waiting_infant"
        self.protocol.waiting_infant_timeout_steps = 2
        # 1 回目: RETRY（タイマー戻し、状態は waiting_infant）
        self.protocol.state_timer = 3
        self.protocol._manage_state(_signals_empty())
        self.assertEqual(self.protocol.state, "waiting_infant")
        self.assertEqual(self.protocol.retry_count, 1)
        self.assertFalse(self.protocol.escalated)
        # 2 回目: FAILED（掲げ・取り換えに行かない）
        self.protocol.state_timer = 3
        self.protocol._manage_state(_signals_empty())
        self.assertEqual(self.protocol.state, "failure")
        self.assertTrue(self.protocol.is_finished)
        self.assertFalse(self.protocol.escalated)

    def test_default_escalates_normally(self):
        self.protocol.start({"fixed_target": 101})
        self.protocol.parent_utterance = False
        self.protocol.state = "waiting_infant"
        self.protocol.waiting_infant_timeout_steps = 2
        # 1 回目: RETRY
        self.protocol.state_timer = 3
        self.protocol._manage_state(_signals_empty())
        self.assertEqual(self.protocol.retry_count, 1)
        self.assertEqual(self.protocol.state, "waiting_infant")
        # 2 回目: ESCALATE → fetching
        self.protocol.state_timer = 3
        self.protocol._manage_state(_signals_empty())
        self.assertEqual(self.protocol.retry_count, 2)
        self.assertTrue(self.protocol.escalated)
        self.assertEqual(self.protocol.state, "fetching")


class TestGivingDenseReward(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.env.reach_radius = 0.45
        self._palm = np.array([0.0, 0.0, 0.2], dtype=np.float32)
        self._forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._obj = {101: np.array([1.0, 0.0, 0.2], dtype=np.float32)}

        def _get_palm_pose(agent):
            return self._palm.copy(), self._forward.copy()

        def _obj_pos(oid):
            return self._obj[int(oid)].copy()

        self.env._get_palm_pose = _get_palm_pose
        self.env._obj_pos = _obj_pos
        self.env._angle_between = BaseEnv._angle_between

        self.protocol = ObjectGivingProtocol(self.env, rejection_check=False)

    def test_inactive_before_waiting(self):
        self.protocol.start({"fixed_target": 101})
        self.assertEqual(self.protocol.dense_reward_function({}), 0.0)

    def test_distance_decrease_positive(self):
        self.protocol.start({"fixed_target": 101})
        # waiting_infant 進入で起点を記録（e = 1.0、正面）
        self.protocol.change_state("waiting_infant")
        self.assertTrue(self.protocol._dense_active)
        self.assertAlmostEqual(self.protocol._dense_prev_e, 1.0, places=5)
        # 初回呼び出しは prev 記録済みなので 0（JA と同じく prev がある場合の差分）
        self.assertEqual(self.protocol.dense_reward_function({}), 0.0)

        # 掌を 0.5 m 近づける → e が 0.5 減る → +0.5
        self._obj[101] = np.array([0.5, 0.0, 0.2], dtype=np.float32)
        r = self.protocol.dense_reward_function({})
        self.assertAlmostEqual(r, 0.5, places=5)

    def test_distance_increase_negative(self):
        self.protocol.start({"fixed_target": 101})
        self.protocol.change_state("waiting_infant")
        self.protocol.dense_reward_function({})  # 0（変化なし）

        # 掌から遠ざける → 負
        self._obj[101] = np.array([1.5, 0.0, 0.2], dtype=np.float32)
        r = self.protocol.dense_reward_function({})
        self.assertAlmostEqual(r, -0.5, places=5)

    def test_angle_factor(self):
        self.protocol.start({"fixed_target": 101})
        # 正面・距離 1 → e = 1
        self.protocol.change_state("waiting_infant")
        e_front = self.protocol._dense_prev_e
        self.assertAlmostEqual(e_front, 1.0, places=5)

        # 同じ距離で真後ろ向き → e = 3d = 3
        self._forward = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        e_back = self.protocol._compute_palm_target_metric()
        self.assertAlmostEqual(e_back, 3.0, places=5)
        self.assertGreater(e_back, e_front)

    def test_grasp_bonus_then_zero(self):
        self.protocol.start({"fixed_target": 101, "close_on_grasp": True})
        self.protocol.parent_utterance = False
        self.protocol.change_state("waiting_infant")
        self.protocol.dense_reward_function({})  # 変化なし

        self.env.infant["held_obj"] = 101
        self.protocol._manage_state(_signals_empty())
        self.assertTrue(self.protocol.is_finished)
        self.assertTrue(self.protocol._dense_success_pending)
        self.assertAlmostEqual(
            self.protocol.dense_reward_function({}), DENSE_GRASP_BONUS
        )
        self.assertEqual(self.protocol.dense_reward_function({}), 0.0)

    def test_retry_does_not_reset_origin(self):
        self.protocol.start({"fixed_target": 101, "close_on_grasp": True})
        self.protocol.parent_utterance = False
        self.protocol.change_state("waiting_infant")
        prev = self.protocol._dense_prev_e
        self.protocol.waiting_infant_timeout_steps = 2
        self.protocol.state_timer = 3
        self.protocol._manage_state(_signals_empty())
        self.assertEqual(self.protocol.retry_count, 1)
        # 起点・prev は戻さない
        self.assertEqual(self.protocol._dense_prev_e, prev)
        self.assertTrue(self.protocol._dense_active)


class TestProtocolManagerGivingDense(unittest.TestCase):
    def test_dense_reward_strips_preference_and_injects(self):
        base = _make_base(
            num_objects=1,
            object_counts={"RedCube": 1},
            layout_freeze="full",
        )
        env = ProtocolManager(
            InfantSignalDetector(ParentTools(base)),
            dense_reward=True,
            household_interval_steps=10**9,
            rejection=False,
        )
        try:
            env.reset(seed=0)
            red = next(
                oid
                for oid, obj in base.all_objects.items()
                if getattr(obj, "concept_id", "") == "red_cube"
            )
            env.set_fixed_protocol(
                "object_giving",
                {
                    "fixed_target": red,
                    "close_on_grasp": True,
                },
            )
            proto = env.current_protocol
            self.assertIsNotNone(proto)
            proto.parent_utterance = False
            # 選好が足されていても dense モードでは差し引き後に dense だけ残る
            proto.dense_reward_function = MagicMock(return_value=1.5)
            isd = env.env  # InfantSignalDetector（ParentTools の外側）
            while not hasattr(isd, "_compute_preference_reward"):
                isd = isd.env
            isd._compute_preference_reward = MagicMock(
                return_value=(0.7, {"hold": 0.0, "look": 0.7, "total": 0.7})
            )
            _, r, _, _, info = env.step(_zero_actions(base))
            self.assertAlmostEqual(info["dense_reward"], 1.5)
            self.assertEqual(info["preference_reward"]["total"], 0.0)
            self.assertAlmostEqual(float(r[0]), 1.5)
        finally:
            base.close()


if __name__ == "__main__":
    unittest.main()
