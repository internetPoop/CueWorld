"""見るだけ共同注意の切り分けテスト（ja_looking_isolation.md）のスイッチ確認。"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager
from cueworld.protocols.base import ATTENTION_WAIT_STEPS
from cueworld.protocols.joint_attention import (
    DENSE_SUCCESS_BONUS,
    JointAttentionProtocol,
)
from cueworld.tests.test_ja_infant_initiative import _MockEnv, _signals_for_target
from cueworld.tests.test_ja_success_hold import _signals_empty


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


def _pose(body, client_id):
    import pybullet as p

    pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=client_id)
    return (tuple(np.round(pos, 5)), tuple(np.round(orn, 5)))


class TestInfantObsKeys(unittest.TestCase):
    def test_filters_observation_space_and_obs(self):
        keys = ["image", "joint_angles"]
        base = _make_base(
            infant_obs_keys=keys,
            infant_action_names=["torso_joint", "head_pan_joint", "head_tilt_joint"],
        )
        try:
            spaces = base.observation_space[0].spaces
            self.assertEqual(list(spaces.keys()), keys)
            # torso は sin/cos → 4 次元
            self.assertEqual(spaces["joint_angles"].shape, (4,))

            obs, _ = base.reset(seed=0)
            self.assertEqual(list(obs[0].keys()), keys)
            self.assertEqual(obs[0]["joint_angles"].shape, (4,))
            self.assertNotIn("is_first", obs[0])
            self.assertNotIn("tactile", obs[0])

            obs2, _, _, _, _ = base.step(_zero_actions(base))
            self.assertEqual(list(obs2[0].keys()), keys)
        finally:
            base.close()

    def test_default_keeps_all_keys(self):
        base = _make_base()
        try:
            keys = set(base.observation_space[0].spaces.keys())
            obs, _ = base.reset(seed=0)
            self.assertTrue({"image", "tactile", "is_first", "infant_voice"} <= keys)
            self.assertTrue({"image", "tactile", "is_first", "infant_voice"} <= set(obs[0].keys()))
        finally:
            base.close()


class TestLayoutFreeze(unittest.TestCase):
    def test_full_repeats_poses_and_target(self):
        base = _make_base(layout_freeze="full", num_objects=2)
        try:
            base.reset(seed=0)
            infant0 = _pose(base.infant_body, base.client_id)
            parent0 = _pose(base.parent_body, base.client_id)
            objs0 = {
                bid: _pose(bid, base.client_id) for bid in base.active_objects
            }
            target0 = base.target_obj_id
            active0 = list(base.active_objects)

            base.reset(seed=1)
            self.assertEqual(_pose(base.infant_body, base.client_id), infant0)
            self.assertEqual(_pose(base.parent_body, base.client_id), parent0)
            self.assertEqual(list(base.active_objects), active0)
            for bid, pose0 in objs0.items():
                self.assertEqual(_pose(bid, base.client_id), pose0)
            self.assertEqual(base.target_obj_id, target0)
        finally:
            base.close()

    def test_rotate_target_cycles_movable(self):
        base = _make_base(layout_freeze="rotate_target", num_objects=2)
        try:
            base.reset(seed=0)
            movable = [
                bid
                for bid in base.active_objects
                if getattr(base.all_objects[bid], "move_type", None) is not None
            ]
            self.assertGreaterEqual(len(movable), 2)
            first = base.target_obj_id
            self.assertIn(first, movable)
            infant0 = _pose(base.infant_body, base.client_id)

            base.reset(seed=0)
            second = base.target_obj_id
            self.assertIn(second, movable)
            self.assertNotEqual(second, first)
            self.assertEqual(_pose(base.infant_body, base.client_id), infant0)

            base.reset(seed=0)
            third = base.target_obj_id
            self.assertEqual(third, first)
            self.assertEqual(_pose(base.infant_body, base.client_id), infant0)
        finally:
            base.close()

    def test_none_does_not_freeze(self):
        base = _make_base(layout_freeze="none", num_objects=2, random_parent_pos=True)
        try:
            base.reset(seed=0)
            parent0 = _pose(base.parent_body, base.client_id)
            # 別シードなら親位置が変わりうる（常に変わる保証は弱いので、frozen が無いことだけ見る）
            self.assertIsNone(base._frozen_layout)
            base.reset(seed=1)
            self.assertIsNone(base._frozen_layout)
            # 少なくとも 2 回目も通常リセット経路で例外にならない
            _ = _pose(base.parent_body, base.client_id)
            self.assertIsNotNone(parent0)
        finally:
            base.close()


class TestLockCue(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.protocol = JointAttentionProtocol(
            self.env, cue_type="looking", rejection_check=False
        )

    def test_lock_cue_keeps_looking(self):
        self.protocol.start(
            {
                "fixed_target": 101,
                "intent_type": "informative",
                "lock_cue": True,
            }
        )
        self.protocol.parent_utterance = False
        self.protocol.state = "looking"
        self.protocol._cue_started = True
        self.protocol.attention_tracker.hold_steps = 30
        self.protocol.attention_tracker.wait_steps = 3
        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(102))
        self.assertEqual(self.protocol.cue_type, "looking")
        self.assertEqual(self.protocol.state, "looking")
        self.assertEqual(self.protocol.retry_count, 1)

        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(102))
        self.assertEqual(self.protocol.state, "failure")
        self.assertTrue(self.protocol.is_finished)

    def test_default_escalates_looking_to_pointing(self):
        self.protocol.start(
            {"fixed_target": 101, "intent_type": "informative"}
        )
        self.protocol.parent_utterance = False
        self.protocol.state = "looking"
        self.protocol._cue_started = True
        self.protocol.attention_tracker.hold_steps = 30
        self.protocol.attention_tracker.wait_steps = 3
        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(102))
        self.assertFalse(self.protocol.lock_cue)
        self.assertEqual(self.protocol.cue_type, "pointing")
        self.assertEqual(self.protocol.state, "pointing")


class TestCloseOnSuccess(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.protocol = JointAttentionProtocol(
            self.env, cue_type="looking", rejection_check=False
        )

    def test_close_on_success_finishes_immediately(self):
        self.protocol.start(
            {
                "fixed_target": 101,
                "intent_type": "informative",
                "close_on_success": True,
            }
        )
        self.protocol.parent_utterance = False
        self.protocol.state = "looking"
        self.protocol._cue_started = True
        self.protocol.attention_tracker.hold_steps = 3
        self.protocol.attention_tracker.wait_steps = ATTENTION_WAIT_STEPS
        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "completed")
        self.assertTrue(self.protocol.is_finished)
        self.assertTrue(self.protocol.session_success)
        self.assertNotEqual(self.protocol.state, "post_shared_attention")

    def test_default_enters_post_shared_attention(self):
        self.protocol.start(
            {"fixed_target": 101, "intent_type": "informative"}
        )
        self.protocol.parent_utterance = False
        self.protocol.state = "looking"
        self.protocol._cue_started = True
        self.protocol.attention_tracker.hold_steps = 3
        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(101))
        self.assertFalse(self.protocol.close_on_success)
        self.assertEqual(self.protocol.state, "post_shared_attention")
        self.assertFalse(self.protocol.is_finished)


class TestDenseRewardFunction(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        # 角度計算用のスタブ
        self._eye = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._obj = {101: np.array([1.0, 0.0, 1.0], dtype=np.float32)}

        def _agent_eye_pose(agent):
            return self._eye.copy(), self._forward.copy()

        def _obj_pos(oid):
            return self._obj[int(oid)].copy()

        self.env._agent_eye_pose = _agent_eye_pose
        self.env._obj_pos = _obj_pos
        self.env._angle_between = BaseEnv._angle_between

        self.protocol = JointAttentionProtocol(
            self.env, cue_type="looking", rejection_check=False
        )

    def test_inactive_before_cue(self):
        self.protocol.start({"fixed_target": 101, "intent_type": "informative"})
        self.assertEqual(self.protocol.dense_reward_function({}), 0.0)

    def test_angle_decrease_positive_increase_negative(self):
        self.protocol.start({"fixed_target": 101, "intent_type": "informative"})
        # 合図開始: 対象は正面（角度 0）
        self.protocol.state = "looking"
        self.protocol._ensure_cue_started()
        self.assertTrue(self.protocol._dense_active)
        # 初回は prev 記録のみ
        self.assertEqual(self.protocol.dense_reward_function({}), 0.0)

        # 90° へ離す → 負
        self._obj[101] = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        r = self.protocol.dense_reward_function({})
        self.assertAlmostEqual(r, -0.5, places=5)

        # 再び正面へ → 正
        self._obj[101] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        r = self.protocol.dense_reward_function({})
        self.assertAlmostEqual(r, 0.5, places=5)

    def test_success_bonus_then_zero(self):
        self.protocol.start(
            {
                "fixed_target": 101,
                "intent_type": "informative",
                "close_on_success": True,
            }
        )
        self.protocol.parent_utterance = False
        self.protocol.state = "looking"
        self.protocol._ensure_cue_started()
        self.protocol.attention_tracker.hold_steps = 2
        self.protocol.dense_reward_function({})  # 起点

        self.protocol._manage_state(_signals_for_target(101))
        self.protocol._manage_state(_signals_for_target(101))
        self.assertTrue(self.protocol.is_finished)
        self.assertTrue(self.protocol._dense_success_pending)
        self.assertAlmostEqual(
            self.protocol.dense_reward_function({}), DENSE_SUCCESS_BONUS
        )
        self.assertEqual(self.protocol.dense_reward_function({}), 0.0)

    def test_retry_does_not_reset_origin(self):
        self.protocol.start(
            {
                "fixed_target": 101,
                "intent_type": "informative",
                "lock_cue": True,
            }
        )
        self.protocol.parent_utterance = False
        self.protocol.state = "looking"
        self.protocol._ensure_cue_started()
        prev = self.protocol._dense_prev_angle
        self.protocol.attention_tracker.wait_steps = 2
        self.protocol.attention_tracker.hold_steps = 30
        for _ in range(2):
            self.protocol._manage_state(_signals_empty())
        self.assertEqual(self.protocol.retry_count, 1)
        # 起点・prev は戻さない
        self.assertEqual(self.protocol._dense_prev_angle, prev)
        self.assertTrue(self.protocol._dense_active)


class TestProtocolManagerDenseReward(unittest.TestCase):
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
                "joint_attention",
                {
                    "fixed_target": red,
                    "cue_type": "looking",
                    "intent_type": "informative",
                    "close_on_success": True,
                },
            )
            proto = env.current_protocol
            self.assertIsNotNone(proto)
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

    def test_default_dense_reward_off(self):
        base = _make_base(num_objects=1, object_counts={"RedCube": 1})
        env = ProtocolManager(
            InfantSignalDetector(ParentTools(base)),
            household_interval_steps=10**9,
            rejection=False,
        )
        try:
            self.assertFalse(env.dense_reward)
            env.reset(seed=0)
            _, _, _, _, info = env.step(_zero_actions(base))
            self.assertNotIn("dense_reward", info)
            self.assertIn("preference_reward", info)
        finally:
            base.close()


if __name__ == "__main__":
    unittest.main()
