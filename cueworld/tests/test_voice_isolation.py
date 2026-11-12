"""音声切り分けテスト（voice_isolation.md）のスイッチ確認。"""

import unittest
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np

from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.language.vocabulary import CONCEPT_TO_PHONEMES
from cueworld.language.voice_recognizer import GETA_BONUS
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager
from cueworld.protocols.voice_echo import (
    MAX_CUE_ROUNDS,
    MAX_EXTRA_WAIT_STEPS,
    MIN_SILENCE_STEPS,
    OBJECT_CONCEPTS,
    VoiceEchoProtocol,
)


class _MockEnv(gym.Env):
    action_num = 15
    action_map = {
        "volume": 11,
        "v_1": 12,
        "v_2": 13,
        "v_3": 14,
    }
    phoneme_mapping = {
        "p0": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        "p1": np.array([1.0, -1.0, -1.0], dtype=np.float32),
        "p2": np.array([-1.0, 1.0, -1.0], dtype=np.float32),
        "p3": np.array([1.0, 1.0, -1.0], dtype=np.float32),
        "p4": np.array([-1.0, -1.0, 1.0], dtype=np.float32),
        "p5": np.array([1.0, -1.0, 1.0], dtype=np.float32),
        "p6": np.array([-1.0, 1.0, 1.0], dtype=np.float32),
        "p7": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    client_id = 0
    voice_volume_threshold = 0.25

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(15,), dtype=np.float32)
        self.parent = {"body": 1}
        self.infant = {"body": 2, "eyes_link": 0}
        self.step_count = 0
        self.last_all_word_scores = {}
        self.last_raw_word_scores = {}
        self.voice_recognizer = MagicMock()
        self.voice_recognizer.in_segment = False
        self.voice_recognizer.set_target = MagicMock()

    def apply_phoneme_to_action(self, action, phoneme_coords, *, volume=0.5):
        coords = np.asarray(phoneme_coords, dtype=np.float32).reshape(-1)
        action[self.action_map["v_1"]] = float(coords[0])
        action[self.action_map["v_2"]] = float(coords[1])
        action[self.action_map["v_3"]] = float(coords[2])
        action[self.action_map["volume"]] = float(volume)
        return action

    def gaze_agent(self, agent_from, agent_to, yaw_link="head", action=None):
        return 0.0, 0.0

    def reset_arm_pose(self, agent, action=None):
        pass

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        return np.zeros(1, dtype=np.float32), [0.0, 0.0], False, False, {}


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


def _drain_cue(protocol: VoiceEchoProtocol, signals=None):
    """見本キューを吐き切って waiting に入る。"""
    signals = signals or {}
    guard = 0
    while protocol.state == "cueing" and guard < 200:
        protocol._manage_state(signals)
        protocol._determine_action()
        guard += 1
    return guard


def _wait_min_silence(protocol: VoiceEchoProtocol, signals=None):
    signals = signals or {}
    for _ in range(MIN_SILENCE_STEPS):
        if protocol.state != "waiting":
            break
        protocol._manage_state(signals)


class TestObjectConcepts(unittest.TestCase):
    def test_twenty_object_words_in_dict_order(self):
        self.assertEqual(len(OBJECT_CONCEPTS), 20)
        self.assertEqual(OBJECT_CONCEPTS[0], "red_cube")
        self.assertEqual(OBJECT_CONCEPTS[-1], "yellow_board")
        for cid in OBJECT_CONCEPTS:
            self.assertIn(cid, CONCEPT_TO_PHONEMES)
            self.assertEqual(len(CONCEPT_TO_PHONEMES[cid]), 5)


class TestVoiceEchoStateMachine(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.protocol = VoiceEchoProtocol(self.env)

    def test_start_queues_target_and_sets_cueing(self):
        self.protocol.start({"target_concept": "red_cube", "use_geta": False})
        self.assertEqual(self.protocol.state, "cueing")
        self.assertEqual(self.protocol.target_concept, "red_cube")
        # 5 音 × 6 step = 30
        self.assertEqual(len(self.protocol.utterance_queue), 30)
        self.env.voice_recognizer.set_target.assert_called_with(None)

    def test_use_geta_sets_target(self):
        self.protocol.start({"target_concept": "blue_cube", "use_geta": True})
        self.env.voice_recognizer.set_target.assert_called_with("blue_cube")

    def test_five_rounds_then_completed(self):
        self.protocol.start(
            {
                "target_concept": "red_cube",
                "max_rounds": MAX_CUE_ROUNDS,
            }
        )
        for round_i in range(MAX_CUE_ROUNDS):
            self.assertEqual(self.protocol.state, "cueing", msg=f"round {round_i}")
            _drain_cue(self.protocol)
            self.assertEqual(self.protocol.state, "waiting", msg=f"round {round_i}")
            _wait_min_silence(self.protocol)
            if round_i < MAX_CUE_ROUNDS - 1:
                self.assertEqual(self.protocol.state, "cueing")
            else:
                self.assertEqual(self.protocol.state, "completed")
                self.assertTrue(self.protocol.is_finished)
                self.assertTrue(self.protocol.session_success)

    def test_extra_wait_while_in_segment(self):
        self.protocol.start({"target_concept": "red_cube", "max_rounds": 1})
        _drain_cue(self.protocol)
        self.env.voice_recognizer.in_segment = True
        # 下限 12 ではまだ進まない
        for _ in range(MIN_SILENCE_STEPS):
            self.protocol._manage_state({})
        self.assertEqual(self.protocol.state, "waiting")
        # 追加 30 まで待つと終わる
        for _ in range(MAX_EXTRA_WAIT_STEPS):
            self.protocol._manage_state({})
        self.assertEqual(self.protocol.state, "completed")
        self.assertTrue(self.protocol.is_finished)


class TestDenseRewardFunction(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.protocol = VoiceEchoProtocol(self.env)

    def test_no_judgment_is_zero(self):
        self.protocol.start({"target_concept": "red_cube", "use_geta": False})
        self.env.last_all_word_scores = {}
        self.env.last_raw_word_scores = {}
        self.assertEqual(self.protocol.dense_reward_function({}), 0.0)

    def test_perfect_match_without_geta_is_one(self):
        self.protocol.start({"target_concept": "red_cube", "use_geta": False})
        self.env.last_raw_word_scores = {"red_cube": 0.0, "blue_cube": 5.0}
        self.assertAlmostEqual(self.protocol.dense_reward_function({}), 1.0)

    def test_perfect_match_with_geta_is_one(self):
        self.protocol.start({"target_concept": "red_cube", "use_geta": True})
        # 下駄後の完全一致は -GETA_BONUS
        self.env.last_all_word_scores = {"red_cube": -float(GETA_BONUS)}
        self.assertAlmostEqual(self.protocol.dense_reward_function({}), 1.0)

    def test_cost_two_without_geta(self):
        self.protocol.start({"target_concept": "red_cube", "use_geta": False})
        self.env.last_raw_word_scores = {"red_cube": 2.0}
        # 1 - (2 - 0) = -1
        self.assertAlmostEqual(self.protocol.dense_reward_function({}), -1.0)

    def test_unadopted_still_scores(self):
        self.protocol.start({"target_concept": "red_cube", "use_geta": False})
        # 不採用でも見本語のコストがあれば出る
        self.env.last_raw_word_scores = {"red_cube": 4.0, "blue_cube": 3.5}
        self.assertAlmostEqual(self.protocol.dense_reward_function({}), 1.0 - 4.0)


class TestWordRotation(unittest.TestCase):
    def test_rotates_across_starts(self):
        env = _MockEnv()
        protocol = VoiceEchoProtocol(env)
        seen = []
        for _ in range(len(OBJECT_CONCEPTS) + 1):
            protocol.start({})
            seen.append(protocol.target_concept)
        self.assertEqual(seen[:20], OBJECT_CONCEPTS)
        self.assertEqual(seen[20], OBJECT_CONCEPTS[0])

    def test_override_does_not_consume_index(self):
        env = _MockEnv()
        protocol = VoiceEchoProtocol(env)
        protocol.start({"target_concept": "yellow_board"})
        self.assertEqual(protocol.target_concept, "yellow_board")
        self.assertEqual(protocol._word_index, 0)
        protocol.start({})
        self.assertEqual(protocol.target_concept, OBJECT_CONCEPTS[0])
        self.assertEqual(protocol._word_index, 1)


class TestInfantObsAndActionKeys(unittest.TestCase):
    def test_filters_observation_space_and_action(self):
        keys = ["infant_voice", "parent_voice"]
        actions = ["volume", "v_1", "v_2", "v_3"]
        base = _make_base(infant_obs_keys=keys, infant_action_names=actions)
        try:
            spaces = base.observation_space[0].spaces
            self.assertEqual(list(spaces.keys()), keys)
            self.assertEqual(tuple(base.action_space[0].shape), (4,))
            obs, _ = base.reset(seed=0)
            infant_obs = obs[0]
            self.assertEqual(list(infant_obs.keys()), keys)
            self.assertEqual(infant_obs["infant_voice"].shape, (4,))
            self.assertEqual(infant_obs["parent_voice"].shape, (4,))
        finally:
            base.close()


class TestProtocolManagerVoiceDense(unittest.TestCase):
    def test_dense_reward_strips_preference_and_injects(self):
        base = _make_base(
            num_objects=1,
            object_counts={"RedCube": 1},
            infant_obs_keys=["infant_voice", "parent_voice"],
            infant_action_names=["volume", "v_1", "v_2", "v_3"],
        )
        env = ProtocolManager(
            InfantSignalDetector(ParentTools(base)),
            dense_reward=True,
            household_interval_steps=10**9,
            rejection=False,
        )
        try:
            env.reset(seed=0)
            env.set_fixed_protocol(
                "voice_echo",
                {"target_concept": "red_cube", "use_geta": False},
            )
            proto = env.current_protocol
            self.assertIsInstance(proto, VoiceEchoProtocol)
            self.assertEqual(proto.target_concept, "red_cube")
            proto.dense_reward_function = MagicMock(return_value=0.75)
            isd = env.env
            while not hasattr(isd, "_compute_preference_reward"):
                isd = isd.env
            isd._compute_preference_reward = MagicMock(
                return_value=(0.7, {"hold": 0.0, "look": 0.7, "total": 0.7})
            )
            infant_act = np.zeros(base.action_space[0].shape, dtype=np.float32)
            parent_act = np.zeros(base.action_space[1].shape, dtype=np.float32)
            _, r, _, _, info = env.step((infant_act, parent_act))
            self.assertAlmostEqual(info["dense_reward"], 0.75)
            self.assertEqual(info["preference_reward"]["total"], 0.0)
            self.assertAlmostEqual(float(r[0]), 0.75)
        finally:
            base.close()

    def test_fixed_protocol_finishes_episode(self):
        base = _make_base(
            infant_obs_keys=["infant_voice", "parent_voice"],
            infant_action_names=["volume", "v_1", "v_2", "v_3"],
        )
        env = ProtocolManager(
            InfantSignalDetector(ParentTools(base)),
            dense_reward=True,
            household_interval_steps=10**9,
            rejection=False,
        )
        try:
            _, info = env.reset(seed=0)
            env.set_fixed_protocol(
                "voice_echo",
                {
                    "target_concept": "red_cube",
                    "use_geta": False,
                    "max_rounds": 1,
                },
            )
            parent_act = info.get(
                "parent_action",
                np.zeros(base.action_space[1].shape, dtype=np.float32),
            )
            infant_act = np.zeros(base.action_space[0].shape, dtype=np.float32)
            done = False
            steps = 0
            # 1 語 30 + 待ち 12 = 42、余裕を見て 120
            while not done and steps < 120:
                _, _, terminated, _, info = env.step((infant_act, parent_act))
                parent_act = info.get("parent_action", parent_act)
                infant_act = np.zeros(base.action_space[0].shape, dtype=np.float32)
                done = bool(terminated)
                steps += 1
            self.assertTrue(done)
            self.assertLess(steps, 120)
            self.assertTrue(env.current_protocol.is_finished)
        finally:
            base.close()


class TestLastAllWordScores(unittest.TestCase):
    def test_commit_keeps_scores_when_unadopted(self):
        base = _make_base()
        detector = InfantSignalDetector(ParentTools(base))
        try:
            detector.reset(seed=0)
            scores = {"red_cube": 4.0, "blue_cube": 3.0}
            detector.voice_recognizer.last_raw_word_scores = {
                "red_cube": 4.0,
                "blue_cube": 3.0,
            }
            detector.voice_recognizer.last_all_word_scores = dict(scores)
            detector.voice_recognizer.word_eval_seq = 1
            detector.voice_recognizer.last_word_recognition_metrics = {
                "best_raw_score": 3.0,
                "best_boosted_score": 3.0,
                "length": 5.0,
            }
            detector._commit_word_observation(None, scores)
            self.assertEqual(detector.last_all_word_scores["red_cube"], 4.0)
            self.assertEqual(detector.word_scores_history[-1], {})
            self.assertIsNone(detector.word_history[-1])
        finally:
            base.close()


if __name__ == "__main__":
    unittest.main()
