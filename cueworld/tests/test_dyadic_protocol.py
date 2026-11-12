"""DyadicProtocol のモータキュー・状態遷移のユニットテスト。"""

import unittest

import gymnasium as gym
import numpy as np

from cueworld.protocols.base import ATTENTION_WAIT_STEPS
from cueworld.protocols.dyadic import (
    ARM_RAISE_HIT_RATE_MIN,
    DyadicProtocol,
    FACE_AWAY_TIMEOUT,
    MATCH_QUANTITY_THRESHOLDS,
    MOTOR_CHANNEL_CONFIG,
    MOTOR_IDLE_TIMEOUT,
    MOTOR_MAX_STEPS,
    PARENT_INIT_CHUNK,
    PHONEME_ECHO_STEPS,
    VOICE_CHANNEL,
)


class _MockEnv(gym.Env):
    action_num = 17
    action_map = {
        "head_pan_joint": 1,
        "head_tilt_joint": 2,
        "v_1": 14,
        "v_2": 15,
        "v_3": 16,
        "volume": 12,
    }
    phoneme_mapping = {
        "p0": np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        "p1": np.array([-1.0, -1.0, 1.0], dtype=np.float32),
    }
    client_id = 0

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(17,), dtype=np.float32)
        self.parent = {"body": 1}
        self.infant = {"body": 2, "eyes_link": 0}
        self.motor_calls = []
        self.reach_calls = []
        self.reset_arm_called = False
        self._cleared_babble = False

    def clear_babbling_chunk(self):
        self._cleared_babble = True

    def gaze_agent(self, agent_from, agent_to, yaw_link="head", action=None):
        return 0.0, 0.0

    def reset_arm_pose(self, agent, action=None):
        self.reset_arm_called = True

    def wave_arm(self, agent, action=None, phase=0.0, gesture=False):
        self.motor_calls.append(("wave_arm", phase, gesture))

    def nod_head(self, agent, action=None, phase=0.0):
        self.motor_calls.append(("nod_head", phase))

    def animate_hand_open_close(self, agent, action=None, phase=0.0, gesture=False):
        self.motor_calls.append(("animate_hand_open_close", phase, gesture))

    def hand_reach_location(self, agent, target_pos, action=None, ik_every=0, oid=-1):
        self.reach_calls.append(np.asarray(target_pos, dtype=np.float32).copy())
        return action if action is not None else {}

    def palm_reach_error(self, agent, target_pos):
        # テストでは常に到達済み扱い（到達待ちをスキップ）
        return 0.0

    def _agent_torso_pose(self, agent):
        return np.array([0.0, 0.0, 0.5], dtype=np.float32), np.array(
            [1.0, 0.0, 0.0], dtype=np.float32
        )

    def reset(self, **kwargs):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def _motor_signals(raise_on=True, open_close=0.0, face=1.0):
    return {
        "hand_height_margin": ARM_RAISE_HIT_RATE_MIN if raise_on else 0.0,
        "hand_open_close_periodicity": float(open_close),
        "babbling_salience": 0.0,
        "instant_parent_face_hit": float(face),
    }


def _face_away_signals():
    return {
        "hand_height_margin": 0.0,
        "hand_open_close_periodicity": 0.0,
        "babbling_salience": 0.0,
        "instant_parent_face_hit": 0.0,
    }


class TestDyadicProtocol(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.protocol = DyadicProtocol(self.env, rejection_check=False)
        self.protocol._compute_gesture_pos = lambda: np.array(
            [0.35, 0.0, 0.8], dtype=np.float32
        )

    def test_queue_channels_for_imitation_voice_and_motor(self):
        self.protocol._queue_channels_for_imitation(
            [("arm_raise", 0.8), (VOICE_CHANNEL, 0.6)],
            babble_chunk=["p0", "p1", "p0"],
        )
        self.assertIn("arm_raise", self.protocol.active_motor_channels)
        self.assertEqual(
            len(self.protocol.utterance_queue), 3 * PHONEME_ECHO_STEPS
        )
        self.assertEqual(self.protocol.source_phoneme, "p0")

    def test_queue_open_close_adds_arm_raise(self):
        self.protocol._queue_channels_for_imitation([("hand_open_close", 0.8)])
        self.assertIn("arm_raise", self.protocol.active_motor_channels)
        self.assertIn("hand_open_close", self.protocol.active_motor_channels)

    def test_begin_infant_imitation_opens_motor_lane(self):
        signals = {
            "hand_open_close_periodicity": 0.8,
            "hand_height_margin": 0.8,
            "babbling_salience": 0.0,
            "instant_parent_face_hit": 1.0,
        }
        ranked = DyadicProtocol.get_salient_channels(signals)
        self.protocol._begin_infant_imitation_from_salience(ranked, infant_signals=signals)
        self.assertTrue(self.protocol.motor_active)
        self.assertTrue(self.protocol.session_success)
        self.assertIn("arm_raise", self.protocol.active_motor_channels)
        self.assertIn("hand_open_close", self.protocol.active_motor_channels)
        self.assertIsNotNone(self.protocol.gesture_pos)

    def test_imitation_channels_complete_after_motor_duration(self):
        self.protocol.active_motor_channels = ["arm_raise"]
        self.protocol.motor_timers = {"arm_raise": 0}
        self.protocol.gesture_reached = True
        duration = MOTOR_CHANNEL_CONFIG["arm_raise"]["duration_steps"]

        for _ in range(duration):
            self.assertFalse(self.protocol._imitation_channels_complete())
            self.protocol._tick_motor_timers()

        self.assertTrue(self.protocol._imitation_channels_complete())

    def test_arm_raise_timer_waits_for_gesture_reach(self):
        self.protocol.active_motor_channels = ["arm_raise"]
        self.protocol.motor_timers = {"arm_raise": 0}
        self.protocol.gesture_pos = np.array([0.35, 0.0, 0.8], dtype=np.float32)
        self.protocol.gesture_reached = False
        self.protocol.gesture_wait_timer = 0
        # palm_reach_error を未到達にする
        self.env.palm_reach_error = lambda agent, target_pos: 1.0

        self.protocol._tick_motor_timers()
        self.assertEqual(self.protocol.motor_timers["arm_raise"], 0)
        self.assertFalse(self.protocol.gesture_reached)

        # タイムアウトで到達扱い
        self.protocol.gesture_wait_timer = self.protocol.gesture_reach_timeout_steps
        self.protocol._tick_motor_timers()
        self.assertTrue(self.protocol.gesture_reached)
        self.assertEqual(self.protocol.motor_timers["arm_raise"], 1)

    def test_determine_action_applies_wave_during_parent_cue(self):
        self.protocol.infant_initiative = False
        self.protocol.state = "performing_parent_cue"
        self.protocol.active_motor_channels = ["arm_raise"]
        self.protocol.motor_timers = {"arm_raise": 0}
        self.protocol.gesture_pos = np.array([0.35, 0.0, 0.8], dtype=np.float32)
        self.protocol.gesture_reached = True

        self.protocol._determine_action()
        self.assertEqual(len(self.env.motor_calls), 1)
        self.assertEqual(self.env.motor_calls[0][0], "wave_arm")
        self.assertTrue(self.env.motor_calls[0][2])  # gesture=True
        self.assertEqual(len(self.env.reach_calls), 1)
        self.assertFalse(self.env.reset_arm_called)

    def test_determine_action_waves_during_infant_motor(self):
        """幼児始動の真似でも腕を振る。"""
        self.protocol.infant_initiative = True
        self.protocol.motor_active = True
        self.protocol._motor_above_now = True
        self.protocol.active_motor_channels = ["arm_raise"]
        self.protocol.motor_timers = {"arm_raise": 0}
        self.protocol.gesture_pos = np.array([0.35, 0.0, 0.8], dtype=np.float32)
        self.protocol.gesture_reached = True

        self.protocol._determine_action()
        self.assertEqual(len(self.env.motor_calls), 1)
        self.assertEqual(self.env.motor_calls[0][0], "wave_arm")
        self.assertEqual(len(self.env.reach_calls), 1)
        self.assertFalse(self.env.reset_arm_called)

    def test_manage_state_waiting_infant_triggers_motor_lane(self):
        self.protocol.infant_initiative = True
        self.protocol.state = "waiting_infant"
        signals = _motor_signals(raise_on=True)
        self.protocol._manage_state(signals)
        self.assertTrue(self.protocol.motor_active)
        self.assertTrue(self.protocol.session_success)
        self.assertEqual(self.protocol.state, "waiting_babble")
        self.assertIn("arm_raise", self.protocol.active_motor_channels)
        self.assertNotIn("hand_open_close", self.protocol.active_motor_channels)
        self.assertIsNotNone(self.protocol.gesture_pos)

    def test_waiting_infant_babble_starts_echo(self):
        self.protocol.infant_initiative = True
        self.protocol.state = "waiting_infant"
        signals = {
            "babbling_chunk": ["p0", "p3", "p1"],
            "instant_parent_face_hit": 1.0,
        }
        self.protocol._manage_state(signals)
        self.assertEqual(self.protocol.state, "echoing")
        self.assertTrue(self.protocol.voice_active)
        self.assertEqual(self.protocol.current_babble_chunk, ["p0", "p3", "p1"])
        self.assertEqual(
            len(self.protocol.utterance_queue), 3 * PHONEME_ECHO_STEPS
        )

    def test_parent_cue_voice_uses_init_chunk_once(self):
        self.protocol.infant_initiative = False
        self.protocol._begin_parent_cue([VOICE_CHANNEL])
        expected_len = len(PARENT_INIT_CHUNK) * PHONEME_ECHO_STEPS
        self.assertEqual(len(self.protocol.utterance_queue), expected_len)
        # 先頭は p0 が 6 個
        queued = list(self.protocol.utterance_queue)
        self.assertEqual(queued[:PHONEME_ECHO_STEPS], ["p0"] * PHONEME_ECHO_STEPS)
        self.assertEqual(
            queued[PHONEME_ECHO_STEPS : 2 * PHONEME_ECHO_STEPS],
            ["p1"] * PHONEME_ECHO_STEPS,
        )

    def test_parent_cue_completes_then_waits_for_babble(self):
        self.protocol.infant_initiative = False
        self.protocol._begin_parent_cue([VOICE_CHANNEL])
        self.protocol.utterance_queue.clear()
        self.protocol._manage_state({"instant_parent_face_hit": 1.0})
        self.assertEqual(self.protocol.state, "waiting_babble")
        # voice only → gesture_pos は不要
        self.assertIsNone(self.protocol.gesture_pos)
        self.assertTrue(self.protocol.response_peak_active)
        self.assertFalse(self.protocol.session_success)

    def test_echo_complete_sets_session_success_and_waits(self):
        self.protocol.infant_initiative = True
        self.protocol._begin_echo(["p0", "p1", "p0"])
        self.protocol.utterance_queue.clear()
        self.protocol._manage_state({"instant_parent_face_hit": 1.0})
        self.assertTrue(self.protocol.session_success)
        self.assertEqual(self.protocol.round_count, 1)
        self.assertEqual(self.protocol.state, "waiting_babble")
        self.assertTrue(self.protocol.voice_active)
        self.assertTrue(self.env._cleared_babble)

    def test_max_rounds_closes_voice_lane(self):
        self.protocol.infant_initiative = True
        self.protocol.max_rounds = 2
        self.protocol._begin_echo(["p0", "p1", "p0"])
        self.protocol.utterance_queue.clear()
        self.protocol._manage_state({"instant_parent_face_hit": 1.0})
        self.assertEqual(self.protocol.state, "waiting_babble")
        self.protocol._begin_echo(["p2", "p3", "p4"])
        self.protocol.utterance_queue.clear()
        self.protocol._manage_state({"instant_parent_face_hit": 1.0})
        self.assertEqual(self.protocol.round_count, 2)
        self.assertFalse(self.protocol.voice_active)
        self.assertTrue(self.protocol.session_success)
        # モータが無いのでセッション終了
        self.assertEqual(self.protocol.state, self.protocol.success_state)
        self.assertTrue(self.protocol.is_finished)

    def test_waiting_babble_timeout_without_success_fails(self):
        self.protocol.infant_initiative = False
        self.protocol.state = "waiting_babble"
        self.protocol.session_success = False
        self.protocol.state_timer = self.protocol.waiting_babble_timeout_steps + 1
        self.protocol._manage_state({"instant_parent_face_hit": 1.0})
        self.assertEqual(self.protocol.state, self.protocol.failure_state)
        self.assertTrue(self.protocol.is_finished)
        self.assertFalse(self.protocol.session_success)

    def test_waiting_babble_timeout_after_success_completes(self):
        self.protocol.state = "waiting_babble"
        self.protocol.session_success = True
        self.protocol.voice_active = True
        self.protocol.round_count = 1
        self.protocol.state_timer = self.protocol.waiting_babble_timeout_steps + 1
        self.protocol._manage_state({"instant_parent_face_hit": 1.0})
        self.assertEqual(self.protocol.state, self.protocol.success_state)
        self.assertTrue(self.protocol.is_finished)

    def test_persistence_decrements_per_round(self):
        self.protocol.retry_count = 0
        self.protocol.state = "echoing"
        self.protocol.round_count = 0
        self.protocol._calculate_persistence({})
        self.assertEqual(self.protocol.persistence, 5.0)
        self.protocol.round_count = 1
        self.protocol._calculate_persistence({})
        self.assertEqual(self.protocol.persistence, 4.0)
        self.protocol.round_count = 4
        self.protocol._calculate_persistence({})
        self.assertEqual(self.protocol.persistence, 1.0)
        self.protocol.round_count = 10
        self.protocol._calculate_persistence({})
        self.assertEqual(self.protocol.persistence, 1.0)

    def test_engagement_rejects_degenerate_arm_ext_and_orientation(self):
        """arm_ext / orientation_to_parent の縮退では engagement しない。"""
        signals = {
            "arm_ext": 1.0,
            "orientation_to_parent": 1.0,
            "hand_movement_size": 1.0,
            "hand_movement_regularity": 1.0,
            "hand_open_close_periodicity": 0.0,
            "head_nod_periodicity": 1.0,
            "hand_height_margin": 0.0,
            "babbling_salience": 0.0,
        }
        self.assertFalse(self.protocol._detect_infant_response(signals))

    def test_engagement_accepts_arm_raise_at_tau(self):
        signals = {
            "hand_height_margin": ARM_RAISE_HIT_RATE_MIN,
            "hand_open_close_periodicity": 0.0,
            "babbling_salience": 0.0,
        }
        self.assertTrue(self.protocol._detect_infant_response(signals))

    def test_engagement_rejects_raise_below_tau(self):
        signals = {
            "hand_height_margin": 0.7,
            "hand_open_close_periodicity": 1.0,
            "babbling_salience": 0.0,
        }
        self.assertFalse(self.protocol._detect_infant_response(signals))

    def test_engagement_accepts_babbling_chunk(self):
        self.assertTrue(
            self.protocol._detect_infant_response(
                {"babbling_chunk": ["p0", "p1", "p0"]}
            )
        )

    def test_response_match_raise_keep_without_open_close(self):
        self.protocol.active_channels = ["arm_raise"]
        self.protocol.response_peak_active = True
        signals = {
            "hand_height_margin": ARM_RAISE_HIT_RATE_MIN,
            "hand_open_close_periodicity": 0.0,
            "babbling_salience": 0.0,
        }
        self.protocol._update_response_match(signals)
        self.assertEqual(self.protocol.channel_on["arm_raise"], 1)
        self.assertEqual(self.protocol.channel_on["hand_open_close"], 0)
        self.assertGreaterEqual(
            self.protocol.channel_peak_scores["arm_raise"],
            MATCH_QUANTITY_THRESHOLDS["arm_raise"],
        )
        self.assertEqual(self.protocol.response_matched, 1)

    def test_response_match_raise_and_open_close_requires_both(self):
        self.protocol.active_channels = ["arm_raise", "hand_open_close"]
        self.protocol.response_peak_active = True
        signals = {
            "hand_height_margin": ARM_RAISE_HIT_RATE_MIN,
            "hand_open_close_periodicity": 0.0,
            "babbling_salience": 0.0,
        }
        self.protocol._update_response_match(signals)
        self.assertEqual(self.protocol.response_matched, 0)

        signals["hand_open_close_periodicity"] = MATCH_QUANTITY_THRESHOLDS[
            "hand_open_close"
        ]
        self.protocol._update_response_match(signals)
        self.assertEqual(self.protocol.channel_on["hand_open_close"], 1)
        self.assertEqual(self.protocol.response_matched, 1)

    def test_response_match_extra_open_close_not_penalized(self):
        self.protocol.active_channels = ["arm_raise"]
        self.protocol.response_peak_active = True
        signals = {
            "hand_height_margin": ARM_RAISE_HIT_RATE_MIN,
            "hand_open_close_periodicity": 0.9,
            "babbling_salience": 0.0,
        }
        self.protocol._update_response_match(signals)
        self.assertEqual(self.protocol.channel_on["hand_open_close"], 1)
        self.assertEqual(self.protocol.response_matched, 1)

    def test_response_match_raise_off_zeros_open_close(self):
        self.protocol.active_channels = ["arm_raise"]
        self.protocol.response_peak_active = True
        signals = {
            "hand_height_margin": 0.0,
            "hand_open_close_periodicity": 1.0,
            "babbling_salience": 0.0,
        }
        salience = DyadicProtocol.compute_channel_salience(signals)
        self.assertEqual(salience["arm_raise"], 0.0)
        self.assertEqual(salience["hand_open_close"], 0.0)
        self.assertNotIn("arm_wave", salience)
        self.assertNotIn("head_nod", salience)
        self.protocol._update_response_match(signals)
        self.assertEqual(self.protocol.channel_on["arm_raise"], 0)
        self.assertEqual(self.protocol.response_matched, 0)

    def test_parent_channels_override_from_context(self):
        self.protocol.start(
            {
                "infant_initiative": False,
                "parent_channels": ["arm_raise"],
            }
        )
        self.assertEqual(self.protocol.active_channels, ["arm_raise"])
        self.assertEqual(self.protocol.parent_channels_override, ["arm_raise"])

    def test_parent_channels_override_open_close_adds_raise(self):
        self.protocol.start(
            {
                "infant_initiative": False,
                "parent_channels": ["hand_open_close"],
            }
        )
        self.assertEqual(
            self.protocol.active_channels, ["arm_raise", "hand_open_close"]
        )

    def test_start_infant_with_babbling_chunk_echoes(self):
        self.protocol.start(
            {
                "infant_initiative": True,
                "babbling_chunk": ["p0", "p1", "p2"],
            }
        )
        self.assertEqual(self.protocol.state, "echoing")
        self.assertTrue(self.protocol.voice_active)
        self.assertEqual(self.protocol.current_babble_chunk, ["p0", "p1", "p2"])

    def test_wait_timeouts_are_attention_wait_steps(self):
        self.assertEqual(
            self.protocol.waiting_infant_timeout_steps, ATTENTION_WAIT_STEPS
        )
        self.assertEqual(
            self.protocol.waiting_babble_timeout_steps, ATTENTION_WAIT_STEPS
        )
        self.assertEqual(
            self.protocol.waiting_infant_response_timeout_steps, ATTENTION_WAIT_STEPS
        )
        self.assertEqual(self.protocol.motor_max_steps, MOTOR_MAX_STEPS)
        self.assertEqual(self.protocol.motor_idle_timeout_steps, MOTOR_IDLE_TIMEOUT)
        self.assertEqual(self.protocol.face_away_timeout_steps, FACE_AWAY_TIMEOUT)

    def test_waiting_babble_no_retry_of_parent_cue(self):
        """喃語待ちタイムアウトでは見本を言い直さない。"""
        self.protocol.start(
            {
                "infant_initiative": False,
                "parent_channels": ["arm_raise"],
            }
        )
        self.protocol.utterance_queue.clear()
        self.protocol.active_motor_channels = []
        self.protocol.motor_timers = {}
        self.protocol.gesture_reached = True
        self.protocol._manage_state({"instant_parent_face_hit": 1.0})
        self.assertEqual(self.protocol.state, "waiting_babble")

        self.protocol.state_timer = self.protocol.waiting_babble_timeout_steps + 1
        self.protocol._manage_state({"instant_parent_face_hit": 1.0})
        self.assertEqual(self.protocol.state, self.protocol.failure_state)
        self.assertTrue(self.protocol.is_finished)
        self.assertFalse(self.protocol.session_success)

    def test_blocks_interrupt_during_segment_or_utterance(self):
        self.protocol.state = "waiting_babble"
        self.protocol.voice_segment.in_segment = True
        self.assertTrue(self.protocol.blocks_interrupt({}))
        self.protocol.voice_segment.in_segment = False
        self.assertFalse(self.protocol.blocks_interrupt({}))
        self.protocol.state = "echoing"
        self.protocol.utterance_queue.append("p0")
        self.assertTrue(self.protocol.blocks_interrupt({}))
        self.protocol.utterance_queue.clear()
        self.assertFalse(self.protocol.blocks_interrupt({}))

    def test_sample_parent_channels_always_includes_arm_raise(self):
        for _ in range(40):
            channels = self.protocol._sample_parent_channels()
            self.assertIn("arm_raise", channels)
            self.assertNotIn("arm_wave", channels)
            self.assertNotIn("head_nod", channels)
            if "hand_open_close" in channels:
                self.assertIn("arm_raise", channels)

    # ---- モータ持続 ----

    def test_motor_wall_timer_closes_after_max_steps(self):
        self.protocol.infant_initiative = True
        self.protocol.state = "waiting_infant"
        self.protocol._manage_state(_motor_signals(raise_on=True))
        self.assertTrue(self.protocol.motor_active)
        for _ in range(MOTOR_MAX_STEPS - 1):
            self.protocol._manage_state(_motor_signals(raise_on=True))
            self.assertTrue(self.protocol.motor_active)
            self.assertFalse(self.protocol.is_finished)
        self.protocol._manage_state(_motor_signals(raise_on=True))
        self.assertFalse(self.protocol.motor_active)
        self.assertTrue(self.protocol.is_finished)
        self.assertEqual(self.protocol.state, self.protocol.success_state)

    def test_motor_idle_resumes_within_timeout(self):
        self.protocol.infant_initiative = True
        self.protocol.state = "waiting_infant"
        self.protocol._manage_state(_motor_signals(raise_on=True))
        # 外す
        for _ in range(10):
            self.protocol._manage_state(_motor_signals(raise_on=False))
            self.assertTrue(self.protocol.motor_active)
        # 戻す
        self.protocol._manage_state(_motor_signals(raise_on=True))
        self.assertEqual(self.protocol.motor_idle_timer, 0)
        self.assertTrue(self.protocol.motor_active)

    def test_motor_idle_timeout_closes_lane(self):
        self.protocol.infant_initiative = True
        self.protocol.state = "waiting_infant"
        self.protocol._manage_state(_motor_signals(raise_on=True))
        for _ in range(MOTOR_IDLE_TIMEOUT):
            self.protocol._manage_state(_motor_signals(raise_on=False))
        self.assertFalse(self.protocol.motor_active)
        self.assertTrue(self.protocol.is_finished)
        self.assertTrue(self.protocol.session_success)

    # ---- 声とモータの並走 ----

    def test_voice_and_motor_session_ends_on_slower_lane(self):
        self.protocol.infant_initiative = True
        self.protocol.state = "waiting_infant"
        # 同時に喃語とモータ
        signals = _motor_signals(raise_on=True)
        signals["babbling_chunk"] = ["p0", "p1", "p0"]
        self.protocol._manage_state(signals)
        self.assertTrue(self.protocol.voice_active)
        self.assertTrue(self.protocol.motor_active)
        self.assertEqual(self.protocol.state, "echoing")
        # エコー完了 → 声は waiting、モータは継続
        self.protocol.utterance_queue.clear()
        self.protocol._manage_state(_motor_signals(raise_on=True))
        self.assertTrue(self.protocol.session_success)
        self.assertTrue(self.protocol.voice_active)
        self.assertTrue(self.protocol.motor_active)
        self.assertFalse(self.protocol.is_finished)
        # 声レーンをタイムアウトで閉じる
        self.protocol.state_timer = self.protocol.waiting_babble_timeout_steps + 1
        self.protocol._manage_state(_motor_signals(raise_on=True))
        self.assertFalse(self.protocol.voice_active)
        self.assertTrue(self.protocol.motor_active)
        self.assertFalse(self.protocol.is_finished)
        # モータ壁時計で閉じる
        remaining = MOTOR_MAX_STEPS - self.protocol.motor_wall_timer
        for _ in range(remaining):
            self.protocol._manage_state(_motor_signals(raise_on=True))
        self.assertTrue(self.protocol.is_finished)
        self.assertEqual(self.protocol.state, self.protocol.success_state)

    def test_echoing_adds_motor_midway(self):
        self.protocol.infant_initiative = True
        self.protocol._begin_echo(["p0", "p1", "p0"])
        self.assertFalse(self.protocol.motor_active)
        self.protocol._manage_state(_motor_signals(raise_on=True))
        self.assertTrue(self.protocol.motor_active)
        self.assertEqual(self.protocol.state, "echoing")
        self.assertTrue(self.protocol.voice_active)

    # ---- 親始動の早期モータ反応 ----

    def test_parent_cue_motor_reacts_during_voice_demo(self):
        self.protocol.start(
            {
                "infant_initiative": False,
                "parent_channels": ["voice", "arm_raise"],
            }
        )
        self.assertEqual(self.protocol.state, "performing_parent_cue")
        self.assertTrue(self.protocol._voice_active())
        # 声の見本中でもモータ反応
        self.protocol._manage_state(_motor_signals(raise_on=True))
        self.assertTrue(self.protocol.motor_active)
        self.assertTrue(self.protocol.session_success)
        # 声の見本は残る
        self.assertTrue(self.protocol._voice_active())
        self.assertEqual(self.protocol.state, "performing_parent_cue")

    def test_parent_cue_babble_ignored_while_voice_demo_pending(self):
        self.protocol.start(
            {
                "infant_initiative": False,
                "parent_channels": ["voice", "arm_raise"],
            }
        )
        signals = {
            "babbling_chunk": ["p0", "p1", "p2"],
            "instant_parent_face_hit": 1.0,
            "hand_height_margin": 0.0,
            "hand_open_close_periodicity": 0.0,
            "babbling_salience": 0.0,
        }
        self.protocol._manage_state(signals)
        self.assertEqual(self.protocol.state, "performing_parent_cue")
        self.assertFalse(self.protocol.voice_active)
        # 見本のキューは init chunk のまま
        self.assertGreater(len(self.protocol.utterance_queue), 0)

    def test_parent_cue_babble_after_voice_demo_starts_echo(self):
        self.protocol.start(
            {
                "infant_initiative": False,
                "parent_channels": ["voice"],
            }
        )
        self.protocol.utterance_queue.clear()  # 声の見本終了
        signals = {
            "babbling_chunk": ["p0", "p1", "p2"],
            "instant_parent_face_hit": 1.0,
            "hand_height_margin": 0.0,
            "hand_open_close_periodicity": 0.0,
            "babbling_salience": 0.0,
        }
        self.protocol._manage_state(signals)
        self.assertEqual(self.protocol.state, "echoing")
        self.assertTrue(self.protocol.voice_active)
        self.assertEqual(self.protocol.current_babble_chunk, ["p0", "p1", "p2"])

    # ---- 顔離脱 ----

    def test_face_away_timeout_fails_without_success(self):
        self.protocol.infant_initiative = True
        self.protocol.state = "waiting_infant"
        for _ in range(FACE_AWAY_TIMEOUT):
            self.protocol._manage_state(_face_away_signals())
        self.assertTrue(self.protocol.is_finished)
        self.assertEqual(self.protocol.state, self.protocol.failure_state)
        self.assertFalse(self.protocol.session_success)

    def test_face_away_timeout_succeeds_after_motor(self):
        self.protocol.infant_initiative = True
        self.protocol.state = "waiting_infant"
        self.protocol._manage_state(_motor_signals(raise_on=True, face=1.0))
        self.assertTrue(self.protocol.session_success)
        # モータを外して待ちへ（顔時計が進む）
        for _ in range(FACE_AWAY_TIMEOUT):
            self.protocol._manage_state(_face_away_signals())
        self.assertTrue(self.protocol.is_finished)
        self.assertEqual(self.protocol.state, self.protocol.success_state)

    def test_face_clock_pauses_during_parent_cue(self):
        self.protocol.start(
            {
                "infant_initiative": False,
                "parent_channels": ["arm_raise"],
            }
        )
        for _ in range(FACE_AWAY_TIMEOUT + 5):
            self.protocol._manage_state(_face_away_signals())
        self.assertEqual(self.protocol.state, "performing_parent_cue")
        self.assertFalse(self.protocol.is_finished)
        self.assertEqual(self.protocol.face_away_timer, 0)

    def test_face_clock_pauses_while_motor_above_threshold(self):
        self.protocol.infant_initiative = True
        self.protocol.state = "waiting_infant"
        self.protocol._manage_state(_motor_signals(raise_on=True, face=1.0))
        for _ in range(FACE_AWAY_TIMEOUT + 5):
            # 顔は外れているが、モータ閾値上なので時計停止
            signals = _motor_signals(raise_on=True, face=0.0)
            self.protocol._manage_state(signals)
        self.assertFalse(self.protocol.is_finished)
        self.assertEqual(self.protocol.face_away_timer, 0)

    def test_face_clock_pauses_during_echo_utterance(self):
        self.protocol.infant_initiative = True
        self.protocol._begin_echo(["p0", "p1", "p0"])
        for _ in range(FACE_AWAY_TIMEOUT + 5):
            self.protocol._manage_state(_face_away_signals())
        # 発話キューが残っているあいだは閉じない
        self.assertFalse(self.protocol.is_finished)
        self.assertEqual(self.protocol.state, "echoing")


if __name__ == "__main__":
    unittest.main()
