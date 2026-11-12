"""共同注意セッション（joint_attention_session.md）のユニットテスト。"""

import unittest

import numpy as np

from cueworld.language.vocabulary import CONCEPT_TO_PHONEMES
from cueworld.protocols.base import ATTENTION_WAIT_STEPS
from cueworld.protocols.joint_attention import MAX_JA_WORD_ROUNDS, JointAttentionProtocol
from cueworld.tests.test_ja_infant_initiative import _MockEnv, _signals_for_target
from cueworld.tests.test_ja_success_hold import _signals_empty


class TestJaSession(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.protocol = JointAttentionProtocol(
            self.env, cue_type="looking", rejection_check=False
        )

    def _enter_post(self, hold_steps=3):
        self.protocol.start({"fixed_target": 101, "intent_type": "informative"})
        self.protocol.parent_utterance = True
        self.protocol.state = "looking"
        self.protocol._cue_started = True
        self.protocol.attention_tracker.hold_steps = hold_steps
        self.protocol.attention_tracker.wait_steps = ATTENTION_WAIT_STEPS
        for _ in range(hold_steps):
            self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "post_shared_attention")
        self.assertTrue(self.protocol.session_success)

    def _drain_reply(self):
        self.protocol.utterance_queue.clear()
        self.protocol._maybe_complete_reply_round()

    def test_gaze_away_after_success_closes_success(self):
        self._enter_post(hold_steps=3)
        self.protocol.attention_tracker.wait_steps = 3
        for _ in range(3):
            self.protocol._manage_state(_signals_empty())
        self.assertEqual(self.protocol.state, "completed")
        self.assertTrue(self.protocol.session_success)
        self.assertEqual(self.protocol.cue_type, "looking")

    def test_two_segments_append_word_match(self):
        self._enter_post(hold_steps=3)
        signals = _signals_for_target(101)
        signals["voice_reply"] = "chigauyo"
        signals["words"] = {"green_cube": 3.0}
        signals["voice_segment_seq"] = 1
        self.protocol._manage_state(signals)
        self._drain_reply()
        self.assertEqual(self.protocol.word_match, ["wrong"])
        self.assertEqual(self.protocol.round_count, 1)

        signals["voice_reply"] = "sugoi"
        signals["words"] = {"red_cube": 0.0}
        signals["voice_segment_seq"] = 2
        self.protocol._manage_state(signals)
        self._drain_reply()
        self.assertEqual(self.protocol.word_match, ["wrong", "correct"])
        self.assertEqual(self.protocol.round_count, 2)

    def test_cleared_reply_not_in_word_match(self):
        self._enter_post(hold_steps=3)
        signals = _signals_for_target(101)
        signals["voice_reply"] = "chigauyo"
        signals["words"] = {"green_cube": 3.0}
        signals["voice_segment_seq"] = 1
        self.protocol._manage_state(signals)
        self.assertEqual(self.protocol.utterance_queue[0], CONCEPT_TO_PHONEMES["correct"][0])

        signals["voice_reply"] = "sugoi"
        signals["words"] = {"red_cube": 0.0}
        self.protocol._manage_state(signals)
        self.assertEqual(self.protocol.utterance_queue[0], CONCEPT_TO_PHONEMES["praise"][0])
        self._drain_reply()
        self.assertEqual(self.protocol.word_match, ["correct"])
        self.assertEqual(self.protocol.round_count, 1)

    def test_max_rounds_closes(self):
        self._enter_post(hold_steps=3)
        for i in range(MAX_JA_WORD_ROUNDS):
            signals = _signals_for_target(101)
            signals["voice_reply"] = "sugoi"
            signals["words"] = {"red_cube": 0.0}
            signals["voice_segment_seq"] = i + 1
            self.protocol._manage_state(signals)
            self._drain_reply()
        self.assertEqual(self.protocol.round_count, MAX_JA_WORD_ROUNDS)
        self.assertEqual(self.protocol.state, "completed")

    def test_persistence_drops_per_round(self):
        self._enter_post(hold_steps=3)
        signals = _signals_for_target(101)
        self.protocol._calculate_persistence(signals)
        base = float(self.protocol.persistence)

        signals["voice_reply"] = "sugoi"
        signals["words"] = {"red_cube": 0.0}
        signals["voice_segment_seq"] = 1
        self.protocol._manage_state(signals)
        self._drain_reply()
        self.assertEqual(self.protocol.round_count, 1)
        self.protocol._calculate_persistence(signals)
        self.assertAlmostEqual(self.protocol.persistence, max(1.0, base - 1.0))

    def test_no_reply_before_hold(self):
        self.protocol.start({"fixed_target": 101, "intent_type": "informative"})
        self.protocol.parent_utterance = True
        self.protocol.state = "looking"
        self.protocol._cue_started = True
        self.protocol.attention_tracker.hold_steps = 30
        signals = _signals_for_target(101)
        signals["voice_reply"] = "sugoi"
        signals["words"] = {"red_cube": 0.0}
        signals["voice_segment_seq"] = 1
        self.protocol._manage_state(signals)
        self.assertFalse(self.protocol._reply_pending)
        self.assertEqual(self.protocol.word_match, [])
        self.assertEqual(self.protocol._last_voice_segment_seq, 1)

    def test_rejection_keeps_session_success(self):
        self._enter_post(hold_steps=3)
        self.protocol.rejection_check = True
        self.protocol.detect_rejection = lambda signals: True
        self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "failure")
        self.assertTrue(self.protocol.is_finished)
        self.assertTrue(self.protocol.session_success)


if __name__ == "__main__":
    unittest.main()
