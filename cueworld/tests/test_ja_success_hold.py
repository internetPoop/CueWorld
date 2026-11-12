"""バッチ H: 共同注意の 3 秒ホールド / 5 秒待ち / 1 回強化。"""

import unittest

import numpy as np

from cueworld.language.vocabulary import CONCEPT_TO_PHONEMES
from cueworld.protocols.base import (
    ATTENTION_HOLD_STEPS,
    ATTENTION_WAIT_STEPS,
    AttentionHoldTracker,
)
from cueworld.tests.test_ja_infant_initiative import (
    JointAttentionProtocol,
    _MockEnv,
    _signals_for_target,
)


def _signals_parent_face():
    signals = _signals_for_target(101)
    signals["gaze_targets_now"] = np.array([0.0, 0.0, 1.0], dtype=float)
    return signals


def _signals_empty():
    signals = _signals_for_target(101)
    signals["gaze_targets_now"] = np.zeros(3, dtype=float)
    return signals


class TestAttentionHoldTracker(unittest.TestCase):
    def test_unbroken_look_succeeds(self):
        tracker = AttentionHoldTracker(hold_steps=3, wait_steps=5)
        for i in range(2):
            self.assertEqual(tracker.step(101, 101, 1), "pending")
        self.assertEqual(tracker.step(101, 101, 1), "success")

    def test_other_object_resets_hold(self):
        tracker = AttentionHoldTracker(hold_steps=3, wait_steps=50)
        tracker.step(101, 101, 1)
        tracker.step(101, 101, 1)
        self.assertEqual(tracker.hold, 2)
        tracker.step(102, 101, 1)
        self.assertEqual(tracker.hold, 0)

    def test_parent_face_does_not_reset_or_advance_hold(self):
        tracker = AttentionHoldTracker(hold_steps=3, wait_steps=50)
        tracker.step(101, 101, 1)
        tracker.step(1, 101, 1)
        self.assertEqual(tracker.hold, 1)
        self.assertEqual(tracker.disengage, 0)

    def test_parent_face_resets_disengage(self):
        tracker = AttentionHoldTracker(hold_steps=30, wait_steps=50)
        tracker.step(101, 101, 1)
        tracker.step(102, 101, 1)
        self.assertEqual(tracker.disengage, 1)
        self.assertEqual(tracker.hold, 0)
        tracker.step(1, 101, 1)
        self.assertEqual(tracker.disengage, 0)
        self.assertEqual(tracker.hold, 0)

    def test_never_looked_times_out(self):
        tracker = AttentionHoldTracker(hold_steps=30, wait_steps=3)
        self.assertEqual(tracker.step(102, 101, 1), "pending")
        self.assertEqual(tracker.step(102, 101, 1), "pending")
        self.assertEqual(tracker.step(102, 101, 1), "timeout")

    def test_disengage_after_look(self):
        tracker = AttentionHoldTracker(hold_steps=30, wait_steps=3)
        tracker.step(101, 101, 1)
        self.assertEqual(tracker.step(102, 101, 1), "pending")
        self.assertEqual(tracker.step(102, 101, 1), "pending")
        self.assertEqual(tracker.step(102, 101, 1), "timeout")


class TestJointAttentionSuccessHold(unittest.TestCase):
    def setUp(self):
        self.env = _MockEnv()
        self.protocol = JointAttentionProtocol(
            self.env, cue_type="looking", rejection_check=False
        )

    def _start_looking(self, hold_steps=3, wait_steps=5):
        self.protocol.start({"fixed_target": 101, "intent_type": "informative"})
        self.protocol.parent_utterance = False
        self.protocol.state = "looking"
        self.protocol._cue_started = True
        self.protocol.attention_tracker.hold_steps = hold_steps
        self.protocol.attention_tracker.wait_steps = wait_steps

    def test_parent_informative_succeeds_without_words(self):
        self._start_looking(hold_steps=3)
        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "post_shared_attention")
        self.assertTrue(self.protocol.ja_success)

    def test_other_object_resets_hold(self):
        self._start_looking(hold_steps=3)
        self.protocol._manage_state(_signals_for_target(101))
        self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.attention_tracker.hold, 2)
        self.protocol._manage_state(_signals_for_target(102))
        self.assertEqual(self.protocol.attention_tracker.hold, 0)
        self.assertEqual(self.protocol.state, "looking")

    def test_parent_face_pauses_hold(self):
        self._start_looking(hold_steps=3)
        self.protocol._manage_state(_signals_for_target(101))
        self.protocol._manage_state(_signals_parent_face())
        self.assertEqual(self.protocol.attention_tracker.hold, 1)
        self.assertEqual(self.protocol.attention_tracker.disengage, 0)

    def test_never_looked_escalates_looking_to_pointing_then_fails(self):
        self._start_looking(hold_steps=30, wait_steps=3)
        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(102))
        self.assertEqual(self.protocol.cue_type, "pointing")
        self.assertEqual(self.protocol.state, "pointing")
        self.assertEqual(self.protocol.retry_count, 1)

        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(102))
        self.assertEqual(self.protocol.state, "failure")
        self.assertTrue(self.protocol.is_finished)

    def test_disengage_retries_then_fails_on_pointing(self):
        self.env = _MockEnv()
        self.protocol = JointAttentionProtocol(
            self.env, cue_type="pointing", rejection_check=False
        )
        self.protocol.start({"fixed_target": 101, "intent_type": "informative"})
        self.protocol.parent_utterance = False
        self.protocol.state = "pointing"
        self.protocol._cue_started = True
        self.protocol.attention_tracker.hold_steps = 30
        self.protocol.attention_tracker.wait_steps = 3

        self.protocol._manage_state(_signals_for_target(101))
        for _ in range(3):
            self.protocol._manage_state(_signals_empty())
        self.assertEqual(self.protocol.retry_count, 1)
        self.assertEqual(self.protocol.state, "pointing")
        self.assertFalse(self.protocol.is_finished)

        self.protocol._manage_state(_signals_for_target(101))
        for _ in range(3):
            self.protocol._manage_state(_signals_empty())
        self.assertEqual(self.protocol.state, "failure")

    def _enter_post_ready(self, hold_steps=3):
        self._start_looking(hold_steps=hold_steps)
        for _ in range(hold_steps):
            self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "post_shared_attention")
        self.assertTrue(self.protocol._hold_ready())

    def _drain_reply(self):
        self.protocol.utterance_queue.clear()
        self.protocol._maybe_complete_reply_round()

    def test_voice_wait_completes_after_hold(self):
        self._start_looking(hold_steps=3)
        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "post_shared_attention")
        for _ in range(ATTENTION_WAIT_STEPS):
            self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "completed")

    def test_wrong_word_does_not_increment_retry(self):
        self._enter_post_ready(hold_steps=3)
        self.protocol.parent_utterance = True
        signals = _signals_for_target(101)
        signals["words"] = {"green_cube": 3.0}
        signals["voice_reply"] = "chigauyo"
        signals["voice_segment_seq"] = 1
        self.protocol._manage_state(signals)
        self.assertEqual(self.protocol.retry_count, 0)
        self.assertEqual(self.protocol._pending_word_match, "wrong")
        self.assertTrue(self.protocol.infant_spoke)
        self.assertTrue(self.protocol._word_reacted)
        first_len = len(self.protocol.utterance_queue)
        self.protocol._manage_state(signals)
        self.assertEqual(len(self.protocol.utterance_queue), first_len)
        self.assertEqual(self.protocol.retry_count, 0)

    def test_unadopted_attempt_corrects_once(self):
        self._enter_post_ready(hold_steps=3)
        self.protocol.parent_utterance = True
        signals = _signals_for_target(101)
        signals["words"] = {}
        signals["voice_unadopted_attempt"] = False
        signals["voice_segment_seq"] = 1
        self.protocol._manage_state(signals)
        self.assertFalse(self.protocol.infant_spoke)
        self.assertEqual(len(self.protocol.utterance_queue), 0)

        signals["voice_unadopted_attempt"] = True
        signals["voice_reply"] = "chigauyo"
        signals["voice_segment_seq"] = 2
        self.protocol._manage_state(signals)
        self.assertEqual(self.protocol._pending_word_match, "unadopted")
        self.assertTrue(self.protocol.infant_spoke)
        self.assertTrue(self.protocol._word_reacted)
        self.assertEqual(self.protocol.utterance_queue[0], CONCEPT_TO_PHONEMES["correct"][0])
        first_len = len(self.protocol.utterance_queue)

        self.protocol._manage_state(signals)
        self.assertEqual(len(self.protocol.utterance_queue), first_len)

        # 同じまとまりではもう一度積まない
        signals["voice_segment_seq"] = 2
        self.protocol._manage_state(signals)
        self.assertEqual(len(self.protocol.utterance_queue), first_len)
        self.assertEqual(self.protocol.retry_count, 0)

    def test_praise_replaces_correction_and_blocks_later_correction(self):
        self._enter_post_ready(hold_steps=3)
        self.protocol.parent_utterance = True
        signals = _signals_for_target(101)
        signals["words"] = {"green_cube": 3.0}
        signals["voice_reply"] = "chigauyo"
        signals["voice_segment_seq"] = 1
        self.protocol._manage_state(signals)
        self.assertEqual(self.protocol._pending_word_match, "wrong")
        self.assertEqual(self.protocol.utterance_queue[0], CONCEPT_TO_PHONEMES["correct"][0])

        # 同じまとまりのまま「すごい」へ格上げ（seq を進めない）
        signals["words"] = {"red_cube": 0.1}
        signals["voice_reply"] = "sugoi"
        self.protocol._manage_state(signals)
        self.assertEqual(self.protocol._pending_word_match, "correct")
        self.assertTrue(self.protocol._praised)
        self.assertEqual(self.protocol.utterance_queue[0], CONCEPT_TO_PHONEMES["praise"][0])
        praised_len = len(self.protocol.utterance_queue)

        signals["words"] = {"green_cube": 3.0}
        signals["voice_reply"] = "chigauyo"
        self.protocol._manage_state(signals)
        self.assertEqual(len(self.protocol.utterance_queue), praised_len)
        self.assertEqual(self.protocol._pending_word_match, "correct")
        self.assertEqual(self.protocol.retry_count, 0)

    def test_segment_blocks_complete_until_cap(self):
        self._start_looking(hold_steps=3)
        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "post_shared_attention")
        self.protocol.voice_segment.in_segment = True
        for _ in range(ATTENTION_HOLD_STEPS):
            self.protocol.voice_segment.in_segment = True
            self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "post_shared_attention")

        remaining = ATTENTION_WAIT_STEPS - ATTENTION_HOLD_STEPS
        for _ in range(remaining):
            self.protocol.voice_segment.in_segment = True
            self.protocol._manage_state(_signals_for_target(101))
        self.assertEqual(self.protocol.state, "completed")

    def test_blocks_interrupt_only_after_success(self):
        self._start_looking(hold_steps=3)
        self.protocol.utterance_queue.append("p0")
        self.assertFalse(self.protocol.blocks_interrupt(_signals_for_target(101)))
        for _ in range(3):
            self.protocol._manage_state(_signals_for_target(101))
        self.assertTrue(self.protocol.blocks_interrupt(_signals_for_target(101)))
        self.protocol.utterance_queue.clear()
        self.protocol.voice_segment.in_segment = False
        self.assertFalse(self.protocol.blocks_interrupt(_signals_for_target(101)))

    def test_average_gaze_does_not_advance_hold(self):
        """平均だけ対象でも、瞬間が別なら hold は進まない。"""
        self._start_looking(hold_steps=3)
        signals = _signals_for_target(102)
        signals["gaze_targets"] = np.array([1.0, 0.0, 0.0], dtype=float)
        self.protocol._manage_state(signals)
        self.assertEqual(self.protocol.attention_tracker.hold, 0)
        self.assertEqual(self.protocol.state, "looking")

    def test_infant_is_engaged_uses_average_not_now(self):
        self.protocol.start({"fixed_target": 101})
        signals = _signals_for_target(101)
        signals["gaze_targets_now"] = np.array([0.0, 1.0, 0.0], dtype=float)
        self.assertTrue(self.protocol.infant_is_engaged(signals))
        signals["gaze_targets"] = np.array([0.0, 1.0, 0.0], dtype=float)
        self.assertFalse(self.protocol.infant_is_engaged(signals))


if __name__ == "__main__":
    unittest.main()
