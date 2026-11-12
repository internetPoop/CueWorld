"""畳んだ音素キューと平坦な DP、下駄、返事、語彙の不変条件。"""

import unittest

from cueworld.language.vocabulary import (
    CONCEPT_TO_PHONEMES,
    RESERVED_PHONEME_CODES,
)
from cueworld.language.voice_recognizer import (
    ADOPT_RAW_LIMIT,
    REPLY_CHIGAUYO,
    REPLY_OSHII,
    REPLY_SUGOI,
    VoiceRecognizer,
)


def _recognizer():
    return VoiceRecognizer(dict(CONCEPT_TO_PHONEMES))


class TestVocabularyInvariants(unittest.TestCase):
    def test_length_no_repeat_start_end_and_hamming(self):
        codes = [tuple(v) for v in CONCEPT_TO_PHONEMES.values()]
        reserved = [tuple(v) for v in RESERVED_PHONEME_CODES]
        self.assertEqual(len(codes), 26)
        self.assertEqual(len(reserved), 10)
        starts = set()
        ends = set()
        below4 = []
        all_codes = codes + reserved
        for code in all_codes:
            self.assertEqual(len(code), 5)
            self.assertTrue(all(code[i] != code[i + 1] for i in range(4)))
            starts.add(code[0])
            ends.add(code[-1])
        self.assertFalse(starts & ends)
        for i, a in enumerate(all_codes):
            for b in all_codes[i + 1 :]:
                dist = sum(x != y for x, y in zip(a, b))
                self.assertGreaterEqual(dist, 3)
                if dist < 4:
                    below4.append((a, b, dist))
        word_below4 = []
        for i, a in enumerate(codes):
            for b in codes[i + 1 :]:
                dist = sum(x != y for x, y in zip(a, b))
                if dist < 4:
                    word_below4.append((a, b, dist))
        self.assertGreater(len(word_below4), 0)
        self.assertLess(len(word_below4), 40)

    def test_reserved_codes_are_not_in_dictionary(self):
        word_set = {tuple(v) for v in CONCEPT_TO_PHONEMES.values()}
        for code in RESERVED_PHONEME_CODES:
            self.assertNotIn(tuple(code), word_set)


class TestAlignCosts(unittest.TestCase):
    def setUp(self):
        self.rec = _recognizer()
        self.word = "declare"
        self.y = list(CONCEPT_TO_PHONEMES[self.word])

    def test_exact_match_is_zero(self):
        cost, last = self.rec.align(self.y, self.y, allow_free_trail=False)
        self.assertEqual(cost, 0.0)
        self.assertEqual(last, 4)

    def test_one_substitution_is_one(self):
        x = list(self.y)
        x[2] = "p1" if x[2] != "p1" else "p2"
        cost, _ = self.rec.align(x, self.y, allow_free_trail=False)
        self.assertEqual(cost, 1.0)

    def test_middle_insertion_is_one(self):
        x = self.y[:2] + ["p7"] + self.y[2:]
        cost, _ = self.rec.align(x, self.y, allow_free_trail=True)
        self.assertEqual(cost, 1.0)

    def test_middle_deletion_is_one(self):
        x = self.y[:2] + self.y[3:]
        cost, _ = self.rec.align(x, self.y, allow_free_trail=False)
        self.assertEqual(cost, 1.0)

    def test_one_leading_extra_is_001(self):
        x = ["p3"] + self.y
        cost, last = self.rec.align(x, self.y, allow_free_trail=True)
        self.assertAlmostEqual(cost, 0.01)
        self.assertEqual(last, 5)

    def test_two_leading_extras_are_002(self):
        x = ["p3", "p2"] + self.y
        cost, _ = self.rec.align(x, self.y, allow_free_trail=False)
        self.assertAlmostEqual(cost, 0.02)

    def test_sixth_trail_is_free_and_beats_leading_drop(self):
        x = self.y + ["p0"]
        cost, last = self.rec.align(x, self.y, allow_free_trail=True)
        self.assertEqual(cost, 0.0)
        self.assertEqual(last, 4)
        lead = ["p0"] + self.y
        lead_cost, _ = self.rec.align(lead, self.y, allow_free_trail=True)
        self.assertAlmostEqual(lead_cost, 0.01)
        self.assertLess(cost, lead_cost)

    def test_silence_lengths_have_no_free_trail(self):
        for n in (3, 4, 5):
            x = self.y[:n]
            cost, _ = self.rec.align(x, self.y, allow_free_trail=False)
            self.assertEqual(cost, float(5 - n))


class TestGetaAndAdoption(unittest.TestCase):
    def test_contiguous_triple_applies_geta(self):
        rec = _recognizer()
        target = "red_cube"
        y = list(CONCEPT_TO_PHONEMES[target])
        rec.set_target(target)
        observed = y[:3] + ["p7", "p6", "p5"]
        rows, boosted = rec._rank_words(observed, allow_free_trail=True)
        by_word = {w: (raw, b) for w, raw, b, _ in rows}
        self.assertAlmostEqual(by_word[target][1], by_word[target][0] - 1.0)

    def test_gapped_triple_gets_no_geta(self):
        rec = _recognizer()
        target = "red_cube"
        y = list(CONCEPT_TO_PHONEMES[target])
        rec.set_target(target)
        observed = [y[0], y[1], "p7", y[2], "p6", "p5"]
        self.assertFalse(rec._has_contiguous_triple(observed, y))

    def test_geta_may_beat_sound_only_first(self):
        rec = VoiceRecognizer(
            {
                "other": ["p0", "p1", "p2", "p3", "p4"],
                "target": ["p0", "p1", "p2", "p5", "p6"],
            }
        )
        rec.set_target("target")
        observed = ["p0", "p1", "p2", "p5", "p7"]
        word, scores = rec.evaluate_queue(observed, allow_free_trail=False, cut=False)
        self.assertEqual(word, "target")
        self.assertLess(scores["target"], scores["other"])

    def test_cost_three_is_not_adopted_two_oh_one_is(self):
        rec = VoiceRecognizer({"a": ["p0", "p1", "p2", "p3", "p4"]})
        word_high, _ = rec.evaluate_queue(["p7", "p6"], allow_free_trail=False, cut=False)
        self.assertIsNone(word_high)
        two_oh_one = ["p7", "p0", "p1", "p3", "p4"]
        cost, _ = rec.align(two_oh_one, ["p0", "p1", "p2", "p3", "p4"], allow_free_trail=False)
        self.assertAlmostEqual(cost, 1.01)
        word201, _ = rec.evaluate_queue(two_oh_one, allow_free_trail=False, cut=False)
        self.assertEqual(word201, "a")


class TestReplyKinds(unittest.TestCase):
    def test_perfect_target_is_sugoi(self):
        rec = _recognizer()
        target = "red_cube"
        rec.set_target(target)
        word, _ = rec.evaluate_queue(
            CONCEPT_TO_PHONEMES[target],
            allow_free_trail=False,
            speak_now=True,
            cut=False,
        )
        self.assertEqual(word, target)
        self.assertEqual(rec.last_reply_kind, REPLY_SUGOI)

    def test_one_substitution_is_oshii(self):
        rec = _recognizer()
        target = "red_cube"
        rec.set_target(target)
        x = list(CONCEPT_TO_PHONEMES[target])
        x[2] = "p1" if x[2] != "p1" else "p2"
        rec.evaluate_queue(x, allow_free_trail=False, speak_now=True, cut=False)
        self.assertEqual(rec.last_reply_kind, REPLY_OSHII)

    def test_unrelated_is_chigauyo(self):
        rec = _recognizer()
        rec.set_target("red_cube")
        other = list(CONCEPT_TO_PHONEMES["prompt"])
        rec.evaluate_queue(other, allow_free_trail=False, speak_now=True, cut=False)
        self.assertEqual(rec.last_reply_kind, REPLY_CHIGAUYO)

    def test_no_target_has_no_reply(self):
        rec = _recognizer()
        rec.evaluate_queue(
            CONCEPT_TO_PHONEMES["declare"],
            allow_free_trail=False,
            speak_now=True,
            cut=False,
        )
        self.assertIsNone(rec.last_reply_kind)

    def test_reserved_codes_are_not_candidates(self):
        rec = _recognizer()
        for code in RESERVED_PHONEME_CODES:
            word, scores = rec.evaluate_queue(code, allow_free_trail=False, cut=False)
            self.assertNotIn(tuple(code), {tuple(CONCEPT_TO_PHONEMES[w]) for w in scores})
            self.assertNotIn("reserved", scores)


class TestQueueCut(unittest.TestCase):
    def test_cut_keeps_free_sixth(self):
        rec = _recognizer()
        y = list(CONCEPT_TO_PHONEMES["declare"])
        rec.phoneme_queue = y + ["p0"]
        word, _ = rec.evaluate_queue(allow_free_trail=True, cut=True)
        self.assertEqual(word, "declare")
        self.assertEqual(rec.phoneme_queue, ["p0"])

    def test_speaking_judges_only_at_six(self):
        rec = _recognizer()
        y = list(CONCEPT_TO_PHONEMES["declare"])
        rec.in_segment = True
        for label in y:
            result = rec.update_segment(1.0, kept_labels=[label])
            self.assertIsNone(result)
        self.assertEqual(len(rec.phoneme_queue), 5)
        result = rec.update_segment(1.0, kept_labels=["p0"])
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "declare")

    def test_silence_judges_three(self):
        rec = _recognizer()
        y = list(CONCEPT_TO_PHONEMES["declare"])
        rec.in_segment = True
        rec.phoneme_queue = y[:3]
        result = None
        for _ in range(4):
            result = rec.update_segment(0.0, kept_labels=[])
        self.assertIsNotNone(result)

    def test_one_or_two_on_silence_publish_nothing(self):
        rec = _recognizer()
        rec.in_segment = True
        rec.phoneme_queue = ["p0"]
        result = None
        for _ in range(4):
            result = rec.update_segment(0.0, kept_labels=[])
        self.assertIsNone(result)
        rec.in_segment = True
        rec.phoneme_queue = ["p0", "p1"]
        for _ in range(4):
            result = rec.update_segment(0.0, kept_labels=[])
        self.assertIsNone(result)


class TestJointAttentionReplyQueue(unittest.TestCase):
    def test_same_segment_oshii_then_sugoi_clears_queue(self):
        """同じまとまりではおしいのあとすごいへ上げ、キューを消す。"""
        from cueworld.protocols.joint_attention import JointAttentionProtocol
        from cueworld.tests.test_ja_infant_initiative import _MockEnv, _signals_for_target
        from cueworld.language.vocabulary import CONCEPT_TO_PHONEMES as vocab

        env = _MockEnv()
        protocol = JointAttentionProtocol(env, cue_type="looking", rejection_check=False)
        protocol.start({"fixed_target": 101, "intent_type": "informative"})
        protocol.parent_utterance = True
        protocol.state = "looking"
        protocol._cue_started = True
        protocol.attention_tracker.hold_steps = 3
        for _ in range(3):
            protocol._manage_state(_signals_for_target(101))
        self.assertEqual(protocol.state, "post_shared_attention")

        signals = _signals_for_target(101)
        signals["voice_reply"] = "chigauyo"
        signals["voice_segment_seq"] = 1
        protocol._manage_state(signals)
        self.assertEqual(protocol._pending_word_match, "unadopted")
        self.assertEqual(protocol.utterance_queue[0], vocab["correct"][0])

        signals["voice_reply"] = "oshii"
        protocol._manage_state(signals)
        self.assertEqual(protocol._pending_word_match, "near")
        self.assertTrue(protocol._recast)
        self.assertIn(vocab["recast"][0], list(protocol.utterance_queue))
        after_oshii = list(protocol.utterance_queue)

        signals["voice_reply"] = "chigauyo"
        protocol._manage_state(signals)
        self.assertEqual(list(protocol.utterance_queue), after_oshii)
        self.assertEqual(protocol._pending_word_match, "near")

        signals["voice_reply"] = "sugoi"
        signals["words"] = {"red_cube": 0.0}
        protocol._manage_state(signals)
        self.assertEqual(protocol._pending_word_match, "correct")
        self.assertEqual(protocol.utterance_queue[0], vocab["praise"][0])
        protocol.utterance_queue.clear()
        protocol._maybe_complete_reply_round()
        self.assertEqual(protocol.word_match, ["correct"])
        self.assertEqual(protocol.round_count, 1)


if __name__ == "__main__":
    unittest.main()
