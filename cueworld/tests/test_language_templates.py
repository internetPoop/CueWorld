"""発話テンプレの概念ID列展開と音素長統一の単体テスト。"""
import unittest
import warnings

from cueworld.language.manager import (
    LanguageManager,
    PHONEME_DURATION_FRAMES,
    INTER_WORD_GAP_FRAMES,
)
from cueworld.language.vocabulary import CONCEPT_TO_PHONEMES


class TestLanguageTemplates(unittest.TestCase):
    def setUp(self):
        self.lm = LanguageManager()

    def test_praise_with_target_expands_to_two_words_with_gap(self):
        pairs = self.lm.get_template_phonemes(
            "common", "praise_with_target", concept_id="red_cube"
        )
        praise = CONCEPT_TO_PHONEMES["praise"]
        target = CONCEPT_TO_PHONEMES["red_cube"]

        expected_frames = (
            len(praise) * PHONEME_DURATION_FRAMES
            + INTER_WORD_GAP_FRAMES
            + len(target) * PHONEME_DURATION_FRAMES
        )
        total_frames = sum(d for _, d in pairs)
        self.assertEqual(total_frames, expected_frames)
        self.assertEqual(expected_frames, 61)

        # praise phonemes @6, then none@1, then red_cube @6
        idx = 0
        for p in praise:
            self.assertEqual(pairs[idx], (p, PHONEME_DURATION_FRAMES))
            idx += 1
        self.assertEqual(pairs[idx], ("none", INTER_WORD_GAP_FRAMES))
        idx += 1
        for p in target:
            self.assertEqual(pairs[idx], (p, PHONEME_DURATION_FRAMES))
            idx += 1
        self.assertEqual(idx, len(pairs))

    def test_all_phoneme_durations_are_unified(self):
        pairs = self.lm.get_template_phonemes(
            "object_giving", "present", concept_id="blue_sphere"
        )
        for phoneme, duration in pairs:
            if phoneme == "none":
                self.assertEqual(duration, INTER_WORD_GAP_FRAMES)
            else:
                self.assertEqual(duration, PHONEME_DURATION_FRAMES)

    def test_target_only_without_concept_skips_without_trailing_gap(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pairs = self.lm.get_template_phonemes("common", "target_only", concept_id=None)
        self.assertEqual(pairs, [])
        self.assertTrue(any("TARGET" in str(w.message) for w in caught))

    def test_single_word_prompt_has_no_gap(self):
        pairs = self.lm.get_template_phonemes("dyadic", "prompt")
        prompt = CONCEPT_TO_PHONEMES["prompt"]
        self.assertEqual(len(pairs), len(prompt))
        self.assertFalse(any(p == "none" for p, _ in pairs))
        for (phoneme, duration), expected in zip(pairs, prompt):
            self.assertEqual(phoneme, expected)
            self.assertEqual(duration, PHONEME_DURATION_FRAMES)

    def test_declare_object_uses_declare_and_target(self):
        pairs = self.lm.get_template_phonemes(
            "joint_attention", "declare_object", concept_id="green_cube"
        )
        declare = CONCEPT_TO_PHONEMES["declare"]
        target = CONCEPT_TO_PHONEMES["green_cube"]
        phonemes_only = [p for p, _ in pairs if p != "none"]
        self.assertEqual(phonemes_only, declare + target)

    def test_unknown_template_key_returns_empty(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pairs = self.lm.get_template_phonemes("common", "does_not_exist")
        self.assertEqual(pairs, [])
        self.assertTrue(any("Unknown utterance template" in str(w.message) for w in caught))

    def test_vocab_min_hamming_after_append(self):
        codes = list(CONCEPT_TO_PHONEMES.values())
        best = 99
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                h = sum(1 for a, b in zip(codes[i], codes[j]) if a != b)
                best = min(best, h)
        self.assertGreaterEqual(best, 3)
        self.assertIn("declare", CONCEPT_TO_PHONEMES)
        self.assertIn("prompt", CONCEPT_TO_PHONEMES)


if __name__ == "__main__":
    unittest.main()
