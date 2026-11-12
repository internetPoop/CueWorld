"""喃語チャンク検出・salience・セッション記録のユニットテスト（babbling_bout.md）。"""

import unittest
from collections import deque

from cueworld.infant_signal import InfantSignalDetector
from cueworld.language.voice_recognizer import VoiceRecognizer
from cueworld.protocols.dyadic import DyadicProtocol, VOICE_CHANNEL


class _MinimalEnv:
    easy_voice = True
    step_count = 0
    action_repeat = 1
    client_id = 0
    infant_voice_window = deque()
    voice_mapping = {"volume": 0, "pitch": 1, "v_1": 2, "v_2": 3, "v_3": 4}
    phoneme_index = 2
    infant = {
        "index_joint": 0,
        "thumb_joint": 1,
        "head_tilt_joint": 2,
        "body": 0,
    }

    def get_joint_angle(self, agent, joint_key):
        return 0.0

    def _get_hand_shape(self, agent):
        return "closed", 0.0, 0.0


class TestBabbleChunkDetection(unittest.TestCase):
    def setUp(self):
        # どの単語にも当たらない短い辞書
        self.recognizer = VoiceRecognizer(
            word_dictionary={"red_cube": ["p0", "p4", "p5", "p6", "p7"]}
        )

    def _speak_then_silence(self, labels):
        """ラベルを有声で入れ、無声で区間を閉じる。"""
        for lab in labels:
            self.recognizer.update_segment(1.0, kept_labels=[lab])
        finished = None
        for _ in range(self.recognizer.offset_silence_frames):
            finished = self.recognizer.update_segment(0.0, kept_labels=[])
        return finished

    def test_silence_close_unadopted_sets_babble_chunk(self):
        finished = self._speak_then_silence(["p0", "p1", "p2"])
        self.assertIsNotNone(finished)
        word, scores = finished
        self.assertIsNone(word)
        # 不採用でも密な報酬用に全語スコアは残す。公開の words は commit 側で空。
        self.assertIn("red_cube", scores)
        self.assertEqual(self.recognizer.last_babble_chunk, ["p0", "p1", "p2"])

    def test_short_queue_is_not_babble(self):
        finished = self._speak_then_silence(["p0", "p1"])
        self.assertIsNone(finished)
        self.assertIsNone(self.recognizer.last_babble_chunk)

    def test_adopted_word_clears_babble_chunk(self):
        # 完全一致に近い列で採用させる
        finished = self._speak_then_silence(["p0", "p4", "p5", "p6", "p7"])
        self.assertIsNotNone(finished)
        word, _scores = finished
        self.assertEqual(word, "red_cube")
        self.assertIsNone(self.recognizer.last_babble_chunk)

    def test_reset_clears_babble_chunk(self):
        self._speak_then_silence(["p0", "p1", "p2"])
        self.assertIsNotNone(self.recognizer.last_babble_chunk)
        self.recognizer.reset()
        self.assertIsNone(self.recognizer.last_babble_chunk)


class TestBabblingSalienceSignals(unittest.TestCase):
    def setUp(self):
        self.env = _MinimalEnv()
        self.detector = InfantSignalDetector(self.env, history_length=30)

    def test_babbling_salience_one_when_chunk_held(self):
        self.detector.babbling_chunk = ["p0", "p1", "p0"]
        chunk, salience = self.detector.get_babbling_chunk()
        self.assertEqual(chunk, ["p0", "p1", "p0"])
        self.assertEqual(salience, 1.0)
        signals, _words = self.detector.get_voice_signals()
        self.assertEqual(signals["babbling_chunk"], ["p0", "p1", "p0"])
        self.assertEqual(signals["babbling_salience"], 1.0)
        self.assertIsNone(signals["babbling_target"])
        self.assertEqual(signals["babbling_ratio"], 0.0)

    def test_babbling_salience_zero_when_empty(self):
        self.detector.babbling_chunk = None
        chunk, salience = self.detector.get_babbling_chunk()
        self.assertIsNone(chunk)
        self.assertEqual(salience, 0.0)

    def test_clear_babbling_chunk(self):
        self.detector.babbling_chunk = ["p0", "p1", "p2"]
        self.detector.voice_recognizer.last_babble_chunk = ["p0"]
        self.detector.clear_babbling_chunk()
        self.assertIsNone(self.detector.babbling_chunk)
        self.assertIsNone(self.detector.voice_recognizer.last_babble_chunk)

    def test_voice_channel_salience_follows_babbling_salience(self):
        salience = DyadicProtocol.compute_channel_salience(
            {"babbling_salience": 1.0}
        )
        self.assertEqual(salience[VOICE_CHANNEL], 1.0)
        ranked = DyadicProtocol.get_salient_channels({"babbling_salience": 1.0})
        self.assertEqual(ranked[0][0], VOICE_CHANNEL)


if __name__ == "__main__":
    unittest.main()
