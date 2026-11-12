"""バッチ H: 共同注意の履歴追加欄。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cueworld.protocol_manager import ProtocolManager
from cueworld.tests.test_ja_infant_initiative import (
    JointAttentionProtocol,
    _MockEnv,
    _signals_for_target,
)
from cueworld.tests.test_protocol_manager_history import (
    _FakeProtocol,
    _MockEnv as _PmMockEnv,
)


class TestJaHistoryExtra(unittest.TestCase):
    def test_history_extra_fields(self):
        env = _MockEnv()
        protocol = JointAttentionProtocol(
            env, cue_type="pointing", rejection_check=False
        )
        protocol.start({"fixed_target": 101, "intent_type": "informative"})
        extra = protocol.history_extra()
        self.assertEqual(extra["intent_type"], "informative")
        self.assertFalse(extra["infant_spoke"])
        self.assertEqual(extra["word_match"], [])
        self.assertIsNone(extra["emotion_reacted"])

        protocol.infant_spoke = True
        protocol.word_match = ["near"]
        extra = protocol.history_extra()
        self.assertTrue(extra["infant_spoke"])
        self.assertEqual(extra["word_match"], ["near"])

    def test_correct_word_sets_match(self):
        env = _MockEnv()
        protocol = JointAttentionProtocol(
            env, cue_type="looking", rejection_check=False
        )
        # sharing は enable_emotion=True が要る。単語返事の確認は informative で足りる。
        protocol.start({"fixed_target": 101, "intent_type": "informative"})
        protocol.parent_utterance = True
        protocol.state = "looking"
        protocol._cue_started = True
        protocol.attention_tracker.hold_steps = 3
        for _ in range(3):
            protocol._manage_state(_signals_for_target(101))
        self.assertEqual(protocol.state, "post_shared_attention")
        signals = _signals_for_target(101)
        signals["words"] = {"red_cube": 0.1}
        signals["voice_reply"] = "sugoi"
        signals["voice_segment_seq"] = 1
        protocol._manage_state(signals)
        self.assertEqual(protocol._pending_word_match, "correct")
        protocol.utterance_queue.clear()
        protocol._maybe_complete_reply_round()
        self.assertEqual(protocol.word_match, ["correct"])
        self.assertEqual(protocol.history_extra()["intent_type"], "informative")

    def test_sharing_requires_enable_emotion(self):
        env = _MockEnv()
        env.enable_emotion = False
        protocol = JointAttentionProtocol(
            env, cue_type="looking", rejection_check=False
        )
        with self.assertRaises(ValueError) as ctx:
            protocol.start({"fixed_target": 101, "intent_type": "sharing"})
        self.assertIn("enable_emotion", str(ctx.exception))

    def test_sharing_allowed_when_emotion_enabled(self):
        env = _MockEnv()
        env.enable_emotion = True
        protocol = JointAttentionProtocol(
            env, cue_type="looking", rejection_check=False
        )
        protocol.start({"fixed_target": 101, "intent_type": "sharing"})
        self.assertEqual(protocol.intent_type, "sharing")
        self.assertEqual(protocol.history_extra()["intent_type"], "sharing")

    def test_manager_merges_extra_into_row(self):
        env = _PmMockEnv()
        manager = ProtocolManager(
            env,
            fixed_protocol="observance",
            fixed_context={},
            check_interval=1,
            household_interval_steps=5,
        )
        ja = _FakeProtocol(
            "JointAttentionProtocol",
            request_type=True,
            response_type=True,
            cue_type="pointing",
        )
        ja.history_extra = lambda: {
            "intent_type": "informative",
            "infant_spoke": True,
            "word_match": ["correct"],
            "emotion_reacted": None,
        }
        manager.protocols["joint_attention"] = ja
        manager.current_protocol = ja
        manager.current_protocol_name = "joint_attention"
        manager.current_protocol_start_step = 4
        manager.fixed_protocol = None
        ja.call_success_flag = True
        ja.is_finished = True
        ja.target_object = SimpleNamespace(id=7)
        env.step_count = 9

        with patch.object(manager, "check_for_response", return_value=(False, "", {})):
            manager.update({key: 0.0 for key in manager.signal_keys})

        row = manager.protocol_history[0]
        self.assertEqual(row["intent_type"], "informative")
        self.assertTrue(row["infant_spoke"])
        self.assertEqual(row["word_match"], ["correct"])
        self.assertIsNone(row["emotion_reacted"])
        self.assertEqual(row["kind"], "pointing")

    def test_blocks_interrupt_skips_check_for_response(self):
        env = _PmMockEnv()
        manager = ProtocolManager(
            env,
            fixed_protocol="observance",
            fixed_context={},
            check_interval=1,
            household_interval_steps=5,
        )
        ja = _FakeProtocol(
            "JointAttentionProtocol",
            request_type=True,
            response_type=True,
            cue_type="looking",
        )
        ja.blocks_interrupt = lambda signals: True
        manager.protocols["joint_attention"] = ja
        manager.current_protocol = ja
        manager.current_protocol_name = "joint_attention"
        manager.fixed_protocol = None
        ja.call_success_flag = None
        ja.is_finished = False

        with patch.object(
            manager, "check_for_response", return_value=(True, "dyadic", {})
        ) as mock_check:
            manager.update({key: 0.0 for key in manager.signal_keys})
            mock_check.assert_not_called()
        self.assertEqual(manager.current_protocol_name, "joint_attention")


if __name__ == "__main__":
    unittest.main()
