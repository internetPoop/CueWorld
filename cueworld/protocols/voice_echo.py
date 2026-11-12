"""音声切り分けテスト用プロトコル（voice_isolation.md）。

親が見本の 1 語を繰り返し、幼児の認識コストを密な報酬にする。
共同注意の返事は出さない。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from cueworld.language.vocabulary import CONCEPT_TO_PHONEMES
from cueworld.language.voice_recognizer import GETA_BONUS
from .base import BaseProtocol

# 物の名前 20 語（辞書に並んだ順。機能語は含めない）。
OBJECT_CONCEPTS: List[str] = [
    cid
    for cid in CONCEPT_TO_PHONEMES.keys()
    if cid
    not in ("praise", "recast", "correct", "douzo", "declare", "prompt")
]

MAX_CUE_ROUNDS = 5
# 無声の下限。認識の無声 4 ティック × voice_interval 3 = 12 step。
MIN_SILENCE_STEPS = 12
# 区間が閉じるまでの追加待ち上限。
MAX_EXTRA_WAIT_STEPS = 30


class VoiceEchoProtocol(BaseProtocol):
    """親が見本語を繰り返し、認識コストを密な報酬にする切り分けプロトコル。"""

    def __init__(self, env: Any, rejection_check: bool = False):
        super().__init__(
            "VoiceEchoProtocol",
            env,
            change_target=False,
            rejection_check=rejection_check,
        )
        self.solo = True
        self.has_target = False
        self.response_type = False
        self.request_type = False
        self.uses_pre_release = False
        self.level = 0
        self.typical_steps = MAX_CUE_ROUNDS * (30 + MIN_SILENCE_STEPS)
        self.state_proximities = {
            "init": 0.0,
            "cueing": 0.4,
            "waiting": 0.7,
            "completed": 1.0,
            "failure": 0.0,
        }

        self.use_geta: bool = False
        self.max_rounds: int = MAX_CUE_ROUNDS
        self.round_count: int = 0
        self.wait_timer: int = 0
        # エピソードをまたいで残す。reset / start では戻さない。
        self._word_index: int = 0

    def start(self, context: Dict[str, Any]):
        super().start(context)
        self.use_geta = bool(context.get("use_geta", False))
        self.max_rounds = int(context.get("max_rounds", MAX_CUE_ROUNDS))
        self.round_count = 0
        self.wait_timer = 0
        self.session_success = False

        override = context.get("target_concept")
        if override:
            self.target_concept = str(override)
        else:
            if not OBJECT_CONCEPTS:
                raise RuntimeError("OBJECT_CONCEPTS is empty")
            self.target_concept = OBJECT_CONCEPTS[self._word_index % len(OBJECT_CONCEPTS)]
            self._word_index += 1

        self.target_word = self.language_manager.get_human_word(self.target_concept)
        self._sync_voice_target()
        self._begin_cue()

    def stop(self):
        self._clear_voice_target()
        super().stop()

    def _sync_voice_target(self) -> None:
        rec = getattr(self.env, "voice_recognizer", None)
        if rec is None:
            return
        if self.use_geta and self.target_concept:
            rec.set_target(self.target_concept)
        else:
            rec.set_target(None)

    def _clear_voice_target(self) -> None:
        rec = getattr(self.env, "voice_recognizer", None)
        if rec is not None:
            rec.set_target(None)

    def _begin_cue(self) -> None:
        self.utterance_queue.clear()
        self.queue_utterance("target_only", self.target_concept)
        self.change_state("cueing")
        self.state_timer = 0
        self.wait_timer = 0

    def _infant_in_segment(self, infant_signals: Dict[str, Any]) -> bool:
        rec = getattr(self.env, "voice_recognizer", None)
        if rec is not None and bool(getattr(rec, "in_segment", False)):
            return True
        # プロトコル側の volume トラッカー（ティック間の補完）。
        if self.voice_segment.in_segment:
            return True
        return False

    def _waiting_should_advance(self, infant_signals: Dict[str, Any]) -> bool:
        if self.wait_timer < MIN_SILENCE_STEPS:
            return False
        if not self._infant_in_segment(infant_signals):
            return True
        return self.wait_timer >= MIN_SILENCE_STEPS + MAX_EXTRA_WAIT_STEPS

    def _finish_session(self) -> None:
        self._clear_voice_target()
        self.session_success = True
        self.change_state(self.success_state)
        self.is_finished = True

    def _manage_state(self, infant_signals: Dict[str, Any]) -> None:
        self.state_timer += 1
        if self.state == "cueing":
            if len(self.utterance_queue) == 0 and self.state_timer > 1:
                self.change_state("waiting")
                self.wait_timer = 0
            return

        if self.state == "waiting":
            self.wait_timer += 1
            if not self._waiting_should_advance(infant_signals):
                return
            self.round_count += 1
            if self.round_count >= self.max_rounds:
                self._finish_session()
                return
            self._begin_cue()
            return

    def _determine_action(self) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        # 体は動かさない。見本中だけ発話キューを流す。
        if hasattr(self.env, "gaze_agent"):
            self.env.gaze_agent(
                self.env.parent, self.env.infant, yaw_link="head", action=action
            )
        if hasattr(self.env, "reset_arm_pose"):
            self.env.reset_arm_pose(self.env.parent, action=action)
        if self.state == "cueing":
            action = self._emit_utterance(action, volume=0.5)
        else:
            vol_idx = self.env.action_map.get("volume")
            if vol_idx is not None:
                action[vol_idx] = 0.0
        return action

    def _calculate_persistence(self, infant_signals: Dict[str, Any]):
        # 切り分け固定なので維持は使わない。
        self.persistence = 0.0

    def dense_reward_function(self, infant_signals: Dict[str, Any]) -> float:
        """判定 step だけ `1 - (cost - c*)`。完全一致は両条件とも +1。"""
        if self.state in (self.success_state, self.failure_state):
            return 0.0
        target = self.target_concept
        if not target:
            return 0.0

        if self.use_geta:
            scores = getattr(self.env, "last_all_word_scores", None) or {}
            c_star = -float(GETA_BONUS)
        else:
            scores = getattr(self.env, "last_raw_word_scores", None) or {}
            if not scores:
                # 下駄なしでは boosted == raw。フォールバック。
                scores = getattr(self.env, "last_all_word_scores", None) or {}
            c_star = 0.0

        if target not in scores:
            return 0.0
        cost = float(scores[target])
        return float(1.0 - (cost - c_star))
