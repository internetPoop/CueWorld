import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
from .base import (
    ATTENTION_HOLD_STEPS,
    ATTENTION_WAIT_STEPS,
    AttentionHoldTracker,
    BaseProtocol,
)
from cueworld.intent import pointing_is_complete, weak_arm

# joint_attention_session.md: 言葉の往復の上限
MAX_JA_WORD_ROUNDS = 5
# 切り分けテスト: 真後ろ→正面の角度上限を 1、成功瞬間を 2
DENSE_ANGLE_SCALE = math.pi
DENSE_SUCCESS_BONUS = 2.0


class JointAttentionProtocol(BaseProtocol):
    """共同注意プロトコル（ラベリングと社会的参照・感情共有を統合）"""
    def __init__(self, env: Any, cue_type: Optional[str],change_target: bool = True, rejection_check: bool = True, eval_mode: bool = False, intent_type: str = "informative"):
        super().__init__("JointAttentionProtocol", env, change_target=change_target, rejection_check=rejection_check, eval_mode=eval_mode)
        self.target_object = None

        # プロトコルのメタデータ（属性）
        self.function_type = "declarative" # 宣言的
        self.relation_type = "triadic"     # 三項関係
        self.is_urgent = False             # 非緊急
        self.response_type = True #幼児が自発的に開始できるプロトコルかどうか
        self.request_type = True #親が自発的に開始できるプロトコルかどうか
        self.solo = False
        self.has_target = True #オブジェクトを用意するプロトコルであるかどうか
        self.cue_type = cue_type #共同注意でどのキューを使用するか？
        
        # 状態ごとの受理状態への近さを定義
        self.state_proximities = {
            "init": 0.0,
            "fetching": 0.15,
            "approach": 0.2,
            "looking": 0.4,
            "pointing": 0.4,
            "shaking": 0.4,
            "post_shared_attention": 0.8,
            "completed": 1.0,
            "failure": 0.0,
        }
        
        self.target_word = None
        self.intent_type = intent_type # "informative" or "sharing" or "simple"
        if self.intent_type == "simple":
            self.parent_utterance = False
        else:
            self.parent_utterance = True
        self.last_distance = 10.0
        self.level = 2
        # バッチ E: 幼児始動・指差しの成功1回の実測。値は仮。H では typical_steps は触らない。
        self.typical_steps = 130
        # バッチ H: 成功は途切れない ATTENTION_HOLD_STEPS。待ち・離脱は ATTENTION_WAIT_STEPS。
        self.shared_hold_steps = ATTENTION_HOLD_STEPS
        self.attention_tracker = AttentionHoldTracker()
        # shake cue: 幼児へ見せる接近距離・振る時間（start で env 同期）
        self.shake_stand_dist = None
        self.show_dist = None
        self.shake_steps = 60
        self.show_reach_tol = 0.12
        # T2 較正: seeds 42-44、T=ceil(max(中央値,平均))、timeout=2T
        # 前進系: 掲げ未達でも振りは続ける。T_reach=12 → 24 steps ≒ 2.4s @ action_dt=0.1
        # （seed42 のみ実到達。43/44 は前進 TO）
        self.shake_reach_timeout_steps = 24
        # エラー対策（親モータ → 即 FAILED）
        self.fetch_timeout_steps = 116     # T=58, k=2 ≒ 11.6s @ action_dt=0.1
        self.approach_timeout_steps = 52   # T=26 (escalate 再接近込み), k=2 ≒ 5.2s @ action_dt=0.1
        self.show_pos = None  # shaking 進入時に固定。退出で破棄
        self.shake_timer = 0  # 到達後のみ進行する振り専用タイマ
        self.shake_reached = False
        self.ja_success = False
        self._cue_started = False
        self._word_reacted = False
        self._praised = False
        self._recast = False
        self._corrected = False
        self._last_words: Dict[str, Any] = {}
        self._last_voice_segment_seq = 0
        self.infant_spoke = False
        # 言い終えた返事の種類を出た順に残す（joint_attention_session.md）
        self.word_match: List[str] = []
        self.voice_wait_timer = 0
        self.round_count = 0
        self.max_rounds = MAX_JA_WORD_ROUNDS
        self._reply_pending = False
        self._pending_word_match: Optional[str] = None
        # 切り分けテスト用スイッチ（start の context で上書き）
        self.lock_cue = False
        self.close_on_success = False
        self._dense_active = False
        self._dense_prev_angle = None
        self._dense_done = False
        self._dense_success_pending = False
        
        if eval_mode:
            self.eval_dict = {
                "steps": 0.0,#エピソード内のステップ数．効率的な視線移動や発話ができているかを見るための指標．
                "success": 0.0,
                "delta_sum": 0.0, #結果的に移動した大きさの合計．効率的な視線移動などができているかを見るための指標．
                "ideal_delta_sum": 0.0 #理想的な移動の大きさの合計．幼児が最初から対象を見ている場合は0、そうでない場合はステップ数に応じて増加するなどの形で定義することができる。
            }
            self.previous_score = None
            
    def start(self, context: Dict[str, Any]):
        super().start(context)

        # context で cue / intent を上書き（デモや fixed_protocol 用）
        if context.get("cue_type") is not None:
            self.cue_type = context["cue_type"]
        if context.get("intent_type") is not None:
            self.intent_type = context["intent_type"]
            if self.intent_type == "simple":
                self.parent_utterance = False
            else:
                self.parent_utterance = True
        self.lock_cue = bool(context.get("lock_cue", False))
        self.close_on_success = bool(context.get("close_on_success", False))
        # enable_emotion=False のとき sharing は始められない（プロトコル失敗ではなく呼び出しエラー）
        if self.intent_type == "sharing" and not getattr(self.env, "enable_emotion", False):
            raise ValueError(
                "intent_type='sharing' requires enable_emotion=True"
            )
        
        self.state = "init"
        self.state_timer = 0
        self.last_distance = 10.0
        self.attention_tracker.reset()
        self.ja_success = False
        self._cue_started = False
        self._word_reacted = False
        self._praised = False
        self._recast = False
        self._corrected = False
        self._last_words = {}
        self._last_voice_segment_seq = 0
        self.infant_spoke = False
        self.word_match = []
        self.voice_wait_timer = 0
        self.round_count = 0
        self.max_rounds = MAX_JA_WORD_ROUNDS
        self._reply_pending = False
        self._pending_word_match = None
        self._dense_active = False
        self._dense_prev_angle = None
        self._dense_done = False
        self._dense_success_pending = False
        self._sync_voice_target()
        # shake 距離: env.stand_dist 相当へ下げて掲げ点が腕長内に入るようにする
        env_shake_stand = getattr(self.env, "shake_stand_dist", None)
        self.shake_stand_dist = (
            float(env_shake_stand)
            if env_shake_stand is not None
            else float(self.env.stand_dist)
        )
        env_show = getattr(self.env, "show_dist", None)
        # None のときは shaking 進入時に、目からの距離を対象の大きさから決める。
        # 注入された値は胴体からの place_dist ではなく、幼児の目からの水平距離。
        self.show_dist = float(env_show) if env_show is not None else None
        self.show_pos = None
        self.shake_timer = 0
        self.shake_reached = False
        if self.eval_mode:
            self.eval_dict = {
                "steps": 0.0,#エピソード内のステップ数．効率的な視線移動や発話ができているかを見るための指標．
                "success": 0.0,
                "delta_sum": 0.0, #結果的に移動した大きさの合計．効率的な視線移動などができているかを見るための指標．
                "ideal_delta_sum": 0.0 #理想的な移動の大きさの合計．幼児が最初から対象を見ている場合は0、そうでない場合はステップ数に応じて増加するなどの形で定義することができる。
            }
            self.previous_score = None

    def _queue_cue_utterance(self) -> None:
        """looking / pointing / shaking 進入時の発話キュー。合図と同時。"""
        if self.intent_type == "informative":
            c_id = self.language_manager.get_concept_for_object(self.target_object)
            self.queue_utterance("declare_object", concept_id=c_id)
        elif self.intent_type == "sharing":
            self.queue_utterance("praise") # 共感語

    def _place_show_pos(self) -> None:
        """掲げ点を幼児の目の高さ・目から親の方向へ固定する。"""
        self.show_pos = self.env.compute_show_position(
            self.env.parent,
            self.env.infant,
            show_dist=self.show_dist,
            target_object=self.target_object,
        )

    def _enter_shaking(self) -> None:
        """shaking 進入時: 掲げ点を固定し、到達後タイマをリセット。"""
        self._place_show_pos()
        self.shake_timer = 0
        self.shake_reached = False
        self.change_state("shaking")
        self._ensure_cue_started()

    def _show_held_object(self, action) -> None:
        """掲げ点で持ったまま見せる。手首は振らない。"""
        if self.show_pos is None:
            self._place_show_pos()
        self.env.hand_reach_location(self.env.parent, self.show_pos, action=action)
        self.env.grip_object(self.env.parent, action=action)

    def _exit_shaking(self, next_state: str) -> None:
        """shaking 退出時: show_pos を破棄して次状態へ。"""
        self.show_pos = None
        self.shake_reached = False
        self.change_state(next_state)

    def _ensure_cue_started(self) -> None:
        if self._cue_started:
            return
        self._queue_cue_utterance()
        self.attention_tracker.reset()
        self._cue_started = True
        # 密な報酬: 合図が最初に始まった瞬間から角度を測る（やり直しでも起点は戻さない）
        if not self._dense_active:
            self._dense_active = True
            self._dense_prev_angle = self._compute_gaze_target_angle()

    def _enter_cue(self, state_name: str) -> None:
        self.change_state(state_name)
        self._ensure_cue_started()

    def _enter_post_shared_attention(self) -> None:
        """視線ホールド成功後。手の合図は続け、顔は幼児へ。発話は再キューしない。"""
        self.ja_success = True
        self.session_success = True
        self._dense_done = True
        self._dense_success_pending = True
        if self.close_on_success:
            self.change_state(self.success_state)
            self.is_finished = True
            return
        self.voice_wait_timer = 0
        self.change_state("post_shared_attention")

    def _parent_body_id(self) -> Optional[int]:
        parent = getattr(self.env, "parent", None) or {}
        return parent.get("body")

    def _hold_ready(self) -> bool:
        """成立済みかつホールドが途切れていないときだけ返事する。"""
        return bool(self.ja_success) and (
            self.attention_tracker.hold >= self.attention_tracker.hold_steps
        )

    def _update_cue_attention(self, infant_signals: Dict[str, Any]) -> None:
        infant_target = self.get_infant_target(infant_signals)
        result = self.attention_tracker.step(
            infant_target,
            self._current_target_id(),
            self._parent_body_id(),
        )
        if result == "success":
            self._enter_post_shared_attention()
        elif result == "timeout":
            self._handle_cue_timeout()

    def _update_post_attention(self, infant_signals: Dict[str, Any]) -> None:
        """成立後の視線。タイムアウトは合図を強めず成功のまま閉じる。"""
        infant_target = self.get_infant_target(infant_signals)
        result = self.attention_tracker.step(
            infant_target,
            self._current_target_id(),
            self._parent_body_id(),
        )
        if result == "timeout":
            self.change_state(self.success_state)

    def _handle_cue_timeout(self) -> None:
        """5秒未到達または離脱。1回だけ強化し、2回目で失敗。対象は変えない。"""
        if self.retry_count < 1:
            self.retry_count = 1
            if not self.lock_cue and self.cue_type == "looking":
                self.cue_type = "pointing"
                self.change_state("pointing")
                self._queue_cue_utterance()
                self.attention_tracker.reset()
            else:
                # pointing / shake / lock_cue: 同じ合図のまま発話をもう一度。振るは取りに戻らない。
                self._queue_cue_utterance()
                self.attention_tracker.reset()
                self.state_timer = 0
            return
        self.change_state(self.failure_state)
        self.is_finished = True

    def _compute_gaze_target_angle(self) -> float:
        """目の前方と今の対象の中心との角度差 [rad]。対象が無ければ 0。"""
        target_id = self._current_target_id()
        if target_id is None:
            return 0.0
        eye_pos, forward_vec = self.env._agent_eye_pose(self.env.infant)
        obj_pos = self.env._obj_pos(target_id)
        to_target = np.asarray(obj_pos, dtype=np.float32) - np.asarray(eye_pos, dtype=np.float32)
        return float(self.env._angle_between(forward_vec, to_target))

    def dense_reward_function(self, infant_signals: Dict[str, Any]) -> float:
        """切り分けテスト用。合図開始〜視線成功まで、角度が減った割合 + 成功で +2。"""
        if self._dense_success_pending:
            self._dense_success_pending = False
            return float(DENSE_SUCCESS_BONUS)
        if self._dense_done or not self._dense_active:
            return 0.0
        angle = self._compute_gaze_target_angle()
        if self._dense_prev_angle is None:
            self._dense_prev_angle = angle
            return 0.0
        delta = (float(self._dense_prev_angle) - float(angle)) / DENSE_ANGLE_SCALE
        self._dense_prev_angle = angle
        return float(delta)

    def _sync_voice_target(self) -> None:
        rec = getattr(self.env, "voice_recognizer", None)
        if rec is None:
            return
        if self.is_finished:
            rec.set_target(None)
        else:
            rec.set_target(self.target_concept)

    def _finish_reply_round(self) -> None:
        """返事を最後まで言い終えたとき。消された途中の返事は列に入れない。"""
        if self._pending_word_match is not None:
            self.word_match.append(self._pending_word_match)
            self.infant_spoke = True
        self._pending_word_match = None
        self._reply_pending = False
        self.round_count += 1
        self.voice_wait_timer = 0
        if self.round_count >= self.max_rounds:
            self.change_state(self.success_state)

    def _maybe_complete_reply_round(self) -> None:
        if self._reply_pending and len(self.utterance_queue) == 0:
            self._finish_reply_round()

    def _queue_word_reply(self, kind: str, template_key: str, concept_id, *, clear: bool) -> None:
        """返事を積む。clear なら残っている発話を消してから入れる（すごい）。"""
        if clear:
            self.utterance_queue.clear()
        self.queue_utterance(template_key, concept_id=concept_id)
        self._pending_word_match = kind
        self._reply_pending = True
        self._word_reacted = True
        self.infant_spoke = True

    def _maybe_react_to_words(self, infant_signals: Dict[str, Any]) -> None:
        """認識器が出した返事の種類で、ほめ／おしい／訂正する。成功は変えない。

        ホールドが成立しているあいだだけ積む。seq はホールド不足でも消費し、
        成立前に出た返事を後から言わない。
        """
        self._sync_voice_target()
        words = infant_signals.get("words") or {}
        seq = infant_signals.get("voice_segment_seq")
        new_segment = seq is not None and int(seq) != self._last_voice_segment_seq
        if seq is not None:
            self._last_voice_segment_seq = int(seq)
        if new_segment:
            # まとまりごとに「もう言った」を戻す。
            self._praised = False
            self._recast = False
            self._corrected = False
            self._word_reacted = False
        reply = infant_signals.get("voice_reply")
        new_words = bool(words) and words != self._last_words
        if not words:
            self._last_words = {}
        if new_words:
            self._last_words = dict(words)
        if not new_segment and not new_words and not reply:
            return
        # ホールド不足でも seq は上で消費済み。返事は出さない。
        if not self._hold_ready():
            return

        c_id = None
        if self.target_object is not None:
            c_id = self.language_manager.get_concept_for_object(self.target_object)

        if reply == "sugoi":
            if self._praised:
                return
            self._praised = True
            # 残っている「ちがうよ」や合図を切って、次の音から「すごいね」にする。
            self._queue_word_reply("correct", "praise_with_target", c_id, clear=True)
            return

        if reply == "oshii":
            if self._praised or self._recast:
                return
            self._recast = True
            self._queue_word_reply("near", "recast_with_target", c_id, clear=False)
            return

        if reply == "chigauyo":
            if self._praised or self._recast or self._corrected:
                return
            self._corrected = True
            adopted_word = min(words, key=words.get) if words else None
            kind = "wrong" if adopted_word is not None else "unadopted"
            self._queue_word_reply(kind, "correct_with_target", c_id, clear=False)

    def _handle_post_shared_attention(self, infant_signals: Dict[str, Any]) -> None:
        """次の言葉を待つ。キューが空で区間の外だけ 50 step。区間内の見切りも 50 step。"""
        if len(self.utterance_queue) > 0:
            self.voice_wait_timer = 0
            return
        in_segment = self.voice_segment.in_segment
        self.voice_wait_timer += 1
        if in_segment:
            if (
                not self._word_reacted
                and self.voice_wait_timer >= ATTENTION_WAIT_STEPS
            ):
                self.change_state(self.success_state)
            return
        if self.voice_wait_timer >= ATTENTION_WAIT_STEPS:
            self.change_state(self.success_state)


    def _get_target_id_from_index(self, target_index: int) -> Optional[int]:
        if target_index < 0 or target_index >= len(self.env.trackable_targets):
            return None
        return self.env.trackable_targets[target_index]

    def _get_signal_target(
        self,
        targets: Optional[np.ndarray],
        *,
        ignore_parent: bool = False,
    ) -> Optional[int]:
        if targets is None or len(targets) == 0:
            return None
        target_array = np.asarray(targets, dtype=float)
        if target_array.size == 0 or float(np.max(target_array)) <= 0.0:
            return None

        sorted_indices = np.argsort(target_array)[::-1]
        for target_index in sorted_indices:
            if target_array[target_index] <= 0.0:
                break
            target_id = self._get_target_id_from_index(int(target_index))
            if target_id is None:
                continue
            if ignore_parent and target_id == self.env.parent["body"]:
                continue
            return target_id
        return None

    def _pointing_is_reliable(self, infant_signals: Dict[str, Any]) -> bool:
        if not pointing_is_complete(infant_signals):
            return False
        gesture_targets = infant_signals.get("gesture_targets", [])
        target_id = self._get_signal_target(
            np.asarray(gesture_targets, dtype=float) if gesture_targets is not None else None,
            ignore_parent=True,
        )
        return target_id is not None and target_id not in self.ignored_targets

    def _hand_target_is_usable(self, infant_signals: Dict[str, Any]) -> bool:
        """揃った指差しまたは弱い腕で、手の先の物が使えるか。"""
        if not (pointing_is_complete(infant_signals) or weak_arm(infant_signals)):
            return False
        gesture_targets = infant_signals.get("gesture_targets", [])
        target_id = self._get_signal_target(
            np.asarray(gesture_targets, dtype=float) if gesture_targets is not None else None,
            ignore_parent=True,
        )
        return target_id is not None and target_id not in self.ignored_targets

    def select_target(self, infant_signals: Optional[Dict[str, Any]] = None):
        sig = infant_signals or self.infant_signals

        # §6-D2: 幼児始動かつ（揃った指差し or 弱い腕）なら手の先
        if self.infant_initiative and self._hand_target_is_usable(sig):
            if getattr(self, "target_object", None) is not None:
                old_target_id = (
                    self.target_object.id if hasattr(self.target_object, "id") else self.target_object
                )
                self.ignored_targets.add(old_target_id)

            target_id = self._get_signal_target(sig.get("gesture_targets", []), ignore_parent=True)
            if target_id is not None and target_id not in self.ignored_targets:
                if hasattr(self.env.unwrapped, "all_objects") and target_id in self.env.unwrapped.all_objects:
                    self.target_object = self.env.unwrapped.all_objects[target_id]
                else:
                    self.target_object = target_id
                return

        super().select_target(infant_signals)

    def _manage_state(self, infant_signals: Dict[str, Any]) -> None:
        self._sync_voice_target()
        # 1. 最優先: 拒否シグナルの検知
        if self.detect_rejection(infant_signals):
            self.show_pos = None
            self.change_state(self.failure_state)
            self.utterance_queue.clear() # 拒否時は発話を即中断する
            self.is_finished = True
            return

        self.state_timer += 1
        
        if self.state == "init":
            if self.infant_initiative:
                self.change_state("approach")
            elif self.cue_type == "shake":
                self.change_state("fetching")
            else:
                self.change_state("approach")

        elif self.state == "fetching":
            # shake cue: 対象物を掴む（既把持なら approach へ）
            if self.env.parent["held_obj"] == self.target_object.id:
                if hasattr(self.env, "first_hand_status"):
                    self.env.first_hand_status.pop(self.env.parent["body"], None)
                self.change_state("approach")
            elif self.state_timer > self.fetch_timeout_steps:
                # エラー対策: 親把持が終わらない → 即 FAILED
                self.fail_on_error_timeout()

        elif self.state == "approach":
            if self.cue_type == "shake":
                # 把持したまま幼児に近づき、見せる位置まで来たら shaking へ
                if self.last_distance < self.shake_stand_dist + 0.1:
                    self._enter_shaking()
                elif self.state_timer > self.approach_timeout_steps:
                    # エラー対策: 接近が終わらない → 即 FAILED
                    self.fail_on_error_timeout()
            else:
                # 親が幼児（または対象物付近）に十分近づいたら
                if self.last_distance < 4.2:
                    if self.cue_type == "pointing":
                        self._enter_cue("pointing")
                    else:
                        self._enter_cue("looking")
                elif self.state_timer > self.approach_timeout_steps:
                    # エラー対策: 非shake 接近も進行阻害なら即 FAILED
                    self.fail_on_error_timeout()
        
        elif self.state == "looking" or self.state == "pointing":
            self._ensure_cue_started()
            self._maybe_react_to_words(infant_signals)
            self._update_cue_attention(infant_signals)

        elif self.state == "shaking":
            # 落下したら拾い直し（トラッカーはリセットしない）
            if self.env.parent["held_obj"] != self.target_object.id:
                self._exit_shaking("fetching")
            else:
                if self.show_pos is not None:
                    err = self.env.palm_reach_error(self.env.parent, self.show_pos)
                    self.shake_reached = err < float(self.show_reach_tol)
                if self.shake_reached:
                    self.shake_timer += 1
                self._ensure_cue_started()
                self._maybe_react_to_words(infant_signals)
                self._update_cue_attention(infant_signals)

        elif self.state == "post_shared_attention":
            self._maybe_complete_reply_round()
            if self.state != "post_shared_attention":
                pass
            else:
                self._update_post_attention(infant_signals)
                if self.state == "post_shared_attention":
                    self._maybe_react_to_words(infant_signals)
                    self._handle_post_shared_attention(infant_signals)
                    
        if self.eval_mode:
            return self.get_eval(infant_signals)
        else:
            return 

    def _determine_action(self) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        # 把持・振り中は腕リセットをしない（grasp/shake と競合するため）
        if self.state not in ("fetching", "shaking") and not (
            self.cue_type == "shake" and self.state in ("approach", "post_shared_attention", self.success_state, self.failure_state)
        ):
            self.env.reset_arm_pose(self.env.parent, action=action)
        
        # --- 物理的アクション ---
        if self.state == "fetching":
            self.env.grasp_object(
                self.env.parent,
                self.target_object.id,
                action=action,
                approach_dist=float(self.env.grasp_approach_dist),
            )

        elif self.state == "approach":
            if self.cue_type == "shake":
                yaw, dist = self.env.approach_infant(
                    self.env.parent, self.env.infant,
                    action=action, approach_dist=self.shake_stand_dist,
                )
                if dist is not None:
                    self.last_distance = dist
                if self.env.parent["held_obj"] == self.target_object.id:
                    self.env.grip_object(self.env.parent, action=action)
            else:
                yaw, dist = self.env.approach_infant_for_ja(
                    self.env.parent, self.env.infant, self.target_object.id,
                    action=action, approach_dist=4.0
                )
                if dist is not None:
                    self.last_distance = dist
                
        elif self.state == "shaking":
            # 固定掲げ点で持ったまま見せる。対象物を注視（置く・渡す・手首を振ることはしない）
            if self.env.parent["held_obj"] == self.target_object.id:
                self._show_held_object(action)
                self.env.gaze_object(
                    self.env.parent, self.target_object.id, yaw_link="head", action=action
                )
            else:
                # 落下した場合は拾い直しへ
                pass

        elif self.state == "pointing" or self.state == "looking": #TODO: 指差し以外のキューにも対応．
            # 対象物の方を向く
            self.env.point_object(self.env.parent, self.target_object.id, action=action)
            yaw, pitch = self.env.gaze_object(self.env.parent, self.target_object.id, yaw_link="head", action=action)
            
            
            if self.intent_type == "sharing":
                # 感情共有の場合は笑顔を見せる処理（仮として顎を開くなどを想定）
                # 現在は特にIKによる動作はせず対象を見るだけにとどめる
                pass
            
        elif self.state in ["post_shared_attention", self.success_state, self.failure_state]:
            if self.cue_type == "shake" and self.env.parent["held_obj"] == self.target_object.id:
                # 締めのあいだは掲げ続ける。成功・失敗のあと手首は振らない。
                if self.state == "post_shared_attention":
                    self._show_held_object(action)
                else:
                    self.env.grip_object(self.env.parent, action=action)
                yaw, pitch = self.env.gaze_agent(
                    self.env.parent, self.env.infant, yaw_link="head", action=action
                )
            else:
                # 手の合図は続け、顔は幼児へ（成功後の確認）
                self.env.point_object(self.env.parent, self.target_object.id, action=action)
                yaw, pitch = self.env.gaze_agent(
                    self.env.parent, self.env.infant, yaw_link="head", action=action
                )
            

        # --- 発話アクション ---
        self._emit_utterance(action)
        
        return action
    
    def get_infant_target(self, infant_signals: Dict[str, Any]):
        """
        幼児の視線の先を返す．
        §6-D2: 成功判定はいつも視線（決めた対象を見ているか）。
        その step のベクトル（gaze_targets_now）。キーが無い／全要素 0 なら None。
        """
        return self._get_signal_target(
            infant_signals.get("gaze_targets_now"),
            ignore_parent=False,
        )

    def infant_is_engaged(self, infant_signals: Dict[str, Any]) -> bool:
        """同じ物を見ている（視線。ATTENTION_HOLD_STEPS 成功条件とは独立）。

        30 step 平均の gaze_targets の最大が今の対象と一致するときだけ真。
        成功ホールドの瞬間視線とは別。
        """
        current_id = self._current_target_id()
        if current_id is None:
            return False
        gaze_id = self._get_signal_target(
            infant_signals.get("gaze_targets", []),
            ignore_parent=False,
        )
        return gaze_id == current_id

    def keeps_held_object(self, held_id) -> bool:
        if self.cue_type != "shake":
            return False
        current_id = self._current_target_id()
        return current_id is not None and held_id == current_id

    def blocks_interrupt(self, infant_signals: Dict[str, Any]) -> bool:
        """成立後の親発話中・幼児の声が出ているあいだだけ割り込みを止める。"""
        if self.state != "post_shared_attention":
            return False
        if len(self.utterance_queue) > 0:
            return True
        return bool(self.voice_segment.in_segment)

    def _calculate_persistence(self, infant_signals: Dict[str, Any]):
        """言い終えた往復ごとに 2.7 / 5.0 から 1.0 引く。下限は 1.0。"""
        if self.typical_steps is None:
            return
        super()._calculate_persistence(infant_signals)
        decrement = 1.0 * float(self.round_count)
        self.persistence = max(1.0, float(self.persistence) - decrement)

    def history_extra(self) -> Dict[str, Any]:
        return {
            "intent_type": self.intent_type,
            "infant_spoke": bool(self.infant_spoke),
            "word_match": list(self.word_match),
            "emotion_reacted": None,
        }

    def get_eval(self, infant_signals: Dict[str, Any]) -> Dict[str, float]:
        # 共同注意の評価指標を計算する
        if self.state == self.success_state:
            self.eval_dict["success"] = 1.0
        else:
            self.eval_dict["success"] = 0.0

        # ステップ数をカウント
        self.eval_dict["steps"] += 1.0
        
        # 幼児の視線移動の大きさを加算（効率的な視線移動ができているかの指標）
        gaze_targets = (infant_signals or self.infant_signals).get("gaze_targets", [])
        #ターゲットオブジェクトの順番を環境から取得
        keys = list(self.env.all_objects.keys())
        target_index = keys.index(self.target_object.id) if self.target_object is not None else -1
        if target_index >= 0 and target_index < len(gaze_targets):
            target_gaze_prob = gaze_targets[target_index]
            if self.previous_score is None:
                self.previous_score = target_gaze_prob
                self.eval_dict["ideal_delta_sum"] = target_gaze_prob
            else:
                self.eval_dict["delta_sum"] += abs(target_gaze_prob - self.previous_score)
                self.previous_score = target_gaze_prob
        return self.eval_dict
    
    def reward_function(self, infant_signals: Dict[str, Any]) -> float:
        # 共同注意の報酬関数を定義する
        if self.state in {"pointing", "looking", "shaking", "post_shared_attention"}:
            looking_at_parent = float(infant_signals.get("looking_at_parent_eye", 0.0))
            prev_looking = getattr(self, "previous_looking_at_parent", 0.0)
            delta_looking = looking_at_parent - prev_looking
            self.previous_looking_at_parent = looking_at_parent
            return delta_looking
        elif self.state == self.success_state:
            return 1.0  # 成功時報酬
        else:
            return 0.0  # その他の状態では報酬なし
