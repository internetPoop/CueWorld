import numpy as np
import pybullet as p
from typing import Dict, Any
from collections import deque
from .base import BaseProtocol

# 切り分けテスト: 掴んだ瞬間のボーナス。指標 e は 1 m を 1 とする。
DENSE_GRASP_BONUS = 2.0


class ObjectGivingProtocol(BaseProtocol):
    """対象物を幼児の到達圏内へ配置し、把持（または拒否）を待つプロトコル。

    直接の hand_over は使わず、親子を結ぶ線上・幼児側への place で実現する（§3）。
    """
    def __init__(self, env: Any, rejection_check: bool = True):
        super().__init__("ObjectGivingProtocol", env, rejection_check=rejection_check)
        
        # プロトコルのメタデータ（属性）
        self.function_type = "imperative" # 命令的
        self.relation_type = "triadic"    # 三項関係
        self.is_urgent = False
        self.response_type = True #幼児が自発的に開始できるプロトコルかどうか
        self.request_type = True #親が自発的に開始できるプロトコルかどうか
        self.solo = False
        self.has_target = True
        # バッチ E: 成功1回の典型 T。値は仮。
        self.typical_steps = 175
        self.state_proximities = {
            "init": 0.0,
            "fetching": 0.2,
            "approach": 0.4,
            "showing": 0.5,
            "placing": 0.6,
            "waiting_infant": 0.8,
            "reaction": 0.9,
            "completed": 1.0
        }

        # 配置パラメータ（env を正とする。start 時に同期）
        self.place_dist = None
        self.stand_dist = None
        self.hold_height = 0.15
        self.release_settle_steps = 8  # 手放し後、落下・静止を待つ step 数

        # showing（ESCALATE 経路）用。None なら目からの既定距離。env.show_dist は目からの水平距離で上書き可
        self.show_dist = None
        self.show_reach_tol = 0.12
        self.show_min_steps = 30  # 1単語 ≈ 30step（PHONEME_DURATION_FRAMES=6 × 5）
        # T2 較正: seeds 42-44、T=ceil(max(中央値,平均))、timeout=2T
        # show_timeout: 前進系。fixed に showing 無し → shake_reach T=12 流用。24 steps ≒ 2.4s @ action_dt=0.1
        self.show_timeout_steps = 24
        # エラー対策（親モータ張り付き → 即 FAILED）
        self.fetch_timeout_steps = 116   # T=58 (med58/mean53.3), k=2 ≒ 11.6s @ action_dt=0.1
        # approach: max(med17,mean20)=20→40 だと escalate 再接近(~43)で切れるため T=max(..., max_sample26)
        self.approach_timeout_steps = 52  # T=26, k=2 ≒ 5.2s @ action_dt=0.1
        self.placing_timeout_steps = 84   # T=42 (med42/mean33.3), k=2 ≒ 8.4s @ action_dt=0.1
        # 失敗系（幼児把持待ち）
        self.waiting_infant_timeout_steps = 34  # T=17 (med15/mean16.7), k=2 ≒ 3.4s @ action_dt=0.1
        
        self.state_timer = 0
        self.target_object = None
        self.last_distance = 10.0
        self.reaction_flag = False
        self.level = 1
        self.place_pos = None
        self.placed = False
        self.release_timer = 0
        self.infant_pos_at_place = None
        self.escalated = False  # waiting_infant ESCALATE 後の専用経路フラグ（1回きり）
        self.show_reached = False
        self.show_pos = None  # showing 進入時に固定。退出で破棄（ライブ追従しない）
        # 切り分けテスト用スイッチ（start の context で上書き）
        self.close_on_grasp = False
        self._dense_active = False
        self._dense_prev_e = None
        self._dense_done = False
        self._dense_success_pending = False

    def start(self, context: Dict[str, Any]):
        super().start(context)

        # env を正とする距離パラメータを同期
        self.place_dist = float(self.env.place_dist)
        self.stand_dist = float(self.env.stand_dist)
        # show_dist: None なら compute_show_position の既定（目から 0.1 + 2×半幅）。
        # env.show_dist があれば、その値を目からの水平距離として使う。
        env_show = getattr(self.env, "show_dist", None)
        self.show_dist = float(env_show) if env_show is not None else None
        self.close_on_grasp = bool(context.get("close_on_grasp", False))
        
        self.change_state("init")
        self.last_distance = 10.0
        self.reaction_flag = False
        self.place_pos = None
        self.placed = False
        self.release_timer = 0
        self.infant_pos_at_place = None
        self.escalated = False
        self.show_reached = False
        self.show_pos = None
        self._dense_active = False
        self._dense_prev_e = None
        self._dense_done = False
        self._dense_success_pending = False

    def infant_is_engaged(self, infant_signals: Dict[str, Any]) -> bool:
        """提示対象への見る / 触る / 持つ（対象一致。parent_object_cue のどれか物ではない）。"""
        return self._infant_attending_current_object(infant_signals)

    def keeps_held_object(self, held_id) -> bool:
        tid = self._current_target_id()
        return tid is not None and held_id == tid

    def change_state(self, new_state: str):
        super().change_state(new_state)
        if new_state == "approach":
            # 進入直後の stale last_distance による即 placing 脱出を防ぐ
            self.last_distance = 10.0
        elif new_state == "waiting_infant":
            # 密な報酬: 最初の待ちに入った瞬間から指標を測る（やり直しでも起点は戻さない）
            if not self._dense_active:
                self._dense_active = True
                self._dense_prev_e = self._compute_palm_target_metric()

    def _prepare_placing_from_show(self) -> None:
        """showing → placing 遷移時の共通セットアップ。present は再キューしない。"""
        self.show_pos = None  # showing 退出
        self.place_pos = self.env.reach_place_position(
            self.env.infant,
            self.env.parent,
            place_dist=self.place_dist,
            hold_height=self.hold_height,
        )
        infant_pos, _ = self._get_agent_base_xy(self.env.infant)
        self.infant_pos_at_place = infant_pos
        self.placed = False
        self.release_timer = 0
        self.escalated = False  # 1回きり: 以降は happy path と同じ approach → placing
        self.change_state("placing")

    def _on_infant_grasp_success(self) -> None:
        """幼児が対象を掴んだときの共通処理。"""
        self.session_success = True
        self._dense_done = True
        self._dense_success_pending = True
        if self.close_on_grasp:
            self.change_state(self.success_state)
            self.is_finished = True
            return
        self.change_state("reaction")

    def _compute_palm_target_metric(self) -> float:
        """掌中心と対象の距離に、掌の向きを織り込んだ指標 e。1 m を 1 とする。

        e = d * (1 + (1 - cos θ))。θ は掌の前方と掌→対象方向の角度。
        向いているとき（θ=0）は e = d。真後ろなら e = 3d。
        """
        target_id = self._current_target_id()
        if target_id is None:
            return 0.0
        palm_center, palm_forward = self.env._get_palm_pose(self.env.infant)
        obj_pos = np.asarray(self.env._obj_pos(target_id), dtype=np.float32)
        palm_center = np.asarray(palm_center, dtype=np.float32)
        diff = obj_pos - palm_center
        d = float(np.linalg.norm(diff))
        theta = float(self.env._angle_between(palm_forward, diff))
        cos_theta = float(np.cos(theta))
        return d * (1.0 + (1.0 - cos_theta))

    def dense_reward_function(self, infant_signals: Dict[str, Any]) -> float:
        """切り分けテスト用。待ち開始〜掴み成功まで、指標 e の減少分 + 成功で +2。"""
        if self._dense_success_pending:
            self._dense_success_pending = False
            return float(DENSE_GRASP_BONUS)
        if self._dense_done or not self._dense_active:
            return 0.0
        e = self._compute_palm_target_metric()
        if self._dense_prev_e is None:
            self._dense_prev_e = e
            return 0.0
        delta = float(self._dense_prev_e) - float(e)
        self._dense_prev_e = e
        return float(delta)

    def _manage_state(self, infant_signals: Dict[str, Any]) -> None:
        # 1. 最優先: 拒否シグナルの検知
        if self.detect_rejection(infant_signals):
            self.show_pos = None
            self.change_state(self.failure_state)
            self.utterance_queue.clear()
            self.is_finished = True
            return
            
        self.state_timer += 1
        
        if self.state == "init":
            self.change_state("fetching")

        elif self.state == "fetching":
            # 対象物を取りに行く（既把持なら即 approach へ）
            if self.env.parent["held_obj"] == self.target_object.id:
                if hasattr(self.env, "first_hand_status"):
                    self.env.first_hand_status.pop(self.env.parent["body"], None)
                self.change_state("approach")
            elif self.state_timer > self.fetch_timeout_steps:
                # エラー対策: 親把持が終わらない → 即 FAILED
                self.fail_on_error_timeout()

        elif self.state == "approach":
            # 親が幼児に近づく
            if self.last_distance < self.stand_dist + 0.1:
                if self.escalated:
                    # ESCALATE 経路: showing へ（present は showing 進入時にキュー）
                    self.show_pos = self.env.compute_show_position(
                        self.env.parent,
                        self.env.infant,
                        show_dist=self.show_dist,
                        target_object=self.target_object,
                    )
                    self.show_reached = False
                    self.change_state("showing")
                    c_id = self.language_manager.get_concept_for_object(self.target_object)
                    self.queue_utterance("present", concept_id=c_id)
                else:
                    self.place_pos = self.env.reach_place_position(
                        self.env.infant,
                        self.env.parent,
                        place_dist=self.place_dist,
                        hold_height=self.hold_height,
                    )
                    infant_pos, _ = self._get_agent_base_xy(self.env.infant)
                    self.infant_pos_at_place = infant_pos
                    self.placed = False
                    self.release_timer = 0
                    self.change_state("placing")
                    c_id = self.language_manager.get_concept_for_object(self.target_object)
                    self.queue_utterance("present", concept_id=c_id)
            elif self.state_timer > self.approach_timeout_steps:
                # エラー対策: 接近が終わらない → 即 FAILED
                self.fail_on_error_timeout()

        elif self.state == "showing":
            # ESCALATE 専用: 把持物を幼児の目線へ掲げる
            if self.env.infant["held_obj"] == self.target_object.id:
                # 掲げている最中に取られたら成功
                self.show_pos = None
                self._on_infant_grasp_success()
            elif self.env.parent["held_obj"] != self.target_object.id:
                # 落とした場合は拾い直す（escalated は維持 → 再 approach 後も showing）
                self.show_pos = None
                self.change_state("fetching")
            elif self.show_reached and self.state_timer >= self.show_min_steps:
                self._prepare_placing_from_show()
            elif self.state_timer > self.show_timeout_steps:
                # 届かなくても placing へ（escalate 全体を落とさない）
                self._prepare_placing_from_show()

        elif self.state == "placing":
            # 未解放のまま幼児が大きく動いたら狙い点を更新
            if (
                not self.placed
                and self.env.parent["held_obj"] == self.target_object.id
                and self.infant_pos_at_place is not None
            ):
                infant_xy, _ = self._get_agent_base_xy(self.env.infant)
                if float(np.linalg.norm(infant_xy - self.infant_pos_at_place)) > 0.25:
                    self.place_pos = self.env.reach_place_position(
                        self.env.infant,
                        self.env.parent,
                        place_dist=self.place_dist,
                        hold_height=self.hold_height,
                    )
                    self.infant_pos_at_place = infant_xy

            # 手放し完了 → 落下・静止待ち → 到達圏チェック
            if self.placed or self.env.parent["held_obj"] is None:
                if not self.placed:
                    self.placed = True
                    self.release_timer = 0
                self.release_timer += 1
                if self.release_timer >= self.release_settle_steps:
                    obj_pos = self.env._obj_pos(self.target_object.id)
                    # IK 頭打ち手放し後、place_pos より手前に落ちて reach を数 cm 超えることがある
                    reach_r = float(getattr(self.env, "reach_radius", 0.45))
                    obj_dist = float(self.env._dist_body_to(self.env.infant, obj_pos))
                    if obj_dist <= reach_r + 0.03:
                        self.change_state("waiting_infant")
                    else:
                        # エラー対策: 到達圏外配置 → 即 FAILED（イベント、滞在計測対象外）
                        self.fail_on_error_timeout()
            elif self.state_timer > self.placing_timeout_steps:
                # エラー対策: 手放し前の張り付き → 即 FAILED
                self.fail_on_error_timeout()

        elif self.state == "waiting_infant":
            # 幼児の把持を待つ（失敗系: handle_timeout のみ）
            if self.env.infant["held_obj"] == self.target_object.id:
                self._on_infant_grasp_success()
            elif self.state_timer > self.waiting_infant_timeout_steps:
                # 切り分けテスト: 掲げ・取り換えをせず、1 回リトライ後に失敗
                max_retries = 1 if self.close_on_grasp else 3
                action = self.handle_timeout(max_retries=max_retries)
                if action == "RETRY":
                    c_id = self.language_manager.get_concept_for_object(self.target_object)
                    self.queue_utterance("present", concept_id=c_id) # 「ほら、どうぞ」と促し直す
                    self.state_timer = 0
                elif action == "ESCALATE":
                    # 関与の強化: 再把持 → showing → 再 place（発話は showing 進入時）
                    self.placed = False
                    self.release_timer = 0
                    self.place_pos = None
                    self.escalated = True
                    self.change_state("fetching")
                elif action == "CHANGE_TARGET":
                    self.select_target(infant_signals) # 別のオブジェクトに変える
                    self.placed = False
                    self.release_timer = 0
                    self.place_pos = None
                    self.change_state("fetching")
                elif action == "FAILED":
                    self.change_state(self.failure_state)
                    self.is_finished = True

        elif self.state == "reaction":
            # 反応する
            if not self.reaction_flag:
                c_id = self.language_manager.get_concept_for_object(self.target_object)
                self.queue_utterance("praise_with_target", concept_id=c_id)
                self.reaction_flag = True
            
            if len(self.utterance_queue) == 0:
                self.change_state(self.success_state)

    def _get_agent_base_xy(self, agent: Dict[str, Any]):
        """エージェント base の XY とフル位置を返す。"""
        pos, _ = p.getBasePositionAndOrientation(
            agent["body"], physicsClientId=self.env.client_id
        )
        return np.asarray(pos, dtype=np.float32)[:2], np.asarray(pos, dtype=np.float32)

    def _determine_action(self) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        # --- 物理的アクション ---
        if self.state == "fetching":
            self.env.grasp_object(
                self.env.parent,
                self.target_object.id,
                action=action,
                approach_dist=float(self.env.grasp_approach_dist),
            )
        elif self.state == "approach":
            yaw, dist = self.env.approach_infant(
                self.env.parent, self.env.infant, action=action, approach_dist=self.stand_dist
            )
            if dist is not None:
                self.last_distance = dist
            # 運搬中の落下防止
            if self.env.parent["held_obj"] == self.target_object.id:
                self.env.grip_object(self.env.parent, action=action)
        elif self.state == "showing":
            self.env.gaze_agent(
                self.env.parent, self.env.infant, yaw_link="head", action=action
            )
            if self.env.parent["held_obj"] == self.target_object.id:
                # 固定点へ到達するだけ（毎ステップ自ら目を読まない）
                if self.show_pos is None:
                    self.show_pos = self.env.compute_show_position(
                        self.env.parent,
                        self.env.infant,
                        show_dist=self.show_dist,
                        target_object=self.target_object,
                    )
                _, reached = self.env.show_object(
                    self.env.parent,
                    self.show_pos,
                    action=action,
                    reach_tol=float(self.show_reach_tol),
                )
                self.show_reached = bool(reached)
            else:
                # 未把持: 次 step の _manage_state で fetching へ戻す
                self.env.grip_object(self.env.parent, action=action)
                self.show_reached = False
        elif self.state == "placing":
            if self.place_pos is not None and self.env.parent["held_obj"] == self.target_object.id:
                self.env.gaze_object(
                    self.env.parent, self.target_object.id, yaw_link="head", action=action
                )
                _, released = self.env.place_object_at(
                    self.env.parent, self.place_pos, action=action
                )
                if released:
                    # 次の _manage_state で held_obj クリアを検出する
                    pass
            elif self.placed or self.env.parent["held_obj"] is None:
                # 手放し後は腕を戻しつつ幼児を見る
                self.env.gaze_agent(
                    self.env.parent, self.env.infant, yaw_link="head", action=action
                )
                self.env.reset_arm_pose(self.env.parent, action=action)
            else:
                # place_pos 未設定のフォールバック（接近しつつ保持）
                self.env.approach_infant(
                    self.env.parent, self.env.infant, action=action, approach_dist=self.stand_dist
                )
                self.env.grip_object(self.env.parent, action=action)
        elif self.state == "waiting_infant":
            self.env.gaze_agent(
                self.env.parent, self.env.infant, yaw_link="head", action=action
            )
            self.env.reset_arm_pose(self.env.parent, action=action)
        elif self.state == "reaction":
            # 幼児の方を見る
            self.env.gaze_agent(self.env.parent, self.env.infant, yaw_link="head", action=action)
            # 腕を使わないので初期姿勢に戻す
            self.env.reset_arm_pose(self.env.parent, action=action)

        self.debug_info["show_reached"] = self.show_reached
        self.debug_info["escalated"] = self.escalated

        # --- 発話アクション ---
        self._emit_utterance(action)
        
        return action
