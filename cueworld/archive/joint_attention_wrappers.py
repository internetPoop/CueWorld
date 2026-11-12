from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Any, List
from enum import IntEnum


import numpy as np
import gymnasium as gym
from cueworld.base_wrapper import WrapperBase
import pybullet as p  # 環境内部で使用しているため依存


class RJAWrapper(WrapperBase):
    """
    共同注意タスクの報酬ラッパー（差分整形 + フェーズ管理 + 親の行動制御）。

    目的:
      親(Parent)と幼児(Infant)間の共同注意成立プロセスを
      簡易ステートマシンで段階管理し、幼児側の報酬 shaping を行う。
      加えて、親の行動を自動生成してルールベースで制御する。

    フェーズ概要:
      phase0:
        相互注視を形成（双方が相手の頭方向を十分に見る）。
        一定ステップ連続で閾値を満たすと phase1 へ遷移。
      phase1:
        親が音声を発しつつ対象物体を指差し/注視/把持振り上げでキュー提示。
        安定して同一対象が選択されたらターゲットを確定し phase2 へ。
      phase2:
        幼児が音声を発しつつ同じ物体へ指差し/注視/保持操作を継続。
        連続ステップ条件を満たせば phase3 へ。
      phase3:
        幼児が親の方を向く視線交互期間。
        一定ステップ継続後、ja_count をインクリメント。
        ja_count が target_ja_count に達したら success。
      success:
        共同注意達成。成功ボーナス付与。terminate_on_success=True ならエピソード終了。

    報酬設計（幼児のみ）:
      - ステップ毎の微小ペナルティ (step_penalty)
      - phase0: 相互注視スコア増分 + 状態キープ報酬
      - phase1以降: 
        * オブジェクト視線スコア増分 × weight_obj
        * 親の目視線スコア増分 × weight_parent_eye (weight_obj > weight_parent_eye)
        * phase0の状態キープ報酬を継続
      - phase2: phase1と同様の報酬構造
      - phase3: 親への視線スコア増分 + 状態キープ報酬
      - success 到達時に success_bonus

    主な前提:
      - 下位 env は simulation_env.JointAttentionBulletEnv 互換の構造を持つ
      - env.parent / env.infant: 各エージェントメタ情報
      - env.objects: オブジェクト body id のリスト
      - env._agent_nose_pose / env._arm_tip_pose: 幾何計算ユーティリティ
      - env.parent_voice_level / env.infant_voice_level: 現在の音声強度
    """

    # 属性一覧（Attributes）
    # - gaze_partner_thresh (float): 相互注視判定で用いる“相手頭方向”との視線角 閾値[rad]。
    # - gaze_obj_thresh (float): 物体注視の角度閾値[rad]。
    # - point_obj_thresh (float): 指差しの角度閾値[rad]。
    # - select_gaze_thresh (float): 親側のターゲット選択用 視線角の厳しめ閾値[rad]。
    # - select_point_thresh (float): 親側のターゲット選択用 指差し角の厳しめ閾値[rad]。
    # - voice_thresh (float): 音声が有効とみなされるしきい値[-1..1]。
    # - grasp_raise_dot_thresh (float): 保持物の“持ち上げ”判定（腕先+ZのZ成分）しきい値。
    # - shake_vel_thresh (float): 保持物の“振り”とみなす線速度[m/s]しきい値。
    # - w_partner (float): phase0 の相互注視スコア増分の重み。
    # - w_parent_cue (float): phase1 の親キュー（指差し/注視/保持）増分の重み。
    # - w_infant_follow (float): phase2 の幼児追従（指差し/注視/保持）増分の重み。
    # - step_penalty (float): 毎ステップの微小負報酬。
    # - success_bonus (float): 共同注意成功ボーナス。
    # - terminate_on_success (bool): Trueで成功時にエピソード終了。
    # - phase (str): 現在のフェーズ ["phase0","phase1","phase2","success"]。
    # - target_obj_id (int|None): 共同注意対象のオブジェクトID。
    # - steps_in_phase (int): 現フェーズに留まっている連続ステップ数。
    # - eye_contact_gate (float): 相互注視スコアがこの値以上で“良好”と判定。
    # - eye_contact_min_steps (int): phase0で gate 以上を要求する連続ステップ数。
    # - parent_cue_min_steps (int): phase1でターゲット候補の継続要求ステップ数。
    # - infant_follow_min_steps (int): phase2で追従を継続要求するステップ数。
    # - prev_scores (dict): 差分整形用の直前スコア保持 {phase: score}。
    # - eye_contact_steps (int): 相互注視 継続カウンタ。
    # - parent_cue_steps (int): 親キュー 継続カウンタ。
    # - infant_follow_steps (int): 幼児追従 継続カウンタ。
    # - pending_target (int|None): phase1 で継続観察中の暫定ターゲット。
    # - eye_contact (float): 最新の相互注視スコア[0..1]。

    def __init__(self,
                 env: gym.Env,
                 prefer_near_infant: bool = True,
                 parent_cue = "pointing", #親の使用キューのデフォルト設定．
                 function = "heuristic", #共同注意の種類 (affective,heuristic,pragmatic)
                 # 声のしきい値
                 voice_thresh: float = 0.5,
                 infant_voice : bool = True,
                 # 報酬（差分整形）
                 eyecontact_phase_reward_weight: float = 0.1,
                 first_phase_reward_weight_for_obj: float = 0.2,  # phase1以降のオブジェクト視線重み
                 first_phase_reward_weight_for_parent_eye: float = 0.1,  # phase1以降の親の目視線重み
                 second_phase_reward_weight_for_obj: float = 0.3,  # phase2のオブジェクト視線重み
                 second_phase_reward_weight_for_parent_eye: float = 0.15,  # phase2の親の目視線重み
                 success_bonus: float = 3.0,
                 failure_penalty: float = -3.0,  # 追加: 誤った物体を注視した場合の負の報酬
                 # 絶対的報酬の設定
                 eye_contact_reward: float = 0.1,
                 gaze_ip_reward: float = 0.05,
                 gaze_face_ip_reward: float = 0.03,
                 gaze_p_reward: float = 0.01,
                 gaze_obj_ip_reward: float = 0.08,
                 dyadic_reward: float = 1.0,
                 step_penalty: float = -0.001,
                 voice_reward: float = 0.02,
                 # フェーズ継続条件
                 eye_contact_min_steps: int = 8,
                 parent_cue_min_steps: int = 30,
                 infant_follow_min_steps: int = 30,
                 max_phase_steps: int = 400,
                 max_ja_count: int = 1,
                 # 親の行動制御パラメータ(接近距離)
                 approach_distance: float | None = None,
                 approach_distance_for_grasp: float | None = None, #0.5 安定で掴める
                 approach_distance_for_point: float | None = None, #0.8: ちょうど触れる ~ 1.0: 離れる
                 approach_distance_for_gaze: float | None = None,
                 approach_distance_for_show: float | None = None,
                 # 親の行動制御パラメータ(角度しきい値)
                 gaze_thresh: float = math.radians(10.0),
                 point_thresh: float = math.radians(10.0),
                 approach_thresh: float = math.radians(15.0),
                 voice_level: float = 1.0,
                 shake_period: int = 6,
                 shake_amp: float = None,
                 use_ik: bool = True,
                 ik_every: int = 0,
                 #評価用
                 eval_mode: bool = False,
                 ideal_infant_action: bool = False,
                 no_gaze: bool = False,
                 no_voice: bool = False,
                 *args, **kwargs):
        """パラメータ説明（要点のみ）:
        gaze_partner_thresh_deg:
          相互注視判定で用いる“相手頭方向”との視線角の閾値[deg]
        gaze_obj_thresh_deg / point_obj_thresh_deg:
          物体注視 / 指差し判定に用いる角度閾値[deg]
        select_*_thresh_deg:
          親ターゲット選択用の少し厳しめの閾値（未指定なら通常閾値使い回し）
        voice_thresh:
          音声が「有効キュー」とみなされる最小強度
        grasp_raise_dot_thresh:
          把持物体の“持ち上げ”判定用 上向き(Z成分)しきい値
        shake_vel_thresh:
          “振っている”とみなす線速度しきい値[m/s]
        shaping_weight_*:
          各フェーズの差分スコアに掛ける重み
        step_penalty:
          毎ステップ一定の微小負報酬
        success_bonus:
          共同注意成功時のボーナス
        *_min_steps:
          各フェーズで条件を連続何ステップ満たす必要があるか
        terminate_on_success:
          True なら success 到達で done=True
        """
        super().__init__(env)
        
        # 共同注意の種類
        self.function = function
        

        # 閾値設定
        self.voice_thresh = float(voice_thresh)
        self.approach_thresh = float(approach_thresh)
        self.gaze_thresh = float(gaze_thresh)
        self.point_thresh = float(point_thresh)
        
        # 差分整形重み
        self.eyecontact_phase_reward_weight = float(eyecontact_phase_reward_weight)
        self.first_phase_reward_weight_for_obj = float(first_phase_reward_weight_for_obj)
        self.first_phase_reward_weight_for_parent_eye = float(first_phase_reward_weight_for_parent_eye)
        self.second_phase_reward_weight_for_obj = float(second_phase_reward_weight_for_obj)
        self.second_phase_reward_weight_for_parent_eye = float(second_phase_reward_weight_for_parent_eye)
        
        # 絶対的報酬の設定
        self.eye_contact_reward = float(eye_contact_reward)  # 相互注視成功報酬
        self.gaze_ip_reward = float(gaze_ip_reward)  # 幼児が親の目を見る報酬
        self.gaze_face_ip_reward = float(gaze_face_ip_reward)  # 幼児が親の顔を見る報酬
        self.gaze_p_reward = float(gaze_p_reward)  # 幼児が親を見る報酬
        self.gaze_obj_ip_reward = float(gaze_obj_ip_reward)  # 幼児が対象オブジェクトを見る報酬
        self.step_penalty = float(step_penalty)  # ステップペナルティ
        self.voice_reward = float(voice_reward)  # 幼児の音声報酬
        
        self.dyadic_reward = float(dyadic_reward)  # アイコンタクト達成報酬
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)  # 追加

        # フェーズ管理
        self.phase: str = "phase0"
        self.parent_target_obj: Optional[int] = None
        self.steps_in_phase: int = 0
        self.prefer_near_infant = prefer_near_infant
        self.max_phase_steps = max_phase_steps

        # 継続ゲートと連続必要ステップ
        self.eye_contact_min_steps = int(eye_contact_min_steps)
        self.parent_cue_min_steps = int(parent_cue_min_steps)
        self.infant_follow_min_steps = int(infant_follow_min_steps)
        
        # 共同注意の繰り返し回数管理
        self.max_ja_count = int(max_ja_count)

        # 差分報酬用の直前スコア
        self.prev_scores = {
            "parent_gaze_score": None,
            "object_gaze_score": None,
        }
        
        # 連続カウンタと保留ターゲット
        self.eye_contact_steps = 0
        self.parent_cue_steps = 0
        self.infant_follow_steps = 0
        self.infant_wrong_follow_steps = 0  # 追加: 誤った物体を注視するカウンタ
        self.parent_gaze_steps = 0  # phase3用
        self.success_num = 0  # 共同注意成功回数
        
        # phase0の状態キープ報酬値を保存
        self.phase0_keep_reward = 0.0
        
        
        # 親の使用キューに関するモード
        self.parent_cue = parent_cue
        self.parent_target_link = "eye" #phase1で幼児が注目するべきところ
        self.parent_target_link_reset_flag = False  # 追加: リセットフラグ
        self.grasp_type_cue_ok = False #親が掴む系のキューを使う場合に，キューが完了しているかどうかのフラグ

        # 親の行動制御パラメータ
        self.approach_distance = float(approach_distance) if approach_distance is not None else None
        self.approach_distance_for_grasp = float(approach_distance_for_grasp) if approach_distance_for_grasp is not None else None
        self.approach_distance_for_point = float(approach_distance_for_point) if approach_distance_for_point is not None else None
        self.approach_distance_for_gaze = float(approach_distance_for_gaze) if approach_distance_for_gaze is not None else None
        self.approach_distance_for_show = float(approach_distance_for_show) if approach_distance_for_show is not None else None
        
        self.voice_level = float(voice_level)
        self.shake_period = max(2, int(shake_period))
        if shake_amp is None:
            if self.env.use_cinematic_model:
                self.shake_amp = 0.2
            else:
                self.shake_amp = 1.0
        else:
            self.shake_amp = float(np.clip(shake_amp, 0.0, 1.0))

        # 親の内部状態
        self.shake_t: int = 0
        self.shake_flag = False
        self.arm_raise_flag = False
        self.repeat_flag: bool = False #視線交互やタイムアウト時などを行ったかどうかのフラグ
        self.emotion = 1.0
        
        self.relocate_object_id = None
        self.yaw_to_relocate = None
        self.previous_forward = 0.0 #邪魔なオブジェクトを避けるときに使う(その時しかこの値は更新しないので注意)
        
        self.ik_every = ik_every #IKを使うステップ数（0でIKを1回のみ呼び出す）
        self.use_ik = use_ik
        
        # 親の位置調整フラグ/位置
        self.adjust_location = False if self.approach_distance is not None else True #TODO: locationを調整したかどうか
        self.point_phase = False
        self.get_location = False
        self.good_location = None
        self.adjust_torso_angle = True #TODO 簡単のためtrueにした
        self.first_dir_pos = None
        
        #幼児の声を使うかどうか
        self.infant_voice = infant_voice
        
        #評価のモード
        self.eval_mode = eval_mode
        if eval_mode:
            self.ideal_infant_action = ideal_infant_action
            self.eval_dict = {
                "joint_attention_phase": False,
                "gaze_object": False,
                "shortest": 0.0, #結果的に移動したスコア量
                "steps": 0.0, #そのエピソードでのステップ数を数える
                "delta": 0.0, #スコアの差分
                "success": False,
            }
            self.first_score = None
        else:
            self.ideal_infant_action = False
        # 視線/音声無効モード
        self.no_gaze = no_gaze
        self.no_voice = no_voice
                

    def _select_parent_target_object(self) -> Optional[int]:
        #親が対象とするオブジェクトを選択
        if hasattr(self.env, "target_obj"):
            return self.env.target_obj
        return self.np_random.choice(self.env.active_objects) if len(self.env.active_objects) > 0 else None

    # ========= Wrapper: reset/step =========
    def reset(self, **kwargs):
        """下位環境をリセットし、共同注意フェーズ/ターゲット/カウンタを初期化して返す。"""
        obs, info = self.env.reset(**kwargs)
        self.phase = "phase0"
        self.parent_target_obj = self._select_parent_target_object()
        if self.function == "affective":
            self.emotion = self.env.object_attributes[self.parent_target_obj]["emotion"]
        self.steps_in_phase = 0
        self.success_num = 0  # 共同注意成功回数
        
        
        # 差分・継続の状態初期化
        self.prev_scores = {
            "parent_gaze_score": None,
            "object_gaze_score": None,
        }
        self.eye_contact_steps = 0
        self.parent_cue_steps = 0
        self.infant_follow_steps = 0
        self.infant_wrong_follow_steps = 0
        self.parent_gaze_steps = 0
        self.eye_contact = 0.0
        self.phase0_keep_reward = 0.0

        # 親の行動制御の初期化
        self.shake_t = 0
        self.shake_flag = False
        self.arm_raise_flag = False
        self.repeat_flag = False
        self.close_to_infant = False
        # 追加: 回避/再配置用内部状態
        self.relocate_object_id = None
        self.yaw_to_relocate = None
        self.previous_forward = 0.0
        
        # 親の位置調整フラグ/位置 #TODO: これらのリセットが正しく行われているか確認
        self.adjust_location = True
        self.point_phase = False
        self.get_location = False
        self.good_location = None
        self.adjust_torso_angle = True #TODO 簡単のためtrueにした
        self.first_dir_pos = None
        
        if self.eval_mode:
            self.eval_dict = {
                "joint_attention_phase": False,
                "gaze_object": False,
                "shortest": 0.0,
                "steps": 0.0,
                "delta": 0.0,
                "success": False,
            }

        # 情報追加
        info = dict(info or {})
        info["parent_action"] = self.subgoal_management()
        if self.ideal_infant_action:
            info["ideal_infant_action"] = self.ideal_infant_action_step()
        
        return obs, info

    def ideal_infant_action_step(self):
        parent = self.env.parent
        infant = self.env.infant
        action = len(self.env.action_map) * [0.0]
        am = self.env.action_map
        if self.phase == "phase2":
            yaw, _ = self.gaze_object(infant, self.parent_target_obj, action=action, yaw_link="head")
        else:
            yaw, _ =self.gaze_agent(infant, parent, action=action, yaw_link="head")
        
        for name, step_size in self.env.infant_joint_mappings.items():
            if name == "voice":
                continue
            action[am[name]] = action[am[name]]/(step_size*self.env.action_repeat)
        clipped = tuple(float(np.clip(v, -1.0, 1.0)) for v in action)
        return clipped
        
    def subgoal_management(self):
        """
        親の行動を決定するためのサブゴール管理ロジック。
        """
        parent = self.env.parent
        infant = self.env.infant
        action = len(self.env.action_map) * [0.0]
        am = self.env.action_map
        main_link = "eye" #その過程で使う主なリンク．幼児が注目するべきところ

        if self.phase is None or self.phase == "phase0":
            action[am["voice"]] = 1.0
            action[am["lip"]] = 0.08
            if self.adjust_location:
                if not self.get_location: #TODO: オブジェクトの位置が変わった場合に対応する
                    self.good_location, self.target_obj_pos, self.first_dir_pos = self.spot_for_joint_attention(parent, self.parent_target_obj, approach_dist=self.approach_distance_for_point)
                    self.get_location = True
                if self.good_location is None:
                    yaw, distance = self.approach_object(parent, self.parent_target_obj, action=action, thresh=self.approach_thresh, approach_dist=self.approach_distance_for_point)
                    if abs(yaw) < math.radians(10.0) and distance is not None and distance <= self.approach_distance_for_point:
                        #位置調整が完了
                        self.adjust_location = False
                else:
                    if not self.adjust_torso_angle:
                        yaw, distance = self.approach_location(parent, self.good_location, action=action, thresh=self.approach_thresh)
                        if distance is not None and distance <= 0.05:
                            self.adjust_torso_angle = True
                    elif self.adjust_torso_angle:
                        yaw = self.gaze_location(parent, self.first_dir_pos, action=action, yaw_link="torso")
                        if abs(yaw) < math.radians(5.0):
                            self.adjust_torso_angle = False
                            #位置調整が完了
                            self.adjust_location = False
            if self.parent_cue == "shaking" and self.repeat_flag:
                action = self.reset_arm_pose(parent, action)
            
            yaw, _ =self.gaze_agent(parent, infant, action=action, yaw_link="head")
            
        elif self.phase == "success":
            pass
        else:
            #TODO: いったんこれらなし．声や表情は簡単に扱う
            # #指示を始めたときから声を出す(affectiveでは表情も)
            # if self.function == "heuristic":
            #     action[am["voice"]] = 1.0
            # elif self.function == "affective":
            #     if self.emotion > 0.0:
            #         action[am["voice"]] = 1.0
            #         action[am["lip"]] = 1.0
            #     elif self.emotion < 0.0:
            #         action[am["voice"]] = -1.0
            #         action[am["lip"]] = -1.0
            # ターゲットが無ければ何もしない
            if self.no_voice or self.parent_cue == "nothing":
                action[am["voice"]] = 0.0
            else:
                action[am["voice"]] = -1.0
            action[am["lip"]] = 0.08
            if self.parent_target_obj is None:
                print("ターゲットの未設定")
                breakpoint()
                return None
            #キューごとに処理
            if self.parent_cue == "shaking":
                held_obj = parent.get("held_obj")
                holding = (held_obj == self.parent_target_obj)
                # 1) まだ持っていないなら、対象へ接近 → 掴む
                if not holding: #仮にrelocate stepで今回の対象オブジェクトを掴んでしまってもここで分岐するので問題なし
                    if held_obj is not None:
                        self.relocate_object_id = held_obj
                    elif self.env.disable_all_body_object_collision == False and self.previous_forward > 0.0:
                        self.relocate_object_id = None if self.env.parent_collide == self.parent_target_obj else self.env.parent_collide
                    if self.relocate_object_id is not None:
                        return self.relocate_step()
                    #掴む
                    action, main_link = self.grasp_object(parent, self.parent_target_obj, action=action, approach_dist=self.approach_distance_for_grasp, main_link_need=True)
                else:
                    self.grasp_type_cue_ok = True
                    self.shake_object(parent, infant, oid=self.parent_target_obj, action=action, thresh=self.approach_thresh, sub_thresh=self.gaze_thresh, approach_dist=self.approach_distance_for_show, shake_amp=self.shake_amp, shake_period=self.shake_period)
            elif self.parent_cue == "looming":
                held_obj = parent.get("held_obj")
                holding = (held_obj == self.parent_target_obj)
                # 1) まだ持っていないなら、対象へ接近 → 掴
                if not holding: #仮にrelocate stepで今回の対象オブジェクトを掴んでしまってもここで分岐するので問題なし
                    if held_obj is not None:
                        self.relocate_object_id = held_obj
                    elif self.env.disable_all_body_object_collision == False and self.previous_forward > 0.0:
                        self.relocate_object_id = None if self.env.parent_collide == self.parent_target_obj else self.env.parent_collide
                    if self.relocate_object_id is not None:
                        return self.relocate_step()
                    #掴む
                    action, main_link = self.grasp_object(parent, self.parent_target_obj, action=action, approach_dist=self.approach_distance_for_grasp, main_link_need=True)
                else:
                    self.grasp_type_cue_ok = True
                    self.show_object(parent, infant, oid=self.parent_target_obj, action=action, thresh=self.approach_thresh, sub_thresh=self.gaze_thresh, approach_dist=self.approach_distance_for_show)
            elif self.parent_cue == "pointing":
                if self.no_gaze:
                    self.point_phase = True
                elif self.point_phase:
                    pass
                else:
                    yaw, _ = self.gaze_object(parent, self.parent_target_obj, action=action, yaw_link="torso")
                    if self.no_gaze or abs(yaw) < math.radians(5.0):
                        main_link = "index"
                        self.point_phase = True
                if not self.repeat_flag and self.point_phase:
                    action = self.point_object(parent, self.parent_target_obj, action=action, ik_every=self.ik_every) #TODO ここでtorsoが回転したら次に幼児エージェントを見るときに，head_panのlimitに引っかかり幼児を見れなくなる可能性がある
                    self.gaze_object(parent, self.parent_target_obj, action=action, yaw_link="head")
            elif self.parent_cue == "gaze":
                # 視線が十分に対象へ向いているか判定
                
                if self.no_gaze:
                    pass
                else:
                    yaw, _ = self.gaze_object(parent, self.parent_target_obj, action=action, yaw_link="head")
                # if self.repeat_flag:
                #     yaw, _ = self.gaze_object(parent, self.parent_target_obj, action=action, yaw_link="head")
                #     distance = self._dist_body_to(parent, self.env._obj_pos(self.parent_target_obj))
                # else:
                #     yaw, distance = self.approach_object(parent, self.parent_target_obj, action=action, thresh=self.approach_thresh, approach_dist=self.approach_distance_for_gaze)
                # if abs(yaw) < math.radians(10.0) and distance is not None and distance <= self.approach_distance_for_gaze:
                #     main_link = "eye"
            elif self.parent_cue == "nothing":
                pass
            else:
                NotImplementedError(f"Unknown parent cue type: {self.parent_cue}")   
        
        # torsoの回転による効果を考慮したhead_panの修正
        #loomingやshakingと相性悪いのでコメントアウト
        # torso = action[am["torso_joint"]]
        # torso_pn = 1 if torso >= 0 else -1
        # torso = min(abs(torso),self.env.joint_mappings["torso_joint"])*torso_pn
        # action[am["head_pan_joint"]] -= torso 
        # 追加: 最終アクションをクリップ
        # 正規化には action_repeat を考慮したステップ幅を使用する
        for name, step_size in self.env.parent_joint_mappings.items():
            if name == "voice":
                continue 
            action[am[name]] = action[am[name]]/(step_size*self.env.action_repeat)
        clipped = tuple(float(np.clip(v, -1.0, 1.0)) for v in action)
        
        if self.parent_target_link != main_link:
            self.parent_target_link_reset_flag = True
        self.parent_target_link = main_link
   
        return clipped

    def check_rja(self, parent_action) -> tuple[float, bool, Dict[str, Any]]:
        """フェーズ遷移と幼児報酬計算。
        戻り値:
          (幼児報酬, successで早期終了フラグ, 情報辞書)
        """
        infant_reward = 0.0
        ja_done = False

        parent = self.env.parent
        infant = self.env.infant
        
        # ステップペナルティ
        infant_reward += self.step_penalty
        
        if self.success_num >= 1 and self.eval_mode:
            self.eval_dict["joint_attention_phase"] = True
        
        
        # フェーズ更新と整形報酬（差分）
        if self.phase == "phase0":
            # 相互注視判定(絶対的報酬)
            eye_contact_flag, gaze_eye_ip, gaze_face_ip, gaze_ip = self.is_eye_contact()
            if eye_contact_flag:
                #アイコンタクト成立
                infant_reward += self.eye_contact_reward
                if not self.adjust_location:
                    self.eye_contact_steps += 1
                else:
                    self.eye_contact_steps = 0
            elif gaze_eye_ip:
                #目を見ることには成功
                infant_reward += self.gaze_ip_reward
                self.eye_contact_steps = 0
            elif gaze_face_ip:
                #顔を見ることには成功
                infant_reward += self.gaze_face_ip_reward
                self.eye_contact_steps = 0
            elif gaze_ip:
                #親を見ることには成功
                infant_reward += self.gaze_p_reward
                self.eye_contact_steps = 0
            else: 
                self.eye_contact_steps = 0
        
            # 差分（増加分のみ）を整形報酬
            angle_to_parent = self.gaze_angle_to_agent(infant, parent)
            gaze_parent_score = self._angle_to_score(angle_to_parent)
            if self.prev_scores["parent_gaze_score"] is not None:
                infant_reward += (gaze_parent_score - self.prev_scores["parent_gaze_score"]) * self.eyecontact_phase_reward_weight
            self.prev_scores["parent_gaze_score"] = gaze_parent_score
            
            # 継続: アイコンタクトが一定期間続いたら phase1 へ
            if self.eye_contact_steps >= self.eye_contact_min_steps:
                self.phase = "phase1"
                self.steps_in_phase = 0
                self.parent_cue_steps = 0
                infant_reward += self.dyadic_reward

        elif self.phase == "phase1":
            if self.eval_mode:
                self.eval_dict["joint_attention_phase"] = True
            # # オブジェクトへの視線スコア差分
            # object_gaze_angle = self.gaze_angle_to_object(infant, self.parent_target_obj)
            # object_gaze_score = self._angle_to_score(object_gaze_angle)
            # object_delta = 0.0
            # if self.prev_scores["object_gaze_score"] is not None:
            #     object_delta = object_gaze_score - self.prev_scores["object_gaze_score"]
            # self.prev_scores["object_gaze_score"] = object_gaze_score
            
            # # 親の目への視線スコア差分
            # parent_eye_angle = self.gaze_angle_to_agent(infant, parent)
            # parent_eye_score = self._angle_to_score(parent_eye_angle)
            # parent_delta = 0.0
            # if self.prev_scores["parent_gaze_score"] is not None:
            #     parent_delta = parent_eye_score - self.prev_scores["parent_gaze_score"]
            # self.prev_scores["parent_gaze_score"] = parent_eye_score
            
            # # 組み合わせ報酬（オブジェクト重視）
            # infant_reward += (self.first_phase_reward_weight_for_obj * object_delta + 
            #                 self.first_phase_reward_weight_for_parent_eye * parent_delta)
            # 相互注視判定(絶対的報酬)
            if self.eval_mode:
                object_gaze_angle = self.gaze_angle_to_object(infant, self.parent_target_obj)
                object_gaze_score = self._angle_to_score(object_gaze_angle)
                self.first_score = object_gaze_score if self.first_score is None else self.first_score
                object_delta = 0.0
                if self.prev_scores["object_gaze_score"] is not None:
                    object_delta = object_gaze_score - self.prev_scores["object_gaze_score"]
                self.prev_scores["object_gaze_score"] = object_gaze_score
                self.eval_dict["delta"] = abs(object_delta)
                self.eval_dict["steps"] = 1.0
            eye_contact_flag, gaze_eye_ip ,_,_ = self.is_eye_contact()
            if eye_contact_flag:
                #アイコンタクト成立
                infant_reward += self.eye_contact_reward
            elif gaze_eye_ip:
                #目を見ることには成功
                infant_reward += self.gaze_ip_reward
        
            # 差分（増加分のみ）を整形報酬
            angle_to_parent = self.gaze_angle_to_agent(infant, parent)
            gaze_parent_score = self._angle_to_score(angle_to_parent)
            if self.prev_scores["parent_gaze_score"] is not None:
                infant_reward += (gaze_parent_score - self.prev_scores["parent_gaze_score"]) * self.eyecontact_phase_reward_weight
            self.prev_scores["parent_gaze_score"] = gaze_parent_score

            # 親のキュー完了判定
            if self.parent_target_obj is not None:
                cue_ok = False
                if self.parent_cue in ["shaking", "looming", "grasp"]:
                    #掴む系のキュー
                    cue_ok = self.grasp_type_cue_ok
                elif self.no_gaze or self.parent_cue == "nothing":
                    #評価用なので，常にOKとする
                    cue_ok = True
                else:
                    cue_ok = self.object_gaze_check(parent, self.parent_target_obj, mode="angle")
                #cue_okに基づき継続カウンタ更新
                if cue_ok:
                    self.parent_cue_steps += 1
                else:
                    self.parent_cue_steps = 0
            else:
                self.parent_cue_steps = 0
            
            # 確定: 親のキューが一定期間続いたら phase2 へ
            if self.parent_cue_steps >= self.parent_cue_min_steps:
                self.phase = "phase2"
                self.steps_in_phase = 0
                self.infant_follow_steps = 0

        elif self.phase == "phase2":
            # オブジェクトへの視線スコア差分
            object_gaze_angle = self.gaze_angle_to_object(infant, self.parent_target_obj)
            object_gaze_score = self._angle_to_score(object_gaze_angle)
            if self.eval_mode:
                self.eval_dict["joint_attention_phase"] = True
                self.first_score = object_gaze_score if self.first_score is None else self.first_score
                self.eval_dict["steps"] = 1.0
            object_delta = 0.0
            if self.prev_scores["object_gaze_score"] is not None:
                object_delta = object_gaze_score - self.prev_scores["object_gaze_score"]
            self.prev_scores["object_gaze_score"] = object_gaze_score
            if self.eval_mode:
                self.eval_dict["delta"] = abs(object_delta)
            
            # 親の目への視線スコア差分
            parent_eye_angle = self.gaze_angle_to_agent(infant, parent)
            parent_eye_score = self._angle_to_score(parent_eye_angle)
            parent_delta = 0.0
            if self.prev_scores["parent_gaze_score"] is not None:
                parent_delta = parent_eye_score - self.prev_scores["parent_gaze_score"]
            self.prev_scores["parent_gaze_score"] = parent_eye_score
            
            # 組み合わせ報酬（オブジェクト重視）
            infant_reward += (self.first_phase_reward_weight_for_obj * object_delta + 
                            self.first_phase_reward_weight_for_parent_eye * parent_delta)
            
            # 幼児の追従判定
            if self.infant_voice:
                infant_voice = self._infant_voice_on()
                if self.function == "heuristic":
                    infant_voice_ok = infant_voice >= self.voice_thresh
                elif self.function == "affective":
                    if self.emotion > 0.0:
                        infant_voice_ok = infant_voice >= self.voice_thresh
                    elif self.emotion < 0.0:
                        infant_voice_ok = infant_voice <= self.voice_thresh
            
            infant_gazing_obj = self.what_is_looking_at(infant)
            object_gaze = infant_gazing_obj == self.parent_target_obj
            if self.eval_mode:
                self.eval_dict["gaze_object"] = object_gaze
            if self.function == "heuristic":
                infant_ok = object_gaze #TODO:将来的には声も使う
            elif self.function == "affective":
                infant_ok = object_gaze and infant_voice_ok
            
            if infant_ok:
                infant_reward += self.gaze_obj_ip_reward
                self.infant_follow_steps += 1
                self.infant_wrong_follow_steps = 0  # 正しい物体を見ているのでリセット
                if self.infant_voice and infant_voice_ok:
                    infant_reward += self.voice_reward
            else:
                # 幼児が親のターゲットと異なる物体を注視しているかチェック
                if infant_gazing_obj is not None and infant_gazing_obj != self.parent_target_obj:
                    # 誤った物体を注視している
                    self.infant_wrong_follow_steps += 1
                    self.infant_follow_steps = 0
                    
                    # 一定期間誤った物体を注視し続けた場合に負の報酬
                    if self.infant_wrong_follow_steps >= self.infant_follow_min_steps:
                        infant_reward += self.failure_penalty
                        # ペナルティを与えた後、phase0にリセット
                        self.phase = "phase0"
                        self.steps_in_phase = 0
                        self.parent_gaze_steps = 0
                        self.infant_wrong_follow_steps = 0
                        self.repeat_flag = True
                else:
                    # 何も見ていないか、視線が定まっていない
                    self.infant_follow_steps = 0
                    self.infant_wrong_follow_steps = 0

            if self.infant_follow_steps >= self.infant_follow_min_steps:
                infant_reward += self.success_bonus
                self.success_num += 1
                self.infant_wrong_follow_steps = 0  # リセット
                if self.eval_mode:
                    self.eval_dict["success"] = True
                if self.success_num == self.max_ja_count:
                    # 1回で終了の場合は直接successへ
                    self.phase = "success"
                    self.steps_in_phase = 0
                    ja_done = True
                else:
                    # 複数回繰り返す場合はphase0へ
                    self.phase = "phase0"
                    self.steps_in_phase = 0
                    self.parent_gaze_steps = 0
                    self.repeat_flag = True
                    # phase3の初期スコアを設定

        elif self.phase == "success":
            # 追加の shaping は特に無し
            pass
        

        self.steps_in_phase += 1
        if self.steps_in_phase >= self.max_phase_steps:
            # フェーズタイムアウト時は phase0 にリセット
            self.phase = "phase0"
            self.steps_in_phase = 0
            self.eye_contact_steps = 0
            self.infant_wrong_follow_steps = 0  # 追加
            self.prev_scores["phase0"] = 0.0
            self.repeat_flag = True
            self.shake_flag = False
            self.grasp_type_cue_ok = False
            self.success_num = 0
        
        info_update = {
            "ja_phase": self.phase,
            "parent_target_obj": self.parent_target_obj,
        }
        return float(infant_reward), bool(ja_done), info_update

    def step(self, action):
        """
        actionが (infant_action, parent_action) のときは親を上書き。
        actionが1エージェント分のときは幼児のみとみなし親を自動生成。
        """
        self.previous_parent_voice_level = self._parent_voice_on() #1step前の音声
        self.previous_infant_voice_level = self._infant_voice_on()
        obs, base_reward, done, truncated, info = self.env.step(action)
        self.eval_dict = {
                "joint_attention_phase": False,
                "gaze_object": False,
                "steps": 0.0,
                "shortest": 0.0,
                "delta": 0.0,
                "success": False,
            }
        info = dict(info or {})
        # 共同注意の評価
        infant_reward, ja_done, ja_info = self.check_rja(action[1])
        parent_action = self.subgoal_management()

        # 親の報酬は0固定（差分報酬に置換）
        reward = [float(infant_reward), 0.0]
        done = bool(done or ja_done)
        if done and self.eval_mode:
            # どれくらい近づいたか(スコア)
            object_gaze_angle = self.gaze_angle_to_object(self.env.infant, self.parent_target_obj)
            object_gaze_score = self._angle_to_score(object_gaze_angle)
            self.eval_dict["shortest"] = abs(object_gaze_score - self.first_score) if self.first_score is not None else None

        info.update(ja_info)
        info["parent_action"] = parent_action
        if self.ideal_infant_action:
            info["ideal_infant_action"] = self.ideal_infant_action_step()
        info["eval_dict"] = self.eval_dict if self.eval_mode else None
        
        return obs, reward, done, truncated, info

    def __getattr__(self, name):
        """env へ属性を委譲するフォールバック。Wrapper自身に無い属性を下位環境から引き出す。"""
        # 無限再帰を防ぐ
        if name == "env":
            return super().__getattribute__(name)
        return getattr(self.env, name)

class IJAWrapper(WrapperBase):
    """
    共同注意タスク（IJA: 幼児発信）のラッパー。
    RJAWrapperと対称的に、幼児がIJAを開始し、親(Parent)がそれにルールベースで応答するシナリオ。

    フェーズ概要:
      phase0:
        相互注視を形成。一定ステップ連続で閾値を満たすと phase1 へ。
      phase1:
        幼児が対象物体を注視/指差しでキュー提示（Initiation）。
        安定して同一対象が選択されたらターゲット(infant_target_obj)を確定し phase2 へ。
      phase2:
        親が幼児の対象に応答(注視・発声等)。
        親が追従を完了する間、幼児が対象や親への注意を継続すれば success。
      success:
        共同注意達成。成功ボーナス付与。

    報酬設計（幼児のみ）:
      - RJAと同様、各フェーズでの差分スコアを基にした shaping 報酬を付与。
    """
    def __init__(self,
                 env: gym.Env,
                 function = "heuristic",
                 # 声のしきい値
                 voice_thresh: float = 0.5,
                 infant_voice : bool = True,
                 # 報酬（差分整形）
                 eyecontact_phase_reward_weight: float = 0.1,
                 first_phase_reward_weight_for_obj: float = 0.2, 
                 first_phase_reward_weight_for_parent_eye: float = 0.1,
                 second_phase_reward_weight_for_obj: float = 0.3,
                 second_phase_reward_weight_for_parent_eye: float = 0.15,
                 success_bonus: float = 3.0,
                 failure_penalty: float = -3.0,
                 # 絶対的報酬の設定
                 eye_contact_reward: float = 0.1,
                 gaze_ip_reward: float = 0.05,
                 gaze_face_ip_reward: float = 0.03,
                 gaze_p_reward: float = 0.01,
                 gaze_obj_ip_reward: float = 0.08,
                 dyadic_reward: float = 1.0,
                 step_penalty: float = -0.001,
                 voice_reward: float = 0.02,
                 # フェーズ継続条件
                 eye_contact_min_steps: int = 8,
                 infant_cue_min_steps: int = 8,
                 parent_follow_min_steps: int = 8,
                 max_phase_steps: int = 400,
                 max_ja_count: int = 1,
                 # 親の行動制御用
                 approach_distance: float | None = None,
                 approach_thresh: float = math.radians(15.0),
                 # IJA特有
                 attention_threshold: float = 0.6,
                 eval_mode: bool = False,
                 *args, **kwargs):
        super().__init__(env)
        
        self.function = function
        self.voice_thresh = float(voice_thresh)
        self.infant_voice = infant_voice
        self.approach_thresh = float(approach_thresh)

        # 報酬重み
        self.eyecontact_phase_reward_weight = float(eyecontact_phase_reward_weight)
        self.first_phase_reward_weight_for_obj = float(first_phase_reward_weight_for_obj)
        self.first_phase_reward_weight_for_parent_eye = float(first_phase_reward_weight_for_parent_eye)
        self.second_phase_reward_weight_for_obj = float(second_phase_reward_weight_for_obj)
        self.second_phase_reward_weight_for_parent_eye = float(second_phase_reward_weight_for_parent_eye)
        
        self.eye_contact_reward = float(eye_contact_reward)
        self.gaze_ip_reward = float(gaze_ip_reward)
        self.gaze_face_ip_reward = float(gaze_face_ip_reward)
        self.gaze_p_reward = float(gaze_p_reward)
        self.gaze_obj_ip_reward = float(gaze_obj_ip_reward)
        self.step_penalty = float(step_penalty)
        self.voice_reward = float(voice_reward)
        self.dyadic_reward = float(dyadic_reward)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)

        self.eye_contact_min_steps = int(eye_contact_min_steps)
        self.infant_cue_min_steps = int(infant_cue_min_steps)
        self.parent_follow_min_steps = int(parent_follow_min_steps)
        self.max_phase_steps = int(max_phase_steps)
        self.max_ja_count = int(max_ja_count)
        self.approach_distance = float(approach_distance) if approach_distance is not None else None
        
        self.attention_threshold = float(attention_threshold)
        
        self.eval_mode = eval_mode

        # 状態
        self.phase: str = "phase0"
        self.infant_target_obj: Optional[int] = None
        self.steps_in_phase: int = 0
        self.success_num = 0
        
        self.prev_scores = {
            "parent_gaze_score": None,
            "object_gaze_score": None,
            "attention_score": None,
        }
        
        self.eye_contact_steps = 0
        self.infant_cue_steps = 0
        self.parent_follow_steps = 0
        self.attention_scores: List[float] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.phase = "phase0"
        self.infant_target_obj = None
        self.steps_in_phase = 0
        self.success_num = 0
        
        self.prev_scores = {
            "parent_gaze_score": None,
            "object_gaze_score": None,
            "attention_score": None,
        }
        self.eye_contact_steps = 0
        self.infant_cue_steps = 0
        self.parent_follow_steps = 0
        
        num_objects = len(getattr(self.env, "active_objects", []))
        self.attention_scores = [0.0] * num_objects

        info = dict(info or {})
        info["parent_action"] = self.subgoal_management()
        return obs, info

    def _update_attention_scores(self) -> Tuple[Optional[int], float]:
        infant = self.env.infant
        obj_ids = getattr(self.env, "active_objects", [])
        
        self.attention_scores = [0.0] * len(obj_ids)
        best_oid = None
        best_score = 0.0

        for i, oid in enumerate(obj_ids):
            g_ang = self.gaze_angle_to_object(infant, oid)
            p_ang = self.pointing_angle_to_object(infant, oid)
            
            # しきい値は20度程度として
            g_score = self._angle_to_score(g_ang, math.radians(20.0))
            p_score = self._angle_to_score(p_ang, math.radians(20.0))
            
            score = max(g_score, p_score)
            self.attention_scores[i] = score
            if score > best_score:
                best_score = score
                best_oid = oid
                
        return best_oid, best_score

    def subgoal_management(self):
        parent = self.env.parent
        infant = self.env.infant
        action = len(self.env.action_map) * [0.0]
        am = self.env.action_map

        if self.phase == "phase0" or self.phase == "phase1":
            # 幼児に注目して待機
            yaw, distance = self.approach_infant(parent, infant, action=action, thresh=self.approach_thresh, approach_dist=self.approach_distance or 1.0)
            if abs(yaw) < math.radians(10.0):
                self.gaze_agent(parent, infant, action=action, yaw_link="head")
        
        elif self.phase == "phase2":
            # 幼児が注目したオブジェクトを見る / 反応する
            if self.infant_target_obj is not None:
                yaw, distance = self.approach_object(parent, self.infant_target_obj, action=action, thresh=self.approach_thresh, approach_dist=self.approach_distance or 1.0)
                if abs(yaw) < math.radians(10.0):
                    self.gaze_object(parent, self.infant_target_obj, action=action, yaw_link="head")
                # 発声等での好意的反応
                action[am["voice"]] = 1.0
                action[am["lip"]] = 0.08
            else:
                self.gaze_agent(parent, infant, action=action, yaw_link="head")
                
        elif self.phase == "success":
            # 成功時、幼児に視線を戻して褒める
            self.gaze_agent(parent, infant, action=action, yaw_link="head")
            action[am["voice"]] = 1.0
            action[am["lip"]] = 1.0

        for name, step_size in self.env.parent_joint_mappings.items():
            if name == "voice":
                continue 
            action[am[name]] = action[am[name]]/(step_size*self.env.action_repeat)
            
        clipped = tuple(float(np.clip(v, -1.0, 1.0)) for v in action)
        return clipped

    def check_ija(self, parent_action) -> tuple[float, bool, Dict[str, Any]]:
        infant_reward = 0.0
        ja_done = False
        parent = self.env.parent
        infant = self.env.infant

        # action_repeat を考慮したステップペナルティ
        action_repeat = getattr(self.env, "action_repeat", 1)
        infant_reward += self.step_penalty * action_repeat

        if self.phase == "phase0":
            eye_contact_flag, gaze_eye_ip, gaze_face_ip, gaze_ip = self.is_eye_contact()
            if eye_contact_flag:
                infant_reward += self.eye_contact_reward
                self.eye_contact_steps += 1
            elif gaze_eye_ip:
                infant_reward += self.gaze_ip_reward
                self.eye_contact_steps = 0
            elif gaze_face_ip:
                infant_reward += self.gaze_face_ip_reward
                self.eye_contact_steps = 0
            elif gaze_ip:
                infant_reward += self.gaze_p_reward
                self.eye_contact_steps = 0
            else: 
                self.eye_contact_steps = 0

            angle_to_parent = self.gaze_angle_to_agent(infant, parent)
            gaze_parent_score = self._angle_to_score(angle_to_parent)
            if self.prev_scores["parent_gaze_score"] is not None:
                infant_reward += max(0.0, gaze_parent_score - self.prev_scores["parent_gaze_score"]) * self.eyecontact_phase_reward_weight
            self.prev_scores["parent_gaze_score"] = gaze_parent_score

            if self.eye_contact_steps >= self.eye_contact_min_steps:
                self.phase = "phase1"
                self.steps_in_phase = 0
                self.infant_cue_steps = 0
                infant_reward += self.dyadic_reward
                self.prev_scores["attention_score"] = 0.0

        elif self.phase == "phase1":
            # Initiation phase
            best_oid, best_score = self._update_attention_scores()
            
            # 注意スコアベースの整形報酬
            if self.prev_scores["attention_score"] is not None:
                delta = best_score - self.prev_scores["attention_score"]
                infant_reward += max(0.0, delta) * self.first_phase_reward_weight_for_obj
            self.prev_scores["attention_score"] = best_score
            
            # 親への視線整形報酬
            angle_to_parent = self.gaze_angle_to_agent(infant, parent)
            gaze_parent_score = self._angle_to_score(angle_to_parent)
            if self.prev_scores["parent_gaze_score"] is not None:
                delta_p = gaze_parent_score - self.prev_scores["parent_gaze_score"]
                infant_reward += max(0.0, delta_p) * self.first_phase_reward_weight_for_parent_eye
            self.prev_scores["parent_gaze_score"] = gaze_parent_score

            if best_score > self.attention_threshold:
                if self.infant_target_obj == best_oid:
                    self.infant_cue_steps += 1
                else:
                    self.infant_target_obj = best_oid
                    self.infant_cue_steps = 1
                
                # 絶対的報酬
                infant_reward += self.gaze_obj_ip_reward
            else:
                self.infant_cue_steps = 0

            if self.infant_cue_steps >= self.infant_cue_min_steps:
                if self.infant_target_obj is not None:
                    self.phase = "phase2"
                    self.steps_in_phase = 0
                    self.parent_follow_steps = 0
                    self.prev_scores["attention_score"] = best_score

        elif self.phase == "phase2":
            # Parent Response Phase
            # 幼児は対象物への注意を続ける
            best_oid, infant_target_score = self._update_attention_scores()
            
            if self.prev_scores["attention_score"] is not None:
                delta = infant_target_score - self.prev_scores["attention_score"]
                infant_reward += max(0.0, delta) * self.second_phase_reward_weight_for_obj
            self.prev_scores["attention_score"] = infant_target_score
            
            if infant_target_score > 0.0 and best_oid == self.infant_target_obj:
                infant_reward += self.gaze_obj_ip_reward
            
            if infant_target_score < self.attention_threshold or best_oid != self.infant_target_obj:
                # 注意が切れたらやり直し
                self.phase = "phase1"
                self.steps_in_phase = 0
            else:
                # 親が幼児のターゲットを見ているか判定
                gaze_ok = self.object_gaze_check(parent, self.infant_target_obj, mode="angle")
                if gaze_ok:
                    self.parent_follow_steps += 1
                else:
                    self.parent_follow_steps = 0
                
                if self.parent_follow_steps >= self.parent_follow_min_steps:
                    infant_reward += self.success_bonus
                    self.success_num += 1
                    
                    if self.success_num >= self.max_ja_count:
                        self.phase = "success"
                        ja_done = True
                    else:
                        self.phase = "phase0"
                        self.steps_in_phase = 0

        elif self.phase == "success":
            pass

        self.steps_in_phase += 1
        if self.steps_in_phase >= self.max_phase_steps:
            self.phase = "phase0"
            self.steps_in_phase = 0
            self.eye_contact_steps = 0
            self.success_num = 0

        info_update = {
            "ija_phase": self.phase,
            "ija_target_obj": self.infant_target_obj,
        }
        return float(infant_reward), bool(ja_done), info_update

    def step(self, action):
        if hasattr(self, "_parent_voice_on"):
            self.previous_parent_voice_level = self._parent_voice_on()
        if hasattr(self, "_infant_voice_on"):
            self.previous_infant_voice_level = self._infant_voice_on()
        
        obs, base_reward, done, truncated, info = self.env.step(action)
        info = dict(info or {})
        
        parent_action = self.subgoal_management()
        infant_reward, ja_done, ja_info = self.check_ija(parent_action)

        reward = [float(infant_reward), 0.0]
        done = bool(done or ja_done)
        
        info.update(ja_info)
        info["parent_action"] = parent_action
        
        return obs, reward, done, truncated, info

    def __getattr__(self, name):
        if name == "env":
            return super().__getattribute__(name)
        return getattr(self.env, name)


