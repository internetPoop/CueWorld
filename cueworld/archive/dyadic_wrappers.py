from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from cueworld.base_wrapper import WrapperBase

class DyadicWrapper(WrapperBase):
    """
    相互注視と交互発話（ターンテーキング）タスクの報酬ラッパー（フェーズ管理 + 親の行動制御）。

    目的:
      親(Parent)と幼児(Infant)間の社会的相互作用である「見つめ合い」と
      「交互の発話（コール＆レスポンス）」を簡易ステートマシンで段階管理し、
      幼児側の報酬 shaping を行う。加えて、親の行動を自動生成してルールベースで制御する。

    フェーズ概要:
      phase0:
        相互注視を形成（双方が相手の頭方向を十分に見る）。
        一定ステップ連続で閾値を満たすと waiting_phase へ遷移。
      waiting_phase:
        どちらが先に発話するかを待つ。
        - 幼児が先に発話: infant_phase へ
        - 親が先に発話: parent_phase へ
        - タイムアウト後: 親が発話開始
      parent_phase:
        親が発話し、幼児は傾聴（親を見続ける）。
        一定ステップ連続で条件を満たすと infant_phase へ遷移。
      infant_phase:
        幼児が発話し、親は傾聴（幼児を見続ける）。
        一定ステップ連続で条件を満たすと success または parent_phase へ。
      success:
        ターンテーキング達成。成功ボーナス付与。

    報酬設計（幼児のみ）:
      - ステップ毎の微小ペナルティ (step_penalty)
      - 常時報酬: 親への注視と相互注視
      - 視線差分報酬: 親への視線角度改善分
      - シェイピング報酬: 各フェーズでのスコア増加分
      - 不適切発話ペナルティ: parent_phase で幼児が発話した場合
      - success 到達時に success_bonus

    親の行動制御:
      phase0/waiting_phase: 幼児を見る（voice off）
      parent_phase: 幼児を見て発話（voice on）
      infant_phase: 幼児を見る（voice off）
      success: 停止（voice off）

    主な前提:
      - 下位 env は simulation_env.BaseEnv 互換の構造を持つ
      - env.parent / env.infant: 各エージェントメタ情報
      - env._agent_eye_pose / env._agent_torso_pose: 幾何計算ユーティリティ
      - env.parent_voice_level / env.infant_voice_level: 現在の音声強度
    """

    def __init__(self,
                 env: gym.Env,
                 # 報酬の重み付けパラメータ
                 shaping_weight_voice: float = 0.1,
                 gaze_angle_reward_weight: float = 0.05,
                 step_penalty: float = -0.001,
                 success_bonus: float = 1.0,
                 wrong_voice_penalty: float = -0.05,
                 # 絶対的報酬の追加設定
                 eye_contact_reward: float = 0.1,
                 gaze_ip_reward: float = 0.05,
                 gaze_face_ip_reward: float = 0.03,
                 gaze_p_reward: float = 0.01,
                 # 判定の閾値と継続条件
                 voice_thresh: float = 0.5,
                 eye_contact_min_steps: int = 10,
                 parent_call_min_steps: int = 10,
                 infant_response_min_steps: int = 10,
                 waiting_phase_timeout: int = 0,
                 max_phase_steps: int | None = 64,
                 terminate_on_success: bool = True,
                 done_episode_num: int = 1,
                 # 親の行動制御パラメータ
                 approach_distance: float = 1.0,
                 approach_thresh: float = math.radians(15.0),
                 target_angle: Optional[float] = math.radians(30.0), #これを設定する場合は，approach_distanceの大きさを考慮すること
                 infant_voice :bool = False,
                 eval_mode: bool = True,
                 *args, **kwargs):
        """パラメータ説明:
        shaping_weight_gaze: 視線シェイピング報酬の重み
        shaping_weight_voice: 発話シェイピング報酬の重み
        constant_gaze_reward: 親への注視に対する常時報酬
        constant_eye_contact_reward: 相互注視に対する常時報酬
        gaze_angle_reward_weight: 視線角度差分報酬の重み
        step_penalty: 毎ステップの微小負報酬
        success_bonus: タスク成功時のボーナス
        wrong_voice_penalty: 不適切なタイミングでの発話ペナルティ
        eye_contact_gate: 相互注視とみなすスコアの閾値
        gaze_gate: 親への注視とみなすスコアの閾値
        voice_thresh: 音声が「ON」であるとみなす閾値
        eye_contact_min_steps: phase0 クリアに必要な連続ステップ数
        parent_call_min_steps: parent_phase クリアに必要な連続ステップ数
        infant_response_min_steps: infant_phase クリアに必要な連続ステップ数
        waiting_phase_timeout: waiting_phase で親が発話開始するまでの時間
        max_phase_steps: 各フェーズで進展がない場合のタイムアウト
        terminate_on_success: True なら success 到達で done=True
        done_episode_num: 成功回数がこれに達したら終了
        approach_distance: 親が幼児に近づく目標距離
        approach_thresh: 親の方向転換の角度閾値
        """
        super().__init__(env)
        
        # 報酬パラメータ
        self.w_voice = float(shaping_weight_voice)
        self.gaze_angle_reward_weight = float(gaze_angle_reward_weight)
        self.step_penalty = float(step_penalty)
        self.success_bonus = float(success_bonus)
        self.wrong_voice_penalty = float(wrong_voice_penalty)
        
        # 絶対的報酬の設定
        self.eye_contact_reward = float(eye_contact_reward)
        self.gaze_ip_reward = float(gaze_ip_reward)
        self.gaze_face_ip_reward = float(gaze_face_ip_reward)
        self.gaze_p_reward = float(gaze_p_reward)
        
        # 判定閾値
        self.voice_thresh = float(voice_thresh)
        
        # フェーズ継続条件
        self.eye_contact_min_steps = int(eye_contact_min_steps)
        self.parent_call_min_steps = int(parent_call_min_steps)
        self.infant_response_min_steps = int(infant_response_min_steps)
        self.waiting_phase_timeout = int(waiting_phase_timeout)
        self.max_phase_steps = int(max_phase_steps) if max_phase_steps is not None else None
        
        self.terminate_on_success = bool(terminate_on_success)
        self.done_episode_num = int(done_episode_num)
        
        # 親の行動制御パラメータ
        self.approach_distance = float(approach_distance) if approach_distance is not None else None
        self.approach_thresh = float(approach_thresh) if approach_thresh is not None else None
        self.target_angle = None if target_angle is None else float(target_angle) #幼児に対しての目標角度
        
        self.get_location = False if approach_distance is not None else True
        self.good_location = None
        self.gaze_phase = False
        
        # フェーズ管理
        self.phase: str = "phase0"
        self.eye_contact_steps: int = 0
        self.steps_in_phase: int = 0
        self.active_steps: int = 0  # 追加: parent_phase/infant_phase用のカウンタ
        
        # 差分報酬用の直前スコア
        self.previous_score = None
        
        # 成功判定用
        self.previous_phase: Optional[str] = None
        self.success_num: int = 0
        
        #幼児の声を使うかどうか
        self.infant_voice = infant_voice
        
        self.eval_mode = eval_mode
        if eval_mode:
            self.eval_dict = {
                "gaze_parent": False,
                "gaze_face": False,
                "gaze_eye": False,
                "eyecontact": False,
                "keep_eyecontact": False,
                "listen_parent": False,
                "respond_infant": False,
                "success": False,
            }
        
        # debug_reward
        self.delta_reward = 0.0
        self.pulse_reward = 0.0

    def reset(self, **kwargs):
        """下位環境をリセットし、フェーズ/カウンタを初期化して返す。"""
        obs, info = self.env.reset(**kwargs)
        self.phase = "phase0"
        self.eye_contact_steps = 0
        self.steps_in_phase = 0
        self.active_steps = 0  # 追加: リセット時に初期化
        self.previous_score = 0.0
        self.previous_phase = None
        self.success_num = 0
        self.get_location = False if self.approach_distance is not None else True
        self.good_location = None
        self.gaze_phase = False #親が幼児を見るフェーズかどうか(親が適切な位置への移動を行っている間はFalseにする)
        
        if self.eval_mode:#TODO: 幼児からの自発的な関わりには対応していない
            self.eval_dict = {
                "gaze_parent": False,
                "gaze_face": False,
                "gaze_eye": False,
                "eyecontact": False,
                "keep_eyecontact": False,
                "listen_parent": False,
                "respond_infant": False,
                "success": False,
            }
        
        # debug_reward
        self.delta_reward = 0.0
        self.pulse_reward = 0.0
        
        # 情報追加
        info = dict(info or {})
        info["tt_phase"] = self.phase
        info["parent_action"] = self.subgoal_management()
        return obs, info

    def subgoal_management(self):
        """
        親の行動を決定するためのサブゴール管理ロジック。
        """
        parent = self.env.parent
        infant = self.env.infant
        action = len(self.env.action_map) * [0.0]
        am = self.env.action_map
        if self.target_angle is not None and not self.get_location:
            self.good_location = self.spot_for_dyadic(parent, infant,approach_dist=self.approach_distance, angle=self.target_angle)
            self.get_location = True
        else:
            if self.good_location is None:
                # 基本動作: 幼児に近づき、幼児の方を向く
                self.gaze_phase = True
                if self.approach_distance is None:
                    yaw, _ = self.gaze_agent(parent, infant, action=action, yaw_link="torso")
                else:
                    yaw, distance = self.approach_infant(parent, infant, action=action, 
                                                        thresh=self.approach_thresh, 
                                                        approach_dist=self.approach_distance)
            else:
                if self.gaze_phase:
                    # 幼児を見るフェーズ
                    yaw, _ = self.gaze_agent(parent, infant, action=action, yaw_link="torso")
                else:
                    # 位置へ近づくフェーズ
                    yaw, distance = self.approach_location(parent, self.good_location, action=action)
                    if distance is not None and distance <= 0.05:
                        self.gaze_phase = True
                
                
                
        # フェーズごとの発話制御
        if self.phase == "phase0":
            if self.gaze_phase:
                # 位置調整中: 声なし
                action[am["voice"]] = 1.0
                action[am["lip"]] = - self.env.get_joint_angle(parent, "left_lip_joint")
            else:
                # 幼児に注視開始
                action[am["voice"]] = 1.0
                action[am["lip"]] = 0.08
                
        if self.phase != "phase0":
            action[am["voice"]] = 1.0
            action[am["lip"]] = 0.08
        # 変更前
        # if self.phase == "phase0":
        #     # 相互注視待ち: 声なし
        #     action[am["voice"]] = 0.0
            
        # elif self.phase == "waiting_phase":
        #     # 発話者決定待機: 一定時間後に親が発話開始
        #     if self.steps_in_phase >= self.waiting_phase_timeout:
        #         action[am["voice"]] = 1.0
        #     else:
        #         action[am["voice"]] = 0.0
            
                
        # elif self.phase == "parent_phase":
        #     # 親の発話: 声あり
        #     action[am["voice"]] = 1.0
            
        # elif self.phase == "infant_phase":
        #     # 幼児の応答待ち: 声なし
        #     action[am["voice"]] = 0.0
            
        # elif self.phase == "success":
        #     action[am["voice"]] = 0.0
        
        # アクションを正規化してクリップ
        for name, step_size in self.env.parent_joint_mappings.items():
            if name == "voice":
                continue
            action[am[name]] = action[am[name]]/(step_size*self.env.action_repeat)
        clipped = tuple(float(np.clip(v, -1.0, 1.0)) for v in action)
        
        return clipped

    def check_dyadic(self, parent_action) -> tuple[float, bool, Dict[str, Any]]:
        """フェーズ遷移と幼児報酬計算。
        戻り値:
          (幼児報酬, successで早期終了フラグ, 情報辞書)
        """
        infant_reward = 0.0
        done = False
        
        parent = self.env.parent
        infant = self.env.infant
        
        # 視線角度と相互注視スコア
        angle_to_parent = self.gaze_angle_to_agent(infant, parent)
        gaze_parent_score = self._angle_to_score(angle_to_parent)
        
        eye_contact_flag, gaze_eye_ip, gaze_face_ip, gaze_ip = self.is_eye_contact()
        
        # 音声レベルの取得
        p_voice = self._parent_voice_on()
        if self.infant_voice:
            i_voice = self._infant_voice_on()
        else:
            i_voice = 0.0
        
        # 常時報酬
        if eye_contact_flag:
            #アイコンタクト成立
            infant_reward += self.eye_contact_reward
            self.pulse_reward = self.eye_contact_reward
            self.eye_contact_steps += 1
            if self.eval_mode:
                self.eval_dict["eyecontact"] = True
                self.eval_dict["gaze_eye"] = True
                self.eval_dict["gaze_parent"] = True
                self.eval_dict["gaze_face"] = True
        elif gaze_eye_ip:
            #目を見ることには成功
            infant_reward += self.gaze_ip_reward
            self.pulse_reward = self.gaze_ip_reward
            self.eye_contact_steps = 0
            if self.eval_mode:
                self.eval_dict["gaze_eye"] = True
                self.eval_dict["eyecontact"] = False
                self.eval_dict["gaze_parent"] = True
                self.eval_dict["gaze_face"] = True
        elif gaze_face_ip:
            #顔を見ることには成功
            infant_reward += self.gaze_face_ip_reward
            self.pulse_reward = self.gaze_face_ip_reward
            self.eye_contact_steps = 0
            if self.eval_mode:
                self.eval_dict["gaze_face"] = True
                self.eval_dict["eyecontact"] = False
                self.eval_dict["gaze_eye"] = False
                self.eval_dict["gaze_parent"] = True
        elif gaze_ip:
            #親を見ることには成功
            infant_reward += self.gaze_p_reward
            self.pulse_reward = self.gaze_p_reward
            self.eye_contact_steps = 0
            if self.eval_mode:
                self.eval_dict["gaze_parent"] = True
                self.eval_dict["gaze_face"] = False
                self.eval_dict["gaze_eye"] = False
                self.eval_dict["eyecontact"] = False
        else: 
            self.eye_contact_steps = 0
            if self.eval_mode:
                self.eval_dict["gaze_parent"] = False
                self.eval_dict["gaze_face"] = False
                self.eval_dict["gaze_eye"] = False
                self.eval_dict["eyecontact"] = False
        
        # 視線角度差分報酬
        if self.previous_score is None:
            self.previous_score = gaze_parent_score
        else:
            delta = gaze_parent_score - self.previous_score
            self.delta_reward = self.gaze_angle_reward_weight * delta
            infant_reward += self.gaze_angle_reward_weight * delta
            self.previous_score = gaze_parent_score
        
        # ステップペナルティ
        infant_reward += self.step_penalty
        
        # waiting_phase以降で目線合わせが崩れたらphase0に戻る
        if not eye_contact_flag or self.gaze_phase == False:
            self.phase = "phase0"
            self.eye_contact_steps = 0
            self.steps_in_phase = 0
        
        if self.eval_mode:
            if self.previous_phase == "parent_phase" and self.phase == "infant_phase":
                self.eval_dict["listen_parent"] = True
            if self.previous_phase == "infant_phase" and self.phase == "parent_phase":
                self.eval_dict["respond_infant"] = True
        
        # フェーズ別の処理
        if self.phase == "phase0":
            if self.eye_contact_steps >= self.eye_contact_min_steps:
                self.phase = "waiting_phase"
                self.steps_in_phase = 0
        
        elif self.phase == "waiting_phase":
            if self.eval_mode:
                self.eval_dict["keep_eyecontact"] = True
            # 発話者決定待機
            p_voice_active = (abs(p_voice) >= self.voice_thresh)
            i_voice_active = (abs(i_voice) >= self.voice_thresh)
            if p_voice_active and i_voice_active:
                # 両方同時発話: ノーカウント
                pass
            elif p_voice_active and not i_voice_active:
                # 親が先に発話
                self.phase = "parent_phase"
                self.steps_in_phase = 0
                self.active_steps = 0
            elif i_voice_active and not p_voice_active:
                # 幼児が先に発話
                self.phase = "infant_phase"
                self.steps_in_phase = 0
                self.active_steps = 0
        
        elif self.phase == "parent_phase":
            if self.eval_mode:
                self.eval_dict["keep_eyecontact"] = True
            # 親の発話
            i_voice_active = (abs(i_voice) >= self.voice_thresh)
            if i_voice_active:
                # 不適切な発話にペナルティ
                infant_reward += self.wrong_voice_penalty
                self.active_steps = 0
                
            else:
                p_voice_active = (abs(p_voice) >= self.voice_thresh)
                
                if p_voice_active:
                    self.active_steps += 1
                else:
                    self.active_steps = 0
                
                if self.active_steps >= self.parent_call_min_steps:
                    if not self.infant_voice or self.previous_phase == "infant_phase":
                        infant_reward += self.success_bonus
                        self.success_num += 1
                        self.eval_dict["success"] = True
                    self.previous_phase = "parent_phase"
                    self.phase = "infant_phase"
                    self.eye_contact_steps = 0
                    self.steps_in_phase = 0
                    self.active_steps = 0
            
            # タイムアウト処理
            if self.max_phase_steps is not None and self.steps_in_phase >= self.max_phase_steps:
                self.previous_phase = None
                self.phase = "waiting_phase"
                self.active_steps = 0
                self.steps_in_phase = 0
        
        elif self.phase == "infant_phase":
            if self.eval_mode:
                self.eval_dict["keep_eyecontact"] = True
            # 幼児の応答
            if self.max_phase_steps is not None and self.steps_in_phase >= self.max_phase_steps:
                #タイムアウト
                self.previous_phase = None
                self.phase = "waiting_phase"
                self.active_steps = 0
                self.steps_in_phase = 0
                infant_reward += -0.1
            else:
                # 発話スコア
                voice_score = min(1.0, abs(i_voice) / self.voice_thresh) if self.voice_thresh > 0 else 0
                infant_reward += voice_score * self.w_voice
                
                i_voice_active = (abs(i_voice) >= self.voice_thresh)
                
                if i_voice_active:
                    self.active_steps += 1
                else:
                    self.active_steps = 0
                
                if self.active_steps >= self.infant_response_min_steps:
                    if self.previous_phase == "parent_phase":
                        infant_reward += self.success_bonus
                        self.success_num += 1
                        if self.eval_mode:
                            #これは成功したときだけでいい．回数数えたいから
                            self.eval_dict["success"] = True
                    self.phase = "parent_phase"
                    self.previous_phase = "infant_phase"
                    self.active_steps = 0
                    self.steps_in_phase = 0
        
        elif self.phase == "success":
            pass
        
        # フェーズ総ステップ数を更新
        self.steps_in_phase += 1
        # 終了判定
        #TODO successされたらdoneとそうでないのみたいに分ける
        if self.success_num >= self.done_episode_num:
            if self.terminate_on_success:
                done = True
        
        info_update = {
            "tt_phase": self.phase,
            "phase_steps": self.eye_contact_steps,
            "infant_reward": infant_reward,
            "gaze_score": gaze_parent_score,
            "delta_reward": self.delta_reward,
            "pulse_reward": self.pulse_reward,
        }
        return float(infant_reward), bool(done), info_update

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)
        if self.eval_mode:
            self.eval_dict = {
                "gaze_parent": False,
                "gaze_face": False,
                "gaze_eye": False,
                "eyecontact": False,
                "keep_eyecontact": False,
                "listen_parent": False,
                "respond_infant": False,
                "success": False,
            }
        info = dict(info or {})
        
        self.delta_reward = 0.0
        self.pulse_reward = 0.0
        
        # ターンテーキングの評価
        infant_reward, tt_done, tt_info = self.check_dyadic(action[1])
        parent_action = self.subgoal_management()
        
        # 親の報酬は0固定
        reward = [float(infant_reward), 0.0]
        done = bool(done or tt_done)
        
        info.update(tt_info)
        info["parent_action"] = parent_action
        info["eval_dict"] = self.eval_dict if self.eval_mode else None
        return obs, reward, done, truncated, info

    def __getattr__(self, name):
        """env へ属性を委譲するフォールバック。"""
        if name == "env":
            return super().__getattribute__(name)
        return getattr(self.env, name)
