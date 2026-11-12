from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import gymnasium as gym
from cueworld.base_wrapper import WrapperBase
import pybullet as p

class HandOverWrapper(WrapperBase):
    """
    ハンドオーバータスクのラッパー。
    幼児が対象物体に対してリーチング動作を行い、親がそのオブジェクトを取ってきて幼児に渡す流れを定義する。

    フェーズ概要:
      phase0 (Infant Reaching):
        幼児が対象物体を見つめ、リーチングしている状態を観察する。
        対象が確定したら phase1 へ。
      phase1 (Parent Fetching):
        親が対象物体へ近づき、把持する。
      phase2 (Parent Handing Over):
        親が幼児へ近づき、対象物体を差し出す。
        幼児が対象物体を把持したら success へ。
      success:
        共同作業(ハンドオーバー)成功。
    """

    def __init__(self,
                 env: gym.Env,
                 reaching_min_steps: int = 10,
                 max_phase_steps: int = 400,
                 approach_distance_for_grasp: float = 0.5,
                 approach_distance_for_handover: float = 0.6,
                 *args, **kwargs):
        super().__init__(env)
        
        self.reaching_min_steps = reaching_min_steps
        self.max_phase_steps = max_phase_steps
        self.approach_distance_for_grasp = approach_distance_for_grasp
        self.approach_distance_for_handover = approach_distance_for_handover
        
        # フェーズ管理
        self.phase: str = "phase0"
        self.parent_target_obj: Optional[int] = None
        self.steps_in_phase: int = 0
        self.infant_reaching_steps: int = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.phase = "phase0"
        self.parent_target_obj = None
        self.steps_in_phase = 0
        self.infant_reaching_steps = 0
        self.hand_over_phase = False
        
        info = dict(info or {})
        info["parent_action"] = self.subgoal_management()
        return obs, info

    def subgoal_management(self):
        """親の行動を決定する"""
        parent = self.env.parent
        infant = self.env.infant
        action = len(self.env.action_map) * [0.0]
        am = self.env.action_map

        # フェーズごとの親の行動
        if self.phase == "phase0":
            # 幼児を観察する (幼児を見る)
            self.gaze_agent(parent, infant, action=action, yaw_link="head")
        
        elif self.phase == "phase1":
            if self.parent_target_obj is not None:
                # ターゲットをつかむ
                if not self.agent_holding_object_check(parent, self.parent_target_obj):
                    action = self.grasp_object(parent, self.parent_target_obj, action=action, approach_dist=self.approach_distance_for_grasp)
        
        elif self.phase == "phase2":
            if self.parent_target_obj is not None:
                # 幼児へ渡すために近づき手を伸ばす
                self.hand_over(parent,action=action)
                
        elif self.phase == "success":
            # 完了後は幼児を見守るなど
            self.gaze_agent(parent, infant, action=action, yaw_link="head")
            
        # アクション制限と正規化
        for name, step_size in self.env.parent_joint_mappings.items():
            if name == "voice":
                continue
            action[am[name]] = action[am[name]] / (step_size * self.env.action_repeat)
            
        clipped = tuple(float(np.clip(v, -1.0, 1.0)) for v in action)
        return clipped

    def check_hand_over(self, parent_action) -> tuple[float, bool, Dict[str, Any]]:
        """フェーズ遷移と報酬計算（報酬は現在は0.0とする）。"""
        infant_reward = 0.0
        ho_done = False
        
        parent = self.env.parent
        infant = self.env.infant

        if self.phase == "phase0":
            # 幼児の欲しいものを特定
            reaching_obj = self._detect_infant_reaching(infant)
            
            if reaching_obj is not None:
                if self.parent_target_obj == reaching_obj:
                    self.infant_reaching_steps += 1
                else:
                    self.parent_target_obj = reaching_obj
                    self.infant_reaching_steps = 1
            else:
                self.infant_reaching_steps = 0

            # 幼児が一定期間同じオブジェクトにリーチングしたら phase1 へ
            if self.infant_reaching_steps >= self.reaching_min_steps:
                self.phase = "phase1"
                self.steps_in_phase = 0
                
        elif self.phase == "phase1":
            # 親が対象オブジェクトを把持したか確認
            if self.agent_holding_object_check(parent, self.parent_target_obj):
                self.phase = "phase2"
                self.steps_in_phase = 0

        elif self.phase == "phase2":
            # 幼児が対象オブジェクトを受け取ったか確認
            if self.agent_holding_object_check(infant, self.parent_target_obj):
                self.phase = "success"
                self.steps_in_phase = 0
                ho_done = True

        elif self.phase == "success":
            pass
            
        self.steps_in_phase += 1
        if self.steps_in_phase >= self.max_phase_steps:
            # タイムアウト
            self.phase = "phase0"
            self.steps_in_phase = 0
            self.infant_reaching_steps = 0
            self.parent_target_obj = None

        info_update = {
            "hand_over_phase": self.phase,
            "parent_target_obj": self.parent_target_obj,
        }
        return float(infant_reward), bool(ho_done), info_update

    def _detect_infant_reaching(self, agent) -> Optional[int]:
        """幼児が手を伸ばしているオブジェクトを判定する。"""
        gaze_obj = self.what_is_looking_at(infant)
        if gaze_obj is not None:
            roll_angle = self.get_palm_roll_angle_to_object(infant, gaze_obj) 
            if abs(roll_angle) < math.radians(15.0):
                return gaze_obj
        return None

    def step(self, action):
        """
        環境を 1 step 進める。
        action は幼児の行動と親の行動を含むタプルだが、親の行動は subgoal_management() によって上書きされる。
        """
        obs, base_reward, env_done, truncated, info = self.env.step(action)
        info = dict(info or {})
        # フェーズ遷移チェックと報酬（ベースはここでは幼児のみ）
        # action[1]相当は親のものだが自動計算されるため使わないことが多い
        infant_reward, ho_done, ho_info = self.check_hand_over(action[1])
        info.update(ho_info)
        info["parent_action"] = self.subgoal_management()

        # 親の報酬は現状0.0とする
        reward = [float(infant_reward), 0.0]
        done = bool(env_done or ho_done)
        
        return obs, reward, done, truncated, info
