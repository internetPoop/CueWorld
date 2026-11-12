from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Any, List
from enum import IntEnum


import numpy as np
import gymnasium as gym
from cueworld.base_wrapper import WrapperBase
import pybullet as p  # 環境内部で使用しているため依存
class NavigationWrapper(WrapperBase):
    """
    幼児のナビゲーションタスク用報酬ラッパー。
    init時に指定されたターゲット（親または特定のオブジェクト、またはNone）への接近を学習させる。
    target=Noneの場合は、全オブジェクトの中から最も視線スコアが高いものを動的に選択する。
    """
    def __init__(self, env: gym.Env, 
                 target=None, # 文字(parent)、objectの特徴タプル(形状，色,サイズ)、またはNone
                 success_dist: float = 0.6,   # 成功とみなす距離(m)
                 shaping_weight: float = 8.0, # 距離差分報酬の重み
                 gaze_shaping_weight: float = 2.0, # 追加: 視線差分報酬の重み
                 gaze_reward: float = 0.05,  # 追加: 視線維持報酬の重み
                 first_gaze_reward: float = 0.1,  # 追加: 最初に視線を向けたときの報酬
                 step_penalty: float = -0.01,
                 success_bonus: float = 10.0,
                 terminate_on_success: bool = True,
                 gaze_min_steps: int = 8,  # 追加: 視線を向け続ける必要がある最小ステップ数
                 eval_mode: bool = False,
                 *args, **kwargs):
        super().__init__(env)
        if target is not None and target != "parent":
            target = tuple(target)
        self.target = target #target=Noneの場合は全オブジェクト対象
        self.success_dist = float(success_dist)
        self.shaping_weight = float(shaping_weight)
        self.gaze_shaping_weight = float(gaze_shaping_weight)
        self.gaze_reward = float(gaze_reward)
        self.first_gaze_reward = float(first_gaze_reward)  # 追加: 最初に視線を向けたときの報酬
        self.step_penalty = float(step_penalty)
        self.success_bonus = float(success_bonus)
        self.terminate_on_success = bool(terminate_on_success)
        self.target_body_id = None
        self.previous_obj_id = None  # 追加: 前回の最良オブジェクトID
        
        self.prev_dist = None
        self.previous_score = None
        self.first_gaze = False  # 追加: 最初に視線を向けたかどうかのフラグ
        
        # 追加: 視線継続判定用のパラメータ
        self.gaze_min_steps = int(gaze_min_steps)
        self.gaze_steps = 0  # 視線を向け続けているステップ数

        # infant_action_namesに"forward"が含まれているか確認
        self.infant_can_move = (
            self.env.infant_action_names is None or 
            "forward" in self.env.infant_action_names
        )
        
        self.eval_mode = eval_mode
        if eval_mode:
            self.eval_dict = {
                "gaze_object": False,
                "success": False,
            }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if self.target is None:
            # 全オブジェクトが対象
            self.target_body_id = None
        elif self.target == "parent":
            self.target_body_id = self.env.parent_body
        else:
            obj_idx = self.env.obj_feat_to_id[self.target]
            self.target_body_id = self.env.active_objects[obj_idx]
            
        if self.infant_can_move:
            if self.target_body_id is not None:
                self.prev_dist = self._get_distance()
            else:
                # target=Noneの場合は最も近いオブジェクトの距離を初期値とする
                self.prev_dist = self._get_best_object_distance()
                
        if self.target_body_id is not None:
            self.previous_score = self._get_gaze_angle()
        else:
            # target=Noneの場合は最も視線スコアが高いオブジェクトのスコアを初期値とする
            self.previous_score = self._get_best_object_score()
        
        # 追加: 視線継続カウンタのリセット
        self.previous_obj_id = None
        self.gaze_steps = 0
        self.first_gaze = False
        
        # info更新
        info = dict(info or {})
        info["parent_action"] = [0.0]*7  # 親の行動は無関係なのでゼロ固定
        if self.eval_mode:
            self.eval_dict = {
                "gaze_object": False,
                "success": False,
            }
        
        return obs, info

    def _get_distance(self) -> float:
        if self.target_body_id is None:
            return 0.0
            
        # 幼児の位置
        infant_pos, _ = p.getBasePositionAndOrientation(self.env.infant_body, physicsClientId=self.client_id)
        infant_pos = np.array(infant_pos[:2], dtype=np.float32)
        
        # ターゲットの位置
        target_pos, _ = p.getBasePositionAndOrientation(self.target_body_id, physicsClientId=self.client_id)
        target_pos = np.array(target_pos[:2], dtype=np.float32)
        
        # 2次元ユークリッド距離
        return float(np.linalg.norm(infant_pos - target_pos))

    def _get_gaze_angle(self) -> float:
        if self.target == "parent":
            return self.gaze_angle_to_agent(self.env.infant, self.env.parent)
        else:
            return self.gaze_angle_to_object(self.env.infant, self.target_body_id)
    
    def _get_best_object_distance(self) -> float:
        """全オブジェクトの中から最も近いものの距離を返す"""
        if not hasattr(self.env, 'objects') or len(self.env.active_objects) == 0:
            return 0.0
            
        infant_pos, _ = p.getBasePositionAndOrientation(self.env.infant_body, physicsClientId=self.client_id)
        infant_pos = np.array(infant_pos[:2], dtype=np.float32)
        
        min_dist = float('inf')
        for obj_id in self.env.active_objects:
            obj_pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client_id)
            obj_pos = np.array(obj_pos[:2], dtype=np.float32)
            dist = float(np.linalg.norm(infant_pos - obj_pos))
            min_dist = min(min_dist, dist)
            
        return min_dist if min_dist != float('inf') else 0.0
    
    def _get_best_object_score(self) -> float:
        """全オブジェクトの中から最も視線スコアが高いものを返す"""
        best_score = 0.0
        for obj_id in self.env.active_objects:
            angle = self.gaze_angle_to_object(self.env.infant, obj_id)
            score = self._angle_to_score(angle)
            best_score = max(best_score, score)
            
        return best_score
    
    def _get_best_object_info(self) -> tuple[int | None, float, float, bool]:
        """全オブジェクトの中から最も視線スコアが高いものの情報を返す
        Returns: (obj_id, distance, angle_score, gaze_ok)
        """ 
        infant_pos, _ = p.getBasePositionAndOrientation(self.env.infant_body, physicsClientId=self.client_id)
        infant_pos = np.array(infant_pos[:2], dtype=np.float32)
        
        best_obj_id = None
        best_score = -1.0
        best_distance = 0.0
        best_gaze = False
        
        for obj_id in self.env.active_objects:
            angle = self.gaze_angle_to_object(self.env.infant, obj_id)
            score = self._angle_to_score(angle)
            
            if score > best_score:
                best_score = score
                best_obj_id = obj_id
                best_gaze = self.object_gaze_check(self.env.infant, obj_id)
                
                obj_pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client_id)
                obj_pos = np.array(obj_pos[:2], dtype=np.float32)
                best_distance = float(np.linalg.norm(infant_pos - obj_pos))
        
        return best_obj_id, best_distance, best_score, best_gaze

    def step(self, action):
        obs, base_reward, done, truncated, info = self.env.step(action)
        info = dict(info or {})
        
        
        # 報酬計算
        reward = 0.0
        # 1. ステップペナルティ (時間経過で減点)
        reward += self.step_penalty
        
        # target指定の有無で処理を分岐
        if self.target_body_id is not None:
            # 特定ターゲットが指定されている場合
            if self.infant_can_move:
                current_dist = self._get_distance()
                # 2. 距離シェイピング (近づいたらプラス)
                if self.prev_dist is not None:
                    delta_dist = self.prev_dist - current_dist
                    reward += self.shaping_weight * delta_dist
                self.prev_dist = current_dist
                    
            current_angle = self.gaze_angle_to_object(self.env.infant, self.target_body_id)
            angle_score = self._angle_to_score(current_angle)
            gaze = self.object_gaze_check(self.env.infant, self.target_body_id)

            # 3. 視線シェイピング (向いたらプラス)
            if self.previous_score is not None:
                delta = angle_score - self.previous_score
                reward += self.gaze_shaping_weight * delta
            else:
                delta = 0.0
            
            self.previous_score = angle_score
            
            # 追加: 視線継続カウンタの更新
            if gaze:
                self.gaze_steps += 1
            else:
                self.gaze_steps = 0
            
            # 4. 成功判定（視線を一定ステップ連続で向け続ける必要がある）
            is_success = False
            if not self.infant_can_move:
                # 移動不可で視線角度のみの判定
                if self.gaze_steps >= self.gaze_min_steps:
                    is_success = True
                    reward += self.success_bonus
                    if self.terminate_on_success:
                        done = True
                    
            elif current_dist < self.success_dist and self.gaze_steps >= self.gaze_min_steps:
                # 移動可能な場合は距離と視線の両方を満たす必要がある
                is_success = True
                reward += self.success_bonus
                if self.terminate_on_success:
                    done = True
            
        else:
            # target=Noneの場合：最も視線スコアが高いオブジェクトを選択
            best_obj_id, current_dist, angle_score, gaze = self._get_best_object_info()
            
            if self.infant_can_move and best_obj_id is not None:
                # 2. 距離シェイピング（最も近いオブジェクトとの距離差分）
                if self.prev_dist is not None:
                    delta_dist = self.prev_dist - current_dist
                    reward += self.shaping_weight * delta_dist
                self.prev_dist = current_dist
            
            
            # 3. 視線シェイピング（最も視線スコアが高いオブジェクト）
            if self.previous_score is not None:
                delta = angle_score - self.previous_score
                reward += self.gaze_shaping_weight * delta
            self.previous_score = angle_score
            
            # 追加: 視線継続カウンタの更新
            if gaze and best_obj_id is not None:
                if best_obj_id == self.previous_obj_id:
                    self.gaze_steps += 1
                    reward += self.gaze_reward
                self.previous_obj_id = best_obj_id
                if not self.first_gaze:
                    self.first_gaze = True
                    reward += self.first_gaze_reward  # 最初に視線を向けたときにも報酬を与える
            else:
                self.previous_obj_id = None
                self.gaze_steps = 0
            
            # 4. 成功判定（視線を一定ステップ連続で向け続ける必要がある）
            is_success = False
            if best_obj_id is not None:
                if not self.infant_can_move:
                    # 移動不可で視線角度のみの判定
                    if self.gaze_steps >= self.gaze_min_steps:
                        is_success = True
                        reward += self.success_bonus
                        if self.terminate_on_success:
                            done = True
                elif current_dist < self.success_dist and self.gaze_steps >= self.gaze_min_steps:
                    # 移動可能な場合は距離と視線の両方を満たす必要がある
                    is_success = True
                    reward += self.success_bonus
                    if self.terminate_on_success:
                        done = True
            
        if self.eval_mode:
            self.eval_dict = {
                "gaze_object": gaze,
                "success": is_success,
            }
        info["eval_dict"] = self.eval_dict if self.eval_mode else {}
        info["parent_action"] = [0.0]*len(self.env.action_map)  # 親の行動は無関係なのでゼロ固定
        
        # 幼児の報酬を上書き (親は0)
        new_reward = [float(reward), 0.0]
        return obs, new_reward, done, truncated, info
        
    def __getattr__(self, name):
        if name == "env":
            return super().__getattribute__(name)
        return getattr(self.env, name)


