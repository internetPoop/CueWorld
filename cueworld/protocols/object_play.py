import numpy as np
from typing import Dict, Any
from collections import deque
from .base import BaseProtocol

class ObjectPlayProtocol(BaseProtocol):
    """オブジェクト操作遊び・アフォーダンスプロトコル"""
    def __init__(self, env: Any, rejection_check: bool = True):
        super().__init__("ObjectPlayProtocol", env, rejection_check=rejection_check)
        
        # プロトコルのメタデータ（属性）
        self.function_type = "declarative"
        self.relation_type = "triadic"
        self.is_urgent = False
        self.response_type = True #幼児が自発的に開始できるプロトコルかどうか
        self.request_type = True #親が自発的に開始できるプロトコルかどうか
        self.solo = False
        self.has_target = True
        
        self.state_proximities = {
            "init": 0.0,
            "fetching": 0.1,
            "approach_target": 0.2,
            "approach": 0.3,
            "playing": 0.6,
            "showing_result": 0.8,
            "completed": 1.0,
            "failure": 0.0
        }
        
        self.state_timer = 0
        self.target_object = None
        self.last_distance = 10.0
        self.level = 1

    def start(self, context: Dict[str, Any]):
        super().start(context)
        self.target_object = context.get("target_object", None)
        if context.get("infant_initiative"):
            self.state = "approach"
        else:
            self.state = "init"
        self.state_timer = 0
        self.last_distance = 10.0

    def _manage_state(self, infant_signals: Dict[str, Any]) -> None:
        #TODO: 受動的なパターン、自発的なパターンを分ける．
        self.state_timer += 1
        
        if self.state == "init":
            if self.target_object is not None:
                if getattr(self.target_object, "is_graspable", False):
                    self.state = "fetching"
                else:
                    self.state = "approach_target"
                self.state_timer = 0
            else:
                self.is_finished = True
                self.state = "failure"

        elif self.state == "fetching":
            if self.env.parent.get("held_obj") == self.target_object.id:
                self.state = "approach"
                self.state_timer = 0
                
        elif self.state == "approach_target": # オブジェクトに近づく
            if self.last_distance < 0.9:
                self.state = "playing"
                self.state_timer = 0
        
        elif self.state == "approach": # 幼児に近づく（今回は遊びメインなので一部変更）
            if self.last_distance < 0.9:
                self.state = "playing"
                self.state_timer = 0

        elif self.state == "playing": 
            if self.state_timer > 150:
                c_id = self.language_manager.get_concept_for_object(self.target_object)
                # contain / open テンプレは廃止。結果提示は praise_with_target に統一。
                self.queue_utterance("praise_with_target", concept_id=c_id)
                    
                self.state = "showing_result"
                self.state_timer = 0

        elif self.state == "showing_result":
            # 結果を提示し、幼児のリアクションを待つ
            if self.state_timer > 100:
                self.is_finished = True
                self.state = "completed"

    def _determine_action(self) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        # 初期姿勢に戻す
        self.env.reset_arm_pose(self.env.parent, action=action)
        
        # --- 物理的アクション ---
        if self.state == "fetching":
            self.env.grasp_object(self.env.parent, self.target_object.id, action=action, approach_dist=0.5)
        elif self.state == "approach_target":
            yaw, dist = self.env.approach_object(self.env.parent, self.target_object.id, action=action, approach_dist=0.8)
            if dist is not None:
                self.last_distance = dist
        elif self.state == "approach":
            yaw, dist = self.env.approach_infant(self.env.parent, self.env.infant, action=action, approach_dist=0.8)
            self.env.grip_object(self.env.parent, action=action)
            if dist is not None:
                self.last_distance = dist
        elif self.state == "playing":
            self.env.grip_object(self.env.parent, action=action)
            is_openable = getattr(self.target_object, "is_openable", False)
            is_containable = getattr(self.target_object, "is_containable", False)
            
            if is_openable:
                # 開閉アクション
                self.env.open_close_object(self.env.parent, self.target_object.id, action=action, approach_dist=0.7)
            elif is_containable:
                # 入れるコンテナを探す
                container_id = None
                for oid in self.env.active_objects:
                    obj = self.env.all_objects.get(oid)
                    # 自分が容器ではない、かつ入れられる容器を探す
                    if obj and getattr(obj, "can_contain", False) and getattr(obj, "open", True) and oid != self.target_object.id:
                        container_id = oid
                        break
                
                if container_id is not None:
                    self.env.put_in_object(self.env.parent, container_id, action=action, approach_dist=0.6)
            else:
                # 固有アクションがなければ対象物の方を向く
                self.env.gaze_object(self.env.parent, self.target_object.id, yaw_link="head", action=action)
                
        elif self.state == "showing_result":
            # 幼児の方を見る
            self.env.gaze_agent(self.env.parent, self.env.infant, yaw_link="head", action=action)

        # --- 発話アクション ---
        if len(self.utterance_queue) > 0:
            phoneme = self.utterance_queue.popleft()
            if phoneme in self.env.phoneme_mapping:
                v_coords = self.env.phoneme_mapping[phoneme]
                if hasattr(self.env, "apply_phoneme_to_action"):
                    self.env.apply_phoneme_to_action(action, v_coords, volume=0.5)
                else:
                    action[self.env.action_map["v_1"]] = v_coords[0]
                    action[self.env.action_map["v_2"]] = v_coords[1]
                    if "v_3" in self.env.action_map and len(v_coords) > 2:
                        action[self.env.action_map["v_3"]] = v_coords[2]
                    action[self.env.action_map["volume"]] = 0.5
        
        return action
    
    def _calculate_persistence(self, infant_signals: Dict[str, Any]):
        self.persistence = 5.0  # 遊び実行中は割り込みを強く抑制する
