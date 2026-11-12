import numpy as np
from typing import Dict, Any
from collections import deque
from .base import BaseProtocol

class PhysicalContactProtocol(BaseProtocol):
    """身体接触・あやし・寝かしつけプロトコル"""
    def __init__(self, env: Any, rejection_check: bool = True):
        super().__init__("PhysicalContactProtocol", env, rejection_check=rejection_check)

        # プロトコルのメタデータ（属性）
        self.function_type = "imperative"
        self.relation_type = "dyadic"
        self.is_urgent = False # 泣き声などの要求時はTrueになる想定
        self.response_type = True #幼児が自発的に開始できるプロトコルかどうか
        self.request_type = True #親が自発的に開始できるプロトコルかどうか
        self.solo = False
        self.has_target = False
        
        self.state_proximities = {
            "init": 0.0,
            "approach": 0.3,
            "contacting": 0.6,
            "comforting": 0.8,
            "completed": 1.0
        }
        
        # 仮の発話セット
        self.utterance_set = {
            "comfort": [("p4", 20), ("p1", 20), ("p4", 20), ("p1", 30)], # よしよし等
        }
        
        self.utterance_queue = deque()
        self.state_timer = 0
        self.last_distance = 10.0
        self.contact_type = "patting" # "holding", "patting", etc.

    def start(self, context: Dict[str, Any]):
        super().start(context)
        self.state = "init"
        self.state_timer = 0
        self.utterance_queue.clear()
        self.last_distance = 10.0
        self.contact_type = context.get("contact_type", "patting")
        self.is_urgent = context.get("is_urgent", self.is_urgent)

    def queue_utterance(self, word_key: str):
        """発話セットから指定された単語をキューに追加する"""
        if word_key in self.utterance_set:
            for phoneme, duration in self.utterance_set[word_key]:
                for _ in range(duration):
                    self.utterance_queue.append(phoneme)

    def _manage_state(self, infant_signals: Dict[str, Any]) -> None:
        self.state_timer += 1
        
        if self.state == "init":
            self.state = "approach"
            self.state_timer = 0

        elif self.state == "approach":
            if self.last_distance < 1.0: # かなり近づく
                self.state = "contacting"
                self.state_timer = 0

        elif self.state == "contacting":
            # 抱っこや撫でる動作の開始（ここでは一定時間で次へ）
            if self.state_timer > 60:
                self.state = "comforting"
                self.state_timer = 0
                self.queue_utterance("comfort")

        elif self.state == "comforting":
            # 幼児の生体パラメータ（不快感・眠気）の回復を待つ
            # 今回はダミーで、一定時間経過または幼児の感情がポジティブになったら完了とする
            infant_emotion = infant_signals.get("emotion", "neutral")
            if infant_emotion == "positive" or self.state_timer > 300:
                self.state = "completed"
                self.is_finished = True

    def _determine_action(self) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        # --- 物理的アクション ---
        if self.state == "approach":
            yaw, dist = self.env.approach_infant(self.env.parent, self.env.infant, action=action, approach_dist=0.5)
            if dist is not None:
                self.last_distance = dist
                
        elif self.state == "contacting" or self.state == "comforting":
            # 幼児の方を見る
            self.env.gaze_agent(self.env.parent, self.env.infant, yaw_link="head", action=action)
            # 腕を使わないので初期姿勢に戻す
            self.env.reset_arm_pose(self.env.parent, action=action)
            # TODO: 抱っこ・撫でるアニメーション（IK制御、Kinematic Constraint等）のダミー実行箇所
            pass

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
        self.persistence = 0.8 # 接触中は強めに持続する
