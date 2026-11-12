import numpy as np
from typing import Dict, Any
from collections import deque
from .base import BaseProtocol

class DyadicPlayProtocol(BaseProtocol):
    """社会的遊びプロトコル（いないいないばぁ等）"""
    def __init__(self, env: Any, rejection_check: bool = True):
        super().__init__("DyadicPlayProtocol", env, rejection_check=rejection_check)
        # プロトコルのメタデータ（属性）
        self.function_type = "declarative" # または命令的の側面もある
        self.relation_type = "dyadic"
        self.is_urgent = False
        self.response_type = True #幼児が自発的に開始できるプロトコルかどうか
        self.request_type = True #親が自発的に開始できるプロトコルかどうか
        self.solo = False
        self.has_target = False
        
        self.state_proximities = {
            "init": 0.0,
            "preparation": 0.2,
            "waiting": 0.5,
            "climax": 0.8,
            "evaluating": 0.9,
            "completed": 1.0
        }
        
        self.state_timer = 0
        self.play_type = "peekaboo"

    def start(self, context: Dict[str, Any]):
        super().start(context)
        self.state = "init"
        self.state_timer = 0
        self.play_type = context.get("play_type", "peekaboo")

    def _manage_state(self, infant_signals: Dict[str, Any]) -> None:
        self.state_timer += 1
        
        if self.state == "init":
            self.state = "preparation"
            self.state_timer = 0

        elif self.state == "preparation":
            # 親の準備アクション（顔を隠す等）
            if self.state_timer > 60:
                self.state = "waiting"
                self.state_timer = 0

        elif self.state == "waiting":
            # 待機状態（タメ）
            if self.state_timer > 40:
                self.state = "climax"
                self.state_timer = 0

        elif self.state == "climax":
            # クライマックス（ばぁ！等）
            if self.state_timer > 60:
                self.state = "evaluating"
                self.state_timer = 0

        elif self.state == "evaluating":
            # 幼児の情動的反応を待機
            infant_emotion = infant_signals.get("emotion", "neutral")
            
            # ポジティブなら成功、または一定時間経過で終了
            if infant_emotion == "positive":
                self.queue_utterance("praise")
                self.state = "completed"
                self.is_finished = True
            elif self.state_timer > 100:
                self.state = "completed"
                self.is_finished = True

    def _determine_action(self) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        # 腕を使わないので初期姿勢に戻す
        self.env.reset_arm_pose(self.env.parent, action=action)
        # --- 物理的アクション ---
        # 基本的に幼児の方を見続ける
        self.env.gaze_agent(self.env.parent, self.env.infant, yaw_link="head", action=action)
        
        if self.state == "preparation":
            # TODO: 顔を隠すアニメーション（IK制御等）のダミー実行箇所
            pass
        elif self.state == "climax":
            # TODO: 顔を出す、笑顔になるアニメーションダミー実行箇所
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
        self.persistence = 5.0  # 遊び実行中は割り込みを強く抑制する
