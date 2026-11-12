import numpy as np
from typing import Dict, Any
from .base import BaseProtocol

class DummyProtocol(BaseProtocol):
    """
    親エージェントが何もしない（ゼロの行動ベクトルを返す）ダミープロトコル。
    親エージェントを退避状態にする場合や、親の介入を完全に無効化したいシナリオで使用します。
    """
    def __init__(self, env: Any):
        super().__init__("DummyProtocol", env)
        self.solo = True
        self.has_target = True  # ターゲットオブジェクトを対象にするため True に変更
        self.state_proximities = {
            "dummy": 1.0  
        }
        self.previous_gaze_prob = None
    
    def start(self, context: Dict[str, Any]):
        super().start(context)
        self.state = "dummy"
        self.previous_gaze_prob = None

    def _manage_state(self, infant_signals: Dict[str, Any]) -> None:
        self.state = "dummy"

    def _determine_action(self) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        return action

    def _calculate_persistence(self, infant_signals: Dict[str, Any]):
        self.persistence = 1.0

    def reward_function(self, infant_signals: Dict[str, Any]) -> float:
        """
        エージェントがターゲットオブジェクトを見ている度合いの変化に基づく報酬を返す
        """
        if self.env.target_obj_id is None:
            return 0.0
            
        gaze_targets = infant_signals.get("gaze_targets", [])
        if len(gaze_targets) == 0:
            return 0.0
            
        # ターゲットオブジェクトのインデックスを取得
        if hasattr(self.env, "all_objects") and isinstance(self.env.all_objects, dict):
            keys = list(self.env.all_objects.keys())
            target_id = getattr(self.env.target_obj_id, "id", self.env.target_obj_id)
            if target_id in keys:
                target_index = keys.index(target_id)
                if 0 <= target_index < len(gaze_targets):
                    current_gaze_prob = float(gaze_targets[target_index])
                    reward = 0.0
                    if self.previous_gaze_prob is not None:
                        # 視線確率の変化量（増加すればプラス、減少すればマイナス）
                        reward = current_gaze_prob - self.previous_gaze_prob
                        
                    self.previous_gaze_prob = current_gaze_prob
                    return reward
                    
        return 0.0
