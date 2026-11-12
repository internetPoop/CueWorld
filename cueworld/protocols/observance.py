import numpy as np
from typing import Dict, Any
from .base import BaseProtocol

class ObservanceProtocol(BaseProtocol):
    """見守りプロトコル。基本状態。"""
    def __init__(self, env: Any):
        super().__init__("ObservanceProtocol", env)
        # 見守りなのではじめから受理状態(完了)とみなす
        self.solo = True
        self.uses_pre_release = False  # 見守りは腕リセットで指が開く（バッチ F）
        self.state_proximities = {
            "observing": 1.0  
        }
        
    def start(self, context: Dict[str, Any]):
        super().start(context)
        self.state = "observing"

    def _manage_state(self, infant_signals: Dict[str, Any]) -> None:
        # 見守りプロトコルなので特に状態は遷移せずそのまま
        self.state = "observing"

    def _determine_action(self) -> np.ndarray:
        # 特別な行動はせず、見守る（初期姿勢のままなど）ためゼロベクトルを返す
        action = np.zeros(self.action_dim, dtype=np.float32)
        self.env.gaze_agent(self.env.parent, self.env.infant, yaw_link="head", action=action)
        # 腕を使わないので初期姿勢に戻す
        self.env.reset_arm_pose(self.env.parent, action=action)
        return action

    def _calculate_persistence(self, infant_signals: Dict[str, Any]):
        self.persistence = 1.0 #TODO 数値の調整
