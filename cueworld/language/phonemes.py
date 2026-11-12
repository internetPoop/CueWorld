"""
easy_voice 用の抽象音素 inventory．

音素ラベルは日本語音ではなく p0..p7 の識別コード．
各ラベルは 3 ビット符号から (v_1, v_2, v_3) 座標を自己記述的に生成する．

  p_i のビット i = b2 b1 b0 に対し
    v_1 = +1 if b0 else -1
    v_2 = +1 if b1 else -1
    v_3 = +1 if b2 else -1

ユークリッド距離 = 2 * sqrt(異なるビット数)．
隣接プロトタイプ間距離は常に 2.0．
"""
from typing import Dict, List

import numpy as np

PHONEME_LABELS: List[str] = [f"p{i}" for i in range(8)]
DEFAULT_PHONEME: str = "p0"
PHONEME_RADIUS: float = 0.8


def phoneme_index_to_coords(index: int) -> np.ndarray:
    """3 ビット符号から (v_1, v_2, v_3) 座標を生成する．"""
    bits = [(index >> b) & 1 for b in range(3)]
    return np.array([1.0 if bit else -1.0 for bit in bits], dtype=np.float32)


def build_phoneme_prototypes(radius: float = PHONEME_RADIUS) -> Dict[str, Dict[str, object]]:
    """ラベル -> {center, radius} のプロトタイプ辞書を返す．"""
    prototypes = {}
    for i, label in enumerate(PHONEME_LABELS):
        prototypes[label] = {
            "center": phoneme_index_to_coords(i),
            "radius": float(radius),
        }
    return prototypes


def build_phoneme_mapping(radius: float = PHONEME_RADIUS) -> Dict[str, np.ndarray]:
    """ラベル -> 座標の単純マッピングを返す．"""
    return {
        label: spec["center"].copy()
        for label, spec in build_phoneme_prototypes(radius).items()
    }
