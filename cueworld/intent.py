"""§6-D2: 訴えの 0/1 セット（意図レイヤ）。

infant_signal の原子信号から、PM が行列前に合成するセットを定義する。
env 由来の事実（対象 ID 解決・到達圏）は PM 側で渡す／判定する。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# 手の対象の揃え（JA の _pointing_is_reliable と同じ）
HAND_TARGET_STABILITY_MIN = 0.6
# 30 step 平均 gesture_targets の最大値。cone_min_accuracy（検出器）とは別レイヤ
HAND_TARGET_MASS_MIN = 0.2

# 単語: 尖り + 今声がある + 距離が JA 内の判定と同水準
WORD_STABILITY_MIN = 0.6
WORD_MATCH_MAX_DISTANCE = 0.6
DEFAULT_VOICE_VOLUME_THRESHOLD = 0.25

# 親の手がかり
PARENT_GAZE_CUE_MIN = 0.6
PARENT_EYE_CUE_MIN = 0.6

INTENT_KEYS = ("pointing", "reach", "weak_arm", "word", "dyadic")


def _gesture_targets_array(signals: Dict[str, Any]) -> np.ndarray:
    gesture_targets = signals.get("gesture_targets", [])
    if gesture_targets is None:
        return np.zeros(0, dtype=float)
    return np.asarray(gesture_targets, dtype=float)


def hand_target_is_clear(
    signals: Dict[str, Any],
    *,
    mass_min: float = HAND_TARGET_MASS_MIN,
    stability_min: float = HAND_TARGET_STABILITY_MIN,
) -> bool:
    """手が同じ物を向いている（安定）かつ対象がはっきりしている。"""
    if float(signals.get("hand_target_stability", 0.0)) < stability_min:
        return False
    gesture_array = _gesture_targets_array(signals)
    if gesture_array.size == 0:
        return False
    return float(np.max(gesture_array)) >= mass_min


def pointing_is_complete(
    signals: Dict[str, Any],
    *,
    mass_min: float = HAND_TARGET_MASS_MIN,
    stability_min: float = HAND_TARGET_STABILITY_MIN,
) -> bool:
    """揃った指差し（形が pointing・安定・質量）。ignored_targets は含めない。"""
    if float(signals.get("point_intensity", 0.0)) <= 0.0:
        return False
    return hand_target_is_clear(signals, mass_min=mass_min, stability_min=stability_min)


def reach_is_complete(
    signals: Dict[str, Any],
    *,
    mass_min: float = HAND_TARGET_MASS_MIN,
    stability_min: float = HAND_TARGET_STABILITY_MIN,
) -> bool:
    """揃ったリーチ（開き手・安定・質量・持っていない）。"""
    if float(signals.get("open_intensity", 0.0)) <= 0.0:
        return False
    if float(signals.get("holding", 0.0)) > 0.0:
        return False
    return hand_target_is_clear(signals, mass_min=mass_min, stability_min=stability_min)


def weak_arm(
    signals: Dict[str, Any],
    *,
    mass_min: float = HAND_TARGET_MASS_MIN,
    stability_min: float = HAND_TARGET_STABILITY_MIN,
) -> bool:
    """弱い腕: 形を問わず、持っていない物への向きがはっきりしている。"""
    if float(signals.get("holding", 0.0)) > 0.0:
        return False
    return hand_target_is_clear(signals, mass_min=mass_min, stability_min=stability_min)


def word_candidate(
    signals: Dict[str, Any],
    *,
    voice_volume_threshold: float = DEFAULT_VOICE_VOLUME_THRESHOLD,
) -> Optional[str]:
    """
    単語の訴えが立つとき最良の concept_id を返す。立たなければ None。

    単語らしい AND 今声がある AND 距離が十分小さい。
    """
    if float(signals.get("voice_volume", 0.0)) < float(voice_volume_threshold):
        return None
    if float(signals.get("voice_target_stability", 0.0)) < WORD_STABILITY_MIN:
        return None
    if float(signals.get("voice_best_distance", float("inf"))) > WORD_MATCH_MAX_DISTANCE:
        return None
    words = signals.get("words") or {}
    if not words:
        return None
    best_word = min(words, key=words.get)
    return str(best_word)


def dyadic_is_salient(signals: Dict[str, Any]) -> bool:
    """二項: DyadicProtocol と同じ salience・τ。どれか 1 チャネルが τ 以上なら True。"""
    from cueworld.protocols.dyadic import DyadicProtocol

    salience = DyadicProtocol.compute_channel_salience(signals)
    for name, score in salience.items():
        if float(score) >= DyadicProtocol.channel_tau(name):
            return True
    return False


def compute_intent_flags(
    signals: Dict[str, Any],
    *,
    voice_volume_threshold: float = DEFAULT_VOICE_VOLUME_THRESHOLD,
) -> Dict[str, float]:
    """0/1 の意図ベクトル（ゲート前）。word は 1.0/0.0。concept_id は別途 word_candidate。"""
    word_id = word_candidate(signals, voice_volume_threshold=voice_volume_threshold)
    return {
        "pointing": 1.0 if pointing_is_complete(signals) else 0.0,
        "reach": 1.0 if reach_is_complete(signals) else 0.0,
        "weak_arm": 1.0 if weak_arm(signals) else 0.0,
        "word": 1.0 if word_id is not None else 0.0,
        "dyadic": 1.0 if dyadic_is_salient(signals) else 0.0,
    }


def parent_object_cue(signals: Dict[str, Any]) -> float:
    """JA / giving の親手がかり: 見る OR 触る OR 持つ（0/1）。"""
    gaze = float(signals.get("gaze_target_stability", 0.0)) >= PARENT_GAZE_CUE_MIN
    touching = float(signals.get("touching", 0.0)) > 0.0
    holding = float(signals.get("holding", 0.0)) > 0.0
    return 1.0 if (gaze or touching or holding) else 0.0


def parent_dyadic_cue(signals: Dict[str, Any]) -> float:
    """二項の親手がかり: 親の目を見ている（0/1）。"""
    return (
        1.0
        if float(signals.get("looking_at_parent_eye", 0.0)) >= PARENT_EYE_CUE_MIN
        else 0.0
    )
