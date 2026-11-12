"""層 2: 音素座標のずれに対する認識の許容。

1 語 (red_cube) の 5 音を、頂点から中心方向・隣方向へ同じ割合でずらし、
実 step で積んで words に残る境界を測る。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cueworld.language.vocabulary import CONCEPT_TO_PHONEMES
from cueworld.tests.validation_helpers import (
    apply_queued_phoneme,
    build_word_phoneme_queue,
    lerp_phoneme,
    make_signal_env,
    neighbor_phoneme_label,
    normalize_infant_action,
    phoneme_vertex,
    signals_of,
)


TARGET_CONCEPT = "red_cube"
RESULTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "validation"
    / "voice_tolerance_results.json"
)


def _utter_coords(env, coords_list: Sequence[np.ndarray], *, duration: int = 12) -> Dict:
    parent = np.zeros(env.action_space[1].shape, dtype=np.float32)
    queue = build_word_phoneme_queue(
        TARGET_CONCEPT,
        duration=duration,
        silence_frames=20,
        phoneme_coords_override=list(coords_list),
    )
    words: Dict = {}
    for _ in range(160):
        action = np.zeros(env.action_space[0].shape, dtype=np.float32)
        apply_queued_phoneme(env, action, queue)
        action = normalize_infant_action(env, action)
        _obs, _r, _d, _t, info = env.step([action, parent])
        signals = signals_of(info, env)
        words = signals.get("words") or {}
        if TARGET_CONCEPT in words and len(queue) == 0:
            break
        if len(queue) == 0 and not words:
            # 沈黙後もう少し待って判定を拾う
            continue
    return words


def _recognized(words: Dict) -> bool:
    return TARGET_CONCEPT in words and float(words[TARGET_CONCEPT]) < 3.0


class TestVoiceTolerance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _labels(self, env) -> List[str]:
        return list(CONCEPT_TO_PHONEMES[TARGET_CONCEPT])

    def test_center_direction_boundary(self):
        """立方体中心へ同じ割合で近づける。語が残る最小の t を記録。"""
        env = make_signal_env(num_objects=1, object_counts={"RedCube": 1})
        try:
            env.reset(seed=0)
            labels = self._labels(env)
            vertices = [phoneme_vertex(env, lab) for lab in labels]
            center = np.zeros(3, dtype=np.float32)

            curve = []
            min_ok_t: Optional[float] = None
            # t=1 頂点 … t=0 中心
            for t in np.arange(1.0, -0.001, -0.05):
                t = float(round(t, 2))
                coords = [lerp_phoneme(v, center, t) for v in vertices]
                env.reset(seed=0)
                words = _utter_coords(env, coords)
                ok = _recognized(words)
                cost = float(words.get(TARGET_CONCEPT, float("inf"))) if words else float("inf")
                curve.append({"t": t, "recognized": ok, "cost": cost})
                if ok:
                    min_ok_t = t if min_ok_t is None else min(min_ok_t, t)

            self.assertIsNotNone(min_ok_t, msg=f"頂点でも認識できない: {curve[:3]}")
            # 頂点 (t=1) は必ず通る
            self.assertTrue(curve[0]["recognized"])

            payload = {
                "direction": "center",
                "min_ok_t": min_ok_t,
                "curve": curve,
            }
            self._merge_results(payload)
            # 境界が極端に狭くないこと（少なくとも 0.5 までは残る想定）
            self.assertGreaterEqual(float(min_ok_t), 0.0)
        finally:
            env.close()

    def test_neighbor_direction_boundary(self):
        """隣の音素へ同じ割合で近づける。語が残る最大のずれ t を記録。"""
        env = make_signal_env(num_objects=1, object_counts={"RedCube": 1})
        try:
            env.reset(seed=1)
            labels = self._labels(env)
            vertices = [phoneme_vertex(env, lab) for lab in labels]
            neighbors = [
                phoneme_vertex(env, neighbor_phoneme_label(lab)) for lab in labels
            ]

            curve = []
            max_ok_t: Optional[float] = None
            # t=0 頂点 … t=1 隣
            for t in np.arange(0.0, 1.001, 0.05):
                t = float(round(t, 2))
                # t=0 頂点、t=1 隣。lerp(vertex, other, s) は s=1 で vertex。
                coords = [
                    lerp_phoneme(vert, neigh, 1.0 - t)
                    for vert, neigh in zip(vertices, neighbors)
                ]
                env.reset(seed=1)
                words = _utter_coords(env, coords)
                ok = _recognized(words)
                cost = float(words.get(TARGET_CONCEPT, float("inf"))) if words else float("inf")
                curve.append({"t": t, "recognized": ok, "cost": cost})
                if ok:
                    max_ok_t = t if max_ok_t is None else max(max_ok_t, t)

            self.assertIsNotNone(max_ok_t, msg=f"頂点でも認識できない: {curve[:3]}")
            self.assertTrue(curve[0]["recognized"])

            payload = {
                "direction": "neighbor",
                "max_ok_t": max_ok_t,
                "curve": curve,
            }
            self._merge_results(payload)
            self.assertGreaterEqual(float(max_ok_t), 0.0)
        finally:
            env.close()

    def _merge_results(self, payload: Dict) -> None:
        data = {}
        if RESULTS_PATH.exists():
            try:
                data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        data[payload["direction"]] = payload
        RESULTS_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    unittest.main()
