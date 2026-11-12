"""
easy_voice 認識の評価基盤．

軸 A: 音素置換数 k x 元単語長 L
軸 B: 惜しい発話 level 0-3
別枠: 無音 / 一様ランダムの偽陽性
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cueworld.language.phonemes import PHONEME_LABELS, build_phoneme_prototypes
from cueworld.language.vocabulary import CONCEPT_TO_PHONEMES


@dataclass
class EvalResult:
    label: str
    n: int = 0
    correct: int = 0
    wrong: int = 0
    rejected: int = 0
    confusion: Dict[Tuple[str, str], int] = field(default_factory=dict)

    @property
    def top1(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def false_accept(self) -> float:
        return self.wrong / self.n if self.n else 0.0

    @property
    def reject_rate(self) -> float:
        return self.rejected / self.n if self.n else 0.0

    def record(self, true_word: Optional[str], pred: Optional[str]) -> None:
        self.n += 1
        if pred is None:
            self.rejected += 1
        elif true_word is not None and pred == true_word:
            self.correct += 1
        else:
            self.wrong += 1
            key = (true_word or "<none>", pred)
            self.confusion[key] = self.confusion.get(key, 0) + 1


class _MinimalVoiceEnv:
    """PyBullet 不要のダミー env．"""

    easy_voice = True
    step_count = 0
    action_repeat = 1
    client_id = 0
    infant_voice_window = deque()
    voice_mapping = {"volume": 0, "pitch": 1, "v_1": 2, "v_2": 3, "v_3": 4}
    phoneme_index = 2
    infant = {"body": 0}

    def __init__(self, ignore_v3: bool = False):
        self.ignore_v3 = ignore_v3
        self.phoneme_mapping = {
            label: spec["center"].copy()
            for label, spec in build_phoneme_prototypes().items()
        }
        if ignore_v3:
            for label, coords in self.phoneme_mapping.items():
                coords[2] = 0.0


def make_detector(ignore_v3: bool = False):
    from cueworld.infant_signal import InfantSignalDetector

    env = _MinimalVoiceEnv(ignore_v3=ignore_v3)
    detector = InfantSignalDetector(env, history_length=30, phoneme_history_length=20)
    return detector


def phoneme_to_voice(
    phoneme: str,
    volume: float = 0.8,
    pitch: float = 0.0,
    offset: Optional[np.ndarray] = None,
    ignore_v3: bool = False,
) -> np.ndarray:
    prototypes = build_phoneme_prototypes()
    center = prototypes[phoneme]["center"].astype(np.float32).copy()
    if offset is not None:
        center = center + np.asarray(offset, dtype=np.float32)
    if ignore_v3:
        center[2] = 0.0
    center = np.clip(center, -1.0, 1.0)
    return np.array([volume, pitch, center[0], center[1], center[2]], dtype=np.float32)


def adjacent_phoneme(phoneme: str, rng: np.random.Generator) -> str:
    """1 ビット違いの隣接音素を返す．"""
    idx = PHONEME_LABELS.index(phoneme)
    bit = int(rng.integers(0, 3))
    neighbor = idx ^ (1 << bit)
    return PHONEME_LABELS[neighbor]


def synthesize_phoneme_scores(
    phonemes: Sequence[str],
    detector,
    *,
    coord_blend: float = 0.0,
    duration_jitter: float = 0.0,
    drop_indices: Optional[Sequence[int]] = None,
    elongate_indices: Optional[Sequence[int]] = None,
    ignore_v3: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Optional[str]], List[Dict[str, float]]]:
    """
    音素列から (phoneme_history, phoneme_scores_history) を合成する．
    coord_blend: 隣接音素方向への寄せ量（0=正確，0.3 / 0.6 が near-miss）．
    """
    rng = rng or np.random.default_rng(0)
    drop_set = set(drop_indices or [])
    elongate_set = set(elongate_indices or [])
    history: List[Optional[str]] = []
    scores_hist: List[Dict[str, float]] = []

    for i, p in enumerate(phonemes):
        if i in drop_set:
            continue
        reps = 1
        if i in elongate_set:
            reps = 2
        if duration_jitter > 0:
            if rng.random() < duration_jitter:
                reps = max(1, reps + int(rng.choice([-1, 1])))

        offset = None
        if coord_blend > 0:
            neighbor = adjacent_phoneme(p, rng)
            n_center = build_phoneme_prototypes()[neighbor]["center"]
            p_center = build_phoneme_prototypes()[p]["center"]
            # p から neighbor 方向へ coord_blend * (neighbor - p) / ||.|| * scale
            # より単純に: center = (1-b)*p + b*neighbor 相当のオフセット
            offset = coord_blend * (n_center - p_center)

        for _ in range(reps):
            voice = phoneme_to_voice(p, offset=offset, ignore_v3=ignore_v3)
            label, scores = detector.get_phoneme(voice)
            history.append(label)
            scores_hist.append(scores)
    return history, scores_hist


def recognize_word(
    detector,
    phonemes: Sequence[str],
    **synth_kwargs,
) -> Tuple[Optional[str], Dict[str, float]]:
    history, scores = synthesize_phoneme_scores(phonemes, detector, **synth_kwargs)
    detector.phoneme_history = deque(history, maxlen=max(len(history), 1))
    return detector.get_word(scores)


def substitute_k(
    phonemes: Sequence[str],
    k: int,
    rng: np.random.Generator,
) -> List[str]:
    """k 箇所を別音素へ置換した列を返す．"""
    out = list(phonemes)
    L = len(out)
    if L == 0 or k <= 0:
        return out
    positions = list(rng.choice(L, size=min(k, L), replace=False))
    for pos in positions:
        choices = [p for p in PHONEME_LABELS if p != out[pos]]
        out[pos] = str(rng.choice(choices))
    return out


def eval_substitution_matrix(
    detector=None,
    *,
    k_values: Sequence[int] = (0, 1, 2, 3),
    n_trials: int = 5,
    seed: int = 0,
    ignore_v3: bool = False,
    vocabulary: Optional[Dict[str, List[str]]] = None,
) -> Dict[Tuple[int, int], EvalResult]:
    """軸 A: (k, L) -> EvalResult．"""
    detector = detector or make_detector(ignore_v3=ignore_v3)
    vocab = vocabulary or CONCEPT_TO_PHONEMES
    rng = np.random.default_rng(seed)
    results: Dict[Tuple[int, int], EvalResult] = {}

    for word, phonemes in vocab.items():
        L = len(phonemes)
        for k in k_values:
            key = (k, L)
            if key not in results:
                results[key] = EvalResult(label=f"k={k},L={L}")
            for trial in range(n_trials):
                perturbed = substitute_k(phonemes, k, rng)
                pred, _ = recognize_word(detector, perturbed, ignore_v3=ignore_v3, rng=rng)
                results[key].record(word, pred)
    return results


def eval_near_miss_levels(
    detector=None,
    *,
    levels: Sequence[int] = (0, 1, 2, 3),
    n_trials: int = 3,
    seed: int = 0,
    ignore_v3: bool = False,
    vocabulary: Optional[Dict[str, List[str]]] = None,
) -> Dict[int, EvalResult]:
    """軸 B: level -> EvalResult．"""
    detector = detector or make_detector(ignore_v3=ignore_v3)
    vocab = vocabulary or CONCEPT_TO_PHONEMES
    rng = np.random.default_rng(seed)
    results: Dict[int, EvalResult] = {lv: EvalResult(label=f"level={lv}") for lv in levels}

    for word, phonemes in vocab.items():
        L = len(phonemes)
        for level in levels:
            for trial in range(n_trials):
                kwargs: Dict[str, Any] = {"ignore_v3": ignore_v3, "rng": rng}
                if level == 0:
                    pass
                elif level == 1:
                    kwargs["coord_blend"] = 0.3
                    kwargs["duration_jitter"] = 0.2
                elif level == 2:
                    kind = int(rng.integers(0, 3))
                    if kind == 0:
                        kwargs["coord_blend"] = 0.6
                    elif kind == 1:
                        kwargs["drop_indices"] = [int(rng.integers(0, L))]
                    else:
                        kwargs["elongate_indices"] = [int(rng.integers(0, L))]
                else:  # level 3: two perturbations
                    kwargs["coord_blend"] = 0.6
                    kwargs["drop_indices"] = [int(rng.integers(0, L))]
                pred, _ = recognize_word(detector, phonemes, **kwargs)
                results[level].record(word, pred)
    return results


def eval_false_positives(
    detector=None,
    *,
    n_silence: int = 100,
    n_random: int = 100,
    seed: int = 0,
    ignore_v3: bool = False,
) -> Dict[str, EvalResult]:
    detector = detector or make_detector(ignore_v3=ignore_v3)
    rng = np.random.default_rng(seed)
    silence = EvalResult(label="silence")
    random_fp = EvalResult(label="uniform_random")

    for _ in range(n_silence):
        # volume=0 -> get_phoneme returns None scores
        history = [None] * 5
        scores = [{} for _ in range(5)]
        detector.phoneme_history = deque(history, maxlen=15)
        pred, _ = detector.get_word(scores)
        silence.record(None, pred)

    for _ in range(n_random):
        phonemes = [str(rng.choice(PHONEME_LABELS)) for _ in range(5)]
        # 座標を一様ランダムに
        history = []
        scores_hist = []
        for _p in phonemes:
            coords = rng.uniform(-1, 1, size=3).astype(np.float32)
            if ignore_v3:
                coords[2] = 0.0
            voice = np.array([0.8, 0.0, coords[0], coords[1], coords[2]], dtype=np.float32)
            label, scores = detector.get_phoneme(voice)
            history.append(label)
            scores_hist.append(scores)
        detector.phoneme_history = deque(history, maxlen=15)
        pred, _ = detector.get_word(scores_hist)
        random_fp.record(None, pred)

    return {"silence": silence, "uniform_random": random_fp}


def format_substitution_table(results: Dict[Tuple[int, int], EvalResult]) -> str:
    ks = sorted({k for k, _ in results})
    Ls = sorted({L for _, L in results})
    lines = ["| k \\ L | " + " | ".join(str(L) for L in Ls) + " |",
             "|---|" + "|".join(["---"] * len(Ls)) + "|"]
    for k in ks:
        cells = []
        for L in Ls:
            r = results.get((k, L))
            if r is None or r.n == 0:
                cells.append("-")
            else:
                cells.append(
                    f"top1={r.top1:.2f}/fa={r.false_accept:.2f}/rej={r.reject_rate:.2f} (n={r.n})"
                )
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_level_table(results: Dict[int, EvalResult]) -> str:
    lines = [
        "| level | top-1 | false_accept | reject | n |",
        "|---|---|---|---|---|",
    ]
    for level in sorted(results):
        r = results[level]
        lines.append(
            f"| {level} | {r.top1:.3f} | {r.false_accept:.3f} | {r.reject_rate:.3f} | {r.n} |"
        )
    return "\n".join(lines)


def run_full_eval(
    *,
    n_trials_sub: int = 5,
    n_trials_level: int = 3,
    seed: int = 0,
    ignore_v3: bool = False,
) -> str:
    detector = make_detector(ignore_v3=ignore_v3)
    sub = eval_substitution_matrix(
        detector, n_trials=n_trials_sub, seed=seed, ignore_v3=ignore_v3
    )
    levels = eval_near_miss_levels(
        detector, n_trials=n_trials_level, seed=seed + 1, ignore_v3=ignore_v3
    )
    fps = eval_false_positives(detector, seed=seed + 2, ignore_v3=ignore_v3)

    parts = [
        f"# Voice recognition eval (ignore_v3={ignore_v3})",
        "",
        "## Axis A: phoneme substitutions (k x L)",
        format_substitution_table(sub),
        "",
        "## Axis B: near-miss levels",
        format_level_table(levels),
        "",
        "## False positives",
        f"- silence accept rate: {1.0 - fps['silence'].reject_rate:.4f} (n={fps['silence'].n})",
        f"- uniform_random accept rate: {1.0 - fps['uniform_random'].reject_rate:.4f} (n={fps['uniform_random'].n})",
    ]
    return "\n".join(parts)
