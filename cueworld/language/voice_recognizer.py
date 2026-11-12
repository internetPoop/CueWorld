"""
easy_voice 用の音声認識器．

InfantSignalDetector から音素・単語認識のコアロジックを切り出し，
将来の別実装（スペクトル特徴など）と差し替え可能なインターフェースにする．
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cueworld.language.phonemes import build_phoneme_mapping, build_phoneme_prototypes
from cueworld.intent import DEFAULT_VOICE_VOLUME_THRESHOLD

REPLY_SUGOI = "sugoi"
REPLY_OSHII = "oshii"
REPLY_CHIGAUYO = "chigauyo"

QUEUE_MAX = 6
COST_LEAD = 0.01
COST_DEL = 1.0
COST_INS = 1.0
COST_SUB = 1.0
GETA_BONUS = 1.0
ADOPT_RAW_LIMIT = 3.0
SUGOI_RAW_LIMIT = 0.01


class VoiceRecognizer:
    """観測系列から音素・単語候補を返す認識器．"""

    def __init__(
        self,
        word_dictionary: Optional[Dict[str, List[str]]] = None,
        *,
        phoneme_cost_scale: float = 2.0,
        phoneme_match_threshold: float = 0.4,
        phoneme_keep_threshold: float = 0.39,
        phoneme_ambiguity_margin: float = 0.06,
        voice_volume_threshold: float = DEFAULT_VOICE_VOLUME_THRESHOLD,
        use_radius: bool = True,
    ):
        self.phoneme_prototypes = build_phoneme_prototypes()
        self.phoneme_mapping = build_phoneme_mapping()
        self.word_dictionary = word_dictionary or {}
        self.phoneme_cost_scale = phoneme_cost_scale
        self.phoneme_match_threshold = phoneme_match_threshold
        # 頂点の音は通り、隣り合う音素の中点（スコア 0.4）は落ちる。
        self.phoneme_keep_threshold = phoneme_keep_threshold
        self.phoneme_ambiguity_margin = phoneme_ambiguity_margin
        self.voice_volume_threshold = voice_volume_threshold
        self.use_radius = use_radius

        self.cost_del = COST_DEL
        self.cost_ins = COST_INS
        self.cost_lead = COST_LEAD
        self.cost_sub = COST_SUB
        # 後方互換の属性。新しい DP では使わない。
        self.cost_del_blank = 0.05
        self.cost_free_edge = 0.1
        self.cost_elongate = 0.05
        self.rejection_threshold = 0.6
        self.word_margin_threshold = 0.08
        self.word_sequence_quality_threshold = 0.18
        self.word_length_norm = "I"
        self.max_absolute_word_cost = ADOPT_RAW_LIMIT
        self.last_word_recognition_metrics: Dict[str, float] = {}
        # 直近の evaluate_queue が付けた全語スコア（切り分けテストの密な報酬用）。
        # boosted は下駄後、raw は下駄前。公開の words とは別。
        self.last_all_word_scores: Dict[str, float] = {}
        self.last_raw_word_scores: Dict[str, float] = {}
        self.word_eval_seq: int = 0

        self.target_concept: Optional[str] = None
        self.last_reply_kind: Optional[str] = None
        self.last_speak_now: bool = False

        # セグメント状態
        self.in_segment = False
        self.phoneme_queue: List[str] = []
        self.silence_frames = 0
        self.onset_hangover = 2
        self.offset_silence_frames = 4
        self.min_word_frames = 10
        self._pending_reply: Optional[str] = None
        self._spoken_sugoi = False
        self._spoken_oshii = False
        self._spoken_chigauyo = False
        self.segment_phonemes: List[Optional[str]] = []
        self.segment_scores: List[Dict[str, float]] = []
        # 沈黙で区間が閉じ、3〜6 音で単語にならなかったときの列（喃語）。
        self.last_babble_chunk: Optional[List[str]] = None

    def reset(self) -> None:
        self.in_segment = False
        self.phoneme_queue = []
        self.silence_frames = 0
        self.last_word_recognition_metrics = {}
        self.last_all_word_scores = {}
        self.last_raw_word_scores = {}
        self.word_eval_seq = 0
        self.last_reply_kind = None
        self.last_speak_now = False
        self._pending_reply = None
        self._spoken_sugoi = False
        self._spoken_oshii = False
        self._spoken_chigauyo = False
        self.segment_phonemes = []
        self.segment_scores = []
        self.last_babble_chunk = None

    def set_target(self, concept_id: Optional[str]) -> None:
        if concept_id != self.target_concept:
            self.target_concept = concept_id
            self._pending_reply = None
            self._spoken_sugoi = False
            self._spoken_oshii = False
            self._spoken_chigauyo = False

    def get_phoneme(
        self,
        avg_voice: np.ndarray,
        *,
        volume_index: int = 0,
        phoneme_index: int = 2,
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """
        1 フレームの平均音声から音素を推定する．
        スコアは phoneme_cost_scale で正規化した [0,1] コスト．
        """
        phoneme = None
        phoneme_scores: Dict[str, float] = {}
        volume = float(avg_voice[volume_index])
        if volume < self.voice_volume_threshold:
            return None, phoneme_scores

        phoneme = "unknown"
        voice_features = np.array(avg_voice[phoneme_index:], dtype=np.float32)
        for p, spec in self.phoneme_prototypes.items():
            target = spec["center"]
            dist = float(np.linalg.norm(voice_features - target))
            if self.use_radius:
                radius = float(spec.get("radius", 0.0))
                # radius 内はコストを抑える（境界で 0 に落とさず滑らかに）
                dist = max(0.0, dist - 0.25 * radius)
            phoneme_scores[p] = float(min(dist / max(self.phoneme_cost_scale, 1e-6), 1.0))

        if phoneme_scores:
            ranked = sorted(phoneme_scores.items(), key=lambda item: item[1])
            best_phoneme, best_dist = ranked[0]
            second_dist = ranked[1][1] if len(ranked) > 1 else float("inf")
            best_margin = second_dist - best_dist
            if (
                best_dist <= self.phoneme_match_threshold
                and best_margin >= self.phoneme_ambiguity_margin
            ):
                phoneme = best_phoneme
        return phoneme, phoneme_scores

    def label_passes_keep(self, phoneme: Optional[str], scores: Dict[str, float]) -> bool:
        if phoneme in (None, "unknown") or not scores:
            return False
        best = min(float(v) for v in scores.values())
        return best <= self.phoneme_keep_threshold

    def _compute_sequence_quality(
        self,
        phoneme_history: List[Optional[str]],
        phoneme_scores_history: List[Dict[str, float]],
    ) -> Dict[str, float]:
        length = min(len(phoneme_history), len(phoneme_scores_history))
        if length == 0:
            return {
                "length": 0.0,
                "valid_ratio": 0.0,
                "mean_best_cost": 1.0,
                "mean_margin": 0.0,
                "quality": 0.0,
            }

        valid_count = 0
        best_costs = []
        margins = []
        for idx in range(length):
            phoneme = phoneme_history[idx]
            scores = phoneme_scores_history[idx]
            if phoneme not in (None, "unknown"):
                valid_count += 1
            if isinstance(scores, dict) and len(scores) > 0:
                ranked = sorted(float(v) for v in scores.values())
                best_costs.append(ranked[0])
                if len(ranked) > 1:
                    margins.append(ranked[1] - ranked[0])
                else:
                    margins.append(self.phoneme_ambiguity_margin)
            else:
                best_costs.append(1.0)
                margins.append(0.0)

        valid_ratio = float(valid_count) / max(length, 1)
        mean_best_cost = float(np.mean(best_costs)) if best_costs else 1.0
        mean_margin = float(np.mean(margins)) if margins else 0.0
        best_cost_quality = float(
            np.clip(1.0 - (mean_best_cost / max(self.phoneme_match_threshold, 1e-6)), 0.0, 1.0)
        )
        margin_quality = float(
            np.clip(mean_margin / max(self.phoneme_ambiguity_margin, 1e-6), 0.0, 1.0)
        )
        quality = valid_ratio * 0.5 + best_cost_quality * 0.3 + margin_quality * 0.2
        return {
            "length": float(length),
            "valid_ratio": valid_ratio,
            "mean_best_cost": mean_best_cost,
            "mean_margin": mean_margin,
            "quality": float(np.clip(quality, 0.0, 1.0)),
        }

    def align(
        self,
        observed: Sequence[str],
        dictionary: Sequence[str],
        allow_free_trail: bool,
    ) -> Tuple[float, int]:
        """観測列と辞書語の生コストと、単語に使った最後の観測位置を返す。"""
        X = list(observed)
        Y = list(dictionary)
        I = len(X)
        J = len(Y)
        if J == 0:
            return float("inf"), -1
        D = [[float("inf")] * (J + 1) for _ in range(I + 1)]
        bp: List[List[Optional[Tuple[int, int, str]]]] = [
            [None] * (J + 1) for _ in range(I + 1)
        ]
        D[0][0] = 0.0
        for i in range(1, I + 1):
            D[i][0] = i * self.cost_lead
            bp[i][0] = (i - 1, 0, "lead")
        for j in range(1, J + 1):
            D[0][j] = j * self.cost_ins
            bp[0][j] = (0, j - 1, "ins")

        op_rank = {"sub": 0, "trail": 1, "lead": 2, "del": 3, "ins": 4}
        for i in range(1, I + 1):
            for j in range(1, J + 1):
                if (
                    allow_free_trail
                    and j == J
                    and I == QUEUE_MAX
                    and i == I
                ):
                    del_cost = 0.0
                    del_op = "trail"
                else:
                    del_cost = self.cost_del
                    del_op = "del"
                cand_del = D[i - 1][j] + del_cost
                cand_ins = D[i][j - 1] + self.cost_ins
                sub_cost = 0.0 if X[i - 1] == Y[j - 1] else self.cost_sub
                cand_sub = D[i - 1][j - 1] + sub_cost
                candidates = [
                    (cand_sub, (i - 1, j - 1, "sub")),
                    (cand_del, (i - 1, j, del_op)),
                    (cand_ins, (i, j - 1, "ins")),
                ]
                best_cost, best_bp = min(candidates, key=lambda item: (item[0], op_rank[item[1][2]]))
                D[i][j] = best_cost
                bp[i][j] = best_bp

        last_used = -1
        i, j = I, J
        seen = 0
        while i > 0 or j > 0:
            prev = bp[i][j]
            if prev is None:
                break
            pi, pj, op = prev
            if op == "sub":
                last_used = max(last_used, i - 1)
            i, j = pi, pj
            seen += 1
            if seen > (I + J + 2):
                break
        return float(D[I][J]), last_used

    def _has_contiguous_triple(self, observed: Sequence[str], target: Sequence[str]) -> bool:
        if len(observed) < 3 or len(target) < 3:
            return False
        needles = {tuple(target[i : i + 3]) for i in range(len(target) - 2)}
        for i in range(len(observed) - 2):
            if tuple(observed[i : i + 3]) in needles:
                return True
        return False

    def _rank_words(
        self,
        observed: Sequence[str],
        allow_free_trail: bool,
    ) -> Tuple[List[Tuple[str, float, float, int]], Dict[str, float]]:
        raw_scores: Dict[str, float] = {}
        last_used: Dict[str, int] = {}
        for word, phonemes in self.word_dictionary.items():
            cost, used = self.align(observed, phonemes, allow_free_trail)
            raw_scores[word] = cost
            last_used[word] = used

        boosted = dict(raw_scores)
        target = self.target_concept
        if target in raw_scores and self._has_contiguous_triple(
            observed, self.word_dictionary[target]
        ):
            boosted[target] = raw_scores[target] - GETA_BONUS

        ranked = sorted(boosted.items(), key=lambda item: (item[1], item[0]))
        rows = [
            (word, raw_scores[word], boosted[word], last_used[word])
            for word, _ in ranked
        ]
        return rows, boosted

    def _choose_reply(
        self,
        rows: List[Tuple[str, float, float, int]],
    ) -> Optional[str]:
        target = self.target_concept
        if not target or target not in self.word_dictionary:
            return None
        by_word = {word: (raw, boosted) for word, raw, boosted, _ in rows}
        if target not in by_word:
            return REPLY_CHIGAUYO
        target_raw, target_boosted = by_word[target]
        under3 = [(w, b) for w, _raw, b, _u in rows if b < ADOPT_RAW_LIMIT][:3]
        in_top3 = any(w == target for w, _ in under3)
        best_word = rows[0][0] if rows else None
        if best_word == target and target_raw <= SUGOI_RAW_LIMIT:
            return REPLY_SUGOI
        if in_top3 and target_boosted < ADOPT_RAW_LIMIT and target_raw > SUGOI_RAW_LIMIT:
            return REPLY_OSHII
        return REPLY_CHIGAUYO

    def _note_pending(self, reply: Optional[str]) -> None:
        if reply is None:
            return
        if reply == REPLY_SUGOI:
            return
        if self._spoken_oshii and reply == REPLY_CHIGAUYO:
            return
        if reply == REPLY_OSHII:
            self._pending_reply = REPLY_OSHII
            return
        if self._pending_reply == REPLY_OSHII:
            return
        if not self._spoken_oshii:
            self._pending_reply = REPLY_CHIGAUYO

    def _reset_spoken_flags(self) -> None:
        """区間が閉じたあと、次のまとまりでも同じ種類を言えるようにする。"""
        self._spoken_sugoi = False
        self._spoken_oshii = False
        self._spoken_chigauyo = False

    def _emit_reply(self, reply: Optional[str], speak_now: bool) -> Optional[str]:
        self.last_reply_kind = None
        self.last_speak_now = False
        if reply == REPLY_SUGOI:
            if self._spoken_sugoi:
                return None
            self._spoken_sugoi = True
            self._pending_reply = None
            self.last_reply_kind = REPLY_SUGOI
            self.last_speak_now = True
            return REPLY_SUGOI
        # 同じまとまりで「すごい」を出したあとは、閉じたときのおしい／ちがうよを出さない。
        if self._spoken_sugoi:
            self._pending_reply = None
            return None
        self._note_pending(reply)
        if not speak_now:
            return None
        pending = self._pending_reply
        if pending is None:
            return None
        if pending == REPLY_OSHII:
            if self._spoken_oshii:
                self._pending_reply = None
                return None
            self._spoken_oshii = True
            self._pending_reply = None
            self.last_reply_kind = REPLY_OSHII
            self.last_speak_now = True
            return REPLY_OSHII
        if pending == REPLY_CHIGAUYO:
            if self._spoken_oshii or self._spoken_chigauyo:
                self._pending_reply = None
                return None
            self._spoken_chigauyo = True
            self._pending_reply = None
            self.last_reply_kind = REPLY_CHIGAUYO
            self.last_speak_now = True
            return REPLY_CHIGAUYO
        return None

    def evaluate_queue(
        self,
        observed: Optional[Sequence[str]] = None,
        *,
        allow_free_trail: Optional[bool] = None,
        speak_now: bool = False,
        cut: bool = False,
    ) -> Tuple[Optional[str], Dict[str, float]]:
        X = list(observed) if observed is not None else list(self.phoneme_queue)
        I = len(X)
        if allow_free_trail is None:
            allow_free_trail = I == QUEUE_MAX
        if I == 0:
            self.last_word_recognition_metrics = {
                "best_raw_score": float("inf"),
                "best_boosted_score": float("inf"),
                "length": 0.0,
            }
            self.last_all_word_scores = {}
            self.last_raw_word_scores = {}
            self.last_reply_kind = None
            self.last_speak_now = False
            return None, {}

        rows, boosted = self._rank_words(X, allow_free_trail)
        best_word = None
        best_raw = float("inf")
        best_boosted = float("inf")
        last_used = -1
        if rows:
            best_word, best_raw, best_boosted, last_used = rows[0]
        scores = {name: boosted_cost for name, _raw, boosted_cost, _u in rows}
        raw_scores = {name: raw for name, raw, _boosted, _u in rows}
        # 切り分けテスト用。採用の有無にかかわらず全語のコストを残す。
        self.last_all_word_scores = {str(k): float(v) for k, v in scores.items()}
        self.last_raw_word_scores = {str(k): float(v) for k, v in raw_scores.items()}
        self.word_eval_seq += 1
        reply = self._choose_reply(rows)
        adopted = best_word is not None and best_boosted < ADOPT_RAW_LIMIT
        if not adopted:
            best_word = None

        self.last_word_recognition_metrics = {
            "best_raw_score": float(best_raw),
            "best_boosted_score": float(best_boosted),
            "length": float(I),
            "reply_kind": reply or "",
        }
        emitted = self._emit_reply(reply, speak_now=speak_now or reply == REPLY_SUGOI)
        if emitted is not None:
            self.last_word_recognition_metrics["reply_kind"] = emitted

        if adopted and cut and last_used >= 0:
            self.phoneme_queue = X[last_used + 1 :]
        return best_word, scores

    def get_word(
        self,
        phoneme_history: Sequence[Optional[str]],
        phoneme_scores_history: Sequence[Dict[str, float]] = (),
        context=None,
    ) -> Tuple[Optional[str], Dict[str, float]]:
        labels = [p for p in phoneme_history if p not in (None, "unknown")]
        word, scores = self.evaluate_queue(
            labels,
            allow_free_trail=len(labels) == QUEUE_MAX,
            speak_now=False,
            cut=False,
        )
        return word, scores

    def _enqueue(self, label: str) -> Optional[Tuple[Optional[str], Dict[str, float]]]:
        if self.phoneme_queue and self.phoneme_queue[-1] == label:
            return None
        self.phoneme_queue.append(label)
        if len(self.phoneme_queue) > QUEUE_MAX:
            self.phoneme_queue.pop(0)
        if len(self.phoneme_queue) == QUEUE_MAX:
            word, scores = self.evaluate_queue(
                allow_free_trail=True,
                speak_now=False,
                cut=True,
            )
            if word is not None or self.last_speak_now:
                return word, scores
        return None

    def update_segment(
        self,
        volume: float,
        phoneme: Optional[str] = None,
        phoneme_scores: Optional[Dict[str, float]] = None,
        kept_labels: Optional[Sequence[str]] = None,
    ) -> Optional[Tuple[Optional[str], Dict[str, float]]]:
        """
        volume による onset / offset 検出。
        話しているあいだは畳んだキューが 6 個のときだけ判定し、
        無声が offset_silence_frames 続いたら 3 個以上なら判定する。
        """
        voiced = volume >= self.voice_volume_threshold
        labels: List[str] = []
        if kept_labels is not None:
            labels = [lab for lab in kept_labels if lab not in (None, "unknown")]
        elif phoneme_scores is not None and self.label_passes_keep(phoneme, phoneme_scores):
            labels = [phoneme] if phoneme not in (None, "unknown") else []

        if voiced:
            self.silence_frames = 0
            if not self.in_segment:
                self.in_segment = True
            found = None
            for label in labels:
                result = self._enqueue(label)
                if result is not None and found is None:
                    found = result
            return found

        if self.in_segment:
            self.silence_frames += 1
            if self.silence_frames >= self.offset_silence_frames:
                self.in_segment = False
                self.silence_frames = 0
                length = len(self.phoneme_queue)
                if length <= 2:
                    self.phoneme_queue = []
                    self.last_babble_chunk = None
                    emitted = self._emit_reply(self._pending_reply, speak_now=True)
                    self._reset_spoken_flags()
                    if emitted is not None:
                        return None, {}
                    return None
                # クリア前に保存。単語採用なら喃語ではない。
                babble_candidate = list(self.phoneme_queue)
                word, scores = self.evaluate_queue(
                    allow_free_trail=length == QUEUE_MAX,
                    speak_now=True,
                    cut=True,
                )
                self.phoneme_queue = []
                self._reset_spoken_flags()
                if word is not None or self.last_speak_now:
                    self.last_babble_chunk = None
                    return word, scores
                # 3〜6 音でどの単語にも採用されなかった列が喃語。
                # scores は密な報酬用に残す（公開の words は commit 側で空のまま）。
                self.last_babble_chunk = babble_candidate
                return None, scores
        return None
