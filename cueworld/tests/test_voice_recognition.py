"""幼児の発話を親が認識する契約テストと、話し方ごとのレポート。"""

import unittest
from collections import deque
from pathlib import Path

import numpy as np

from cueworld.infant_signal import InfantSignalDetector
from cueworld.language.phonemes import PHONEME_LABELS, build_phoneme_mapping
from cueworld.language.vocabulary import CONCEPT_TO_PHONEMES
from cueworld.language.voice_recognizer import REPLY_CHIGAUYO, REPLY_OSHII, REPLY_SUGOI, VoiceRecognizer


REPORT_PATH = Path(__file__).resolve().parents[1] / "docs" / "voice_recognition_report.md"
VOICE_ACTIVE = [
    "voice_volume",
    "words",
    "voice_unadopted_attempt",
    "voice_segment_seq",
    "voice_reply",
]
PHONEME_COORDS = build_phoneme_mapping()
WORD_NAMES = list(CONCEPT_TO_PHONEMES)


class _VoiceEnv:
    easy_voice = True
    voice_dim = 5
    voice_mapping = {"volume": 0, "pitch": 1, "v_1": 2, "v_2": 3, "v_3": 4}
    phoneme_index = 2
    voice_volume_threshold = 0.25

    def __init__(self):
        self.step_count = 0
        self.infant = {"body": 0}
        self.infant_voice = np.zeros(self.voice_dim, dtype=np.float64)
        self.infant_voice_window = deque(maxlen=3)


def phoneme_vec(label: str, volume: float = 1.0) -> np.ndarray:
    vec = np.zeros(5, dtype=np.float64)
    vec[0] = volume
    vec[2:] = PHONEME_COORDS[label]
    return vec


def silent_vec() -> np.ndarray:
    return np.zeros(5, dtype=np.float64)


def first_foreign(label: str) -> str:
    for other in PHONEME_LABELS:
        if other != label:
            return other
    return "p0"


def unique_mixed_seq(name: str):
    phonemes = list(CONCEPT_TO_PHONEMES[name])
    rec = VoiceRecognizer(dict(CONCEPT_TO_PHONEMES))
    for other in PHONEME_LABELS:
        if other == phonemes[1]:
            continue
        seq = phonemes[:2] + [other] + phonemes[2:]
        rows, _ = rec._rank_words(seq, allow_free_trail=True)
        if not rows:
            continue
        best_word, _raw, best_cost, _u = rows[0]
        second = rows[1][2] if len(rows) > 1 else float("inf")
        if best_word == name and best_cost < second:
            return seq
    return phonemes[:2] + [first_foreign(phonemes[1])] + phonemes[2:]


def fmt_num(value) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "inf"
    return f"{number:.3f}"


class VoiceSession:
    def __init__(self):
        self.env = _VoiceEnv()
        self.detector = InfantSignalDetector(self.env, active_signals=VOICE_ACTIVE)
        self.commits = []

    def reset(self, phase: int = 0, target: str = None) -> None:
        self.env.step_count = int(phase)
        self.env.infant_voice.fill(0.0)
        self.env.infant_voice_window.clear()
        self.detector.voice_recognizer.reset()
        self.detector.voice_recognizer.set_target(target)
        self.detector.word_history.clear()
        self.detector.word_scores_history.clear()
        self.detector.phoneme_history.clear()
        self.detector.phoneme_scores_history.clear()
        self.detector.voice_volume_history.clear()
        self.detector.voice_pitch_history.clear()
        self.detector.voice_segment_seq = 0
        self.detector.voice_unadopted_attempt = False
        self.detector.last_voice_reply = None
        self.detector.last_word_recognition_metrics = {}
        self.commits = []

    def step(self, vec: np.ndarray) -> None:
        self.env.step_count += 1
        self.env.infant_voice[:] = vec
        self.env.infant_voice_window.append(np.array(vec, copy=True))
        before_seq = self.detector.voice_segment_seq
        self.detector.update()
        if self.detector.voice_segment_seq > before_seq:
            word = self.detector.word_history[-1] if self.detector.word_history else None
            metrics = dict(self.detector.last_word_recognition_metrics)
            self.commits.append(
                {
                    "word": word,
                    "unadopted": bool(self.detector.voice_unadopted_attempt),
                    "reply": self.detector.last_voice_reply,
                    "metrics": metrics,
                }
            )

    def hold(self, phonemes, dwell: int) -> None:
        for label in phonemes:
            vec = phoneme_vec(label)
            for _ in range(dwell):
                self.step(vec)

    def hold_varied(self, phonemes, dwells) -> None:
        for label, dwell in zip(phonemes, dwells):
            vec = phoneme_vec(label)
            for _ in range(int(dwell)):
                self.step(vec)

    def silence_ticks(self, n_ticks: int) -> None:
        silent = silent_vec()
        for _ in range(48):
            was_in = self.detector.voice_recognizer.in_segment
            self.step(silent)
            if n_ticks >= 4 and was_in and not self.detector.voice_recognizer.in_segment:
                return
            if (
                n_ticks < 4
                and self.detector.voice_recognizer.in_segment
                and self.detector.voice_recognizer.silence_frames >= n_ticks
            ):
                return

    def close_with_silence(self) -> None:
        self.silence_ticks(4)
        silent = silent_vec()
        for _ in range(6):
            self.step(silent)


def metrics_of(commit) -> dict:
    if commit is None:
        return {}
    return commit.get("metrics") or {}


def row_from_trial(spoken, phase, commits, expected, still_in_segment: bool) -> dict:
    heard = [c["word"] for c in commits]
    heard_text = ",".join("—" if w is None else str(w) for w in heard) if heard else "—"
    expected_text = ",".join(expected)
    spoken_text = spoken if isinstance(spoken, str) else "+".join(spoken)
    last = commits[-1] if commits else None
    metrics = metrics_of(last)
    replies = [c.get("reply") for c in commits if c.get("reply")]
    reply_text = ",".join(str(r) for r in replies) if replies else "—"
    correct = heard == list(expected)
    if correct:
        reason = ""
    elif still_in_segment:
        reason = "無音前に確定せず"
    elif not commits:
        reason = "短すぎて破棄"
    elif all(c["word"] is None for c in commits):
        reason = "ゲート不通過"
    elif any(c["word"] is None for c in commits):
        reason = "ゲート不通過"
    else:
        reason = "誤認識"
    return {
        "spoken": spoken_text,
        "phase": phase,
        "heard": heard_text,
        "correct": correct,
        "expected": expected_text,
        "raw": metrics.get("best_raw_score"),
        "boosted": metrics.get("best_boosted_score"),
        "reply": reply_text,
        "reason": reason,
        "commits": commits,
    }


def run_single(session: VoiceSession, name: str, phase: int, phonemes, dwells) -> dict:
    session.reset(phase)
    if isinstance(dwells, int):
        session.hold(phonemes, dwells)
    else:
        session.hold_varied(phonemes, dwells)
    session.close_with_silence()
    return row_from_trial(
        name,
        phase,
        session.commits,
        [name],
        session.detector.voice_recognizer.in_segment,
    )


def run_pair(session: VoiceSession, first: str, second: str, phase: int, dwell: int) -> dict:
    session.reset(phase)
    session.hold(CONCEPT_TO_PHONEMES[first], dwell)
    session.hold(CONCEPT_TO_PHONEMES[second], dwell)
    session.close_with_silence()
    return row_from_trial(
        (first, second),
        phase,
        session.commits,
        [first, second],
        session.detector.voice_recognizer.in_segment,
    )


def md_table(rows) -> str:
    lines = [
        "| 話した語 | 位相 | 出た語 | 正誤 | 生コスト | 下駄後 | 返事 | 理由 |",
        "|---|---:|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {spoken} | {phase} | {heard} | {ok} | {raw} | {boosted} | {reply} | {reason} |".format(
                spoken=row["spoken"],
                phase=row["phase"],
                heard=row["heard"],
                ok="正" if row["correct"] else "誤",
                raw=fmt_num(row["raw"]),
                boosted=fmt_num(row["boosted"]),
                reply=row["reply"],
                reason=row["reason"] or "",
            )
        )
    return "\n".join(lines)


def failed_rows(rows):
    return [row for row in rows if not row["correct"]]


def summarize_pairs(rows) -> str:
    ok = [row for row in rows if row["correct"]]
    bad = failed_rows(rows)
    raws = [float(row["raw"]) for row in ok if row["raw"] is not None and np.isfinite(row["raw"])]
    parts = [
        f"成功 {len(ok)} / {len(rows)}。",
    ]
    if raws:
        parts.append(f"生コストは最小 {min(raws):.3f}、最大 {max(raws):.3f}。")
    if not bad:
        parts.append("認識できなかった組はない。")
        return " ".join(parts)
    parts.append("認識できなかった組は次のとおり。")
    return " ".join(parts) + "\n\n" + md_table(bad)


def section(title: str, body: str, rows, *, pair: bool = False) -> str:
    failed = failed_rows(rows)
    chunks = [f"## {title}", "", body, ""]
    if pair:
        chunks.append(summarize_pairs(rows))
    else:
        chunks.append(md_table(rows))
        chunks.append("")
        chunks.append("### 認識できなかったパターン")
        chunks.append("")
        if failed:
            chunks.append(md_table(failed))
        else:
            chunks.append("なし。")
    chunks.append("")
    return "\n".join(chunks)


def collect_report(session: VoiceSession) -> dict:
    fast = [
        run_single(session, name, 0, CONCEPT_TO_PHONEMES[name], 3)
        for name in WORD_NAMES
    ]
    stretched = []
    for phase in (0, 1, 2):
        for name in WORD_NAMES:
            stretched.append(run_single(session, name, phase, CONCEPT_TO_PHONEMES[name], 6))
    long_hold = [
        run_single(session, name, 0, CONCEPT_TO_PHONEMES[name], 9)
        for name in WORD_NAMES
    ]
    mixed = []
    for name in WORD_NAMES:
        mixed.append(run_single(session, name, 0, unique_mixed_seq(name), 6))
    silence_short = []
    silence_long = []
    for name in WORD_NAMES:
        phonemes = list(CONCEPT_TO_PHONEMES[name])
        session.reset(0)
        session.hold(phonemes[:2], 6)
        session.silence_ticks(2)
        session.hold(phonemes[2:], 6)
        session.close_with_silence()
        silence_short.append(
            row_from_trial(
                name,
                0,
                session.commits,
                [name],
                session.detector.voice_recognizer.in_segment,
            )
        )
        session.reset(0)
        session.hold(phonemes[:2], 6)
        session.silence_ticks(4)
        session.hold(phonemes[2:], 6)
        session.close_with_silence()
        silence_long.append(
            row_from_trial(
                name,
                0,
                session.commits,
                [name],
                session.detector.voice_recognizer.in_segment,
            )
        )
    partial_short = []
    partial_long = []
    dropped = []
    for name in WORD_NAMES:
        phonemes = list(CONCEPT_TO_PHONEMES[name])
        short_dwells = [6, 6, 3, 6, 6]
        long_dwells = [6, 6, 9, 6, 6]
        partial_short.append(run_single(session, name, 0, phonemes, short_dwells))
        partial_long.append(run_single(session, name, 0, phonemes, long_dwells))
        dropped.append(run_single(session, name, 0, phonemes[:2] + phonemes[3:], 6))
    pairs_6 = [
        run_pair(session, first, second, 0, 6)
        for first in WORD_NAMES
        for second in WORD_NAMES
    ]
    pairs_9 = [
        run_pair(session, first, second, 0, 9)
        for first in WORD_NAMES
        for second in WORD_NAMES
    ]
    replies = collect_replies(session)
    return {
        "fast": fast,
        "stretched": stretched,
        "long_hold": long_hold,
        "mixed": mixed,
        "silence_short": silence_short,
        "silence_long": silence_long,
        "partial_short": partial_short,
        "partial_long": partial_long,
        "dropped": dropped,
        "pairs_6": pairs_6,
        "pairs_9": pairs_9,
        "replies": replies,
    }


def collect_replies(session: VoiceSession) -> list:
    target = "red_cube"
    y = list(CONCEPT_TO_PHONEMES[target])
    rows = []

    def trial(label, phonemes, expected_reply):
        session.reset(0, target=target)
        session.hold(phonemes, 6)
        session.close_with_silence()
        replies = [c.get("reply") for c in session.commits if c.get("reply")]
        heard = [c["word"] for c in session.commits]
        got = replies[-1] if replies else None
        last = session.commits[-1] if session.commits else None
        metrics = metrics_of(last)
        rows.append(
            {
                "spoken": label,
                "phase": 0,
                "heard": ",".join("—" if w is None else str(w) for w in heard) if heard else "—",
                "correct": got == expected_reply,
                "raw": metrics.get("best_raw_score"),
                "boosted": metrics.get("best_boosted_score"),
                "reply": got or "—",
                "reason": "" if got == expected_reply else f"期待 {expected_reply}",
                "commits": session.commits,
            }
        )

    trial("完全一致", y, REPLY_SUGOI)
    sub = list(y)
    sub[2] = first_foreign(sub[2])
    trial("置換1", sub, REPLY_OSHII)
    contig = y[:3] + [first_foreign(y[3]), first_foreign(y[4])]
    trial("連続3音", contig, REPLY_OSHII)
    other = list(CONCEPT_TO_PHONEMES["prompt"])
    trial("無関係", other, REPLY_CHIGAUYO)
    session.reset(0, target=target)
    session.hold(y[:1], 6)
    session.close_with_silence()
    replies = [c.get("reply") for c in session.commits if c.get("reply")]
    rows.append(
        {
            "spoken": "1音",
            "phase": 0,
            "heard": "—",
            "correct": not replies and not session.commits,
            "raw": None,
            "boosted": None,
            "reply": "—",
            "reason": "" if not session.commits else "返事が出た",
            "commits": session.commits,
        }
    )
    session.reset(0, target=target)
    session.hold(y[:2], 6)
    session.close_with_silence()
    replies = [c.get("reply") for c in session.commits if c.get("reply")]
    rows.append(
        {
            "spoken": "2音",
            "phase": 0,
            "heard": "—",
            "correct": not replies and not session.commits,
            "raw": None,
            "boosted": None,
            "reply": "—",
            "reason": "" if not session.commits else "返事が出た",
            "commits": session.commits,
        }
    )
    return rows


def write_report(results: dict) -> None:
    parts = [
        "# いくつかの単語を幼児エージェントに話させて，親エージェントが認識するテスト",
        "",
        "親が読んでいるのは `InfantSignalDetector`。発声は音素頂点へ直接置き、`infant_voice_window` と `update()` に流した。",
        "生コストと下駄後は、その発話で最後に公開された単語の値。出なかったときは最後の公開が無いので空欄。",
        "",
        section(
            "単語を素早く言う",
            "音素あたり 3 step（採取 1 回）。畳んで 5 音になり、黙ったときに判定する。",
            results["fast"],
        ),
        section(
            "単語を少し伸ばして言う",
            "音素あたり 6 step（採取 2 回）。開始 step を 0、1、2 の 3 位相。",
            results["stretched"],
        ),
        section(
            "単語を伸ばして言う",
            "音素あたり 9 step（採取 3 回）。畳み込みで 5 音になり、黙ったときに判定する。",
            results["long_hold"],
        ),
        section(
            "単語の中に関係ない音素を一つだけ混ぜて言う",
            "6 step の語の 2 音素目のあとに、その位置と違う最初の音素を 6 step 入れる。キューが 6 個になったときに判定する。",
            results["mixed"],
        ),
        section(
            "単語の中に沈黙を混ぜて言う（無音 2 ティック）",
            "6 step の語の 2 音素目のあと、区間が閉じない無音 2 ティックを挟む。",
            results["silence_short"],
        ),
        section(
            "単語の中に沈黙を混ぜて言う（無音 4 ティック）",
            "6 step の語の 2 音素目のあと、区間が閉じる無音 4 ティックを挟む。割れたあとの 3 音は別の語になり得る。",
            results["silence_long"],
        ),
        section(
            "単語の一部のみを短く言う",
            "3 音素目だけ 3 step。他は 6 step。",
            results["partial_short"],
        ),
        section(
            "単語の一部のみを長く言う",
            "3 音素目だけ 9 step。他は 6 step。",
            results["partial_long"],
        ),
        section(
            "単語の一部のみ欠損",
            "3 音素目を抜く。残り 4 音素は 6 step。黙ったときに判定する。",
            results["dropped"],
        ),
        section(
            "異なる単語を連続して話す（6 step）",
            f"無音なし。全 {len(WORD_NAMES) * len(WORD_NAMES)} 組。6 音目で先の語を切り、残りから次の語を取る。",
            results["pairs_6"],
            pair=True,
        ),
        section(
            "異なる単語を連続して話す（9 step）",
            f"無音なし。全 {len(WORD_NAMES) * len(WORD_NAMES)} 組。畳み込みで 6 step と同じ長さになる。",
            results["pairs_9"],
            pair=True,
        ),
        section(
            "対象を red_cube に固定した返事",
            "完全一致は「すごい」、置換 1 つと連続 3 音は「おしい」、無関係は「ちがうよ」、1 音と 2 音は返事なし。",
            results["replies"],
        ),
    ]
    REPORT_PATH.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


class TestVoiceRecognition(unittest.TestCase):
    def test_contract_and_write_report(self):
        session = VoiceSession()
        results = collect_report(session)
        write_report(results)

        for label, rows in (
            ("3 step", results["fast"]),
            ("6 step", results["stretched"]),
            ("9 step", results["long_hold"]),
        ):
            for row in rows:
                self.assertTrue(
                    row["correct"],
                    f"{label} の {row['spoken']} が {row['heard']}",
                )
                self.assertAlmostEqual(float(row["raw"]), 0.0, places=3)

        for row in results["pairs_6"]:
            self.assertTrue(
                row["correct"],
                f"6 step 連続 {row['spoken']} が {row['heard']}",
            )
        for row in results["pairs_9"]:
            self.assertTrue(
                row["correct"],
                f"9 step 連続 {row['spoken']} が {row['heard']}",
            )
        for row in results["silence_short"]:
            self.assertTrue(
                row["correct"],
                f"無音 2 ティック {row['spoken']} が {row['heard']}",
            )
        for row in results["mixed"]:
            self.assertTrue(
                row["correct"],
                f"混入 {row['spoken']} が {row['heard']}",
            )
        for row in results["dropped"]:
            self.assertTrue(
                row["correct"],
                f"欠損 {row['spoken']} が {row['heard']}",
            )
        for row in results["replies"]:
            self.assertTrue(
                row["correct"],
                f"返事 {row['spoken']} が {row['reply']} ({row['reason']})",
            )

        session.reset(0)
        session.hold(CONCEPT_TO_PHONEMES["declare"][:1], 3)
        session.close_with_silence()
        self.assertEqual(list(session.detector.word_history), [])
        self.assertFalse(session.detector.voice_unadopted_attempt)
        self.assertEqual(session.detector.voice_segment_seq, 0)

        session.reset(0)
        session.hold(CONCEPT_TO_PHONEMES["declare"][:2], 3)
        session.close_with_silence()
        self.assertEqual(list(session.detector.word_history), [])
        self.assertFalse(session.detector.voice_unadopted_attempt)


if __name__ == "__main__":
    unittest.main()
