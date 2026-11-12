"""プロトコルデモ共通。発話帯、日英台本、配置、幼児の動き。"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import deque
from typing import Optional

import numpy as np
import pybullet as p

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from camera_views import get_third_person_view  # noqa: E402
from cueworld.paths import ABS_PATH  # noqa: E402
from cueworld.base import BaseEnv  # noqa: E402
from cueworld.infant_signal import InfantSignalDetector  # noqa: E402
from cueworld.language.manager import (  # noqa: E402
    INTER_WORD_GAP_FRAMES,
    PHONEME_DURATION_FRAMES,
)
from cueworld.language.templates import PROTOCOL_TEMPLATES  # noqa: E402
from cueworld.language.vocabulary import CONCEPT_TO_HUMAN_WORD  # noqa: E402
from cueworld.parent_tools import ParentTools  # noqa: E402
from cueworld.protocol_manager import ProtocolManager  # noqa: E402
from cueworld.protocols import joint_attention as ja_protocol  # noqa: E402
from demo_fixed_ja import (  # noqa: E402
    apply_queued_phoneme,
    build_word_phoneme_queue,
    normalize_infant_action,
    raw_view_from_info,
    _target_concept_id,
)
from demo_ja_voice import (  # noqa: E402
    INFANT_PHONEME_HOLD,
    INFANT_SILENCE_FRAMES,
    nearest_phoneme,
    paint_voice_bands,
    voice_pair_from_obs,
)
from cueworld.video import images_to_video  # noqa: E402

SLIDE_ARENA = (1.8, 2.0)
FPS = 12
TAIL_AFTER_DONE = 12
PARENT_PAUSE_STEPS = 12
POST_SHARED_HOLD_STEPS = 200
POST_SHARED_WAIT_STEPS = 200
VOICE_HISTORY_STEPS = 280
HOUSEHOLD_OFF = 20000
IDLE_AFTER_PLAY = 20
RAISE_FORWARD_DIST = 0.35
MOTOR_CYCLE_STEPS = 30
RAISE_SETTLE_STEPS = 10

SAVE_DIR = f"{ABS_PATH}/results/3d_test"
LINES_DIR = f"{SAVE_DIR}/lines"

ENGLISH_WORDS = {
    "declare": "this",
    "red_cube": "red square",
    "green_cube": "green square",
    "blue_cube": "blue square",
    "yellow_cube": "yellow square",
    "praise": "well done",
    "recast": "that's right",
    "correct": "that's not it",
    "douzo": "here you go",
    "prompt": "hey",
    "mother": "mama",
}

PROTOCOL_JA = {
    "observance": "見守り",
    "joint_attention": "共同注意",
    "object_giving": "受け渡し",
    "dyadic": "二項",
    "household_solo": "家事",
}
PROTOCOL_EN = {
    "observance": "observance",
    "joint_attention": "joint attention",
    "object_giving": "object giving",
    "dyadic": "dyadic",
    "household_solo": "household",
}


def english_word(concept_id: str | None, ja_label: str | None = None) -> str:
    if concept_id and concept_id in ENGLISH_WORDS:
        return ENGLISH_WORDS[concept_id]
    if ja_label == "喃語":
        return "babble (not a word)"
    return ja_label or (concept_id or "")


def human_word(concept_id: str | None) -> str:
    if concept_id is None:
        return ""
    return CONCEPT_TO_HUMAN_WORD.get(concept_id, concept_id)


def slide_object_placements() -> list[dict]:
    """赤を正面から外し、緑と青は同じ奥の列で左右に置く。"""
    yh = SLIDE_ARENA[1]
    z = 0.06
    return [
        {"class": "Red_Cube", "x": 0.0, "y": -yh + 0.28, "z": z},
        {"class": "Green_Cube", "x": -0.42, "y": -yh + 0.42, "z": z},
        {"class": "Blue_Cube", "x": 0.42, "y": -yh + 0.42, "z": z},
    ]


def close_object_placements() -> list[dict]:
    """幼児側の端行に赤を寄せる。受け渡し除外は request_protocol 側で行う。"""
    yh = SLIDE_ARENA[1]
    z = 0.06
    return [
        {"class": "Red_Cube", "x": -1.05, "y": -yh + 0.28, "z": z},
        {"class": "Green_Cube", "x": -0.42, "y": -yh + 0.42, "z": z},
        {"class": "Blue_Cube", "x": 0.42, "y": -yh + 0.42, "z": z},
    ]


def giving_object_placements() -> list[dict]:
    """赤を親側の端行へ置き、幼児からは届かないようにする。"""
    yh = SLIDE_ARENA[1]
    z = 0.06
    return [
        {"class": "Red_Cube", "x": 0.95, "y": -yh + 0.28, "z": z},
        {"class": "Green_Cube", "x": -0.42, "y": -yh + 0.42, "z": z},
        {"class": "Blue_Cube", "x": 0.42, "y": -yh + 0.42, "z": z},
    ]


def sequence_object_placements() -> list[dict]:
    """赤と緑は見える奥、青は親側の端行（受け渡し用）。"""
    yh = SLIDE_ARENA[1]
    z = 0.06
    return [
        {"class": "Red_Cube", "x": 0.0, "y": -yh + 0.28, "z": z},
        {"class": "Green_Cube", "x": -0.42, "y": -yh + 0.42, "z": z},
        {"class": "Blue_Cube", "x": 0.95, "y": -yh + 0.28, "z": z},
    ]


def make_env(
    placements: list[dict],
    *,
    max_steps: int = 1200,
    household_interval_steps: int = HOUSEHOLD_OFF,
    seed: int = 42,
    fixed_protocol: str | None = None,
    fixed_context: dict | None = None,
    wait_base_score: float | None = None,
) -> ProtocolManager:
    kwargs = {}
    if wait_base_score is not None:
        kwargs["wait_base_score"] = wait_base_score
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=len(placements),
        object_counts={},
        max_steps=max_steps,
        return_views=True,
        view_width=256,
        view_height=256,
        action_repeat=1,
        arena_half_extents=SLIDE_ARENA,
        arena_rows=3,
        arena_cols=4,
        fixed_placements=placements,
        debug_info=True,
        placement_method="grid",
        use_cinematic_model=True,
    )
    env = ParentTools(base)
    env = InfantSignalDetector(env)
    env = ProtocolManager(
        env,
        household_interval_steps=household_interval_steps,
        rejection=False,
        fixed_protocol=fixed_protocol,
        fixed_context=fixed_context or {},
        **kwargs,
    )
    ja_protocol.ATTENTION_HOLD_STEPS = POST_SHARED_HOLD_STEPS
    ja_protocol.ATTENTION_WAIT_STEPS = POST_SHARED_WAIT_STEPS
    return env


def find_object(env, class_name: str) -> Optional[int]:
    for obj_id, obj in env.all_objects.items():
        if obj.__class__.__name__ == class_name:
            return obj_id
    return None


def protocol_target_id(protocol) -> Optional[int]:
    if protocol is None:
        return None
    target = getattr(protocol, "target_object", None)
    if target is None:
        return None
    if hasattr(target, "id"):
        return target.id
    return target


def find_signal_detector(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        # __getattr__ で内側の属性が見えるラッパは、代入が自分自身に付くので除外する。
        owned = vars(current)
        if "word_history" in owned and "voice_recognizer" in owned:
            return current
        current = getattr(current, "env", None)
    return None


def silence_infant_voice(env) -> None:
    """前の発話の残りが次の区間に入らないようにする。"""
    if hasattr(env, "infant_voice"):
        env.infant_voice[:] = 0
    if hasattr(env, "infant_voice_window"):
        env.infant_voice_window.clear()
    detector = find_signal_detector(env)
    if detector is not None and hasattr(detector, "voice_recognizer"):
        detector.voice_recognizer.reset()
    # 前の「すごい」が次の共同注意の成立直後に再放送されないようにする。
    if detector is not None and hasattr(detector, "last_voice_reply"):
        detector.last_voice_reply = None


def skip_point_while_looking(manager) -> None:
    """state が looking のときだけ、親の指差しを出さない。"""
    protocol = manager.current_protocol
    if protocol is None:
        return
    inner = protocol.env
    point = inner.point_object

    def point_object(agent, oid, action=None, ik_every=0):
        if getattr(protocol, "state", None) == "looking":
            return action
        return point(agent, oid, action=action, ik_every=ik_every)

    inner.point_object = point_object


def neutralize_appeal(signals: dict, env) -> dict:
    out = dict(signals)
    out["point_intensity"] = 0.0
    out["open_intensity"] = 0.0
    out["hand_target_stability"] = 0.0
    out["hand_height_margin"] = 0.0
    out["hand_open_close_periodicity"] = 0.0
    out["babbling_ratio"] = 0.0
    out["voice_volume"] = 0.0
    out["looking_at_parent_eye"] = 0.0
    n = len(getattr(env, "trackable_targets", []) or [])
    out["gesture_targets"] = np.zeros(n, dtype=float)
    return out


def inject_complete_pointing(signals: dict, env, target_id: Optional[int]) -> dict:
    if target_id is None:
        return signals
    out = neutralize_appeal(signals, env)
    gesture_targets = np.zeros(len(env.trackable_targets), dtype=float)
    if target_id in env.trackable_targets:
        gesture_targets[env.trackable_targets.index(target_id)] = 1.0
    out["gesture_targets"] = gesture_targets
    out["point_intensity"] = 1.0
    out["hand_target_stability"] = 1.0
    return out


def inject_complete_reach(signals: dict, env, target_id: Optional[int]) -> dict:
    if target_id is None:
        return signals
    out = neutralize_appeal(signals, env)
    gesture_targets = np.zeros(len(env.trackable_targets), dtype=float)
    if target_id in env.trackable_targets:
        gesture_targets[env.trackable_targets.index(target_id)] = 1.0
    out["gesture_targets"] = gesture_targets
    out["open_intensity"] = 1.0
    out["hand_target_stability"] = 1.0
    return out


def inject_gaze_cue(signals: dict, env, target_id: Optional[int]) -> dict:
    out = neutralize_appeal(signals, env)
    n = len(getattr(env, "trackable_targets", []) or [])
    gaze = np.zeros(n, dtype=float)
    if target_id is not None and target_id in env.trackable_targets:
        gaze[env.trackable_targets.index(target_id)] = 1.0
    out["gaze_targets"] = gaze
    out["gaze_target_stability"] = 1.0
    # 視線の安定は窓平均なので、開始直後は 0.6 に届かないことがある。
    # 親の物手がかりは見る OR 触るなので、デモでは触るも立てて開始を安定させる。
    out["touching"] = 1.0
    return out


def inject_dyadic_raise(signals: dict, env) -> dict:
    out = neutralize_appeal(signals, env)
    out["hand_height_margin"] = 1.0
    return out


def install_signal_override(manager: ProtocolManager, runtime: dict) -> None:
    detector = manager.env
    orig = detector.get_normalized_signals

    def wrapped(ignored_targets=None):
        sigs = orig(ignored_targets=ignored_targets)
        mode = runtime.get("mode") or "idle"
        target_id = runtime.get("inject_id")
        if mode == "idle":
            return neutralize_appeal(sigs, manager)
        if mode == "point":
            return inject_complete_pointing(sigs, manager, target_id)
        if mode == "reach":
            return inject_complete_reach(sigs, manager, target_id)
        if mode == "gaze":
            return inject_gaze_cue(sigs, manager, target_id)
        if mode == "dyadic":
            return inject_dyadic_raise(sigs, manager)
        return sigs

    detector.get_normalized_signals = wrapped


def infant_raise_pos(env) -> np.ndarray:
    infant = env.infant
    torso_pos, fwd = env._agent_torso_pose(infant)
    direction = np.asarray([fwd[0], fwd[1]], dtype=np.float32)
    dnorm = float(np.linalg.norm(direction))
    if dnorm < 1e-4:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / dnorm
    xy = np.asarray(torso_pos, dtype=np.float32)[:2] + direction * float(RAISE_FORWARD_DIST)
    eyes_state = p.getLinkState(
        infant["body"],
        infant["eyes_link"],
        computeForwardKinematics=True,
        physicsClientId=env.client_id,
    )
    return np.array(
        [float(xy[0]), float(xy[1]), float(eyes_state[4][2])], dtype=np.float32
    )


def apply_infant_raise(env, action: np.ndarray) -> None:
    env.hand_reach_location(env.infant, infant_raise_pos(env), action=action)


def template_word_spans(template_key: str, concept_id: str | None, language_manager) -> list[tuple[int, int, str, str]]:
    """テンプレートを展開したときの単語区間 [start, end), 日本語, concept。"""
    template_list = None
    for group in PROTOCOL_TEMPLATES.values():
        if template_key in group:
            template_list = group[template_key]
            break
    if not template_list:
        return []
    spans = []
    frame = 0
    words_emitted = 0
    for item in template_list:
        resolved = concept_id if item == "TARGET" else item
        if resolved is None:
            continue
        phonemes = language_manager.get_phonemes(resolved)
        if not phonemes:
            continue
        if words_emitted > 0 and INTER_WORD_GAP_FRAMES > 0:
            frame += INTER_WORD_GAP_FRAMES
        length = len(phonemes) * PHONEME_DURATION_FRAMES
        spans.append((frame, frame + length, language_manager.get_human_word(resolved), resolved))
        frame += length
        words_emitted += 1
    return spans


class SpeechGate:
    """親が無音になってから PARENT_PAUSE_STEPS 空いたら幼児が話し始める。"""

    def __init__(self, pause_steps: int = PARENT_PAUSE_STEPS):
        self.pause_steps = int(pause_steps)
        self.ready = False
        self.heard = False
        self.silent = 0

    def update(self, parent_phoneme: str | None, allow: bool) -> bool:
        if self.ready or not allow:
            return self.ready
        if parent_phoneme is not None:
            self.heard = True
            self.silent = 0
        elif self.heard:
            self.silent += 1
            if self.silent >= self.pause_steps:
                self.ready = True
        return self.ready


class DemoRecorder:
    def __init__(self, env, history_steps: int = VOICE_HISTORY_STEPS):
        self.env = env
        self.history_steps = history_steps
        self.movies = {"infant": [], "third": []}
        self.parent_phonemes: list[str | None] = []
        self.infant_phonemes: list[str | None] = []
        self.lines: list[dict] = []
        self.switches: list[dict] = []
        self.detector = find_signal_detector(env)
        self.threshold = float(getattr(env, "voice_volume_threshold", 0.25))
        self.mapping = getattr(env, "phoneme_mapping", {})
        self.prev_utt_key = None
        self.prev_word_count = len(self.detector.word_history) if self.detector is not None else 0
        self.prev_protocol_name = getattr(env, "current_protocol_name", None)
        self.parent_voiced_open: dict | None = None
        self.infant_voiced_open: dict | None = None

    def _language_manager(self):
        protocol = getattr(self.env, "current_protocol", None)
        if protocol is not None and hasattr(protocol, "language_manager"):
            return protocol.language_manager
        return None

    def append_voices(self, obs) -> None:
        parent_voice, infant_voice = voice_pair_from_obs(obs)
        self.parent_phonemes.append(nearest_phoneme(parent_voice, self.mapping, self.threshold))
        self.infant_phonemes.append(nearest_phoneme(infant_voice, self.mapping, self.threshold))

    def note_switch(self, t: int) -> None:
        name = getattr(self.env, "current_protocol_name", None)
        if name != self.prev_protocol_name:
            self.switches.append(
                {
                    "t": t,
                    "from": self.prev_protocol_name,
                    "to": name,
                    "infant_initiative": bool(getattr(self.env, "current_infant_initiative", False)),
                }
            )
            print(f"t={t:03d} protocol {self.prev_protocol_name} -> {name}")
            self.prev_protocol_name = name

    def note_parent_utterance(self, t: int, q_before: int) -> None:
        protocol = getattr(self.env, "current_protocol", None)
        if protocol is None:
            return
        utt = (getattr(protocol, "debug_info", {}) or {}).get("last_utterance") or {}
        utt_key = (utt.get("template_key"), utt.get("concept_id"))
        if not utt.get("template_key") or utt_key == self.prev_utt_key:
            return
        lm = self._language_manager()
        if lm is None:
            return
        spans = template_word_spans(utt_key[0], utt_key[1], lm)
        origin = (len(self.parent_phonemes) - 1) + q_before
        proto = getattr(self.env, "current_protocol_name", None)
        for start, end, label, concept in spans:
            self.lines.append(
                {
                    "t0": origin + start,
                    "t1": origin + end,
                    "speaker": "parent",
                    "protocol": proto,
                    "ja": label,
                    "en": english_word(concept, label),
                    "concept": concept,
                    "kind": "word",
                }
            )
        self.prev_utt_key = utt_key
        print(
            f"t={t:03d} parent utterance {utt_key[0]} "
            f"words={[label for _, _, label, _ in spans]}"
        )

    def note_infant_word(self, t: int) -> None:
        if self.detector is None or len(self.detector.word_history) <= self.prev_word_count:
            return
        word = self.detector.word_history[-1]
        self.prev_word_count = len(self.detector.word_history)
        if not word:
            print(f"t={t:03d} infant recognition returned None")
            return
        i = len(self.infant_phonemes) - 1
        while i >= 0 and self.infant_phonemes[i] is None:
            i -= 1
        if i < 0:
            return
        end = i + 1
        while i >= 0 and self.infant_phonemes[i] is not None:
            i -= 1
        start = i + 1
        label = human_word(str(word))
        self.lines.append(
            {
                "t0": start,
                "t1": end,
                "speaker": "infant",
                "protocol": getattr(self.env, "current_protocol_name", None),
                "ja": label,
                "en": english_word(str(word), label),
                "concept": str(word),
                "kind": "word",
            }
        )
        print(f"t={t:03d} infant recognized {word} ({label})")

    def _close_babble(self, open_span: dict | None, t1: int, speaker: str) -> None:
        if open_span is None:
            return
        covered = False
        for line in self.lines:
            if line["speaker"] != speaker or line.get("kind") != "word":
                continue
            if line["t0"] < t1 and line["t1"] > open_span["t0"]:
                covered = True
                break
        if covered:
            return
        self.lines.append(
            {
                "t0": open_span["t0"],
                "t1": t1,
                "speaker": speaker,
                "protocol": open_span.get("protocol"),
                "ja": "喃語",
                "en": "babble (not a word)",
                "concept": None,
                "kind": "babble",
            }
        )

    def flush_babble(self) -> None:
        """単語に載らない有声区間を喃語として残す。"""
        def walk(phonemes: list[str | None], speaker: str) -> None:
            start = None
            proto = None
            for i, ph in enumerate(phonemes):
                if ph is not None:
                    if start is None:
                        start = i
                        proto = getattr(self.env, "current_protocol_name", None)
                elif start is not None:
                    self._close_babble({"t0": start, "protocol": proto}, i, speaker)
                    start = None
            if start is not None:
                self._close_babble({"t0": start, "protocol": proto}, len(phonemes), speaker)

        walk(self.parent_phonemes, "parent")
        walk(self.infant_phonemes, "infant")

    def after_step(self, obs, t: int, q_before: int) -> None:
        self.append_voices(obs)
        self.note_switch(t)
        self.note_parent_utterance(t, q_before)
        self.note_infant_word(t)

    def capture(self, info) -> None:
        frame = raw_view_from_info(info or {}, "infant_rgb")
        if frame is not None:
            self.movies["infant"].append(
                paint_voice_bands(
                    frame,
                    self.parent_phonemes,
                    self.infant_phonemes,
                    history_steps=self.history_steps,
                )
            )
        third = get_third_person_view(self.env, orbit_deg=32, distance=3.1)
        if third is not None:
            self.movies["third"].append(
                paint_voice_bands(
                    third,
                    self.parent_phonemes,
                    self.infant_phonemes,
                    history_steps=self.history_steps,
                )
            )

    def save_videos(self, stem: str) -> dict[str, str]:
        os.makedirs(SAVE_DIR, exist_ok=True)
        paths = {}
        for view_name, frames in self.movies.items():
            if not frames:
                print(f"[warn] no frames for view={view_name}")
                continue
            if view_name == "infant":
                video_path = f"{SAVE_DIR}/{stem}.mp4"
            else:
                video_path = f"{SAVE_DIR}/{stem}_{view_name}.mp4"
            print(f"Saving video to {video_path} ({len(frames)} frames)...")
            images_to_video(frames, video_path, fps=FPS, scale=1)
            paths[view_name] = video_path
        return paths

    def write_lines(
        self,
        stem: str,
        *,
        title: str,
        situation_ja: str,
        situation_en: str,
        result: dict,
    ) -> str:
        self.flush_babble()
        os.makedirs(LINES_DIR, exist_ok=True)
        path = f"{LINES_DIR}/{stem}.md"
        rows = sorted(self.lines, key=lambda r: (r["t0"], 0 if r["speaker"] == "parent" else 1))
        lines = [
            f"# {title}",
            "",
            "## 状況 / Situation",
            "",
            situation_ja,
            "",
            situation_en,
            "",
            "## セリフ / Lines",
            "",
            "| 秒 / s | 話者 / Speaker | プロトコル / Protocol | 日本語 | English |",
            "|---:|---|---|---|---|",
        ]
        speaker_ja = {"parent": "親", "infant": "幼児"}
        speaker_en = {"parent": "parent", "infant": "infant"}
        for row in rows:
            sec = row["t0"] / float(FPS)
            proto = row.get("protocol") or ""
            proto_cell = f"{PROTOCOL_JA.get(proto, proto)} / {PROTOCOL_EN.get(proto, proto)}"
            who = f"{speaker_ja.get(row['speaker'], row['speaker'])} / {speaker_en.get(row['speaker'], row['speaker'])}"
            lines.append(
                f"| {sec:.1f} | {who} | {proto_cell} | {row['ja']} | {row['en']} |"
            )
        if not rows:
            lines.append("| — | — | — | （発話なし） | (no speech) |")
        lines.extend(
            [
                "",
                "## 結果 / Result",
                "",
                f"- 終了プロトコル / final protocol: `{result.get('final_protocol')}`",
                f"- 終了状態 / final state: `{result.get('final_state')}`",
                f"- 成功 / success: `{result.get('success')}`",
                f"- word_match: `{result.get('word_match')}`",
                f"- 切替回数 / switches: `{result.get('n_switches', len(self.switches))}`",
                f"- フレーム / frames: `{result.get('frames')}`",
                "",
            ]
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Wrote lines to {path}")
        return path


def write_result_json(stem: str, payload: dict) -> str:
    os.makedirs(LINES_DIR, exist_ok=True)
    path = f"{LINES_DIR}/{stem}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


CATALOG_ORDER = [
    "demo_pm_infant_ja",
    "demo_pm_parent_ja",
    "demo_single_ja_looking",
    "demo_single_ja_pointing",
    "demo_single_ja_shake",
    "demo_single_dyadic_voice",
    "demo_single_dyadic_arm",
    "demo_single_dyadic_hand",
    "demo_single_giving",
    "demo_pm_sequence",
    "demo_pm_ja_retarget",
    "demo_pm_ja_point_to_shake",
    "demo_pm_parent_ja_away",
]


def word_match_label(value) -> str | None:
    """返事の種類。列なら最後に言い終えたもの。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return str(value[-1])
    text = str(value)
    return text or None


def result_sentence(payload: dict) -> str:
    result = payload.get("result") or {}
    success = result.get("success")
    word = word_match_label(result.get("word_match"))
    proto = result.get("final_protocol")
    n_sw = result.get("n_switches")
    bits = []
    if success is True:
        bits.append("成功")
    elif success is False:
        bits.append("失敗")
    if word:
        bits.append(f"word_match={word}")
    if proto:
        bits.append(f"最後のプロトコルは {proto}")
    if n_sw is not None:
        bits.append(f"切替 {n_sw} 回")
    return "、".join(bits) if bits else "記録なし"


def write_catalog_from_jsons() -> str:
    entries = []
    for stem in CATALOG_ORDER:
        path = f"{LINES_DIR}/{stem}.json"
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        videos = payload.get("videos") or {}
        entries.append(
            {
                "title": payload.get("title", stem),
                "situation_ja": payload.get("situation_ja", ""),
                "result_ja": result_sentence(payload),
                "infant_video": videos.get("infant"),
                "third_video": videos.get("third"),
                "lines": payload.get("lines"),
            }
        )
    return write_catalog(entries)


def write_catalog(entries: list[dict]) -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = f"{SAVE_DIR}/demo_catalog.md"
    lines = [
        "# プロトコルデモ一覧",
        "",
        "幼児一人称と三人称の動画、発話の日英台本、状況と結果。",
        "",
    ]
    for entry in entries:
        lines.append(f"## {entry['title']}")
        lines.append("")
        lines.append(entry["situation_ja"])
        lines.append("")
        lines.append(f"- 結果: {entry['result_ja']}")
        infant = entry.get("infant_video")
        third = entry.get("third_video")
        script = entry.get("lines")
        if infant:
            lines.append(f"- 幼児視点: [{os.path.basename(infant)}]({os.path.relpath(infant, SAVE_DIR)})")
        if third:
            lines.append(f"- 三人称: [{os.path.basename(third)}]({os.path.relpath(third, SAVE_DIR)})")
        if script:
            lines.append(f"- セリフ: [{os.path.basename(script)}]({os.path.relpath(script, SAVE_DIR)})")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote catalog to {path}")
    return path


def empty_action(env) -> np.ndarray:
    return np.zeros(env.action_space[0].shape, dtype=np.float32)


def maybe_queue_target_word(
    env,
    target_obj,
    speech_queue: deque,
    speech_started: list,
    speech_ready: bool,
) -> None:
    if not speech_ready or speech_started[0] or len(speech_queue) > 0:
        return
    concept_id = _target_concept_id(env, target_obj)
    if concept_id is None:
        return
    speech_queue.extend(
        build_word_phoneme_queue(
            concept_id,
            duration=INFANT_PHONEME_HOLD,
            silence_frames=INFANT_SILENCE_FRAMES,
        )
    )
    speech_started[0] = True
    print(f"[infant] queued word {concept_id} hold={INFANT_PHONEME_HOLD}")


def ja_engage_action(
    env,
    target_obj,
    speech_queue: deque,
    speech_started: list,
    speech_ready: bool,
    *,
    point: bool = False,
) -> np.ndarray:
    action = empty_action(env)
    if target_obj is not None:
        env.gaze_object(env.infant, target_obj, yaw_link="head", action=action)
        if point:
            env.point_object(env.infant, target_obj, action=action)
    maybe_queue_target_word(env, target_obj, speech_queue, speech_started, speech_ready)
    if len(speech_queue) > 0 or speech_started[0]:
        apply_queued_phoneme(env, action, speech_queue)
    return normalize_infant_action(env, action)


def giving_engage_action(env, target_obj, reaching_phase: list) -> np.ndarray:
    action = empty_action(env)
    protocol = getattr(env, "current_protocol", None)
    state = getattr(protocol, "state", None) if protocol else None
    if target_obj is None:
        return normalize_infant_action(env, action)
    if state in ("init", "fetching", "approach", "showing", "placing"):
        if not reaching_phase[0]:
            yaw, _pitch = env.gaze_object(env.infant, target_obj, yaw_link="torso", action=action)
            if abs(yaw) < math.radians(5.0):
                reaching_phase[0] = True
        else:
            env.hand_reach_object(env.infant, target_obj, action=action)
            env.gaze_object(env.infant, target_obj, yaw_link="head", action=action)
    elif state == "waiting_infant":
        env.grasp_object(
            env.infant,
            target_obj,
            action=action,
            approach_dist=float(env.grasp_approach_dist),
        )
    else:
        env.gaze_object(env.infant, target_obj, yaw_link="head", action=action)
    return normalize_infant_action(env, action)


# 二項は見本の途中からモータが立ち、見本のあと喃語を待つ。
DYADIC_RESPOND_STATES = (
    "performing_parent_cue",
    "waiting_babble",
    "waiting_infant",
    "waiting_infant_response",
    "echoing",
    "acknowledging",
    "imitating",
    "completed",
)
# 3 音。待ちは 50 step なので、音を伸ばしすぎると往復の前に失敗する。
# p0 p1 p0 は「どうぞ」に採用される。p0 p2 p0 は単語にならない。
DEMO_BABBLE = ("p0", "p2", "p0")


def dyadic_respond_action(env, t: int, *, voice: bool = False, hand: bool = False) -> np.ndarray:
    action = empty_action(env)
    env.gaze_agent(env.infant, env.parent, yaw_link="head", action=action)
    protocol = getattr(env, "current_protocol", None)
    state = getattr(protocol, "state", "") if protocol else ""
    responding = state in DYADIC_RESPOND_STATES or bool(
        getattr(protocol, "motor_active", False)
    )
    # 声の見本では、喃語の往復が始まる前に掲げるとモータだけ先に成功して終わる。
    raise_now = responding and protocol is not None and (
        not voice or state == "echoing" or int(getattr(protocol, "round_count", 0)) > 0
    )
    if raise_now:
        apply_infant_raise(env, action)
        held = int(getattr(protocol, "_demo_raise_steps", 0)) + 1
        protocol._demo_raise_steps = held
        if hand and held >= RAISE_SETTLE_STEPS:
            env.animate_hand_open_close(
                env.infant,
                action=action,
                phase=float((t % MOTOR_CYCLE_STEPS) / MOTOR_CYCLE_STEPS),
                gesture=True,
            )
    if voice and state == "waiting_babble" and protocol is not None:
        _apply_demo_babble(env, action, protocol)
    return normalize_infant_action(env, action)


def _apply_demo_babble(env, action: np.ndarray, protocol) -> None:
    """見本のあと、単語にならない 3 音を一度だけ言って黙る。

    幼児の声は加算なので、音の境目で別の音素が混ざる。
    返事のあいだだけ発話ベクトルを音素そのものに置く。
    """
    if not hasattr(env, "phoneme_mapping") or not hasattr(env, "infant_voice"):
        return
    queue = getattr(protocol, "_demo_babble", None)
    if queue is None:
        queue = deque()
        for phoneme in DEMO_BABBLE:
            queue.extend([phoneme] * 6)
        queue.extend([None] * 15)
        protocol._demo_babble = queue
    if not queue:
        return
    phoneme = queue.popleft()
    voice = np.zeros(env.voice_dim, dtype=np.float32)
    if phoneme is not None and phoneme in env.phoneme_mapping:
        coords = np.asarray(env.phoneme_mapping[phoneme], dtype=np.float32).reshape(-1)
        voice[env.voice_mapping["volume"]] = 1.0
        for axis, key in enumerate(("v_1", "v_2", "v_3")):
            if axis < coords.size and key in env.voice_mapping:
                voice[env.voice_mapping[key]] = float(coords[axis])
        protocol._demo_babble_silenced = False
    elif not getattr(protocol, "_demo_babble_silenced", False):
        if hasattr(env, "infant_voice_window"):
            env.infant_voice_window.clear()
        protocol._demo_babble_silenced = True
    env.infant_voice[:] = voice


def look_parent_action(env) -> np.ndarray:
    action = empty_action(env)
    env.gaze_agent(env.infant, env.parent, yaw_link="head", action=action)
    return normalize_infant_action(env, action)


def look_object_action(env, target_obj, *, point: bool = False, reach: bool = False) -> np.ndarray:
    action = empty_action(env)
    if target_obj is not None:
        env.gaze_object(env.infant, target_obj, yaw_link="head", action=action)
        if point:
            env.point_object(env.infant, target_obj, action=action)
        if reach:
            env.hand_reach_object(env.infant, target_obj, action=action)
    return normalize_infant_action(env, action)


def raise_action(env) -> np.ndarray:
    action = empty_action(env)
    env.gaze_agent(env.infant, env.parent, yaw_link="head", action=action)
    apply_infant_raise(env, action)
    return normalize_infant_action(env, action)


def step_parent(env, info, infant_action):
    parent_action = np.zeros(env.action_space[1].shape, dtype=np.float32)
    if isinstance(info, dict) and "parent_action" in info:
        parent_action = info["parent_action"]
    protocol = getattr(env, "current_protocol", None)
    q_before = len(getattr(protocol, "utterance_queue", [])) if protocol is not None else 0
    obs, reward, done, trunc, info = env.step((infant_action, parent_action))
    return obs, reward, done, trunc, info, q_before


def ja_result(env, recorder: DemoRecorder, frames: int, prefer: str | None = None) -> dict:
    protocol = getattr(env, "current_protocol", None)
    name = getattr(env, "current_protocol_name", None)
    hist = list(getattr(env, "protocol_history", []) or [])
    last_play = None
    if prefer:
        for item in reversed(hist):
            if item.get("name") == prefer:
                last_play = item
                break
    if last_play is None:
        for item in reversed(hist):
            if item.get("name") in ("joint_attention", "object_giving", "dyadic"):
                last_play = item
                break
    success = None
    if last_play is not None:
        success = bool(last_play.get("success"))
    elif protocol is not None:
        success = getattr(protocol, "state", None) == getattr(protocol, "success_state", "completed")
    word_match = None
    if name == "joint_attention":
        word_match = word_match_label(getattr(protocol, "word_match", None))
    elif last_play is not None:
        word_match = word_match_label(last_play.get("word_match"))
    return {
        "final_protocol": name if last_play is None else last_play.get("name"),
        "final_state": last_play.get("end_reason") if last_play is not None else (
            getattr(protocol, "state", None) if protocol is not None else None
        ),
        "success": success,
        "word_match": word_match,
        "ja_success": last_play.get("success") if last_play is not None else None,
        "n_switches": len(recorder.switches),
        "frames": frames,
        "switches": recorder.switches,
    }
