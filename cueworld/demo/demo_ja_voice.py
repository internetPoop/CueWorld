"""
共同注意 1 回の映像に、親子の声の色の流れを重ねるデモ。

声は波形ではなく観測の 5 次元ベクトル。有声フレームは (v_1, v_2, v_3) に
最も近い音素 p0..p7 を、淡い色と記号で置く。凡例・単語・音量は動画に載せない。
上の帯が親、下の帯が幼児。床の空きに重ね、姿や対象には被せない。

対象の立方体は親子の正面から外し、見るときに首が左右へ振れる配置にする。

幼児の音素保持は親の 6 step より少し長くする（積算で角に届かせる）。
幼児の対象語は、親の「これ」＋対象語が言い終わった次の step から始める。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
from cueworld.video import images_to_video  # noqa: E402

# 積算で対角へ移るのに 4 step。5 音素 × 7 で角に届いたサンプルが窓に残る。
INFANT_PHONEME_HOLD = 7
INFANT_SILENCE_FRAMES = 12
# プロトコル終了後、幼児側の区間閉じが観測に残るまでの余韻。
TAIL_AFTER_DONE = 12
# 成功後の無音離脱・発話中の打ち切り。既定は 30 / 50 step で、親のラベリングが
# 終わる頃に post_shared を閉じる。このデモでは幼児の返答と「すごい」が
# 入るまで開けておく。視線成功の AttentionHoldTracker（30 step）は別物で、変えない。
POST_SHARED_HOLD_STEPS = 150
POST_SHARED_WAIT_STEPS = 150

LANE_BG = (28, 30, 36)
PLAYHEAD = (255, 255, 255)
# 1024 フレームの下端。足の下の床に載せ、姿には届かせない。
VOICE_BAND_H = 136
VOICE_BAND_GAP = 8
VOICE_BAND_BOTTOM = 12
VOICE_BAND_SIDE = 12
# この本数までは右端揃えで全部残す。超えた過去だけ左へ流す。
VOICE_HISTORY_STEPS = 220
# 親子は中段で向かい合う。立方体は -Y の端行（三人称カメラの奥）。
SLIDE_ARENA = (1.8, 2.0)
# 立方体（赤・緑・青）と親（黄）・幼児（青）の原色から外した淡い色。黒文字が乗る明るさ。
PHONEME_COLORS = {
    "p0": (214, 176, 138),
    "p1": (232, 154, 86),
    "p2": (176, 150, 208),
    "p3": (128, 184, 176),
    "p4": (154, 162, 176),
    "p5": (196, 150, 124),
    "p6": (204, 164, 188),
    "p7": (188, 176, 154),
}
# 左から p0。細くて似る ! | ~ は使わない。
PHONEME_SYMBOLS = {
    "p0": "#",
    "p1": "$",
    "p2": "%",
    "p3": "&",
    "p4": "*",
    "p5": "=",
    "p6": "?",
    "p7": "+",
}
PHONEME_INK = (24, 24, 28)
_PHONEME_FONT = None


def phoneme_font(size: int = 18) -> ImageFont.ImageFont:
    global _PHONEME_FONT
    if _PHONEME_FONT is not None and getattr(_PHONEME_FONT, "size", None) == size:
        return _PHONEME_FONT
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/local/share/fonts/ipag.ttf",
    )
    font = None
    for path in candidates:
        if os.path.isfile(path):
            font = ImageFont.truetype(path, size=size)
            break
    if font is None:
        font = ImageFont.load_default()
    _PHONEME_FONT = font
    return font
def parse_args():
    parser = argparse.ArgumentParser(description="JA video with parent/infant voice roll")
    parser.add_argument(
        "--views",
        type=str,
        default="infant,third",
        help="Comma-separated views: infant,third",
    )
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def slide_object_placements() -> list[dict]:
    """赤を正面から外し、緑と青は同じ奥の列で左右に置く。"""
    yh = SLIDE_ARENA[1]
    z = 0.06
    return [
        {"class": "Red_Cube", "x": 0.0, "y": -yh + 0.28, "z": z},
        {"class": "Green_Cube", "x": -0.42, "y": -yh + 0.42, "z": z},
        {"class": "Blue_Cube", "x": 0.42, "y": -yh + 0.42, "z": z},
    ]


def voice_pair_from_obs(obs) -> tuple[np.ndarray, np.ndarray]:
    infant_obs = obs[0] if isinstance(obs, (tuple, list)) else obs
    parent = np.asarray(infant_obs["parent_voice"], dtype=np.float32)
    infant = np.asarray(infant_obs["infant_voice"], dtype=np.float32)
    return parent, infant


def nearest_phoneme(voice: np.ndarray, mapping: dict, threshold: float) -> str | None:
    """有声なら (v_1, v_2, v_3) に最も近い音素。無声は None。"""
    if float(voice[0]) < float(threshold):
        return None
    # enable_emotion=False は 4 次元（volume の直後が v_1）。True は pitch を挟んで 5 次元。
    start = 1 if int(np.asarray(voice).shape[0]) <= 4 else 2
    feat = np.asarray(voice[start:start + 3], dtype=np.float32)
    best_name = None
    best_dist = 1e9
    for name in sorted(mapping.keys()):
        dist = float(np.linalg.norm(feat - np.asarray(mapping[name], dtype=np.float32)))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def find_signal_detector(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "word_history") and hasattr(current, "voice_recognizer"):
            return current
        current = getattr(current, "env", None)
    return None


def template_word_spans(template_key: str, concept_id: str | None, language_manager) -> list[tuple[int, int, str]]:
    """テンプレートを展開したときの単語区間 [start, end) をフレーム単位で返す。"""
    template_list = PROTOCOL_TEMPLATES.get("joint_attention", {}).get(template_key)
    if not template_list:
        template_list = PROTOCOL_TEMPLATES.get("common", {}).get(template_key)
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
        spans.append((frame, frame + length, language_manager.get_human_word(resolved)))
        frame += length
        words_emitted += 1
    return spans


def build_infant_action(
    env,
    protocol,
    target_obj,
    speech_queue: deque,
    speech_started: list,
    speech_ready: bool,
) -> np.ndarray:
    """合図中は対象を見る。親のラベリングが終わってから対象語を言う。"""
    infant_action = np.zeros(env.action_space[0].shape, dtype=np.float32)
    state = getattr(protocol, "state", None) if protocol else None
    cue = state in ("looking", "pointing", "shaking")
    after = state in ("post_shared_attention", "completed")
    if cue and target_obj is not None:
        env.gaze_object(env.infant, target_obj, yaw_link="head", action=infant_action)
    elif after:
        env.gaze_agent(env.infant, env.parent, yaw_link="head", action=infant_action)
    elif target_obj is not None:
        env.gaze_object(env.infant, target_obj, yaw_link="head", action=infant_action)

    if speech_ready and not speech_started[0] and len(speech_queue) == 0:
        concept_id = _target_concept_id(env, target_obj)
        if concept_id is not None:
            speech_queue.extend(
                build_word_phoneme_queue(
                    concept_id,
                    duration=INFANT_PHONEME_HOLD,
                    silence_frames=INFANT_SILENCE_FRAMES,
                )
            )
            speech_started[0] = True
            print(f"[infant] queued word {concept_id} hold={INFANT_PHONEME_HOLD}")
    if len(speech_queue) > 0 or speech_started[0]:
        apply_queued_phoneme(env, infant_action, speech_queue)
    return normalize_infant_action(env, infant_action)


def paint_voice_bands(
    camera: np.ndarray,
    parent_phonemes: list[str | None],
    infant_phonemes: list[str | None],
    out: int = 1024,
    history_steps: int = VOICE_HISTORY_STEPS,
) -> np.ndarray:
    """カメラ画像の下端に、親（上）と幼児（下）の音素を描く。

    同じ音素の連続は一つの短冊にし、左端に記号を一つ置く。
    """
    cam = Image.fromarray(camera).convert("RGB")
    cam = cam.resize((out, out), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(cam)
    side = VOICE_BAND_SIDE
    plot_w = max(1, out - 2 * side)
    px = max(4, plot_w // max(1, int(history_steps)))
    n = len(parent_phonemes)
    visible = max(1, plot_w // px)
    visible_start = max(0, n - visible)
    shown = min(n, visible)
    x_origin = side + (plot_w - shown * px)

    infant_y = out - VOICE_BAND_BOTTOM - VOICE_BAND_H
    parent_y = infant_y - VOICE_BAND_GAP - VOICE_BAND_H

    font = phoneme_font(22)

    def draw_lane(y: int, phonemes: list[str | None]) -> None:
        draw.rectangle([side, y, out - side, y + VOICE_BAND_H], fill=LANE_BG)
        frame_i = 0
        while frame_i < n:
            phoneme = phonemes[frame_i]
            run_end = frame_i + 1
            while run_end < n and phonemes[run_end] == phoneme:
                run_end += 1
            if phoneme in PHONEME_COLORS and run_end > visible_start and frame_i < n:
                vis_lo = max(frame_i, visible_start)
                vis_hi = min(run_end, n)
                x0 = x_origin + (vis_lo - visible_start) * px
                x1 = x_origin + (vis_hi - visible_start) * px - 1
                draw.rectangle(
                    [x0, y + 4, x1, y + VOICE_BAND_H - 4],
                    fill=PHONEME_COLORS[phoneme],
                )
                label = PHONEME_SYMBOLS.get(phoneme)
                glyph_x = x_origin + (frame_i - visible_start) * px + 3
                if label and glyph_x >= side and glyph_x < out - side:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    gw = bbox[2] - bbox[0]
                    gh = bbox[3] - bbox[1]
                    run_w = (run_end - frame_i) * px
                    if gw <= run_w - 2:
                        gy = y + (VOICE_BAND_H - gh) / 2 - bbox[1]
                        draw.text((glyph_x, gy), label, fill=PHONEME_INK, font=font)
            frame_i = run_end
        if n > 0:
            play_x = x_origin + (n - 1 - visible_start) * px
            draw.line([(play_x, y), (play_x, y + VOICE_BAND_H)], fill=PLAYHEAD, width=2)

    draw_lane(parent_y, parent_phonemes)
    draw_lane(infant_y, infant_phonemes)
    return np.asarray(cam)


def infant_word_span(phonemes: list[str | None], end_index: int) -> tuple[int, int] | None:
    """認識が乗ったフレームから、直前の有声区間 [start, end) を返す。"""
    i = end_index
    while i >= 0 and phonemes[i] is None:
        i -= 1
    if i < 0:
        return None
    end = i + 1
    while i >= 0 and phonemes[i] is not None:
        i -= 1
    return i + 1, end


def main():
    args = parse_args()
    view_set = {v.strip() for v in args.views.split(",") if v.strip()}
    print(
        f"Starting JA voice demo (views={sorted(view_set)}, "
        f"infant_hold={INFANT_PHONEME_HOLD}, silence={INFANT_SILENCE_FRAMES})..."
    )
    placements = slide_object_placements()
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=len(placements),
        object_counts={},
        max_steps=max(args.max_frames + 100, 700),
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
    env = ProtocolManager(env)
    # _handle_post_shared_attention が実行時に読む名前だけを、このプロセスで差し替える。
    ja_protocol.ATTENTION_HOLD_STEPS = POST_SHARED_HOLD_STEPS
    ja_protocol.ATTENTION_WAIT_STEPS = POST_SHARED_WAIT_STEPS

    obs, info = env.reset(seed=args.seed)
    target_obj = None
    for obj_id, obj in env.all_objects.items():
        if obj.__class__.__name__ == "RedCube":
            print(f"Found target object: {obj.name} (ID: {obj_id})")
            target_obj = obj_id
            break
    context = {}
    if target_obj is not None:
        context = {
            "fixed_target": target_obj,
            "cue_type": "looking",
            "intent_type": "informative",
        }
    env.set_fixed_protocol("joint_attention", context)

    detector = find_signal_detector(env)
    threshold = float(getattr(env, "voice_volume_threshold", 0.25))
    mapping = getattr(env, "phoneme_mapping", {})
    language_manager = env.current_protocol.language_manager

    parent_phonemes: list[str | None] = []
    infant_phonemes: list[str | None] = []
    parent_words: list[dict] = []
    infant_words: list[dict] = []
    movies = {k: [] for k in ("infant", "third") if k in view_set}

    prev_utt_key = None
    prev_word_count = len(detector.word_history) if detector is not None else 0
    speech_queue: deque = deque()
    speech_started = [False]
    speech_ready = False
    declare_end = None
    parent_action = np.zeros(env.action_space[1].shape, dtype=np.float32)
    reached_post_shared = False
    tail = None

    def append_voices(obs_now) -> None:
        parent_voice, infant_voice = voice_pair_from_obs(obs_now)
        parent_phonemes.append(nearest_phoneme(parent_voice, mapping, threshold))
        infant_phonemes.append(nearest_phoneme(infant_voice, mapping, threshold))

    def capture(info_now) -> None:
        if "infant" in movies:
            frame = raw_view_from_info(info_now or {}, "infant_rgb")
            if frame is not None:
                movies["infant"].append(paint_voice_bands(frame, parent_phonemes, infant_phonemes))
        if "third" in movies:
            third = get_third_person_view(env, orbit_deg=32, distance=3.1)
            if third is not None:
                movies["third"].append(paint_voice_bands(third, parent_phonemes, infant_phonemes))

    def note_infant_word() -> None:
        nonlocal prev_word_count
        if detector is None or len(detector.word_history) <= prev_word_count:
            return
        word = detector.word_history[-1]
        prev_word_count = len(detector.word_history)
        if not word:
            print(f"t={t:03d} infant recognition returned None")
            return
        span = infant_word_span(infant_phonemes, len(infant_phonemes) - 1)
        label = language_manager.get_human_word(str(word))
        if span is not None:
            infant_words.append({"start": span[0], "end": span[1], "label": label})
        print(f"t={t:03d} infant recognized {word} ({label}) span={span}")

    append_voices(obs)
    capture(info)

    t = 0
    while t < args.max_frames:
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        if state == "post_shared_attention":
            reached_post_shared = True
        q_before = len(getattr(protocol, "utterance_queue", [])) if protocol is not None else 0

        infant_action = build_infant_action(
            env, protocol, target_obj, speech_queue, speech_started, speech_ready
        )
        if isinstance(info, dict) and "parent_action" in info:
            parent_action = info["parent_action"]
        obs, reward, done, trunc, info = env.step((infant_action, parent_action))

        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        utt = {}
        if protocol is not None:
            utt = (getattr(protocol, "debug_info", {}) or {}).get("last_utterance") or {}
        utt_key = (utt.get("template_key"), utt.get("concept_id"))
        append_voices(obs)
        if utt.get("template_key") and utt_key != prev_utt_key and protocol is not None:
            spans = template_word_spans(utt_key[0], utt_key[1], language_manager)
            # append 済みの今フレームが、空だったキューなら新発話の先頭。
            origin = (len(parent_phonemes) - 1) + q_before
            for start, end, label in spans:
                parent_words.append({"start": origin + start, "end": origin + end, "label": label})
            if utt_key[0] == "declare_object" and spans:
                declare_end = origin + spans[-1][1]
            prev_utt_key = utt_key
            print(
                f"t={t:03d} parent utterance {utt_key[0]} "
                f"words={[label for _, _, label in spans]}"
            )
        note_infant_word()
        if declare_end is not None and len(parent_phonemes) >= declare_end:
            speech_ready = True
        capture(info)

        if t % 50 == 0:
            print(
                f"t={t:03d} state={state} ja_success={getattr(protocol, 'ja_success', None)} "
                f"queue={len(getattr(protocol, 'utterance_queue', []))} "
                f"speech_left={len(speech_queue)} done={done}"
            )

        # 幼児の語が終わっても、称賛のキューが空になる（done）まで回す。
        infant_finished = speech_started[0] and len(speech_queue) == 0
        if infant_finished and (done or trunc):
            if tail is None:
                tail = TAIL_AFTER_DONE
            tail -= 1
            if tail <= 0:
                print(f"Stop at t={t} after infant utterance.")
                break
        if t >= args.max_frames - 1:
            break
        t += 1

    protocol = getattr(env, "current_protocol", None)
    print("===== Summary =====")
    print(f"  final_state    : {getattr(protocol, 'state', None)}")
    print(f"  ja_success     : {getattr(protocol, 'ja_success', None)}")
    print(f"  word_match     : {getattr(protocol, 'word_match', None)}")
    print(f"  reached_post   : {reached_post_shared}")
    print(f"  infant_spoke   : {speech_started[0]}")
    print(f"  parent_words   : {[w['label'] for w in parent_words]}")
    print(f"  infant_words   : {[w['label'] for w in infant_words]}")
    print(f"  frames         : {t + 1}")
    print("===================")
    env.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    for view_name, frames in movies.items():
        if not frames:
            print(f"[warn] no frames for view={view_name}")
            continue
        if view_name == "infant":
            video_path = f"{save_dir}/demo_ja_voice.mp4"
        else:
            video_path = f"{save_dir}/demo_ja_voice_{view_name}.mp4"
        print(f"Saving video to {video_path} ({len(frames)} frames)...")
        images_to_video(frames, video_path, fps=12, scale=1)
    print("Done!")


if __name__ == "__main__":
    main()
