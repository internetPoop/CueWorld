"""
共同注意を「見るだけ」から始め、気づかれなければ指差しに上がるデモ。

幼児は指差しが始まるまで親を見て、対象は見ない。
見るだけの約 5 秒が空振りすると、プロトコルが指差しに切り替える。
指差しが始まったら対象を見る。親が言い終わってから少し間を空けて、対象語を言う。
視線が約 3 秒続けば成功する。

声は淡い色と記号で床に重ねる。上の帯が親、下の帯が幼児。

見るだけの状態でも _determine_action は point_object を呼ぶ（指差し以外の
キューは未対応の TODO）。このデモでは、その呼び出しだけを見てるあいだ
飛ばし、腕は下ろしたまま頭だけ向ける。プロトコルの遷移自体は変えない。
"""
from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from camera_views import get_third_person_view  # noqa: E402
from cueworld.paths import ABS_PATH  # noqa: E402
from cueworld.base import BaseEnv  # noqa: E402
from cueworld.infant_signal import InfantSignalDetector  # noqa: E402
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
TAIL_AFTER_DONE = 12
# 指差しのあと、親の合図が終わって幼児が対象語を言い終わるまで、成功後の待ちを開けておく。
# 視線成功の AttentionHoldTracker（30 step）は別物で、変えない。
# 音の長さは demo_ja_voice と同じ。開始だけ、親が無音になってから約 1 秒空ける。
PARENT_PAUSE_STEPS = 12
POST_SHARED_HOLD_STEPS = 200
POST_SHARED_WAIT_STEPS = 200
VOICE_HISTORY_STEPS = 280


def slide_object_placements() -> list[dict]:
    """赤を正面から外し、緑と青は同じ奥の列で左右に置く。"""
    yh = SLIDE_ARENA[1]
    z = 0.06
    return [
        {"class": "Red_Cube", "x": 0.0, "y": -yh + 0.28, "z": z},
        {"class": "Green_Cube", "x": -0.42, "y": -yh + 0.42, "z": z},
        {"class": "Blue_Cube", "x": 0.42, "y": -yh + 0.42, "z": z},
    ]


def skip_point_while_looking(manager) -> None:
    """state が looking のときだけ、親の指差しを出さない。

    プロトコルが持っているのは ProtocolManager の内側の env なので、
    そちらに付ける。
    """
    protocol = manager.current_protocol
    inner = protocol.env
    point = inner.point_object

    def point_object(agent, oid, action=None, ik_every=0):
        if getattr(protocol, "state", None) == "looking":
            return action
        return point(agent, oid, action=action, ik_every=ik_every)

    inner.point_object = point_object


def build_infant_action(
    env,
    protocol,
    target_obj,
    speech_queue: deque,
    speech_started: list,
    speech_ready: bool,
) -> np.ndarray:
    """指差しまでは親を見る。指差しが始まったら対象を見て、間を置いて対象語を言う。"""
    infant_action = np.zeros(env.action_space[0].shape, dtype=np.float32)
    state = getattr(protocol, "state", None) if protocol else None
    if state in ("pointing", "post_shared_attention", "completed") and target_obj is not None:
        env.gaze_object(env.infant, target_obj, yaw_link="head", action=infant_action)
    else:
        env.gaze_agent(env.infant, env.parent, yaw_link="head", action=infant_action)

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


def main(stem: str = "demo_ja_look_to_point"):
    print(f"Starting look-to-point JA demo ({stem})...")
    placements = slide_object_placements()
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=len(placements),
        object_counts={},
        max_steps=700,
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
    ja_protocol.ATTENTION_HOLD_STEPS = POST_SHARED_HOLD_STEPS
    ja_protocol.ATTENTION_WAIT_STEPS = POST_SHARED_WAIT_STEPS

    obs, info = env.reset(seed=42)
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
    skip_point_while_looking(env)

    movies = {"infant": [], "third": []}
    parent_action = np.zeros(env.action_space[1].shape, dtype=np.float32)
    prev_state = None
    prev_cue = None
    saw_looking = False
    saw_pointing = False
    tail = None
    speech_queue: deque = deque()
    speech_started = [False]
    speech_ready = False
    heard_parent_after_point = False
    parent_silent = 0
    threshold = float(getattr(env, "voice_volume_threshold", 0.25))
    mapping = getattr(env, "phoneme_mapping", {})
    parent_phonemes: list[str | None] = []
    infant_phonemes: list[str | None] = []

    def append_voices(obs_now) -> None:
        parent_voice, infant_voice = voice_pair_from_obs(obs_now)
        parent_phonemes.append(nearest_phoneme(parent_voice, mapping, threshold))
        infant_phonemes.append(nearest_phoneme(infant_voice, mapping, threshold))

    def capture(info_now) -> None:
        frame = raw_view_from_info(info_now or {}, "infant_rgb")
        if frame is not None:
            movies["infant"].append(
                paint_voice_bands(
                    frame, parent_phonemes, infant_phonemes, history_steps=VOICE_HISTORY_STEPS
                )
            )
        third = get_third_person_view(env, orbit_deg=32, distance=3.1)
        if third is not None:
            movies["third"].append(
                paint_voice_bands(
                    third, parent_phonemes, infant_phonemes, history_steps=VOICE_HISTORY_STEPS
                )
            )

    append_voices(obs)
    capture(info)
    t = 0
    while t < 500:
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        cue = getattr(protocol, "cue_type", None) if protocol else None
        if state == "looking":
            saw_looking = True
        if state == "pointing":
            saw_pointing = True
        if state != prev_state or cue != prev_cue:
            print(f"t={t:03d} state {prev_state} -> {state} cue {prev_cue} -> {cue}")
            prev_state = state
            prev_cue = cue

        infant_action = build_infant_action(
            env, protocol, target_obj, speech_queue, speech_started, speech_ready
        )
        if isinstance(info, dict) and "parent_action" in info:
            parent_action = info["parent_action"]
        obs, reward, done, trunc, info = env.step((infant_action, parent_action))
        append_voices(obs)
        # 指差し以降に親の声を聞いてから、無音が PARENT_PAUSE_STEPS 続いたら話し始める。
        # 単語のあいだの 1 step では始まらない。
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        if (
            not speech_ready
            and state in ("pointing", "post_shared_attention", "completed")
        ):
            parent_voiced = bool(parent_phonemes) and parent_phonemes[-1] is not None
            if parent_voiced:
                heard_parent_after_point = True
                parent_silent = 0
            elif heard_parent_after_point:
                parent_silent += 1
                if parent_silent >= PARENT_PAUSE_STEPS:
                    speech_ready = True
                    print(f"t={t:03d} parent finished, pause={PARENT_PAUSE_STEPS}")
        capture(info)

        if t % 25 == 0:
            print(
                f"t={t:03d} state={state} cue={getattr(protocol, 'cue_type', None)} "
                f"ja_success={getattr(protocol, 'ja_success', None)} "
                f"speech_left={len(speech_queue)} done={done}"
            )
        infant_finished = speech_started[0] and len(speech_queue) == 0
        if infant_finished and (done or trunc):
            if tail is None:
                tail = TAIL_AFTER_DONE
            tail -= 1
            if tail <= 0:
                print(f"Stop at t={t}")
                break
        t += 1

    protocol = getattr(env, "current_protocol", None)
    print("===== Summary =====")
    print(f"  final_state  : {getattr(protocol, 'state', None)}")
    print(f"  cue_type     : {getattr(protocol, 'cue_type', None)}")
    print(f"  ja_success   : {getattr(protocol, 'ja_success', None)}")
    print(f"  word_match   : {getattr(protocol, 'word_match', None)}")
    print(f"  saw_looking  : {saw_looking}")
    print(f"  saw_pointing : {saw_pointing}")
    print(f"  infant_spoke : {speech_started[0]}")
    print(f"  frames       : {t + 1}")
    print("===================")
    env.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    for view_name, frames in movies.items():
        if not frames:
            print(f"[warn] no frames for view={view_name}")
            continue
        if view_name == "infant":
            video_path = f"{save_dir}/{stem}.mp4"
        else:
            video_path = f"{save_dir}/{stem}_{view_name}.mp4"
        print(f"Saving video to {video_path} ({len(frames)} frames)...")
        images_to_video(frames, video_path, fps=12, scale=1)
    print("Done!")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_ja_look_to_point")
