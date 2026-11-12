"""
バッチ E 作業順 4: ProtocolManager 自律切替の comprehensive デモ。

見守り中は沈黙と完成形（揃った指差し）を混ぜ、関わり中は幼児 3 本
（忠実に付き合う / 動かず失敗 / 別の完成形）で乗車・未乗車・割れを見る。
親は PM が自律制御。fixed_protocol は使わない。拒否パターンは入れない。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import deque
from typing import Optional

import numpy as np
import pybullet as p

from cueworld.paths import ABS_PATH
from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.language.vocabulary import CONCEPT_TO_PHONEMES
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager
from cueworld.video import images_to_video

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from camera_views import get_third_person_view  # noqa: E402
from demo_ja_shake import annotated_view_from_info, normalize_infant_action  # noqa: E402
from dwell_tracker import DwellTracker  # noqa: E402


def raw_view_from_info(info: dict, view_key: str) -> np.ndarray | None:
    if not isinstance(info, dict):
        return None
    views = info.get("views") or {}
    base = views.get(view_key)
    if base is None:
        return None
    return np.array(base, copy=True)

PLAY_PROTOCOLS = frozenset({"joint_attention", "object_giving", "dyadic"})
POST_HOUSEHOLD_IDLE = 40
RAISE_FORWARD_DIST = 0.35
ARENA_HALF_EXTENTS = (1.6, 1.2)
YH = ARENA_HALF_EXTENTS[1]


def household_fixed_placements() -> list[dict]:
    """家事候補用の board と、指差し対象の RGB キューブ。"""
    return [
        {"class": "Green_Board", "x": 0.4, "y": YH - 0.05, "z": 0.5, "yaw": 90.0},
        {"class": "Yellow_Board", "x": 1.2, "y": YH - 0.05, "z": 0.5, "yaw": 90.0},
        {"class": "Red_Board", "x": 0.4, "y": -YH + 0.05, "z": 0.5, "yaw": 90.0},
        {"class": "Blue_Board", "x": 1.2, "y": -YH + 0.05, "z": 0.5, "yaw": 90.0},
        {"class": "Red_Cube", "x": -1.25, "y": -YH + 0.05, "z": 0.5, "yaw": 0.0},
        {"class": "Green_Cube", "x": -0.75, "y": -YH + 0.05, "z": 0.5, "yaw": 0.0},
        {"class": "Blue_Cube", "x": -0.25, "y": -YH + 0.05, "z": 0.5, "yaw": 0.0},
    ]


def _find_cube(env, class_name: str) -> Optional[int]:
    for obj_id, obj in env.all_objects.items():
        if obj.__class__.__name__ == class_name:
            return obj_id
    return None


def _protocol_target_id(protocol) -> Optional[int]:
    if protocol is None:
        return None
    target = getattr(protocol, "target_object", None)
    if target is None:
        return None
    if hasattr(target, "id"):
        return target.id
    return target


def _target_name(env, target_id) -> str:
    if target_id is None:
        return "None"
    obj = env.all_objects.get(target_id)
    if obj is None:
        return str(target_id)
    return getattr(obj, "concept_id", getattr(obj, "name", str(target_id)))


def _household_remaining(env) -> int:
    interval = int(getattr(env, "household_interval_steps", 200))
    last = int(getattr(env, "last_household_end_step", 0))
    step = int(getattr(env, "step_count", 0))
    return max(0, interval - (step - last))


def _target_concept_id(env, target_id) -> Optional[str]:
    if target_id is None:
        return None
    obj = env.all_objects.get(target_id)
    if obj is None:
        return None
    return getattr(obj, "concept_id", None)


def build_word_phoneme_queue(
    concept_id: str,
    duration: int = 12,
    silence_frames: int = 12,
) -> deque:
    phonemes = CONCEPT_TO_PHONEMES.get(concept_id, [])
    queue: deque = deque()
    for phoneme in phonemes:
        for _ in range(int(duration)):
            queue.append(phoneme)
    for _ in range(int(silence_frames)):
        queue.append(None)
    return queue


def apply_queued_phoneme(env, infant_action: np.ndarray, speech_queue: deque) -> None:
    if len(speech_queue) == 0:
        return
    phoneme = speech_queue.popleft()
    if phoneme is None:
        if "volume" in env.action_map:
            infant_action[env.action_map["volume"]] = -1.0
        return
    mapping = getattr(env, "phoneme_mapping", {})
    if phoneme not in mapping:
        return
    env.apply_phoneme_to_action(infant_action, mapping[phoneme], volume=1.0)


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


def _get_infant_reach_action(env, agent, target_obj_id, action, reaching_phase: bool):
    if not reaching_phase:
        yaw, pitch = env.gaze_object(agent, target_obj_id, yaw_link="torso", action=action)
        if abs(yaw) < math.radians(5.0):
            reaching_phase = True
    else:
        env.hand_reach_object(agent, target_obj_id, action=action)
        env.gaze_object(agent, target_obj_id, yaw_link="head", action=action)
    return action, reaching_phase


def neutralize_appeal(signals: dict, env) -> dict:
    """沈黙区間で、直前モータの履歴が訴えとして残らないようにする。"""
    out = dict(signals)
    out["point_intensity"] = 0.0
    out["open_intensity"] = 0.0
    out["hand_target_stability"] = 0.0
    out["hand_height_margin"] = 0.0
    out["hand_open_close_periodicity"] = 0.0
    out["babbling_ratio"] = 0.0
    out["voice_volume"] = 0.0
    n = len(getattr(env, "trackable_targets", []) or [])
    out["gesture_targets"] = np.zeros(n, dtype=float)
    return out


def inject_complete_pointing(signals: dict, env, target_id: Optional[int]) -> dict:
    """揃った指差しをデモ専用に立てる（J2 と同じ注入）。"""
    if target_id is None:
        return signals
    out = dict(signals)
    gesture_targets = np.zeros(len(env.trackable_targets), dtype=float)
    if target_id in env.trackable_targets:
        gesture_targets[env.trackable_targets.index(target_id)] = 1.0
    out["gesture_targets"] = gesture_targets
    out["point_intensity"] = 1.0
    out["hand_target_stability"] = 1.0
    return out


def install_signal_override(manager: ProtocolManager, runtime: dict) -> None:
    detector = manager.env
    orig = detector.get_normalized_signals

    def wrapped(ignored_targets=None):
        sigs = orig(ignored_targets=ignored_targets)
        mode = runtime.get("mode") or "idle"
        if mode == "idle":
            return neutralize_appeal(sigs, manager)
        target_id = runtime.get("inject_id")
        if target_id is not None:
            return inject_complete_pointing(sigs, manager, target_id)
        return sigs

    detector.get_normalized_signals = wrapped


class InfantScheduler:
    """見守り沈黙 → 家事 → 短い沈黙 → 指差し。関わり中は 3 本の pattern。"""

    def __init__(self, pattern: str, post_hh_idle: int = POST_HOUSEHOLD_IDLE):
        self.pattern = pattern
        self.post_hh_idle = post_hh_idle
        self.seen_household = False
        self.post_hh_idle_count = 0
        self.mode = "idle"

    def update(self, protocol_name: str) -> str:
        if protocol_name == "household_solo":
            self.seen_household = True
            self.mode = "idle"
            return self.mode
        if protocol_name in PLAY_PROTOCOLS:
            if self.pattern == "engage":
                self.mode = "engage"
            elif self.pattern == "other_complete":
                self.mode = "other_complete"
            else:
                self.mode = "idle"
            return self.mode
        if not self.seen_household:
            self.mode = "idle"
            return self.mode
        self.post_hh_idle_count += 1
        if self.post_hh_idle_count < self.post_hh_idle:
            self.mode = "idle"
            return self.mode
        self.mode = "complete_form"
        return self.mode


def build_infant_action(
    env,
    protocol,
    *,
    mode: str,
    primary_id: Optional[int],
    other_id: Optional[int],
    reaching_phase: bool,
    speech_queue: deque,
    speech_started: list,
) -> tuple[np.ndarray, bool]:
    infant_action = np.zeros(env.action_space[0].shape, dtype=np.float32)
    env.reset_arm_pose(agent=env.infant, action=infant_action)
    protocol_name = getattr(env, "current_protocol_name", "observance") or "observance"
    protocol_state = getattr(protocol, "state", None) if protocol else None
    play_target = _protocol_target_id(protocol) or primary_id

    if mode == "idle":
        return normalize_infant_action(env, infant_action), reaching_phase

    if mode == "complete_form":
        if primary_id is not None:
            env.gaze_object(env.infant, primary_id, yaw_link="head", action=infant_action)
            env.point_object(env.infant, primary_id, action=infant_action)
        return normalize_infant_action(env, infant_action), reaching_phase

    if mode == "other_complete":
        if other_id is not None:
            env.gaze_object(env.infant, other_id, yaw_link="head", action=infant_action)
            env.point_object(env.infant, other_id, action=infant_action)
        return normalize_infant_action(env, infant_action), reaching_phase

    # engage: 今のプロトコルに付き合う（JA は対象を見続ける＝乗車）
    if protocol_name == "joint_attention":
        in_response = protocol_state in ("completed", "post_shared_attention")
        if play_target is not None:
            env.gaze_object(env.infant, play_target, yaw_link="head", action=infant_action)
        if in_response:
            if (
                protocol_state == "post_shared_attention"
                and not speech_started[0]
                and len(speech_queue) == 0
            ):
                concept_id = _target_concept_id(env, play_target)
                if concept_id is not None:
                    speech_queue.extend(build_word_phoneme_queue(concept_id))
                    speech_started[0] = True
            apply_queued_phoneme(env, infant_action, speech_queue)

    elif protocol_name == "object_giving":
        if play_target is not None:
            if protocol_state in ("init", "fetching", "approach", "showing", "placing"):
                infant_action, reaching_phase = _get_infant_reach_action(
                    env,
                    env.infant,
                    play_target,
                    action=infant_action,
                    reaching_phase=reaching_phase,
                )
            elif protocol_state == "waiting_infant":
                env.grasp_object(
                    env.infant,
                    play_target,
                    action=infant_action,
                    approach_dist=float(env.grasp_approach_dist),
                )
            else:
                env.gaze_object(
                    env.infant, play_target, yaw_link="head", action=infant_action
                )

    elif protocol_name == "dyadic":
        env.gaze_agent(env.infant, env.parent, yaw_link="head", action=infant_action)
        apply_infant_raise(env, infant_action)

    return normalize_infant_action(env, infant_action), reaching_phase


def build_overlay_lines(
    env,
    protocol,
    *,
    pattern: str,
    mode: str,
    primary_id: Optional[int],
    other_id: Optional[int],
) -> list[str]:
    lines = [
        f"Pattern: {pattern}",
        f"Mode: {mode}",
        f"Persistence: {getattr(protocol, 'persistence', None)}",
        f"InfantInit: {getattr(env, 'current_infant_initiative', False)}",
        f"HHDueIn: {_household_remaining(env)}",
        f"Primary: {_target_name(env, primary_id)}",
        f"Other: {_target_name(env, other_id)}",
    ]
    target_id = _protocol_target_id(protocol)
    if target_id is not None:
        lines.append(f"ProtoTarget: {_target_name(env, target_id)}")
    return lines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive autonomous protocol-switch demo (batch E)"
    )
    parser.add_argument(
        "--infant_pattern",
        type=str,
        default="engage",
        choices=["engage", "idle_fail", "other_complete"],
        help="Infant script during interaction protocols",
    )
    parser.add_argument(
        "--views",
        type=str,
        default="infant,third",
        help="Comma-separated views to save: infant,parent,third",
    )
    parser.add_argument("--max_frames", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_annotate",
        action="store_true",
        help="Save raw camera frames without overlay text",
    )
    parser.add_argument(
        "--inject_pointing",
        dest="inject_pointing",
        action="store_true",
        help="Inject complete pointing signals (default on)",
    )
    parser.add_argument(
        "--no_inject_pointing",
        dest="inject_pointing",
        action="store_false",
        help="Use physical pointing only",
    )
    parser.set_defaults(inject_pointing=True)
    return parser.parse_args()


def _household_candidates(env) -> list:
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        cands = getattr(current, "household_task_candidates", None)
        if cands:
            return list(cands)
        current = getattr(current, "env", None)
    return []


def main():
    args = parse_args()
    view_set = {v.strip() for v in args.views.split(",") if v.strip()}
    print(
        f"Starting comprehensive demo "
        f"(pattern={args.infant_pattern}, views={sorted(view_set)}, "
        f"inject_pointing={args.inject_pointing}, no_annotate={args.no_annotate})..."
    )

    placements = household_fixed_placements()
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=len(placements),
        object_counts={},
        max_steps=max(args.max_frames + 100, 1700),
        return_views=True,
        output_parent_view=("parent" in view_set),
        view_width=256,
        view_height=256,
        action_repeat=1,
        arena_half_extents=ARENA_HALF_EXTENTS,
        arena_rows=4,
        arena_cols=5,
        fixed_placements=placements,
        debug_info=True,
        placement_method="grid",
        use_cinematic_model=True,
    )

    env = ParentTools(base)
    env = InfantSignalDetector(env)
    env = ProtocolManager(env, household_interval_steps=200, rejection=False)

    movies = {k: [] for k in ("infant", "parent", "third") if k in view_set}
    obs, info = env.reset(seed=args.seed)

    primary_id = _find_cube(env, "RedCube")
    other_id = _find_cube(env, "GreenCube")
    hh_cands = _household_candidates(env)
    print(
        f"Primary (complete form): {_target_name(env, primary_id)} "
        f"Other: {_target_name(env, other_id)}"
    )
    print(f"Household candidates: {[c.get('task_type') for c in hh_cands]}")
    if not hh_cands:
        print("[warn] no household_task_candidates; household clock will not fire")

    scheduler = InfantScheduler(args.infant_pattern)
    runtime: dict = {"mode": "idle", "inject_id": None}
    install_signal_override(env, runtime)

    dwell = DwellTracker()
    parent_action = np.zeros(env.action_space[1].shape, dtype=np.float32)
    reaching_phase = False
    speech_queue: deque = deque()
    speech_started = [False]
    last_protocol_name = getattr(env, "current_protocol_name", "observance")
    switches: list[str] = []
    saw_household = False
    saw_infant_ja = False

    def capture_frames(info_dict, mode: str):
        if args.no_annotate:
            if "infant" in movies:
                frame = raw_view_from_info(info_dict or {}, "infant_rgb")
                if frame is not None:
                    movies["infant"].append(frame)
            if "parent" in movies:
                frame = raw_view_from_info(info_dict or {}, "parent_rgb")
                if frame is not None:
                    movies["parent"].append(frame)
            if "third" in movies:
                third = get_third_person_view(env)
                if third is not None:
                    movies["third"].append(third)
            return

        protocol = getattr(env, "current_protocol", None)
        extra = build_overlay_lines(
            env,
            protocol,
            pattern=args.infant_pattern,
            mode=mode,
            primary_id=primary_id,
            other_id=other_id,
        )
        if "infant" in movies and "views" in (info_dict or {}):
            annot = annotated_view_from_info(
                info_dict, env, "infant_rgb", text_scale=0.1, extra_lines=extra
            )
            if annot is not None:
                movies["infant"].append(annot)
        if "parent" in movies and "views" in (info_dict or {}):
            annot = annotated_view_from_info(
                info_dict, env, "parent_rgb", text_scale=0.1, extra_lines=extra
            )
            if annot is not None:
                movies["parent"].append(annot)
        if "third" in movies:
            third = get_third_person_view(env)
            if third is not None:
                annot = annotated_view_from_info(
                    info_dict or {},
                    env,
                    image=third,
                    text_scale=0.1,
                    extra_lines=extra,
                )
                if annot is not None:
                    movies["third"].append(annot)

    protocol_name = getattr(env, "current_protocol_name", "observance")
    mode = scheduler.update(protocol_name)
    capture_frames(info, mode)

    t = 0
    while t < args.max_frames:
        protocol_name = getattr(env, "current_protocol_name", "observance") or "observance"
        protocol = getattr(env, "current_protocol", None)
        mode = scheduler.update(protocol_name)

        runtime["mode"] = mode
        if args.inject_pointing and mode == "complete_form":
            runtime["inject_id"] = primary_id
        elif args.inject_pointing and mode == "other_complete":
            runtime["inject_id"] = other_id
        else:
            runtime["inject_id"] = None

        if protocol_name != last_protocol_name:
            switches.append(f"t={t}:{last_protocol_name}->{protocol_name}")
            print(
                f"Protocol switch at t={t}: {last_protocol_name} -> {protocol_name} "
                f"(initiative={getattr(env, 'current_infant_initiative', False)})"
            )
            last_protocol_name = protocol_name
            reaching_phase = False
            speech_queue.clear()
            speech_started[0] = False
        if protocol_name == "household_solo":
            saw_household = True
        if (
            protocol_name == "joint_attention"
            and bool(getattr(env, "current_infant_initiative", False))
        ):
            saw_infant_ja = True

        infant_action, reaching_phase = build_infant_action(
            env,
            protocol,
            mode=mode,
            primary_id=primary_id,
            other_id=other_id,
            reaching_phase=reaching_phase,
            speech_queue=speech_queue,
            speech_started=speech_started,
        )

        if "parent_action" in info:
            parent_action = info["parent_action"]

        obs, reward, done, trunc, info = env.step((infant_action, parent_action))
        protocol = getattr(env, "current_protocol", None)
        protocol_name = getattr(env, "current_protocol_name", "observance") or "observance"
        state = getattr(protocol, "state", None) if protocol else None
        dwell.update(f"{protocol_name}:{state}")
        dwell.observe_protocol(protocol)
        capture_frames(info, mode)

        if t % 50 == 0:
            persistence = getattr(protocol, "persistence", None)
            print(
                f"t={t:04d}, protocol={protocol_name}, state={state}, mode={mode}, "
                f"persist={persistence}, init={getattr(env, 'current_infant_initiative', False)}, "
                f"hh_due_in={_household_remaining(env)}"
            )

        if done or trunc:
            print(f"Episode finished early at t={t} (done={done}, trunc={trunc}).")
            break
        t += 1

    protocol = getattr(env, "current_protocol", None)
    final_state = getattr(protocol, "state", "None") if protocol else "None"
    final_name = getattr(env, "current_protocol_name", "None")
    success = 1 if final_state == "completed" else 0

    print("===== Summary =====")
    print(f"  pattern             : {args.infant_pattern}")
    print(f"  final_protocol      : {final_name}")
    print(f"  final_state         : {final_state}")
    print(f"  success             : {success}")
    print(f"  saw_household       : {saw_household}")
    print(f"  saw_infant_ja       : {saw_infant_ja}")
    print(f"  switches            : {switches}")
    print(f"  frames              : {t}")
    print("===================")
    dwell.print_summary(
        seed=args.seed,
        success=success,
        run=f"comprehensive_{args.infant_pattern}",
    )

    env.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    pattern = args.infant_pattern
    for view_name, frames in movies.items():
        if not frames:
            print(f"[warn] no frames for view={view_name}")
            continue
        if view_name == "infant":
            video_path = f"{save_dir}/demo_comprehensive_{pattern}.mp4"
        else:
            video_path = f"{save_dir}/demo_comprehensive_{pattern}_{view_name}.mp4"
        print(f"Saving video to {video_path} ({len(frames)} frames)...")
        images_to_video(frames, video_path, fps=12, scale=4)
    print("Done!")


if __name__ == "__main__":
    main()
