"""
JointAttentionProtocol 幼児始動デモ（J2）。

幼児が親を見る -> 対象を見る / 指す -> 親が追従する、という
再現性優先の scripted パターンを固定プロトコルで再生する。
"""

import argparse
import os

import numpy as np

from cueworld.paths import ABS_PATH
from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager
from dwell_tracker import DwellTracker
from demo_ja_shake import (
    annotated_view_from_info,
    build_status_lines,
    get_third_person_view,
    normalize_infant_action,
)
from cueworld.video import images_to_video


def parse_args():
    parser = argparse.ArgumentParser(description="JointAttention infant-initiated demo")
    parser.add_argument(
        "--infant_pattern",
        type=str,
        default="gaze",
        choices=["gaze", "point", "speak"],
        help="Infant behavior pattern",
    )
    parser.add_argument(
        "--views",
        type=str,
        default="infant,third",
        help="Comma-separated views to save: infant,parent,third",
    )
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _target_name(env, target_id):
    if target_id is None:
        return "None"
    if target_id == env.parent["body"]:
        return "parent"
    obj = env.all_objects.get(target_id)
    if obj is None:
        return str(target_id)
    return getattr(obj, "concept_id", getattr(obj, "name", str(target_id)))


def _signal_argmax_target(env, targets, *, ignore_parent=False):
    if targets is None or len(targets) == 0:
        return None
    target_array = np.asarray(targets, dtype=float)
    if target_array.size == 0 or float(np.max(target_array)) <= 0.0:
        return None
    sorted_indices = np.argsort(target_array)[::-1]
    for idx in sorted_indices:
        if target_array[idx] <= 0.0:
            break
        if idx >= len(env.trackable_targets):
            continue
        target_id = env.trackable_targets[int(idx)]
        if ignore_parent and target_id == env.parent["body"]:
            continue
        return target_id
    return None


def build_infant_action(env, protocol, target_obj, pattern: str, t: int, warmup_steps: int):
    infant_action = np.zeros(env.action_space[0].shape, dtype=np.float32)
    env.reset_arm_pose(agent=env.infant, action=infant_action)

    if target_obj is None:
        return normalize_infant_action(env, infant_action)

    if t < warmup_steps:
        env.gaze_agent(env.infant, env.parent, yaw_link="head", action=infant_action)
        return normalize_infant_action(env, infant_action)

    env.gaze_object(env.infant, target_obj, yaw_link="head", action=infant_action)

    if pattern in {"point", "speak"}:
        env.point_object(env.infant, target_obj, action=infant_action)

    protocol_state = getattr(protocol, "state", None) if protocol else None
    if pattern == "speak" and protocol_state == "post_shared_attention":
        if hasattr(env, "phoneme_mapping") and "p0" in env.phoneme_mapping:
            env.apply_phoneme_to_action(
                infant_action, env.phoneme_mapping["p0"], volume=0.7
            )

    return normalize_infant_action(env, infant_action)


def build_context_infant_signals(env, info, target_obj, pattern: str):
    signals = dict(info.get("infant_signals", {}) if isinstance(info, dict) else {})
    if pattern != "point" or target_obj is None:
        return signals

    gesture_targets = np.zeros(len(env.trackable_targets), dtype=float)
    if target_obj in env.trackable_targets:
        gesture_targets[env.trackable_targets.index(target_obj)] = 1.0
    signals["gesture_targets"] = gesture_targets
    signals["point_intensity"] = 1.0
    signals["hand_target_stability"] = 1.0
    return signals


def main():
    args = parse_args()
    view_set = {v.strip() for v in args.views.split(",") if v.strip()}
    print(
        f"Starting JA infant-initiated demo "
        f"(pattern={args.infant_pattern}, views={sorted(view_set)})..."
    )

    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=3,
        object_counts={"RedCube": 1, "GreenCube": 1, "BlueCube": 1},
        max_steps=max(args.max_frames + args.warmup_steps + 100, 700),
        return_views=True,
        output_parent_view=("parent" in view_set),
        view_width=256,
        view_height=256,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        debug_info=True,
        placement_method="grid",
        use_cinematic_model=True,
    )

    env = ParentTools(base)
    env = InfantSignalDetector(env)
    env = ProtocolManager(env)

    movies = {k: [] for k in ("infant", "parent", "third") if k in view_set}
    obs, info = env.reset(seed=args.seed)

    target_obj = None
    for obj_id, obj in env.all_objects.items():
        if obj.__class__.__name__ == "RedCube":
            print(f"Found target object: {obj.name} (ID: {obj_id})")
            target_obj = obj_id
            break

    env.set_fixed_protocol("observance", {})

    dwell = DwellTracker()
    parent_action = np.zeros(env.action_space[1].shape, dtype=np.float32)
    reached_post_shared = False

    def capture_frames(info_dict):
        protocol = getattr(env, "current_protocol", None)
        extra = build_status_lines(env, protocol, target_obj)
        if protocol is not None:
            extra.append(f"JASuccess: {getattr(protocol, 'ja_success', False)}")
            extra.append(f"FallbackUttered: {getattr(protocol, 'fallback_uttered', False)}")
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

    capture_frames(info)

    total_steps = args.warmup_steps + args.max_frames
    for t in range(total_steps):
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None

        if t == args.warmup_steps:
            env.set_fixed_protocol(
                "joint_attention",
                {
                    "infant_initiative": True,
                    "infant_signals": build_context_infant_signals(
                        env, info, target_obj, args.infant_pattern
                    ),
                    "cue_type": "looking",
                    "intent_type": "informative",
                },
            )
            protocol = getattr(env, "current_protocol", None)
            state = getattr(protocol, "state", None) if protocol else None

        if state == "post_shared_attention":
            reached_post_shared = True

        infant_action = build_infant_action(
            env, protocol, target_obj, args.infant_pattern, t, args.warmup_steps
        )

        if "parent_action" in info:
            parent_action = info["parent_action"]

        obs, reward, done, trunc, info = env.step((infant_action, parent_action))
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        dwell.update(state)
        dwell.observe_protocol(protocol)
        capture_frames(info)

        if t % 50 == 0:
            print(
                f"t={t:03d}, protocol={getattr(env, 'current_protocol_name', 'None')}, "
                f"state={state}, reached_post_shared={reached_post_shared}"
            )

        if done or trunc:
            print(f"Episode finished early at t={t} (done={done}, trunc={trunc}).")
            break

        if state in ("completed", "failure") and t > args.warmup_steps:
            if getattr(protocol, "state_timer", 0) > 30:
                break

    protocol = getattr(env, "current_protocol", None)
    final_state = getattr(protocol, "state", "None") if protocol else "None"
    success = 1 if final_state == "completed" else 0
    selected_target = (
        getattr(getattr(protocol, "target_object", None), "id", None)
        if protocol is not None
        else None
    )
    if selected_target is None and protocol is not None:
        selected_target = getattr(protocol, "target_object", None)
    infant_signals = info.get("infant_signals", {}) if isinstance(info, dict) else {}
    gaze_argmax_target = _signal_argmax_target(env, infant_signals.get("gaze_targets", []))
    gesture_argmax_target = _signal_argmax_target(
        env, infant_signals.get("gesture_targets", []), ignore_parent=True
    )
    fallback_uttered = bool(getattr(protocol, "fallback_uttered", False))

    print("===== Summary =====")
    print(f"  final_state         : {final_state}")
    print(f"  success             : {success}")
    print(f"  selected_target     : {_target_name(env, selected_target)}")
    print(f"  gaze_argmax_target  : {_target_name(env, gaze_argmax_target)}")
    print(f"  gesture_argmax_tgt  : {_target_name(env, gesture_argmax_target)}")
    print(f"  reached_post_shared : {reached_post_shared}")
    print(f"  fallback_uttered    : {fallback_uttered}")
    print(f"  frames              : {t}")
    print("===================")
    dwell.print_summary(
        seed=args.seed,
        success=success,
        run=f"ja_infant_initiated_{args.infant_pattern}",
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
            video_path = f"{save_dir}/demo_ja_infant_initiated_{pattern}.mp4"
        else:
            video_path = f"{save_dir}/demo_ja_infant_initiated_{pattern}_{view_name}.mp4"
        print(f"Saving video to {video_path} ({len(frames)} frames)...")
        images_to_video(frames, video_path, fps=12, scale=4)
    print("Done!")


if __name__ == "__main__":
    main()
