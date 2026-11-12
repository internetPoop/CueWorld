"""
振る JA → くれの強制連鎖デモ（バッチ F）。

親が対象を持って振り始め（shake_reached）、くじを使わず object_giving へ固定切替。
親は持ったまま。幼児は振っているあいだ対象を見て、置かれたら取る。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import deque

import numpy as np

from cueworld.paths import ABS_PATH
from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager
from cueworld.video import images_to_video

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from camera_views import get_third_person_view  # noqa: E402
from demo_comprehensive import build_infant_action  # noqa: E402
from demo_ja_shake import annotated_view_from_info, build_status_lines  # noqa: E402
from dwell_tracker import DwellTracker  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="JA shake → object giving chain demo")
    parser.add_argument(
        "--views",
        type=str,
        default="infant,third",
        help="Comma-separated views to save: infant,parent,third",
    )
    parser.add_argument("--max_frames", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    view_set = {v.strip() for v in args.views.split(",") if v.strip()}
    print(f"Starting JA shake → giving demo (views={sorted(view_set)})...")

    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=3,
        object_counts={"RedCube": 1, "GreenCube": 1, "BlueCube": 1},
        max_steps=max(args.max_frames + 100, 1000),
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

    context = {}
    if target_obj is not None:
        context = {
            "fixed_target": target_obj,
            "target_word": "あか",
            "cue_type": "shake",
            "intent_type": "simple",
        }
    env.set_fixed_protocol("joint_attention", context)

    parent_action = np.zeros(env.action_space[1].shape, dtype=np.float32)
    t = 0
    switched = False
    reaching_phase = False
    speech_queue: deque = deque()
    speech_started = [True]
    held_log = []
    dwell = DwellTracker()

    def capture_frames(info_dict):
        protocol = getattr(env, "current_protocol", None)
        extra = build_status_lines(env, protocol, target_obj)
        extra.append(f"Switched: {switched}")
        extra.append(f"Blocked: {sorted(env.blocked_pairs)}")
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

    while t < args.max_frames:
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None

        if (
            not switched
            and env.current_protocol_name == "joint_attention"
            and state == "shaking"
            and bool(getattr(protocol, "shake_reached", False))
        ):
            pair = ("joint_attention", target_obj)
            print(
                f"t={t:03d}: shake_reached → interrupt JA, set_fixed_protocol(object_giving)"
            )
            env.save_protocol_history(
                "joint_attention", False, end_reason="interrupt"
            )
            env.set_fixed_protocol(
                "object_giving",
                {"fixed_target": target_obj},
            )
            switched = True
            if pair in env.blocked_pairs:
                print(f"[warn] interrupt put {pair} on blocked_pairs: {env.blocked_pairs}")
            else:
                print(f"blocked_pairs after interrupt: {env.blocked_pairs}")
            protocol = getattr(env, "current_protocol", None)
            state = getattr(protocol, "state", None) if protocol else None

        infant_action, reaching_phase = build_infant_action(
            env,
            protocol,
            mode="engage",
            primary_id=target_obj,
            other_id=None,
            reaching_phase=reaching_phase,
            speech_queue=speech_queue,
            speech_started=speech_started,
        )

        if "parent_action" in info:
            parent_action = info["parent_action"]

        obs, reward, done, trunc, info = env.step((infant_action, parent_action))
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        dwell.update(state)
        dwell.observe_protocol(protocol)
        capture_frames(info)

        held_log.append(
            (
                t,
                env.current_protocol_name,
                state,
                env.parent.get("held_obj"),
                env.infant.get("held_obj"),
            )
        )

        if t % 50 == 0:
            print(
                f"t={t:03d}, Protocol: {env.current_protocol_name}, State: {state}, "
                f"ParentHeld: {env.parent.get('held_obj')}, "
                f"InfantHeld: {env.infant.get('held_obj')}"
            )

        if done or trunc:
            print(f"Episode finished early at t={t} (done={done}, trunc={trunc}).")
            break

        if (
            switched
            and state in ("completed", "failure")
            and getattr(protocol, "state_timer", 0) > 20
        ):
            break

        t += 1

    protocol = getattr(env, "current_protocol", None)
    final_state = getattr(protocol, "state", "None") if protocol else "None"
    print("===== Summary =====")
    print(f"  final_protocol      : {env.current_protocol_name}")
    print(f"  final_state         : {final_state}")
    print(f"  switched            : {switched}")
    print(f"  parent_held         : {env.parent.get('held_obj')}")
    print(f"  infant_held         : {env.infant.get('held_obj')}")
    print(f"  blocked_pairs       : {env.blocked_pairs}")
    hist = list(env.protocol_history)
    if hist:
        last = hist[-1]
        print(
            f"  last_history        : name={last.get('name')} "
            f"success={last.get('success')} reason={last.get('end_reason')}"
        )
    print("  held_obj transitions:")
    prev = None
    for row in held_log:
        key = row[1:]
        if key != prev:
            print(
                f"    t={row[0]:03d} proto={row[1]} state={row[2]} "
                f"parent={row[3]} infant={row[4]}"
            )
            prev = key
    print("===================")
    dwell.print_summary(seed=args.seed, success=int(final_state == "completed"), run="ja_shake_to_giving")

    env.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    for view_name, frames in movies.items():
        if not frames:
            print(f"[warn] no frames for view={view_name}")
            continue
        video_path = f"{save_dir}/demo_ja_shake_to_giving_{view_name}.mp4"
        print(f"Saving video to {video_path} ({len(frames)} frames)...")
        images_to_video(frames, video_path, fps=12, scale=4)
    print("Done!")


if __name__ == "__main__":
    main()
