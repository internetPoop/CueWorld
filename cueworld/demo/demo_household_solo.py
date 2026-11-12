"""
HouseholdSoloProtocol デモ

4 色の board と可動 item を固定配置し、色ベースの家事タスク
（collect_red / collect_blue / collect_green / collect_yellow）および
foraging・inspection 状態を録画する。

ランダム配置だと非対象物が手元に割り込んで grasp 失敗を繰り返すことがあるため、
このデモでは item も明示配置して家事タスクの動作確認を安定化する。
"""

import argparse
import os

import cv2
import numpy as np

from cueworld.paths import ABS_PATH
from cueworld.base import BaseEnv
from cueworld.household_schemas import HOUSEHOLD_INSPECTION_DEMO_ON, TASK_SCHEMAS
from cueworld.infant_signal import InfantSignalDetector
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager
from cueworld.video import images_to_video


def get_base_env(env):
    current = env
    while hasattr(current, "env"):
        if isinstance(current, BaseEnv):
            return current
        current = current.env
    return current


def normalize_infant_action(env, action: np.ndarray) -> np.ndarray:
    """joint delta を action_repeat 込みの正規化値に変換する。"""
    for name, step_size in env.infant_joint_mappings.items():
        if name in ("volume", "pitch"):
            continue
        try:
            idx = env.action_map[name]
            action[idx] = action[idx] / (step_size * getattr(env, "action_repeat", 1))
        except KeyError:
            pass
    return np.clip(action, -1.0, 1.0)


def build_infant_action(env, protocol) -> np.ndarray:
    """幼児は親の作業対象（item / board）または親本人を見る。"""
    action = np.zeros(env.action_space[0].shape, dtype=np.float32)
    target_obj = None
    if protocol is not None:
        if getattr(protocol, "current_item_id", None) is not None:
            target_obj = protocol.current_item_id
        elif getattr(protocol, "target_board_id", None) is not None:
            target_obj = protocol.target_board_id

    if target_obj is not None:
        env.gaze_object(env.infant, target_obj, yaw_link="torso", action=action)
    else:
        env.gaze_agent(env.infant, env.parent, yaw_link="torso", action=action)
    return normalize_infant_action(env, action)


def _object_name(env, oid) -> str:
    if oid is None:
        return "-"
    obj = env.all_objects.get(oid)
    if obj is None:
        return str(oid)
    return getattr(obj, "name", str(oid))


def annotated_view_from_info(
    info: dict,
    env: ProtocolManager,
    view_key: str,
    *,
    text_scale: float = 1.0,
    line_gap: int | None = None,
    padding: int = 4,
    box_alpha: float = 0.9,
) -> np.ndarray | None:
    if not isinstance(info, dict):
        return None
    views = info.get("views") or {}
    base_img = views.get(view_key)
    if base_img is None:
        return None

    img = np.array(base_img, copy=True)
    h0, w0 = img.shape[:2]
    if min(h0, w0) <= 64:
        scale = 4
    elif min(h0, w0) <= 128:
        scale = 2
    else:
        scale = 1
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, _ = img.shape[:2]

    lines: list[str] = []
    lines.append(f"Protocol: {getattr(env, 'current_protocol_name', 'None')}")

    protocol = getattr(env, "current_protocol", None)
    base = get_base_env(env)
    if protocol is not None:
        lines.append(f"State: {getattr(protocol, 'state', '-')}")
        intent = getattr(protocol, "global_intent", None)
        if intent is not None:
            label = TASK_SCHEMAS.get(intent, {}).get("label_ja", intent)
            lines.append(f"Intent: {intent} ({label})")
        style = getattr(protocol, "household_style", None)
        if style is not None:
            lines.append(f"Style: {style}")
        item_id = getattr(protocol, "current_item_id", None)
        board_id = getattr(protocol, "target_board_id", None)
        lines.append(f"Item: {_object_name(base, item_id)}")
        lines.append(f"Board: {_object_name(base, board_id)}")
        lines.append(f"Placed: {getattr(protocol, 'items_placed', 0)}")
        insp = getattr(protocol, "inspection_history", None) or []
        if insp:
            lines.append(f"Inspections: {len(insp)} ({insp[-1].get('kind', '?')})")

    candidates = getattr(base, "household_task_candidates", None) or []
    if candidates:
        task_types = ", ".join(c["task_type"] for c in candidates)
        lines.append(f"Feasible: {task_types}")

    text = "\n".join(lines)
    if not text:
        return img

    try:
        from PIL import Image, ImageDraw, ImageFont

        font_px = max(8, int(h * 0.14 * float(text_scale)))
        font = None
        for path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, font_px)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()

        pil_img = Image.fromarray(img).convert("RGBA")
        measure = ImageDraw.Draw(pil_img)
        line_sizes = []
        for line in text.split("\n"):
            bbox = measure.textbbox((0, 0), line, font=font)
            w, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            line_sizes.append((w, lh))
        if line_gap is None:
            base_h = line_sizes[0][1] if line_sizes else max(12, font_px)
            line_gap_px = max(2, int(round(base_h * 0.2)))
        else:
            line_gap_px = int(line_gap)

        max_w = max((w for w, _ in line_sizes), default=0)
        total_h = sum((lh for _, lh in line_sizes)) + line_gap_px * (max(0, len(line_sizes) - 1))

        x0, y0 = padding, padding
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        alpha = int(max(0.0, min(1.0, box_alpha)) * 255)
        draw.rectangle(
            [x0 - padding, y0 - padding, x0 + max_w + padding, y0 + total_h + padding],
            fill=(0, 0, 0, alpha),
        )
        pil_img = Image.alpha_composite(pil_img, overlay)
        draw = ImageDraw.Draw(pil_img)
        y = y0
        for line, (_, lh) in zip(text.split("\n"), line_sizes):
            draw.text((x0, y), line, font=font, fill=(255, 255, 255, 255))
            y += lh + line_gap_px
        return np.array(pil_img.convert("RGB"))
    except Exception:
        pass

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.3, 0.45 * float(text_scale))
    thickness = 1
    y = padding + 12
    for line in text.split("\n"):
        cv2.putText(img, line, (padding, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += int(18 * scale) + 2
    return img


def combine_demo_views(infant_img: np.ndarray | None, parent_img: np.ndarray | None) -> np.ndarray | None:
    """幼児視点と親視点を横に並べて 1 本の確認用動画にする。"""
    if infant_img is None:
        return parent_img
    if parent_img is None:
        return infant_img
    if infant_img.shape[0] != parent_img.shape[0]:
        target_h = min(infant_img.shape[0], parent_img.shape[0])
        infant_img = cv2.resize(
            infant_img,
            (int(infant_img.shape[1] * target_h / infant_img.shape[0]), target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        parent_img = cv2.resize(
            parent_img,
            (int(parent_img.shape[1] * target_h / parent_img.shape[0]), target_h),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.hstack([infant_img, parent_img])


def build_protocol_context(args: argparse.Namespace) -> dict:
    ctx: dict = {"household_style": args.household_style}
    if args.fixed_task is not None:
        ctx["fixed_task"] = args.fixed_task
    if args.inspection:
        ctx.update(HOUSEHOLD_INSPECTION_DEMO_ON)
    return ctx


def run_demo(args: argparse.Namespace) -> str:
    print(
        f"Starting Household Solo Demo "
        f"(task={args.fixed_task or 'random'}, seed={args.seed}, style={args.household_style})..."
    )
    arena_half_extents = (1.6, 1.2)
    arena_rows = 4
    arena_cols = 5
    _, yh = arena_half_extents

    # class 名は BaseEnv の正規化規則に合わせ snake_case（Red_Board → RedBoard）
    fixed_placements = [
        {"class": "Green_Board", "x": 0.4, "y": yh - 0.05, "z": 0.5, "yaw": 90.0},
        {"class": "Yellow_Board", "x": 1.2, "y": yh - 0.05, "z": 0.5, "yaw": 90.0},
        {"class": "Red_Board", "x": 0.4, "y": -yh + 0.05, "z": 0.5, "yaw": 90.0},
        {"class": "Blue_Board", "x": 1.2, "y": -yh + 0.05, "z": 0.5, "yaw": 90.0},
        # item も固定配置して、非対象物の grasp 割り込みを避ける
        {"class": "Red_Cube", "x": -1.25, "y": -yh + 0.05, "z": 0.5, "yaw": 0.0},
        {"class": "Blue_Sphere", "x": -0.85, "y": -yh + 0.05, "z": 0.5, "yaw": 0.0},
        {"class": "Green_Cylinder", "x": -0.45, "y": -yh + 0.05, "z": 0.5, "yaw": 0.0},
        {"class": "Yellow_Cone", "x": -0.05, "y": -yh + 0.05, "z": 0.5, "yaw": 0.0},
    ]

    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=len(fixed_placements),
        object_counts={},
        max_steps=max(args.max_frames + 100, 1000),
        return_views=True,
        output_parent_view=True,
        view_width=256,
        view_height=256,
        action_repeat=1,
        arena_half_extents=arena_half_extents,
        arena_rows=arena_rows,
        arena_cols=arena_cols,
        fixed_placements=fixed_placements,
        debug_info=True,
        placement_method="grid",
        use_cinematic_model=True,
    )

    env = ParentTools(base)
    env = InfantSignalDetector(env)
    env = ProtocolManager(env, demo_mode=False)

    demo_movie: list[np.ndarray] = []
    obs, info = env.reset(seed=args.seed)

    protocol_ctx = build_protocol_context(args)
    env.set_fixed_protocol("household_solo", protocol_ctx)
    protocol = env.current_protocol
    base = get_base_env(env)

    feasible = getattr(base, "household_task_candidates", [])
    print(f"Feasible household tasks: {[c['task_type'] for c in feasible]}")
    if protocol is not None:
        print(
            f"Selected intent={getattr(protocol, 'global_intent', None)}, "
            f"state={getattr(protocol, 'state', None)}, "
            f"candidates={getattr(protocol, 'candidate_items', [])}"
        )

    if "views" in (info or {}):
        infant_annot = annotated_view_from_info(info, env, "infant_rgb", text_scale=0.1)
        parent_annot = annotated_view_from_info(info, env, "parent_rgb", text_scale=0.1)
        combined = combine_demo_views(infant_annot, parent_annot)
        if combined is not None:
            demo_movie.append(combined)

    parent_action = np.zeros(base.action_space[1].shape, dtype=np.float32)
    t = 0

    while t < args.max_frames:
        protocol = env.current_protocol
        infant_action = build_infant_action(env, protocol)

        if "parent_action" in info:
            parent_action = info["parent_action"]

        obs, reward, done, trunc, info = env.step((infant_action, parent_action))

        if "views" in info:
            infant_annot = annotated_view_from_info(info, env, "infant_rgb", text_scale=0.1)
            parent_annot = annotated_view_from_info(info, env, "parent_rgb", text_scale=0.1)
            combined = combine_demo_views(infant_annot, parent_annot)
            if combined is not None:
                demo_movie.append(combined)

        if t % 50 == 0 and protocol is not None:
            item_name = _object_name(base, getattr(protocol, "current_item_id", None))
            print(
                f"t={t:03d}, state={protocol.state}, "
                f"intent={getattr(protocol, 'global_intent', None)}, "
                f"item={item_name}, placed={getattr(protocol, 'items_placed', 0)}"
            )

        if done or trunc:
            print(f"Episode finished at t={t} (done={done}, trunc={trunc}).")
            break

        t += 1

    if protocol is not None and getattr(protocol, "is_finished", False):
        print(
            f"Household protocol completed: intent={protocol.global_intent}, "
            f"placed={protocol.items_placed}, inspections={len(protocol.inspection_history)}"
        )

    env.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    task_suffix = args.fixed_task or "random"
    video_path = f"{save_dir}/demo_household_solo_{task_suffix}_seed{args.seed}.mp4"
    print(f"Saving video to {video_path}...")
    images_to_video(demo_movie, video_path, fps=12, scale=4)
    print("Done!")
    return video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HouseholdSoloProtocol demo")
    parser.add_argument(
        "--fixed-task",
        choices=["collect_red", "collect_blue", "collect_green", "collect_yellow"],
        default=None,
        help="固定する家事タスク種別（未指定時は成立候補からランダム）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument(
        "--household-style",
        choices=["committed", "opportunistic", "monitoring", "interrupted_but_returning"],
        default="committed",
    )
    parser.add_argument(
        "--inspection",
        action="store_true",
        help="item / board inspection を必ず出して目視する（hesitate は出さない）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
