"""
JointAttentionProtocol デモ（cue_type="shake"）

親が対象物を掴み → 幼児へ接近 → 振って見せ → 共同注意チェックする流れを
三人称カメラ付きで録画する。置く・渡すは含まない。
"""
import argparse
import math
import os

import cv2
import numpy as np
import pybullet as p

from cueworld.paths import ABS_PATH
from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager
from camera_views import get_third_person_view
from dwell_tracker import DwellTracker
from cueworld.video import images_to_video


def annotated_view_from_info(
    info: dict,
    env: ProtocolManager,
    view_key: str = None,
    *,
    image: np.ndarray = None,
    text_scale: float = 1.0,
    line_gap: int | None = None,
    padding: int = 4,
    box_alpha: float = 0.9,
    cv_thickness: int | None = None,
    extra_lines: list[str] | None = None,
) -> np.ndarray | None:
    if image is not None:
        base = image
    else:
        if not isinstance(info, dict):
            return None
        views = info.get("views") or {}
        base = views.get(view_key)
    if base is None:
        return None

    img = np.array(base, copy=True)
    H0, W0 = img.shape[:2]
    if min(H0, W0) <= 64:
        SCALE = 4
    elif min(H0, W0) <= 128:
        SCALE = 2
    else:
        SCALE = 1
    if SCALE != 1:
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
    H, W = img.shape[:2]

    # 追加する情報
    lines: list[str] = []

    current_protocol = getattr(env, "current_protocol_name", "None")
    lines.append(f"Protocol: {current_protocol}")

    protocol = getattr(env, "current_protocol", None)
    if protocol is not None:
        if hasattr(protocol, "state"):
            lines.append(f"State: {protocol.state}")
        if hasattr(protocol, "ja_success"):
            lines.append(f"JASuccess: {protocol.ja_success}")
        if hasattr(protocol, "cue_type"):
            lines.append(f"Cue: {protocol.cue_type}")
        if hasattr(protocol, "state_timer") and protocol.state_timer is not None:
            lines.append(f"State Timer: {protocol.state_timer}")
        if hasattr(protocol, "debug_info") and protocol.debug_info:
            if protocol.debug_info.get("last_utterance"):
                for key, value in protocol.debug_info["last_utterance"].items():
                    lines.append(f"{key}: {value}")

    if extra_lines:
        lines.extend(extra_lines)

    infant_signals = info.get("infant_signals", {}) if isinstance(info, dict) else {}
    lines.append("Signals:")
    if infant_signals:
        gaze_targets = infant_signals.get("gaze_targets", [])
        if len(gaze_targets) > 0:
            gaze_target = np.argmax(gaze_targets)
            gaze_target_id = (
                env.trackable_targets[gaze_target]
                if gaze_target < len(env.trackable_targets)
                else None
            )
            if gaze_target_id in env.all_objects:
                gaze_target_name = env.all_objects[gaze_target_id].concept_id
                lines.append(f"  Gaze Target: {gaze_target_name}")
        words = infant_signals.get("words", {})
        if words:
            best_word = min(words, key=words.get)
            best_dist = words[best_word]
            if best_dist <= env.rejection_threshold:
                lines.append(f"  latest_word: {best_word}")
            else:
                lines.append("  latest_word: None")

    text = "\n".join(lines) if lines else ""
    if not text:
        return img

    try:
        from PIL import Image, ImageDraw, ImageFont

        font_px = max(8, int(H * 0.16 * float(text_scale)))
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
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            line_sizes.append((w, h))
        if line_gap is None:
            base_h = line_sizes[0][1] if line_sizes else max(12, font_px)
            line_gap_px = max(2, int(round(base_h * 0.2)))
        else:
            line_gap_px = int(line_gap)

        max_w = max((w for w, _ in line_sizes), default=0)
        total_h = sum((h for _, h in line_sizes)) + line_gap_px * (max(0, len(line_sizes) - 1))

        x0, y0 = padding, padding
        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        alpha = int(max(0.0, min(1.0, box_alpha)) * 255)
        draw.rectangle(
            [x0 - padding, y0 - padding, x0 + max_w + padding, y0 + total_h + padding],
            fill=(0, 0, 0, alpha),
        )
        yy = y0
        for (line, (_, h)) in zip(text.split("\n"), line_sizes):
            draw.text((x0, yy), line, fill=(255, 255, 255, 255), font=font)
            yy += h + line_gap_px

        out = Image.alpha_composite(pil_img, overlay).convert("RGB")
        return np.asarray(out)
    except Exception:
        pass

    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4 * (H / 64.0) * float(text_scale)
        thickness = (
            int(cv_thickness) if cv_thickness is not None else max(1, int(round(scale * 2)))
        )
        color_text = (255, 255, 255)
        color_bg = (0, 0, 0)

        sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in text.split("\n")]
        max_w = max((s[0] for s in sizes), default=0)
        if line_gap is None:
            base_h = sizes[0][1] if sizes else max(12, int(10 * text_scale))
            line_gap_px = max(2, int(round(base_h * 0.2)))
        else:
            line_gap_px = int(line_gap)
        total_h = sum((s[1] for s in sizes)) + line_gap_px * (max(0, len(sizes) - 1))

        x0, y0 = padding, padding
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (x0 - padding, y0 - padding),
            (x0 + max_w + padding, y0 + total_h + padding),
            color_bg,
            thickness=-1,
        )
        a = float(max(0.0, min(1.0, box_alpha)))
        img = cv2.addWeighted(overlay, a, img, 1.0 - a, 0.0)

        y = y0 + (sizes[0][1] if sizes else 0)
        for line, sz in zip(text.split("\n"), sizes):
            cv2.putText(img, line, (x0, y), font, scale, color_text, thickness, cv2.LINE_AA)
            y += sz[1] + line_gap_px
        return img
    except Exception:
        pass

    return img


def build_status_lines(env, protocol, target_obj_id: int | None) -> list[str]:
    lines: list[str] = []
    if target_obj_id is not None and target_obj_id in env.all_objects:
        parent_held = env.parent.get("held_obj") == target_obj_id
        infant_held = env.infant.get("held_obj") == target_obj_id
        lines.append(f"ParentHeld: {parent_held}")
        lines.append(f"InfantHeld: {infant_held}")
        sound = getattr(env, "_object_sound", None)
        if sound is not None:
            fired = bool(np.any(np.asarray(sound) != 0.0))
            lines.append(f"ShakeSound: {fired}")
    return lines


def normalize_infant_action(env, infant_action: np.ndarray) -> np.ndarray:
    for name, step_size in env.infant_joint_mappings.items():
        if name in ["volume", "pitch"]:
            continue
        try:
            idx = env.action_map[name]
            infant_action[idx] = infant_action[idx] / (step_size * getattr(env, "action_repeat", 1))
        except KeyError:
            pass
    return np.clip(infant_action, -1.0, 1.0)


def build_infant_action(env, protocol, target_obj, pattern: str):
    """JA-shake 用の幼児行動。"""
    infant_action = np.zeros(env.action_space[0].shape, dtype=np.float32)

    if pattern == "fixed":
        # 対象を注視し続けて共同注意成功を狙う
        if target_obj is not None:
            env.gaze_object(env.infant, target_obj, yaw_link="head", action=infant_action)

    elif pattern == "passive":
        # 対象は見るが積極的な反応はしない
        if target_obj is not None:
            env.gaze_object(env.infant, target_obj, yaw_link="head", action=infant_action)

    elif pattern == "reject":
        parent_pos, _ = p.getBasePositionAndOrientation(
            env.parent["body"], physicsClientId=env.client_id
        )
        infant_pos, _ = p.getBasePositionAndOrientation(
            env.infant["body"], physicsClientId=env.client_id
        )
        away = np.asarray(infant_pos, dtype=np.float32)[:2] - np.asarray(parent_pos, dtype=np.float32)[:2]
        if float(np.linalg.norm(away)) < 1e-3:
            away = np.array([1.0, 0.0], dtype=np.float32)
        else:
            away = away / float(np.linalg.norm(away))
        _, fwd = env._agent_torso_pose(env.infant)
        yaw = env.rotation_to_target(
            np.array([fwd[0], fwd[1]], dtype=np.float32),
            away,
        )
        infant_action[env.action_map["torso_joint"]] = yaw
        infant_action[env.action_map["forward"]] = 0.3

    return normalize_infant_action(env, infant_action)


def parse_args():
    parser = argparse.ArgumentParser(description="JointAttention shake-cue demo")
    parser.add_argument(
        "--infant_pattern",
        type=str,
        default="fixed",
        choices=["fixed", "passive", "reject"],
        help="Infant behavior pattern",
    )
    parser.add_argument(
        "--views",
        type=str,
        default="infant,third",
        help="Comma-separated views to save: infant,parent,third",
    )
    parser.add_argument("--max_frames", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    view_set = {v.strip() for v in args.views.split(",") if v.strip()}
    print(
        f"Starting JA Shake Demo "
        f"(pattern={args.infant_pattern}, views={sorted(view_set)})..."
    )

    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=3,
        object_counts={"RedCube": 1, "GreenCube": 1, "BlueCube": 1},
        max_steps=max(args.max_frames + 100, 700),
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
    reached_shaking = False
    reached_post_shared = False
    shake_sound_firings = 0
    dwell = DwellTracker()

    def capture_frames(info_dict):
        protocol = getattr(env, "current_protocol", None)
        extra = build_status_lines(env, protocol, target_obj)
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
        if state == "shaking":
            reached_shaking = True
        if state == "post_shared_attention":
            reached_post_shared = True

        infant_action = build_infant_action(env, protocol, target_obj, args.infant_pattern)

        if "parent_action" in info:
            parent_action = info["parent_action"]

        obs, reward, done, trunc, info = env.step((infant_action, parent_action))
        protocol = getattr(env, "current_protocol", None)
        state = getattr(protocol, "state", None) if protocol else None
        dwell.update(state)
        dwell.observe_protocol(protocol)
        capture_frames(info)

        # shaking 中の object_sound 発火をカウント
        if state == "shaking":
            sound = None
            if isinstance(obs, (list, tuple)) and len(obs) > 0:
                sound = obs[0].get("object_sound") if isinstance(obs[0], dict) else None
            if sound is None:
                sound = getattr(env, "_object_sound", None)
            if sound is not None and np.any(np.asarray(sound) != 0.0):
                shake_sound_firings += 1

        if t % 50 == 0:
            protocol_name = getattr(env, "current_protocol_name", "None")
            held = env.parent.get("held_obj") == target_obj
            print(
                f"t={t:03d}, Protocol: {protocol_name}, State: {state}, "
                f"ParentHeld: {held}, ShakeFirings: {shake_sound_firings}"
            )

        if done or trunc:
            print(f"Episode finished early at t={t} (done={done}, trunc={trunc}).")
            break

        # 成功で終わったら早めに終了（余韻を少し残す）
        if state in ("completed", "failure") and t > 80:
            # 成功/失敗後も少し録画してから抜ける
            if getattr(protocol, "state_timer", 0) > 30:
                break

        t += 1

    # --- サマリ ---
    protocol = getattr(env, "current_protocol", None)
    final_state = getattr(protocol, "state", "None") if protocol else "None"
    parent_held = env.parent.get("held_obj") == target_obj if target_obj is not None else False
    success = 1 if final_state == "completed" else 0

    print("===== Summary =====")
    print(f"  final_state         : {final_state}")
    print(f"  success             : {success}")
    print(f"  parent_held         : {parent_held}")
    print(f"  shake_sound_firings : {shake_sound_firings}")
    print(f"  reached_shaking     : {reached_shaking}")
    print(f"  reached_post_shared : {reached_post_shared}")
    print(f"  frames              : {t}")
    print("===================")
    dwell.print_summary(
        seed=args.seed,
        success=success,
        run=f"ja_shake_{args.infant_pattern}",
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
            video_path = f"{save_dir}/demo_ja_shake_{pattern}.mp4"
        else:
            video_path = f"{save_dir}/demo_ja_shake_{pattern}_{view_name}.mp4"
        print(f"Saving video to {video_path} ({len(frames)} frames)...")
        images_to_video(frames, video_path, fps=12, scale=4)
    print("Done!")


if __name__ == "__main__":
    main()
