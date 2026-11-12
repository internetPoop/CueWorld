import numpy as np
from cueworld.base import BaseEnv
from cueworld.parent_tools import ParentTools
from cueworld.infant_signal import InfantSignalDetector
from cueworld.protocol_manager import ProtocolManager
from cueworld.video import images_to_video
import random
import math
import pybullet as p
import cv2
import argparse
import os
from cueworld.paths import ABS_PATH

def annotated_view_from_info(
    info: dict,
    env: ProtocolManager,
    view_key: str,
    *,
    text_scale: float = 1.0,
    line_gap: int | None = None,
    padding: int = 4,
    box_alpha: float = 0.9,
    cv_thickness: int | None = None
) -> np.ndarray | None:
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
    
    if hasattr(env, "current_protocol") and env.current_protocol is not None:
        if hasattr(env.current_protocol, "state"):
            lines.append(f"State: {env.current_protocol.state}")
        if hasattr(env.current_protocol, "state_timer") and env.current_protocol.state_timer is not None:
            lines.append(f"State Timer: {env.current_protocol.state_timer}s")
        if hasattr(env.current_protocol, "debug_info") and env.current_protocol.debug_info:
            if env.current_protocol.debug_info["last_utterance"]:
                for key, value in env.current_protocol.debug_info["last_utterance"].items():
                    lines.append(f"{key}: {value}")
            
    infant_signals = info.get("infant_signals", {})
    lines.append(f"Signals:")
    if infant_signals:
        gaze_targets = infant_signals.get("gaze_targets", [])
        if len(gaze_targets) > 0:
            gaze_target = np.argmax(gaze_targets)
            gaze_target_id = env.trackable_targets[gaze_target] if gaze_target < len(env.trackable_targets) else None
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
                lines.append(f"  latest_word: None")

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
        draw.rectangle([x0 - padding, y0 - padding, x0 + max_w + padding, y0 + total_h + padding],
                       fill=(0, 0, 0, alpha))
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
        thickness = int(cv_thickness) if cv_thickness is not None else max(1, int(round(scale * 2)))
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
        cv2.rectangle(overlay, (x0 - padding, y0 - padding), (x0 + max_w + padding, y0 + total_h + padding), color_bg, thickness=-1)
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

def _get_infant_reach_action(env, agent, target_obj_id, action, reaching_phase: bool = False) -> tuple[np.ndarray, bool]:
    if not reaching_phase:
        yaw, pitch = env.gaze_object(agent, target_obj_id, yaw_link="torso", action=action)
        if abs(yaw) < math.radians(5.0):
            reaching_phase = True
    else:
        env.hand_reach_object(agent, target_obj_id, action=action)
        env.gaze_object(agent, target_obj_id, yaw_link="head", action=action)
    return action, reaching_phase

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infant_pattern", type=str, default="fixed", choices=["fixed", "adaptive"], help="Infant behavior pattern")
    parser.add_argument("--save_parent_video", action="store_true", help="Save parent view video")
    args = parser.parse_args()

    print(f"Starting Fixed Object Giving Demo (Infant Pattern: {args.infant_pattern})...")
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=3,
        object_counts={"RedCube": 1, "GreenCube": 1, "BlueCube": 1},
        max_steps=500,
        return_views=True,
        output_parent_view=args.save_parent_video,
        view_width=256,
        view_height=256,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        debug_info=True,
        placement_method="grid",
        use_cinematic_model=True
    )
    
    env = ParentTools(base)
    env = InfantSignalDetector(env)
    env = ProtocolManager(env, demo_mode=True)

    infant_movie = []
    parent_movie = []
    obs, info = env.reset(seed=42)

    # Find target object (RedCube)
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
            "intent_type": "informative"
        }

    env.set_fixed_protocol("object_giving", context)

    if "views" in (info or {}):
        annot_infant = annotated_view_from_info(info, env, "infant_rgb", text_scale=0.1)
        if annot_infant is not None:
            infant_movie.append(annot_infant)
        if args.save_parent_video:
            annot_parent = annotated_view_from_info(info, env, "parent_rgb", text_scale=0.1)
            if annot_parent is not None:
                parent_movie.append(annot_parent)

    max_frames = 400
    t = 0
    parent_action = np.zeros(env.action_space[1].shape, dtype=np.float32)
    reaching_phase = False

    while t < max_frames:
        infant_action = np.zeros(env.action_space[0].shape, dtype=np.float32)
        
        protocol = getattr(env, "current_protocol", None)
        protocol_state = getattr(protocol, "state", "None") if protocol else "None"
        
        # 幼児の行動
        if args.infant_pattern == "fixed":
            if target_obj is not None:
                if t <= 50:
                    pass
                elif 50 < t <= 150:
                    infant_action, reaching_phase = _get_infant_reach_action(env, env.infant, target_obj, action=infant_action, reaching_phase=reaching_phase)
                    infant_action[env.action_map["volume"]] = 1.0 # 強い声
                else:
                    infant_action, reaching_phase = _get_infant_reach_action(env, env.infant, target_obj, action=infant_action, reaching_phase=reaching_phase)
                    
                    if protocol_state == "presenting":
                        # 親が持ってきたら受け取る動作
                        infant_action[env.action_map["index_joint"]] = 3.14
                        infant_action[env.action_map["other_fingers"]] = 3.14
                        
        elif args.infant_pattern == "adaptive":
            if target_obj is not None:
                if protocol_state in ["init", "fetching", "approach"]:
                    # 要求する
                    infant_action, reaching_phase = _get_infant_reach_action(env, env.infant, target_obj, action=infant_action, reaching_phase=reaching_phase)
                    infant_action[env.action_map["volume"]] = 1.0
                elif protocol_state == "presenting":
                    # 受け取る
                    infant_action, reaching_phase = _get_infant_reach_action(env, env.infant, target_obj, action=infant_action, reaching_phase=reaching_phase)
                    infant_action[env.action_map["index_joint"]] = 3.14
                    infant_action[env.action_map["other_fingers"]] = 3.14
                elif protocol_state in ["waiting_infant", "reaction", "completed"]:
                    # 受け取った後、オブジェクトを見て声を出すなど
                    env.gaze_object(env.infant, target_obj, yaw_link="head", action=infant_action)
                    infant_action[env.action_map["index_joint"]] = 3.14
                    infant_action[env.action_map["other_fingers"]] = 3.14
                    if hasattr(env, "phoneme_mapping") and "p0" in env.phoneme_mapping:
                        v_coords = env.phoneme_mapping["p0"]
                        if hasattr(env, "apply_phoneme_to_action"):
                            env.apply_phoneme_to_action(infant_action, v_coords, volume=0.5)
                        else:
                            infant_action[env.action_map["v_1"]] = v_coords[0]
                            infant_action[env.action_map["v_2"]] = v_coords[1]
                            if "v_3" in env.action_map and len(v_coords) > 2:
                                infant_action[env.action_map["v_3"]] = v_coords[2]
                            infant_action[env.action_map["volume"]] = 0.5
                else:
                    env.gaze_object(env.infant, target_obj, yaw_link="head", action=infant_action)

        # 正規化
        for name, step_size in env.infant_joint_mappings.items():
            if name in ["volume", "pitch"]: continue
            try:
                idx = env.action_map[name]
                infant_action[idx] = infant_action[idx] / (step_size * getattr(env, "action_repeat", 1))
            except KeyError:
                pass
        infant_action = np.clip(infant_action, -1.0, 1.0)

        if "parent_action" in info:
            parent_action = info["parent_action"]

        obs, reward, done, trunc, info = env.step((infant_action, parent_action))

        if "views" in info:
            annot_infant = annotated_view_from_info(info, env, "infant_rgb", text_scale=0.1)
            if annot_infant is not None:
                infant_movie.append(annot_infant)
            if args.save_parent_video:
                annot_parent = annotated_view_from_info(info, env, "parent_rgb", text_scale=0.1)
                if annot_parent is not None:
                    parent_movie.append(annot_parent)

        if t % 50 == 0:
            protocol_name = getattr(env, "current_protocol_name", "None")
            state = getattr(env.current_protocol, "state", "None") if getattr(env, "current_protocol", None) else "None"
            print(f"t={t:03d}, Protocol: {protocol_name}, State: {state}")

        if done or trunc:
            print("Episode finished early.")
            break

        t += 1

    env.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    video_path = f"{save_dir}/demo_fixed_give.mp4"
    print(f"Saving video to {video_path}...")
    images_to_video(infant_movie, video_path, fps=12, scale=4)
    if args.save_parent_video:
        parent_video_path = f"{save_dir}/demo_fixed_give_parent.mp4"
        print(f"Saving parent video to {parent_video_path}...")
        images_to_video(parent_movie, parent_video_path, fps=12, scale=4)
    print("Done!")

if __name__ == "__main__":
    main()
