import numpy as np
from cueworld.base import BaseEnv
from cueworld.parent_tools import ParentTools
from cueworld.infant_signal import InfantSignalDetector
from cueworld.protocol_manager import ProtocolManager
from cueworld.parent_tools import ParentTools
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

    lines: list[str] = []
    
    current_protocol = getattr(env, "current_protocol_name", "None")
    lines.append(f"Protocol: {current_protocol}")
    
    if hasattr(env, "current_protocol") and env.current_protocol is not None:
        if hasattr(env.current_protocol, "state"):
            lines.append(f"State: {env.current_protocol.state}")
            
    infant_signals = info.get("infant_signals", {})
    if infant_signals:
        lines.append(f"Signals:")
        for key in ["gaze_target_stability", "open_intensity", "voice_target_stability"]:
            val = infant_signals.get(key, 0.0)
            if isinstance(val, (int, float)):
                lines.append(f"  {key}: {val:.2f}")

    text = "\n".join(lines) if lines else ""
    if not text:
        return img

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

def _get_infant_reach_action(env, agent, target_obj_id, reaching_phase: bool = False) -> tuple[np.ndarray, bool]:
    action = np.zeros(env.action_num, dtype=np.float32)
    if not reaching_phase:
        yaw, pitch = env.gaze_object(agent, target_obj_id, yaw_link="torso", action=action)
        if abs(yaw) < math.radians(5.0):
            reaching_phase = True
    else:
        env.hand_reach_object(agent, target_obj_id, action=action)
        env.gaze_object(agent, target_obj_id, yaw_link="head", action=action)
    
    for name, step_size in env.infant_joint_mappings.items():
        if name == "volume" or name == "pitch":
            continue
        try:
            idx = env.action_map[name]
            action[idx] = action[idx] / (step_size * getattr(env, "action_repeat", 1))
        except KeyError:
            pass
            
    clipped = np.clip(action, -1.0, 1.0)
    return clipped, reaching_phase

def main():
    parser = argparse.ArgumentParser(description="Test Parent's reaction to a single infant modality signal.")
    parser.add_argument("--modality", type=str, choices=["voice", "gaze", "gesture"], default="gaze", 
                        help="Choose which modality the infant should strongly exhibit (default: voice).")
    args = parser.parse_args()

    print(f"Starting Single Modality Demo. Testing: {args.modality.upper()} signal")
    
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=3,
        object_counts={"RedCube": 1, "GreenCube": 1, "BlueCube": 1},
        max_steps=500,
        return_views=True,
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
    env = ProtocolManager(env) # デモモードをFalseにして親の自律切り替えをテスト

    infant_movie = []
    obs, info = env.reset(seed=42)

    if "views" in (info or {}):
        annot = annotated_view_from_info(info, env, "infant_rgb", text_scale=0.1)
        if annot is not None:
            infant_movie.append(annot)

    max_frames = 1000
    t = 0
    parent_action = np.zeros(base.action_space[1].shape, dtype=np.float32)
    reaching_phase = False

    target_obj = None
    for obj_id, obj in env.all_objects.items():
        if obj.__class__.__name__ == "RedCube":
            target_obj = obj_id
            break

    while t < max_frames:
        infant_action = np.zeros(env.action_space[0].shape, dtype=np.float32)
        
        # 50フレーム経過後から、指定したモダリティの強いシグナルを出し続ける
        if t > 50:
            if args.modality == "voice":
                # ボリュームを最大にし、他の動作はしない
                if "volume" in env.action_map:
                    infant_action[env.action_map["volume"]] = 1.0
                    
            elif args.modality == "gaze":
                # ターゲットを見つめ続けるが、声は出さず手も伸ばさない
                if target_obj is not None:
                    yaw, pitch = env.gaze_object(env.infant, target_obj, yaw_link="head", action=infant_action)
                    # 正規化
                    for name, step_size in env.infant_joint_mappings.items():
                        if name in ["volume", "pitch"]: continue
                        try:
                            idx = env.action_map[name]
                            infant_action[idx] = infant_action[idx] / (step_size * getattr(env, "action_repeat", 1))
                        except KeyError:
                            pass
                    infant_action = np.clip(infant_action, -1.0, 1.0)
                    
            elif args.modality == "gesture":
                # ターゲットに手を伸ばし続ける
                if target_obj is not None:
                    infant_action, reaching_phase = _get_infant_reach_action(env, env.infant, target_obj, reaching_phase=reaching_phase)
                    # 声は出さない
                    if "volume" in env.action_map:
                        infant_action[env.action_map["volume"]] = 0.0

        if "parent_action" in info:
            parent_action = info["parent_action"]

        obs, reward, done, trunc, info = env.step((infant_action, parent_action))

        if "views" in info:
            annot = annotated_view_from_info(info, env, "infant_rgb", text_scale=0.1)
            if annot is not None:
                infant_movie.append(annot)

        if done or trunc:
            break

        t += 1

    env.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    video_path = f"{save_dir}/demo_single_{args.modality}.mp4"
    print(f"Saving video to {video_path}...")
    images_to_video(infant_movie, video_path, fps=12, scale=4)
    print("Done!")

if __name__ == "__main__":
    main()