import numpy as np
from cueworld.base import BaseEnv
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
    
    # ProtocolManagerとInfantSignalDetectorから情報取得
    current_protocol = getattr(env, "current_protocol_name", "None")
    lines.append(f"Protocol: {current_protocol}")
    
    if hasattr(env, "current_protocol") and env.current_protocol is not None:
        if hasattr(env.current_protocol, "state"):
            lines.append(f"State: {env.current_protocol.state}")
            
    infant_signals = info.get("infant_signals", {})
    if infant_signals:
        lines.append(f"Signals:")
        for key in ["gaze_target_stability", "point_intensity", "voice_best_distance"]:
            val = infant_signals.get(key, 0.0)
            if isinstance(val, (int, float)):
                lines.append(f"  {key}: {val:.2f}")

    text = "\n".join(lines) if lines else ""
    if not text:
        return img

    try:
        from PIL import Image, ImageDraw, ImageFont
        import os

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

def _get_agent_look_at_target_action(env, agent, target_agent, action_names=None) -> np.ndarray:
    from base_wrapper import WrapperBase
    wrapper = WrapperBase(env)
    eye, fwd = env._agent_eye_pose(agent)
    target_pos = p.getLinkState(target_agent["body"], target_agent["eyes_link"], computeForwardKinematics=True)[4]
    target_pos = np.array(target_pos, dtype=np.float32)
    v = target_pos - eye
    
    fwd_yaw = np.array([fwd[0], fwd[1]], dtype=np.float32)
    to_target_yaw = np.array([v[0], v[1]], dtype=np.float32)
    yaw_diff = wrapper.rotation_to_target(fwd_yaw, to_target_yaw)
    
    fwd_xy = np.sqrt(fwd[0]**2 + fwd[1]**2)
    fwd_pitch = np.array([fwd[2], fwd_xy], dtype=np.float32)
    to_target_xy = np.sqrt(v[0]**2 + v[1]**2)
    to_target_pitch = np.array([v[2], to_target_xy], dtype=np.float32)
    pitch_diff = wrapper.rotation_to_target(fwd_pitch, to_target_pitch)
    
    action_repeat = getattr(env, "action_repeat", 1)
    torso_step = env.parent_joint_mappings["torso_joint"] * action_repeat
    head_step = env.parent_joint_mappings["head_tilt_joint"] * action_repeat
    
    torso_action = float(np.clip(yaw_diff / torso_step, -1.0, 1.0))
    pitch_action = float(np.clip(pitch_diff / head_step, -1.0, 1.0))
    
    if action_names is not None:
        action = np.zeros(len(action_names), dtype=np.float32)
        for i, name in enumerate(action_names):
            if name == "torso_joint":
                action[i] = torso_action
            elif name == "head_tilt_joint":
                action[i] = pitch_action
        return action
    else:
        action = np.zeros(env.action_num, dtype=np.float32)
        action[env.action_map["torso_joint"]] = torso_action
        action[env.action_map["head_tilt_joint"]] = pitch_action
        return action

def _get_agent_look_at_object_action(env, agent, target_obj_id, action_names=None) -> np.ndarray:
    from base_wrapper import WrapperBase
    wrapper = WrapperBase(env)
    eye, fwd = env._agent_eye_pose(agent)
    target_pos = env._obj_pos(target_obj_id)
    v = target_pos - eye
    
    fwd_yaw = np.array([fwd[0], fwd[1]], dtype=np.float32)
    to_target_yaw = np.array([v[0], v[1]], dtype=np.float32)
    yaw_diff = wrapper.rotation_to_target(fwd_yaw, to_target_yaw)
    
    fwd_xy = np.sqrt(fwd[0]**2 + fwd[1]**2)
    fwd_pitch = np.array([fwd[2], fwd_xy], dtype=np.float32)
    to_target_xy = np.sqrt(v[0]**2 + v[1]**2)
    to_target_pitch = np.array([v[2], to_target_xy], dtype=np.float32)
    pitch_diff = wrapper.rotation_to_target(fwd_pitch, to_target_pitch)
    
    action_repeat = getattr(env, "action_repeat", 1)
    torso_step = env.parent_joint_mappings["torso_joint"] * action_repeat
    head_step = env.parent_joint_mappings["head_tilt_joint"] * action_repeat
    
    torso_action = float(np.clip(yaw_diff / torso_step, -1.0, 1.0))
    pitch_action = float(np.clip(pitch_diff / head_step, -1.0, 1.0))
    
    if action_names is not None:
        action = np.zeros(len(action_names), dtype=np.float32)
        for i, name in enumerate(action_names):
            if name == "torso_joint":
                action[i] = torso_action
            elif name == "head_tilt_joint":
                action[i] = pitch_action
        return action
    else:
        action = np.zeros(env.action_num, dtype=np.float32)
        action[env.action_map["torso_joint"]] = torso_action
        action[env.action_map["head_tilt_joint"]] = pitch_action
        return action

def main():
    print("Starting Imitation Demo...")
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=3,
        object_counts={"RedCube": 1, "BlueSphere": 1, "YellowCylinder": 1},
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
    
    env = InfantSignalDetector(base)
    env = ProtocolManager(env, demo_mode=False)

    infant_movie = []
    obs, info = env.reset(seed=42)

    # Force dyadic protocol
    env.switch_protocol("dyadic", {"target_object": None})

    if "views" in (info or {}):
        annot = annotated_view_from_info(info, env, "infant_rgb", text_scale=0.1)
        if annot is not None:
            infant_movie.append(annot)

    max_frames = 400
    t = 0
    parent_action = np.zeros(base.action_space[1].shape, dtype=np.float32)

    while t < max_frames:
        infant_action = np.zeros(base.action_space[0].shape, dtype=np.float32)
        
        # 幼児は発声する (あー、うーなど)
        infant_action = _get_agent_look_at_target_action(base, base.infant, base.parent)
        infant_action[base.action_map["volume"]] = 1.0 # 声を出す
        # 時間によって母音を変化させる
        if t < 100:
            infant_action[base.action_map["v_1"]] = 1.0 # a
        elif t < 200:
            infant_action[base.action_map["v_2"]] = 1.0 # i
        elif t < 300:
            infant_action[base.action_map["v_1"]] = -1.0 # u
        else:
            infant_action[base.action_map["v_2"]] = -1.0 # e

        if "parent_action" in info:
            parent_action = info["parent_action"]

        obs, reward, done, trunc, info = env.step((infant_action, parent_action))

        if "views" in info:
            annot = annotated_view_from_info(info, env, "infant_rgb", text_scale=0.1)
            if annot is not None:
                infant_movie.append(annot)

        if t % 50 == 0:
            protocol = getattr(env, "current_protocol_name", "None")
            print(f"t={t:03d}, Protocol: {protocol}")

        if done or trunc:
            print("Episode finished early.")
            break

        t += 1

    env.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    video_path = f"{save_dir}/demo_imitation.mp4"
    print(f"Saving video to {video_path}...")
    images_to_video(infant_movie, video_path, fps=12, scale=4)
    print("Done!")

if __name__ == "__main__":
    main()
