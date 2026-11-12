import numpy as np
from cueworld.base import BaseEnv
from cueworld.parent_tools import ParentTools
from cueworld.infant_signal import InfantSignalDetector
from cueworld.protocol_manager import ProtocolManager
from cueworld.video import images_to_video
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
            
    if "reward" in info:
        r = info["reward"]
        lines.append(f"Reward: Infant={r[0]:.3f}, Parent={r[1]:.3f}")

    if "protocol_reward" in info:
        lines.append(f"Protocol Reward: {info['protocol_reward']:.3f}")
            
    infant_signals = info.get("infant_signals", {})
    if infant_signals:
        lines.append(f"Signals:")
        for key in ["gaze_target_stability", "point_intensity", "voice_target_stability"]:
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

def _get_agent_look_at_object_action(env, agent, target_obj_id) -> np.ndarray:
    action = np.zeros(env.action_num, dtype=np.float32)
    yaw, pitch = env.gaze_object(agent, target_obj_id, yaw_link="head", action=action)
    
    for name, step_size in env.infant_joint_mappings.items():
        if name in ["volume", "pitch"]: continue
        try:
            idx = env.action_map[name]
            action[idx] = action[idx] / (step_size * getattr(env, "action_repeat", 1))
        except KeyError:
            pass
            
    clipped = np.clip(action, -1.0, 1.0)
    return clipped

def main():
    print("Starting Fixed Joint Attention with Reward Demo...")
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
        placement_method="geometric",
        debug_info=True,
        use_cinematic_model=True
    )
    env = ParentTools(base)
    env = InfantSignalDetector(env, active_signals=["gaze_targets", "looking_at_parent_eye"])
    env = ProtocolManager(env, demo_mode=True, fixed_protocol="joint_attention", chnge_target=False, rejection=False)

    infant_movie = []
    obs, info = env.reset(seed=42)

    

    if "views" in (info or {}):
        annot = annotated_view_from_info(info, env, "infant_rgb", text_scale=0.5)
        if annot is not None:
            infant_movie.append(annot)

    max_frames = 400
    t = 0
    parent_action = np.zeros(base.action_space[1].shape, dtype=np.float32)

    while t < max_frames:
        infant_action = np.zeros(env.action_space[0].shape, dtype=np.float32)
        
        if env.target_obj_id is not None:
            infant_action = _get_agent_look_at_object_action(env, env.infant, env.target_obj_id)

        if "parent_action" in info:
            parent_action = info["parent_action"]

        obs, reward, done, trunc, info = env.step((infant_action, parent_action))
        info["reward"] = reward

        if "views" in info:
            annot = annotated_view_from_info(info, env, "infant_rgb", text_scale=0.5)
            if annot is not None:
                infant_movie.append(annot)

        if t % 50 == 0:
            protocol = getattr(env, "current_protocol_name", "None")
            state = getattr(env.current_protocol, "state", "None") if getattr(env, "current_protocol", None) else "None"
            print(f"t={t:03d}, Protocol: {protocol}, State: {state}, Reward: {reward}")

        if done or trunc:
            break

        t += 1

    env.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    video_path = f"{save_dir}/demo_fixed_ja_reward.mp4"
    print(f"Saving video to {video_path}...")
    images_to_video(infant_movie, video_path, fps=12, scale=4)
    print("Done!")

if __name__ == "__main__":
    main()