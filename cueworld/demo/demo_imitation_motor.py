"""
案F: 二項モータ模倣デモ

幼児の周期性モータ（手開閉・手振り・うなずき・音声+手振り）を parent_tools で生成し、
DyadicProtocol が salience に基づいて親の模倣を行う様子を録画する。
"""

import argparse
import os

import cv2
import numpy as np

from cueworld.paths import ABS_PATH
from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager
from cueworld.video import images_to_video

MOTOR_CYCLE_STEPS = 30
WARMUP_STEPS = 20

INFANT_INITIATED_SCENARIOS = ("hand_open_close", "arm_wave", "head_nod", "voice_arm_wave")
ALL_SCENARIOS = INFANT_INITIATED_SCENARIOS + ("parent_cue",)


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


def motor_phase(t: int, cycle_steps: int = MOTOR_CYCLE_STEPS) -> float:
    return float((t % cycle_steps) / cycle_steps)


def build_infant_action(env, t: int, scenario: str) -> np.ndarray:
    """幼児の行動を parent_tools で構築する（他デモと同様）。"""
    action = np.zeros(env.action_space[0].shape, dtype=np.float32)
    env.gaze_agent(env.infant, env.parent, yaw_link="head", action=action)
    phase = motor_phase(t)

    protocol = getattr(env, "current_protocol", None)
    protocol_state = getattr(protocol, "state", "") if protocol else ""

    if scenario == "parent_cue":
        # 親 cue 後の喃語／反応待ち（旧 waiting_infant_response）。
        if protocol_state == "waiting_babble" and t >= WARMUP_STEPS:
            env.wave_arm(env.infant, action=action, phase=phase)
        return normalize_infant_action(env, action)

    if t < WARMUP_STEPS:
        return normalize_infant_action(env, action)

    if scenario == "hand_open_close":
        env.animate_hand_open_close(env.infant, action=action, phase=phase)
    elif scenario == "arm_wave":
        env.wave_arm(env.infant, action=action, phase=phase)
    elif scenario == "head_nod":
        env.nod_head(env.infant, action=action, phase=phase)
    elif scenario == "voice_arm_wave":
        env.wave_arm(env.infant, action=action, phase=phase)
        if hasattr(env, "phoneme_mapping") and "p0" in env.phoneme_mapping:
            v_coords = env.phoneme_mapping["p0"]
            if hasattr(env, "apply_phoneme_to_action"):
                env.apply_phoneme_to_action(action, v_coords, volume=1.0)
            else:
                action[env.action_map["v_1"]] = v_coords[0]
                action[env.action_map["v_2"]] = v_coords[1]
                if "v_3" in env.action_map and len(v_coords) > 2:
                    action[env.action_map["v_3"]] = v_coords[2]
                action[env.action_map["volume"]] = 1.0

    return normalize_infant_action(env, action)


def scenario_context(scenario: str) -> dict:
    if scenario == "parent_cue":
        # 声チャネルを含めると p0 p1 p0 p1 の見本が出る。腕だけならモータ cue。
        return {
            "infant_initiative": False,
            "parent_channels": ["voice", "arm_raise"],
        }
    return {"infant_initiative": True}


def annotated_view_from_info(
    info: dict,
    env: ProtocolManager,
    view_key: str,
    scenario: str,
    *,
    text_scale: float = 1.0,
    line_gap: int | None = None,
    padding: int = 4,
    box_alpha: float = 0.9,
    cv_thickness: int | None = None,
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
    lines.append(f"Scenario: {scenario}")
    lines.append(f"Protocol: {getattr(env, 'current_protocol_name', 'None')}")

    if hasattr(env, "current_protocol") and env.current_protocol is not None:
        if hasattr(env.current_protocol, "state"):
            lines.append(f"State: {env.current_protocol.state}")
        if hasattr(env.current_protocol, "state_timer") and env.current_protocol.state_timer is not None:
            lines.append(f"State Timer: {env.current_protocol.state_timer}s")
        salient = getattr(env.current_protocol, "salient_channels", None)
        if salient:
            lines.append(f"Salient: {salient}")
        active_motor = getattr(env.current_protocol, "active_motor_channels", None)
        if active_motor:
            lines.append(f"Parent motor: {active_motor}")
        active_channels = getattr(env.current_protocol, "active_channels", None)
        if active_channels:
            lines.append(f"Parent cue: {active_channels}")

    infant_signals = info.get("infant_signals", {})
    lines.append("Signals:")
    if infant_signals:
        for key in (
            "hand_open_close_periodicity",
            "hand_height_margin",
            "babbling_ratio",
            "voice_volume",
        ):
            val = infant_signals.get(key)
            if isinstance(val, (int, float)):
                lines.append(f"  {key}: {val:.2f}")

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
        for line, (_, h) in zip(text.split("\n"), line_sizes):
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


def run_demo(
    scenario: str = "hand_open_close",
    max_frames: int = 450,
    seed: int = 42,
    save_parent_video: bool = False,
    save_dir: str | None = None,
) -> str:
    if scenario not in ALL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    print(f"Starting Motor Imitation Demo (scenario={scenario}, seed={seed})...")
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=3,
        object_counts={"RedCube": 1, "BlueSphere": 1, "YellowCylinder": 1},
        max_steps=max(max_frames + 50, 500),
        return_views=True,
        output_parent_view=save_parent_video,
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

    infant_movie: list[np.ndarray] = []
    parent_movie: list[np.ndarray] = []
    obs, info = env.reset(seed=seed)
    env.set_fixed_protocol("dyadic", scenario_context(scenario))

    if "views" in (info or {}):
        annot_infant = annotated_view_from_info(info, env, "infant_rgb", scenario, text_scale=0.1)
        if annot_infant is not None:
            infant_movie.append(annot_infant)
        if save_parent_video:
            annot_parent = annotated_view_from_info(info, env, "parent_rgb", scenario, text_scale=0.1)
            if annot_parent is not None:
                parent_movie.append(annot_parent)

    parent_action = np.zeros(base.action_space[1].shape, dtype=np.float32)
    t = 0

    while t < max_frames:
        infant_action = build_infant_action(env, t, scenario)

        if "parent_action" in info:
            parent_action = info["parent_action"]

        obs, reward, done, trunc, info = env.step((infant_action, parent_action))

        if "views" in info:
            annot_infant = annotated_view_from_info(info, env, "infant_rgb", scenario, text_scale=0.1)
            if annot_infant is not None:
                infant_movie.append(annot_infant)
            if save_parent_video:
                annot_parent = annotated_view_from_info(info, env, "parent_rgb", scenario, text_scale=0.1)
                if annot_parent is not None:
                    parent_movie.append(annot_parent)

        if t % 50 == 0:
            protocol_name = getattr(env, "current_protocol_name", "None")
            state = getattr(env.current_protocol, "state", "None") if getattr(env, "current_protocol", None) else "None"
            signals = info.get("infant_signals", {})
            print(
                f"t={t:03d}, Protocol: {protocol_name}, State: {state}, "
                f"hoc={signals.get('hand_open_close_periodicity', 0):.2f}, "
                f"raise={signals.get('hand_height_margin', 0):.2f}"
            )

        if done or trunc:
            print(f"Episode finished at t={t} (done={done}, trunc={trunc}).")
            break
        t += 1

    env.close()

    out_dir = save_dir or f"{ABS_PATH}/results/3d_test"
    os.makedirs(out_dir, exist_ok=True)
    video_path = f"{out_dir}/demo_imitation_motor_{scenario}.mp4"
    print(f"Saving video to {video_path}...")
    images_to_video(infant_movie, video_path, fps=12, scale=4)

    if save_parent_video and parent_movie:
        parent_video_path = f"{out_dir}/demo_imitation_motor_{scenario}_parent.mp4"
        print(f"Saving parent video to {parent_video_path}...")
        images_to_video(parent_movie, parent_video_path, fps=12, scale=4)

    print("Done!")
    return video_path


def main():
    parser = argparse.ArgumentParser(description="Dyadic motor imitation demo (Plan F).")
    parser.add_argument(
        "--scenario",
        type=str,
        default="hand_open_close",
        choices=list(ALL_SCENARIOS),
        help="Infant motor pattern or parent-initiated cue demo.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for env reset.")
    parser.add_argument("--max_frames", type=int, default=450, help="Maximum simulation frames.")
    parser.add_argument("--save_parent_video", action="store_true", help="Also save parent view video.")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory for videos.")
    args = parser.parse_args()

    run_demo(
        scenario=args.scenario,
        max_frames=args.max_frames,
        seed=args.seed,
        save_parent_video=args.save_parent_video,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
