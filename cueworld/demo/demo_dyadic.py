"""
DyadicProtocol デモ（二項・声／身体模倣）

幼児のモータ（掲げキープ・手開閉・音声+掲げ）を parent_tools で生成し、
DyadicProtocol が salience に基づいて親の模倣を行う様子を、三人称カメラ付きで録画する。
物体は扱わない。
"""
import argparse
import os

import cv2
import numpy as np
import pybullet as p

from cueworld.paths import ABS_PATH
try:
    from camera_views import get_third_person_view
except ImportError:
    from cueworld.demo.camera_views import get_third_person_view
from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.parent_tools import ParentTools
from cueworld.protocol_manager import ProtocolManager
from cueworld.video import images_to_video

MOTOR_CYCLE_STEPS = 30
WARMUP_STEPS = 20
RAISE_FORWARD_DIST = 0.35  # 親 gesture_forward_dist と同値
RAISE_SETTLE_STEPS = 10

INFANT_INITIATED_SCENARIOS = ("arm_raise", "hand_open_close", "voice_arm_raise")
PARENT_FIXED_SCENARIOS = {
    "parent_arm_raise": ("arm_raise",),
    "parent_hand_open_close": ("arm_raise", "hand_open_close"),
}
ALL_SCENARIOS = INFANT_INITIATED_SCENARIOS + ("parent_cue",) + tuple(PARENT_FIXED_SCENARIOS.keys())

# D2: 親始動格子で幼児がモータを続ける状態（peak 窓が acknowledging/completed まで開く）
PARENT_RESPONSE_STATES = ("waiting_infant_response", "acknowledging", "completed")


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


def infant_raise_pos(env) -> np.ndarray:
    """幼児 torso 前方・目線高さの掲げ位置（デモ専用）。"""
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
    """幼児を目線高さ付近へ hand_reach_location で掲げる（デモ専用）。"""
    env.hand_reach_location(env.infant, infant_raise_pos(env), action=action)


def scenario_context(scenario: str) -> dict:
    if scenario == "parent_cue":
        return {"infant_initiative": False, "target_phoneme": "p0"}
    if scenario in PARENT_FIXED_SCENARIOS:
        return {
            "infant_initiative": False,
            "parent_channels": list(PARENT_FIXED_SCENARIOS[scenario]),
            "target_phoneme": "p0",
        }
    return {"infant_initiative": True}


def intensity_tag(intensity: float) -> str:
    """動画ファイル名用の強度接尾（例: 1.0 → i100, 0.5 → i050, 0.25 → i025）。"""
    return f"i{int(round(float(intensity) * 100)):03d}"


def build_infant_action(
    env, t: int, scenario: str, intensity: float = 1.0
) -> np.ndarray:
    """幼児の行動を parent_tools で構築する（他デモと同様）。

    intensity: D2 親始動格子用の amplitude_scale（既定 1.0。probe 互換のため必須でない）。
    """
    action = np.zeros(env.action_space[0].shape, dtype=np.float32)
    env.gaze_agent(env.infant, env.parent, yaw_link="head", action=action)
    phase = motor_phase(t)
    scale = float(intensity)

    protocol = getattr(env, "current_protocol", None)
    protocol_state = getattr(protocol, "state", "") if protocol else ""

    if scenario == "parent_cue":
        # ランダム cue への応答デモ: 掲げキープ + 開閉（raise 単独・raise+開閉の両方に応答しうる）
        if protocol_state in PARENT_RESPONSE_STATES and t >= WARMUP_STEPS:
            apply_infant_raise(env, action)
            settle_elapsed = int(getattr(protocol, "_response_wait_elapsed", 0) or 0)
            if settle_elapsed < RAISE_SETTLE_STEPS:
                return normalize_infant_action(env, action)
            env.animate_hand_open_close(
                env.infant,
                action=action,
                phase=phase,
                gesture=True,
                amplitude_scale=scale,
            )
        return normalize_infant_action(env, action)

    if scenario in PARENT_FIXED_SCENARIOS:
        channels = PARENT_FIXED_SCENARIOS[scenario]
        if protocol_state in PARENT_RESPONSE_STATES and t >= WARMUP_STEPS:
            apply_infant_raise(env, action)
            settle_elapsed = int(getattr(protocol, "_response_wait_elapsed", 0) or 0)
            if settle_elapsed < RAISE_SETTLE_STEPS:
                return normalize_infant_action(env, action)
            if "hand_open_close" in channels:
                env.animate_hand_open_close(
                    env.infant,
                    action=action,
                    phase=phase,
                    gesture=True,
                    amplitude_scale=scale,
                )
        return normalize_infant_action(env, action)

    if t < WARMUP_STEPS:
        return normalize_infant_action(env, action)

    if scenario in ("arm_raise", "hand_open_close", "voice_arm_raise"):
        apply_infant_raise(env, action)

    if t < WARMUP_STEPS + RAISE_SETTLE_STEPS and scenario in (
        "arm_raise",
        "hand_open_close",
        "voice_arm_raise",
    ):
        return normalize_infant_action(env, action)

    if scenario == "hand_open_close":
        env.animate_hand_open_close(
            env.infant, action=action, phase=phase, gesture=True, amplitude_scale=scale
        )
    elif scenario == "voice_arm_raise":
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


def annotated_view_from_info(
    info: dict,
    env: ProtocolManager,
    view_key: str = None,
    *,
    image: np.ndarray = None,
    scenario: str = "",
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

    lines: list[str] = []
    if scenario:
        lines.append(f"Scenario: {scenario}")

    current_protocol = getattr(env, "current_protocol_name", "None")
    lines.append(f"Protocol: {current_protocol}")

    protocol = getattr(env, "current_protocol", None)
    if protocol is not None:
        if hasattr(protocol, "state"):
            lines.append(f"State: {protocol.state}")
        if hasattr(protocol, "state_timer") and protocol.state_timer is not None:
            lines.append(f"State Timer: {protocol.state_timer}")
        salient = getattr(protocol, "salient_channels", None)
        if salient:
            lines.append(f"Salient: {salient}")
        active_motor = getattr(protocol, "active_motor_channels", None)
        if active_motor:
            lines.append(f"Parent motor: {active_motor}")
        active_channels = getattr(protocol, "active_channels", None)
        if active_channels:
            lines.append(f"Parent cue: {active_channels}")
        if hasattr(protocol, "debug_info") and protocol.debug_info:
            if protocol.debug_info.get("last_utterance"):
                for key, value in protocol.debug_info["last_utterance"].items():
                    lines.append(f"{key}: {value}")

    if extra_lines:
        lines.extend(extra_lines)

    infant_signals = info.get("infant_signals", {}) if isinstance(info, dict) else {}
    lines.append("Signals:")
    if infant_signals:
        for key in (
            "hand_height_margin",
            "hand_open_close_periodicity",
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


def build_status_lines(env, protocol, scenario: str) -> list[str]:
    lines: list[str] = []
    lines.append(f"Scenario: {scenario}")
    if protocol is not None:
        salient = getattr(protocol, "salient_channels", None)
        if salient:
            lines.append(f"Salient: {salient}")
        active_motor = getattr(protocol, "active_motor_channels", None)
        if active_motor:
            lines.append(f"ParentMotor: {active_motor}")
        active_channels = getattr(protocol, "active_channels", None)
        if active_channels:
            lines.append(f"ParentCue: {active_channels}")
        source = getattr(protocol, "source_phoneme", None)
        if source:
            lines.append(f"Phoneme: {source}")
        peaks = getattr(protocol, "channel_peak_scores", None) or {}
        if peaks:
            peak_str = " ".join(f"{k}={v:.2f}" for k, v in peaks.items())
            lines.append(f"Peak: {peak_str}")
        channel_on = getattr(protocol, "channel_on", None) or {}
        on_names = [k for k, v in channel_on.items() if v]
        if on_names:
            lines.append(f"On: {on_names}")
        if hasattr(protocol, "response_matched"):
            lines.append(f"Matched: {protocol.response_matched}")
        tte = getattr(protocol, "time_to_engagement", None)
        if tte is not None:
            lines.append(f"TTE: {tte}")
    return lines


def parse_args():
    parser = argparse.ArgumentParser(description="Dyadic (voice/motor imitation) demo")
    parser.add_argument(
        "--scenario",
        type=str,
        default="arm_raise",
        choices=list(ALL_SCENARIOS),
        help="Infant motor pattern or parent-initiated cue demo.",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=1.0,
        help="Infant motor amplitude_scale for parent-initiated grid demos (default 1.0).",
    )
    parser.add_argument(
        "--views",
        type=str,
        default="infant,third",
        help="Comma-separated views to save: infant,parent,third",
    )
    parser.add_argument("--max_frames", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_annotate",
        action="store_true",
        help="Save raw camera frames without overlay text.",
    )
    return parser.parse_args()


def raw_view_from_info(info: dict, view_key: str) -> np.ndarray | None:
    """info['views'] から生画像を取り出す（アノテーションなし）。"""
    if not isinstance(info, dict):
        return None
    views = info.get("views") or {}
    base = views.get(view_key)
    if base is None:
        return None
    return np.array(base, copy=True)


def main():
    # dwell_tracker は cueworld/demo 実行時のみ必要（eval からの import では不要）
    from dwell_tracker import DwellTracker

    args = parse_args()
    view_set = {v.strip() for v in args.views.split(",") if v.strip()}
    print(
        f"Starting Dyadic Demo "
        f"(scenario={args.scenario}, intensity={args.intensity}, "
        f"views={sorted(view_set)}, annotate={not args.no_annotate})..."
    )

    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=3,
        object_counts={"RedCube": 1, "BlueSphere": 1, "YellowCylinder": 1},
        max_steps=max(args.max_frames + 50, 500),
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
    env.set_fixed_protocol("dyadic", scenario_context(args.scenario))

    parent_action = np.zeros(env.action_space[1].shape, dtype=np.float32)
    t = 0
    dwell = DwellTracker()

    def capture_frames(info_dict):
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
        extra = build_status_lines(env, protocol, args.scenario)
        if "infant" in movies and "views" in (info_dict or {}):
            annot = annotated_view_from_info(
                info_dict,
                env,
                "infant_rgb",
                scenario=args.scenario,
                text_scale=0.1,
                extra_lines=extra,
            )
            if annot is not None:
                movies["infant"].append(annot)
        if "parent" in movies and "views" in (info_dict or {}):
            annot = annotated_view_from_info(
                info_dict,
                env,
                "parent_rgb",
                scenario=args.scenario,
                text_scale=0.1,
                extra_lines=extra,
            )
            if annot is not None:
                movies["parent"].append(annot)
        if "third" in movies:
            # 三人称は画面を見やすくするため、アノテーションなしで保存する
            third = get_third_person_view(env)
            if third is not None:
                movies["third"].append(third)

    capture_frames(info)

    while t < args.max_frames:
        protocol = getattr(env, "current_protocol", None)
        infant_action = build_infant_action(
            env, t, args.scenario, intensity=args.intensity
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
            protocol_name = getattr(env, "current_protocol_name", "None")
            signals = info.get("infant_signals", {}) if isinstance(info, dict) else {}
            print(
                f"t={t:03d}, Protocol: {protocol_name}, State: {state}, "
                f"margin={signals.get('hand_height_margin', 0):.2f}, "
                f"hoc={signals.get('hand_open_close_periodicity', 0):.2f}, "
                f"on={getattr(protocol, 'channel_on', None)}, "
                f"matched={getattr(protocol, 'response_matched', None)}"
            )

        if done or trunc:
            print(f"Episode finished early at t={t} (done={done}, trunc={trunc}).")
            break

        t += 1

    protocol = getattr(env, "current_protocol", None)
    final_state = getattr(protocol, "state", "None") if protocol else "None"
    salient = getattr(protocol, "salient_channels", None) if protocol else None
    active_motor = getattr(protocol, "active_motor_channels", None) if protocol else None
    success = 1 if final_state == "completed" else 0

    print("===== Summary =====")
    print(f"  scenario    : {args.scenario}")
    print(f"  intensity   : {args.intensity}")
    print(f"  final_state : {final_state}")
    print(f"  salient     : {salient}")
    print(f"  parent_motor: {active_motor}")
    print(f"  peaks       : {getattr(protocol, 'channel_peak_scores', None)}")
    print(f"  on          : {getattr(protocol, 'channel_on', None)}")
    print(f"  matched     : {getattr(protocol, 'response_matched', None)}")
    print(f"  tte         : {getattr(protocol, 'time_to_engagement', None)}")
    print(f"  frames      : {t}")
    print("===================")
    dwell.print_summary(
        seed=args.seed,
        success=success,
        run=f"dyadic_{args.scenario}_{intensity_tag(args.intensity)}",
    )

    env.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    scenario = args.scenario
    itag = intensity_tag(args.intensity)
    use_intensity_suffix = scenario in PARENT_FIXED_SCENARIOS
    for view_name, frames in movies.items():
        if not frames:
            print(f"[warn] no frames for view={view_name}")
            continue
        if use_intensity_suffix:
            base_name = f"demo_dyadic_{scenario}_{itag}"
        else:
            base_name = f"demo_dyadic_{scenario}"
        if view_name == "infant":
            video_path = f"{save_dir}/{base_name}.mp4"
        else:
            video_path = f"{save_dir}/{base_name}_{view_name}.mp4"
        print(f"Saving video to {video_path} ({len(frames)} frames)...")
        images_to_video(frames, video_path, fps=12, scale=4)
    print("Done!")


if __name__ == "__main__":
    main()
