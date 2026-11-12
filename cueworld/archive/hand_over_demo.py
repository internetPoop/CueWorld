import numpy as np
from cueworld.base import BaseEnv
from cueworld.hand_over import HandOverWrapper
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
    view_key: str,
    *,
    text_scale: float = 1.0,
    line_gap: int | None = None,
    padding: int = 4,
    box_alpha: float = 0.9,
    cv_thickness: int | None = None
) -> np.ndarray | None:
    """
    info["views"][view_key] を取り出し、現在の環境のメタ情報を左上に注釈して返す。
    """
    if not isinstance(info, dict):
        return None
    views = info.get("views") or {}
    base = views.get(view_key)
    if base is None:
        return None

    img = np.array(base, copy=True)
    H0, W0 = img.shape[:2]
    # 自動スケール: 小サイズを見やすく拡大
    if min(H0, W0) <= 64:
        SCALE = 4
    elif min(H0, W0) <= 128:
        SCALE = 2
    else:
        SCALE = 1
    if SCALE != 1:
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
    H, W = img.shape[:2]

    # 表示優先キー（HandOverWrapper用）
    keys_order = [
        "hand_over_phase",
        "infant_reward",
        "infant_action",
        "parent_action",
        "parent_target_obj",
    ]
    lines: list[str] = []
    for k in keys_order:
        if k in info:
            v = info[k]
            if isinstance(v, float):
                lines.append(f"{k}={v:.2f}")
            else:
                lines.append(f"{k}={v}")
                
    for k, v in info.items():
        if k == "views" or k in keys_order:
            continue
        if isinstance(v, np.ndarray):
            lines.append(f"{k}=ndarray{list(v.shape)}")
        else:
            if isinstance(v, (list, tuple)) and len(v) <= 15:
                try:
                    s = ",".join(f"{float(x):.2f}" for x in v)
                    lines.append(f"{k}=[{s}]")
                except Exception:
                    lines.append(f"{k}={v}")
            else:
                lines.append(f"{k}={v}")

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
            
        base_h = line_sizes[0][1] if line_sizes else max(12, font_px)
        line_gap_px = int(line_gap) if line_gap is not None else max(2, int(round(base_h * 0.2)))

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
        
        base_h = sizes[0][1] if sizes else max(12, int(10 * text_scale))
        line_gap_px = int(line_gap) if line_gap is not None else max(2, int(round(base_h * 0.2)))
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


def _get_infant_reach_action(env, agent, target_obj_id, action_names=None, reaching_phase: bool = False) -> np.ndarray:
    """幼児が対象オブジェクトを見て、リーチングするためのアクションを生成する"""
    action = np.zeros(env.env.action_num, dtype=np.float32)
    # infant reaches to the object
    #env.hand_reach_object(agent, target_obj_id, action=action)
    # infant looks at the object
    if not reaching_phase:
        yaw, pitch =env.gaze_object(agent, target_obj_id, yaw_link="torso", action=action)
        if abs(yaw) < math.radians(5.0):
            reaching_phase = True
    else:
        env.hand_reach_object(agent, target_obj_id, action=action)
        env.gaze_object(agent, target_obj_id, yaw_link="head", action=action)
    
    
    # normalize action
    for name, step_size in env.env.infant_joint_mappings.items():
        if name == "voice":
            continue
        try:
            idx = env.env.action_map[name]
            action[idx] = action[idx] / (step_size * getattr(env.env, "action_repeat", 1))
        except KeyError:
            pass
            
    clipped = np.clip(action, -1.0, 1.0)
    
    if action_names is not None:
        filtered_action = np.zeros(len(action_names), dtype=np.float32)
        for i, name in enumerate(action_names):
            if name in env.env.action_map:
                filtered_action[i] = clipped[env.env.action_map[name]]
        return filtered_action
    
    return clipped,reaching_phase


def is_ready_to_receive(env, target_obj_id) -> bool:
    """親の持っているオブジェクトを受け取れるような状態(親と幼児の距離など)になったらTrueを返す"""
    parent = env.env.parent
    infant = env.env.infant
    
    # 親が対象オブジェクトを持っているか
    if not env.agent_holding_object_check(parent, target_obj_id):
        return False
        
    # 幼児の受け取りが可能な範囲に親の持つオブジェクトがあるか
    if env.env.detect_grasp_object(infant) == target_obj_id:
        return True
        
    return False

def main(enable_annotation: bool = True, random_infant: bool = False, infant_action_names: list[str] | None = None):
    # ベース環境 + ハンドオーバーラッパー
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=1,
        max_steps=300,
        return_views=True,
        output_parent_view=True,
        view_width=256,
        view_height=256,
        action_repeat=4,
        arena_half_extents=(1.6, 1.2),
        debug_info=True,
        placement_method="grid",
        use_cinematic_model=True,
        object_sizes=[0.05],
        infant_action_names=infant_action_names,
    )
    
    # HandOverWrapper パラメータ設定
    env = HandOverWrapper(
        base,
        reaching_min_steps=10,
        approach_distance_for_grasp=0.5,
        approach_distance_for_handover=0.6,
        max_phase_steps=300
    )

    infant_movie = []
    parent_movie = []

    annot_opts = {
        "text_scale": 0.1,
        "padding": 4,
        "box_alpha": 0.9,
    }

    obs, info = env.reset(seed=42)
    if "views" in (info or {}):
        if enable_annotation:
            infant_annot = annotated_view_from_info(info, "infant_rgb", **annot_opts)
            parent_annot = annotated_view_from_info(info, "parent_rgb", **annot_opts)
        else:
            infant_annot = np.array(info["views"].get("infant_rgb"))
            if info["views"].get("parent_rgb") is not None:
                parent_annot = np.array(info["views"].get("parent_rgb"))
            else:
                parent_annot = None
        if infant_annot is not None:
            infant_movie.append(infant_annot)
        if parent_annot is not None:
            parent_movie.append(parent_annot)

    max_frames = 1000
    t = 0
    current_phase = "phase0"
    
    # 幼児のリーチング対象（ランダムでどれか１つのオブジェクトを選ぶ）
    target_obj = env.env.active_objects[0]
    reaching_phase = False
    
    while t < max_frames:
        if "hand_over_phase" in info:
            current_phase = info["hand_over_phase"]
        
        if random_infant:
            dim = len(infant_action_names) if infant_action_names is not None else env.env.action_num
            infant_action = np.random.uniform(-1.0, 1.0, dim).astype(np.float32)
        else:
            if current_phase == "phase0":
                # 幼児がオブジェクトにリーチング
                if t > 5:
                    infant_action, reaching_phase = _get_infant_reach_action(env, env.env.infant, target_obj, infant_action_names, reaching_phase)
                else:
                    infant_action = np.zeros(len(infant_action_names) if infant_action_names is not None else env.env.action_num, dtype=np.float32)
            elif current_phase == "phase1" or current_phase == "phase2":
                # 親が取ってくるのを待つ間もリーチし続ける
                infant_action, reaching_phase = _get_infant_reach_action(env, env.env.infant, target_obj, infant_action_names, reaching_phase)
                
                # phase2 で親が渡しに来たら、親の持つアイテムを受け取ろうとする
                if current_phase == "phase2":
                    if is_ready_to_receive(env, target_obj):
                        # 親の持っている物を保持する動作を入れる(把持ボタン)
                        if infant_action_names is None:
                            infant_action[env.env.action_map["index_joint"]] = 1.0
                            infant_action[env.env.action_map["other_fingers"]] = 1.0
                        elif "index_joint" in infant_action_names:
                            infant_action[infant_action_names.index("index_joint")] = 1.0
                            if "other_fingers" in infant_action_names:
                                infant_action[infant_action_names.index("other_fingers")] = 1.0
            else:
                infant_action = np.zeros(len(infant_action_names) if infant_action_names is not None else env.env.action_num, dtype=np.float32)

        parent_action = info["parent_action"]
        
        obs, reward, done, trunc, info = env.step((infant_action, parent_action))

        if "views" in info:
            if enable_annotation:
                infant_annot = annotated_view_from_info(info, "infant_rgb", **annot_opts)
                parent_annot = annotated_view_from_info(info, "parent_rgb", **annot_opts)
            else:
                infant_annot = np.array(info["views"].get("infant_rgb"))
                if info["views"].get("parent_rgb") is not None:
                    parent_annot = np.array(info["views"].get("parent_rgb"))
                else:
                    parent_annot = None
            if infant_annot is not None:
                infant_movie.append(infant_annot)
            if parent_annot is not None:
                parent_movie.append(parent_annot)

        if t % 25 == 0:
            phase = info.get("hand_over_phase")
            # HandOverWrapper では info["parent_target_obj"] として記録されるはず
            print(f"t={t:04d} phase={phase} target_obj={target_obj} r_infant={reward[0]:.3f}")

        if done or trunc:
            obs, info = env.reset()
            print(f"--- reset at t={t+1} ---")
            target_obj = env.env.active_objects[0]
            reaching_phase = False
            if "views" in (info or {}):
                if enable_annotation:
                    infant_annot = annotated_view_from_info(info, "infant_rgb", **annot_opts)
                    parent_annot = annotated_view_from_info(info, "parent_rgb", **annot_opts)
                else:
                    infant_annot = np.array(info["views"].get("infant_rgb"))
                    if info["views"].get("parent_rgb") is not None:
                        parent_annot = np.array(info["views"].get("parent_rgb"))
                    else:
                        parent_annot = None
                if infant_annot is not None:
                    infant_movie.append(infant_annot)
                if parent_annot is not None:
                    parent_movie.append(parent_annot)

        t += 1

    env.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)

    prefix = "hand_over"
    if random_infant:
        prefix += "_random"

    if enable_annotation:
        print(f"save movie {save_dir}/{prefix}_anno_infant_view.mp4")
        images_to_video(infant_movie, f"{save_dir}/{prefix}_anno_infant_view.mp4", fps=12, scale=4)
        print(f"save movie {save_dir}/{prefix}_anno_parent_view.mp4")
        images_to_video(parent_movie, f"{save_dir}/{prefix}_anno_parent_view.mp4", fps=12, scale=4)
    else:
        images_to_video(infant_movie, f"{save_dir}/{prefix}_infant_view.mp4", fps=12, scale=4)
        images_to_video(parent_movie, f"{save_dir}/{prefix}_parent_view.mp4", fps=12, scale=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-annot", action="store_true", help="注釈を無効化")
    parser.add_argument("--random-infant", action="store_true", help="幼児をランダムに動かす")
    parser.add_argument("--infant-actions", nargs="+", default=None, help="幼児の行動空間を指定")
    args = parser.parse_args()
    main(enable_annotation=not args.no_annot, random_infant=args.random_infant, infant_action_names=args.infant_actions)
