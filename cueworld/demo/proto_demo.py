import numpy as np
from cueworld.base import BaseEnv
from cueworld.joint_attention_wrappers import RJAWrapper  # RespondingJointAttentionRewardWrapper -> RJAWrapper
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
    - view_key: "infant_rgb" | "parent_rgb"
    - 画像が小さい場合は自動スケール（SCALE=4/2/1）
    - Pillow があれば TrueType で描画、無ければ OpenCV にフォールバック
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

    # 表示優先キー（現在のラッパが生成する情報に対応）
    keys_order = [
        "ja_phase",
        "infant_reward",
        "infant_action",
        "parent_action",
        "ja_target_obj",
    ]
    # 1) 優先キー
    lines: list[str] = []
    for k in keys_order:
        if k in info:
            v = info[k]
            if isinstance(v, float):
                lines.append(f"{k}={v:.2f}")
            else:
                lines.append(f"{k}={v}")
    # 2) それ以外（views を除く）も追記（重複除外）
    for k, v in info.items():
        if k == "views" or k in keys_order:
            continue
        if isinstance(v, np.ndarray):
            lines.append(f"{k}=ndarray{list(v.shape)}")
        else:
            # parent_action などのベクトル/タプルは短く整形
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

    # Pillow での描画
    try:
        from PIL import Image, ImageDraw, ImageFont
        import os

        font_px = max(8, int(H * 0.16 * float(text_scale)))
        # フォント候補
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
        # 行ごとの bbox 採寸
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

    # OpenCV フォールバック
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
    """
    エージェントがターゲットエージェントの方を向くためのアクション生成。
    最新のURDF構造とaction_mapに対応。
    """
    from base_wrapper import WrapperBase
    
    # WrapperBaseのメソッドを使用してyaw, pitch差分を計算
    wrapper = WrapperBase(env)
    eye, fwd = env._agent_eye_pose(agent)
    target_pos = p.getLinkState(target_agent["body"], target_agent["eyes_link"], computeForwardKinematics=True)[4]
    target_pos = np.array(target_pos, dtype=np.float32)
    v = target_pos - eye
    
    # yawとpitchの計算
    fwd_yaw = np.array([fwd[0], fwd[1]], dtype=np.float32)
    to_target_yaw = np.array([v[0], v[1]], dtype=np.float32)
    yaw_diff = wrapper.rotation_to_target(fwd_yaw, to_target_yaw)
    
    fwd_xy = np.sqrt(fwd[0]**2 + fwd[1]**2)
    fwd_pitch = np.array([fwd[2], fwd_xy], dtype=np.float32)
    to_target_xy = np.sqrt(v[0]**2 + v[1]**2)
    to_target_pitch = np.array([v[2], to_target_xy], dtype=np.float32)
    pitch_diff = wrapper.rotation_to_target(fwd_pitch, to_target_pitch)
    
    # action_repeatを考慮したステップサイズ
    action_repeat = getattr(env, "action_repeat", 1)
    torso_step = env.parent_joint_mappings["torso_joint"] * action_repeat
    head_step = env.parent_joint_mappings["head_tilt_joint"] * action_repeat
    
    # 正規化された行動値を計算
    torso_action = float(np.clip(yaw_diff / torso_step, -1.0, 1.0))
    pitch_action = float(np.clip(pitch_diff / head_step, -1.0, 1.0))
    
    # アクション名が指定されている場合（幼児用）
    if action_names is not None:
        action = np.zeros(len(action_names), dtype=np.float32)
        for i, name in enumerate(action_names):
            if name == "torso_joint":
                action[i] = torso_action
            elif name == "head_tilt_joint":
                action[i] = pitch_action
            elif name == "voice":
                # たまに声を出す
                action[i] = 1.0 if random.random() < 0.05 else 0.0
        return action
    else:
        # フルアクション（親用）
        action = np.zeros(env.action_num, dtype=np.float32)
        action[env.action_map["torso_joint"]] = torso_action
        action[env.action_map["head_tilt_joint"]] = pitch_action
        # たまに声を出す
        if random.random() < 0.05:
            action[env.action_map["voice"]] = 1.0
        return action


def _get_agent_look_at_object_action(env, agent, target_obj_id, action_names=None) -> np.ndarray:
    """
    エージェントがターゲットオブジェクトの方を向くためのアクション生成。
    """
    from base_wrapper import WrapperBase
    
    wrapper = WrapperBase(env)
    eye, fwd = env._agent_eye_pose(agent)
    target_pos = env._obj_pos(target_obj_id)
    v = target_pos - eye
    
    # yawとpitchの計算
    fwd_yaw = np.array([fwd[0], fwd[1]], dtype=np.float32)
    to_target_yaw = np.array([v[0], v[1]], dtype=np.float32)
    yaw_diff = wrapper.rotation_to_target(fwd_yaw, to_target_yaw)
    
    fwd_xy = np.sqrt(fwd[0]**2 + fwd[1]**2)
    fwd_pitch = np.array([fwd[2], fwd_xy], dtype=np.float32)
    to_target_xy = np.sqrt(v[0]**2 + v[1]**2)
    to_target_pitch = np.array([v[2], to_target_xy], dtype=np.float32)
    pitch_diff = wrapper.rotation_to_target(fwd_pitch, to_target_pitch)
    
    # action_repeatを考慮したステップサイズ
    action_repeat = getattr(env, "action_repeat", 1)
    torso_step = env.parent_joint_mappings["torso_joint"] * action_repeat
    head_step = env.parent_joint_mappings["head_tilt_joint"] * action_repeat
    
    # 正規化された行動値を計算
    torso_action = float(np.clip(yaw_diff / torso_step, -1.0, 1.0))
    pitch_action = float(np.clip(pitch_diff / head_step, -1.0, 1.0))
    
    # アクション名が指定されている場合（幼児用）
    if action_names is not None:
        action = np.zeros(len(action_names), dtype=np.float32)
        for i, name in enumerate(action_names):
            if name == "torso_joint":
                action[i] = torso_action
            elif name == "head_tilt_joint":
                action[i] = pitch_action
            elif name == "voice":
                # phase2でたまに声を出す
                action[i] = 1.0 if random.random() < 0.05 else 0.0
        return action
    else:
        # フルアクション（親用）
        action = np.zeros(env.action_num, dtype=np.float32)
        action[env.action_map["torso_joint"]] = torso_action
        action[env.action_map["head_tilt_joint"]] = pitch_action
        if random.random() < 0.05:
            action[env.action_map["voice"]] = 1.0
        return action


def main(enable_annotation: bool = True, fixed_parent_cue: str | None = None, random_infant: bool = False, infant_action_names: list[str] | None = None):
    # ベース環境 + 共同注意報酬ラッパー（RJAWrapper）
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=3,
        max_steps=200,
        return_views=True,
        view_width=256,
        view_height=256,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        debug_info=True,
        placement_method="grid",
        use_cinematic_model=True,
        object_sizes=[0.05],
        infant_action_names=infant_action_names,
    )
    
    # RJAWrapperのパラメータ設定
    env = RJAWrapper(
        base,
        parent_cue=fixed_parent_cue or "pointing",
        function="heuristic",  # "heuristic" or "affective"
        # 報酬関連パラメータ
        zero_phase_reward_weight=0.1,
        first_phase_reward_weight_for_eye=0.2,
        first_phase_reward_weight_for_index=0.2,
        first_phase_reward_weight_for_hand=0.2,
        second_phase_reward_weight=0.3,
        approach_distance_for_point = 1.0,
        approach_distance_for_grasp=0.5,
        approach_distance_for_show=0.8,
        success_bonus=1.0,
        # 絶対的報酬パラメータ
        eye_contact_reward=0.1,
        gaze_ip_reward=0.05,
        gaze_obj_ip_reward=0.08,
        voice_reward=0.02,
        step_penalty=-0.001,
        # フェーズ継続条件
        eye_contact_min_steps=8,
        parent_cue_min_steps=8,
        infant_follow_min_steps=8,
        max_phase_steps = 20,
        # 親の行動制御パラメータ
        voice_level=1.0,
        shake_period=6,
        shake_amp=0.8
    )

    # 収集用バッファ
    infant_movie, parent_movie = [], []

    # 注釈オプション（wrapper_test.pyと同等）
    annot_opts = {
        "text_scale": 0.1,
        "padding": 4,
        "box_alpha": 0.9,
    }

    obs, info = env.reset(seed=42)
    # 初期フレーム
    if "views" in (info or {}):
        if enable_annotation:
            infant_annot = annotated_view_from_info(info, "infant_rgb", **annot_opts)
            #parent_annot = annotated_view_from_info(info, "parent_rgb", **annot_opts)
        else:
            infant_annot = np.array(info["views"].get("infant_rgb"))
            #parent_annot = np.array(info["views"].get("parent_rgb"))
        if infant_annot is not None:
            infant_movie.append(infant_annot)
        # if parent_annot is not None:
        #     parent_movie.append(parent_annot)

    # 実行ループ
    max_frames = 1000
    t = 0
    current_phase = "phase0"  # フェーズを追跡
    
    while t < max_frames:
        # フェーズの更新
        if "ja_phase" in info:
            current_phase = info["ja_phase"]
        
        if random_infant:
            # 幼児アクション: ランダム
            dim = len(infant_action_names) if infant_action_names is not None else env.action_num
            infant_action = np.random.uniform(-1.0, 1.0, dim).astype(np.float32)
        else:
            # フェーズに応じて幼児の行動を切り替え
            if current_phase == "phase0" or current_phase == "phase1":
                # phase0, phase1: 親を見る
                if t > 10:
                    infant_action = _get_agent_look_at_target_action(
                        env, env.infant, env.parent, infant_action_names
                    )
                else:
                    infant_action = np.zeros(len(infant_action_names) if infant_action_names is not None else env.action_num, dtype=np.float32)
            elif current_phase == "phase2":
                # phase2: ターゲットオブジェクトを見る
                target_obj = info.get("parent_target_obj")
                if target_obj is None:
                    target_obj = env.parent_target_obj
                if target_obj is not None:
                    infant_action = _get_agent_look_at_object_action(
                        env, env.infant, target_obj, infant_action_names
                    )
                else:
                    
                    # ターゲットが不明な場合は親を見続ける
                    infant_action = _get_agent_look_at_target_action(
                        env, env.infant, env.parent, infant_action_names
                    )
            else:
                # その他のフェーズ（success等）: 何もしない
                infant_action = np.zeros(len(infant_action_names) if infant_action_names is not None else env.action_num, dtype=np.float32)

        # 親アクションはRJAWrapper内で自動生成される
        parent_action = info["parent_action"]
        
        # 2エージェント分のアクションでステップ
        obs, reward, done, trunc, info = env.step((infant_action, parent_action))

        # フレーム保存（注釈付き or 生画像）
        if "views" in info:
            if enable_annotation:
                infant_annot = annotated_view_from_info(info, "infant_rgb", **annot_opts)
                #parent_annot = annotated_view_from_info(info, "parent_rgb", **annot_opts)
            else:
                infant_annot = np.array(info["views"].get("infant_rgb"))
                #parent_annot = np.array(info["views"].get("parent_rgb"))
            if infant_annot is not None:
                infant_movie.append(infant_annot)
            # if parent_annot is not None:
            #     parent_movie.append(parent_annot)

        # 進捗ログ - RJAWrapperの情報キーに合わせて更新
        if t % 25 == 0:
            phase = info.get("ja_phase")
            target_obj = info.get("parent_target_obj")
            infant_reward = reward[0] if isinstance(reward, list) else reward
            print(f"t={t:04d} phase={phase} target_obj={target_obj} r_infant={infant_reward:.3f}")

        # エピソード管理
        if done or trunc:
            obs, info = env.reset()
            print(f"--- reset at t={t+1} ---")
            if "views" in (info or {}):
                if enable_annotation:
                    infant_annot = annotated_view_from_info(info, "infant_rgb", **annot_opts)
                    #parent_annot = annotated_view_from_info(info, "parent_rgb", **annot_opts)
                else:
                    infant_annot = np.array(info["views"].get("infant_rgb"))
                    #parent_annot = np.array(info["views"].get("parent_rgb"))
                if infant_annot is not None:
                    infant_movie.append(infant_annot)
                # if parent_annot is not None:
                # #     parent_movie.append(parent_annot)

        t += 1

    env.close()

    # 動画保存
    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)

    prefix = str(fixed_parent_cue) if fixed_parent_cue else "default"
    if random_infant:
        prefix += "_random"

    if enable_annotation:
        print(f"save movie {save_dir}/{prefix}_anno_infant_view.mp4")
        images_to_video(infant_movie, f"{save_dir}/{prefix}_anno_infant_view.mp4", fps=12, scale=4)
        #images_to_video(parent_movie, f"{save_dir}/{prefix}_anno_parent_view.mp4", fps=12, scale=4)
    else:
        images_to_video(infant_movie, f"{save_dir}/{prefix}_infant_view.mp4", fps=12, scale=4)
        #  images_to_video(parent_movie, f"{save_dir}/{prefix}_parent_view.mp4", fps=24, scale=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-annot", action="store_true", help="注釈を無効化")
    parser.add_argument("--cue", type=str, default=None, help="親キューを固定（例: gaze,pointing,shakingなど）")
    parser.add_argument("--random-infant", action="store_true", help="幼児をランダムに動かす")
    parser.add_argument("--infant-actions", nargs="+", default=None, help="幼児の行動空間を指定（例: torso head_tilt voice）")
    args = parser.parse_args()
    main(enable_annotation=not args.no_annot, fixed_parent_cue=args.cue, random_infant=args.random_infant, infant_action_names=args.infant_actions)
