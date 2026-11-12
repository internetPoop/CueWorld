import numpy as np
from cueworld.base import BaseEnv
from cueworld.dyadic_wrappers import DyadicWrapper
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

    # 表示優先キー（ターンテーキングタスク用）
    keys_order = [
        "tt_phase",
        "tt_steps", 
        "tt_phase2_total_steps",
        "infant_reward",
        "gaze_score",
        "eye_contact_score",
        "angle_to_parent",
        "parent_action",
    ]
    # 1) 優先キー
    lines: list[str] = []
    for k in keys_order:
        if k in info:
            v = info[k]
            if isinstance(v, float):
                lines.append(f"{k}={v:.3f}")
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
            if isinstance(v, (list, tuple)) and len(v) <= 8:
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
    try:
        eye, fwd = env._agent_eye_pose(agent)
        target_pos = p.getLinkState(target_agent["body"], target_agent["eyes_link"], computeForwardKinematics=True)[4]
        target_pos = np.array(target_pos, dtype=np.float32)
        v = target_pos - eye
        
        # yawとpitchの計算
        fwd_yaw = np.array([fwd[0], fwd[1]], dtype=np.float32)
        to_target_yaw = np.array([v[0], v[1]], dtype=np.float32)
        yaw_diff = env.rotation_to_target(fwd_yaw, to_target_yaw)
        
        fwd_xy = np.sqrt(fwd[0]**2 + fwd[1]**2)
        fwd_pitch = np.array([fwd[2], fwd_xy], dtype=np.float32)
        to_target_xy = np.sqrt(v[0]**2 + v[1]**2)
        to_target_pitch = np.array([v[2], to_target_xy], dtype=np.float32)
        pitch_diff = env.rotation_to_target(fwd_pitch, to_target_pitch)
        
        # action_repeatを考慮したステップサイズ
        action_repeat = getattr(env, "action_repeat", 1)
        torso_step = env.torso_step * action_repeat
        head_step = env.head_step * action_repeat
        
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
                    # フェーズに応じた発話（後で設定）
                    action[i] = 0.0
            return action
        else:
            # フルアクション（親用）
            action = np.zeros(env.action_num, dtype=np.float32)
            action[env.action_map["torso_joint"]] = torso_action
            action[env.action_map["head_tilt_joint"]] = pitch_action
            return action
    except Exception as e:
        print(f"Warning: Failed to generate look-at action: {e}")
        # フォールバック: ゼロアクション
        if action_names is not None:
            return np.zeros(len(action_names), dtype=np.float32)
        else:
            return np.zeros(env.action_num, dtype=np.float32)


def main(enable_annotation: bool = True, random_infant: bool = False, head_shake_infant: bool = False, infant_action_names: list[str] | None = None):
    # ベース環境 + ターンテーキング報酬ラッパ（親ポリシーも統合済み）
    base = BaseEnv(
        gui=False,
        num_agents=2,
        return_views=True,
        view_width=64,
        view_height=64,
        action_repeat=24,
        debug_info=True,
        use_cinematic_model=True,
        infant_action_names=infant_action_names,
        random_parent_pos=True,
        layout_grid = [["_", "_","xp","xp"],["i", "_","xp","xp"],["_", "_","xp","xp"]]
    )
    
    # 新しい統合ラッパーを適用
    env = DyadicWrapper(base,
        # 報酬パラメータの調整
        shaping_weight_voice=0.3,
        gaze_angle_reward_weight=0.5,
        step_penalty=-0.002,
        success_bonus=2.0,
        wrong_voice_penalty=-0.1,
        
        # 絶対的報酬設定
        eye_contact_reward=0.1,
        gaze_ip_reward=0.05,
        gaze_face_ip_reward=0.03,
        gaze_p_reward=0.01,
        
        # フェーズ遷移条件
        voice_thresh=0.5,
        eye_contact_min_steps=15,
        parent_call_min_steps=15,
        infant_response_min_steps=15,
        waiting_phase_timeout=8,
        max_phase_steps=40,
        
        terminate_on_success=True,  # 成功後すぐ終了
        done_episode_num=1,          # 1回成功したら終了
        target_angle= None,
        
        # 親の行動制御パラメータ
        approach_distance=2.0,
    )

    # 収集用バッファ
    infant_movie, parent_movie = [], []

    # 注釈オプション
    annot_opts = {
        "text_scale": 0.12,
        "padding": 4,
        "box_alpha": 0.85,
    }

    obs, info = env.reset()
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
    max_frames = 1000  # 長めに設定（複数回の成功を観測するため）
    t = 0
    success_count = 0
    
    # 首振り用の状態変数
    shake_direction = 1.0  # 1.0で右向き、-1.0で左向き
    shake_step_count = 0
    shake_duration = 15    # 片方向への振り続け時間（フレーム数）
    
    while t < max_frames:
        if random_infant:
            # 幼児アクション: ランダム
            dim = len(infant_action_names) if infant_action_names is not None else env.action_num
            infant_action = np.random.uniform(-1.0, 1.0, dim).astype(np.float32)
        elif head_shake_infant:
            # 幼児アクション: 首振り
            dim = len(infant_action_names) if infant_action_names is not None else env.action_num
            infant_action = np.zeros(dim, dtype=np.float32)
            
            # 首振りロジック
            shake_step_count += 1
            if shake_step_count >= shake_duration:
                # 方向転換
                shake_direction *= -1.0
                shake_step_count = 0
            
            # 首の動きを設定（torso_jointを使用）
            shake_intensity = 0.8  # 振りの強さ
            if infant_action_names is not None:
                for i, name in enumerate(infant_action_names):
                    if name == "torso_joint":
                        infant_action[i] = shake_direction * shake_intensity
                    elif name == "voice":
                        # 時々発話
                        infant_action[i] = 1.0 if random.random() < 0.05 else 0.0
                    else:
                        infant_action[i] = 0.0
            else:
                # フル次元の場合（base.pyのaction_mapに対応）
                infant_action[env.action_map["torso_joint"]] = shake_direction * shake_intensity
                infant_action[env.action_map["voice"]] = 1.0 if random.random() < 0.05 else 0.0
                # その他のアクションは0に設定済み（初期化時）
        else:
            # 幼児アクション: 親の方を向く（最新のURDF構造に対応）
            infant_action = _get_agent_look_at_target_action(
                env, env.infant, env.parent, infant_action_names
            )
            
            # 新しいフェーズに応じた発話戦略
            phase = info.get("tt_phase", "phase0")
            if phase in ["phase0", "waiting_phase", "parent_phase"]:
                # 相互注視、待機、親発話フェーズでは静かに
                voice_val = 0.0
            elif phase == "infant_phase":
                # 幼児の応答フェーズでは積極的に発話
                voice_val = 1.0 if random.random() < 0.99 else 0.0
            else:
                # その他（successなど）は時々発話
                voice_val = 1.0 if random.random() < 0.01 else 0.0
            
            # voiceアクションを設定
            if infant_action_names is not None:
                for i, name in enumerate(infant_action_names):
                    if name == "voice":
                        infant_action[i] = voice_val
            else:
                infant_action[env.action_map["voice"]] = voice_val

        # 親アクションは統合ラッパ内で自動生成される
        parent_action = info.get("parent_action")
        if parent_action is None:
            parent_action = np.zeros(env.action_num, dtype=np.float32)
        else:
            parent_action = np.asarray(parent_action, dtype=np.float32)

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

        # 進捗ログ（新しい情報キーに対応）
        if t % 25 == 0:
            phase = info.get("tt_phase", "unknown")
            steps = info.get("phase_steps", 0)
            success_num = info.get("success", 0)
            mode = "random" if random_infant else ("head_shake" if head_shake_infant else "look_at_parent")
            print(f"t={t:04d} phase={phase} steps={steps} r_infant={reward[0]:.3f} success={success_num} mode={mode}")

        # エピソード管理
        if done or trunc:
            final_success_num = info.get("success", 0)
            print(f"Episode finished at t={t}")
            obs, info = env.reset()
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
    final_success_num = info.get("success", 0)
    print(f"Demo completed! Final success count: {final_success_num}")

    # 動画保存
    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)

    prefix = "turntaking_bidirectional"  # 双方向ターンテーキング対応を示すprefix
    if random_infant:
        prefix += "_random"
    elif head_shake_infant:
        prefix += "_headshake"

    if enable_annotation:
        images_to_video(infant_movie, f"{save_dir}/{prefix}_anno_infant_view.mp4", fps=24, scale=4)
        #images_to_video(parent_movie, f"{save_dir}/{prefix}_anno_parent_view.mp4", fps=24, scale=4)
        print(f"Saved annotated videos to {save_dir}/{prefix}_anno_*_view.mp4")
    else:
        images_to_video(infant_movie, f"{save_dir}/{prefix}_infant_view.mp4", fps=24, scale=4)
        #images_to_video(parent_movie, f"{save_dir}/{prefix}_parent_view.mp4", fps=24, scale=4)
        print(f"Saved videos to {save_dir}/{prefix}_*_view.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-annot", action="store_true", help="注釈を無効化")
    parser.add_argument("--random-infant", action="store_true", help="幼児をランダムに動かす")
    parser.add_argument("--head-shake-infant", action="store_true", help="幼児が首を左右に振り続ける")
    parser.add_argument("--infant-actions", nargs="+", default=None, 
                       help="幼児の行動空間を指定（例: torso_joint head_tilt_joint voice）")
    args = parser.parse_args()
    
    # random_infantとhead_shake_infantの同時指定をチェック
    if args.random_infant and args.head_shake_infant:
        print("Warning: --random-infant と --head-shake-infant が同時に指定されました。--head-shake-infant を優先します。")
        args.random_infant = False
    
    main(
        enable_annotation=not args.no_annot, 
        random_infant=args.random_infant,
        head_shake_infant=args.head_shake_infant,
        infant_action_names=args.infant_actions
    )
