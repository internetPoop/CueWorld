import pybullet as p
import numpy as np
import time
import math
from cueworld.base import BaseEnv
from cueworld.video import images_to_video
import os
from cueworld.paths import ABS_PATH
import cv2
import argparse

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

    # 表示優先キー（IKテスト用情報）
    keys_order = [
        "frame",
        "target_distance",
        "ee_position",
        "target_position",
        "ik_error",
        "joint_angles"
    ]
    
    # 1) 優先キー
    lines: list[str] = []
    for k in keys_order:
        if k in info:
            v = info[k]
            if isinstance(v, float):
                lines.append(f"{k}={v:.3f}")
            elif isinstance(v, np.ndarray):
                if len(v) <= 3:
                    formatted = ",".join(f"{x:.3f}" for x in v)
                    lines.append(f"{k}=[{formatted}]")
                else:
                    lines.append(f"{k}=array({v.shape})")
            else:
                lines.append(f"{k}={v}")
    
    # 2) それ以外（views を除く）も追記（重複除外）
    for k, v in info.items():
        if k == "views" or k in keys_order:
            continue
        if isinstance(v, np.ndarray):
            lines.append(f"{k}=ndarray{list(v.shape)}")
        else:
            # アクションなどのベクトル/タプルは短く整形
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

def main(enable_annotation: bool = True):
    # 1. BaseEnv環境の初期化（ヘッドレスモード）
    env = BaseEnv(
        gui=False,
        num_agents=2,
        return_views=True,
        view_width=128,
        z_box = 0.5,
        view_height=128,
        action_repeat=1,
        debug_info=True,
        use_cinematic_model=True
    )
    
    obs, info = env.reset()
    
    
    # 2. 動画保存用バッファ
    infant_movie, parent_movie = [], []
    
    # 注釈オプション
    annot_opts = {
        "text_scale": 0.8,
        "padding": 4,
        "box_alpha": 0.9,
    }
    
    # 初期フレーム保存
    if "views" in info:
        if enable_annotation:
            # IKテスト用の情報を追加
            info["frame"] = 0
            info["target_distance"] = 0.0
            info["ee_position"] = np.array([0, 0, 0])
            info["target_position"] = np.array([0, 0, 0])
            info["ik_error"] = 0.0
            info["joint_angles"] = np.array([0, 0, 0, 0, 0])
            
            infant_annot = annotated_view_from_info(info, "infant_rgb", **annot_opts)
            parent_annot = annotated_view_from_info(info, "parent_rgb", **annot_opts)
        else:
            infant_annot = np.array(info["views"]["infant_rgb"])
            parent_annot = np.array(info["views"]["parent_rgb"])
        
        if infant_annot is not None:
            infant_movie.append(infant_annot)
        if parent_annot is not None:
            parent_movie.append(parent_annot)
    
    # 3. 親エージェントのIK関連情報を取得
    parent_body = env.parent_body
    
    # エンドエフェクタ（手先）のリンクインデックスを取得
    target_link_name = "hand_link"
    ee_link_index = -1
    
    # すべての可動関節をIK対象とする
    movable_joint_indices = []
    num_joints = p.getNumJoints(parent_body)
    print(f"Parent total joints: {num_joints}")
    
    for i in range(num_joints):
        info_joint = p.getJointInfo(parent_body, i)
        joint_name = info_joint[1].decode("utf-8")
        joint_type = info_joint[2]
        link_name = info_joint[12].decode("utf-8")
        
        if link_name == target_link_name:
            ee_link_index = i
            print(f"Found End Effector: {target_link_name} at ID {i}")
        
        # すべての可動関節（REVOLUTE）を対象とする
        if joint_type == p.JOINT_REVOLUTE:
            movable_joint_indices.append(i)
            print(f"Movable joint {i}: {joint_name}")
    if ee_link_index == -1:
        print(f"Error: Link {target_link_name} not found.")
        return
    
    # 4. IK計算用のパラメータ設定（腕関節のみ）
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    rest_poses = []
    
    for i in movable_joint_indices: #TODO: 動かしたくない関節はlimitで固定したい
        info_joint = p.getJointInfo(parent_body, i)
        lower_limits.append(info_joint[8])
        upper_limits.append(info_joint[9])
        joint_ranges.append(info_joint[9] - info_joint[8])
        rest_poses.append(0)
    
    # 5. 環境内のオブジェクトを目標として使用
    if not env.objects:
        print("No objects found in environment!")
        return
    
    # 最初のオブジェクトを目標として選択
    target_object = env.objects[1]
    print(f"Target object ID: {target_object}")
    target_pos = p.getBasePositionAndOrientation(target_object)[0]
    #target_pos = (target_pos+ee_pos) / 2.0
    t = 0
    max_frames = 300
    print("Start IK Simulation...")
    joint_poses = p.calculateInverseKinematics(
            parent_body,
            ee_link_index,
            target_pos,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=50,
            residualThreshold=1e-4
        )
    
    while t < max_frames:
        t += 1
        
        # --- 目標座標の取得（環境内オブジェクトの位置） ---
        
        # 目標までの距離を計算
        
        
        # --- Inverse Kinematics (IK) の計算 ---

        
        # --- 親のIK結果をすべての可動関節に反映 ---
        parent_action = np.zeros(env.action_num, dtype=np.float32)
        
        # 現在の関節角度を取得してIK結果を適用
        current_joint_angles = []
        for i, joint_idx in enumerate(movable_joint_indices):
            if i < len(joint_poses):
                current_angle = p.getJointState(parent_body, joint_idx)[0]
                current_joint_angles.append(current_angle)
                target_angle = joint_poses[i]
                angle_diff = target_angle - current_angle
                
                # 関節名を取得してアクション配列にマッピング
                joint_info = p.getJointInfo(parent_body, joint_idx)
                joint_name = joint_info[1].decode("utf-8")
                
                # URDFのすべての可動関節に対応
                if "torso_joint" in joint_name:
                    # torso_jointは通常移動に含まれるため、必要に応じて対応
                    pass  # 必要であれば実装
                elif "head_pan_joint" in joint_name:
                    if hasattr(env, 'action_map') and "neck_yaw" in env.action_map:
                        parent_action[env.action_map["neck_yaw"]] = np.clip(angle_diff / env.neck_step, -1, 1)
                elif "head_tilt_joint" in joint_name:
                    if hasattr(env, 'action_map') and "neck_pitch" in env.action_map:
                        parent_action[env.action_map["neck_pitch"]] = np.clip(angle_diff / env.neck_step, -1, 1)
                elif "left_lip_joint" in joint_name or "right_lip_joint" in joint_name:
                    # 口の動きは通常IKでは制御しない
                    pass
                elif "shoulder_yaw_joint" in joint_name:
                    parent_action[env.action_map["shoulder_yaw_joint"]] = np.clip(angle_diff / env.shoulder_step, -1, 1)
                elif "shoulder_pitch_joint" in joint_name:
                    parent_action[env.action_map["shoulder_pitch_joint"]] = np.clip(angle_diff / env.shoulder_step, -1, 1)
                elif "elbow_joint" in joint_name:
                    parent_action[env.action_map["elbow_joint"]] = np.clip(angle_diff / env.elbow_step, -1, 1)
                elif "roll_wrist_joint" in joint_name:
                    parent_action[env.action_map["roll_wrist_joint"]] = np.clip(angle_diff / env.wrist_step, -1, 1)
                elif "pitch_wrist_joint" in joint_name:
                    parent_action[env.action_map["pitch_wrist_joint"]] = np.clip(angle_diff / env.wrist_step, -1, 1)
                elif "index_joint" in joint_name:
                    # 指の関節は通常IKでは制御しないが、必要であれば実装
                    pass
                elif "thumb_joint" in joint_name:
                    # 指の関節は通常IKでは制御しないが、必要であれば実装
                    pass
                elif "little_finger_joint" in joint_name:
                    # 指の関節は通常IKでは制御しないが、必要であれば実装
                    pass
        
        # 幼児は全てゼロ（フリーズ）
        infant_action = np.zeros(len(env.infant_action_indices), dtype=np.float32)
        
        # --- 環境ステップ実行 ---
        obs, reward, done, trunc, info = env.step((infant_action, parent_action))
        
        
        # IKテスト用の情報を追加
        if enable_annotation and "views" in info:
            info["frame"] = t
            info["target_position"] = np.array(target_pos)
            info["joint_angles"] = np.array(current_joint_angles[:5])  # 最初の5つの関節角度を表示
            
            infant_annot = annotated_view_from_info(info, "infant_rgb", **annot_opts)
            parent_annot = annotated_view_from_info(info, "parent_rgb", **annot_opts)
        else:
            if "views" in info:
                infant_annot = np.array(info["views"]["infant_rgb"])
                parent_annot = np.array(info["views"]["parent_rgb"])
            else:
                infant_annot = None
                parent_annot = None
        
        
        # フレーム保存
        if infant_annot is not None:
            infant_movie.append(infant_annot)
        if parent_annot is not None:
            parent_movie.append(parent_annot)
        
        # 進捗表示
        if t % 50 == 0:
            print(f"Frame {t}/{max_frames}")
            
        if done or trunc:
            break
    
    env.close()
    
    # 動画保存
    save_dir = f"{ABS_PATH}/results/ik_test"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Saving videos...")
    suffix = "_annotated" if enable_annotation else ""
    images_to_video(infant_movie, f"{save_dir}/ik_test_infant_view{suffix}.mp4", fps=24, scale=4)
    images_to_video(parent_movie, f"{save_dir}/ik_test_parent_view{suffix}.mp4", fps=24, scale=4)
    
    print("IK test completed and videos saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-annot", action="store_true", help="アノテーションを無効化")
    args = parser.parse_args()
    main(enable_annotation=not args.no_annot)