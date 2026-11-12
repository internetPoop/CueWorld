"""幼児エージェントの静止テスト - 幾何学的配置モードで幼児視点の画像を保存"""

import numpy as np
import cv2
from cueworld.base import BaseEnv
from pathlib import Path
from cueworld.paths import ABS_PATH
import os


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

def main():
    """幼児が静止している状態でのテスト - 幼児視点の画像を保存"""
    
    # 環境の初期化（geometricモード）
    env = BaseEnv(
        gui=False,
        num_objects=1,
        num_agents=2,
        max_steps=200,
        placement_method="geometric",
        object_colors=["red"],
        object_shapes=["cube"],
        object_sizes=[0.15],
        z_box=0.15,
        #infant_obj_angle_range_deg=5.0,  
        #infant_obj_dist_range=1.0,
        #infant_parent_angle_range_deg=10.0,
        #infant_parent_dist_range=1.8,
        #obj_parent_angle_range_deg=None,
        obj_parent_dist_range = 1.0,
        #obj_parent_dist_range=None,
        #shielding_thresh_deg=None,
        return_views=True,
        view_fov=90,
        ssaa=2,
        debug_info=True,
    )
    
    # 出力ディレクトリの作成
    save_dir = f"{ABS_PATH}/results/view_test"
    os.makedirs(save_dir, exist_ok=True)
    
    # リセット回数
    num_resets = 10
    
    print(f"テスト開始: 幼児が静止した状態での画像保存 ({num_resets}回)")
    
    # 複数回リセットして画像を保存
    for i in range(num_resets):
        # 環境のリセット
        obs, info = env.reset(seed=i)
        
        print(f"\n[{i+1}/{num_resets}] リセット実行 (seed={i})")
        
        # 幼児視点の画像を取得
        if "views" in info and "infant_rgb" in info["views"]:
            # デバッグ情報を画像に描画
            infant_view_with_info = annotated_view_from_info(info, "infant_rgb")
            
            # 画像が取得できた場合のみ保存
            if infant_view_with_info is not None:
                # OpenCVはBGR形式なので変換
                infant_view_bgr = cv2.cvtColor(infant_view_with_info, cv2.COLOR_RGB2BGR)
                
                # 画像を保存（連番付き）
                image_path = os.path.join(save_dir, f"infant_view_static_{i:03d}.png")
                cv2.imwrite(image_path, infant_view_bgr)
                
                print(f"  画像保存完了: {image_path}")
                print(f"  画像サイズ: {infant_view_with_info.shape}")
            else:
                print(f"  エラー: annotated_view_from_info が None を返しました")
        else:
            print(f"  エラー: 画像が取得できませんでした")
    
    # リソースの解放
    env.close()
    
    print(f"\nテスト完了: {num_resets}枚の画像を保存しました")
    print(f"保存先: {save_dir}")


if __name__ == "__main__":
    main()
