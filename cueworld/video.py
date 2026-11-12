"""デモ用のフレーム列を mp4 にする。"""
from PIL import Image
import imageio
import numpy as np


def images_to_video(image_list, output_file, fps=30, scale=None, size=None):
    """
    PillowのImage型の配列を動画に変換する
    Args:
        image_list (list of Image): PillowのImageオブジェクトのリスト
        output_file (str): 出力する動画ファイル名
        fps (int): 動画のフレームレート
    """
    # PillowのImage型をNumpy配列に変換
    processed_images = []
    for img in image_list:
        # --- torch.Tensor対応 ---
        if hasattr(img, '__torch_function__'):  # torch.Tensorの判定
            img = img.detach().cpu().numpy()
            # PyTorchのテンソルは(C, H, W)形式なので(H, W, C)に変換
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
        # --- NumPy配列ならPILに変換 ---
        if isinstance(img, np.ndarray):
            arr = img
            # dtype 正規化（floatは0-1想定、超えてたらクリップ）
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0 + 0.5).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            # チャンネル整形
            if arr.ndim == 2:                 # (H,W) -> L
                img = Image.fromarray(arr, mode="L")
            elif arr.ndim == 3 and arr.shape[2] == 1:
                img = Image.fromarray(arr.squeeze(-1), mode="L")
            elif arr.ndim == 3 and arr.shape[2] in (3, 4):
                img = Image.fromarray(arr)     # RGB/RGBA
            else:
                raise ValueError(f"Unsupported array shape: {arr.shape}")
        elif isinstance(img, Image.Image):
            pass  # そのまま
        else:
            raise TypeError(f"Unsupported frame type: {type(img)}")

        # 透過PNGを黒背景で合成（必要なら色を変えてOK）
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (0, 0, 0))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        if scale is not None:
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        elif size is not None:
            img = img.resize(size, Image.Resampling.LANCZOS)
        processed_images.append(img)

    # 動画に保存
    with imageio.get_writer(output_file, fps=fps, codec='libx264') as writer:
        for img in processed_images:
            # PIL ImageをNumPy配列に変換してwriterに渡す
            frame = np.array(img)
            writer.append_data(frame)
