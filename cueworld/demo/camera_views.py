"""デモ用の三人称カメラ描画。"""

import math

import numpy as np
import pybullet as p

# 親子の初期配置距離 (~3.8m) を踏まえ、両者の全身が枠内に収まる値。
_THIRD_SIDE_DISTANCE = 2.0
_THIRD_TARGET_Z = 0.55
_THIRD_EYE_Z = 1.4
_THIRD_FOV = 76.0


def get_third_person_view(
    env,
    width: int = 256,
    height: int = 256,
    *,
    hide_walls: bool = True,
    hide_boards: bool = True,
    orbit_deg: float = 0.0,
    distance: float | None = None,
) -> np.ndarray | None:
    """親子の中点を注視する三人称視点。両エージェントの全身が映るよう調整。

    hide_walls / hide_boards が True なら、衝突はそのまま、アリーナ壁と board の
    visual だけ撮るあいだ消す。幼児／親の観察カメラは env.step 側で撮るので、
    そちらでは壁も board も見える。
    orbit_deg は親子を結ぶ軸まわりのカメラ回転（度）。0 は真横。
    distance は中点からの水平距離。None なら既定の真横用距離。
    """
    wall_setter = getattr(env, "set_wall_visual_visible", None) if hide_walls else None
    board_setter = getattr(env, "set_platform_visual_visible", None) if hide_boards else None
    try:
        if callable(wall_setter):
            wall_setter(False)
        if callable(board_setter):
            board_setter(False)
        client_id = env.client_id
        parent_pos, _ = p.getBasePositionAndOrientation(
            env.parent["body"], physicsClientId=client_id
        )
        infant_pos, _ = p.getBasePositionAndOrientation(
            env.infant["body"], physicsClientId=client_id
        )
        mid = 0.5 * (np.asarray(parent_pos, dtype=np.float32) + np.asarray(infant_pos, dtype=np.float32))
        diff = np.asarray(parent_pos, dtype=np.float32)[:2] - np.asarray(infant_pos, dtype=np.float32)[:2]
        if float(np.linalg.norm(diff)) < 1e-3:
            side = np.array([0.0, 1.0], dtype=np.float32)
        else:
            side = np.array([-diff[1], diff[0]], dtype=np.float32)
            side = side / float(np.linalg.norm(side))
        if orbit_deg:
            ang = math.radians(float(orbit_deg))
            c, s = math.cos(ang), math.sin(ang)
            side = np.array(
                [c * side[0] - s * side[1], s * side[0] + c * side[1]],
                dtype=np.float32,
            )
        eye = np.array(
            [
                mid[0] + side[0] * _THIRD_SIDE_DISTANCE,
                mid[1] + side[1] * (_THIRD_SIDE_DISTANCE if distance is None else float(distance)),
                _THIRD_EYE_Z,
            ],
            dtype=np.float32,
        )
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=[float(mid[0]), float(mid[1]), _THIRD_TARGET_Z],
            cameraUpVector=[0, 0, 1],
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=_THIRD_FOV,
            aspect=float(width) / float(height),
            nearVal=0.1,
            farVal=20.0,
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client_id,
        )
        img = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
        return img
    except Exception as e:
        print(f"[warn] third-person view failed: {e}")
        return None
    finally:
        if callable(wall_setter):
            wall_setter(True)
        if callable(board_setter):
            board_setter(True)
