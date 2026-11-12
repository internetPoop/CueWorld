import pybullet as p
import pybullet_data
import time
import numpy as np
import os

from cueworld.video import images_to_video
from cueworld.paths import ABS_PATH

def get_full_aabb(body_id):
    """
    モデル全体のすべてのリンクを加味したAABBを計算して返す
    """
    # まずベースリンク（linkIndex = -1）のAABBを取得して初期値にする
    aabb_min, aabb_max = p.getAABB(body_id, -1)
    global_min = list(aabb_min)
    global_max = list(aabb_max)
    
    # モデルの全リンク（関節）の数を取得
    num_joints = p.getNumJoints(body_id)
    
    # 各リンクのAABBを取得し、全体の最小・最大座標を更新する
    for i in range(num_joints):
        link_min, link_max = p.getAABB(body_id, i)
        for j in range(3): # X, Y, Z軸それぞれで比較
            if link_min[j] < global_min[j]:
                global_min[j] = link_min[j]
            if link_max[j] > global_max[j]:
                global_max[j] = link_max[j]
                
    return global_min, global_max

def draw_aabb(body_id, line_ids=None):
    """
    全体のAABB（バウンディングボックス）を描画・更新する関数
    カメラ画像に映し込むため、addUserDebugLineの代わりに円柱（Cylinder）を使用します。
    """
    # 古いCylinderのボディがあれば削除する
    if line_ids is not None:
        for bid in line_ids:
            p.removeBody(bid)

    global_min, global_max = get_full_aabb(body_id)
    # 修正した関数を呼び出す
    x_min, y_min, z_min = global_min
    x_max, y_max, z_max = global_max
    
    # バウンディングボックスの8つの頂点を定義
    pts = [
        [x_min, y_min, z_min], # 0: 底面
        [x_max, y_min, z_min], # 1
        [x_max, y_max, z_min], # 2
        [x_min, y_max, z_min], # 3
        [x_min, y_min, z_max], # 4: 上面
        [x_max, y_min, z_max], # 5
        [x_max, y_max, z_max], # 6
        [x_min, y_max, z_max]  # 7
    ]
    
    # 線で結ぶ頂点のペアと軸方向を定義
    edges = [
        (0, 1, 'x'), (1, 2, 'y'), (2, 3, 'x'), (3, 0, 'y'), # 底面の4辺
        (4, 5, 'x'), (5, 6, 'y'), (6, 7, 'x'), (7, 4, 'y'), # 上面の4辺
        (0, 4, 'z'), (1, 5, 'z'), (2, 6, 'z'), (3, 7, 'z')  # 側面（上下を繋ぐ4辺）
    ]
    
    color = [1, 0, 0, 1] # 線の色（赤、RGBA）
    radius = 0.02        # 円柱の太さ
    
    new_line_ids = []
    for (i, j, axis) in edges:
        p1 = np.array(pts[i])
        p2 = np.array(pts[j])
        
        # 中心座標と長さを計算
        center = (p1 + p2) / 2.0
        length = np.linalg.norm(p2 - p1)
        
        if length == 0:
            continue
            
        # 軸に応じた姿勢（Quaternion）を決定（初期状態の円柱はZ軸向き）
        if axis == 'x':
            orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
        elif axis == 'y':
            orn = p.getQuaternionFromEuler([-np.pi/2, 0, 0])
        else: # 'z'
            orn = [0, 0, 0, 1]
            
        # 円柱のVisual Shapeを作成
        vis_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=length,
            rgbaColor=color
        )
        
        # Visual Shapeを静的ボディとして配置
        edge_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis_shape,
            basePosition=center,
            baseOrientation=orn
        )
        
        new_line_ids.append(edge_body)
        
    return new_line_ids

# ====== メイン処理 ======
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)
plane_id = p.loadURDF("plane.urdf")
obj_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 2.0])

# 初回のバウンディングボックス描画
bbox_line_ids = draw_aabb(obj_id)

# Camera settings
viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[2.5, 2.5, 2],
    cameraTargetPosition=[0, 0, 0.5],
    cameraUpVector=[0, 0, 1])
projectionMatrix = p.computeProjectionMatrixFOV(
    fov=60.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=100.0)

frames = []
print("Rendering frames...")

for _ in range(100):
    p.stepSimulation()
    
    # オブジェクトの動きに合わせてAABBの線を更新
    bbox_line_ids = draw_aabb(obj_id, line_ids=bbox_line_ids)
    
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=256, 
        height=256,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix)
    
    img = np.reshape(rgbImg, (height, width, 4))[:, :, :3]
    img = img.astype(np.uint8)
    frames.append(img)

p.disconnect()

# Save video
save_dir = f"{ABS_PATH}/results/3d_test"
os.makedirs(save_dir, exist_ok=True)
video_path = f"{save_dir}/getAABB.mp4"
print(f"Saving video to {video_path}...")
images_to_video(frames, video_path, fps=24, scale=4)
print("Done!")
