import pybullet as p
import pybullet_data
import time
import numpy as np
import os

from cueworld.video import images_to_video
from cueworld.paths import ABS_PATH, PACKAGE_DIR

def main():
    # ====== メイン処理 ======
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF("plane.urdf")
    
    # RedSphere オブジェクトのURDFパスを取得して配置
    apple_urdf_path = os.path.join(PACKAGE_DIR, "objects", "assets", "red_sphere.urdf")
    
    if not os.path.exists(apple_urdf_path):
        print(f"Error: URDF file not found at {apple_urdf_path}")
        p.disconnect()
        return

    # 中心付近（少し高い位置から落とす）に配置
    apple_id = p.loadURDF(apple_urdf_path, basePosition=[0, 0, 0.5])
    
    # Camera settings (RedSphere がよく見えるように少し近寄る)
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[0.5, 0.5, 0.5],
        cameraTargetPosition=[0, 0, 0.0],
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
    video_path = f"{save_dir}/test_red_sphere.mp4"
    print(f"Saving video to {video_path}...")
    images_to_video(frames, video_path, fps=24, scale=4)
    print("Done!")

if __name__ == "__main__":
    main()
