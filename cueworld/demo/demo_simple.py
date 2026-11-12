import numpy as np
from cueworld.base import BaseEnv
from cueworld.video import images_to_video
import os
from cueworld.paths import ABS_PATH

def main():
    print("Starting Simple Still Demo...")
    # 総数とnum_objectsを一致させることで確実な配置を保証
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=3,
        object_counts={"RedCube": 1, "BlueSphere": 1, "YellowCylinder": 1},
        max_steps=200,
        return_views=True,
        view_width=256,
        view_height=256,
        action_repeat=1,
        arena_half_extents=(1.6, 1.2),
        debug_info=True,
        placement_method="grid",
        use_cinematic_model=True
    )
    
    infant_movie = []
    obs, info = base.reset(seed=42)

    if "views" in (info or {}) and "infant_rgb" in info["views"]:
        infant_movie.append(info["views"]["infant_rgb"])

    max_frames = 200
    t = 0
    
    # 静止させるため、アクションはゼロベクトルのままにする
    infant_action = np.zeros(base.action_space[0].shape, dtype=np.float32)
    parent_action = np.zeros(base.action_space[1].shape, dtype=np.float32)

    while t < max_frames:
        # 幼児が体をぐるっと回す（torso_jointを一定速度で動かし続ける）
        infant_action[base.action_map["torso_joint"]] = 1.0
        obs, reward, done, trunc, info = base.step((infant_action, parent_action))

        if "views" in info and "infant_rgb" in info["views"]:
            infant_movie.append(info["views"]["infant_rgb"])

        if t % 50 == 0:
            print(f"t={t:03d}")

        if done or trunc:
            print("Episode finished early.")
            break

        t += 1

    base.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    video_path = f"{save_dir}/demo_simple.mp4"
    print(f"Saving video to {video_path}...")
    images_to_video(infant_movie, video_path, fps=12, scale=4)
    print("Done!")

if __name__ == "__main__":
    main()
