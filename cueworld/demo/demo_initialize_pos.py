import numpy as np
from cueworld.base import BaseEnv
from cueworld.video import images_to_video
import os
from cueworld.paths import ABS_PATH


def main():
    print("Starting Fixed Object Initialization Demo...")

    arena_half_extents = (2.0, 1.5)
    arena_rows = 4
    arena_cols = 5
    xh, yh = arena_half_extents
    cell_h = (2 * yh) / arena_rows

    # row=0 / row=max は固定オブジェクト専用。エージェント・可動オブジェクトは中間行のみ。
    test_layout = [
        ["_", "_", "_", "_", "_"],
        ["_", "i", "_", "p", "_"],
        ["_", "_", "x", "x", "_"],
        ["_", "_", "_", "_", "_"],
    ]

    row0_y = -yh + cell_h * 0.5
    row_max_y = -yh + cell_h * (arena_rows - 0.5)

    fixed_placements = [
        # 北壁（row=max）に1枚だけ。壁面と一体化した Board パネル。
        {"class": "Board", "x": 0.0, "y": yh - 0.05, "z": 0.5, "yaw": 90.0},
        {"class": "Red_Board", "x": 1.6, "y": row0_y, "z": 0.5, "yaw": 0.0},
        {"class": "Blue_Board", "x": -1.6, "y": row_max_y, "z": 0.5, "yaw": 180.0},
        {"class": "Green_Board", "x": 1.6, "y": row0_y, "z": 0.5, "yaw": 90.0},
    ]

    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=2,
        object_counts={"RedCube": 1, "BlueSphere": 1},
        max_steps=150,
        return_views=True,
        view_width=256,
        view_height=256,
        action_repeat=1,
        arena_half_extents=arena_half_extents,
        arena_rows=arena_rows,
        arena_cols=arena_cols,
        layout_grid=test_layout,
        fixed_placements=fixed_placements,
        debug_info=True,
        placement_method="grid",
        use_cinematic_model=True,
    )

    infant_movie = []

    num_episodes = 2
    for ep in range(num_episodes):
        print(f"--- Episode {ep+1}/{num_episodes} ---")
        obs, info = base.reset(seed=42 + ep)

        if "views" in (info or {}) and "infant_rgb" in info["views"]:
            infant_movie.append(info["views"]["infant_rgb"])

        max_frames = 150
        t = 0

        infant_action = np.zeros(base.action_space[0].shape, dtype=np.float32)
        parent_action = np.zeros(base.action_space[1].shape, dtype=np.float32)

        while t < max_frames:
            infant_action[base.action_map["torso_joint"]] = 0.8
            obs, reward, done, trunc, info = base.step((infant_action, parent_action))

            if "views" in info and "infant_rgb" in info["views"]:
                infant_movie.append(info["views"]["infant_rgb"])

            if t % 50 == 0:
                print(f"t={t:03d}")

            if done or trunc:
                break
            t += 1

    base.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    video_path = f"{save_dir}/demo_initialize_pos.mp4"
    print(f"Saving video to {video_path}...")
    images_to_video(infant_movie, video_path, fps=12, scale=4)
    print("Done!")


if __name__ == "__main__":
    main()
