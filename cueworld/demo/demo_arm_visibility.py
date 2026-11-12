import numpy as np
from cueworld.base import BaseEnv
from cueworld.video import images_to_video
import cv2
import os
import math
import pybullet as p
from cueworld.paths import ABS_PATH
from cueworld.parent_tools import ParentTools

def main():
    print("Starting Arm Visibility Demo (Smaller Arena)...")
    # BaseEnvの設定。両方のビューを取る＋アリーナを小さめにする
    base = BaseEnv(
        gui=False,
        num_agents=2,
        num_objects=1,
        max_steps=100,
        return_views=True,
        output_parent_view=True,  
        view_width=256,
        view_height=256,
        view_fov=90,
        action_repeat=1,
        arena_half_extents=(0.8, 0.8), # 親を近づけるためにアリーナを小さく
        debug_info=True,
    )
    env = ParentTools(base)

    movie = []
    obs, info = env.reset(seed=42)

    max_frames = 100
    t = 0
    pitch_flag = True
    yaw_flag = True

    while t < max_frames:
        infant_action = np.zeros(env.action_space[0].shape, dtype=np.float32)
        parent_action = np.zeros(env.action_space[1].shape, dtype=np.float32)
        
        # 1. 腕を目標角度（肩-90度、肘-40度）に近づける
        target_shoulder = -math.radians(90)
        target_elbow    = -math.radians(0)
        
        cur_shoulder = env.get_joint_angle(env.infant, "shoulder_pitch_joint")
        cur_elbow    = env.get_joint_angle(env.infant, "elbow_joint")
        
        shoulder_step = env.infant_joint_mappings["shoulder_pitch_joint"] * env.action_repeat
        elbow_step    = env.infant_joint_mappings["elbow_joint"] * env.action_repeat
        
        # 差分をステップ幅で割って -1〜1 にクリップ
        infant_action[env.action_map["shoulder_pitch_joint"]] = np.clip((target_shoulder - cur_shoulder) / shoulder_step, -1.0, 1.0)
        infant_action[env.action_map["elbow_joint"]] = np.clip((target_elbow - cur_elbow) / elbow_step, -1.0, 1.0)
        
        # # 2. 幼児が自分の手を見るように視線を計算
        # hand_pos = p.getLinkState(env.infant["body"], env.infant["hand_link"])[4]
        # eye, fwd = env._agent_eye_pose(env.infant)
        # v = np.array(hand_pos) - np.array(eye)
        
        # # Yaw角のズレ計算
        # fwd_yaw = np.array([fwd[0], fwd[1]], dtype=np.float32)
        # to_target_yaw = np.array([v[0], v[1]], dtype=np.float32)
        # yaw_diff = env.rotation_to_target(fwd_yaw, to_target_yaw)
        
        # # Pitch角のズレ計算
        # fwd_xy = np.sqrt(fwd[0]**2 + fwd[1]**2)
        # fwd_pitch = np.array([fwd[2], fwd_xy], dtype=np.float32)
        # to_target_xy = np.sqrt(v[0]**2 + v[1]**2)
        # to_target_pitch = np.array([v[2], to_target_xy], dtype=np.float32)
        # pitch_diff = env.rotation_to_target(fwd_pitch, to_target_pitch)
        # if pitch_diff < 0.01:
        #     pitch_flag = False
        # if yaw_diff < 0.01:
        #     yaw_flag = False
        
        # head_pan_step = env.infant_joint_mappings["head_pan_joint"] * env.action_repeat
        # head_tilt_step  = env.infant_joint_mappings["head_tilt_joint"] * env.action_repeat
        
        # if yaw_flag:
        #     infant_action[env.action_map["head_pan_joint"]] = np.clip(yaw_diff / head_pan_step, -1.0, 1.0)
        # else:
        #     infant_action[env.action_map["head_pan_joint"]] = -0.01
        # if pitch_flag:
        #     infant_action[env.action_map["head_tilt_joint"]] = np.clip(pitch_diff / head_tilt_step, -1.0, 1.0)
        # else:
        #     infant_action[env.action_map["head_tilt_joint"]] = -0.01
        if t>=50:
            infant_action[env.action_map["head_tilt_joint"]] = 0.01
        
        # 3. 親も幼児に注目するようにする (ParentToolsを使う)
        env.gaze_agent(env.parent, env.infant, yaw_link="torso", action=parent_action)
        
        obs, reward, done, trunc, info = env.step((infant_action, parent_action))

        if "views" in info:
            infant_img = info["views"].get("infant_rgb")
            parent_img = info["views"].get("parent_rgb")
            
            if infant_img is not None and parent_img is not None:
                # 幼児視点と親視点を横に並べる
                infant_annot = cv2.putText(np.array(infant_img), "Infant View", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                infant_annot = cv2.putText(infant_annot, "Infant View", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                parent_annot = cv2.putText(np.array(parent_img), "Parent View", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                parent_annot = cv2.putText(parent_annot, "Parent View", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                combined = np.concatenate((infant_annot, parent_annot), axis=1)
                movie.append(combined)

        if t % 20 == 0:
            print(f"t={t:03d}, cur_shoulder={math.degrees(cur_shoulder):.1f} deg, cur_elbow={math.degrees(cur_elbow):.1f} deg")

        if done or trunc:
            print("Episode finished early.")
            break

        t += 1

    env.close()

    save_dir = f"{ABS_PATH}/results/3d_test"
    os.makedirs(save_dir, exist_ok=True)
    video_path = f"{save_dir}/demo_arm_visibility_2.mp4"
    print(f"Saving video to {video_path}...")
    images_to_video(movie, video_path, fps=12, scale=2)
    print("Done!")

if __name__ == "__main__":
    main()
