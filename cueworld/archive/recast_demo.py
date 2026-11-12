import numpy as np
import os
import argparse
import random
from cueworld.base import BaseEnv
from cueworld.infant_signal import InfantSignalDetector
from cueworld.protocol_manager import ProtocolManager
from cueworld.parent_tools import ParentTools
from cueworld.video import images_to_video
from cueworld.paths import ABS_PATH

def main(enable_annotation: bool = True):
    # ベース環境
    base = BaseEnv(
        gui=False,
        num_agents=2,
        max_steps=800,
        return_views=True,
        view_width=256,
        view_height=256,
        action_repeat=1,
        debug_info=True,
        use_cinematic_model=True,
        output_parent_view=True
    )
    
    # 幼児シグナルを検知するラッパー
    env = InfantSignalDetector(
        base,
        active_modalities=[ "gaze", "voice"]
    )
    env = ParentTools(env)
    
    # プロトコルマネージャーラッパー
    env = ProtocolManager(env, demo_mode=True)

    infant_movie, parent_movie = [], []
    
    obs, info = env.reset()
    
    # 今回のデモターゲット（例：赤いボックスを対象）
    # ※ 環境が複数のオブジェクトを持っていると仮定し適当に選択
    target_obj_idx = 0
    target_obj_id = env.active_objects[target_obj_idx]
        
    print(f"Target Object ID: {target_obj_id}")

    max_frames = 800
    t = 0
    episode_t = 0
    
    # 親のアクションはProtocolManagerがinfoを通じて渡してくるため、それを環境に入れる
    parent_action = np.zeros(env.unwrapped.action_num, dtype=np.float32)

    while t < max_frames:
        # デモにおける幼児のスクリプト挙動
        infant_action = np.zeros(env.unwrapped.action_num, dtype=np.float32)
        
        curr_proto = env.current_protocol.name if hasattr(env, "current_protocol") and env.current_protocol else "None"
        state = env.current_protocol.state if hasattr(env, "current_protocol") and hasattr(env.current_protocol, "state") else "None"

        if curr_proto == "LabelingProtocol" and state == "completed":
            env.gaze_agent(env.infant, env.parent, yaw_link="torso", action=infant_action)
        elif target_obj_id is not None:
            # 常にターゲットの方向を見続ける
            env.gaze_object(env.infant, target_obj_id, yaw_link="torso", action=infant_action)
        
        # 現在の音声の状態を取得
        curr_v1 = env.unwrapped.infant_voice[env.unwrapped.voice_mapping["v_1"]]
        curr_v2 = env.unwrapped.infant_voice[env.unwrapped.voice_mapping["v_2"]]
        curr_volume = env.unwrapped.infant_voice[env.unwrapped.voice_mapping["volume"]]

        # 親が "checking" （幼児を見ながら発話）状態になったと仮定し、
        # 親の発話後（episode_t > 300頃）に幼児が間違った発話「あか (a, k, a)」をするとする。
        # 実際には適当な母音(e, i)等に設定
        if 350 < episode_t < 450:
            target_v1 = -1.0 if (episode_t - 350) % 60 < 30 else -1.0  # e.g., e -> i
            target_v2 = 0.0 if (episode_t - 350) % 60 < 30 else -1.0
            target_vol = 0.8
            
            infant_action[env.unwrapped.action_map["v_1"]] = np.clip((target_v1 - curr_v1) / 0.5, -1.0, 1.0)
            infant_action[env.unwrapped.action_map["v_2"]] = np.clip((target_v2 - curr_v2) / 0.5, -1.0, 1.0)
            infant_action[env.unwrapped.action_map["volume"]] = np.clip((target_vol - curr_volume) / 0.5, -1.0, 1.0)
            print(env.phoneme_history)
            print(env.word_history)
        elif episode_t > 450 and episode_t < 650:
            # 正解 "りんご" (i, o) を発話
            target_v1 = -1.0 if (episode_t - 450) % 60 < 30 else 1.0
            target_v2 = -1.0 if (episode_t - 450) % 60 < 30 else 0.0
            target_vol = 0.8
            
            infant_action[env.unwrapped.action_map["v_1"]] = np.clip((target_v1 - curr_v1) / 0.5, -1.0, 1.0)
            infant_action[env.unwrapped.action_map["v_2"]] = np.clip((target_v2 - curr_v2) / 0.5, -1.0, 1.0)
            infant_action[env.unwrapped.action_map["volume"]] = np.clip((target_vol - curr_volume) / 0.5, -1.0, 1.0)
            print(env.phoneme_history)
            print(env.word_history)
            
        else:
            # 発話しない時はボリュームを下げ、口を閉じる
            infant_action[env.unwrapped.action_map["v_1"]] = np.clip((0.0 - curr_v1) / 0.5, -1.0, 1.0)
            infant_action[env.unwrapped.action_map["v_2"]] = np.clip((0.0 - curr_v2) / 0.5, -1.0, 1.0)
            infant_action[env.unwrapped.action_map["volume"]] = np.clip((-1.0 - curr_volume) / 0.5, -1.0, 1.0)

        
        # Action is a tuple of (infant_action, parent_action)
        action = (infant_action, parent_action)
        obs, reward, done, trunc, info = env.step(action)
        
        # update parent_action from info calculated by manager
        if "parent_action" in info:
            parent_action = np.asarray(info["parent_action"], dtype=np.float32)

        # 映像保存
        if "views" in info:
            infant_annot = np.array(info["views"].get("infant_rgb", np.zeros((64,64,3))))
            parent_annot = np.array(info["views"].get("parent_rgb", np.zeros((64,64,3))))
            infant_movie.append(infant_annot)
            parent_movie.append(parent_annot)

        # ステータス表示
        curr_proto = env.current_protocol.name if hasattr(env, "current_protocol") and env.current_protocol else "None"
        state = env.current_protocol.state if hasattr(env, "current_protocol") and hasattr(env.current_protocol, "state") else "None"
        
        if t % 50 == 0:
            print(f"t={t:04d} Protocol={curr_proto} State={state}")
            
        # #終了条件
        # if curr_proto == "LabelingProtocol" and state == "completed":
        #     done = True

        if done or trunc:
            print("Episode Done. Resetting environment...")
            obs, info = env.reset()
            if len(env.active_objects) > target_obj_idx:
                target_obj_id = env.active_objects[target_obj_idx]
            else:
                target_obj_id = None
            parent_action = np.zeros(env.unwrapped.action_num, dtype=np.float32)
            episode_t = 0
            t += 1
            continue
            
        t += 1
        episode_t += 1

    env.close()

    # 動画の保存
    save_dir = f"{ABS_PATH}/results/recast_demo"
    os.makedirs(save_dir, exist_ok=True)
    images_to_video(infant_movie, f"{save_dir}/infant_view.mp4", fps=24, scale=4)
    images_to_video(parent_movie, f"{save_dir}/parent_view.mp4", fps=24, scale=4)
    print(f"Videos saved to {save_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-annot", action="store_true")
    args = parser.parse_args()
    main(enable_annotation=not args.no_annot)
