#親の行動決定までで使用する細かい関数をまとめたラッパー
from typing import Optional, Tuple, Dict, Any, List
import math
import warnings
import numpy as np
import gymnasium as gym
import pybullet as p  # 環境内部で使用しているため依存

from cueworld.infant_signal import GAZE_RAY_ORIGIN_OFFSET


def show_distance_from_eyes(target_object: Any = None) -> float:
    """掲げ点を相手の目からどれだけ離すか。

    視線レイは目の GAZE_RAY_ORIGIN_OFFSET 前方から出る。半幅を足すと
    手前の面がその起点の外に出る。もう半幅は、手が目標より顔側へ寄っても
    面が起点の内側に入らないように残す分。
    共同注意の見せる合図と、渡しの掲げ（showing）が同じ式を使う。
    """
    half = 0.06
    extents = getattr(target_object, "bbox_half_extents", None)
    if extents:
        half = float(max(extents))
    return float(GAZE_RAY_ORIGIN_OFFSET) + 2.0 * half


class ParentTools(gym.Wrapper):
    # 近接で大きく下を向くと、目が tilt 軸より約 25cm 前にあるため
    # 1ステップの pitch が視線誤差の符号を反転させて頭が振動する。
    # 目標付近だけゲインを落とし、1°未満は動かさない。
    _GAZE_PITCH_DAMP_RAD = 0.35  # ~20deg
    _GAZE_PITCH_NEAR_GAIN = 0.3
    _GAZE_DEADZONE_RAD = math.radians(1.5)

    def __init__(self, env, *args, **kwargs):
        super().__init__(env)
        self.visibility_cache = {}
        self.ik_count = {}
        self.ik = {}
        self.ik_target = {}
        self.shake_t = 0
        self.arm_raise_flag = False
        self.shake_flag = False
        self.hand_over_phase = False  # ハンドオーバーのフェーズ管理用フラグ
        self.first_hand_status = {} #エージェントが掴みを始めたときの指の状態を保存する辞書
        # place_object_at: IK 頭打ち時の手放し判定用
        self._place_xy_best = {}
        self._place_xy_stagnant = {}
        self._place_target = {}
        self._place_releasing = {}

    def clear_cache(self):
        """IKのターゲットや視線回避の目標地点などのキャッシュをクリアする"""
        self.ik_count = {}
        self.ik = {}
        self.ik_target = {}
        self.visibility_cache = {}
        self._place_xy_best = {}
        self._place_xy_stagnant = {}
        self._place_target = {}
        self._place_releasing = {}

    def reset(self, **kwargs):
        """Resets the environment and returns the initial observation."""
        self.shake_t = 0
        self.clear_cache()

        self.arm_raise_flag = False
        self.shake_flag = False
        self.hand_over_phase = False
        self.first_hand_status = {}
        
        return self.env.reset(**kwargs)  
    
    def step(self, action):
        return super().step(action)

    
    #-----------------------親の行動を作るメソッド--------------------
    @staticmethod
    def _wrap_pi(delta: float) -> float:
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta <= -math.pi:
            delta += 2.0 * math.pi
        return float(delta)

    def _head_pan_yaw_error(self, agent: Dict[str, Any], target) -> float:
        """head_pan の残り角。

        目の前方のワールド XY は大きく下を向くと潰れて atan2 が不安定になる。
        pan 軸は胴体鉛直なので、胴体前方と「胴体→対象」の XY 角から目標 pan を求め、
        現在の head_pan との差を返す。
        """
        torso_pos, torso_fwd = self.env._agent_torso_pose(agent)
        v = np.array(target, dtype=np.float32) - np.array(torso_pos, dtype=np.float32)
        desired = self.rotation_to_target(
            np.array([torso_fwd[0], torso_fwd[1]], dtype=np.float32),
            np.array([v[0], v[1]], dtype=np.float32),
        )
        current = float(self.env.get_joint_angle(agent, "head_pan_joint"))
        return self._wrap_pi(desired - current)

    def _gaze_pitch_error(self, agent: Dict[str, Any], target) -> float:
        """目の前方と「目→対象」の仰角差。水平成分が極小なら 0。"""
        eye, fwd = self.env._agent_eye_pose(agent)
        v = np.array(target, dtype=np.float32) - np.array(eye, dtype=np.float32)
        fwd_xy = float(np.sqrt(fwd[0] ** 2 + fwd[1] ** 2))
        v_xy = float(np.sqrt(v[0] ** 2 + v[1] ** 2))
        if fwd_xy < 1e-4 or v_xy < 1e-4:
            return 0.0
        return float(
            self.rotation_to_target(
                np.array([fwd[2], fwd_xy], dtype=np.float32),
                np.array([v[2], v_xy], dtype=np.float32),
            )
        )

    def _stabilize_gaze_deltas(self, yaw: float, pitch: float) -> Tuple[float, float]:
        """近接の下向きで pitch が符号反転しないよう、目標付近のゲインとデッドゾーンを掛ける。"""
        yaw = float(yaw)
        pitch = float(pitch)
        if abs(pitch) < self._GAZE_PITCH_DAMP_RAD:
            pitch *= self._GAZE_PITCH_NEAR_GAIN
        if abs(yaw) < self._GAZE_DEADZONE_RAD:
            yaw = 0.0
        if abs(pitch) < self._GAZE_DEADZONE_RAD:
            pitch = 0.0
        return yaw, pitch

    def gaze_agent(self, agent_from: Dict[str, Any], agent_to: Dict[str, Any], yaw_link, action=None):
        """相手エージェントに視線を合わせるアクションを返す。"""
        am = self.env.action_map
        if action is None:
            action = {}
        target = p.getLinkState(agent_to["body"], agent_to["eyes_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4]
        if yaw_link == "torso":
            action[am["head_pan_joint"]] = -self.env.get_joint_angle(agent_from, "head_pan_joint")
            torso_pos, fwd = self.env._agent_torso_pose(agent_from)  # 胴体向き更新
            v = np.array(target) - np.array(torso_pos)
            yaw = self.rotation_to_target(np.array([fwd[0], fwd[1]], dtype=np.float32), np.array([v[0], v[1]], dtype=np.float32)) 
            action[am["torso_joint"]] = yaw
        elif yaw_link == "head":
            # 前のバージョン
            # centor_pos = p.getLinkState(agent_from["body"], agent_from["torso_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4]
            # v = np.array(target) - np.array(centor_pos)
            # _, fwd = self.env._agent_eye_pose(agent_from)  # 目の向き更新
            # yaw = self.rotation_to_target(np.array([fwd[0], fwd[1]], dtype=np.float32), np.array([v[0], v[1]], dtype=np.float32))
            # action[am["head_pan_joint"]] = yaw
            # 現在のバージョン
            yaw = self._head_pan_yaw_error(agent_from, target)
        
        
        # pitch成分
        # 前のバージョン
        # eye, fwd = self.env._agent_eye_pose(agent_from)
        # v = np.array(target) - np.array(eye)
        # fwd_xy = np.sqrt(fwd[0]**2 + fwd[1]**2)
        # v_xy = np.sqrt(v[0]**2 + v[1]**2)
        # pitch = self.rotation_to_target(np.array([fwd[2] ,fwd_xy], dtype=np.float32), np.array([v[2], v_xy], dtype=np.float32))
        # 現在のバージョン
        pitch = self._gaze_pitch_error(agent_from, target)
        yaw, pitch = self._stabilize_gaze_deltas(yaw, pitch)
        if yaw_link == "head":
            action[am["head_pan_joint"]] = yaw
        else:
            action[am["torso_joint"]] = yaw
        action[am["head_tilt_joint"]] = pitch
        return yaw, pitch
    
    def gaze_object(self, agent: Dict[str, Any], oid: int, yaw_link, action=None, option=None):
        """
        物体に視線を合わせるアクションを返す。
        頭を対象に向けるときはyawもpitchも動かす場合振動する可能性があるので適時optionを使用
        """
        am = self.env.action_map
        if action is None:
            action = {}
        target = self.env._obj_pos(oid)
        if yaw_link == "torso":
            action[am["head_pan_joint"]] = -self.env.get_joint_angle(agent, "head_pan_joint")
            torso_pos, fwd = self.env._agent_torso_pose(agent)  # 胴体向き更新
            v = np.array(target) - np.array(torso_pos)
            yaw = self.rotation_to_target(np.array([fwd[0], fwd[1]], dtype=np.float32), np.array([v[0], v[1]], dtype=np.float32))
        elif yaw_link == "head":
            # 前のバージョン
            # centor_pos = p.getLinkState(agent["body"], agent["torso_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4]
            # v = np.array(target) - np.array(centor_pos)
            # _, fwd = self.env._agent_eye_pose(agent)  # 目の向き更新
            # yaw = self.rotation_to_target(np.array([fwd[0], fwd[1]], dtype=np.float32), np.array([v[0], v[1]], dtype=np.float32))
            # 現在のバージョン
            yaw = self._head_pan_yaw_error(agent, target)
              
        # pitch のみ成分で角度計算
        if option == "yaw_only":
            pitch = 0.0
        else:
            # pitch 成分
            # 前のバージョン
            # eye, fwd = self.env._agent_eye_pose(agent)
            # v = np.array(target) - np.array(eye)
            # fwd_xy = np.sqrt(fwd[0]**2 + fwd[1]**2)
            # v_xy = np.sqrt(v[0]**2 + v[1]**2)
            # pitch = self.rotation_to_target(np.array([fwd[2], fwd_xy], dtype=np.float32), np.array([v[2], v_xy], dtype=np.float32))
            # # if pitch < 0.05:
            # #     pitch = pitch * 0.5  # 微小な上下動を抑制
            # 現在のバージョン
            pitch = self._gaze_pitch_error(agent, target)
            
        if option == "pitch_only":
            yaw = 0.0
        # 現在のバージョン
        yaw, pitch = self._stabilize_gaze_deltas(yaw, pitch)

        if yaw_link == "head":
            action[am["head_pan_joint"]] = yaw
        else:
            action[am["torso_joint"]] = yaw
        action[am["head_tilt_joint"]] = pitch
        return yaw, pitch

    def point_object(self, agent, oid: int, action=None, ik_every: int = 0):
        if action is None:
            action = {}
        agent_id = agent["body"]
        target_pos = self.env._obj_pos(oid)
        #ikのターゲットが変わった場合はリセット
        current_target = self.ik_target.get(agent_id)
        if current_target is None or (current_target[0] != (oid) or np.linalg.norm(np.array(current_target[1]) - np.array(target_pos)) > 0.01):
            self.ik_target[agent_id] = (oid, target_pos)
            self.ik_count[agent_id] = 0
            self.ik[agent_id] = None
        #特定のタイミングでIKを計算
        if self.ik.get(agent_id) is None or (ik_every != 0 and self.ik_count.get(agent_id, 0) % ik_every == 0):
            agent_body = agent["body"]
            ee_link_index = agent["point_link"]  # 指差し用のリンク
            torso_index = agent["joints_for_point"]["torso_joint"]
            # torsoの可動域を現在のtorsoの状態に固定する．
            current_torso_angle = self.env.get_joint_angle(agent, "torso_joint")
            agent["lower_limits"][torso_index] = current_torso_angle
            agent["upper_limits"][torso_index] = current_torso_angle
            self.ik[agent_id] = p.calculateInverseKinematics(
                agent_body,
                ee_link_index,
                target_pos,
                lowerLimits=agent["lower_limits"],
                upperLimits=agent["upper_limits"],
                jointRanges=agent["joint_ranges"],
                restPoses=agent["restposes"],
                maxNumIterations=50,
                residualThreshold=1e-4,
                physicsClientId=self.env.client_id
            )
        #指差しに関する関節角度をアクションにセット
        ik_sol = self.ik[agent_id]
        for joint_name, joint_index in agent["joints_for_point"].items():
            current_angle = p.getJointState(agent["body"], agent[joint_name], physicsClientId=self.env.client_id)[0]
            target_angle = ik_sol[joint_index]
            angle_diff = target_angle - current_angle
            action[self.env.action_map[joint_name]] = angle_diff
        #IKカウンタ更新
        if ik_every != 0:
            self.ik_count[agent_id] = self.ik_count.get(agent_id, 0) + 1
        action[self.env.action_map["roll_wrist_joint"]] = -self.env.get_joint_angle(agent, "roll_wrist_joint")
        action[self.env.action_map["pitch_wrist_joint"]] = -self.env.get_joint_angle(agent, "pitch_wrist_joint")
        action[self.env.action_map["other_fingers"]] = 3.14 #閉じる
        action[self.env.action_map["index_joint"]] = -3.14 #伸ばす
        return action
    
    def approach_infant(self, agent, infant, action=None, thresh=None, approach_dist: float = 0):
        """幼児に近づく行動"""
        distance = None
        if action is None:
            action = {}
        if thresh is None:
            thresh = float(math.radians(20.0))
        yaw, _ = self.gaze_agent(agent, infant, yaw_link="torso", action=action)
        if abs(yaw) <= thresh:
            # 十分向いている → 前進
            infant_pos, _ = p.getBasePositionAndOrientation(infant["body"], physicsClientId=self.env.client_id)
            distance = self._dist_body_to(agent, infant_pos)
            forward = distance - approach_dist
            forward = max(0.0, float(forward))  # クリップ
            action[self.env.action_map["forward"]] = forward
        return yaw, distance
    
    def approach_object(self, agent, oid: int, action=None, thresh=None, approach_dist: float = 0):
        """
        オブジェクトに近づく行動
        approach_dist: 近づきたい距離 これがあまりにも小さいと有効数字の問題で不可能な場合がある
        """
        distance = None
        if action is None:
            action = {}
        if thresh is None:
            thresh = float(math.radians(15.0))
        yaw, _ = self.gaze_object(agent, oid, yaw_link="torso", action=action)
        if abs(yaw) <= thresh:
            # 十分向いている → 前進
            distance = self._dist_body_to(agent, self.env._obj_pos(oid))
            forward = distance - approach_dist
            forward = max(0.0, float(forward))  # クリップ
            action[self.env.action_map["forward"]] = forward
        return yaw, distance
    
    def approach_location(self, agent, target_pos: np.ndarray, action=None, thresh=None, approach_dist: float = 0.05):
        """指定位置に近づくための前進量を計算。"""
        distance = None
        if action is None:
            action = {}
        if thresh is None:
            thresh = float(getattr(self.env, "gaze_obj_thresh", math.radians(20.0)))
        torso, fwd = self.env._agent_torso_pose(agent)
        # yaw の成分で角度計算(ある場所に向かうときはpitchは考慮しない)
        fwd_yaw = np.array([fwd[0], fwd[1]], dtype=np.float32)
        to_target_yaw = np.array([target_pos[0] - torso[0], target_pos[1] - torso[1]], dtype=np.float32)
        yaw = self.rotation_to_target(fwd_yaw, to_target_yaw)
        action[self.env.action_map["torso_joint"]] = yaw
        
        if abs(yaw) <= thresh:
            # 十分向いている → 前進
            distance = self._dist_body_to(agent, target_pos)
            forward = distance - approach_dist
            forward = max(0.00, float(forward))  # クリップ
            action[self.env.action_map["forward"]] = forward
                
        self.yaw = yaw
        self.distance = distance
        return yaw, distance
    
    def hand_reach_location(self, agent: Dict[str, Any], target_pos, action=None, ik_every: int = 0, oid: int = -1):
        """指定した座標に手を伸ばす"""
        if action is None:
            action = {}
        agent_id = agent["body"]
        #ikのターゲットが変わった場合はリセット
        current_target = self.ik_target.get(agent_id)
        if current_target is None or (current_target[0] != (oid) or np.linalg.norm(np.array(current_target[1]) - np.array(target_pos)) > 0.01):
            self.ik_target[agent_id] = (oid, target_pos)
            self.ik_count[agent_id] = 0
            self.ik[agent_id] = None
        #特定のタイミングでIKを計算
        if self.ik.get(agent_id) is None or (ik_every != 0 and self.ik_count.get(agent_id, 0) % ik_every == 0):
            agent_body = agent["body"]
            ee_link_index = agent["grasp_link"]  # 手先のリンク
            torso_index = agent["joints_for_reach"]["torso_joint"]
            # torsoの可動域を現在のtorsoの状態に固定する．
            current_torso_angle = self.env.get_joint_angle(agent, "torso_joint")
            agent["lower_limits"][torso_index] = current_torso_angle
            agent["upper_limits"][torso_index] = current_torso_angle
            self.ik[agent_id] = p.calculateInverseKinematics(
                agent_body,
                ee_link_index,
                target_pos,
                lowerLimits=agent["lower_limits"],
                upperLimits=agent["upper_limits"],
                jointRanges=agent["joint_ranges"],
                restPoses=agent["restposes"],
                maxNumIterations=50,
                residualThreshold=1e-4,
                physicsClientId=self.env.client_id
            )
        #手に関する関節角度をアクションにセット
        ik_sol = self.ik[agent_id]
        for joint_name, joint_index in agent["joints_for_reach"].items(): 
            current_angle = p.getJointState(agent["body"], agent[joint_name], physicsClientId=self.env.client_id)[0]
            target_angle = ik_sol[joint_index]
            angle_diff = target_angle - current_angle
            action[self.env.action_map[joint_name]] = angle_diff
        #IKカウンタ更新
        if ik_every != 0:
            self.ik_count[agent_id] = self.ik_count.get(agent_id, 0) + 1
        action[self.env.action_map["index_joint"]] = -3.14 
        action[self.env.action_map["other_fingers"]] = -3.14 
        if oid != -1:
            action[self.env.action_map["roll_wrist_joint"]] = self.get_palm_roll_angle_to_object(agent, oid)
        return action

    def hand_reach_object(self, agent: Dict[str, Any], oid: int, action=None, ik_every: int = 0):
        """
        オブジェクトへ手を伸ばす
        手を広げてオブジェクトを掴もうとするところも含む
        """
        if action is None:
            action = {}
        target_pos = self.env._obj_pos(oid)
        return self.hand_reach_location(agent, target_pos, action=action, ik_every=ik_every, oid=oid)
    
    def get_palm_roll_angle_to_object(self, agent: Dict[str, Any], oid: int) -> float:
        """手のひらロール方向(+X)と物体方向の角度。"""
        link_state = p.getLinkState(agent["body"], agent["hand_link"], physicsClientId=self.env.client_id)
        hand_pos = link_state[4]
        obj_pos = self.env._obj_pos(oid)
        if np.linalg.norm(np.array(obj_pos) - np.array(hand_pos)) < 1e-4:
            return -self.env.get_joint_angle(agent, "roll_wrist_joint")
        else:
            hand_orn = link_state[5]
            #リンクの世界座標を逆変換
            inv_hand_pos, inv_hand_orn = p.invertTransform(hand_pos, hand_orn)
            _, fwd = self.env._get_palm_pose(agent)  # 手の姿勢更新
            #手のリンクの局所座標系でオブジェクト位置を取得
            obj_local_pos, _ = p.multiplyTransforms(
                inv_hand_pos,
                inv_hand_orn,
                obj_pos,
                [0, 0, 0, 1]
            )
            angle = self.rotation_to_target(np.array([-1.0, 0.0], dtype=np.float32), np.array([obj_local_pos[0], obj_local_pos[1]], dtype=np.float32))
            return angle
            
    def grasp_object(self, agent, oid: int, action=None, approach_dist: float = 0):
        """
        オブジェクトを掴むアクションを返す。
        これをプロトコルで呼び出してものを掴んだらself.env.first_hand_status = {}を呼び出して初期化することを忘れずに行う．
        """
        if action is None:
            action = {}
        am = self.env.action_map
        main_link = "eye" #その過程で使う主なリンク
        yaw, distance = self.approach_object(agent, oid, action, thresh=None, approach_dist=approach_dist) #approach_distはオブジェクトとエージェントの大きさの合計分
        #print(f"approach_object: yaw={math.degrees(yaw):.2f} deg, distance={distance} m") #デバッグ用
        if distance is not None and abs(yaw) <= math.radians(10.0) and distance <= approach_dist+0.01:
            action = self.hand_reach_object(agent, oid, action=action)
            graspable_obj, _ = self.env.detect_grasp_object(agent)
            print(f"graspable_obj: {graspable_obj}, target_oid: {oid}") #デバッグ用
            if oid == graspable_obj:
                if agent["body"] not in self.first_hand_status:
                    self.first_hand_status[agent["body"]] = agent["previous_hand_state"]
                elif self.first_hand_status[agent["body"]] == "closed" and agent["previous_hand_state"] == "open":
                    self.first_hand_status[agent["body"]] = "open"
                    
                if self.first_hand_status[agent["body"]] == "open":
                    action[am["index_joint"]] = 3.14
                    action[am["other_fingers"]] = 3.14
                else:
                    action[am["index_joint"]] = -3.14
                    action[am["other_fingers"]] = -3.14
            else:
                action[am["index_joint"]] = -3.14
                action[am["other_fingers"]] = -3.14
        return action
    
    def grip_object(self, agent, action=None):
        """すでに手に持っているオブジェクトをしっかり握るアクションを返す。"""
        if action is None:
            action = {}
        am = self.env.action_map
        action[am["index_joint"]] = 3.14
        action[am["other_fingers"]] = 3.14
        return action

    def open_hand(self, agent, action=None):
        """今の手の位置で指を開く。開始時の手放し（バッチ F）。place の解放と同じ値。"""
        if action is None:
            action = {}
        am = self.env.action_map
        action[am["index_joint"]] = -3.14
        action[am["other_fingers"]] = -3.14
        return action
    
    def open_close_object(self, agent, oid: int, action=None, approach_dist: float = 0.5):
        """オブジェクトを開閉するアクション"""
        if action is None:
            action = {}
        am = self.env.action_map
        yaw, distance = self.approach_object(agent, oid, action, thresh=None, approach_dist=approach_dist)
        if distance is not None and abs(yaw) <= math.radians(10.0) and distance <= approach_dist:
            # 蓋の座標を取得
            obj = self.env.all_objects.get(oid)
            target_pos = self.env._obj_pos(oid)
            if obj and hasattr(obj, "lid_link_id") and obj.lid_link_id is not None:
                lid_state = p.getLinkState(oid, obj.lid_link_id, computeForwardKinematics=True, physicsClientId=self.env.client_id)
                target_pos = lid_state[4]
            
            self.hand_reach_location(agent, target_pos, action=action, oid=oid)
            
            graspable_obj, _ = self.env.detect_grasp_object(agent)
            if oid == graspable_obj:
                action[am["index_joint"]] = 3.14
                action[am["other_fingers"]] = 3.14
        return action

    def put_in_object(self, agent, container_id: int, action=None, approach_dist: float = 0.5):
        """持っているオブジェクトをコンテナに入れるアクション"""
        if action is None:
            action = {}
        am = self.env.action_map
        yaw, distance = self.approach_object(agent, container_id, action, thresh=None, approach_dist=approach_dist)
        if distance is not None and abs(yaw) <= math.radians(10.0) and distance <= approach_dist:
            # 開いているコンテナの少し上に手を伸ばす
            container_pos = self.env._obj_pos(container_id)
            target_pos = (container_pos[0], container_pos[1], container_pos[2] + 0.15)
            self.hand_reach_location(agent, target_pos, action=action)
            # 手を開いて離す
            action[am["index_joint"]] = -3.14
            action[am["other_fingers"]] = -3.14
        return action

    def hand_over(self,agent,target_agent,action=None, mouth_feeding=False, target_distance=None):
        """
        エージェントが物を手渡すアクションを返す。
        mouth_feeding: Trueの場合、口元に運ぶ
        §3 以降 object_giving からは使わない（到達圏への place に移行）。
        """
        #TODO: まだ仮実装．実際に使うときに調整が必要
        if action is None:
            action = {}
        am = self.env.action_map
        #幼児に近づく
        if not self.hand_over_phase:
            if target_distance is None:
                target_distance = 0.8 if not mouth_feeding else 0.6
            yaw, distance = self.approach_infant(agent, target_agent, action=action, thresh=None, approach_dist=target_distance)
            if distance is not None and abs(yaw) <= math.radians(5.0) and distance <= target_distance:
                self.hand_over_phase = True
        #幼児に食べさせる/わたす
        if self.hand_over_phase:
            # 非 *_2 URDF には jaw_link がないため、口元運びは手元へフォールバック
            use_mouth = mouth_feeding and target_agent.get("jaw_link") is not None
            if mouth_feeding and not use_mouth:
                warnings.warn(
                    "mouth_feeding requested but target_agent has no jaw_link; falling back to hand",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if use_mouth:
                target_pos = p.getLinkState(target_agent["body"], target_agent["jaw_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4]
                self.hand_reach_location(agent, target_pos, action=action)
            else:
                target_pos = p.getLinkState(target_agent["body"], target_agent["grasp_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4]
                self.hand_reach_location(agent, target_pos, action=action)
                little_finger_angle = self.env.get_joint_angle(agent, "little_finger_joint")
                index_angle = self.env.get_joint_angle(agent, "index_joint")
                action[am["index_joint"]] = 1.6 - index_angle
                action[am["other_fingers"]] = 1.6 - little_finger_angle
        return action
    
    def _dist_body_to(self, agent: Dict[str, Any], target_pos: np.ndarray) -> float:
        """エージェント胴体(base)位置から target_pos までの水平距離[XY]を返す。"""
        pos, _ = p.getBasePositionAndOrientation(agent["body"], physicsClientId=self.env.client_id)
        ppos = np.asarray(pos, dtype=np.float32)[:2]
        tgt = np.asarray(target_pos, dtype=np.float32)[:2]
        diff = tgt - ppos
        return float(np.linalg.norm(diff))

    def reach_place_position(
        self,
        target_agent: Dict[str, Any],
        from_agent: Dict[str, Any],
        place_dist: float = 0.35,
        hold_height: float = 0.15,
    ) -> np.ndarray:
        """相手の到達圏内（親子を結ぶ線上・相手側）に置く狙い点を返す。

        XY は target_agent base から from_agent 方向へ place_dist 進んだ点。
        z は床 + hold_height（簡易重力で落とす前提なので厳密に合わせない）。
        """
        target_pos, _ = p.getBasePositionAndOrientation(
            target_agent["body"], physicsClientId=self.env.client_id
        )
        from_pos, _ = p.getBasePositionAndOrientation(
            from_agent["body"], physicsClientId=self.env.client_id
        )
        txy = np.asarray(target_pos, dtype=np.float32)[:2]
        fxy = np.asarray(from_pos, dtype=np.float32)[:2]
        diff = fxy - txy
        norm = float(np.linalg.norm(diff))
        if norm < 1e-4:
            # 縮退時は from_agent の torso 前方へフォールバック
            _, fwd = self.env._agent_torso_pose(from_agent)
            direction = np.asarray([fwd[0], fwd[1]], dtype=np.float32)
            dnorm = float(np.linalg.norm(direction))
            if dnorm < 1e-4:
                direction = np.array([1.0, 0.0], dtype=np.float32)
            else:
                direction = direction / dnorm
            place_xy = fxy + direction * float(place_dist)
        else:
            direction = diff / norm
            place_xy = txy + direction * float(place_dist)

        # アリーナ内にクランプ
        xh, yh = self.env.arena_half_extents
        margin = float(getattr(self.env, "margin", 0.1))
        place_xy[0] = float(np.clip(place_xy[0], -xh + margin, xh - margin))
        place_xy[1] = float(np.clip(place_xy[1], -yh + margin, yh - margin))
        return np.array([place_xy[0], place_xy[1], float(hold_height)], dtype=np.float32)

    def compute_show_position(
        self,
        agent: Dict[str, Any],
        target_agent: Dict[str, Any],
        show_dist: Optional[float] = None,
        from_eyes: bool = True,
        target_object: Any = None,
    ) -> np.ndarray:
        """掲げ目標点を計算する（プロトコル進入時に1回呼ぶ想定）。

        既定（from_eyes=True）は、相手の目の高さで、目から agent 方向へ水平に進んだ点。
        show_dist が None なら show_distance_from_eyes(target_object)。
        渡した値は目からの水平距離。from_eyes=False のときは相手の base からの距離で、
        show_dist が None なら env.place_dist。いずれもライブ追従はしない。
        """
        if from_eyes:
            distance = (
                show_distance_from_eyes(target_object)
                if show_dist is None
                else float(show_dist)
            )
            return self._show_position_from_eyes(agent, target_agent, distance)
        if show_dist is None:
            show_dist = float(getattr(self.env, "place_dist", 0.3))
        show_pos = self.reach_place_position(
            target_agent, agent, place_dist=float(show_dist), hold_height=0.0
        )
        eyes_state = p.getLinkState(
            target_agent["body"],
            target_agent["eyes_link"],
            computeForwardKinematics=True,
            physicsClientId=self.env.client_id,
        )
        show_pos = np.asarray(show_pos, dtype=np.float32).copy()
        show_pos[2] = float(eyes_state[4][2])
        return show_pos

    def _show_position_from_eyes(
        self,
        agent: Dict[str, Any],
        target_agent: Dict[str, Any],
        distance: float,
    ) -> np.ndarray:
        """相手の目の高さで、目から agent 方向へ distance だけ離した点。"""
        eyes_state = p.getLinkState(
            target_agent["body"],
            target_agent["eyes_link"],
            computeForwardKinematics=True,
            physicsClientId=self.env.client_id,
        )
        eye = np.asarray(eyes_state[4], dtype=np.float32)
        target_pos, _ = p.getBasePositionAndOrientation(
            target_agent["body"], physicsClientId=self.env.client_id
        )
        from_pos, _ = p.getBasePositionAndOrientation(
            agent["body"], physicsClientId=self.env.client_id
        )
        direction = np.asarray(from_pos[:2], dtype=np.float32) - np.asarray(target_pos[:2], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-4:
            _, fwd = self.env._agent_torso_pose(agent)
            direction = np.asarray(fwd[:2], dtype=np.float32)
            norm = float(np.linalg.norm(direction))
        direction = direction / max(norm, 1e-8)
        show_pos = eye.copy()
        show_pos[0] = float(eye[0] + direction[0] * float(distance))
        show_pos[1] = float(eye[1] + direction[1] * float(distance))
        xh, yh = self.env.arena_half_extents
        margin = float(getattr(self.env, "margin", 0.1))
        show_pos[0] = float(np.clip(show_pos[0], -xh + margin, xh - margin))
        show_pos[1] = float(np.clip(show_pos[1], -yh + margin, yh - margin))
        return show_pos

    def palm_reach_error(self, agent: Dict[str, Any], target_pos) -> float:
        """palm 中心と target_pos の 3D 距離[m]。"""
        palm_pos, _ = self.env._get_palm_pose(agent)
        return float(
            np.linalg.norm(
                np.asarray(palm_pos, dtype=np.float32)
                - np.asarray(target_pos, dtype=np.float32)
            )
        )

    def place_object_at(
        self,
        agent: Dict[str, Any],
        target_pos,
        action=None,
        release_tol: float = 0.15,
        release_slack_tol: float = 0.20,
        stagnant_steps: int = 30,
    ):
        """指定座標へ手を伸ばし、XY が揃ったら手放す。

        hand_reach_location は常に指を開くため、XY が合うまでは grip で上書きする。
        高さは合わせない（簡易重力で落とす）。

        手放し条件:
        - palm XY 誤差 < release_tol（通常）
        - または、一度も release_tol 未満まで届かず、release_slack_tol 未満のまま
          stagnant_steps 連続（IK が ~0.17 で頭打ちするケース）。
          release_tol を 0.18 等に単純引き上げると happy path が早手放しで到達圏外になる。

        Returns:
            (action, released): released は今ステップで手放し指令を出したか
        """
        if action is None:
            action = {}
        am = self.env.action_map
        self.hand_reach_location(agent, target_pos, action=action)
        palm_pos, _ = self.env._get_palm_pose(agent)
        xy_err = float(
            np.linalg.norm(
                np.asarray(palm_pos, dtype=np.float32)[:2]
                - np.asarray(target_pos, dtype=np.float32)[:2]
            )
        )

        agent_id = agent["body"]
        # 一度手放し判定したら、held が消えるまで開き続ける
        # （指は step 制限でゆっくり開く。途中で grip に戻すと closed のままになる）
        if self._place_releasing.get(agent_id):
            if agent.get("held_obj") is None:
                self._place_releasing.pop(agent_id, None)
                self._place_xy_best.pop(agent_id, None)
                self._place_xy_stagnant.pop(agent_id, None)
                self._place_target.pop(agent_id, None)
                return action, True
            action[am["index_joint"]] = -3.14
            action[am["other_fingers"]] = -3.14
            return action, True

        tgt_xy = np.asarray(target_pos, dtype=np.float32)[:2]
        prev_tgt = self._place_target.get(agent_id)
        if prev_tgt is None or float(np.linalg.norm(prev_tgt - tgt_xy)) > 0.01:
            self._place_target[agent_id] = tgt_xy.copy()
            self._place_xy_best[agent_id] = xy_err
            self._place_xy_stagnant[agent_id] = 0
        else:
            best = self._place_xy_best.get(agent_id, xy_err)
            if xy_err < best:
                self._place_xy_best[agent_id] = xy_err
            if xy_err < float(release_slack_tol):
                self._place_xy_stagnant[agent_id] = self._place_xy_stagnant.get(agent_id, 0) + 1
            else:
                self._place_xy_stagnant[agent_id] = 0

        best = self._place_xy_best.get(agent_id, xy_err)
        can_release = xy_err < float(release_tol) or (
            best >= float(release_tol)
            and xy_err < float(release_slack_tol)
            and self._place_xy_stagnant.get(agent_id, 0) >= int(stagnant_steps)
        )
        if not can_release:
            self.grip_object(agent, action)  # XY がまだ合っていない → 握り続ける
            return action, False

        self._place_releasing[agent_id] = True
        self._place_xy_best.pop(agent_id, None)
        self._place_xy_stagnant.pop(agent_id, None)
        self._place_target.pop(agent_id, None)
        action[am["index_joint"]] = -3.14  # 手放す（base 側で解放処理 → 簡易重力で落下）
        action[am["other_fingers"]] = -3.14
        return action, True

    def show_object(
        self,
        agent: Dict[str, Any],
        target_pos,
        action=None,
        reach_tol: float = 0.12,
    ):
        """把持済みのオブジェクトを固定 target_pos へ掲げる。

        接近・視線・目標計算は含めない（呼び出し側プロトコルで行う）。
        渡された固定点へ hand_reach_location するだけ（毎ステップ自ら目を読まない）。
        hand_reach_location は常に指を開くため、grip で上書きして把持を維持する。

        Returns:
            (action, reached): reached は palm 中心と target_pos の 3D 距離が reach_tol 未満か
        """
        if action is None:
            action = {}
        self.hand_reach_location(agent, target_pos, action=action)
        self.grip_object(agent, action)  # 掲げる間は握り続ける
        err = self.palm_reach_error(agent, target_pos)
        return action, err < float(reach_tol)

    def is_within_reach(
        self,
        agent: Dict[str, Any],
        target_pos,
        reach_radius: float = None,
    ) -> bool:
        """エージェント base から target_pos までの XY 距離が到達圏内か。"""
        if reach_radius is None:
            reach_radius = float(getattr(self.env, "reach_radius", 0.45))
        return self._dist_body_to(agent, target_pos) <= float(reach_radius)
    
    def approach_infant_for_ja(self, agent, infant, oid: int, action=None, thresh=None, approach_dist: float = 1.0, min_angle_deg: float = 30.0):
        """共同注意用の位置調整付き接近。
        
        親がオブジェクトと幼児の間に入らないように、位置を調整しながら幼児に近づく。
        - 親が幼児よりオブジェクトに近い場合（間に入る配置）:
          幼児-オブジェクト直線から min_angle_deg 以上ずれた位置を目標に移動する。
        - そうでない場合:
          従来通り幼児に直接近づく。
        
        Args:
            agent: 親エージェント情報
            infant: 幼児エージェント情報
            oid: 対象オブジェクトのbody ID
            action: アクション辞書（省略時は新規作成）
            thresh: 前進開始のYaw閾値（rad）
            approach_dist: 目標距離
            min_angle_deg: 幼児-オブジェクト直線に対する最小角度（度）
        
        Returns:
            (yaw, distance): 方向調整量と現在の距離
        """
        distance = None
        if action is None:
            action = {}
        if thresh is None:
            thresh = float(math.radians(20.0))
        
        # 各位置を取得（2Dの水平座標で計算）
        parent_pos, _ = p.getBasePositionAndOrientation(agent["body"], physicsClientId=self.env.client_id)
        infant_pos, _ = p.getBasePositionAndOrientation(infant["body"], physicsClientId=self.env.client_id)
        obj_pos = self.env._obj_pos(oid)
        
        parent_xy = np.array(parent_pos[:2], dtype=np.float32)
        infant_xy = np.array(infant_pos[:2], dtype=np.float32)
        obj_xy = np.array(obj_pos[:2], dtype=np.float32)
        
        dist_parent_infant = float(np.linalg.norm(parent_xy - infant_xy))
        dist_obj_infant = float(np.linalg.norm(obj_xy - infant_xy))
        
        # 親がオブジェクトより幼児に近い場合（遮る可能性がある）
        if dist_parent_infant < dist_obj_infant:
            # 幼児→オブジェクトのベクトル
            io_vec = obj_xy - infant_xy
            io_len = float(np.linalg.norm(io_vec))
            if io_len < 1e-4:
                # オブジェクトと幼児が同位置の場合はフォールバック
                return self.approach_infant(agent, infant, action=action, thresh=thresh, approach_dist=approach_dist)
            
            # 親→幼児のベクトル
            pi_vec = infant_xy - parent_xy
            
            # 幼児-オブジェクト直線と親-幼児ベクトルの成す角を計算
            io_unit = io_vec / io_len
            pi_len = float(np.linalg.norm(pi_vec))
            if pi_len < 1e-4:
                return 0.0, 0.0  # すでに幼児の位置にいる
            pi_unit = pi_vec / pi_len
            
            cos_angle = float(np.clip(np.dot(io_unit, pi_unit), -1.0, 1.0))
            angle_rad = math.acos(cos_angle)
            min_angle_rad = math.radians(min_angle_deg)
            
            if angle_rad < min_angle_rad:
                # 直線に近い位置にいるので、ずれた位置を目標とする
                # 幼児-オブジェクト直線に対して垂直方向のベクトルを計算
                perp_vec = np.array([-io_unit[1], io_unit[0]], dtype=np.float32)
                
                # 親が直線のどちら側にいるかを判定し、その側にずらす
                side = float(np.dot(pi_vec, perp_vec))
                if side >= 0:
                    offset_dir = perp_vec
                else:
                    offset_dir = -perp_vec
                
                # 目標位置: 幼児から approach_dist の距離で、min_angle_deg ずれた位置
                target_angle = min_angle_rad
                # 幼児-オブジェクト方向を基準に回転した目標方向
                cos_t = math.cos(target_angle)
                sin_t = math.sin(target_angle)
                # io_unit方向からtarget_angle回転した方向に幼児からapproach_distの距離
                if side >= 0:
                    target_dir = np.array([
                        io_unit[0] * cos_t - io_unit[1] * sin_t,
                        io_unit[0] * sin_t + io_unit[1] * cos_t
                    ], dtype=np.float32)
                else:
                    target_dir = np.array([
                        io_unit[0] * cos_t + io_unit[1] * sin_t,
                        -io_unit[0] * sin_t + io_unit[1] * cos_t
                    ], dtype=np.float32)
                
                # 目標座標 = 幼児位置 - target_dir * approach_dist（幼児の手前に立つ）
                target_xy = infant_xy - target_dir * approach_dist
                target_pos_3d = (float(target_xy[0]), float(target_xy[1]), float(parent_pos[2]))
                
                # 目標座標へ向かう
                torso_pos, fwd = self.env._agent_torso_pose(agent)
                v = np.array(target_pos_3d[:2]) - np.array(torso_pos[:2])
                yaw = self.rotation_to_target(
                    np.array([fwd[0], fwd[1]], dtype=np.float32),
                    np.array([v[0], v[1]], dtype=np.float32)
                )
                am = self.env.action_map
                action[am["torso_joint"]] = yaw
                action[am["head_pan_joint"]] = -self.env.get_joint_angle(agent, "head_pan_joint")
                
                if abs(yaw) <= thresh:
                    distance = float(np.linalg.norm(np.array(target_pos_3d[:2]) - np.array(torso_pos[:2])))
                    forward = max(0.0, distance - 0.1)
                    action[am["forward"]] = forward
                
                # 返り値の distance は幼児までの実距離を返す
                distance = dist_parent_infant
                return yaw, distance
        
        # 遮る心配がない場合は通常の接近
        return self.approach_infant(agent, infant, action=action, thresh=thresh, approach_dist=approach_dist)

    def _apply_joint_target(
        self,
        agent: Dict[str, Any],
        action,
        joint_name: str,
        target_angle: float,
        additive: bool = True,
    ) -> None:
        """関節の目標角に向けた差分を action に書き込む（複数チャネル同時実行時は加算）。"""
        am = self.env.action_map
        if joint_name == "other_fingers":
            if "other_fingers" not in am:
                return
            current_angle = self.env.get_joint_angle(agent, "little_finger_joint")
            idx = am["other_fingers"]
        elif joint_name in am:
            current_angle = self.env.get_joint_angle(agent, joint_name)
            idx = am[joint_name]
        else:
            return

        delta = float(target_angle) - float(current_angle)
        if additive:
            action[idx] += delta
        else:
            action[idx] = delta

    def wave_arm(
        self,
        agent: Dict[str, Any],
        action=None,
        phase: float = 0.0,
        gesture: bool = False,
        amplitude_scale: float = 1.0,
    ):
        """手振り: 腕を上げた姿勢で肩を phase に応じて往復させる。

        gesture=False（既定）: 現行どおり肩・肘・手首・指の関節角ポーズ。
        gesture=True: shoulder_pitch に正弦波を加算（他は IK のまま。掲げオーバーレイ用）。
        amplitude_scale: 振れ幅の倍率（既定 1.0。親の本番キューは既定のまま）。
        """
        if action is None:
            action = {}
        phase = float(phase) % 1.0
        scale = float(amplitude_scale)

        if gesture:
            # 掲げ IK の上に肩 pitch を上下加算（幼児の wave と同軸で見やすくする）
            # hand_reach_location の後に呼ばれる前提: action に IK 差分が既にある
            am = self.env.action_map
            pitch_idx = am["shoulder_pitch_joint"]
            pitch_amplitude = 0.35 * scale
            action[pitch_idx] = float(action[pitch_idx]) + pitch_amplitude * math.sin(
                2.0 * math.pi * phase
            )
            return action

        base_pitch = -0.8
        amplitude = 0.35 * scale
        target_pitch = base_pitch + amplitude * math.sin(2.0 * math.pi * phase)

        self._apply_joint_target(agent, action, "shoulder_yaw_joint", 0.35)
        self._apply_joint_target(agent, action, "shoulder_pitch_joint", target_pitch)
        self._apply_joint_target(agent, action, "elbow_joint", 1.1)
        self._apply_joint_target(agent, action, "roll_wrist_joint", 0.0)
        self._apply_joint_target(agent, action, "pitch_wrist_joint", 0.0)
        self._apply_joint_target(agent, action, "index_joint", -0.3)
        self._apply_joint_target(agent, action, "other_fingers", -0.3)
        return action

    def nod_head(
        self,
        agent: Dict[str, Any],
        action=None,
        phase: float = 0.0,
        amplitude_scale: float = 1.0,
    ):
        """うなずき: 頭ピッチを phase に応じて往復させる（視線制御との加算を想定）。

        amplitude_scale: 振れ幅の倍率（既定 1.0。親の本番キューは既定のまま）。
        """
        if action is None:
            action = {}
        phase = float(phase) % 1.0
        amplitude = 0.25 * float(amplitude_scale)
        nod_offset = amplitude * math.sin(2.0 * math.pi * phase)
        current_tilt = self.env.get_joint_angle(agent, "head_tilt_joint")
        self._apply_joint_target(agent, action, "head_tilt_joint", current_tilt + nod_offset)
        return action

    def animate_hand_open_close(
        self,
        agent: Dict[str, Any],
        action=None,
        phase: float = 0.0,
        gesture: bool = False,
        amplitude_scale: float = 1.0,
    ):
        """手の開閉: phase に応じて開く相・閉じる相をループさせる。

        gesture=False（既定）: 現行どおり腕ポーズ＋指開閉。
        gesture=True: 指のみ開閉（他は IK のまま。掲げオーバーレイ用）。
        amplitude_scale: 開閉振幅の倍率（中点まわり。既定 1.0。親の本番キューは既定のまま）。
        """
        if action is None:
            action = {}
        phase = float(phase) % 1.0
        # 0→1 を開→閉→開の1周期にマップ（位相シフトに依存しない往復）
        open_close_ratio = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
        open_angle = -1.2
        closed_angle = 1.2
        mid = 0.5 * (open_angle + closed_angle)
        full_target = open_angle + (closed_angle - open_angle) * open_close_ratio
        target_finger = mid + (full_target - mid) * float(amplitude_scale)

        if gesture:
            self._apply_joint_target(
                agent, action, "index_joint", target_finger, additive=False
            )
            self._apply_joint_target(
                agent, action, "other_fingers", target_finger, additive=False
            )
            return action

        self._apply_joint_target(agent, action, "shoulder_pitch_joint", -0.45)
        self._apply_joint_target(agent, action, "elbow_joint", 0.9)
        self._apply_joint_target(agent, action, "roll_wrist_joint", 0.0)
        self._apply_joint_target(agent, action, "pitch_wrist_joint", 0.0)
        self._apply_joint_target(agent, action, "index_joint", target_finger)
        self._apply_joint_target(agent, action, "other_fingers", target_finger)
        return action

    def reset_arm_pose(self, agent, action=None):
        """腕・肩・手先の関節を初期角度（0）へ徐々に戻すアクションをセットする。
        
        各関節の現在角度と初期角度(0)の差分を計算し、action にセットする。
        環境の apply_action が1ステップあたりのステップ幅を制限するため、
        大きな角度差でも安全に徐々に戻る。
        
        Args:
            agent: エージェント情報辞書
            action: アクション辞書（省略時は新規作成）
        
        Returns:
            action: 更新されたアクション辞書
        """
        if action is None:
            action = {}
        am = self.env.action_map
        
        # リセット対象の関節リスト
        arm_joints = [
            "shoulder_yaw_joint",
            "shoulder_pitch_joint",
            "elbow_joint",
            "roll_wrist_joint",
            "pitch_wrist_joint",
            "index_joint",
            "other_fingers",
        ]
        
        for joint_name in arm_joints:
            if joint_name == "other_fingers":
                # 各指をリセット
                current_angle = self.env.get_joint_angle(agent, "little_finger_joint")
                action[am["other_fingers"]] = -current_angle
            elif joint_name in am:
                current_angle = self.env.get_joint_angle(agent, joint_name)
                # 目標は初期角度 0 なので、差分 = -current_angle
                action[am[joint_name]] = -current_angle
        
        return action


    def check_ray_visibility(self, start_pos, target_pos, target_id, ignore_ids=None):
        """p.rayTestを用いて、start_posからtarget_posへの視線が通るかチェックする。"""
        if ignore_ids is None:
            ignore_ids = []
        ray_results = p.rayTest(start_pos, target_pos, physicsClientId=self.env.client_id)
        # ray_result: (objectUniqueId, linkIndex, hit_fraction, hit_position, hit_normal)
        hit_obj_id = ray_results[0][0]
        if hit_obj_id == -1: # 何もヒットしなかった
            return True
        if hit_obj_id == target_id: # ターゲット自身にヒット
            return True
        if hit_obj_id in ignore_ids: # 無視するオブジェクトにヒットした場合は一応OKとする
            return True
        return False

    def get_grid_rc_from_pos(self, pos):
        """座標からグリッドの (row, col) を返す。"""
        xh, yh = self.env.arena_half_extents
        x, y = pos[0], pos[1]
        c = int((x + xh) / self.env.cell_w)
        r = int((y + yh) / self.env.cell_h)
        # 範囲内にクリップ
        c = max(0, min(self.env.arena_cols - 1, c))
        r = max(0, min(self.env.arena_rows - 1, r))
        return r, c

    def get_cell_center(self, r, c):
        """(row, col) からセルの中心 (x, y) を返す。"""
        xh, yh = self.env.arena_half_extents
        cx = -xh + self.env.cell_w * (c + 0.5)
        cy = -yh + self.env.cell_h * (r + 0.5)
        return cx, cy

    def find_visible_cell_for_target(self, agent, target_pos, target_body_id):
        """周辺グリッドを探索して、ターゲットが見えるセルの中心座標を返す。"""
        # 親の目の位置
        eye_pos, _ = self.env._agent_eye_pose(agent)
        
        # 直接見えるかチェック（1回のrayTestは軽いため常に確認する）
        ignore_ids = [agent["body"], self.env.infant_body]
        if self.check_ray_visibility(eye_pos, target_pos, target_body_id, ignore_ids):
            if target_body_id in getattr(self, "visibility_cache", {}):
                del self.visibility_cache[target_body_id]
            return "direct", None  # 直視可能
            
        # キャッシュの確認（ターゲットが大きく移動していなければ同じ目的セルを使う）
        if hasattr(self, "visibility_cache") and target_body_id in self.visibility_cache:
            cache = self.visibility_cache[target_body_id]
            if np.linalg.norm(np.array(cache["target_pos"]) - np.array(target_pos)) < 0.1:
                return cache["status"], cache["cell_pos"]

        r_agent, c_agent = self.get_grid_rc_from_pos(eye_pos)
        r_target, c_target = self.get_grid_rc_from_pos(target_pos)
        
        # 自分のいるグリッドから同心円状に探索
        max_dist = max(self.env.arena_rows, self.env.arena_cols)
        
        for dist in range(1, max_dist):
            candidates = []
            for dr in range(-dist, dist + 1):
                for dc in range(-dist, dist + 1):
                    # マンハッタン距離ではなくチェビシェフ距離で矩形に探索
                    if max(abs(dr), abs(dc)) != dist:
                        continue
                    
                    r_cand = r_agent + dr
                    c_cand = c_agent + dc
                    
                    if 0 <= r_cand < self.env.arena_rows and 0 <= c_cand < self.env.arena_cols:
                        # ターゲットと同じグリッドは除外
                        if r_cand == r_target and c_cand == c_target:
                            continue
                        candidates.append((r_cand, c_cand))
            
            # 各候補について視線チェック
            for r, c in candidates:
                cx, cy = self.get_cell_center(r, c)
                # 高さは親の目線を想定
                cand_eye_pos = [cx, cy, eye_pos[2]]
                
                if self.check_ray_visibility(cand_eye_pos, target_pos, target_body_id, ignore_ids):
                    self.visibility_cache[target_body_id] = {"target_pos": target_pos, "status": "found", "cell_pos": (cx, cy)}
                    return "found", (cx, cy)
                    
        self.visibility_cache[target_body_id] = {"target_pos": target_pos, "status": "not_found", "cell_pos": None}
        return "not_found", None

    def approach_object_for_visibility(self, agent, oid: int, action=None, thresh=None, approach_dist: float = 0.05):
        """オブジェクトが見えなければ回避先のセルへ移動し、見えれば注視する"""
        if action is None:
            action = {}
        target_pos = self.env._obj_pos(oid)
        status, cell_pos = self.find_visible_cell_for_target(agent, target_pos, oid)
        
        if status == "direct" or status == "not_found":
            self.gaze_object(agent, oid, action=action)
            return 0.0, None
        
        # 迂回先が見つかった場合、迂回先セルへ移動
        if thresh is None:
            thresh = float(math.radians(15.0))
            
        yaw, distance = self.approach_location(agent, cell_pos, action=action, thresh=thresh, approach_dist=approach_dist)
        if distance is not None and abs(yaw) <= thresh and distance <= 0.05:
            self.gaze_object(agent, oid, yaw_link="torso", action=action)
                
        return yaw, distance

    def approach_infant_for_visibility(self, agent, infant, action=None, thresh=None, approach_dist: float = 0.05):
        """幼児が見えなければ回避先のセルへ移動し、見えれば注視する"""
        if action is None:
            action = {}
        
        # 幼児の目は eyes_link から取得
        target_pos = p.getLinkState(infant["body"], infant["eyes_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4]
        target_body_id = infant["body"]
        status, cell_pos = self.find_visible_cell_for_target(agent, target_pos, target_body_id)
        
        if status == "direct" or status == "not_found":
            self.gaze_agent(agent, infant, action=action)
            return 0.0, None
        
        # 迂回先が見つかった場合、迂回先セルへ移動
        if thresh is None:
            thresh = float(math.radians(15.0)) # 幼児用のアプローチのデフォルト値に合わせる等適宜変更
            
        yaw, distance = self.approach_location(agent, cell_pos, action=action, thresh=thresh, approach_dist=approach_dist)
        if distance is not None and abs(yaw) <= thresh and distance <= 0.05:
            self.gaze_agent(agent, infant, yaw_link="torso", action=action)
                
        return yaw, distance


    def shake_object(
        self,
        agent: Dict[str, Any],
        action=None,
        shake_period: int = 20,
        shake_amp: float = 0.5,
        target_pos=None,
    ):
        """把持済みのオブジェクトを振るモータープリミティブ（最小版）。

        把持済み前提（`agent["held_obj"] is not None`）で呼ぶこと。
        接近・共同注意との組み合わせは呼び出し側プロトコルで行う。
        shake_t を内部位相として使用する。フェーズフラグは持たない。

        Args:
            agent: エージェント情報辞書
            action: アクション辞書（省略時は新規作成）
            shake_period: 1往復のステップ数。
            shake_amp: 手首ピッチの振れ幅 [rad]。
            target_pos: 指定時は固定点へ IK 維持＋手首振り。None なら現行の手首振りのみ。

        Returns:
            action: 更新されたアクション辞書
        """
        if action is None:
            action = {}

        # shake_t を更新して位相を計算
        self.shake_t = (self.shake_t + 1) % shake_period
        phase = self.shake_t / shake_period  # 0.0 → 1.0
        wrist_target = shake_amp * math.sin(2.0 * math.pi * phase)

        if target_pos is not None:
            # 固定点へ IK → 手首ピッチを上書き（加算だと IK と二重になる）→ 指は把持維持
            self.hand_reach_location(agent, target_pos, action=action)
            self._apply_joint_target(
                agent, action, "pitch_wrist_joint", wrist_target, additive=False
            )
            self.grip_object(agent, action)
            return action

        # 手首ピッチを正弦波で往復させる（把持を維持しながら腕を振る）
        self._apply_joint_target(agent, action, "pitch_wrist_joint", wrist_target)

        # 指は閉じて把持を維持
        self._apply_joint_target(agent, action, "index_joint", 3.14)
        self._apply_joint_target(agent, action, "other_fingers", 3.14)

        return action

    def __getattr__(self, name):
        """env へ属性を委譲するフォールバック。Wrapper自身に無い属性を下位環境から引き出す。"""
        # 無限再帰を防ぐ
        if name == "env":
            return super().__getattribute__(name)
        return getattr(self.env, name)
