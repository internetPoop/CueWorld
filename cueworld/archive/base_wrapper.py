from typing import Optional, Tuple, Dict, Any, List
import math
import numpy as np
import gymnasium as gym
import pybullet as p  # 環境内部で使用しているため依存


class WrapperBase(gym.Wrapper):
    """Base class for all custom wrappers in CueWorld environment."""

    def __init__(self, env):
        super().__init__(env)
        self.ik_count = {}
        self.ik = {}
        self.ik_target = {}
        self.shake_t = 0
        self.arm_raise_flag = False
        self.shake_flag = False
        self.hand_over_phase = False  # ハンドオーバーのフェーズ管理用フラグ

    def reset(self, **kwargs):
        """Resets the environment and returns the initial observation."""
        self.shake_t = 0
        self.ik_count = {}
        self.ik = {}
        self.ik_target = {}

        self.arm_raise_flag = False
        self.shake_flag = False
        self.hand_over_phase = False
        return self.env.reset(**kwargs)  
    
    def step(self, action):
        return super().step(action)
    
    # ========= 補助: 幾何/姿勢 =========
    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
        """2ベクトル間角度[rad]。長さ極小なら最大を返し学習安定性確保。"""
        a = a.astype(np.float32); b = b.astype(np.float32)
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na < eps or nb < eps:
            return math.pi  # ベクトル長が極小 → 最大角度を返す．親が物を持つだけでphase2になるのを防ぐため
        cosv = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        return float(math.acos(cosv))
    
    @staticmethod
    def rotation_to_target(vec_from: np.ndarray, vec_to: np.ndarray) -> float:
        """
        2次元ベクトル vec_from を vec_to の方向に合わせるための
        最小回転角（ラジアン）を返す。
        
        時計回りを負、反時計回りを正とする。
        vec_from, vec_to は shape=(2,) の NumPy 配列を想定。
        """
        # 各ベクトルの角度
        theta_from = np.arctan2(vec_from[1], vec_from[0])
        theta_to   = np.arctan2(vec_to[1],   vec_to[0])
        
        # 差をとり、(-π, π] に正規化
        delta = theta_to - theta_from
        while delta > np.pi:
            delta -= 2 * np.pi
        while delta <= -np.pi:
            delta += 2 * np.pi
        
        return delta

    @staticmethod
    def _angle_to_score(angle_rad: float, cutoff_rad: float=math.pi) -> float:
        """角度→整形スコア[0..1]:
        0 以下の角度で 1, cutoff 以上で 0, その間は線形減衰。
        角度が小さいほど一致度が高い指標。
        """
        a = max(0.0, min(cutoff_rad, float(angle_rad)))
        return float(1.0 - a / cutoff_rad)
    
    @staticmethod
    def find_coordinates(P, Q, r, alpha, x_range, y_range, r_2 = None):
        """
        P: 円の中心 (px, py)
        Q: 角度の頂点 (qx, qy)
        r: 半径
        r_2: 求めたい点がQから離れていてほしい最低距離
        alpha: 角度(ラジアン) ∠PQX(0~60度程度)
        """
        candidate_points = []
        px, py = P
        qx, qy = Q
        
        # 1. QからPへの距離 d と 基準角度 theta_base
        dx = px - qx
        dy = py - qy
        d = math.sqrt(dx**2 + dy**2)
        #円の内部に点がある場合なし
        if d <= r:
            return []
        theta_base = math.atan2(dy, dx)
        
        # 2. 二次方程式の判別式などを計算して距離 L 出す
        D_quarter = r**2 - (d * math.sin(alpha))**2
        
        if D_quarter >= 0:
            if r_2 is not None and d < r + r_2:
                cos_alpha_thresh = (d**2 + r_2**2 - r**2) / (2 * d * r_2)
                if cos_alpha_thresh < math.cos(alpha):
                    print("指定の距離での位置が存在しないため，交点を使用")
                    #指定の角度でやると親と幼児が距離r_2より近くに来てしまう場合
                    # オブジェクト周りの円と、親と幼児の距離r_2の円の交点を求め，それを使用
                    # 余弦定理の変形から、Pから垂線の足までの距離 a を求める
                    # a = (r1^2 - r2^2 + d^2) / (2d)
                    a = (r**2 - r_2**2 + d**2) / (2 * d)
                    h = np.sqrt(np.abs(r**2 - a**2))
                    u = np.array([dx, dy]) / d  # PからQへの単位ベクトル
                    perp_u = np.array([-u[1], u[0]])  # 直交ベクトル
                    #交点1
                    inter1 = np.array([qx, qy]) + a * u + h * perp_u
                    #交点2
                    inter2 = np.array([qx, qy]) + a * u - h * perp_u
                    points =  [inter1, inter2]
                    candidate_points = []
                    for point in points:
                        x, y = point
                        # 4. 範囲チェック
                        if (x_range[0] <= x <= x_range[1]) and (y_range[0] <= y <= y_range[1]):
                            candidate_points.append((x, y)) 
            #　オブジェクトから指定した距離で，幼児からある角度をなす場所を探す
            #余弦法則に基づく二次方程式
            term1 = d * math.cos(alpha)
            term2 = math.sqrt(D_quarter)
            l = term1 - term2
            if l < 0:
                return []
            directions = [theta_base + alpha, theta_base - alpha]
            for theta in directions:
                # 座標計算 (始点は Q であることに注意)
                x = qx + l * math.cos(theta)
                y = qy + l * math.sin(theta)
                
                # 4. 範囲チェック
                if (x_range[0] <= x <= x_range[1]) and (y_range[0] <= y <= y_range[1]):
                    candidate_points.append((x, y))
        else:
            #上記位置が存在しない場合は，幼児から伸ばしたオブジェクト中心の半径rの円への接線の交点を使用
            l = math.sqrt(d**2 - r**2) #虚数解の場合、接線距離を採用
            cos = l / d
            sin = r / d
            for _ in range(2):
                x = qx + l * cos
                y = qy + l * sin
                # 4. 範囲チェック
                if (x_range[0] <= x <= x_range[1]) and (y_range[0] <= y <= y_range[1]):
                    candidate_points.append((x, y))
                    
        return candidate_points
    
    def angle_bisector_theorem(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
        """
        ∠ABC の二等分線上にある点 D(辺 AC 上) を求める。
        A, B, C はそれぞれ np.ndarray で座標を表す。
        返り値: 点 D の座標 (np.ndarray)
        """
        AB = A - B
        CB = C - B
        len_AB = np.linalg.norm(AB)
        len_CB = np.linalg.norm(CB)
        if len_AB + len_CB == 0:
            return 0.0
        ratio = len_AB / (len_AB + len_CB)
        AC = C - A
        D = A + ratio * AC
        return D
        

    # ========= 角度/近さ =========
    def gaze_angle_to_object(self, agent: Dict[str, Any], oid: int) -> float:
        """
        視線と物体方向の角度。環境状態依存の瞬時評価。
        yaw_and_pitchがTrueの場合、(yaw, pitch)のタプルを返す。これは親の行動決定に使用する
        """
        eye, fwd = self.env._agent_eye_pose(agent)
        v = self.env._obj_pos(oid) - eye
        return self._angle_between(fwd, v)
    
    def gaze_angle_to_location(self, agent: Dict[str, Any], target_pos: np.ndarray) -> float:
        """
        視線と指定位置方向の角度。環境状態依存の瞬時評価。
        yaw_and_pitchがTrueの場合、(yaw, pitch)のタプルを返す。これは親の行動決定に使用する
        """
        eye, fwd = self.env._agent_eye_pose(agent)
        v = target_pos - eye
        return self._angle_between(fwd, v)
    
    def object_gaze_check(self, agent: Dict[str, Any], oid: int, length: float=5.0, mode=None) -> bool:
        """エージェントが物体を見ているかの判定。raycastで視線上に物体があるか確認。"""
        if mode == "angle":
            angle = self.gaze_angle_to_object(agent, oid)
            return angle < math.radians(15.0)
        else:
            eye_pos, fwd = self.env._agent_eye_pose(agent)
            eye_radius = agent["eyes_radius"]
            eye_pos = eye_pos + fwd * eye_radius  # 目の中心から少し前にずらす
            end_pos = eye_pos + fwd * length
            ray_res = p.rayTest(eye_pos, end_pos, physicsClientId=self.env.client_id)
            return ray_res[0][0] == oid
        
    def what_is_looking_at(self, agent: Dict[str, Any], length: float=5.0) -> Optional[int]:
        """エージェントが見ている物体のIDを返す。見ていない場合は None を返す。"""
        eye_pos, fwd = self.env._agent_eye_pose(agent)
        eye_radius = agent["eyes_radius"]
        eye_pos = eye_pos + fwd * eye_radius  # 目の中心から少し前にずらす
        end_pos = eye_pos + fwd * length
        ray_res = p.rayTest(eye_pos, end_pos, physicsClientId=self.env.client_id)
        oid = ray_res[0][0]
        if oid >= 0 and oid in self.env.active_objects:
            return oid
        else:
            return None

    
    def agent_gaze_check(self, agent: Dict[str, Any], target_agent: Dict[str, Any], length: float=5.0) -> bool:
        """
        エージェントが相手エージェントを見ているかの判定。
        raycastで視線上に相手エージェントがあるか確認。
        返り値: (見ているか(bool), 見ているリンクのインデックス(int))
        """
        eye_pos, fwd = self.env._agent_eye_pose(agent)
        eye_radius = agent["eyes_radius"]
        eye_pos = eye_pos + fwd * eye_radius  # 目の中心から少し前にずらす
        end_pos = eye_pos + fwd * length
        ray_res = p.rayTest(eye_pos, end_pos, physicsClientId=self.env.client_id)
        
        return ray_res[0][0] == target_agent["body"], ray_res[0][1]
    
    def agent_eye_gaze_check(self, agent: Dict[str, Any], target_agent: Dict[str, Any], length: float=5.0) -> bool:
        """
        エージェントが相手エージェントの目を見ているかの判定。
        raycastで視線上に相手エージェントの目のリンクがあるか確認。
        返り値: (見ているか(bool), 見ているリンクのインデックス(int))
        """
        eye_gaze = False
        face_gaze = False
        agent_gaze, gaze_link = self.agent_gaze_check(agent, target_agent, length)
        if agent_gaze:
            eye_gaze = gaze_link == target_agent["eyes_link"]
            face_gaze = gaze_link in target_agent["face_links"]
        return agent_gaze, eye_gaze, face_gaze
    
    def agent_hand_gaze_check(self, agent: Dict[str, Any], target_agent: Dict[str, Any], length: float=5.0, target_link="hand") -> bool:
        """
        エージェントが相手エージェントの手を見ているかの判定。
        raycastで視線上に相手エージェントの手のリンクがあるか確認。
        返り値: (見ているか(bool), 見ているリンクのインデックス(int))
        """
        hand_gaze = False
        agent_gaze, gaze_link = self.agent_gaze_check(agent, target_agent, length)
        if agent_gaze:
            hand_gaze = (gaze_link in target_agent["fingers"] or gaze_link == target_agent["hand_link"])
            arm_gaze = gaze_link in target_agent["arm_links"]
            if target_link == "index":
                target_gaze = gaze_link == target_agent["index_link"]
                return agent_gaze, arm_gaze, hand_gaze, target_gaze
        return agent_gaze, arm_gaze, hand_gaze
    
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
            centor_pos = p.getLinkState(agent["body"], agent["torso_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4]
            v = np.array(target) - np.array(centor_pos)
            _, fwd = self.env._agent_eye_pose(agent)  # 目の向き更新
            yaw = self.rotation_to_target(np.array([fwd[0], fwd[1]], dtype=np.float32), np.array([v[0], v[1]], dtype=np.float32))
              
        # pitch のみ成分で角度計算
        if option == "yaw_only":
            pitch = 0.0
        else:
            # pitch 成分
            eye, fwd = self.env._agent_eye_pose(agent)
            v = np.array(target) - np.array(eye)
            fwd_xy = np.sqrt(fwd[0]**2 + fwd[1]**2)
            v_xy = np.sqrt(v[0]**2 + v[1]**2)
            if fwd_xy < 1e-6 or v_xy < 1e-6:
                pitch = 0.0  # 水平成分が極小なら上下動なしとみなす
            pitch = self.rotation_to_target(np.array([fwd[2], fwd_xy], dtype=np.float32), np.array([v[2], v_xy], dtype=np.float32))
            if pitch < 0.05:
                pitch = pitch * 0.5  # 微小な上下動を抑制
        if option == "pitch_only":
            yaw = 0.0
        if yaw_link == "head":
            action[am["head_pan_joint"]] = yaw
        else:
            action[am["torso_joint"]] = yaw
        action[am["head_tilt_joint"]] = pitch
        return yaw, pitch
    
    def gaze_location(self, agent: Dict[str, Any], target_pos: np.ndarray, yaw_link, action=None):
        """
        指定位置に視線を合わせるアクションを返す。
        頭を対象に向けるときはyawもpitchも動かす場合振動する可能性があるので適時optionを使用
        """
        am = self.env.action_map
        if action is None:
            action = {}
        if yaw_link == "torso":
            action[am["head_pan_joint"]] = -self.env.get_joint_angle(agent, "head_pan_joint")
            torso_pos, fwd = self.env._agent_torso_pose(agent)  # 胴体向き更新
            v = np.array(target_pos) - np.array(torso_pos[:2])
            yaw = self.rotation_to_target(np.array([fwd[0], fwd[1]], dtype=np.float32), np.array([v[0], v[1]], dtype=np.float32))
        elif yaw_link == "head":
            centor_pos = p.getLinkState(agent["body"], agent["torso_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4]
            v = np.array(target_pos) - np.array(centor_pos[:2])
            _, fwd = self.env._agent_eye_pose(agent)  # 目の向き更新
            yaw = self.rotation_to_target(np.array([fwd[0], fwd[1]], dtype=np.float32), np.array([v[0], v[1]], dtype=np.float32))
        if yaw_link == "head":
            action[am["head_pan_joint"]] = yaw
        else:
            action[am["torso_joint"]] = yaw
        return yaw

    def pointing_angle_to_object(self, agent: Dict[str, Any], oid: int) -> float:
        """
        指差し仮定方向(+Z)と物体方向の角度。腕無しは inf を返しスコア化で0になる。
        yaw_and_pitchがTrueの場合、(yaw, pitch)のタプルを返す。これは親の行動決定に使用する
        """
        index_pos, point_dir = self.env._agent_index_pose(agent)
        tip_pos = index_pos + point_dir * agent["index_link_size"]
        v = self.env._obj_pos(oid) - tip_pos # 物体方向ベクトル
        return self._angle_between(point_dir, v)
    
    def _lower_arm_to_object(self, agent: Dict[str, Any], oid: int):
        """腕先位置と物体位置の距離。腕無しは大きな値を返す。"""
        tip, fwd = self.env._lower_arm_pose(agent)  # 腕先位置更新
        target = self.env._obj_pos(oid)
        v = target - tip
        #yaw
        fwd_yaw = np.array([fwd[0],fwd[1]], dtype=np.float32)
        to_target_yaw = np.array([v[0],v[1]], dtype=np.float32)
        yaw = self.rotation_to_target(fwd_yaw, to_target_yaw)
        #pitch
        fwd_xy = np.sqrt(fwd[0]**2 + fwd[1]**2)
        fwd_pitch = np.array([fwd[2], fwd_xy], dtype=np.float32)
        to_target_xy = np.sqrt(v[0]**2 + v[1]**2)
        to_target_pitch = np.array([v[2], to_target_xy], dtype=np.float32)
        pitch = self.rotation_to_target(fwd_pitch, to_target_pitch)
        return (yaw, pitch)
    
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
    
    def hand_reach_object(self, agent: Dict[str, Any], oid: int, action=None, ik_every: int = 0):
        """
        オブジェクトへ手を伸ばす
        手を広げてオブジェクトを掴もうとするところも含む
        """
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
        action[self.env.action_map["roll_wrist_joint"]] = self.get_palm_roll_angle_to_object(agent, oid)
        return action
    
    def hand_reach_location(self, agent: Dict[str, Any], target_pos: np.ndarray, action=None, ik_every: int = 0):
        """
        指定位置へ手を伸ばす
        手の動きは指定しない
        """
        agent_id = agent["body"]
        agent_body = agent["body"]
        ee_link_index = agent["grasp_link"]
        if action is None:
            action = {}
        #ikのターゲットが変わった場合はリセット
        current_target = self.ik_target.get(agent_id)
        if current_target is None or current_target[0] != -1 or np.linalg.norm(np.array(current_target[1]) - np.array(target_pos)) > 0.01:
            self.ik_target[agent_id] = (-1, target_pos)
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
            current_angle = p.getJointState(agent_body, agent[joint_name], physicsClientId=self.env.client_id)[0]
            target_angle = ik_sol[joint_index]
            angle_diff = target_angle - current_angle
            action[self.env.action_map[joint_name]] = angle_diff
        link_state = p.getLinkState(agent_body, ee_link_index, computeForwardKinematics=True, physicsClientId=self.env.client_id)  # 手の姿勢更新
        hand_pos = link_state[4] #手のひらの先の位置
        distance = np.linalg.norm(np.array(target_pos) - np.array(hand_pos))
        #IKカウンタ更新
        if ik_every != 0:
            self.ik_count[agent_id] = self.ik_count.get(agent_id, 0) + 1
        return distance
    
    def get_palm_roll_angle_to_object(self, agent: Dict[str, Any], oid: int) -> float:
        """手のひらロール方向(+X)と物体方向の角度。"""
        link_state = p.getLinkState(agent["body"], agent["hand_link"], physicsClientId=self.env.client_id)
        hand_pos = link_state[4]
        obj_pos = self.env._obj_pos(oid)
        if np.linalg.norm(np.array(obj_pos) - np.array(hand_pos)) < self.env.grasp_distance:
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
            
    def agent_holding_object_check(self, agent: Dict[str, Any], oid: Optional[int] = None) -> bool:
        """エージェントが物体(または特定の物体 oid)を把持しているかを判定。"""
        held_obj = agent.get("held_obj")
        if oid is not None:
            return held_obj == oid
        return held_obj is not None

    
    def gaze_angle_to_agent(self, agent_from: Dict[str, Any], agent_to: Dict[str, Any]) -> float:
        """
        相手頭位置への視線角。相互注視スコア計算に利用。
        """
        eye, fwd = self.env._agent_eye_pose(agent_from)
        target = p.getLinkState(agent_to["body"], agent_to["eyes_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4]
        target = np.array(target, dtype=np.float32)
        return self._angle_between(fwd, target - eye)
    
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
            centor_pos = p.getLinkState(agent_from["body"], agent_from["torso_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4]
            v = np.array(target) - np.array(centor_pos)
            _, fwd = self.env._agent_eye_pose(agent_from)  # 目の向き更新
            yaw = self.rotation_to_target(np.array([fwd[0], fwd[1]], dtype=np.float32), np.array([v[0], v[1]], dtype=np.float32))
            action[am["head_pan_joint"]] = yaw
        
        
        # pitch成分
        eye, fwd = self.env._agent_eye_pose(agent_from)
        v = np.array(target) - np.array(eye)
        fwd_xy = np.sqrt(fwd[0]**2 + fwd[1]**2)
        v_xy = np.sqrt(v[0]**2 + v[1]**2)
        pitch = self.rotation_to_target(np.array([fwd[2] ,fwd_xy], dtype=np.float32), np.array([v[2], v_xy], dtype=np.float32))
        action[am["head_tilt_joint"]] = pitch
        return yaw, pitch

    def is_eye_contact(self):
        """
        お互いのエージェントがアイコンタクトを取っているかなど返す
        """
        parent = self.env.parent
        infant = self.env.infant
        gaze_eye_pi = False
        gaze_eye_ip = False
        eye_contact = False
    
        _, gaze_link_pi = self.agent_gaze_check(parent, infant)
        gaze_ip, gaze_eye_ip, gaze_face_ip = self.agent_eye_gaze_check(infant, parent)
        gaze_eye_pi = gaze_link_pi == infant["eyes_link"]
        if gaze_eye_pi and gaze_eye_ip:
            eye_contact = True
        return eye_contact, gaze_eye_ip, gaze_face_ip, gaze_ip
    
    def _dist_body_to(self, agent: Dict[str, Any], target_pos: np.ndarray) -> float:
        """エージェント胴体(base)位置から target_pos までの水平距離[XY]を返す。"""
        pos, _ = p.getBasePositionAndOrientation(agent["body"], physicsClientId=self.env.client_id)
        ppos = np.asarray(pos, dtype=np.float32)[:2]
        tgt = np.asarray(target_pos, dtype=np.float32)[:2]
        diff = tgt - ppos
        return float(np.linalg.norm(diff))
    
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

    
    def away_from_location(self, agent, target_pos: np.ndarray, action=None, thresh=None, target_dist: float = 1.0):
        """指定位置から離れるための後退量を計算。"""
        
        distance = None
        if action is None:
            action = {}
        if thresh is None:
            thresh = float(getattr(self.env, "gaze_obj_thresh", math.radians(20.0)))
        eye, fwd = self.env._agent_eye_pose(agent)
        v = eye - target_pos #離れるため，逆方向ベクトル
        # yaw の成分で角度計算(ある場所に向かうときはpitchは考慮しない)
        fwd_yaw = np.array([fwd[0], fwd[1]], dtype=np.float32)
        to_target_yaw = np.array([v[0], v[1]], dtype=np.float32)
        head_yaw = self.rotation_to_target(fwd_yaw, to_target_yaw)
        action[self.env.action_map["torso_joint"]] = head_yaw
        # 十分向いている → 後退
        if abs(head_yaw) <= thresh:
            distance = self._dist_body_to(agent, target_pos)
            backward = target_dist - distance
            backward = max(0.0, float(backward))  # クリップ
            action[self.env.action_map["forward"]] = backward
        return head_yaw, distance
    # ========= 二項関係 =========
    def spot_for_dyadic(self, agent_from, agent_to, approach_dist: float = 0.05, angle: float = math.radians(30.0)):
        """エージェントが相手エージェントに近づくときのエージェントが向かうべき地点を返す。"""
        torso_pos, torso_fwd = self.env._agent_torso_pose(agent_to)  # 相手の胴体位置更新
        cos = np.cos(angle)
        sin = np.sin(angle)
        rotation_matrix_plus = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        rotation_matrix_minus = np.array([[cos, sin], [-sin, cos]], dtype=np.float32)
        dir_vector = np.array([torso_fwd[0], torso_fwd[1]], dtype=np.float32)
        dir_vector = dir_vector / np.linalg.norm(dir_vector)
        point_plus = torso_pos[:2] + rotation_matrix_plus @ dir_vector * approach_dist
        point_minus = torso_pos[:2] + rotation_matrix_minus @ dir_vector * approach_dist
        
        x_edge = self.env.arena_half_extents[0] - 0.02
        y_edge = self.env.arena_half_extents[1] - 0.02
        candidates = []
        if -x_edge <= point_plus[0] <= x_edge and -y_edge <= point_plus[1] <= y_edge:
            candidates.append(point_plus)
        if -x_edge <= point_minus[0] <= x_edge and -y_edge <= point_minus[1] <= y_edge:
            candidates.append(point_minus)
        if len(candidates) >= 1:
            #pointsの座標のうち最もfrom_agentに近い点を選ぶ
            min_dist = float('inf')
            best_point = None
            for point in candidates:
                point_2d = np.array([point[0], point[1]], dtype=np.float32)
                dist = np.linalg.norm(np.array(p.getBasePositionAndOrientation(agent_from["body"], physicsClientId=self.env.client_id)[0])[:2] - point_2d)
                if dist < min_dist:
                    min_dist = dist
                    best_point = point_2d
            return best_point
        
        return None
    
    # =========共同注意=========
    def find_joint_attention_area(self, from_agent, to_agent, oid, distance: float, alpha: float = math.radians(30.0), min_infant_parent_distance: float = 0.8):
        #TODO 親との最小距離や角度の値をパラメータ化
        """
        指定位置の周囲で近づくエリアを計算。
        """
        object_pos = self.env._obj_pos(oid)
        i_pos, _ = p.getBasePositionAndOrientation(to_agent["body"], physicsClientId=self.env.client_id)
        x_edge = self.env.arena_half_extents[0] - 0.02
        y_edge = self.env.arena_half_extents[1] - 0.02
        x_range = (-x_edge, x_edge)
        y_range = (-y_edge, y_edge)
        points = self.find_coordinates(object_pos[:2], i_pos[:2], distance, alpha, x_range, y_range, r_2=min_infant_parent_distance)
        #pointsの座標のうち最もfrom_agentに近い点を選ぶ
        if len(points) == 2:
            min_dist = float('inf')
            best_point = None
            for point in points:
                point_2d = np.array([point[0], point[1]], dtype=np.float32)
                dist = np.linalg.norm(np.array(p.getBasePositionAndOrientation(from_agent["body"], physicsClientId=self.env.client_id)[0])[:2] - point_2d)
                if dist < min_dist:
                    min_dist = dist
                    best_point = point_2d
        elif len(points) == 1:
            best_point = np.array([points[0][0], points[0][1]], dtype=np.float32)
        else:
            best_point = None
        
        # 向くべき地点を求める
        dir_target = self.angle_bisector_theorem(best_point, object_pos[:2], i_pos[:2]) if best_point is not None else None
        if best_point is None or dir_target is None:
            return None, None
        return best_point, dir_target
    
    def spot_for_joint_attention(self, agent, oid: int, approach_dist: float = 0):
        """エージェントが注視対象のオブジェクトに近づくときのエージェントが向かうべき地点を返す。"""
        obj_pos = self.env._obj_pos(oid)
        distance = self._dist_body_to(agent, obj_pos)
        if distance < approach_dist:
            point, dir_pos = self.find_joint_attention_area(agent, self.env.infant, oid, approach_dist)
            if point is not None:
                return point, obj_pos, dir_pos
        return None, obj_pos, None
    
    # ========= 掴み/振り キュー関連 =========
    def grasp_object(self, agent, oid: int, action=None, approach_dist: float = 0, main_link_need=False):
        """オブジェクトを掴むアクションを返す。"""
        if action is None:
            action = {}
        am = self.env.action_map
        main_link = "eye" #その過程で使う主なリンク
        yaw, distance = self.approach_object(agent, oid, action, thresh=None, approach_dist=approach_dist) #approach_distはオブジェクトとエージェントの大きさの合計分
        if distance is not None and abs(yaw) <= math.radians(10.0) and distance <= approach_dist:
            main_link = "hand"
            action = self.hand_reach_object(agent, oid, action=action)
            graspable_obj = self.env.detect_grasp_object(agent)
            if oid == graspable_obj:
                action[am["index_joint"]] = 3.14
                action[am["other_fingers"]] = 3.14
        if main_link_need:
            return action, main_link
        return action
    
    
    def put_object_to(self, agent, target_pos: np.ndarray, action=None, approach_dist: float = 0):
        """
        物を置くアクションを返す。
        target_posの値は物の大きさなどを意識して与えること。z軸はものの大きさがいいかも
        """
        if action is None:
            action = {}
        am = self.env.action_map
        ang_ok, distance_ok = self.approach_location(agent, target_pos, action, thresh=None, approach_dist=approach_dist) #approach_distはオブジェクトとエージェントの大きさの合計分
        if ang_ok and distance_ok:
            distance = self.hand_reach_location(agent, target_pos, action=action)  # Noneで手を伸ばす
            cur_roll_wrist = self.env.get_joint_angle(agent, "roll_wrist_joint")
            action[am["roll_wrist_joint"]] = -cur_roll_wrist  # 手
            if distance <= 0.05 and abs(cur_roll_wrist) <= 0.1:
                # 開放
                action[am["index_joint"]] = -3.14
                action[am["other_fingers"]] = -3.14
            else:
                # 掴む
                action[am["index_joint"]] = 3.14
                action[am["other_fingers"]] = 3.14
        return action
    
    def show_object(self, from_agent, to_agent, oid=None, action=None, approach_dist: float = 0, thresh=None, sub_thresh=None, ik_every: int = 0):
        """
        オブジェクトなどを見せるアクションを返す。
        手で持ったオブジェクトを相手に見せる動作を行う。
        ここでは相手の目線付近に持ち上げる動作を実装する。
        物は持っている前提。
        """
        if action is None:
            action = {}
        am = self.env.action_map
        #TODO: このapproach_distanceはif not self.close_to_infant:の中で使うべきかも．要確認
        yaw, distance = self.approach_infant(from_agent, to_agent, action, thresh=thresh if thresh is not None else float(math.radians(20.0)), approach_dist=approach_dist)
        if not self.close_to_infant and distance is not None and distance <= approach_dist and abs(yaw) < sub_thresh if sub_thresh is not None else float(math.radians(10.0)):
            self.close_to_infant = True
        if self.close_to_infant:
            # --- 変更: 持ち上げ(pitch) + 縦振り(pitch) ---
            # まずは腕を伸ばす
            if self.arm_raise_flag == False:
                cur_shoulder_pitch = self.env.get_joint_angle(from_agent, "shoulder_pitch_joint")
                cur_shoulder_yaw = self.env.get_joint_angle(from_agent, "shoulder_yaw_joint")
                cur_wrist_pitch = self.env.get_joint_angle(from_agent, "pitch_wrist_joint")
                shoulder_pitch = -1.57 - cur_shoulder_pitch
                action[am["shoulder_pitch_joint"]] = shoulder_pitch
                action[am["shoulder_yaw_joint"]] = -cur_shoulder_yaw
                action[am["pitch_wrist_joint"]] = -cur_wrist_pitch
                cur_head_pan = self.env.get_joint_angle(from_agent, "head_pan_joint")
                action[am["head_pan_joint"]] = -cur_head_pan
                cur_elbow = self.env.get_joint_angle(from_agent, "elbow_joint")
                action[am["elbow_joint"]] = -cur_elbow
                if abs(shoulder_pitch)<0.01 and abs(cur_elbow) < 0.01 and abs(cur_shoulder_yaw) < 0.01 and abs(cur_wrist_pitch) < 0.01 and abs(cur_head_pan) < 0.01:
                    self.arm_raise_flag = True
            else:
                #親と幼児の胴体の真ん中
                parent_torso_pos = np.array(p.getLinkState(from_agent["body"], from_agent["torso_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4])
                infant_torso_pos = np.array(p.getLinkState(to_agent["body"], to_agent["torso_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4])
                mid_pos = (parent_torso_pos + infant_torso_pos) / 2.0
                target_pos = np.array([mid_pos[0], mid_pos[1], 0.7878], dtype=np.float32)
                agent_id = from_agent["body"]
                current_target = self.ik_target.get(agent_id)
                if current_target is None or np.linalg.norm(np.array(current_target[1]) - target_pos) > 0.05: #ここの閾値は緩めのほうがいい？対象が動くことを考慮
                    self.ik_target[agent_id] = (-1, target_pos)
                    self.ik_count[agent_id] = 0
                    self.ik[agent_id] = None
                if self.ik.get(agent_id) is None or (ik_every != 0 and self.ik_count.get(agent_id, 0) % ik_every == 0):
                    agent_body = from_agent["body"]
                    ee_link_index = from_agent["grasp_link"]  # 手先のリンク
                    # torsoの可動域を現在のtorsoの状態に固定する．
                    torso_index = from_agent["joints_for_point"]["torso_joint"]
                    current_torso_angle = self.env.get_joint_angle(from_agent, "torso_joint")
                    from_agent["lower_limits"][torso_index] = current_torso_angle
                    from_agent["upper_limits"][torso_index] = current_torso_angle
                    self.ik[agent_id] = p.calculateInverseKinematics(
                        agent_body,
                        ee_link_index,
                        target_pos,
                        lowerLimits=from_agent["lower_limits"],
                        upperLimits=from_agent["upper_limits"],
                        jointRanges=from_agent["joint_ranges"],
                        restPoses=from_agent["restposes"],
                        maxNumIterations=50,
                        residualThreshold=1e-4,
                        physicsClientId=self.env.client_id
                    )
                
                ik_sol = self.ik[agent_id]
                for joint_name, joint_index in from_agent["joints_for_reach"].items(): #reachと同じ関節群を使用
                    current_angle = p.getJointState(from_agent["body"], from_agent[joint_name], physicsClientId=self.env.client_id)[0]
                    target_angle = ik_sol[joint_index]
                    angle_diff = target_angle - current_angle
                    action[self.env.action_map[joint_name]] = angle_diff
                if ik_every != 0:
                    self.ik_count[agent_id] = self.ik_count.get(agent_id, 0) + 1
        self.gaze_object(from_agent, oid, yaw_link="head", action=action, option="pitch_only")
        action[self.env.action_map["index_joint"]] = 3.14 
        action[self.env.action_map["other_fingers"]] = 3.14 
        return action
    
    def shake_object(self, from_agent, to_agent, oid=None, action=None, thresh=None, sub_thresh=None, approach_dist: float = 0, shake_period: int = 20, shake_amp: float = 0.5, ik_every: int = 0):
        """
        物を振るアクションを返す。
        物は持っている前提。
        """
        if action is None:
            action = {}
        am = self.env.action_map
        yaw, distance = self.approach_infant(from_agent, to_agent, action, thresh=thresh if thresh is not None else float(math.radians(10.0)), approach_dist=approach_dist)
        if not self.close_to_infant and distance is not None and distance <= approach_dist and abs(yaw) < sub_thresh if sub_thresh is not None else float(math.radians(10.0)):
            self.close_to_infant = True
        if self.close_to_infant:
            if not self.arm_raise_flag:
                cur_shoulder_pitch = self.env.get_joint_angle(from_agent, "shoulder_pitch_joint")
                cur_shoulder_yaw = self.env.get_joint_angle(from_agent, "shoulder_yaw_joint")
                cur_wrist_pitch = self.env.get_joint_angle(from_agent, "pitch_wrist_joint")
                shoulder_pitch = -1.57 - cur_shoulder_pitch
                action[am["shoulder_pitch_joint"]] = shoulder_pitch
                action[am["shoulder_yaw_joint"]] = 0
                action[am["pitch_wrist_joint"]] = -cur_wrist_pitch
                cur_head_pan = self.env.get_joint_angle(from_agent, "head_pan_joint")
                action[am["head_pan_joint"]] = -cur_head_pan
                cur_elbow = self.env.get_joint_angle(from_agent, "elbow_joint")
                action[am["elbow_joint"]] = - cur_elbow
                if abs(shoulder_pitch) < 0.01 and abs(cur_elbow) < 0.01 and abs(cur_wrist_pitch) < 0.01 and abs(cur_head_pan) < 0.01:
                    self.arm_raise_flag = True
                    self.shake_flag = False
                    self.shake_t = 0
            elif self.shake_flag == False:
                parent_torso_pos = np.array(p.getLinkState(from_agent["body"], from_agent["torso_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4])
                infant_torso_pos = np.array(p.getLinkState(to_agent["body"], to_agent["torso_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4])
                mid_pos = (parent_torso_pos + infant_torso_pos) / 2.0
                target_pos = np.array([mid_pos[0], mid_pos[1], 0.6], dtype=np.float32)
                agent_id = from_agent["body"]
                current_target = self.ik_target.get(agent_id)
                if current_target is None or np.linalg.norm(np.array(current_target[1]) - target_pos) > 0.05:
                    self.ik_target[agent_id] = (-1, target_pos)
                    self.ik_count[agent_id] = 0
                    self.ik[agent_id] = None
                if self.ik.get(agent_id) is None or (ik_every != 0 and self.ik_count.get(agent_id, 0) % ik_every == 0):
                    agent_body = from_agent["body"]
                    ee_link_index = from_agent["grasp_link"]
                    # torsoの可動域を現在のtorsoの状態に固定する．
                    torso_index = from_agent["joints_for_point"]["torso_joint"]
                    current_torso_angle = self.env.get_joint_angle(from_agent, "torso_joint")
                    from_agent["lower_limits"][torso_index] = current_torso_angle
                    from_agent["upper_limits"][torso_index] = current_torso_angle
                    self.ik[agent_id] = p.calculateInverseKinematics(
                        agent_body,
                        ee_link_index,
                        target_pos,
                        lowerLimits=from_agent["lower_limits"],
                        upperLimits=from_agent["upper_limits"],
                        jointRanges=from_agent["joint_ranges"],
                        restPoses=from_agent["restposes"],
                        maxNumIterations=50,
                        residualThreshold=1e-4,
                        physicsClientId=self.env.client_id
                    )
                complete_flags = []
                ik_sol = self.ik[agent_id]
                for joint_name, joint_index in from_agent["joints_for_reach"].items():
                    current_angle = p.getJointState(from_agent["body"], from_agent[joint_name], physicsClientId=self.env.client_id)[0]
                    target_angle = ik_sol[joint_index]
                    angle_diff = target_angle - current_angle
                    if abs(angle_diff) < 0.01:
                        complete_flags.append(True)
                    else:
                        complete_flags.append(False)
                    action[self.env.action_map[joint_name]] = angle_diff
                if ik_every != 0:
                    self.ik_count[agent_id] = self.ik_count.get(agent_id, 0) + 1
                if all(complete_flags):
                    self.shake_flag = True
                    self.shake_t = 0
            elif self.shake_flag:
                self.shake_t += 1
                shake_angle = shake_amp * self.env.parent_joint_mappings["elbow_joint"] * math.sin(2.0 * math.pi * (self.shake_t / shake_period))
                action[am["elbow_joint"]] += shake_angle
        #これヘドバンしちゃうかも？要調整
        #self.gaze_object(from_agent, oid, yaw_link="head", action=action, option="pitch_only")
        action[am["index_joint"]] = 3.14 
        action[am["other_fingers"]] = 3.14 
        return action
    
    #=========初期姿勢・腕下げ関連=========
    def reset_arm_pose(self, agent, action=None):
        """
        腕を下げてセットポイント(初期姿勢)にするためのアクションを返す。
        """
        if action is None:
            action = {}
        am = self.env.action_map
        
        # 腕や手に関係する関節をリストアップし、それぞれの初期角度(通常は0.0)に戻す
        arm_joints = [
            "shoulder_pitch_joint",
            "shoulder_yaw_joint",
            "elbow_joint"
        ]
        
        for joint in arm_joints:
            if joint in am:
                current_angle = self.env.get_joint_angle(agent, joint)
                # ターゲット角度を 0.0 とする (腕を下ろした状態)
                target_angle = 0.0
                action[am[joint]] = target_angle - current_angle
                
        # 腕を下げた状態に戻すので、フラグ等もリセットする
        self.arm_raise_flag = False
        self.shake_flag = False
        self.shake_t = 0
        
        return action

    #=========譲渡，受け取り関連=========
    def hand_over(self,agent,action=None):
        """
        エージェントが物を手渡すアクションを返す。
        """
        #TODO: まだ仮実装．実際に使うときに調整が必要
        if action is None:
            action = {}
        am = self.env.action_map
        #幼児に近づく
        if not self.hand_over_phase:
            yaw, distance = self.approach_infant(agent, self.env.infant, action=action, thresh=None, approach_dist=0.8)
        #幼児にわたす
            if distance is not None and abs(yaw) <= math.radians(5.0) and distance <= 0.8:
                self.hand_over_phase = True
        if self.hand_over_phase:
            target_agent = self.env.infant
            target_pos = p.getLinkState(target_agent["body"], target_agent["grasp_link"], computeForwardKinematics=True, physicsClientId=self.env.client_id)[4]
            self.hand_reach_location(agent, target_pos, action=action)
            little_finger_angle = self.env.get_joint_angle(agent, "little_finger_joint")
            index_angle = self.env.get_joint_angle(agent, "index_joint")
            action[am["index_joint"]] = 1.6 - index_angle
            action[am["other_fingers"]] = 1.6 - little_finger_angle
        return action
    
    def receive_object(self,agent,action=None):
        """
        エージェントが物を受け取るアクションを返す。
        """
        #TODO: まだ仮実装．実際に使うときに調整が必要
        if action is None:
            action = {}
        am = self.env.action_map
        target_agent = self.env.infant
        oid = target_agent["held_obj"]
        if oid is None:
            return action
        #幼児に近ずく
        yaw, distance = self.approach_infant(agent, target_agent, action=action, thresh=None, approach_dist=0.8)
        #幼児から受け取る
        if distance is not None and abs(yaw) <= math.radians(10.0) and distance <= 0.8:
            graspable_obj = self.env.detect_grasp_object(agent)
            if oid != graspable_obj:
                action[am["shoulder_yaw_joint"]] = - self.env.get_joint_angle(agent, "shoulder_yaw_joint")
                action[am["shoulder_pitch_joint"]] = -1.0 - self.env.get_joint_angle(agent, "shoulder_pitch_joint")
                action[am["elbow_joint"]] = -0.75 - self.env.get_joint_angle(agent, "elbow_joint")
                roll_wrist_angle = self.env.get_joint_angle(agent, "roll_wrist_joint")
                if roll_wrist_angle >= 0:
                    action[am["roll_wrist_joint"]] = 3.14 - roll_wrist_angle
                else:
                    action[am["roll_wrist_joint"]] = -3.14 - roll_wrist_angle
                action[am["pitch_wrist_joint"]] = -0.21 - self.env.get_joint_angle(agent, "pitch_wrist_joint")
                action[am["index_joint"]] = 3.14
                action[am["other_fingers"]] = 3.14
            else:
                action[am["index_joint"]] = -3.14
                action[am["other_fingers"]] = -3.14
        return action
            
    # ========= 親-幼児 関連 =========
    def get_parent_infant_distance(self, kind: str = "3d") -> float:
        """
        親と幼児の胴体(base)位置間距離を返す。
        kind:
          "3d"  : 3次元ユークリッド距離
          "xy"  : 水平距離 (XY平面)
          "xz"  : XZ平面距離
          "yz"  : YZ平面距離
        """
        p_pos, _ = p.getBasePositionAndOrientation(self.parent_body, physicsClientId=self.env.client_id)
        i_pos, _ = p.getBasePositionAndOrientation(self.infant_body, physicsClientId=self.env.client_id)
        p_pos = np.array(p_pos, dtype=np.float32)
        i_pos = np.array(i_pos, dtype=np.float32)
        diff = p_pos - i_pos
        if kind == "xy":
            return float(np.linalg.norm(diff[:2]))
        if kind == "xz":
            return float(np.linalg.norm(diff[[0,2]]))
        if kind == "yz":
            return float(np.linalg.norm(diff[1:]))
        return float(np.linalg.norm(diff))

    # =========表情関連=========
    def expression(self, agent, positive: bool, action=None):
        """
        エージェントが表情をするアクションを返す。
        """
        if action is None:
            action = {}
        am = self.env.action_map
        #ここでは笑顔を作るために口角を上げる関節
        if positive:
            action[am["lip"]] = -1.0  # 口角を上げる
        else:
            action[am["lip"]] = 1.0   # 口角を下げる
        return action
    # ========= 音声キュー関連 =========
    def _parent_voice_on(self) -> float:
        """親の音声キュー強度[-1..1]を返す（env.parent_voice_levelを参照）。"""
        # 環境が保持している現在の声レベルを利用
        return float(getattr(self.env, "parent_voice_level", 0.0))

    def _infant_voice_on(self) -> float:
        """幼児の音声キュー強度[-1..1]を返す（env.infant_voice_levelを参照）。"""
        return float(getattr(self.env, "infant_voice_level", 0.0))















