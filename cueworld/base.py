"""PyBullet を用いた共同注意（Joint Attention）タスク環境。

概要:
- 目的:
  本環境は頭・腕・腰の基本的な運動と、把持・音声キュー・前進（キネマティク）を備える。
- 座標系:
  右手系。前方=+x, 左=+y, 上=+z。カメラの前方は headローカル +X 方向。
- 行動（各エージェント7次元）:
  [Δyaw, Δpitch, forward, voice, arm_pitch, arm_yaw, grasp]
  Δyaw/Δpitch は head_step 乗算（rad），forward は move_step 乗算（m，<=0 で移動無し）。
  voice は [0..1]，grasp は把持コマンド（>0:掴む, <0:離す, 0:何もしない）。
  注意: 本実装では首の yaw は胴体の腰ヨーに適用し、首そのもののyawは 0 に保つ設計とする
       （二重回転や視点の過剰変化を避け、腰回転へ一元化するため）。
- 観測（Dict）:
  "image": 幼児視点RGB画像（getCameraImage）
  "head": 幼児の頭角 [yaw, pitch]
  "infant_voice": 幼児の音声キュー強度（[0..1]）
  "parent_voice": 親の音声キュー強度（[0..1]）
  現状は常にRGB画像を返す（"state+rgb"などの観測モードは未導入）。
- 報酬:
  [幼児, 親] の2要素。現状は急激な頭運動に小ペナルティ（big_move）のみ。
  ターゲット整列・共同注意の整合ボーナスは将来拡張で追加可能。
- 幾何・物理:
  矩形アリーナの周囲に静的な壁（BOX）。床は plane.urdf。
  オブジェクトは親と幼児の間の帯域に密配置（最小距離制約あり）。
- レンダリング:
  ヘッドレス時は可能なら EGL（ER_BULLET_HARDWARE_OPENGL），
  失敗時は TinyRenderer。SSAA で内部高解像度を縮小して返す。

注記:
- compute_forward_dxdy は「頭の前方（+X）ベクトルのXY投影」に沿って前進距離を与える。
  真上/真下でXY成分が極小のときは base(+X) にフォールバック。
- キネマティク移動は「貫通（ContactDistance<0）」のみを衝突とみなして拒否し、
  0以上の接触（押す/擦る）は許可するため、オブジェクトを押せる。
"""

import math
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import pkgutil
from cueworld.paths import PACKAGE_DIR
from cueworld.intent import DEFAULT_VOICE_VOLUME_THRESHOLD
from typing import Optional, Tuple, List, Dict, Any, Union

# fixed_placements の z 省略時、および bbox 未設定オブジェクトの床接触高さフォールバック [m]
DEFAULT_FIXED_PLACEMENT_Z = 0.05
import time
import os
import importlib

#GPUの番号をこれで指定可能
#os.environ["PYBULLET_EGL_DEVICE"] = "1"  # ヘッドレス実行で使うGPUを指定する場合に設定

# ---------- ユーティリティ ----------
def _downsample_lanczos(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """RGB画像を単純平均で縮小する（SSAAのダウンサンプル相当）。
    引数:
      rgb (np.ndarray): 入力RGB画像 (Hs, Ws, 3) uint8
      w (int): 出力幅（入力幅を等分割できる整数）
      h (int): 出力高さ（入力高さを等分割できる整数）
    戻り値:
      np.ndarray: 形状 (h, w, 3) の uint8 画像
    注意:
      - Pillow等を使わず、等分割ブロックの面積平均で縮小する簡易実装。
      - SSAA倍率が整数のときに有効。
    """
    # Pillowなしでの近似：面積平均でダウンサンプル（簡易実装）
    # 64や128に落とすときは、ssaa=2/4のとき「等分割」なので平均で十分効果あり
    Hs, Ws, _ = rgb.shape
    assert Hs % h == 0 and Ws % w == 0, "SSAA倍率は整数で!"
    ky, kx = Hs // h, Ws // w
    rgb = rgb.reshape(h, ky, w, kx, 3).mean(axis=(1,3))
    return rgb.astype(np.uint8)

# ---------- 環境 ----------
# 幼児 joint_angles 観測の正本（action_map の関節スロット。forward / 音声は含めない）
_INFANT_JOINT_OBS_SLOTS_FULL = (
    "torso_joint",
    "head_pan_joint",
    "head_tilt_joint",
    "shoulder_yaw_joint",
    "shoulder_pitch_joint",
    "elbow_joint",
    "roll_wrist_joint",
    "pitch_wrist_joint",
    "index_joint",
    "other_fingers",
    "lip",
)
_INFANT_JOINT_OBS_WRAPPED = frozenset({"torso_joint", "roll_wrist_joint"})
_INFANT_JOINT_OBS_SOURCE = {
    "other_fingers": "little_finger_joint",
    "lip": "left_lip_joint",
}

# 幼児 fatigue チャネル（観測・内部状態の並び。ラップ関節・前進・pitch/v_* は対象外）
_INFANT_FATIGUE_CHANNELS = (
    "head_pan_joint",
    "head_tilt_joint",
    "shoulder_yaw_joint",
    "shoulder_pitch_joint",
    "elbow_joint",
    "pitch_wrist_joint",
    "index_joint",
    "other_fingers",
    "lip",
    "volume",
)
_INFANT_FATIGUE_GAIN = {
    "head_pan_joint": 0.5,
    "head_tilt_joint": 0.5,
    "shoulder_yaw_joint": 1.0,
    "shoulder_pitch_joint": 1.0,
    "elbow_joint": 1.0,
    "pitch_wrist_joint": 0.5,
    "index_joint": 0.5,
    "other_fingers": 0.5,
    "lip": 0.5,
    "volume": 1.0,
}
_INFANT_FATIGUE_DIM = len(_INFANT_FATIGUE_CHANNELS)

# 手の形（_get_hand_shape / classify_hand_shape）— 境界と理想姿勢の正本
HAND_SHAPE_BOUNDARY = 1.57
INDEX_OPEN = 0.0
INDEX_CLOSED = 2.8
THUMB_OPEN = 0.0
THUMB_CLOSED = -2.8

_HAND_SHAPE_IDEAL = {
    "open": (INDEX_OPEN, THUMB_OPEN),
    "closed": (INDEX_CLOSED, THUMB_CLOSED),
    "pointing": (INDEX_OPEN, THUMB_CLOSED),
}
# 各クラス内で理想から最も遠い角（境界付近）までの距離
_HAND_SHAPE_D_MAX = {
    "open": math.hypot(HAND_SHAPE_BOUNDARY, HAND_SHAPE_BOUNDARY),
    "closed": math.hypot(
        INDEX_CLOSED - HAND_SHAPE_BOUNDARY,
        THUMB_CLOSED - (-HAND_SHAPE_BOUNDARY),
    ),
    "pointing": math.hypot(
        HAND_SHAPE_BOUNDARY,
        THUMB_CLOSED - (-HAND_SHAPE_BOUNDARY),
    ),
}


def classify_hand_shape(
    index_angle: float,
    thumb_angle: float,
) -> Tuple[str, float, float]:
    """index / thumb 角から手形・握力・強度 [0, 1] を返す。

    強度はその形の理想姿勢を 1、その形と判定される一番端を 0 とする。
    other の強度は常に 0。
    """
    grip_power = 0.0
    boundary = HAND_SHAPE_BOUNDARY

    if index_angle < boundary and thumb_angle > -boundary:
        hand_state = "open"
        ideal_i, ideal_t = _HAND_SHAPE_IDEAL["open"]
    elif index_angle >= boundary and thumb_angle <= -boundary:
        hand_state = "closed"
        grip_power = thumb_angle
        ideal_i, ideal_t = _HAND_SHAPE_IDEAL["closed"]
    elif index_angle < boundary and thumb_angle < -boundary:
        hand_state = "pointing"
        ideal_i, ideal_t = _HAND_SHAPE_IDEAL["pointing"]
    else:
        return "other", 0.0, 0.0

    dist = math.hypot(index_angle - ideal_i, thumb_angle - ideal_t)
    d_max = _HAND_SHAPE_D_MAX[hand_state]
    intensity = float(np.clip(1.0 - dist / d_max, 0.0, 1.0))
    return hand_state, grip_power, intensity


class BaseEnv(gym.Env):
    """共同注意タスク環境（PyBullet）。

    観測:
      - Dict 形式で常に以下のキーを返す:
        - "image": 幼児視点RGB画像（shape=(H,W,3), uint8）
        - "head": 幼児の頭角 [yaw, pitch]（rad）
        - "infant_voice": 幼児の音声キュー強度（[0..1]）
        - "parent_voice": 親の音声キュー強度（[0..1]）

    カメラ:
      head リンク姿勢から鼻先位置（+X方向オフセット）と前方ベクトルを求め、
      視点（eye）と注視点（target=eye+dir）を構築。上方向は+Z固定。
      return_views=True のときは info["views"] に親/幼児の画像を格納可能。
    """

    # 属性一覧（Attributes）
    # - gui (bool): GUI表示を行うかどうか。False の場合はヘッドレスで実行。
    # - num_agents (int): エージェント数（1 or 2）。
    # - num_objects (int): アリーナに配置するオブジェクト数。
    # - max_steps (int): 1エピソードの最大ステップ数。
    # - head_step (float): |action[1]|=1 のときの首ピッチΔ[rad]。
    # - move_step (float): |action[2]|=1 のときの前進距離[m]（負値は移動無し）。
    # - arm_step (float): |action[4 or 5]|=1 のときの腕角Δ[rad]。
    # - waist_step (float): |action[0]|=1 のときの腰ヨーΔ[rad]。
    # - grasp_distance (float): palm 基準の把持検出しきい値[m]（自動計算可）。
    # - reach_radius (float): エージェント base からの XY 到達圏[m]（高さは見ない。自動計算可）。
    # - place_dist / grasp_approach_dist / stand_dist (float): 配置・把持接近・giving 立ち位置（reach から派生可）。
    # - grasp_margin / reach_factor / place_margin / grasp_approach_delta: 自動計算用チューニング定数。
    # - first_distance (float): 親と幼児の初期X距離（親=+X/幼児=-Xに配置）。
    # - band_width (float): 親子間帯域の幅（Y方向）—オブジェクト配置領域。
    # - margin (float): 壁・胴体等との最小余白[m]（アリーナ配置用。grasp_margin とは別）。
    # - min_dist (float): オブジェクト同士の最小距離[m]。
    # - foward_by_motor (bool): モーターによる前進を有効化するか。
    # - head_pitch_limit (float): 首ピッチの制限角[rad]。
    # - infant_waist_yaw_limit (float): 幼児用腰ヨーの制限角[rad]。
    # - arena_half_extents (tuple[float,float]): アリーナ半径 (x半径, y半径)。
    # - wall_thickness (float): 壁の厚み[m]。
    # - wall_height (float): 壁の高さ[m]。
    # - static_bodies (list[int]): 衝突検査対象の静的ボディID（床を除く壁等）。
    # - return_views (bool): Trueで info["views"] にRGB画像を返す。
    # - view_width, view_height (int): 画像解像度。
    # - view_fov (float): カメラFOV[deg]。
    # - view_near, view_far (float): カメラのnear/far。
    # - ssaa (int): スーパーサンプリング倍率（内部高解像度→縮小）。
    # - infant_voice_level (float): 幼児の発話強度[0..1]。
    # - parent_voice_level (float): 親の発話強度[0..1]。
    # - small_motion_eps (float): 小さな関節ノイズをスナップする閾値（首/腕）。
    # - small_waist_eps (float): 腰ヨーのスナップ閾値。
    # - observation_space (gym.Space): 観測空間。
    # - single_action_space (gym.Space): エージェント1体の行動空間。
    # - action_space (gym.Space): (幼児, 親) のTuple行動空間。
    # - client_id (int|None): PyBulletのクライアントID。
    # - egl_ok (bool): EGLプラグインが利用可能か。
    # - plugin_id (int): EGLプラグインID（未使用時は -1）。
    # - debug_info (bool): デバッグ情報（get_info）の有効化フラグ。
    # - parent, infant (dict): エージェントのメタ情報（body, 各ジョイントindex等）。
    # - parent_body, infant_body (int): エージェントのベースボディID。
    # - objects (list[int]): 生成したオブジェクトのボディID一覧。
    # - step_count (int): 現在のステップカウント。

    def __init__(self,
                 gui=False,
                 id=0,
                 name="base",
                 seed=0,
                 num_agents=2,
                 num_objects=3,
                 parent_no_movement: bool = False,
                 max_steps=200,
                 voice_window_size=3,
                 torso_step=0.06,
                 head_pan_step=0.10,
                 head_tilt_step=0.10,
                 shoulder_yaw_step=0.12,
                 shoulder_pitch_step=0.12,
                 elbow_step=0.12,
                 roll_wrist_step=0.12,
                 pitch_wrist_step=0.12,
                 index_step=0.15,
                 other_fingers_step=0.15,
                 lip_step=0.15,
                 forward_step=0.04,
                 parent_torso_step=0.10,
                 parent_head_pan_step=0.15,
                 parent_head_tilt_step=0.15,
                 parent_shoulder_yaw_step=0.20,
                 parent_shoulder_pitch_step=0.20,
                 parent_elbow_step=0.20,
                 parent_roll_wrist_step=0.20,
                 parent_pitch_wrist_step=0.20,
                 parent_index_step=0.30,
                 parent_other_fingers_step=0.30,
                 parent_lip_step=0.30,
                 parent_forward_step=0.10,
                 grasp_distance=None,  # None=自動（grasp_contact_distance + grasp_margin）
                 grasp_dir_threshold=30.0,
                 grasp_object_offset=0.075,
                 reach_radius=None,  # None=自動（床置き掴み姿勢の FK 計測 × reach_factor）
                 grasp_margin: float = 0.12,
                 reach_factor: float = 0.85,
                 place_margin: float = 0.10,
                 grasp_approach_delta: float = 0.05,
                 place_dist=None,  # None=自動（reach_radius - place_margin）
                 grasp_approach_dist=None,  # None=自動（reach_radius + grasp_approach_delta）
                 stand_dist=None,  # None=自動（place_dist + grasp_approach_dist）
                 margin : float = 0.35,
                 min_dist : float = 0.35,
                 foward_by_motor: bool = False,
                 arena_half_extents=(2.5, 2.0),
                 wall_thickness=0.01,
                 wall_height=1.0,
                 arena_rows: int = 3,
                 arena_cols: int = 4,
                 layout_grid: list[list[str]] = None,
                 fixed_placements: list[dict] = None,
                 placement_method: str = "grid",  # "grid" または "random"
                 return_views: bool = False,
                 output_parent_view: bool = False,
                 view_width: int = 64,
                 view_height: int = 64,
                 view_fov: float = 90.0,
                 view_near: float = 0.01,
                 view_far: float = 5.0,
                 ssaa: int = 1,
                 debug_info = False,
                 small_motion_eps: float = 1e-3,
                 small_waist_eps: float = 1e-3,
                 shake_move_threshold: float = 0.002,  # 振る検出のしきい値（胴体ローカル座標での変位[m]/サブステップ）
                 disable_all_body_object_collision: bool = True,
                 action_repeat: int = 1,
                 infant_action_names: list[str] = None,
                 infant_obs_keys: list[str] = None,
                 layout_freeze: str = "none",
                 non_linear_action: bool = False,
                 easy_voice: bool = True,
                 random_parent_pos: bool = False,
                 object_counts: dict[str, int] = None,
                 eval_mode: bool = False,  # 評価モードフラグ
                 reverse_mode: bool = False,  # True のとき、親と幼児の役割を逆転させる（親が受け取る側、幼児が渡す側になる）
                 parent_invisible_mode: bool = False,  # Trueのとき、親を初期化時に画面外(-100m)に配置する
                 parent_preference: float = 1.2,
                 enable_fatigue: bool = False,
                 enable_emotion: bool = False,
                 fatigue_high: float = 30.0,
                 fatigue_low: float = 10.0,
                 fatigue_recovery: float = 0.3,
                 voice_volume_threshold: float = DEFAULT_VOICE_VOLUME_THRESHOLD,
                 fall_per_step: float = 0.01,
                 *args, **kwargs
                 ):
        """引数の主な意味（抜粋）:
        - gui: True で PyBullet GUI を起動。False はヘッドレス（EGL/TinyRenderer）。
        - num_objects: オブジェクト数。親子間帯域に最小距離制約つきで配置。
        - max_steps: 1エピソード上限ステップ。
        - head_step: 首 yaw/pitch の1ステップ目標変化量（|action|=1 のときのrad）。Noneの場合はモードに応じて自動設定。
        - move_step: 前進の1ステップ距離[m]（後退は禁止: action<=0 で移動無し）。Noneの場合はモードに応じて自動設定。
        - arm_step: 肩の yaw/pitch の1ステップ目標変化量[rad]。Noneの場合はモードに応じて自動設定。
        - waist_step: 腰ヨーの1ステップ目標変化量[rad]（行動からは内部反映）。Noneの場合はモードに応じて自動設定。
        - grasp_distance: palm 基準の把持検出しきい値[m]。None ならロード後に自動計算。
        - reach_radius: エージェント base からの XY 到達圏[m]（高さは見ない）。None なら床置き掴み姿勢の FK 計測から自動。
        - grasp_margin / reach_factor / place_margin / grasp_approach_delta: 自動計算用チューニング定数。
        - place_dist / grasp_approach_dist / stand_dist: 配置・把持接近・giving 立ち位置。None なら reach_radius から派生。
        - first_distance, band_width, margin, min_dist: 親子配置・帯域等の幾何条件（margin はアリーナ余白）。
        - foward_by_motor: True でモーターによる前進を有効。
        - head_pitch_limit_deg: 頭ピッチの上下制限角度[deg]。
        - infant_waist_yaw_limit_deg: 幼児用腰ヨーの上下制限角度[deg]。
        - arena_half_extents, wall_thickness, wall_height: アリーナ（内側）の形状。
        - return_views: True で info["views"] に親/幼児のRGBを格納。
        - view_*, ssaa: カメラの画角・近遠・解像度。SSAA>1 で内部高解像度から縮小。
        - small_motion_eps, small_waist_eps: 関節の微小揺れをスナップする閾値。
        - action_repeat: 常に 1。1回の env.step で行動を何回適用するか（≠1 は拒否）。
          設計前提: 1 env.step ≒ 人間の約 0.1s。時間は step で数え、物理 timestep から導出しない。
        - torso_step / head_*_step / shoulder_*_step 等（幼児）: |action|=1 のときの 1step 変位。
          幼児は親より小さめ（発達的に遅い動きの初期値）。
        - parent_*_step: 親（成人）の 1step 変位。成人の約 0.1s 相当を正とし、現行デフォルトを初期値とする。
        - infant_action_names: 幼児が制御できる行動名のリスト。指定された場合、幼児の行動空間はこのリストの順序・次元に制限される。
          有効な名前: "waist_yaw", "head_pitch", "forward", "voice", "arm_pitch", "arm_yaw", "grasp"
        - infant_obs_keys: 幼児観測に残すキーのリスト。None なら全キー。指定時は observation_space / _get_obs をそのキーだけにする。
        - layout_freeze: 配置の固定モード。"none"（毎回ずらす）/ "full"（位置・対象を固定）/ "rotate_target"（位置は固定、対象だけ順に変える）。
        - random_parent_pos: Trueの場合、親の初期位置をアリーナ内でランダムにする（幼児の方を向く）。
        - layout_grid: 配置レイアウトを指定する二次元リスト。各セルは文字列で、
          'x' でオブジェクト配置可能、'p' で親配置可能、'i' で幼児配置可能を示す。
          複数の文字を組み合わせ可能（例: 'xp' でオブジェクトと親の両方配置可能）。
          None の場合は arena_rows/arena_cols を使用した従来の挙動。
          例: [['x','p','x'],['x','x','x'],['x','i','x']]
          row=0 と row=max は固定オブジェクト用に予約され、可動オブジェクトの配置候補から除外される。
        - fixed_placements: 固定オブジェクトのワールド座標配置リスト（grid のみ）。
          各要素は {"class": "Board", "x": 0.0, "y": 1.95, "z": 0.5, "yaw": 0.0} 形式。
          Board URDF はデフォルト壁（厚 0.02 m, 高 1.0 m, 南北スパン 2*xh）と同寸。
          y は row=0 または row=max の帯域内である必要がある（Y 軸方向の両端のみ）。
        - placement_method: 初期配置方法の選択。
          "grid": グリッド配置（従来の方式、layout_gridを使用）
          "random": アリーナ内完全ランダム配置（layout_gridは無視）
        注意:
        - 行動の Δyaw は首ではなく「腰ヨー」に適用し、首yawは常に0へ維持する方針。
          これにより、視点のヨーは胴体回転に一元化され、二重回転/解釈の混乱を避ける。
        """
        super().__init__()
        self.gui = gui
        self.id = id
        self.name = name
        self.num_agents = num_agents
        self.num_objects = num_objects
        self.parent_no_movement = parent_no_movement
        self.max_steps = max_steps
        if action_repeat != 1:
            raise ValueError(
                f"action_repeat must be 1 (got {action_repeat}). "
                "1 env.step ≈ 0.1s; variable action_repeat is not supported."
            )
        self.action_repeat = action_repeat
        self.voice_window_size = voice_window_size #音声を記録するウィンドウサイズ
        self.infant_action_names = infant_action_names
        self.infant_obs_keys = list(infant_obs_keys) if infant_obs_keys is not None else None
        if layout_freeze not in ("none", "full", "rotate_target"):
            raise ValueError(
                f"layout_freeze must be 'none', 'full', or 'rotate_target' (got {layout_freeze!r})"
            )
        self.layout_freeze = layout_freeze
        self._frozen_layout = None
        self._rotate_target_index = 0
        self.non_linear_action = non_linear_action
        self.easy_voice = easy_voice
        self.enable_emotion = bool(enable_emotion)
        self.eval_mode = eval_mode
        self.parent_invisible_mode = parent_invisible_mode
        self.random_parent_pos = random_parent_pos
        self.obj_feat_to_id = {}
        self.all_objects = {}  # すべてのオブジェクトを管理: body_id -> オブジェクトインスタンス
        self.active_objects = []  # 現在配置されているオブジェクトのbody_id リスト
        # 家事プロトコル用インデックス（household_schemas.TASK_SCHEMAS と連動）
        self.household_boards_by_category = {}
        self.household_items_by_category = {}
        self.household_task_candidates = []
        infant_obj_angle_range_deg = kwargs.get("infant_obj_angle_range", None)
        if infant_obj_angle_range_deg is not None and isinstance(infant_obj_angle_range_deg, tuple):
            self.infant_obj_angle_range = (math.radians(infant_obj_angle_range_deg[0]),math.radians(infant_obj_angle_range_deg[1]))
        else:
            self.infant_obj_angle_range = None
        
        # オブジェクト生成用のパラメータを保存
        self.object_counts = object_counts if object_counts is not None else {"RedCube": 3}
        
        self.is_loaded = False
        
        self.debug_grasp_bbox = kwargs.get("debug_grasp_bbox", False)
        self._grasp_bbox_ids = {}

        # 行動名とインデックスの対応
        # enable_emotion=False のとき lip / pitch を除外し、以降の index を詰める（17 → 15）。
        if easy_voice:
            if self.enable_emotion:
                self.action_map = {
                    # 胴体・頭部
                    "torso_joint": 0,           # torso_joint (base rotation)
                    "head_pan_joint": 1,            # head_pan_joint (head yaw)
                    "head_tilt_joint": 2,           # head_tilt_joint (head pitch)
                    
                    # 左腕
                    "shoulder_yaw_joint": 3,  # left_shoulder_yaw_joint
                    "shoulder_pitch_joint": 4,       # left_shoulder_pitch_joint  
                    "elbow_joint": 5,            # left_elbow_joint
                    "roll_wrist_joint": 6,            # left_wrist_joint
                    "pitch_wrist_joint": 7,      # left_pitch_wrist_joint
                    
                    # 指
                    "index_joint": 8,           # left_index_joint
                    "other_fingers": 9,           # other fingers
                    
                    # 唇
                    "lip": 10,             # lip
                    
                    # その他動作
                    "forward": 11,              # 前進動作
                    "volume": 12,               # volume
                    "pitch": 13,                # pitch

                    #音韻に関わる動作
                    "v_1": 14,
                    "v_2": 15,
                    "v_3": 16,
                }
            else:
                self.action_map = {
                    # 胴体・頭部
                    "torso_joint": 0,
                    "head_pan_joint": 1,
                    "head_tilt_joint": 2,
                    # 左腕
                    "shoulder_yaw_joint": 3,
                    "shoulder_pitch_joint": 4,
                    "elbow_joint": 5,
                    "roll_wrist_joint": 6,
                    "pitch_wrist_joint": 7,
                    # 指
                    "index_joint": 8,
                    "other_fingers": 9,
                    # その他動作（lip / pitch 無し）
                    "forward": 10,
                    "volume": 11,
                    #音韻に関わる動作
                    "v_1": 12,
                    "v_2": 13,
                    "v_3": 14,
                }
        else:
            raise NotImplementedError("vocal tract版はまだ実装されていません")
        

        # 関節 1step 幅（|action|=1 の変位）。時間は step 前提（1 env.step ≒ 0.1s）。
        # 親 = 成人の約 0.1s 相当（parent_*_step）、幼児 = 別表で小さめ。数値は現行デフォルトを初期値とする。
        # if self.use_cinematic_model:
        #     # 頭部系
        #     self.head_step = head_step if head_step is not None else 0.01
        #     self.torso_step = torso_step if torso_step is not None else 0.02
        #     # 腕系
        #     self.shoulder_step = 0.05  # 肩関節用
        #     self.elbow_step = 0.08     # 肘関節用
        #     self.wrist_step = 0.1      # 手首関節用
        #     self.finger_step = 0.2     # 指関節用
        #     self.lip_step = 0.15       # 唇関節用
        #     # その他
        #     self.move_step = move_step if move_step is not None else 0.04
        # else:
        #     # 頭部系
        #     self.head_step = head_step if head_step is not None else 0.08
        #     self.torso_step = torso_step if torso_step is not None else 0.2
        
        # 幼児の関節ステップ設定
        # enable_emotion=False のとき lip / pitch を入れない（pitch は voice_mapping にも無い）。
        if easy_voice:
            self.infant_joint_mappings = {
                # 腰
                "torso_joint": torso_step,
                # 頭
                "head_pan_joint": head_pan_step,
                "head_tilt_joint": head_tilt_step,
                # 左腕
                "shoulder_yaw_joint": shoulder_yaw_step,
                "shoulder_pitch_joint": shoulder_pitch_step,
                "elbow_joint": elbow_step,
                "roll_wrist_joint": roll_wrist_step,
                "pitch_wrist_joint": pitch_wrist_step,
                # 指
                "index_joint": index_step,
                "other_fingers": other_fingers_step,
                # その他動作
                "forward": forward_step,
                "volume": 0.5,
                # 音韻に関わる動作
                "v_1": 0.5,
                "v_2": 0.5,
                "v_3": 0.5,
            }
            if self.enable_emotion:
                self.infant_joint_mappings["lip"] = lip_step
                self.infant_joint_mappings["pitch"] = 0.5

            # 親の関節ステップ設定
            self.parent_joint_mappings = {
                "torso_joint": parent_torso_step if parent_torso_step is not None else torso_step,
                "head_pan_joint": parent_head_pan_step if parent_head_pan_step is not None else head_pan_step,
                "head_tilt_joint": parent_head_tilt_step if parent_head_tilt_step is not None else head_tilt_step,
                "shoulder_yaw_joint": parent_shoulder_yaw_step if parent_shoulder_yaw_step is not None else shoulder_yaw_step,
                "shoulder_pitch_joint": parent_shoulder_pitch_step if parent_shoulder_pitch_step is not None else shoulder_pitch_step,
                "elbow_joint": parent_elbow_step if parent_elbow_step is not None else elbow_step,
                "roll_wrist_joint": parent_roll_wrist_step if parent_roll_wrist_step is not None else roll_wrist_step,
                "pitch_wrist_joint": parent_pitch_wrist_step if parent_pitch_wrist_step is not None else pitch_wrist_step,
                "index_joint": parent_index_step if parent_index_step is not None else index_step,
                "other_fingers": parent_other_fingers_step if parent_other_fingers_step is not None else other_fingers_step,
                "forward": parent_forward_step if parent_forward_step is not None else forward_step,
                "volume": 0.5,
                # 音韻に関わる動作
                "v_1": 0.5,
                "v_2": 0.5,
                "v_3": 0.5,
            }
            if self.enable_emotion:
                self.parent_joint_mappings["lip"] = parent_lip_step if parent_lip_step is not None else lip_step
                self.parent_joint_mappings["pitch"] = 0.5
        else:
            raise NotImplementedError("vocal tract版はまだ実装されていません")

        
        self.joints_for_point = ("torso_joint","shoulder_yaw_joint","shoulder_pitch_joint","elbow_joint")
        self.joints_for_reach = ("torso_joint","shoulder_yaw_joint","shoulder_pitch_joint","elbow_joint","roll_wrist_joint","pitch_wrist_joint")

        #発話ベクトルのmapping
        # enable_emotion=False のとき pitch を除外し、音素軸の index を詰める（5 → 4、phoneme_index 2 → 1）。
        if easy_voice:
            if self.enable_emotion:
                self.voice_mapping = {
                    "volume": 0,
                    "pitch": 1,
                    # v_1, v_2, v_3 は調音的意味を持たない識別軸（値域 [-1, 1]）
                    "v_1": 2,
                    "v_2": 3,
                    "v_3": 4,
                }
                self.phoneme_index = 2
            else:
                self.voice_mapping = {
                    "volume": 0,
                    # v_1, v_2, v_3 は調音的意味を持たない識別軸（値域 [-1, 1]）
                    "v_1": 1,
                    "v_2": 2,
                    "v_3": 3,
                }
                self.phoneme_index = 1
            self.voice_dim = len(self.voice_mapping)
            from cueworld.language.phonemes import build_phoneme_mapping
            self.phoneme_mapping = build_phoneme_mapping()
        else:
            raise NotImplementedError("vocal tract版はまだ実装されていません")
        
        # 物を持つのに関する設定（距離系は _load_urdfs 後に _compute_interaction_distances で確定）
        self._auto_grasp_distance = grasp_distance is None
        self._auto_reach_radius = reach_radius is None
        self._auto_place_dist = place_dist is None
        self._auto_grasp_approach_dist = grasp_approach_dist is None
        self._auto_stand_dist = stand_dist is None
        self.grasp_margin = float(grasp_margin)
        self.reach_factor = float(reach_factor)
        self.place_margin = float(place_margin)
        self.grasp_approach_delta = float(grasp_approach_delta)
        # 明示指定時の仮値。自動の場合はロード後に上書き（未計算時のフォールバック兼用）
        self.grasp_distance = float(grasp_distance) if grasp_distance is not None else 0.25
        # margin なしの接触距離（palm_half + 掴める物の最大半幅）。ロード後に上書き。
        self.grasp_contact_distance = 0.13
        self.grasp_dir_threshold = math.radians(grasp_dir_threshold)  # 物を掴む際の手のひらが向く方向の閾値(ラジアン)
        self.grasp_object_offset = grasp_object_offset  # 掴んだ物を手からどれくらい離すか(m)
        self.reach_radius = float(reach_radius) if reach_radius is not None else 0.45
        self.place_dist = float(place_dist) if place_dist is not None else 0.35
        self.grasp_approach_dist = float(grasp_approach_dist) if grasp_approach_dist is not None else 0.57
        self.stand_dist = float(stand_dist) if stand_dist is not None else 0.92
        self.measured_floor_reach_xy = 0.0
        self._interaction_distances_ready = False

        self.margin = margin
        self.min_dist = min_dist
        self.voice_volume_threshold = float(voice_volume_threshold)
        self.fall_per_step = float(fall_per_step)
        self.foward_by_motor = foward_by_motor
        self.arena_half_extents = tuple(arena_half_extents)
        self.arena_scale = np.sqrt((arena_half_extents[0]*2)**2 + (arena_half_extents[1]*2)**2)
        self.wall_thickness = float(wall_thickness)
        self.wall_height = float(wall_height)
        self.disable_all_body_object_collision = disable_all_body_object_collision

        # 配置方法の検証
        if placement_method not in ["grid", "random", "geometric"]:
            raise ValueError(f"placement_method must be 'grid', 'random', or 'geometric', got: {placement_method}")
        self.placement_method = placement_method
        if fixed_placements is not None and self.placement_method != "grid":
            raise ValueError("fixed_placements is only supported with placement_method='grid'")

        # レイアウトグリッドの設定（グリッド配置の場合のみ）
        if self.placement_method == "grid":
            self._setup_layout_grid(layout_grid, arena_rows, arena_cols)
        else:
            # ランダム配置の場合はグリッド関連を初期化しない
            self.arena_rows = None
            self.arena_cols = None
            self.layout_grid_spec = None
            self.fixed_placements = []

        # 壁など「衝突に使う静的ボディ」をまとめる．床はここに入れない
        self.static_bodies: list[int] = []
        self.wall_body_ids: list[int] = []  # collision は残したまま visual だけ切替したい壁

        # ▼ 追加: 視覚画像関連の設定
        self.return_views = return_views
        self.output_parent_view = output_parent_view
        self.view_width = view_width
        self.view_height = view_height
        self.view_fov = view_fov
        self.view_near = view_near
        self.view_far = view_far
        self.ssaa = int(max(1, ssaa))

        # 追加: 音声状態（将来、行動で幼児発話を制御する準備）
        self.infant_voice = np.zeros(self.voice_dim, dtype=np.float32)
        self.parent_voice = np.zeros(self.voice_dim, dtype=np.float32)
        self.infant_voice_window = deque(maxlen=self.voice_window_size)
        self.parent_voice_window = deque(maxlen=self.voice_window_size)

        # 幼児の内部パラメータ
        self.sleepiness = 0.0
        self.sleepiness_steps = 100000
        self.parent_preference = float(parent_preference)
        self.enable_fatigue = bool(enable_fatigue)
        self.fatigue_high = float(fatigue_high)
        self.fatigue_low = float(fatigue_low)
        self.fatigue_recovery = float(fatigue_recovery)
        # enable_emotion=False のとき lip を fatigue から除外（10 → 9）。
        if self.enable_emotion:
            self._effective_fatigue_channels = _INFANT_FATIGUE_CHANNELS
        else:
            self._effective_fatigue_channels = tuple(
                c for c in _INFANT_FATIGUE_CHANNELS if c != "lip"
            )
        self._fatigue_dim = len(self._effective_fatigue_channels)
        self._fatigue_channel_index = {
            name: idx for idx, name in enumerate(self._effective_fatigue_channels)
        }
        self._infant_fatigue = np.zeros(self._fatigue_dim, dtype=np.float32)
        self._infant_fatigue_lock = np.zeros(self._fatigue_dim, dtype=bool)


        # ▼ しきい値保存（オブジェクト用は今回未使用だが残す）
        self.small_motion_eps = float(small_motion_eps)
        self.small_waist_eps = float(small_waist_eps)
        # 振る検出しきい値（胴体ローカル座標での変位[m]/サブステップ）
        self.shake_move_threshold = float(shake_move_threshold)

        # 振る → 音 / 効果チャネル（毎 env step 先頭でゼロクリア、shake_effect で書き込む）
        self._object_sound = np.zeros(2, dtype=np.float32)
        self._object_effect = np.zeros(2, dtype=np.float32)
        
        # 後で使うものを定義
        self.parent_collide = None  # 親の衝突オブジェクトID保存用

        self._infant_joint_obs_slots = self._resolve_infant_joint_obs_slots()
        self._infant_joint_obs_dim = self._infant_joint_obs_dimension(self._infant_joint_obs_slots)

        # オラクル枠数: object_counts 合計 + fixed_placements 件数（コンストラクタで固定）
        # observation_space 構築より前に確定する（grid の _setup_fixed_placements は後段）
        if self.placement_method == "grid":
            _fp_for_slots = fixed_placements if fixed_placements is not None else []
        else:
            _fp_for_slots = getattr(self, "fixed_placements", []) or []
        self._oracle_slot_count = int(sum(int(v) for v in self.object_counts.values())) + len(_fp_for_slots)

        # 観測空間: "image", "head", "infant_voice", "parent_voice", "object_sound", "object_effect"
        _voice_obs_low, _voice_obs_high = self._voice_obs_bounds()
        infant_obs_spaces = {
            "image": spaces.Box(
                low=0, high=255,
                shape=(self.view_height, self.view_width, 3),
                dtype=np.uint8
            ),
            "infant_voice": spaces.Box(
                low=_voice_obs_low,
                high=_voice_obs_high,
                dtype=np.float32
            ),
            "parent_voice": spaces.Box(
                low=_voice_obs_low,
                high=_voice_obs_high,
                dtype=np.float32
            ),
            "parent_position": spaces.Box(
                low=np.full(2, -1.0, dtype=np.float32),
                high=np.full(2, 1.0, dtype=np.float32),
                dtype=np.float32
            ),
            # 振る → 音チャネル（形由来の固定2次元ベクトル, 無音時 [0,0]）
            "object_sound": spaces.Box(
                low=np.full(2, -1.0, dtype=np.float32),
                high=np.full(2, 1.0, dtype=np.float32),
                dtype=np.float32
            ),
            # 振る → 効果チャネル（色由来の固定2次元ベクトル, 無効果時 [0,0]）
            "object_effect": spaces.Box(
                low=np.full(2, -1.0, dtype=np.float32),
                high=np.full(2, 1.0, dtype=np.float32),
                dtype=np.float32
            ),
            "sleepiness": spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32
            ),
            "joint_angles": spaces.Box(
                low=np.full(self._infant_joint_obs_dim, -1.0, dtype=np.float32),
                high=np.full(self._infant_joint_obs_dim, 1.0, dtype=np.float32),
                dtype=np.float32
            ),
            "tactile": spaces.Box(
                low=np.full(2, -1.0, dtype=np.float32),
                high=np.full(2, 1.0, dtype=np.float32),
                dtype=np.float32
            ),
            "holding": spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32
            ),
            "fatigue": spaces.Box(
                low=np.zeros(self._fatigue_dim, dtype=np.float32),
                high=np.ones(self._fatigue_dim, dtype=np.float32),
                dtype=np.float32
            ),
            "fatigue_lock": spaces.Box(
                low=np.zeros(self._fatigue_dim, dtype=np.float32),
                high=np.ones(self._fatigue_dim, dtype=np.float32),
                dtype=np.float32
            ),
            "is_first": spaces.Discrete(2),
            "is_terminal": spaces.Discrete(2)
        }
        # オラクル観測は infant_obs_keys で要求されたときだけ空間に入れる（本番既定には出さない）
        _oracle_obs_spaces = {
            "oracle_objects": spaces.Box(
                low=-2.0, high=np.inf,
                shape=(self._oracle_slot_count, 7),
                dtype=np.float32,
            ),
            "oracle_parent": spaces.Box(
                low=-2.0, high=np.inf,
                shape=(6,),
                dtype=np.float32,
            ),
            "oracle_cues": spaces.Box(
                low=-2.0, high=np.inf,
                shape=(7,),
                dtype=np.float32,
            ),
        }
        if self.infant_obs_keys is not None:
            for _ok, _osp in _oracle_obs_spaces.items():
                if _ok in self.infant_obs_keys:
                    infant_obs_spaces[_ok] = _osp
            unknown = [k for k in self.infant_obs_keys if k not in infant_obs_spaces]
            if unknown:
                raise ValueError(f"Unknown infant_obs_keys: {unknown}")
            infant_obs_spaces = {
                k: infant_obs_spaces[k] for k in self.infant_obs_keys
            }
        self.observation_space = spaces.Tuple((
            spaces.Dict(infant_obs_spaces),
            spaces.Box(low=0, high=0, shape=(0,), dtype=np.uint8),
        ))
        
        # 行動空間: Tuple([幼児7次元, 親7次元])
        # forward は [-1,1] だが、値が <=0 の場合は移動しない（後退禁止）
        # grasp は [-1,1] をそのまま _apply_grasp_command の value に渡す（>0:掴む, <0:離す, 0:何もしない）
        self.action_num = len(self.action_map)
        # 基本の行動空間定義
        low_full = np.array([-1.0]*self.action_num, dtype=np.float32)
        high_full = np.array([1.0]*self.action_num, dtype=np.float32)

        # 幼児の行動空間設定
        if self.infant_action_names is not None:
            self.infant_action_indices = []
            for name in self.infant_action_names:
                if name not in self.action_map:
                    raise ValueError(f"Unknown action name: {name}")
                self.infant_action_indices.append(self.action_map[name])
            
            self.infant_action_space = spaces.Box(
                low=low_full[self.infant_action_indices],
                high=high_full[self.infant_action_indices],
                dtype=np.float32
            )
        else:
            self.infant_action_indices = list(range(self.action_num))
            self.infant_action_space = spaces.Box(low=low_full, high=high_full, dtype=np.float32)

        # 親は常にフル次元
        self.parent_action_space = spaces.Box(low=low_full, high=high_full, dtype=np.float32)

        self.action_space = spaces.Tuple((self.infant_action_space, self.parent_action_space))

        
        # セル中心リスト作成（グリッド配置の場合のみ）
        if self.placement_method == "grid":
            self._initialize_cells()
            self._setup_fixed_placements(fixed_placements)

        # PyBullet 接続・初期設定
        self.client_id = None
        self.egl_ok, self.plugin_id = self._connect()
        
        self.debug_info = debug_info
        self.reverse_mode = reverse_mode
        self.reset(seed=seed)
        

    def _setup_layout_grid(self, layout_grid, default_rows: int, default_cols: int):
        """レイアウトグリッドの検証と行列数の設定を行う。
        
        引数:
            layout_grid: ユーザー指定のレイアウトグリッド（None可）
            default_rows: グリッド未指定時の行数
            default_cols: グリッド未指定時の列数
        """
        if layout_grid is None:
            # 従来の挙動：指定された行・列数を使用
            self.arena_rows = default_rows
            self.arena_cols = default_cols
            self.layout_grid_spec = None
            return
        
        # グリッドの検証
        if not isinstance(layout_grid, (list, tuple)) or len(layout_grid) == 0:
            raise ValueError("layout_grid は空でない二次元リストで指定してください。")
        
        rows = len(layout_grid)
        cols = len(layout_grid[0])
        
        # 長方形であることを確認
        for i, row in enumerate(layout_grid):
            if not isinstance(row, (list, tuple)):
                raise ValueError(f"layout_grid の行 {i} がリストまたはタプルではありません。")
            if len(row) != cols:
                raise ValueError(f"layout_grid は長方形である必要があります（行 {i} の列数が不一致）。")
        
        # グリッドを正規化（小文字の文字列に変換）
        normalized_grid = []
        for row in layout_grid:
            normalized_row = []
            for cell in row:
                if not isinstance(cell, str):
                    raise ValueError(f"layout_grid の各セルは文字列で指定してください: {cell}")
                normalized_row.append(cell.lower())
            normalized_grid.append(tuple(normalized_row))
        
        self.arena_rows = rows
        self.arena_cols = cols
        self.layout_grid_spec = tuple(normalized_grid)

    def _reserved_grid_rows(self) -> set[int]:
        """固定オブジェクト用に予約される行（Y 軸方向の両端）。"""
        return {0, self.arena_rows - 1}

    def _is_fixed_placement_y_allowed(self, y: float) -> bool:
        """固定オブジェクトの y が row=0 または row=max の帯域内か判定する。"""
        _, yh = self.arena_half_extents
        for r in self._reserved_grid_rows():
            y_lo = -yh + self.cell_h * r
            y_hi = -yh + self.cell_h * (r + 1)
            if y_lo <= y <= y_hi:
                return True
        return False

    def _setup_fixed_placements(self, fixed_placements):
        """固定オブジェクトのワールド座標配置を検証・正規化する。"""
        if fixed_placements is None:
            self.fixed_placements = []
            return

        if not isinstance(fixed_placements, (list, tuple)):
            raise ValueError("fixed_placements はリストで指定してください。")

        normalized = []
        max_row = self.arena_rows - 1
        for i, entry in enumerate(fixed_placements):
            if not isinstance(entry, dict):
                raise ValueError(f"fixed_placements[{i}] は辞書で指定してください。")

            cls_name = entry.get("class") or entry.get("cls")
            if not cls_name or not isinstance(cls_name, str):
                raise ValueError(f"fixed_placements[{i}] に 'class' キーが必要です。")
            if "x" not in entry or "y" not in entry:
                raise ValueError(f"fixed_placements[{i}] に 'x', 'y' キーが必要です。")

            x = float(entry["x"])
            y = float(entry["y"])
            z = float(entry.get("z", DEFAULT_FIXED_PLACEMENT_Z))
            yaw = float(entry.get("yaw", 0.0))

            if not self._is_fixed_placement_y_allowed(y):
                raise ValueError(
                    f"fixed_placements[{i}]: y={y} は row=0 または row={max_row} "
                    f"の帯域外です。固定オブジェクトは Y 軸方向の両端行にのみ配置できます。"
                )

            normalized.append({
                "class": cls_name,
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw,
            })

        self.fixed_placements = normalized

    def _object_cell_candidates_without_reserved_rows(self, candidates: list[int]) -> list[int]:
        """row=0 / row=max を除いたオブジェクト配置候補を返す。"""
        reserved = self._reserved_grid_rows()
        return [
            idx for idx in candidates
            if (idx // self.arena_cols) not in reserved
        ]

    def _initialize_cells(self):
        """セル座標リストと配置候補インデックスを生成する。"""
        xh, yh = self.arena_half_extents
        self.cell_w = (xh * 2) / self.arena_cols
        self.cell_h = (yh * 2) / self.arena_rows
        
        # 全セルの中心座標を計算
        self.cells = []
        for r in range(self.arena_rows):
            for c in range(self.arena_cols):
                cx = -xh + self.cell_w * (c + 0.5)
                cy = -yh + self.cell_h * (r + 0.5)
                self.cells.append((cx, cy))
        
        if self.layout_grid_spec is None:
            # 従来の挙動：固定パターン
            mid_row = self.arena_rows // 2
            self.infant_cell_candidates = [mid_row * self.arena_cols + 0]
            self.default_infant_cell_idx = self.infant_cell_candidates[0]
            
            if self.random_parent_pos:
                self.parent_cell_candidates = [row * self.arena_cols + (self.arena_cols - 1) 
                                               for row in range(self.arena_rows)]
            else:
                self.parent_cell_candidates = [mid_row * self.arena_cols + (self.arena_cols - 1)]
            self.default_parent_cell_idx = mid_row * self.arena_cols + (self.arena_cols - 1)
            
            self.object_cell_candidates = [
                self.arena_cols * row + col
                for row in range(1, self.arena_rows - 1)
                for col in range(1, self.arena_cols - 1)
            ]
        else:
            # レイアウトグリッドから候補を抽出
            parent_cands = []
            infant_cands = []
            obj_cands = []

            for r in range(self.arena_rows):
                for c in range(self.arena_cols):
                    idx = r * self.arena_cols + c
                    cell = self.layout_grid_spec[r][c]

                    if cell == '_' or cell == '':
                        continue
                    if not all(char in ['p', 'i', 'x', '_'] for char in cell):
                        raise ValueError(
                            f"layout_grid[{r}][{c}]='{cell}' は未対応です。"
                            f"固定オブジェクトは fixed_placements で座標指定してください。"
                        )
                    if 'p' in cell:
                        parent_cands.append(idx)
                    if 'i' in cell:
                        infant_cands.append(idx)
                    if 'x' in cell:
                        obj_cands.append(idx)

            obj_cands = self._object_cell_candidates_without_reserved_rows(obj_cands)
            # 必須チェック
            if not infant_cands:
                raise ValueError("layout_grid に幼児配置候補 'i' が少なくとも1つ必要です。")
            if not parent_cands and self.num_agents == 2:
                raise ValueError("layout_grid に親配置候補 'p' が少なくとも1つ必要です。")
            
            self.infant_cell_candidates = infant_cands
            self.parent_cell_candidates = parent_cands
            self.object_cell_candidates = obj_cands
            
            # デフォルト位置（リストの先頭）
            self.default_infant_cell_idx = infant_cands[0]
            if self.num_agents == 2:
                self.default_parent_cell_idx = parent_cands[0]

    # ---------- PyBullet セットアップ ----------
    def _apply_physics_params(self):
        """resetSimulation() 後に毎回適用すべき物理設定。
        - 物体を「押せる」挙動が安定するよう、転がり/スピン摩擦も弱く付与。
        """
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.setTimeStep(1/240, physicsClientId=self.client_id)

    def _connect(self):
        """PyBulletへ接続し、描画と物理の初期設定を行う。
        - ヘッドレス時はデバッグ可視化機能をOFF。
        - EGL プラグインが見つかれば ER_BULLET_HARDWARE_OPENGL を使い高速描画。
        """
        self.client_id = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        if not self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.client_id)
        egl_ok, plugin_id = self.try_load_egl_plugin()
        print("EGL enabled:", egl_ok, "plugin_id:", plugin_id)
        # 重力・タイムステップなど
        self._apply_physics_params()
        return egl_ok, plugin_id
    
    def try_load_egl_plugin(self) -> tuple[bool, int]:
        """EGL プラグインのロードを試行。
        - 成功: (True, plugin_id>=0) を返す
        - 失敗: (False, -1) を返す（TinyRendererにフォールバックされる）
        """
        try:
            egl = pkgutil.get_loader("eglRenderer")
            if egl is None:
                return False, -1
            pid = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin", physicsClientId=self.client_id)
            return (pid >= 0), pid
        except Exception:
            return False, -1

    def close(self):
        """環境のクリーンアップ。
        - EGL プラグインをアンロード
        - PyBullet から切断し、クライアントIDを無効化
        """
        if self.plugin_id is not None and self.plugin_id >= 0:
            try:
                p.unloadPlugin(self.plugin_id, physicsClientId=self.client_id)
            except Exception:
                pass
        """PyBullet から切断してクライアントをクリーンアップ。"""
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None

    ### エージェント・オブジェクト生成ヘルパー ###
    def _create_agent(self,name: str, base_pos, base_yaw=0.0, scale=1.0):
        """エージェント（親/幼児）のボディをロードしてメタ情報を返すヘルパー。

        scale: PyBullet loadURDF の globalScaling。各 link の比率を保ったまま全体を一様スケールする。
        """
        if not self.reverse_mode:
            if name == "parent":
                urdf_file = "cue_robot_parent.urdf"
            else:
                urdf_file = "cue_robot_infant.urdf"
        else:
            if name == "parent":
                urdf_file = "cue_robot_infant.urdf"
            else:
                urdf_file = "cue_robot_parent.urdf"
            
        body = p.loadURDF(os.path.join(PACKAGE_DIR, urdf_file),
                            basePosition=base_pos,
                            baseOrientation=p.getQuaternionFromEuler([0, 0, base_yaw]),
                            useFixedBase=True,
                            globalScaling=scale,
                            flags=p.URDF_MAINTAIN_LINK_ORDER,
                            physicsClientId=self.client_id)
        
        # 軽めの減衰 #TODO: これキネマティックモードでいる？
        # for j in range(p.getNumJoints(body)):
        #     p.changeDynamics(body, j, linearDamping=0.04, angularDamping=0.04)

        # 便利: 関節名/リンク名 → index
        name_to_joint = {p.getJointInfo(body, j, physicsClientId=self.client_id)[1].decode(): j for j in range(p.getNumJoints(body, physicsClientId=self.client_id))}
        name_to_link  = {p.getBodyInfo(body, physicsClientId=self.client_id)[0].decode(): -1} #この行だとbase linkのみ
        #全リンク名をname_to_linkに追加
        for j in range(p.getNumJoints(body, physicsClientId=self.client_id)):
            lname = p.getJointInfo(body, j, physicsClientId=self.client_id)[12].decode()
            name_to_link[lname] = j
        
        visual_shape_data = p.getVisualShapeData(body, physicsClientId=self.client_id)
        eyes_link = name_to_link.get("eyes_link", None)
        head_link = name_to_link.get("head_link", None)
        left_lip_link = name_to_link.get("left_lip_link", None)
        right_lip_link = name_to_link.get("right_lip_link", None)
        index_finger_link = name_to_link.get("index_finger_link", None)
        palm_link = name_to_link.get("palm_link", None)
        lower_arm_link = name_to_link.get("lower_arm_link", None)
        
        eyes_radius = 0.02
        index_finger_length = 0.06
        palm_thickness = 0.004
        palm_half = 0.06
        lower_arm_length = 0.0759375
        for item in visual_shape_data:
            if item[1] == eyes_link:
                eyes_radius = item[3][0] #目の半径
            if item[1] == index_finger_link:
                index_finger_length = item[3][2] #人差し指の長さ
            if item[1] == palm_link:
                palm_thickness = item[3][0] #手のひらの厚み
                # PyBullet box dimensions は half-extents
                palm_half = float(max(item[3][0], item[3][1], item[3][2]))
            if item[1] == lower_arm_link:
                lower_arm_length = item[3][0] # 前腕の長さ
        #関節limitの取得
        joint_limits = {}
        joints_for_point = {}
        joints_for_reach = {}
        lower_limits = []
        upper_limits = []
        joint_ranges = []
        restposes = []
        k = 0
        for j in range(p.getNumJoints(body, physicsClientId=self.client_id)):
            info = p.getJointInfo(body, j, physicsClientId=self.client_id)
            if info[2] == p.JOINT_REVOLUTE:
                joint_name = info[1].decode()
                joint_lower_limit = info[8]
                joint_upper_limit = info[9]
                joint_limits[joint_name] = (joint_lower_limit, joint_upper_limit)
                lower_limits.append(joint_lower_limit)
                upper_limits.append(joint_upper_limit)
                joint_ranges.append(joint_upper_limit - joint_lower_limit)
                restposes.append(0.0)  # 仮の初期姿勢値
                if joint_name in self.joints_for_point:
                    joints_for_point[joint_name] = k
                if joint_name in self.joints_for_reach:
                    joints_for_reach[joint_name] = k
                k+=1
        # J1: 把持中の視線クレジット振替対象（肩・顔・胴は除外）
        _hold_gaze_link_names = (
            "upper_arm_link",
            "lower_arm_link",
            "elbow_link",
            "roll_wrist_link",
            "pitch_wrist_link",
            "hand_link",
            "palm_link",
            "hand_back_link",
            "index_finger_link",
            "thumb_link",
            "little_finger_link",
            "grasp_link",
            "point_link",
        )
        hold_gaze_links = tuple(
            lid
            for lname in _hold_gaze_link_names
            if (lid := name_to_link.get(lname)) is not None
        )
        agent_meta = { #TODO:後でここ見直す．関節のidと関節の状態がある
            "name": name,
            "body": body,
            "torso_joint": name_to_joint.get("torso_joint", None),
            "torso_link": name_to_link.get("torso_link", None),
            "head_pan_joint": name_to_joint.get("head_pan_joint", None),
            "head_tilt_joint": name_to_joint.get("head_tilt_joint", None),
            "eyes_link": eyes_link,
            "eyes_radius": eyes_radius,
            "gaze_link": name_to_link.get("gaze_link_2", None), #現在はgaze_link_2を使用している
            "face_links": (eyes_link, head_link, left_lip_link, right_lip_link),
            "hold_gaze_links": hold_gaze_links,
            "shoulder_yaw_joint": name_to_joint.get("shoulder_yaw_joint", None),
            "shoulder_pitch_joint": name_to_joint.get("shoulder_pitch_joint", None),
            "elbow_joint": name_to_joint.get("elbow_joint", None),
            "roll_wrist_joint": name_to_joint.get("roll_wrist_joint", None),
            "pitch_wrist_joint": name_to_joint.get("pitch_wrist_joint", None),
            "lower_arm_link": name_to_link.get("lower_arm_link", None),
            "lower_arm_length": lower_arm_length/2.0,
            "hand_link": name_to_link.get("hand_link", None),
            "grasp_link": name_to_link.get("grasp_link", None),
            "palm_thickness": palm_thickness,
            "palm_half": palm_half,
            "index_joint": name_to_joint.get("index_joint", None),
            "index_link": index_finger_link,
            "index_link_size": index_finger_length/2.0,  # 人差し指の長さの半分（先端位置計算用）
            "point_link": name_to_link.get("point_link", None),
            "thumb_joint": name_to_joint.get("thumb_joint", None),
            "little_finger_joint": name_to_joint.get("little_finger_joint", None),
            "arm_links": (name_to_link.get("upper_arm_link", None),
                          name_to_link.get("lower_arm_link", None),
                          name_to_link.get("shoulder_link", None),
                          name_to_link.get("shoulder_link", None),
                          name_to_link.get("shoulder_yaw_link", None),
                          name_to_link.get("shoulder_pitch_link", None),
                          name_to_link.get("elbow_link", None),
                          name_to_link.get("roll_wrist_link", None),
                          name_to_link.get("pitch_wrist_link", None)),
            "fingers": (name_to_joint.get("index_joint", None),
                        name_to_joint.get("thumb_joint", None),
                        name_to_joint.get("little_finger_joint", None)),
            "index_finger_tip_link": name_to_link.get("index_finger_tip_link", None),
            "left_lip_joint": name_to_joint.get("left_lip_joint", None),
            "right_lip_joint": name_to_joint.get("right_lip_joint", None),
            "jaw_link": name_to_link.get("jaw_link", None),
            "jaw_joint": name_to_joint.get("jaw_joint", None),
            "previous_mouth_shape": "closed",
            "torso_joint_limits": joint_limits.get("torso_joint", (None, None)),
            "head_pan_joint_limits": joint_limits.get("head_pan_joint", (None, None)),
            "head_tilt_joint_limits": joint_limits.get("head_tilt_joint", (None, None)),
            "shoulder_yaw_joint_limits": joint_limits.get("shoulder_yaw_joint", (None, None)),
            "shoulder_pitch_joint_limits": joint_limits.get("shoulder_pitch_joint", (None, None)),
            "elbow_joint_limits": joint_limits.get("elbow_joint", (None, None)),
            "roll_wrist_joint_limits": joint_limits.get("roll_wrist_joint", (None, None)),
            "pitch_wrist_joint_limits": joint_limits.get("pitch_wrist_joint", (None, None)),
            "index_joint_limits": joint_limits.get("index_joint", (None, None)),
            "thumb_joint_limits": joint_limits.get("thumb_joint", (None, None)),
            "little_finger_joint_limits": joint_limits.get("little_finger_joint", (None, None)),
            "left_lip_joint_limits": joint_limits.get("left_lip_joint", (None, None)),
            "right_lip_joint_limits": joint_limits.get("right_lip_joint", (None, None)),
            "previous_hand_state": "open",   # "open" or "closed,other"
            "held_obj": None,
            "graspable_obj": None,
            "graspable_link": None,
            "held_obj_mass": None,
            "held_obj_prev_local": None,  # 振る検出用: 前サブステップの胴体ローカル座標でのオブジェクト位置
            "eaten_objects": [],
            "lower_limits": lower_limits,
            "upper_limits": upper_limits,
            "joint_ranges": joint_ranges,
            "restposes": restposes,
            "joints_for_point": joints_for_point,
            "joints_for_reach": joints_for_reach,
        }
        return agent_meta

    def _load_urdfs(self):
        """初回のみ実行：床・壁・エージェント・オブジェクトのボディを生成する。"""
        self.static_bodies: list[int] = []
        self.wall_body_ids: list[int] = []
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        p.changeDynamics(plane_id, -1,
                        restitution=0.0,
                        lateralFriction=1.0,
                        rollingFriction=0.01,
                        spinningFriction=0.01,
                        linearDamping=0.04,
                        angularDamping=0.04,
                        physicsClientId=self.client_id)
        
        self._spawn_walls()
    
        # 親と幼児の生成（初期位置は仮）
        # num_agents=1 の場合は親を作成しない
        if self.num_agents == 2:
            self.parent = self._create_agent("parent",[+10, 0.0, 0.0], base_yaw=math.pi)
            self.parent_body = self.parent["body"]
        else:
            self.parent = None
            self.parent_body = None
            
        self.infant = self._create_agent("infant",[-10, 0.0, 0.0])
        self.infant_body = self.infant["body"]
        
        # オブジェクトのプール生成（すべての組み合わせを事前生成）
        self.all_objects = {}  # body_id -> object_instance
        # 辞書に基づいてオブジェクトを生成
        for cls_name, count in self.object_counts.items():
            try:
                module = importlib.import_module("cueworld.objects.items")
                obj_cls = getattr(module, cls_name)
            except (ImportError, AttributeError):
                try:
                    module = importlib.import_module("objects.items")
                    obj_cls = getattr(module, cls_name)
                except (ImportError, AttributeError) as e:
                    print(f"Warning: Failed to load class {cls_name}. Error: {e}")
                    continue
                
            for _ in range(count):
                obj_instance = obj_cls()
                urdf_path = os.path.join(PACKAGE_DIR, "objects", "assets", obj_instance.urdf_file)
                body_id = p.loadURDF(
                    urdf_path,
                    basePosition=[0, 0, -100.0],
                    physicsClientId=self.client_id
                )
                
                p.changeDynamics(body_id, -1,
                    restitution=0.0,
                    lateralFriction=0.9,
                    rollingFriction=0.03,
                    spinningFriction=0.03,
                    linearDamping=0.05,
                    angularDamping=0.2,
                    contactProcessingThreshold=0.0,
                    ccdSweptSphereRadius=0.02,
                    physicsClientId=self.client_id
                )
                
                obj_instance.id = body_id

                obj_instance.lid_link_id = None
                obj_instance.lid_joint = None
                for j in range(p.getNumJoints(body_id, physicsClientId=self.client_id)):
                    joint_info = p.getJointInfo(body_id, j, physicsClientId=self.client_id)
                    link_name = joint_info[12].decode('utf-8')
                    if link_name in ["lid_link", "door_link"]:
                        obj_instance.lid_link_id = j
                        obj_instance.lid_joint = j
                        break
                
                # if self.disable_all_body_object_collision:
                #     self._disable_agent_object_collisions_all(body_id)
                
                self.all_objects[body_id] = obj_instance

        # 固定オブジェクト（ワールド座標指定）の生成
        for spec in self.fixed_placements:
            cls_name = spec["class"]
            normalized_cls_name = "".join(word.capitalize() for word in cls_name.split('_'))
            try:
                try:
                    module = importlib.import_module("cueworld.objects.items")
                    obj_cls = getattr(module, normalized_cls_name)
                except (ImportError, AttributeError):
                    module = importlib.import_module("objects.items")
                    obj_cls = getattr(module, normalized_cls_name)

                obj_instance = obj_cls()
                urdf_path = os.path.join(PACKAGE_DIR, "objects", "assets", obj_instance.urdf_file)
                body_id = p.loadURDF(
                    urdf_path,
                    basePosition=[0, 0, -100.0],
                    physicsClientId=self.client_id
                )

                p.changeDynamics(body_id, -1,
                    restitution=0.0,
                    lateralFriction=0.9,
                    rollingFriction=0.03,
                    spinningFriction=0.03,
                    linearDamping=0.05,
                    angularDamping=0.2,
                    contactProcessingThreshold=0.0,
                    ccdSweptSphereRadius=0.02,
                    physicsClientId=self.client_id
                )

                obj_instance.id = body_id
                obj_instance.lid_link_id = None
                obj_instance.lid_joint = None
                for j in range(p.getNumJoints(body_id, physicsClientId=self.client_id)):
                    joint_info = p.getJointInfo(body_id, j, physicsClientId=self.client_id)
                    link_name = joint_info[12].decode('utf-8')
                    if link_name in ["lid_link", "door_link"]:
                        obj_instance.lid_link_id = j
                        obj_instance.lid_joint = j
                        break

                # if self.disable_all_body_object_collision:
                #     self._disable_agent_object_collisions_all(body_id)

                obj_instance.move_type = None
                obj_instance.initial_pos = [spec["x"], spec["y"], spec["z"]]
                obj_instance.initial_orn = p.getQuaternionFromEuler(
                    [0, 0, math.radians(spec["yaw"])]
                )

                self.all_objects[body_id] = obj_instance
            except (ImportError, AttributeError) as e:
                print(f"Warning: Failed to load fixed object class {normalized_cls_name}. Error: {e}")
            except Exception as e:
                print(f"Warning: Failed to load fixed object {cls_name}. Error: {e}")

        # 初期状態では全オブジェクトを画面外に配置
        for i, body_id in enumerate(self.all_objects.keys()):
            p.resetBasePositionAndOrientation(body_id, [0, 0, -10.0 - i], [0,0,0,1], physicsClientId=self.client_id)

        self._index_household_boards()

    # 床置きオブジェクトを掴む想定の到達計測姿勢
    # コンテナで候補比較: palm z を下げつつ XY を確保（現行腕では z≈0.33 が下限付近）
    # shoulder_pitch: 腕を下げる / elbow: やや伸ばす / wrist: 掌向き調整
    FLOOR_REACH_JOINTS = {
        "shoulder_yaw_joint": 0.0,
        "shoulder_pitch_joint": -1.5,
        "elbow_joint": -0.2,
        "roll_wrist_joint": 0.0,
        "pitch_wrist_joint": 0.3,
    }

    def _palm_half(self, agent) -> float:
        """palm collision/visual の最大半幅[m]。"""
        return float(agent.get("palm_half", 0.06))

    def _max_object_half(self) -> float:
        """掴める物体プール（またはプリミティブ表）の最大半幅[m]。"""
        halves = []
        for obj in getattr(self, "all_objects", {}).values():
            if not getattr(obj, "is_graspable", False):
                continue
            bbox = getattr(obj, "bbox_half_extents", None)
            if bbox is not None:
                halves.append(float(max(bbox)))
        if halves:
            return float(max(halves))
        try:
            from cueworld.objects.items import PRIMITIVE_BBOX_HALF_EXTENTS
            return float(max(max(v) for v in PRIMITIVE_BBOX_HALF_EXTENTS.values()))
        except Exception:
            return 0.065

    def _object_rest_z(self, obj) -> float:
        """可動物の床接触高さ [m]（bbox_half_extents[2]。簡易物理の着地と同じ）。"""
        bbox = getattr(obj, "bbox_half_extents", None)
        if bbox is not None and len(bbox) >= 3:
            return float(bbox[2])
        return float(DEFAULT_FIXED_PLACEMENT_Z)

    def _voice_volume_index(self) -> int:
        return int(self.voice_mapping["volume"])

    def _is_voice_voiced(self, volume: float) -> bool:
        return float(volume) >= float(self.voice_volume_threshold)

    def _clip_voice_state(self, voice_vec: np.ndarray) -> None:
        """infant_voice / parent_voice の volume チャネルを [0, 1] にクリップ。"""
        vol_idx = self._voice_volume_index()
        voice_vec[vol_idx] = float(np.clip(voice_vec[vol_idx], 0.0, 1.0))

    def _voice_obs_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        low = np.full(self.voice_dim, -1.0, dtype=np.float32)
        high = np.full(self.voice_dim, 1.0, dtype=np.float32)
        vol_idx = self._voice_volume_index()
        low[vol_idx] = 0.0
        high[vol_idx] = 1.0
        return low, high

    def _measure_floor_reach_xy(self, agent) -> float:
        """床置き掴み姿勢で base→palm の XY 距離を一回計測する。計測後は関節を 0 に戻す。"""
        joint_keys = list(self.FLOOR_REACH_JOINTS.keys())
        saved = {}
        for key in joint_keys:
            jid = agent.get(key)
            if jid is None:
                continue
            saved[key] = p.getJointState(agent["body"], jid, physicsClientId=self.client_id)[0]
            p.resetJointState(
                agent["body"], jid, float(self.FLOOR_REACH_JOINTS[key]),
                physicsClientId=self.client_id,
            )
        # FK 更新
        p.stepSimulation(physicsClientId=self.client_id)
        base_pos, _ = p.getBasePositionAndOrientation(agent["body"], physicsClientId=self.client_id)
        palm_pos, _ = self._get_palm_pose(agent)
        xy = float(np.linalg.norm(np.asarray(palm_pos[:2], dtype=np.float32) - np.asarray(base_pos[:2], dtype=np.float32)))
        palm_z = float(palm_pos[2])
        for key, angle in saved.items():
            p.resetJointState(agent["body"], agent[key], angle, physicsClientId=self.client_id)
        self._last_floor_reach_palm_z = palm_z
        return xy

    def _compute_interaction_distances(self) -> None:
        """エージェント／物体ロード後に把持・到達・配置距離を確定する。明示指定は維持。"""
        measure_agent = self.parent if self.parent is not None else self.infant
        if measure_agent is None:
            return

        palm_half = self._palm_half(measure_agent)
        obj_half = self._max_object_half()
        self.grasp_contact_distance = float(palm_half + obj_half)
        if self._auto_grasp_distance:
            self.grasp_distance = float(self.grasp_contact_distance + self.grasp_margin)

        measured_xy = None
        if self._auto_reach_radius or self._auto_grasp_approach_dist or self._auto_stand_dist:
            measured_xy = self._measure_floor_reach_xy(measure_agent)
            self.measured_floor_reach_xy = float(measured_xy)

        if self._auto_reach_radius:
            self.reach_radius = float(measured_xy * self.reach_factor)

        if self._auto_place_dist:
            self.place_dist = float(max(0.05, self.reach_radius - self.place_margin))

        if self._auto_grasp_approach_dist:
            # 到達圏の外側で止まるが、実測腕長を超えない（超えると place/grasp 不能）
            desired = float(self.reach_radius + self.grasp_approach_delta)
            max_approach = float(self.measured_floor_reach_xy - 0.02)
            self.grasp_approach_dist = float(min(desired, max_approach))

        if self._auto_stand_dist:
            self.stand_dist = float(self.place_dist + self.grasp_approach_dist)

        self._interaction_distances_ready = True
        palm_z = getattr(self, "_last_floor_reach_palm_z", None)
        palm_z_str = f"{palm_z:.3f}" if palm_z is not None else "n/a"
        measured_str = f"{getattr(self, 'measured_floor_reach_xy', 0.0):.3f}"
        print(
            "[interaction_distances] "
            f"grasp_contact_distance={self.grasp_contact_distance:.3f} "
            f"grasp_distance={self.grasp_distance:.3f} "
            f"reach_radius={self.reach_radius:.3f} "
            f"place_dist={self.place_dist:.3f} "
            f"grasp_approach_dist={self.grasp_approach_dist:.3f} "
            f"stand_dist={self.stand_dist:.3f} "
            f"measured_floor_xy={measured_str} palm_z={palm_z_str}"
        )

    def object_in_agent_reach(self, agent: Dict[str, Any], oid: int) -> bool:
        """物体がエージェントの到達圏内か（床 XY のみ。高さは見ない）。

        §6-D2: giving は届く物では始めない。距離は agent base と物体 base の XY。
        """
        if oid is None or agent is None:
            return False
        try:
            agent_pos, _ = p.getBasePositionAndOrientation(
                agent["body"], physicsClientId=self.client_id
            )
            obj_pos, _ = p.getBasePositionAndOrientation(
                oid, physicsClientId=self.client_id
            )
        except Exception:
            return False
        dist_xy = float(
            np.linalg.norm(
                np.asarray(agent_pos, dtype=float)[:2] - np.asarray(obj_pos, dtype=float)[:2]
            )
        )
        return dist_xy <= float(self.reach_radius)

    def _index_household_boards(self) -> None:
        """_load_urdfs() 後に呼ぶ。is_platform な board を accepts_category で索引化する。"""
        self.household_boards_by_category = {}
        for body_id, obj in self.all_objects.items():
            if not getattr(obj, "is_platform", False):
                continue
            category = getattr(obj, "accepts_category", None)
            if category is not None:
                self.household_boards_by_category.setdefault(category, []).append(body_id)

    def _index_household_items(self) -> None:
        """_select_active_objects() 後に毎 reset 呼ぶ。active な item を色カテゴリ別に索引化する。"""
        from cueworld.household_schemas import HOUSEHOLD_ITEM_CATEGORIES

        self.household_items_by_category = {}
        for body_id in self.active_objects:
            obj = self.all_objects[body_id]
            if getattr(obj, "is_platform", False):
                continue
            category = getattr(obj, "color", None)
            if category in HOUSEHOLD_ITEM_CATEGORIES:
                self.household_items_by_category.setdefault(category, []).append(body_id)

    def _finalize_active_after_placement(self, placed_ids) -> None:
        """床配置できた（または既配置スキップした）ID だけを active_objects に残し、household 索引を更新する。"""
        placed_set = set(placed_ids)
        self.active_objects = [bid for bid in self.active_objects if bid in placed_set]
        self._index_household_items()
        self._build_household_task_candidates()

    def _is_eligible_household_item(self, body_id: int) -> bool:
        """家事タスク候補に含められる可動 item かどうか（現状ベース）。"""
        if body_id not in self.active_objects:
            return False
        obj = self.all_objects.get(body_id)
        if obj is None or getattr(obj, "is_platform", False):
            return False
        if getattr(obj, "put_on", None) is not None:
            return False
        for board_ids in self.household_boards_by_category.values():
            for board_id in board_ids:
                board_obj = self.all_objects[board_id]
                if body_id in getattr(board_obj, "objects_on_board", []):
                    return False
        held_ids = []
        if getattr(self, "parent", None) and self.parent.get("held_obj") is not None:
            held_ids.append(self.parent["held_obj"])
        if getattr(self, "infant", None) and self.infant.get("held_obj") is not None:
            held_ids.append(self.infant["held_obj"])
        if body_id in held_ids:
            return False
        return True

    def _build_household_task_candidates(self) -> None:
        """board / item 索引から TASK_SCHEMAS に基づくタスク候補を組み立てる。"""
        from cueworld.household_schemas import TASK_SCHEMAS

        self.household_task_candidates = []
        for task_type, schema in TASK_SCHEMAS.items():
            category = schema["category"]
            board_ids = self.household_boards_by_category.get(category, [])
            if not board_ids:
                continue
            item_ids = [
                item_id
                for item_id in self.household_items_by_category.get(category, [])
                if self._is_eligible_household_item(item_id)
            ]
            if not item_ids:
                continue
            for board_id in board_ids:
                self.household_task_candidates.append({
                    "task_type": task_type,
                    "board_id": board_id,
                    "item_ids": list(item_ids),
                })

    def _select_active_objects(self):
        if not hasattr(self, '_first_reset_done'):
            self._first_reset_done = True
            fixed_objs = [bid for bid, obj in self.all_objects.items() if obj.move_type is None]
            
            # 固定家具は無条件で全てアクティブにする
            self._fixed_active_objects = fixed_objs
                
        movable_objs = [bid for bid, obj in self.all_objects.items() if obj.move_type is not None]
        num_movable_to_place = min(len(movable_objs), max(0, self.num_objects - len(self._fixed_active_objects)))
        
        if num_movable_to_place > 0:
            active_movable = self.np_random.choice(movable_objs, size=num_movable_to_place, replace=False).tolist()
        else:
            active_movable = []
            
        self.active_objects = self._fixed_active_objects + active_movable
        self._index_household_items()
        self._build_household_task_candidates()

    def _reset_object_states(self):
        for body_id in self.active_objects:
            obj = self.all_objects[body_id]
            obj.contained_in = None
            if hasattr(obj, 'contents'):
                obj.contents = []
            if hasattr(obj, 'objects_on_board'):
                obj.objects_on_board = []
            obj.put_on = None
            if hasattr(obj, 'lid_link_id') and obj.lid_link_id is not None:
                p.resetJointState(body_id, obj.lid_link_id, 0.0, physicsClientId=self.client_id)

    def _pose_of_body(self, body_id):
        pos, orn = p.getBasePositionAndOrientation(int(body_id), physicsClientId=self.client_id)
        return (tuple(pos), tuple(orn))

    def _apply_pose(self, body_id, pose):
        pos, orn = pose
        p.resetBasePositionAndOrientation(
            int(body_id), list(pos), list(orn), physicsClientId=self.client_id
        )
        p.resetBaseVelocity(
            int(body_id), [0, 0, 0], [0, 0, 0], physicsClientId=self.client_id
        )

    def _reset_agent_joints_and_grasp(self, agent):
        body = agent["body"]
        for j in range(p.getNumJoints(body, physicsClientId=self.client_id)):
            p.resetJointState(body, j, 0.0, physicsClientId=self.client_id)
        agent["previous_hand_state"] = "open"
        agent["held_obj"] = None
        agent["graspable_obj"] = None
        agent["graspable_link"] = None

    def _movable_active_ids(self):
        return [
            bid for bid in self.active_objects
            if getattr(self.all_objects.get(bid), "move_type", None) is not None
        ]

    def _save_frozen_layout(self):
        """layout_freeze 用に、最初のリセットで決まった配置を保存する。"""
        object_poses = {}
        for body_id in self.active_objects:
            object_poses[int(body_id)] = self._pose_of_body(body_id)
        layout = {
            "active_objects": list(self.active_objects),
            "infant_pose": self._pose_of_body(self.infant_body),
            "parent_pose": (
                self._pose_of_body(self.parent_body)
                if self.num_agents == 2
                else None
            ),
            "object_poses": object_poses,
            "target_obj_id": getattr(self, "target_obj_id", None),
        }
        movable = self._movable_active_ids()
        if layout["target_obj_id"] is None and movable:
            layout["target_obj_id"] = int(movable[0])
        if layout["target_obj_id"] is not None:
            self.target_obj_id = layout["target_obj_id"]
        self._frozen_layout = layout
        # rotate_target: 次回 restore から 2 番目へ進める（初回は上で 0 番）
        self._rotate_target_index = 1 if (self.layout_freeze == "rotate_target" and movable) else 0

    def _restore_frozen_layout(self):
        """保存した配置を復元し、関節だけ 0 に戻す。"""
        layout = self._frozen_layout
        self.active_objects = list(layout["active_objects"])
        # 非アクティブな可動物は床下へ退避
        active_set = set(self.active_objects)
        for i, body_id in enumerate(self.all_objects.keys()):
            if body_id not in active_set:
                p.resetBasePositionAndOrientation(
                    body_id, [0, 0, -10.0 - i], [0, 0, 0, 1], physicsClientId=self.client_id
                )
                p.resetBaseVelocity(
                    body_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client_id
                )
        self._apply_pose(self.infant_body, layout["infant_pose"])
        if self.num_agents == 2 and layout["parent_pose"] is not None:
            self._apply_pose(self.parent_body, layout["parent_pose"])
        for body_id, pose in layout["object_poses"].items():
            self._apply_pose(body_id, pose)
        self._reset_object_states()
        self._reset_agent_joints_and_grasp(self.infant)
        if self.num_agents == 2:
            self._reset_agent_joints_and_grasp(self.parent)
        self._index_household_items()
        self._build_household_task_candidates()

        movable = self._movable_active_ids()
        if self.layout_freeze == "rotate_target" and movable:
            tid = int(movable[self._rotate_target_index % len(movable)])
            self._rotate_target_index = (self._rotate_target_index + 1) % len(movable)
            self.target_obj_id = tid
        elif layout["target_obj_id"] is not None:
            self.target_obj_id = layout["target_obj_id"]

    def _reset_bodies(self):
        """エピソード毎に実行：配置方法に応じてエージェントとオブジェクトの位置・状態をリセットする。"""
        if self.layout_freeze != "none" and self._frozen_layout is not None:
            self._restore_frozen_layout()
            return
        if self.placement_method == "grid":
            self._reset_bodies_grid()
        elif self.placement_method == "random":
            self._reset_bodies_random()
        elif self.placement_method == "geometric":
            self._reset_bodies_geometric()
        if self.layout_freeze != "none":
            self._save_frozen_layout()

    def _reset_bodies_grid(self):
        """グリッド配置による初期化（従来の方式）"""
        # ヘルパー: セル中心から少しずらした位置を返す
        def get_pos_in_cell(idx):
            cx, cy = self.cells[idx]
            # 中心から ±10% 程度のランダムオフセット（十分小さなランダム距離）
            dx = self.np_random.uniform(-self.cell_w * 0.1, self.cell_w * 0.1)
            dy = self.np_random.uniform(-self.cell_h * 0.1, self.cell_h * 0.1)
            return [cx + dx, cy + dy, 0.0]

        # 使用済みセルを追跡するセット
        used_cells = set()

        # 1. 幼児の配置
        infant_cell_idx = self.default_infant_cell_idx
        infant_pos = get_pos_in_cell(infant_cell_idx)
        # 幼児は +X (親方向、あるいはセンター) を向く
        p.resetBasePositionAndOrientation(self.infant_body, infant_pos, p.getQuaternionFromEuler([0, 0, 0.0]), physicsClientId=self.client_id)
        used_cells.add(infant_cell_idx)

        # 2. 親の配置（num_agents=2 の場合のみ）
        if self.num_agents == 2:
            if self.parent_invisible_mode:
                parent_pos = [0.0, 0.0, -100.0]
                p.resetBasePositionAndOrientation(self.parent_body, parent_pos, p.getQuaternionFromEuler([0, 0, 0.0]), physicsClientId=self.client_id)
            else:
                if self.random_parent_pos:
                    # 使用済みセルを除外して親の候補から選択
                    available_parent_cells = [idx for idx in self.parent_cell_candidates if idx not in used_cells]
                    if available_parent_cells:
                        parent_cell_idx = self.np_random.choice(available_parent_cells)
                    else:
                        # フォールバック: デフォルト位置を使用
                        parent_cell_idx = self.default_parent_cell_idx
                else:
                    parent_cell_idx = self.default_parent_cell_idx
                    
                parent_pos = get_pos_in_cell(parent_cell_idx)
                used_cells.add(parent_cell_idx)
                
                # 親は幼児の方を向く
                dx = infant_pos[0] - parent_pos[0]
                dy = infant_pos[1] - parent_pos[1]
                yaw = math.atan2(dy, dx)
                p.resetBasePositionAndOrientation(self.parent_body, parent_pos, p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=self.client_id)
        else:
            # num_agents=1 の場合は親のリセット処理をスキップ
            pass

        # 関節リセット
        def reset_joints(agent):
            body = agent["body"]
            # 全関節を0に（必要なら初期姿勢へ）
            for j in range(p.getNumJoints(body, physicsClientId=self.client_id)):
                p.resetJointState(body, j, 0.0, physicsClientId=self.client_id)
            
            # 保持状態クリア
            agent["previous_hand_state"] = "open"
            agent["held_obj"] = None
            agent["graspable_obj"] = None
            agent["graspable_link"] = None

        if self.num_agents == 2:
            reset_joints(self.parent)
        reset_joints(self.infant)

        # 3. オブジェクトの再配置
        self._select_active_objects()

        for i, body_id in enumerate(self.all_objects.keys()):
            obj = self.all_objects[body_id]
            if obj.move_type is None and body_id in self._fixed_active_objects and getattr(self, '_first_reset_done_grid_placed', False):
                pass
            else:
                p.resetBasePositionAndOrientation(body_id, [0, 0, -10.0 - i], [0,0,0,1], physicsClientId=self.client_id)
                p.resetBaseVelocity(body_id, [0,0,0], [0,0,0], physicsClientId=self.client_id)
        
        self._reset_object_states()
        
        available_obj_cells = [idx for idx in self.object_cell_candidates if idx not in used_cells]
        self.np_random.shuffle(available_obj_cells)
        
        cell_idx_counter = 0
        placed_ids = []
        for body_id in self.active_objects:
            obj = self.all_objects[body_id]
            if obj.move_type is None and getattr(self, '_first_reset_done_grid_placed', False):
                p.resetBaseVelocity(body_id, [0,0,0], [0,0,0], physicsClientId=self.client_id)
                placed_ids.append(body_id)
                continue
                
            pos = None
            if obj.move_type is None and hasattr(obj, 'initial_pos') and obj.initial_pos is not None:
                pos = list(obj.initial_pos)
            elif cell_idx_counter < len(available_obj_cells):
                idx = available_obj_cells[cell_idx_counter]
                pos = get_pos_in_cell(idx)
                pos[2] = self._object_rest_z(obj)
                used_cells.add(idx)
                cell_idx_counter += 1
                
            if pos is not None:
                orn = getattr(obj, "initial_orn", [0, 0, 0, 1])
                p.resetBasePositionAndOrientation(body_id, pos, orn, physicsClientId=self.client_id)
                placed_ids.append(body_id)
            p.resetBaseVelocity(body_id, [0,0,0], [0,0,0], physicsClientId=self.client_id)

        self._finalize_active_after_placement(placed_ids)
        self._first_reset_done_grid_placed = True

    def _reset_bodies_random(self):
        """ランダム配置による初期化"""
        # 1. 幼児の配置
        infant_pos = self._get_random_position_in_arena()
        p.resetBasePositionAndOrientation(self.infant_body, infant_pos, p.getQuaternionFromEuler([0, 0, 0.0]), physicsClientId=self.client_id)

        # 2. 親の配置（num_agents=2の場合のみ）
        if self.num_agents == 2:
            if self.parent_invisible_mode:
                parent_pos = [0.0, 0.0, -100.0]
                p.resetBasePositionAndOrientation(self.parent_body, parent_pos, p.getQuaternionFromEuler([0, 0, 0.0]), physicsClientId=self.client_id)
            else:
                # 幼児から十分離れた位置を探す
                for _ in range(50):
                    parent_pos = self._get_random_position_in_arena()
                    dx = parent_pos[0] - infant_pos[0]
                    dy = parent_pos[1] - infant_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist >= self.min_dist:
                        break
                
                # 親は幼児の方を向く
                yaw = math.atan2(dy, dx)
                p.resetBasePositionAndOrientation(self.parent_body, parent_pos, p.getQuaternionFromEuler([0, 0, yaw]), physicsClientId=self.client_id)

        # 3. 関節リセット
        def reset_joints(agent):
            body = agent["body"]
            # 全関節を0に（必要なら初期姿勢へ）
            for j in range(p.getNumJoints(body, physicsClientId=self.client_id)):
                p.resetJointState(body, j, 0.0, physicsClientId=self.client_id)
            
            # 保持状態クリア
            agent["previous_hand_state"] = "open"
            agent["held_obj"] = None
            agent["graspable_obj"] = None
            agent["graspable_link"] = None

        if self.num_agents == 2:
            reset_joints(self.parent)
        reset_joints(self.infant)

        # 4. オブジェクトの配置
        self._select_active_objects()

        for i, body_id in enumerate(self.all_objects.keys()):
            obj = self.all_objects[body_id]
            if obj.move_type is None and body_id in self._fixed_active_objects and getattr(self, '_first_reset_done_random_placed', False):
                pass
            else:
                p.resetBasePositionAndOrientation(body_id, [0, 0, -10.0 - i], [0,0,0,1], physicsClientId=self.client_id)
                p.resetBaseVelocity(body_id, [0,0,0], [0,0,0], physicsClientId=self.client_id)
        
        self._reset_object_states()

        # 配置済み位置を記録（重複防止）
        placed_positions = [infant_pos[:2]]
        if self.num_agents == 2 and not self.parent_invisible_mode:
            placed_positions.append(parent_pos[:2])

        placed_ids = []
        for body_id in self.active_objects:
            obj = self.all_objects[body_id]
            if obj.move_type is None and getattr(self, '_first_reset_done_random_placed', False):
                p.resetBaseVelocity(body_id, [0,0,0], [0,0,0], physicsClientId=self.client_id)
                pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
                placed_positions.append(pos[:2])
                placed_ids.append(body_id)
                continue

            pos = None
            if obj.move_type is None and hasattr(obj, 'initial_pos') and obj.initial_pos is not None:
                pos = list(obj.initial_pos)
                placed_positions.append(pos[:2])
            else:
                # 既存の配置から十分離れた位置を探す
                for _ in range(100):
                    cand_pos = self._get_random_position_in_arena()
                    
                    # 距離チェック
                    valid = True
                    for px, py in placed_positions:
                        dx = cand_pos[0] - px
                        dy = cand_pos[1] - py
                        if math.sqrt(dx*dx + dy*dy) < self.min_dist:
                            valid = False
                            break
                    
                    if valid:
                        pos = (cand_pos[0], cand_pos[1], self._object_rest_z(obj))
                        placed_positions.append(pos[:2])
                        break
            
            if pos is not None:
                orn = getattr(obj, "initial_orn", [0, 0, 0, 1])
                p.resetBasePositionAndOrientation(body_id, pos, orn, physicsClientId=self.client_id)
                placed_ids.append(body_id)
            p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client_id)

        self._finalize_active_after_placement(placed_ids)
        self._first_reset_done_random_placed = True
    
    def _reset_bodies_geometric(self):
        """
        幾何学的配置による初期化
        
        配置手順：
        1. 幼児を配置
        2. 幼児-オブジェクト角度・距離にターゲットオブジェクトを配置
        3. オブジェクト-親の角度・距離に親を配置（制約チェック付き）
           または、オブジェクト-親の制約がない場合は幼児-親の制約のみで配置
        4. 残りのオブジェクトをランダムに配置
        
        active_objectsの決め方が特殊なので，self.selrct_active_objects()の呼び出しは行わずこの中で直接書く
        """
        #-------------------配置-------------------------#
        max_parent_trials = 100  # 親配置の最大試行回数
        infant_obj_dist_range = (0.9, 1.5)
        obj_parent_dist_range = (0.9, 1.5)
        infant_parent_angle_range = (math.radians(-30),math.radians(30))
        infant_parent_dist_range = (0.9, 1.5)
        if self.infant_obj_angle_range is None:
            self.infant_obj_angle_range = (math.radians(-30),math.radians(30))
        shielding_thresh = math.radians(15.0)
        obj_parent_angle_range = (math.radians(-120),math.radians(120))
        xpi_angle_thresh = math.radians(120.0)
        
        # 1. 幼児の配置
        infant_pos = ([-2.15, 0, 0.0]) #TODO: 幼児の位置は-x側の端っこ
        p.resetBasePositionAndOrientation(
            self.infant_body, 
            infant_pos, 
            p.getQuaternionFromEuler([0, 0, 0.0]), 
            physicsClientId=self.client_id
        )
        
        # 2. 幼児-オブジェクト角度・距離をサンプリングまたは固定値を使用
        if isinstance(self.infant_obj_angle_range, tuple):
            infant_obj_angle = self.np_random.uniform(*self.infant_obj_angle_range)
        else:
            infant_obj_angle = self.infant_obj_angle_range
        
        if isinstance(infant_obj_dist_range, tuple):
            infant_obj_dist = self.np_random.uniform(*infant_obj_dist_range)
        else:
            infant_obj_dist = infant_obj_dist_range
            
        target_obj_id = None
        if self.num_objects >= 1:
            # 3. ターゲットオブジェクトの配置
            #----------------------------------
            # オブジェクトプールからランダムに選択
            all_body_ids = list(self.all_objects.keys())
            if len(all_body_ids) == 0:
                raise RuntimeError("オブジェクトプールが空です")
            
            # ターゲットオブジェクトを選択
            target_obj_id = self.np_random.choice(range(len(all_body_ids)))
            self.target_obj_id = all_body_ids[target_obj_id]
            target_obj_inst = self.all_objects[self.target_obj_id]

            obj_x = infant_pos[0] + infant_obj_dist * math.cos(infant_obj_angle)
            obj_y = infant_pos[1] + infant_obj_dist * math.sin(infant_obj_angle)
            # アリーナ内に収まるようクリップ
            obj_x, obj_y = self._clamp_xy_inside_arena(obj_x, obj_y, margin=self.margin)
            obj_z = self._object_rest_z(target_obj_inst)
            obj_pos = (obj_x, obj_y, obj_z)
            
            p.resetBasePositionAndOrientation(
                self.target_obj_id, 
                obj_pos, 
                [0, 0, 0, 1], 
                physicsClientId=self.client_id
            )
            self.active_objects = [self.target_obj_id]
            #----------------------------------
        
        # 4-6. 親の配置（制約チェック付き）
        if self.num_agents == 2 and not self.parent_invisible_mode:
            if self.parent_invisible_mode:
                parent_pos = [0.0, 0.0, -100.0]
                parent_x, parent_y = parent_pos[0], parent_pos[1]
                p.resetBasePositionAndOrientation(
                    self.parent_body, 
                    parent_pos, 
                    p.getQuaternionFromEuler([0, 0, 0.0]), 
                    physicsClientId=self.client_id)
            elif obj_parent_dist_range is None or self.num_objects == 0:
                # オブジェクトと親の位置関係の制約がないので親と幼児の位置関係で制約
                parent_placed = False
                for trial in range(max_parent_trials):
                    # 幼児-親の角度・距離をサンプリングまたは固定値を使用
                    if isinstance(infant_parent_angle_range, tuple):
                        infant_parent_angle = self.np_random.uniform(*infant_parent_angle_range)
                    else:
                        infant_parent_angle = infant_parent_angle_range
                    
                    if isinstance(infant_parent_dist_range, tuple):
                        infant_parent_dist = self.np_random.uniform(*infant_parent_dist_range)
                    else:
                        infant_parent_dist = infant_parent_dist_range
                    
                    # 親の位置を計算
                    parent_x = infant_pos[0] + infant_parent_dist * math.cos(infant_parent_angle)
                    parent_y = infant_pos[1] + infant_parent_dist * math.sin(infant_parent_angle)
                    
                    # アリーナ内に収まるかチェック
                    xh, yh = self.arena_half_extents
                    if (abs(parent_x) > xh - self.margin or 
                        abs(parent_y) > yh - self.margin):
                        continue
                    
                    parent_pos = (parent_x, parent_y, 0.0)
                    
                    # 遮蔽チェック（幼児-親とオブジェクトの関係）
                    if self.num_objects > 0 and shielding_thresh is not None:
                        angle_diff = abs(infant_parent_angle - infant_obj_angle)
                        # 角度差を[0, π]に正規化
                        if angle_diff > math.pi:
                            angle_diff = 2 * math.pi - angle_diff
                        
                        shielding_ok = (angle_diff > shielding_thresh or 
                                    infant_parent_dist > infant_obj_dist)
                        if not shielding_ok:
                            continue
                    
                    # 全ての制約を満たした
                    parent_placed = True
                    break
                
                if not parent_placed:
                    # 制約を満たす配置が見つからない場合、警告を出してデフォルト配置
                    print(f"Warning: Could not find valid parent placement after {max_parent_trials} trials. Using fallback.")
                    if isinstance(infant_parent_dist_range, tuple):
                        fallback_dist = infant_parent_dist_range[1]
                    else:
                        fallback_dist = infant_parent_dist_range
                    parent_x = infant_pos[0] + fallback_dist
                    parent_y = infant_pos[1]
                    parent_pos = (parent_x, parent_y, 0.0)
            elif obj_parent_angle_range is None:
                # まず親と幼児の位置関係で制約．オブジェクトと親の距離の制約はチェック
                parent_placed = False
                for trial in range(max_parent_trials):
                    # 幼児-親の角度・距離をサンプリングまたは固定値を使用
                    if isinstance(infant_parent_angle_range, tuple):
                        infant_parent_angle = self.np_random.uniform(*infant_parent_angle_range)
                    else:
                        infant_parent_angle = infant_parent_angle_range
                    
                    if isinstance(infant_parent_dist_range, tuple):
                        infant_parent_dist = self.np_random.uniform(*infant_parent_dist_range)
                    else:
                        infant_parent_dist = infant_parent_dist_range
                    
                    # 親の位置を計算
                    parent_x = infant_pos[0] + infant_parent_dist * math.cos(infant_parent_angle)
                    parent_y = infant_pos[1] + infant_parent_dist * math.sin(infant_parent_angle)
                    
                    # アリーナ内に収まるかチェック
                    xh, yh = self.arena_half_extents
                    if (abs(parent_x) > xh - self.margin or 
                        abs(parent_y) > yh - self.margin):
                        continue
                    
                    parent_pos = (parent_x, parent_y, 0.0)
                    
                    # オブジェクト-親の距離制約チェック
                    dx = parent_x - obj_x
                    dy = parent_y - obj_y
                    obj_parent_dist = math.sqrt(dx*dx + dy*dy)
                    
                    if isinstance(obj_parent_dist_range, tuple):
                        if not (obj_parent_dist_range[0] <= obj_parent_dist <= 
                                obj_parent_dist_range[1]):
                            continue
                    else:
                        # 固定値の場合は許容誤差を設ける（例: ±10%）
                        tolerance = obj_parent_dist_range * 0.1
                        if abs(obj_parent_dist - obj_parent_dist_range) > tolerance:
                            continue
                    
                    # 6-3. ∠XPI (X=オブジェクト, P=親, I=幼児) のチェック
                    # ベクトルPX (親→オブジェクト)
                    px_x = obj_x - parent_x
                    px_y = obj_y - parent_y
                    # ベクトルPI (親→幼児)
                    pi_x = infant_pos[0] - parent_x
                    pi_y = infant_pos[1] - parent_y
                    
                    # 内積から角度を計算
                    dot_product = px_x * pi_x + px_y * pi_y
                    norm_px = math.sqrt(px_x*px_x + px_y*px_y)
                    norm_pi = math.sqrt(pi_x*pi_x + pi_y*pi_y)
                    
                    if norm_px < 1e-6 or norm_pi < 1e-6:
                        continue
                    
                    cos_angle = np.clip(dot_product / (norm_px * norm_pi), -1.0, 1.0)
                    xpi_angle = math.acos(cos_angle)
                    
                    if xpi_angle >= self.xpi_angle_thresh:
                        continue
                    
                    # 遮蔽チェック（幼児-親とオブジェクトの関係）
                    if self.num_objects > 0 and self.shielding_thresh is not None:
                        angle_diff = abs(infant_parent_angle - infant_obj_angle)
                        # 角度差を[0, π]に正規化
                        if angle_diff > math.pi:
                            angle_diff = 2 * math.pi - angle_diff
                        
                        shielding_ok = (angle_diff > self.shielding_thresh or 
                                    infant_parent_dist > infant_obj_dist)
                        if not shielding_ok:
                            continue
                    
                    # 全ての制約を満たした
                    parent_placed = True
                    break
                
                if not parent_placed:
                    # 制約を満たす配置が見つからない場合、警告を出してデフォルト配置
                    print(f"Warning: Could not find valid parent placement after {max_parent_trials} trials. Using fallback.")
                    if isinstance(self.infant_parent_dist_range, tuple):
                        fallback_dist = self.infant_parent_dist_range[1]
                    else:
                        fallback_dist = self.infant_parent_dist_range
                    parent_x = infant_pos[0] + fallback_dist
                    parent_y = infant_pos[1]
                    parent_pos = (parent_x, parent_y, 0.0)
            else:
                parent_placed = False
                #以下のいずれかの範囲に制限．
                #infant_parent_angleをinfant_obj_angleとは正負が逆の値の範囲に制限．
                #infant_obj_dist<infant_parent_distとなる範囲に制限．
                
                for trial in range(max_parent_trials):
                    #まず，親とオブジェクトの位置関係を決定(これの範囲が一番狭いので最初にここを決定)
                    if isinstance(obj_parent_angle_range, tuple):
                        obj_parent_angle = self.np_random.uniform(*obj_parent_angle_range)
                    else:
                        obj_parent_angle = obj_parent_angle_range
                    if isinstance(obj_parent_dist_range, tuple):
                        obj_parent_dist = self.np_random.uniform(*obj_parent_dist_range)
                    else:
                        obj_parent_dist = obj_parent_dist_range
                    
                    # 5. 親の位置を計算
                    parent_x = obj_x + obj_parent_dist * math.cos(obj_parent_angle)
                    parent_y = obj_y + obj_parent_dist * math.sin(obj_parent_angle)
                    
                    parent_pos = (parent_x, parent_y, 0.0)
                    
                    # 幼児-親の角度・距離を計算
                    dx = parent_x - infant_pos[0]
                    dy = parent_y - infant_pos[1]
                    infant_parent_angle = math.atan2(dy, dx)
                    infant_parent_dist = math.sqrt(dx*dx + dy*dy)
                    
                    ip_pos_ok = False
                    if infant_obj_angle <= 0:
                        ip_pos_ok = (infant_parent_angle >= 0)
                    else:
                        ip_pos_ok = (infant_parent_angle <= 0)
                    if not ip_pos_ok:
                        if infant_obj_dist >= infant_parent_dist:
                            continue
                    # 6. 制約チェック
                    # 6-1. 幼児-親の位置が指定範囲内か
                    if isinstance(infant_parent_dist_range, tuple):
                        if not (infant_parent_dist_range[0] <= infant_parent_dist <= 
                                infant_parent_dist_range[1]):
                            continue
                    else:
                        # 固定値の場合は許容誤差を設ける（例: ±10%）
                        tolerance = infant_parent_dist_range * 0.1
                        if abs(infant_parent_dist - infant_parent_dist_range) > tolerance:
                            continue
                    
                    # 角度を[-π, π]に正規化
                    infant_parent_angle_norm = ((infant_parent_angle + math.pi) % (2 * math.pi)) - math.pi
                    if isinstance(infant_parent_angle_range, tuple):
                        if not (infant_parent_angle_range[0] <= infant_parent_angle_norm <= 
                                infant_parent_angle_range[1]):
                            continue
                    else:
                        # 固定値の場合は許容誤差を設ける（例: ±5度）
                        tolerance = math.radians(5.0)
                        if abs(infant_parent_angle_norm - infant_parent_angle_range) > tolerance:
                            continue
                    
                    # 6-2. 遮蔽チェック
                    if shielding_thresh is not None:
                        angle_diff = abs(infant_parent_angle - infant_obj_angle)
                        # 角度差を[0, π]に正規化
                        if angle_diff > math.pi:
                            angle_diff = 2 * math.pi - angle_diff
                        
                        shielding_ok = (angle_diff > shielding_thresh or 
                                    infant_parent_dist > infant_obj_dist)
                        if not shielding_ok:
                            continue
                    
                    # 6-3. ∠XPI (X=オブジェクト, P=親, I=幼児) のチェック
                    # ベクトルPX (親→オブジェクト)
                    px_x = obj_x - parent_x
                    px_y = obj_y - parent_y
                    # ベクトルPI (親→幼児)
                    pi_x = infant_pos[0] - parent_x
                    pi_y = infant_pos[1] - parent_y
                    
                    # 内積から角度を計算
                    dot_product = px_x * pi_x + px_y * pi_y
                    norm_px = math.sqrt(px_x*px_x + px_y*px_y)
                    norm_pi = math.sqrt(pi_x*pi_x + pi_y*pi_y)
                    
                    if norm_px < 1e-6 or norm_pi < 1e-6:
                        continue
                    
                    cos_angle = np.clip(dot_product / (norm_px * norm_pi), -1.0, 1.0)
                    xpi_angle = math.acos(cos_angle)
                    
                    if xpi_angle >= xpi_angle_thresh:
                        continue
                    
                    # アリーナ内に収まるかチェック
                    xh, yh = self.arena_half_extents
                    if (abs(parent_x) > xh - self.margin or 
                        abs(parent_y) > yh - self.margin):
                        continue
                    
                    # 全ての制約を満たした
                    parent_placed = True
                    break
                
                if not parent_placed:
                    # 制約を満たす配置が見つからない場合、警告を出してデフォルト配置
                    print(f"Warning: Could not find valid parent placement after {max_parent_trials} trials. Using fallback.")
                    if isinstance(infant_parent_dist_range, tuple):
                        fallback_dist = infant_parent_dist_range[1]
                    else:
                        fallback_dist = infant_parent_dist_range
                    parent_x = infant_pos[0] + fallback_dist
                    parent_y = infant_pos[1]
                    parent_pos = (parent_x, parent_y, 0.0)
        
            # 親を配置
            if not self.parent_invisible_mode:
                if self.num_objects > 0:
                    # 親の向きを計算（∠IPXの二等分線方向）
                    I = np.array([infant_pos[0], infant_pos[1]])
                    P = np.array([parent_x, parent_y])
                    X = np.array([obj_x, obj_y])
                    D = self.angle_bisector_theorem(I, P, X)
                    
                    # 親からDへ向かうベクトルを計算
                    dx = D[0] - parent_x
                    dy = D[1] - parent_y
                    parent_yaw = math.atan2(dy, dx)
                else:
                    parent_yaw = math.pi
                p.resetBasePositionAndOrientation(
                    self.parent_body, 
                    parent_pos, 
                    p.getQuaternionFromEuler([0, 0, parent_yaw]), 
                    physicsClientId=self.client_id)
        #---------------------------------------------------#
        
        # 関節リセット
        def reset_joints(agent):
            body = agent["body"]
            for j in range(p.getNumJoints(body, physicsClientId=self.client_id)):
                p.resetJointState(body, j, 0.0, physicsClientId=self.client_id)
            
            agent["previous_hand_state"] = "open"
            agent["held_obj"] = None
            agent["graspable_obj"] = None
            agent["graspable_link"] = None
        
        if self.num_agents == 2:
            reset_joints(self.parent)
        reset_joints(self.infant)
        
        #TODO: オブジェクト配置(現在の配置に合わせて)
        # 7. 残りのオブジェクトの配置
        # まず全オブジェクトを画面外に移動
        if target_obj_id is not None:
            del all_body_ids[target_obj_id]
        else:
            all_body_ids = list(self.all_objects.keys())
        for i, body_id in enumerate(all_body_ids):
            p.resetBasePositionAndOrientation(body_id, [0, 0, -10.0 - i], [0,0,0,1], physicsClientId=self.client_id)
            p.resetBaseVelocity(body_id, [0,0,0], [0,0,0], physicsClientId=self.client_id)
        
        #ターゲット以外のオブジェクトを配置
        if self.num_objects > 1:
            # ランダムに選択
            num_objects_to_place = min(self.num_objects-1, len(all_body_ids))
            self.active_objects = [self.target_obj_id] + self.np_random.choice(all_body_ids, size=num_objects_to_place, replace=False).tolist()
            
            # 配置済み位置リスト（衝突回避用）
            placed_positions = [
                (obj_x, obj_y),
            ]
            if self.num_agents == 2 and not self.parent_invisible_mode:
                placed_positions.append(parent_pos[:2])
            
            # ターゲット以外のオブジェクトを配置
            max_placement_trials = 100  # 幾何学的制約を満たす配置の最大試行回数
            fallback_trials = 100  # フォールバック（距離のみ）の最大試行回数
            
            for body_id in self.active_objects[1:]:  # ターゲット以外
                placed = False
                obj = self.all_objects[body_id]
                rest_z = self._object_rest_z(obj)
                
                # まず幾何学的制約（幼児との距離・角度）を満たす配置を試行
                for _ in range(max_placement_trials):
                    # 幼児-オブジェクト角度・距離をサンプリング
                    if isinstance(self.infant_obj_angle_range, tuple):
                        obj_angle = self.np_random.uniform(*self.infant_obj_angle_range)
                    else:
                        obj_angle = self.infant_obj_angle_range
                    
                    if isinstance(infant_obj_dist_range, tuple):
                        obj_dist = self.np_random.uniform(*infant_obj_dist_range)
                    else:
                        obj_dist = infant_obj_dist_range
                    
                    # オブジェクト位置を計算
                    pos_x = infant_pos[0] + obj_dist * math.cos(obj_angle)
                    pos_y = infant_pos[1] + obj_dist * math.sin(obj_angle)
                    
                    # アリーナ内に収まるかチェック
                    xh, yh = self.arena_half_extents
                    if (abs(pos_x) > xh - self.margin or 
                        abs(pos_y) > yh - self.margin):
                        continue
                    
                    # 既存オブジェクトとの距離チェック
                    valid = True
                    for px, py in placed_positions:
                        dx = pos_x - px
                        dy = pos_y - py
                        if math.sqrt(dx*dx + dy*dy) < self.min_dist:
                            valid = False
                            break
                    
                    if valid:
                        pos = (pos_x, pos_y, rest_z)
                        p.resetBasePositionAndOrientation(
                            body_id, 
                            pos, 
                            [0, 0, 0, 1], 
                            physicsClientId=self.client_id
                        )
                        placed_positions.append((pos_x, pos_y))
                        placed = True
                        break
                
                # 幾何学的制約を満たせなかった場合、距離制約のみでランダム配置
                if not placed:
                    for _ in range(fallback_trials):
                        pos = self._get_random_position_in_arena(margin=self.margin)
                        
                        # 既存オブジェクトとの距離チェック
                        valid = True
                        for px, py in placed_positions:
                            dx = pos[0] - px
                            dy = pos[1] - py
                            if math.sqrt(dx*dx + dy*dy) < self.min_dist:
                                valid = False
                                break
                        
                        if valid:
                            pos = (pos[0], pos[1], rest_z)
                            p.resetBasePositionAndOrientation(
                                body_id, 
                                pos, 
                                [0, 0, 0, 1], 
                                physicsClientId=self.client_id
                            )
                            placed_positions.append(pos[:2])
                            placed = True
                            break
                    
                    if not placed:
                        # それでも配置できない場合は警告を出して適当な位置に配置
                        print(f"Warning: Could not place object {body_id} satisfying all constraints. Placing at fallback position.")
                        pos = self._get_random_position_in_arena(margin=self.margin)
                        pos = (pos[0], pos[1], rest_z)
                        p.resetBasePositionAndOrientation(
                            body_id, 
                            pos, 
                            [0, 0, 0, 1], 
                            physicsClientId=self.client_id
                        )
                        placed_positions.append(pos[:2])
                
                p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client_id)
    
    def _spawn_walls(self):
        """
        アリーナ矩形の四方に壁（薄いBOX）を建てる静的ボディ。
        - mass=0 の静的剛体。反発ゼロ、適度な摩擦、軽い減衰。
        - 厚み t, 高さ h、アリーナ半径 (xh,yh) に基づいて ±X/±Y へ配置。
        
        座標系とグリッドの関係:
          - X軸: 列(col)方向 → +X が右側の列、-X が左側の列
          - Y軸: 行(row)方向 → +Y が上側の行、-Y が下側の行
         
          - グリッドインデックス: idx = row * arena_cols + col
        グリッドリストとの対応: 
            [[(0,0), (0,1), ..., (0,arena_cols-1)],
             [(1,0), (1,1), ..., (1,arena_cols-1)],
             ...,
             [(arena_rows-1,0), ..., (arena_rows-1,arena_cols-1)]]
        参考（上から見た図、中央がプレイエリア）:
                    +Y (row=0側)
                     ↑
           ┌─────────────┐
           │             │
        -X │    (0,0)    │ +X
      (col=0)          (col=max)
           │             │
           └─────────────┘
                     ↓
                    -Y (row=max側)
        """
        xh, yh = self.arena_half_extents
        t = self.wall_thickness
        h = self.wall_height
        zc = h / 2.0  # 中心高さ

        wall_col_long  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[xh, t, h/2], physicsClientId=self.client_id)
        wall_vis_long  = p.createVisualShape(   p.GEOM_BOX, halfExtents=[xh, t, h/2], rgbaColor=[0.6,0.6,0.65,1], physicsClientId=self.client_id)
        wall_col_short = p.createCollisionShape(p.GEOM_BOX, halfExtents=[t, yh, h/2], physicsClientId=self.client_id)
        wall_vis_short = p.createVisualShape(   p.GEOM_BOX, halfExtents=[t, yh, h/2], rgbaColor=[0.6,0.6,0.65,1], physicsClientId=self.client_id)

        def mk_wall(col, vis, pos):
            wid = p.createMultiBody(baseMass=0.0,
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=pos,
                                    physicsClientId=self.client_id)
            p.changeDynamics(wid, -1,
                            restitution=0.0,
                            lateralFriction=1.0,
                            rollingFriction=0.01,
                            spinningFriction=0.01,
                            linearDamping=0.05,
                            angularDamping=0.1,
                            physicsClientId=self.client_id)
            self.static_bodies.append(wid)
            self.wall_body_ids.append(wid)
            return wid

        # +Y（北）壁 / -Y（南）壁
        mk_wall(wall_col_long,  wall_vis_long,  [0.0, +yh + t, zc])
        mk_wall(wall_col_long,  wall_vis_long,  [0.0, -yh - t, zc])
        # +X（東）壁 / -X（西）壁
        mk_wall(wall_col_short, wall_vis_short, [+xh + t, 0.0, zc])
        mk_wall(wall_col_short, wall_vis_short, [-xh - t, 0.0, zc])

    def set_wall_visual_alpha(self, alpha: float) -> None:
        """壁の衝突は残したまま、visual の透明度だけを変更する。"""
        clamped_alpha = float(max(0.0, min(1.0, alpha)))
        for wall_body_id in getattr(self, "wall_body_ids", []):
            try:
                p.changeVisualShape(
                    wall_body_id,
                    -1,
                    rgbaColor=[0.6, 0.6, 0.65, clamped_alpha],
                    physicsClientId=self.client_id,
                )
            except Exception:
                continue

    def set_wall_visual_visible(self, visible: bool) -> None:
        """True で通常表示、False で透明化。"""
        self.set_wall_visual_alpha(1.0 if visible else 0.0)

    def set_platform_visual_alpha(self, alpha: float) -> None:
        """board（is_platform）の衝突は残したまま、visual の透明度だけを変更する。"""
        clamped_alpha = float(max(0.0, min(1.0, alpha)))
        all_objects = getattr(self, "all_objects", None) or {}
        for oid, obj in all_objects.items():
            if not getattr(obj, "is_platform", False):
                continue
            try:
                shapes = p.getVisualShapeData(int(oid), physicsClientId=self.client_id)
            except Exception:
                continue
            for item in shapes:
                link_index = int(item[1])
                rgba = list(item[7])
                if len(rgba) < 4:
                    continue
                rgba[3] = clamped_alpha
                try:
                    p.changeVisualShape(
                        int(oid),
                        link_index,
                        rgbaColor=rgba,
                        physicsClientId=self.client_id,
                    )
                except Exception:
                    continue

    def set_platform_visual_visible(self, visible: bool) -> None:
        """True で通常表示、False で透明化。"""
        self.set_platform_visual_alpha(1.0 if visible else 0.0)


    # def _disable_agent_object_collisions_all(self, oid):
    #     """
    #     親/幼児の base(-1) を含む全リンクと全オブジェクトの衝突を無効化。
    #     """
    #     # まだ p がインポートされている前提 (import pybullet as p)
    #     if self.num_agents == 1:
    #         agents = [self.infant]
    #     else:
    #         agents = [self.parent, self.infant]
    #     for agent in agents:
    #         body = agent["body"]
    #         num_joints = p.getNumJoints(body, physicsClientId=self.client_id)
    #         # base(-1) と 全ジョイント(0..N-1) のリストを作成
    #         link_indices = [-1] + list(range(num_joints))
    #         for link_idx in link_indices:
    #             # オブジェクト側は base(-1) のみと仮定
    #             p.setCollisionFilterPair(body, oid, link_idx, -1, enableCollision=0, physicsClientId=self.client_id)

    def reset(self, seed=None, options=None):
        """環境のリセット。
        - ワールド再生成、頭角の初期化、ターゲット物体の選択
        - 現状、親の頭はゼロ姿勢にセット
        - return_views=True の場合は info["views"] に親/幼児のRGBを格納
        """
        super().reset(seed=seed)
        
        # 接続チェックと初回ロード判定
        if self.client_id is None or not p.isConnected(self.client_id):
            self.egl_ok, self.plugin_id = self._connect()
            self.is_loaded = False
        
        if not self.is_loaded:
            p.resetSimulation(physicsClientId=self.client_id)
            self._apply_physics_params()
            self._load_urdfs()
            self.is_loaded = True
            self._compute_interaction_distances()
            # body id はロード順で不変。初回ロード時に 1 回だけ確定する。
            self.trackable_targets = self.get_all_trackable_targets()
        
        # 毎回のリセット処理（位置・状態の初期化）
        self._reset_bodies()
        
        if hasattr(self, '_grasp_bbox_ids'):
            for ids in self._grasp_bbox_ids.values():
                for bid in ids:
                    p.removeBody(bid, physicsClientId=self.client_id)
            self._grasp_bbox_ids.clear()
            
        self.step_count = 0
        self.sleepiness = 0.0
        self._reset_infant_fatigue()
        self.infant_voice.fill(0.0)
        self.infant_voice_window.clear()

        # 振る → 音 / 効果チャネルをリセット
        self._object_sound[:] = 0.0
        self._object_effect[:] = 0.0
        # エージェントの振る検出履歴をリセット
        for agent in [getattr(self, "infant", None), getattr(self, "parent", None)]:
            if agent is not None:
                agent["held_obj_prev_local"] = None
        
        infant_obs = self._get_obs(parent_voice_on=self.parent_voice, infant_voice_on=self.infant_voice)
        if self.infant_obs_keys is None or "is_first" in self.infant_obs_keys:
            infant_obs["is_first"] = True
        obs = (infant_obs, None)  # 親の観測は空
            
        self.parent_collide = None  # 親の衝突オブジェクトID保存用
        
        if self.debug_info:
            info = self.get_info()
        else:
            info = {} 
        if self.return_views:
            info["views"] = {"infant_rgb": self._get_agent_view(self.infant_body)}
            if self.num_agents == 2 and self.output_parent_view:
                info["views"]["parent_rgb"] = self._get_agent_view(self.parent_body)
        return obs, info

    def _resolve_infant_joint_obs_slots(self):
        """幼児 joint_angles の観測スロット列。infant_action_names はデバッグ用の部分列。

        enable_emotion=False のとき lip を落とす（full 13 → 12、sin/cos ラップ込み）。
        """
        if self.infant_action_names is None:
            slots = _INFANT_JOINT_OBS_SLOTS_FULL
        else:
            joint_slots = set(_INFANT_JOINT_OBS_SLOTS_FULL)
            slots = tuple(name for name in self.infant_action_names if name in joint_slots)
        if not self.enable_emotion:
            slots = tuple(s for s in slots if s != "lip")
        return slots

    def _infant_joint_obs_dimension(self, slots) -> int:
        dim = 0
        for slot in slots:
            if slot in _INFANT_JOINT_OBS_WRAPPED:
                dim += 2
            else:
                dim += 1
        return dim

    def _encode_joint_slot_obs(self, agent, slot: str):
        source = _INFANT_JOINT_OBS_SOURCE.get(slot, slot)
        angle = self.get_joint_angle(agent, source)
        if slot in _INFANT_JOINT_OBS_WRAPPED:
            return [math.sin(angle), math.cos(angle)]
        lo, hi = agent[f"{source}_limits"]
        if hi <= lo:
            return [0.0]
        norm = 2.0 * (angle - lo) / (hi - lo) - 1.0
        return [float(np.clip(norm, -1.0, 1.0))]

    def _encode_infant_joint_angles(self) -> np.ndarray:
        values = []
        for slot in self._infant_joint_obs_slots:
            values.extend(self._encode_joint_slot_obs(self.infant, slot))
        return np.array(values, dtype=np.float32)

    def _get_sleepiness_obs(self) -> np.ndarray:
        if self.sleepiness_steps <= 0:
            ratio = 0.0
        else:
            ratio = self.sleepiness / self.sleepiness_steps
        return np.array([float(np.clip(ratio, 0.0, 1.0))], dtype=np.float32)

    def _get_tactile_obs(self) -> np.ndarray:
        oid = self.infant.get("graspable_obj")
        if oid is None:
            return np.zeros(2, dtype=np.float32)
        obj = self.all_objects.get(oid)
        if obj is None:
            return np.zeros(2, dtype=np.float32)
        tactile = getattr(obj, "tactile_vec", None)
        if tactile is None:
            return np.zeros(2, dtype=np.float32)
        return np.array(tactile, dtype=np.float32)

    def _get_holding_obs(self) -> np.ndarray:
        held = 1.0 if self.infant.get("held_obj") is not None else 0.0
        return np.array([held], dtype=np.float32)

    def _reset_infant_fatigue(self) -> None:
        self._infant_fatigue.fill(0.0)
        self._infant_fatigue_lock.fill(False)

    def _joint_normalized_displacement(self, agent, joint_key: str) -> float:
        angle = self.get_joint_angle(agent, joint_key)
        lo, hi = agent[f"{joint_key}_limits"]
        scale = max(abs(float(lo)), abs(float(hi)), 1e-8)
        return float(abs(angle) / scale)

    def _fatigue_channel_displacement(self, agent, channel: str) -> float:
        if channel == "other_fingers":
            little = self._joint_normalized_displacement(agent, "little_finger_joint")
            thumb = self._joint_normalized_displacement(agent, "thumb_joint")
            return max(little, thumb)
        if channel == "lip":
            return self._joint_normalized_displacement(agent, "left_lip_joint")
        if channel == "volume":
            vol_idx = self.voice_mapping["volume"]
            return float(abs(self.infant_voice[vol_idx]))
        return self._joint_normalized_displacement(agent, channel)

    def _update_infant_fatigue(self) -> None:
        if not self.enable_fatigue or self.infant is None:
            return
        agent = self.infant
        for channel in self._effective_fatigue_channels:
            idx = self._fatigue_channel_index[channel]
            gain = _INFANT_FATIGUE_GAIN[channel]
            if not self._infant_fatigue_lock[idx]:
                disp = self._fatigue_channel_displacement(agent, channel)
                self._infant_fatigue[idx] += gain * disp
            self._infant_fatigue[idx] = max(
                0.0, self._infant_fatigue[idx] - self.fatigue_recovery
            )
            if not self._infant_fatigue_lock[idx] and self._infant_fatigue[idx] >= self.fatigue_high:
                self._infant_fatigue_lock[idx] = True
            if self._infant_fatigue_lock[idx] and self._infant_fatigue[idx] <= self.fatigue_low:
                self._infant_fatigue_lock[idx] = False

    def _is_infant_fatigue_locked(self, channel: str) -> bool:
        if not self.enable_fatigue:
            return False
        if channel not in self._fatigue_channel_index:
            return False
        return bool(self._infant_fatigue_lock[self._fatigue_channel_index[channel]])

    def _retract_infant_fatigue_channel(self, agent, channel: str, step_size: float) -> None:
        if channel == "other_fingers":
            for joint_key in ("little_finger_joint", "thumb_joint"):
                angle = self.get_joint_angle(agent, joint_key)
                if abs(angle) <= step_size:
                    self.set_joint_angle(agent, joint_key, 0.0)
                else:
                    self.set_joint_angle(agent, joint_key, angle - math.copysign(step_size, angle))
            return
        if channel == "lip":
            for joint_key, sign in (("left_lip_joint", 1.0), ("right_lip_joint", -1.0)):
                angle = self.get_joint_angle(agent, joint_key)
                if abs(angle) <= step_size:
                    self.set_joint_angle(agent, joint_key, 0.0)
                else:
                    self.set_joint_angle(agent, joint_key, angle - sign * step_size)
            return
        if channel == "volume":
            vol_idx = self.voice_mapping["volume"]
            volume = float(self.infant_voice[vol_idx])
            if abs(volume) <= step_size:
                self.infant_voice[vol_idx] = 0.0
            else:
                self.infant_voice[vol_idx] = volume - math.copysign(step_size, volume)
            return
        angle = self.get_joint_angle(agent, channel)
        if abs(angle) <= step_size:
            self.set_joint_angle(agent, channel, 0.0)
        else:
            self.set_joint_angle(agent, channel, angle - math.copysign(step_size, angle))

    def _fatigue_action_channel(self, joint_key: str) -> Optional[str]:
        if joint_key in self._effective_fatigue_channels:
            return joint_key
        if joint_key == "volume":
            return "volume"
        return None

    def _get_fatigue_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enable_fatigue:
            return (
                np.zeros(self._fatigue_dim, dtype=np.float32),
                np.zeros(self._fatigue_dim, dtype=np.float32),
            )
        high = max(self.fatigue_high, 1e-8)
        fatigue_obs = np.clip(self._infant_fatigue / high, 0.0, 1.0).astype(np.float32)
        lock_obs = self._infant_fatigue_lock.astype(np.float32)
        return fatigue_obs, lock_obs

    # 観測取得
    def _get_obs(self, parent_voice_on=0.0, infant_voice_on=0.0, is_terminal=False):
        """Dict 形式の観測を返す。
        - image: 幼児視点のRGB画像（SSAA後にダウンサンプル済み）。
        - joint_angles: 幼児の関節自己受容（full 13 次元。ラップ関節は sin/cos）。
        - infant_voice: 現在の発話強度（発話）。
        - parent_voice: 親の発話ベクトル。
        """
        # parent_voice_val = float(np.clip(parent_voice_on, -1.0, 1.0))
        # if infant_voice_on is None:
        #     infant_voice_val = float(np.clip(self.infant_voice_level, -1.0, 1.0))
        # else:
        #     infant_voice_val = float(np.clip(infant_voice_on, -1.0, 1.0))

        # # 親の相対位置計算 (幼児から見た親の位置: Parent - Infant)
        # if self.num_agents == 2:
        #     p_pos, _ = p.getBasePositionAndOrientation(self.parent_body, physicsClientId=self.client_id)
        #     i_pos, _ = p.getBasePositionAndOrientation(self.infant_body, physicsClientId=self.client_id)
        #     rel_x = float((p_pos[0] - i_pos[0])/self.arena_scale)
        #     rel_y = float((p_pos[1] - i_pos[1])/self.arena_scale)
        # else:
        #     # num_agents=1 の場合はダミー値
        #     rel_x = 0.0
        #     rel_y = 0.0

        # obs = {
        #     "image": self._get_agent_view(self.infant_body),
        #     #"head": np.array([infant_yaw, infant_pitch], dtype=np.float32),
        #     "infant_voice": np.array([infant_voice_val], dtype=np.float32),
        #     "parent_voice": np.array([parent_voice_val, rel_x, rel_y], dtype=np.float32),
        #     "is_first": False,
        #     "is_terminal": False,
        # }

        if not self._is_voice_voiced(infant_voice_on[self.voice_mapping["volume"]]):
            infant_voice = np.zeros(self.voice_dim, dtype=np.float32)
        else:
            infant_voice = np.array(infant_voice_on, dtype=np.float32)

        parent_position = np.array([0, 0], dtype=np.float32) #TODO: 初期値は何がいいか．とにかく後ろで遠くっぽいほうがいいかも？
        if self._is_voice_voiced(parent_voice_on[self.voice_mapping["volume"]]):
            parent_voice = np.array(parent_voice_on, dtype=np.float32)
            # 親の相対位置計算 (幼児から見た親の位置: Parent - Infant)
            if self.num_agents == 2:
                p_pos, _ = p.getBasePositionAndOrientation(self.parent_body, physicsClientId=self.client_id)
                i_pos, _ = p.getBasePositionAndOrientation(self.infant_body, physicsClientId=self.client_id)
                rel_x = float((p_pos[0] - i_pos[0])/self.arena_scale)
                rel_y = float((p_pos[1] - i_pos[1])/self.arena_scale)
            else:
                # num_agents=1 の場合はダミー値
                rel_x = 0.0
                rel_y = 0.0
            parent_position[0] = rel_x
            parent_position[1] = rel_y
        else:
            parent_voice = np.zeros(self.voice_dim, dtype=np.float32)
        
        
        
        fatigue_obs, fatigue_lock_obs = self._get_fatigue_obs()
        obs = {
            "image": self._get_agent_view(self.infant_body),
            #"head": np.array([infant_yaw, infant_pitch], dtype=np.float32),
            "infant_voice": np.array(infant_voice, dtype=np.float32),
            "parent_voice": np.array(parent_voice, dtype=np.float32),
            "parent_position": np.array(parent_position, dtype=np.float32),
            # 振る → 音 / 効果チャネル（振られたサブステップがあれば固定ベクトル, なければ [0,0]）
            "object_sound": self._object_sound.copy(),
            "object_effect": self._object_effect.copy(),
            "sleepiness": self._get_sleepiness_obs(),
            "joint_angles": self._encode_infant_joint_angles(),
            "tactile": self._get_tactile_obs(),
            "holding": self._get_holding_obs(),
            "fatigue": fatigue_obs,
            "fatigue_lock": fatigue_lock_obs,
            "is_first": False,
            "is_terminal": bool(is_terminal),
        }
        # オラクル観測: infant_obs_keys に要求があるときだけセグメンテーションを撮って組み立てる
        _ORACLE_KEYS = ("oracle_objects", "oracle_parent", "oracle_cues")
        if self.infant_obs_keys is not None and any(k in self.infant_obs_keys for k in _ORACLE_KEYS):
            seg, vm, pm, cam_eye, _cam_dir = self._get_segmentation(self.infant)
            if "oracle_objects" in self.infant_obs_keys:
                obs["oracle_objects"] = self._build_oracle_objects(seg, vm, pm, cam_eye)
            if "oracle_parent" in self.infant_obs_keys:
                obs["oracle_parent"] = self._build_oracle_parent(seg, vm, pm, cam_eye)
            if "oracle_cues" in self.infant_obs_keys:
                obs["oracle_cues"] = self._build_oracle_cues(seg, vm, pm, cam_eye)
        if self.infant_obs_keys is not None:
            obs = {k: obs[k] for k in self.infant_obs_keys}
        return obs

    def apply_phoneme_to_action(
        self,
        action: np.ndarray,
        phoneme_coords: np.ndarray,
        *,
        volume: float = 0.5,
    ) -> np.ndarray:
        """音素座標を easy_voice action の `v_1`,`v_2`,`v_3` に反映する。"""
        coords = np.asarray(phoneme_coords, dtype=np.float32).reshape(-1)
        if coords.size >= 1:
            action[self.action_map["v_1"]] = float(coords[0])
        if coords.size >= 2:
            action[self.action_map["v_2"]] = float(coords[1])
        if "v_3" in self.action_map and coords.size >= 3:
            action[self.action_map["v_3"]] = float(coords[2])
        action[self.action_map["volume"]] = float(np.clip(volume, 0.0, 1.0))
        return action

    # ---------- ステップ補助（共通適用） ----------
    def _apply_agent_action(self, agent, body_id: int, action_vec) -> float:
        """
        単一エージェントに対して行動ベクトルを適用し，授受に関するフラグを返す。
        cue_robot.urdf対応の22次元行動ベクトル:
          [waist_yaw, head_pan, head_tilt, left_shoulder_forward, left_shoulder_up,
           left_elbow, left_wrist, left_pitch_wrist, shoulder_yaw, shoulder_pitch,
           elbow, roll_wrist, pitch_wrist, index, thumb, little_finger,  lip, forward, voice]
        """
        # 幼児の場合のみ、補正を適用
        if body_id == self.infant_body and self.non_linear_action:
            action_vec = np.array(action_vec, dtype=np.float32)
            action_vec = action_vec ** 3

        mappings = self.infant_joint_mappings if body_id == self.infant_body else self.parent_joint_mappings
        is_infant = body_id == self.infant_body
        if is_infant and self.enable_fatigue:
            self._update_infant_fatigue()

        for joint_key, step_size in mappings.items():
            fatigue_channel = self._fatigue_action_channel(joint_key) if is_infant else None
            if is_infant and fatigue_channel is not None and self._is_infant_fatigue_locked(fatigue_channel):
                self._retract_infant_fatigue_channel(agent, fatigue_channel, step_size)
                continue
            if "lip" in joint_key:
                delta = float(np.clip(action_vec[self.action_map["lip"]], -1, 1) * step_size)
                current_angle = self.get_joint_angle(agent, "left_lip_joint")
                self.set_joint_angle(agent, "left_lip_joint", current_angle + delta)
                current_angle = self.get_joint_angle(agent, "right_lip_joint")
                self.set_joint_angle(agent, "right_lip_joint", current_angle - delta)
            elif "other_fingers" in joint_key:
                delta = float(np.clip(action_vec[self.action_map["other_fingers"]], -1, 1) * step_size)
                current_angle = self.get_joint_angle(agent, "little_finger_joint")
                self.set_joint_angle(agent, "little_finger_joint", current_angle + delta)
                current_angle = self.get_joint_angle(agent, "thumb_joint")
                self.set_joint_angle(agent, "thumb_joint", current_angle - delta)
            elif "forward" in joint_key:
                # 前進（forward）
                forward_unit = float(np.clip(action_vec[self.action_map["forward"]], -1.0, 1.0))
                distance = (forward_unit if forward_unit > 0.0 else 0.0) * step_size
                if distance > 0.0:
                    if not self.foward_by_motor:
                        self.compute_forward_dxdy(body_id, distance)
                    else:
                        raise NotImplementedError("foward_by_motor is not implemented")
            elif joint_key in self.voice_mapping:
                # 音声（voice）
                if body_id == self.infant_body:
                    if joint_key != "volume" or not self._is_infant_fatigue_locked("volume"):
                        delta = float(np.clip(action_vec[self.action_map[joint_key]], -1, 1) * step_size)
                        if self.easy_voice:
                            self.infant_voice[self.voice_mapping[joint_key]] += delta
                        else:
                            raise NotImplementedError("vocal tract版はまだ実装されていません")
                else:
                    #親は行動で直接音声を指定する形にしてる．データセットから呼び出して使うのの雛形として
                    if self.easy_voice:
                        value = float(np.clip(action_vec[self.action_map[joint_key]], -1, 1))
                        if joint_key == "volume":
                            value = float(np.clip(value, 0.0, 1.0))
                        self.parent_voice[self.voice_mapping[joint_key]] = value
                    else:
                        raise NotImplementedError("vocal tract版はまだ実装されていません")
            else:
                delta = float(np.clip(action_vec[self.action_map[joint_key]], -1, 1) * step_size)
                current_angle = self.get_joint_angle(agent, joint_key)
                self.set_joint_angle(agent, joint_key, current_angle + delta)
                
        # ループ終了後に音声をクリップしてウィンドウに追加
        if body_id == self.infant_body and self.easy_voice:
            np.clip(self.infant_voice, -1.0, 1.0, out=self.infant_voice)
            self._clip_voice_state(self.infant_voice)
            self.infant_voice_window.append(self.infant_voice.copy())
        elif body_id == self.parent_body and self.easy_voice:
            np.clip(self.parent_voice, -1.0, 1.0, out=self.parent_voice)
            self._clip_voice_state(self.parent_voice)
            self.parent_voice_window.append(self.parent_voice.copy())

        
        #手の形状，手に関する更新(接触，把持など)
        hand_shape, grip_power, _ = self._get_hand_shape(agent)
        oid, link_id = self.detect_grasp_object(agent)
        agent["graspable_obj"] = oid
        agent["graspable_link"] = link_id
        transfer_flag, contain_objs = self._update_grasp(agent, hand_shape, grip_power)
        agent["previous_hand_state"] = hand_shape

        return transfer_flag, contain_objs

    def get_all_trackable_targets(self) -> List[int]:
        """追跡対象となる可能性のある全てのIDリスト（全生成オブジェクト + 親）を返す"""
        targets = []
        if hasattr(self, "all_objects"):
            targets.extend(list(self.all_objects.keys()))
        if hasattr(self, "parent") and self.parent is not None:
            targets.append(self.parent["body"])
        return targets

    # ---------- ステップ ----------
    def step(self, action):
        """
        action は [infant_action, parent_action] の形（Tuple/List長さ2）。
        各エージェントに対して _apply_agent_action を適用し、報酬を計算。
        戻り値:
          - obs: 幼児の観測（親の観測は None）
          - reward: [幼児の報酬, 親の報酬] のリスト
          - done: エピソード終了フラグ
          - info: デバッグ情報
        """
        self.step_count += 1

        #生体パラメータの変化
        self.sleepiness += 1.0
        # 入力バリデーション
        if not isinstance(action, (list, tuple)) or len(action) != 2:
            raise ValueError("action は [infant_action, parent_action] の形（長さ2のリスト/タプル）にしてください。")
        infant_action_input, parent_action = action[0], action[1]

        # 幼児アクションの展開（制限モードの場合）
        if self.infant_action_names is not None:
            infant_action = np.zeros(self.action_num, dtype=np.float32)
            # 入力ベクトルを指定インデックスへマッピング
            for i, idx in enumerate(self.infant_action_indices):
                infant_action[idx] = float(infant_action_input[i])
        else:
            infant_action = infant_action_input

        total_r_infant = 0.0
        total_r_parent = 0.0

        # 振る → 音 / 効果チャネルを毎 env step 先頭でゼロクリア
        self._object_sound[:] = 0.0
        self._object_effect[:] = 0.0

        # action_repeat 回数分シミュレーションを進める
        for _ in range(self.action_repeat):
            # 各エージェントに適用
            transfer_flag_infant, contain_objs_infant = self._apply_agent_action(self.infant, self.infant_body, infant_action)
            if self.num_agents == 2 and not self.parent_no_movement:
                transfer_flag_parent, contain_objs_parent = self._apply_agent_action(self.parent, self.parent_body, parent_action)
                if transfer_flag_infant and transfer_flag_parent:
                    self._update_handover()
                else:
                    if contain_objs_infant:
                        container = self.all_objects[contain_objs_infant[0]]
                        obj = self.all_objects[contain_objs_infant[1]]
                        self.contain_effect(container, obj)
                    if contain_objs_parent:
                        container = self.all_objects[contain_objs_parent[0]]
                        obj = self.all_objects[contain_objs_parent[1]]
                        self.contain_effect(container, obj)
            else:
                transfer_flag_parent = False
        
        #抱っこの判定
        # self._update_holding(agent)

        # 簡易的な物理を適用
        self._apply_simple_physics()

        #睡眠
        done = self.sleepiness >= self.sleepiness_steps

        if self.debug_info:
            info = self.get_info(action)
        else:
            info = {}
        if self.return_views:
            info["views"] = {"infant_rgb": self._get_agent_view(self.infant_body)}
            if self.num_agents == 2 and self.output_parent_view:
                info["views"]["parent_rgb"] = self._get_agent_view(self.parent_body)

        # 観測（幼児中心）
        obs = (self._get_obs(
            parent_voice_on=self.parent_voice,
            infant_voice_on=self.infant_voice,
            is_terminal=done,
            ), None)  # 親の観測は未実装

        # self._clamp_body_orientation(self.infant_body)
        # if self.num_agents == 2:
        #     self._clamp_body_orientation(self.parent_body)

        # 報酬は [幼児, 親] (累積値)
        reward = [float(total_r_infant), float(total_r_parent)]
        

        return obs, reward, done, False, info

    def get_info(self, action=None):
        """action=[infant_action, parent_action] を解析し，各次元の値と現在角度を返す。
        - raw: 入力そのもの
        - scaled: 物理量へスケールした値（m, rad, [0..1]）
        - 返却はデバッグ簡易用（幼児側の主要値を格納）
        """
        info = {
                "step_count": self.step_count,
        }
        
        # pan = self.get_joint_angle(self.parent, "head_pan_joint")
        # if abs(pan) > 1.56:
        #     breakpoint()
        return info

    # def _clamp_body_orientation(self, body):
    #     """胴体の角速度もゼロ化。
    #     - キネマティク制御と物理安定性のための措置
    #     """
    #     lin, ang = p.getBaseVelocity(body, physicsClientId=self.client_id)

    #     # 角速度を強制ゼロ
    #     if lin is None:
    #         lin = [0, 0, 0]
    #     p.resetBaseVelocity(body, linearVelocity=lin, angularVelocity=[0, 0, 0], physicsClientId=self.client_id)
    
    def _clamp_xy_inside_arena(self, x, y, margin=1e-3):
        """アリーナ内側に (x,y) をクリップ。
        - 壁厚み + margin を考慮して、確実に内側へ収める。
        """
        xh, yh = self.arena_half_extents
        # 壁の内側に少しだけ余白（margin）を持たせる
        x = float(np.clip(x, -xh + margin, +xh - margin))
        y = float(np.clip(y, -yh + margin, +yh - margin))
        return x, y
    
    def _apply_deadzone(self, value: float, threshold: float = 0.25) -> float:
        """指定範囲内を0にし、境界で連続になるよう再スケーリングする。"""
        if abs(value) <= threshold:
            return 0.0
        if value > 0:
            return (value - threshold) / (1.0 - threshold)
        else:
            return (value + threshold) / (1.0 - threshold)
    
    def _set_base_xy(self, body, x, y, z_fixed=None):
        """胴体(base)のXYのみをキネマティクに設定（姿勢とZは維持/固定）。"""
        pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=self.client_id)
        z = pos[2] if z_fixed is None else z_fixed
        p.resetBasePositionAndOrientation(body, [float(x), float(y), float(z)], orn, physicsClientId=self.client_id)
    
    def _try_kinematic_slide(self, body, dx, dy):
        """
        キネマティク移動（スライド）。優先順位:
         
          1) 合成 (dx,dy)
          2) Xのみ (dx,0)
          3) Yのみ (0,dy)
        いずれも貫通する場合は据え置き。
        戻り値: 実際に適用された (Δx, Δy)。
        #TODO: エージェントの胴体が重ならないように
        """
        pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=self.client_id)
        x, y = pos[0], pos[1]
        nx, ny = self._clamp_xy_inside_arena(x+dx, y+dy, margin=self.wall_thickness + 1e-3)
        self._set_base_xy(body, nx, ny, z_fixed=pos[2])
        return nx, ny
        # 候補 (合成, Xのみ, Yのみ) をアリーナ内にクリップして評価
        
        # cands = []
        # for nx, ny in [(x+dx, y+dy), (x+dx, y), (x, y+dy)]:
        #     nx, ny = self._clamp_xy_inside_arena(nx, ny, margin=self.wall_thickness + 1e-3)
        #     cands.append((nx, ny))

        # for nx, ny in cands:
        #     if not self._collides_after_move(body, (nx, ny)):
        #         self._set_base_xy(body, nx, ny, z_fixed=pos[2])
        #         return (nx - x, ny - y)
        # return (0.0, 0.0)

    def compute_forward_dxdy(self, body_id: int, distance: float) -> tuple[float, float]:
        """「今向いている方向（頭+XのXY投影）」に distance だけ進むときの (dx,dy) を返す。
        備考:
          - XY成分が極小（ほぼ真上/真下）なら base(+X) にフォールバック。
          - 実際の適用は _try_kinematic_slide で衝突検査しつつ行う。
        """
        # 後退禁止：距離は非負にクランプ
        distance = max(0.0, float(distance))

        # 対象エージェントのメタデータ選択
        agent = self.parent if body_id == self.parent_body else self.infant
        # 頭の前方ベクトル（ワールド座標）
        _, fwd_world = self._agent_torso_pose(agent) #TODO agent_nose_poseではないので変える
        dir_xy = np.array([fwd_world[0], fwd_world[1]], dtype=np.float32)
        norm = float(np.linalg.norm(dir_xy))
        if norm < 1e-6:
            # フォールバック: 胴体(base)の+X方向
            _, base_orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
            Rb = np.array(p.getMatrixFromQuaternion(base_orn), dtype=np.float32).reshape(3, 3)
            base_fwd = Rb @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
            dir_xy = np.array([base_fwd[0], base_fwd[1]], dtype=np.float32)
            norm = float(np.linalg.norm(dir_xy))
            if norm < 1e-6:
                return (0.0, 0.0)
        dir_xy /= norm
        dx = float(distance * dir_xy[0])
        dy = float(distance * dir_xy[1])
        return self._try_kinematic_slide(body_id, dx, dy)

    # ---------- レンダリング ----------
    def render(self):
        """GUI可視化用（ヘッドレス時は案内のみ）。補助線などのデバッグ描画を追加可能。"""
        if not self.gui:
            print("Enable gui=True to visualize.")
        # GUI のときは頭向き可視化を補助線で描くと便利（デバッグ用に line debug を追加可能）
        
    def _get_agent_view(self, body_id: int) -> np.ndarray:
        """head リンク姿勢からカメラを構築してRGB画像を返す（uint8, HxWx3）。
        - 視点: 鼻先（headローカル +X 方向へ nose_forward オフセット）
        - 注視点: 視点 + 前方ベクトル
        - upベクトル: +Z 固定（ロールは考慮しない簡易カメラ）
        - レンダラ: EGL 利用可なら ER_BULLET_HARDWARE_OPENGL、不可なら TinyRenderer
        - SSAA: 内部は高解像度で描画後、面積平均で縮小しジャギーを低減
        """
        # 引数の body_id が親か幼児かを判定し、対応するエージェントメタ情報を選択。
        agent = self.parent if body_id == self.parent_body else self.infant
        # 選んだエージェントの鼻先位置（視点）と前方方向ベクトル（ワールド座標）を取得。
        eye, dir_vec = self._agent_eye_pose(agent)
        eye = eye + dir_vec * agent["eyes_radius"]  # 目の位置オフセットを適用
        # カメラの注視点を「視点＋前方1単位」に設定（方向だけ使うので距離は1で十分）
        target = eye + dir_vec
        # カメラの上方向ベクトルをワールド+Zに固定。
        up_vec = [0, 0, 1]
        # 視点・注視点・Upからビュー行列（カメラの位置と向きを決める）を生成。
        view_matrix = p.computeViewMatrix(eye, target, up_vec)
        
        # ▼ SSAA: 内部は高解像度で描いて、最後に縮小して返す
        W_hi = int(self.view_width  * self.ssaa)
        H_hi = int(self.view_height * self.ssaa)
        # 視野角FOV、アスペクト比（幅/高さ）、ニア・ファーから射影行列（カメラの視野角，画角，遠近を決める）を生成。
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.view_fov, aspect=float(W_hi)/float(H_hi),
            nearVal=self.view_near, farVal=self.view_far
        )
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.egl_ok else p.ER_TINY_RENDERER
        # カメラ画像を取得。戻り値は (width, height, rgba, depth, seg) だが、RGBAのみ使用。
        _, _, rgba, _, _ = p.getCameraImage(W_hi, H_hi, view_matrix, proj_matrix, renderer=renderer, flags=0, physicsClientId=self.client_id)
        # RGBAバッファを(H, W, 4)に整形しuint8へ変換、アルファを捨ててRGBだけを返却
        img_hi = np.reshape(rgba, (H_hi, W_hi, 4)).astype(np.uint8)[:, :, :3]
        # ▼ SSAA縮小（アンチエイリアス相当）
        if self.ssaa > 1:
            img = _downsample_lanczos(img_hi, self.view_width, self.view_height)
        else:
            img = img_hi
        return img

    # ---------- オラクル観測（oracle_isolation.md） ----------
    # 無しの画像座標 / 距離。有効な方向は長さ 1、有効な距離は正なので 0 や -1 と区別できる。
    _ORACLE_ABSENT_XY = (-2.0, -2.0)
    _ORACLE_ABSENT_DIST = -1.0
    _ORACLE_SHOW_REACH_TOL = 0.12  # palm と掲げ点の到達判定 [m]（JA / giving と同じ）

    def _get_segmentation(self, agent):
        """幼児カメラと同じビューでセグメンテーションを撮る。

        戻り値: (seg HxW int32, view_matrix, proj_matrix, cam_eye, cam_dir)
        RGB は捨てる。SSAA は使わず view_width × view_height で撮る。
        """
        eye, dir_vec = self._agent_eye_pose(agent)
        cam_eye = np.asarray(eye, dtype=np.float32) + np.asarray(dir_vec, dtype=np.float32) * float(agent["eyes_radius"])
        cam_dir = np.asarray(dir_vec, dtype=np.float32)
        target = cam_eye + cam_dir
        up_vec = [0, 0, 1]
        view_matrix = p.computeViewMatrix(cam_eye.tolist(), target.tolist(), up_vec)
        W = int(self.view_width)
        H = int(self.view_height)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.view_fov, aspect=float(W) / float(H),
            nearVal=self.view_near, farVal=self.view_far,
        )
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.egl_ok else p.ER_TINY_RENDERER
        _, _, _, _, seg = p.getCameraImage(
            W, H, view_matrix, proj_matrix,
            renderer=renderer,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=self.client_id,
        )
        seg = np.reshape(seg, (H, W)).astype(np.int32)
        return seg, view_matrix, proj_matrix, cam_eye, cam_dir

    @staticmethod
    def _world_to_ndc(point_3d, view_matrix, proj_matrix):
        """ワールド座標を画像正規化座標 (右+1, 上+1) へ投影する。

        カメラ前方（クリップ w > 0）なら ±1 を超えてもその値を返す。
        背後（w <= 0）なら (-2, -2)。
        """
        vm = np.asarray(view_matrix, dtype=np.float64).reshape(4, 4, order="F")
        pm = np.asarray(proj_matrix, dtype=np.float64).reshape(4, 4, order="F")
        hom = np.array(
            [float(point_3d[0]), float(point_3d[1]), float(point_3d[2]), 1.0],
            dtype=np.float64,
        )
        clip = pm @ (vm @ hom)
        if clip[3] <= 0.0:
            return BaseEnv._ORACLE_ABSENT_XY
        return (float(clip[0] / clip[3]), float(clip[1] / clip[3]))

    @staticmethod
    def _parse_seg_buffer(seg: np.ndarray, body_id: int):
        """body_id の画素があれば (左端の列, 上端の行) を返す。無ければ None。

        PyBullet のセグメンテーション画素は
        objectUniqueId + ((linkIndex + 1) << 24)。
        """
        body_id = int(body_id)
        # 背景は -1。符号付きのままだとマスク後に別値になるので、非負画素だけ使う。
        valid = seg >= 0
        if not np.any(valid):
            return None
        bodies = np.where(valid, seg & ((1 << 24) - 1), -1)
        mask = bodies == body_id
        if not np.any(mask):
            return None
        rows, cols = np.where(mask)
        # 一番左の列。同列なら上（行が小さい）
        left_col = int(cols.min())
        top_among_left = int(rows[cols == left_col].min())
        return (left_col, top_among_left)

    @staticmethod
    def _seg_has_link(seg: np.ndarray, body_id: int, link_id: int) -> bool:
        """(body_id, link_id) の画素が 1 つでもあれば True。"""
        if link_id is None:
            return False
        body_id = int(body_id)
        link_id = int(link_id)
        valid = seg >= 0
        if not np.any(valid):
            return False
        bodies = np.where(valid, seg & ((1 << 24) - 1), -1)
        links = np.where(valid, (seg >> 24) - 1, -2)
        return bool(np.any((bodies == body_id) & (links == link_id)))

    def _agent_point_pose(self, agent):
        """point_link の位置と前方（ローカル +Z の負、index と同じ）。"""
        link_id = agent.get("point_link")
        if link_id is None:
            return self._agent_index_pose(agent)
        link_state = p.getLinkState(
            agent["body"], link_id,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        pos = np.array(link_state[4], dtype=np.float32)
        orn = link_state[5]
        R = np.array(p.getMatrixFromQuaternion(orn), dtype=np.float32).reshape(3, 3)
        fwd_local = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        fwd_world = R @ fwd_local
        return pos, fwd_world

    def _oracle_xy_dist(self, point_3d, view_matrix, proj_matrix, cam_eye, visible: bool):
        """可視なら (画像x, 画像y, 距離[m])。不可視なら無し値。

        画素はあるが中心が背後のとき: 画像座標だけ (-2,-2)、距離はメートルのまま。
        """
        if not visible:
            return (
                self._ORACLE_ABSENT_XY[0],
                self._ORACLE_ABSENT_XY[1],
                self._ORACLE_ABSENT_DIST,
            )
        xy = self._world_to_ndc(point_3d, view_matrix, proj_matrix)
        dist = float(np.linalg.norm(
            np.asarray(point_3d, dtype=np.float32) - np.asarray(cam_eye, dtype=np.float32)
        ))
        return (float(xy[0]), float(xy[1]), dist)

    def _oracle_color_shape(self, obj) -> Tuple[float, float, float, float]:
        """色 2 + 形 2。COLOR_SHAKE_EFFECT / SHAPE_SHAKE_SOUND を流用。無しは (0,0)。"""
        from cueworld.objects.items import COLOR_SHAKE_EFFECT, SHAPE_SHAKE_SOUND
        color = getattr(obj, "color", None)
        shape = getattr(obj, "shape_type", None)
        cvec = COLOR_SHAKE_EFFECT.get(color, [0.0, 0.0]) if color is not None else [0.0, 0.0]
        svec = SHAPE_SHAKE_SOUND.get(shape, [0.0, 0.0]) if shape is not None else [0.0, 0.0]
        return (float(cvec[0]), float(cvec[1]), float(svec[0]), float(svec[1]))

    def _oracle_show_position(self, held_obj) -> np.ndarray:
        """幼児の目の高さで、目から親方向へ水平に進んだ掲げ点（compute_show_position と同じ式）。"""
        from cueworld.infant_signal import GAZE_RAY_ORIGIN_OFFSET
        half = 0.06
        extents = getattr(held_obj, "bbox_half_extents", None)
        if extents:
            half = float(max(extents))
        distance = float(GAZE_RAY_ORIGIN_OFFSET) + 2.0 * half
        eyes_state = p.getLinkState(
            self.infant["body"], self.infant["eyes_link"],
            computeForwardKinematics=True, physicsClientId=self.client_id,
        )
        eye = np.asarray(eyes_state[4], dtype=np.float32)
        infant_base, _ = p.getBasePositionAndOrientation(
            self.infant["body"], physicsClientId=self.client_id
        )
        parent_base, _ = p.getBasePositionAndOrientation(
            self.parent["body"], physicsClientId=self.client_id
        )
        direction = (
            np.asarray(parent_base[:2], dtype=np.float32)
            - np.asarray(infant_base[:2], dtype=np.float32)
        )
        norm = float(np.linalg.norm(direction))
        if norm < 1e-4:
            _, fwd = self._agent_torso_pose(self.parent)
            direction = np.asarray([fwd[0], fwd[1]], dtype=np.float32)
            norm = float(np.linalg.norm(direction))
        if norm < 1e-4:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm
        show = eye.copy()
        show[0] = eye[0] + direction[0] * distance
        show[1] = eye[1] + direction[1] * distance
        show[2] = eye[2]
        return show

    def _build_oracle_objects(self, seg, view_matrix, proj_matrix, cam_eye) -> np.ndarray:
        """写っている物を左から並べた (枠数, 7)。余り枠は無し値。"""
        n = int(self._oracle_slot_count)
        out = np.zeros((n, 7), dtype=np.float32)
        out[:, 0] = self._ORACLE_ABSENT_XY[0]
        out[:, 1] = self._ORACLE_ABSENT_XY[1]
        out[:, 2] = self._ORACLE_ABSENT_DIST
        # 残り色・形は 0

        visible = []
        for oid in list(getattr(self, "active_objects", []) or []):
            pix = self._parse_seg_buffer(seg, oid)
            if pix is None:
                continue
            left_col, top_row = pix
            pos = self._obj_pos(oid)
            dist = float(np.linalg.norm(
                np.asarray(pos, dtype=np.float32) - np.asarray(cam_eye, dtype=np.float32)
            ))
            visible.append((left_col, top_row, dist, oid, pos))

        # 左端の列 → 上の行 → 近い距離
        visible.sort(key=lambda t: (t[0], t[1], t[2]))

        for i, (_lc, _tr, _d, oid, pos) in enumerate(visible):
            if i >= n:
                break
            xy = self._world_to_ndc(pos, view_matrix, proj_matrix)
            # 背後なら画像座標は無し、距離はメートルのまま
            dist = float(np.linalg.norm(
                np.asarray(pos, dtype=np.float32) - np.asarray(cam_eye, dtype=np.float32)
            ))
            obj = self.all_objects.get(oid)
            c0, c1, s0, s1 = self._oracle_color_shape(obj) if obj is not None else (0.0, 0.0, 0.0, 0.0)
            out[i, 0] = float(xy[0])
            out[i, 1] = float(xy[1])
            out[i, 2] = dist
            out[i, 3] = c0
            out[i, 4] = c1
            out[i, 5] = s0
            out[i, 6] = s1
        return out

    def _build_oracle_parent(self, seg, view_matrix, proj_matrix, cam_eye) -> np.ndarray:
        """(目x, 目y, 目距離, 指x, 指y, 指距離)。"""
        out = np.array(
            [
                self._ORACLE_ABSENT_XY[0], self._ORACLE_ABSENT_XY[1], self._ORACLE_ABSENT_DIST,
                self._ORACLE_ABSENT_XY[0], self._ORACLE_ABSENT_XY[1], self._ORACLE_ABSENT_DIST,
            ],
            dtype=np.float32,
        )
        if self.num_agents < 2 or getattr(self, "parent", None) is None:
            return out
        parent = self.parent
        body = int(parent["body"])

        eye_visible = self._seg_has_link(seg, body, parent.get("eyes_link"))
        if eye_visible:
            eye_pos, _ = self._agent_eye_pose(parent)
            x, y, d = self._oracle_xy_dist(eye_pos, view_matrix, proj_matrix, cam_eye, True)
            out[0], out[1], out[2] = x, y, d

        point_visible = self._seg_has_link(seg, body, parent.get("point_link"))
        if point_visible:
            tip_pos, _ = self._agent_point_pose(parent)
            x, y, d = self._oracle_xy_dist(tip_pos, view_matrix, proj_matrix, cam_eye, True)
            out[3], out[4], out[5] = x, y, d
        return out

    def _build_oracle_cues(self, seg, view_matrix, proj_matrix, cam_eye) -> np.ndarray:
        """(見るxy, 指差すxy, 掲げるxy, 掲げる0/1)。プロトコル名では出し分けない。"""
        absent = self._ORACLE_ABSENT_XY
        out = np.array(
            [absent[0], absent[1], absent[0], absent[1], absent[0], absent[1], 0.0],
            dtype=np.float32,
        )
        if self.num_agents < 2 or getattr(self, "parent", None) is None:
            return out
        parent = self.parent
        body = int(parent["body"])

        eye_visible = self._seg_has_link(seg, body, parent.get("eyes_link"))
        eye_pos = None
        eye_fwd = None
        if eye_visible:
            eye_pos, eye_fwd = self._agent_eye_pose(parent)
            look_pt = np.asarray(eye_pos, dtype=np.float32) + np.asarray(eye_fwd, dtype=np.float32) * 1.0
            xy = self._world_to_ndc(look_pt, view_matrix, proj_matrix)
            out[0], out[1] = float(xy[0]), float(xy[1])

        point_visible = self._seg_has_link(seg, body, parent.get("point_link"))
        if point_visible:
            tip_pos, tip_fwd = self._agent_point_pose(parent)
            point_pt = np.asarray(tip_pos, dtype=np.float32) + np.asarray(tip_fwd, dtype=np.float32) * 1.0
            xy = self._world_to_ndc(point_pt, view_matrix, proj_matrix)
            out[2], out[3] = float(xy[0]), float(xy[1])

        held_oid = parent.get("held_obj")
        held_visible = False
        if held_oid is not None:
            held_visible = self._parse_seg_buffer(seg, held_oid) is not None

        # 掲げるの座標: 目と持っている物の両方が写っているときだけ
        if eye_visible and held_visible and eye_pos is not None:
            held_pos = self._obj_pos(held_oid)
            to_held = np.asarray(held_pos, dtype=np.float32) - np.asarray(eye_pos, dtype=np.float32)
            norm = float(np.linalg.norm(to_held))
            if norm > 1e-6:
                unit = to_held / norm
                show_dir_pt = np.asarray(eye_pos, dtype=np.float32) + unit * 1.0
                xy = self._world_to_ndc(show_dir_pt, view_matrix, proj_matrix)
                out[4], out[5] = float(xy[0]), float(xy[1])

        # 掲げる 0/1: 持っている物が写っていて、palm が掲げ点から 0.12 m 以内
        if held_visible and held_oid is not None:
            held_obj = self.all_objects.get(held_oid)
            show_pos = self._oracle_show_position(held_obj)
            palm_pos, _ = self._get_palm_pose(parent)
            err = float(np.linalg.norm(
                np.asarray(palm_pos, dtype=np.float32) - np.asarray(show_pos, dtype=np.float32)
            ))
            if err < float(self._ORACLE_SHOW_REACH_TOL):
                out[6] = 1.0
        return out

    ###手の状態###
    def _get_hand_shape(self, agent):
        """
        エージェントの手の形状情報を返す。
        grip_power: 握りの強さ（closed のとき thumb_angle）
        intensity: その形としての強度 [0, 1]（理想 1、形の端 0）
        """
        index_angle = self.get_joint_angle(agent, "index_joint")
        thumb_angle = self.get_joint_angle(agent, "thumb_joint")
        return classify_hand_shape(index_angle, thumb_angle)
    ###掴む###
    def _get_palm_pose(self, agent) -> np.ndarray:
        """手のひらの向きベクトル（ワールド座標系）を返す。"""
        link_state = p.getLinkState(agent["body"], agent["hand_link"], computeForwardKinematics=True, physicsClientId=self.client_id)
        palm_pos = np.array(link_state[4])
        palm_orn = link_state[5]
        R = np.array(p.getMatrixFromQuaternion(palm_orn)).reshape(3, 3)
        palm_forward = R @ np.array([-1.0, 0.0, 0.0])
        palm_center = palm_pos + palm_forward * agent["palm_thickness"]
        return palm_center, palm_forward
    
    def _draw_grasp_bbox(self, agent_name: str, aabb_min: np.ndarray, aabb_max: np.ndarray):
        breakpoint()
        if agent_name in self._grasp_bbox_ids:
            for bid in self._grasp_bbox_ids[agent_name]:
                p.removeBody(bid, physicsClientId=self.client_id)
        x_min, y_min, z_min = aabb_min
        x_max, y_max, z_max = aabb_max
        
        pts = [
            [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
        ]
        edges = [
            (0, 1, 'x'), (1, 2, 'y'), (2, 3, 'x'), (3, 0, 'y'),
            (4, 5, 'x'), (5, 6, 'y'), (6, 7, 'x'), (7, 4, 'y'),
            (0, 4, 'z'), (1, 5, 'z'), (2, 6, 'z'), (3, 7, 'z')
        ]
        
        radius = 0.005 # 邪魔にならない細さ
        color = [1, 0, 0, 1]
        
        new_line_ids = []
        for (i, j, axis) in edges:
            p1 = np.array(pts[i])
            p2 = np.array(pts[j])
            center = (p1 + p2) / 2.0
            length = np.linalg.norm(p2 - p1)
            
            if length == 0: continue
                
            if axis == 'x':
                orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
            elif axis == 'y':
                orn = p.getQuaternionFromEuler([-np.pi/2, 0, 0])
            else:
                orn = [0, 0, 0, 1]
                
            vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=color, physicsClientId=self.client_id)
            # コリジョンを持たせないため collisionShapeIndex=-1、質量0
            edge_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_shape, basePosition=center, baseOrientation=orn, physicsClientId=self.client_id)
            
            new_line_ids.append(edge_body)
            
        self._grasp_bbox_ids[agent_name] = new_line_ids

    def detect_grasp_object(self, agent) -> Tuple[Optional[int], Optional[int]]:
        """手のひらの向きにあるオブジェクトを探索し，掴むオブジェクトIDを返す
        見つからない場合は (None, None) を返す。
        """
        #TODO: 物を既に持っている場合は，その持っているオブジェクトをreturnしてこの関数を即抜けていい
        nearest_obj_id = None
        nearest_link_id = None
        nearest_obj_dist = float('inf')
        palm_pos, palm_forward = self._get_palm_pose(agent)
        # アクティブなオブジェクトのみを対象とする
        #pybulletのgetOverlappingObjects機能を使って対象オブジェクトの候補を絞る．
        raw_aabb_min = palm_pos - np.array([self.grasp_distance, self.grasp_distance, self.grasp_distance])
        raw_aabb_max = palm_pos + np.array([self.grasp_distance, self.grasp_distance, self.grasp_distance])
        
        if getattr(self, "debug_grasp_bbox", False):
            self._draw_grasp_bbox(agent["name"], raw_aabb_min, raw_aabb_max)
            
        candidates = p.getOverlappingObjects(list(raw_aabb_min), list(raw_aabb_max), physicsClientId=self.client_id)
        if candidates is None:
            candidates = []
        for obj_id, link_id in candidates:
            if obj_id not in self.active_objects:
                continue
            obj = self.all_objects.get(obj_id)
            if obj is None or not getattr(obj, "is_graspable", False):
                continue
            pts = p.getClosestPoints(bodyA=obj_id, bodyB=agent["body"], distance=self.grasp_distance, linkIndexA=link_id, linkIndexB=agent["hand_link"], physicsClientId=self.client_id)
            
            if len(pts) == 0:
                continue
            obj_dist = pts[0][8]
            obj_dir = pts[0][7]
            angle_diff = self._angle_between(palm_forward, obj_dir)
            contact_dist = getattr(self, "grasp_contact_distance", 0.13)
            if angle_diff < self.grasp_dir_threshold or obj_dist < contact_dist:
                if obj_dist < nearest_obj_dist:
                    nearest_obj_dist = obj_dist
                    nearest_obj_id = obj_id
                    nearest_link_id = link_id
        
        return nearest_obj_id, nearest_link_id

    def _update_grasp(self, agent, hand_shape, grip_power):
        """
        掴む/離す処理を行う。
        渡す/受け取るの判定も行う
        """
        receive_flag = False
        handover_flag = False
        contain_objs = None
        if agent.get("held_obj") is not None:
            # 既に掴んでいる場合は離す判定をチェック
            oid = agent["held_obj"]
            if hand_shape != "closed":
                #オブジェクトの速度を0にする
                p.resetBaseVelocity(oid, [0,0,0], [0,0,0], physicsClientId=self.client_id)
                # 重力を元に戻す
                p.changeDynamics(oid, -1, linearDamping=0.05, mass=agent["held_obj_mass"], physicsClientId=self.client_id)
                agent["held_obj"] = None
                agent["held_obj_prev_local"] = None  # 振る検出履歴をリセット
                handover_flag = True
                #TODO: オブジェクトの機能を実装したら，ここの返り値を作る
                self.release_effect(oid)
                #オブジェクトの周辺にcontainが可能なオブジェクトがないか確認
                obj_pos = self._obj_pos(oid)
                bbox_half = getattr(self.all_objects[oid], "bbox_half_extents", [0.1, 0.1, 0.1])
                aabb_min = np.array(obj_pos) - np.array(bbox_half)
                aabb_max = np.array(obj_pos) + np.array(bbox_half)
                candidates = p.getOverlappingObjects(aabb_min, aabb_max, physicsClientId=self.client_id)
                for obj_id, link_id in candidates:
                    if obj_id not in self.active_objects:
                        continue  
                    if self.all_objects[obj_id].is_platform:
                        self.all_objects[obj_id].objects_on_board.append(oid)
                        self.all_objects[oid].put_on = obj_id
                        break
                    if not self.all_objects[obj_id].can_contain:
                        continue
                    if self.all_objects[obj_id].open:
                        contain_objs = (obj_id, oid)
                        break
    
            else:
                #掴み続ける
                if grip_power < 2.2:
                    handover_flag = True
                self._update_kinematic_held_objects(agent, oid)
        else:
            # 掴んでいない場合は掴む判定をチェック
            if agent["previous_hand_state"] == "open" and hand_shape == "closed":
                oid = agent.get("graspable_obj")
                obj_link_id = agent.get("graspable_link")
                #print(self.all_objects[oid].name)
                if oid is not None:
                    is_handover = False
                    if agent["name"] == "parent" and hasattr(self, "infant") and self.infant.get("held_obj") == oid:
                        is_handover = True
                    elif agent["name"] == "infant" and hasattr(self, "parent") and self.parent.get("held_obj") == oid:
                        is_handover = True
                        
                    is_grasp = False
                    if is_handover:
                        # 1. 渡す（受け取る）
                        receive_flag = True
                        is_grasp = True
                    elif self.all_objects[oid].can_contain and self.all_objects[oid].contents:
                        # 2. オブジェクトの中身を取り出す
                        self.remove_effect(oid)
                    elif self.all_objects[oid].is_openable and obj_link_id == self.all_objects[oid].lid_link_id:
                        # 3. コンテナオブジェクトを開く・閉じる
                        if not self.all_objects[oid].open:
                            self.open_effect(oid)
                        else:
                            self.close_effect(oid)
                    else:
                        is_grasp = True
                        
                    if is_grasp:
                        # 掴んだ瞬間に重力を無効化
                        agent["held_obj_mass"] = p.getDynamicsInfo(bodyUniqueId=oid, linkIndex=-1, physicsClientId=self.client_id)[0]
                        p.changeDynamics(oid, -1, linearDamping=1.0, mass=0.0, physicsClientId=self.client_id)
                        agent["held_obj"] = oid
                        agent["held_obj_prev_local"] = None  # 把持開始時に振る検出履歴を初期化
                        self._update_kinematic_held_objects(agent, oid)
                        #ここで，掴むオブジェクトの掴む際の機能の発動を行う
                        #TODO: オブジェクトの機能を実装したら，ここの返り値を作る
                        if hasattr(self.all_objects[oid], "grasp_effect"):
                            self.grasp_effect(oid)
        return handover_flag or receive_flag, contain_objs
    
                                      
    def _update_kinematic_held_objects(self,agent, oid: int):
        """キネマティックモード時、保持しているオブジェクトを腕先にスナップさせる。"""
        # オブジェクトの物理シミュレーションを無効化（重力・慣性の影響を受けない）
    
        palm_pos, palm_forward = self._get_palm_pose(agent)
        _ , object_orn = p.getBasePositionAndOrientation(oid, physicsClientId=self.client_id)
        # オブジェクトを腕先に強制移動
        object_pos = palm_pos + palm_forward * self.grasp_object_offset
        p.resetBasePositionAndOrientation(oid, object_pos, object_orn, physicsClientId=self.client_id)
        
        p.resetBaseVelocity(oid, [0,0,0], [0,0,0], physicsClientId=self.client_id)

        # 振る検出: 胴体ローカル座標での変位を比較（歩行による全体移動を除去するため）
        obj = self.all_objects.get(oid)
        if obj is not None and getattr(obj, "can_shake", False):
            link_state = p.getLinkState(agent["body"], agent["torso_link"], computeForwardKinematics=True, physicsClientId=self.client_id)
            torso_pos = np.array(link_state[4])
            torso_R = np.array(p.getMatrixFromQuaternion(link_state[5]), dtype=np.float32).reshape(3, 3)
            # ワールド座標 → 胴体ローカル座標: local = R^T @ (object_pos - torso_pos)
            local_pos = torso_R.T @ (object_pos - torso_pos)
            prev_local = agent.get("held_obj_prev_local")
            if prev_local is not None:
                displacement = np.linalg.norm(local_pos - prev_local)
                if displacement > self.shake_move_threshold:
                    self.shake_effect(oid)
            agent["held_obj_prev_local"] = local_pos
    
        
    ### 授受更新 ###
    def _update_handover(self):
        """親と幼児の間でオブジェクトの授受を行う。"""
        if self.num_agents != 2:
            return  # 親がいない場合は何もしない
            
        # 幼児から親へ
        if self.infant.get("held_obj") is not None:
            oid = self.infant["held_obj"]
            self.infant["held_obj"] = None
            self.parent["held_obj"] = oid
            self._update_kinematic_held_objects(self.parent, oid)
        # 親から幼児へ
        if self.parent.get("held_obj") is not None:
            oid = self.parent["held_obj"]
            self.parent["held_obj"] = None
            self.infant["held_obj"] = oid
            self._update_kinematic_held_objects(self.infant, oid)
    
    def _apply_simple_physics(self):
        """
        エージェントが保持していないアクティブなオブジェクトに簡易的な物理（落下）を適用する。
        地面から一定の高さを超えている場合、1 env.step あたり self.fall_per_step [m] だけ落下させる。
        （0.01 m/step ≒ 0.1 m/s @ 10Hz。物理 timestep からは導出しない。
         0.05 だと配置・視線が不安定になるためゲートで下げた。）
        """
        held_objects = set()
        if self.infant and self.infant.get("held_obj") is not None:
            held_objects.add(self.infant["held_obj"])
        if self.parent and self.parent.get("held_obj") is not None:
            held_objects.add(self.parent["held_obj"])

        fall_distance = self.fall_per_step

        for obj_id in self.active_objects:
            if obj_id in held_objects:
                continue

            obj_instance = self.all_objects.get(obj_id)
            if obj_instance is not None and obj_instance.move_type is None:
                continue
            if hasattr(obj_instance, 'put_on') and obj_instance.put_on is not None:
                continue
            if not obj_instance or getattr(obj_instance, 'contained_in', None) is not None:
                continue

            try:
                pos, orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client_id)
                
                ground_contact_z = obj_instance.bbox_half_extents[2] if hasattr(obj_instance, 'bbox_half_extents') and obj_instance.bbox_half_extents is not None else 0.05

                if pos[2] > ground_contact_z:
                    new_z = max(ground_contact_z, pos[2] - fall_distance)
                    new_pos = [pos[0], pos[1], new_z]
                    p.resetBasePositionAndOrientation(obj_id, new_pos, orn, physicsClientId=self.client_id)
                    p.resetBaseVelocity(obj_id, [0,0,0], [0,0,0], physicsClientId=self.client_id)
            except p.error:
                continue

    # def _update_holding(self, agent):
    #     """
    #     抱っこの判定アルゴリズムを実装
    #     TODO: 実装
    #     """
    #     # 親の手が幼児の近くにあるかの判定
    #     # 親が幼児を持ち上げようとするか？(新しく，その行動項目を追加して)
    #     # 仮にholdingアクションとすると，その値が上がるごとに，幼児の座標が上がっていく．(holdingアクションが0より大きいとき，幼児は前進などができない．)
    #     # 抱っこできたら．親の前の上らへんの位置に幼児を固定．幼児は親とは反対を向く(視界が遮られないように)
    #     # 離すときは，holdingを下げて0にしていく．幼児は徐々に下がっていく．
    #     return False

    ### オブジェクトの機能 ###
    def release_effect(self, oid):
        pass

    def grasp_effect(self, oid):
        pass
    
    def open_effect(self, oid):
        obj = self.all_objects.get(oid)
        if obj is None: return
        
        if getattr(obj, "open_type", None) == "rotate":
            obj.open = True
            obj.can_contain = True
            if hasattr(obj, "lid_joint") and obj.lid_joint is not None:
                p.resetJointState(oid, obj.lid_joint, 1.57, physicsClientId=self.client_id)
        
        elif getattr(obj, "open_type", None) == "detach":
            if oid in self.active_objects:
                self.active_objects.remove(oid)
            p.resetBasePositionAndOrientation(oid, [0, 0, -100], [0, 0, 0, 1], physicsClientId=self.client_id)
            
            if hasattr(obj, "detachable_body_ids") and len(obj.detachable_body_ids) >= 2:
                container_id = obj.detachable_body_ids[0]
                lid_id = obj.detachable_body_ids[1]
                
                if container_id not in self.active_objects:
                    self.active_objects.append(container_id)
                if lid_id not in self.active_objects:
                    self.active_objects.append(lid_id)
                    
                container_obj = self.all_objects.get(container_id)
                if container_obj is not None:
                    container_obj.contents = getattr(obj, "contents", []).copy()
                
                pos, ori = p.getBasePositionAndOrientation(oid, physicsClientId=self.client_id)
                p.resetBasePositionAndOrientation(container_id, pos, ori, physicsClientId=self.client_id)
                p.resetBasePositionAndOrientation(lid_id, (pos[0], pos[1] + 0.2, pos[2]), ori, physicsClientId=self.client_id)
        
    def close_effect(self, oid):
        obj = self.all_objects.get(oid)
        if obj is None: return
        
        if getattr(obj, "open_type", None) == "rotate":
            obj.open = False
            obj.can_contain = False
            if hasattr(obj, "lid_joint") and obj.lid_joint is not None:
                p.resetJointState(oid, obj.lid_joint, 0.0, physicsClientId=self.client_id)
    
    def contain_effect(self, container_obj, target_obj):
        if container_obj is not None:
            if not hasattr(container_obj, "contents"):
                container_obj.contents = []
            if target_obj.id not in container_obj.contents:
                container_obj.contents.append(target_obj.id)
            target_obj.contained_in = container_obj.id
        
        if target_obj.id in self.active_objects:
            self.active_objects.remove(target_obj.id)
        p.resetBasePositionAndOrientation(target_obj.id, [0, 0, -100], [0, 0, 0, 1], physicsClientId=self.client_id)
    
    def remove_effect(self, container_id, target_obj_id=None):
        if target_obj_id is not None and target_obj_id in self.all_objects:
            self.all_objects[target_obj_id].contained_in = None
            if container_id in self.all_objects:
                container = self.all_objects[container_id]
                if hasattr(container, "contents") and target_obj_id in container.contents:
                    container.contents.remove(target_obj_id)
    
    def move_effect(self, oid):
        pass
    
    def shake_effect(self, oid):
        """振る効果: オブジェクトの shake_sound / shake_effect_vec を env チャネルに書き込む。
        同一 env step 内で複数回発火した場合は最後の発火値で上書きする。"""
        obj = self.all_objects.get(oid)
        if obj is None:
            return
        sound = getattr(obj, "shake_sound", None)
        effect = getattr(obj, "shake_effect_vec", None)
        if sound is not None:
            self._object_sound[:] = np.array(sound, dtype=np.float32)
        if effect is not None:
            self._object_effect[:] = np.array(effect, dtype=np.float32)
    
    def eat_effect(self, oid):
        pass


    
    # ---------- 内部状態 ----------
    def get_joint_angle(self, agent, joint_key):
        """指定関節の角度を取得。存在しなければ 0 を返す。"""
        angle = p.getJointState(agent["body"], agent[joint_key], physicsClientId=self.client_id)[0]
        return float(angle)
    
    def set_joint_angle(self, agent, joint_key, angle, force: float = 20.0):
        """指定関節の角度を設定。存在しなければ無視。"""
        if joint_key == "torso_joint" or joint_key == "roll_wrist_joint":
            #腰の回転だけはラップアラウンド
            angle = float(angle)
            if angle > math.pi:
                angle -= 2.0 * math.pi
            elif angle < -math.pi:
                angle += 2.0 * math.pi
        else:
            angle = float(np.clip(angle, agent[f"{joint_key}_limits"][0], agent[f"{joint_key}_limits"][1]))
            
        p.resetJointState(agent["body"], agent[joint_key], angle, physicsClientId=self.client_id)
    
    
    def _agent_torso_pose(self, agent):
        """
        胴体の位置と姿勢を取得
        """
        link_state = p.getLinkState(agent["body"], agent["torso_link"], computeForwardKinematics=True, physicsClientId=self.client_id)
        # torsoリンクのワールド位置ベクトルをNumPy配列化
        torso_pos = np.array(link_state[4])         # worldLinkFramePosition
        # torsoリンクのワールド姿勢（クォータニオン, [x,y,z,w]）
        torso_orn = link_state[5]                   # worldLinkFrameOrientation (quat)
        # クォータニオン → 回転行列
        # クォータニオンを3x3回転行列（フラットな長さ9のリスト, 行優先）に変換
        R = p.getMatrixFromQuaternion(torso_orn)
        # 行列を3x3に整形
        R = np.array(R, dtype=np.float32).reshape(3,3)
        
        # torsoローカルの前方ベクトルは +X
        fwd_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # 回転行列でローカル前方ベクトルをワールド座標系へ変換。単位ベクトルのまま。
        fwd_world = R @ fwd_local                   # 世界座標の前方向き 
        
        return torso_pos, fwd_world
 
    def _agent_eye_pose(self, agent):
        """
        目の位置と姿勢を取得
        """
        "目の位置と姿勢を取得"
        link_state = p.getLinkState(agent["body"], agent["eyes_link"], computeForwardKinematics=True, physicsClientId=self.client_id)
        # eyesリンクのワールド位置ベクトルをNumPy配列化
        eye_pos = np.array(link_state[4])         # worldLinkFramePosition
        # eyesリンクのワールド姿勢（クォータニオン, [x,y,z,w]）
        eye_orn = link_state[5]                   # worldLinkFrameOrientation (quat)
        # クォータニオン → 回転行列
        # クォータニオンを3x3回転行列（フラットな長さ9のリスト, 行優先）に変換
        R = p.getMatrixFromQuaternion(eye_orn)
        # 行列を3x3に整形
        R = np.array(R, dtype=np.float32).reshape(3,3)
        
        # eyesローカルの前方ベクトル（鼻方向）は +X
        fwd_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # 回転行列でローカル前方ベクトルをワールド座標系へ変換。単位ベクトルのまま。
        fwd_world = R @ fwd_local                   # 世界座標の前方向き 
        
        return eye_pos, fwd_world  
    
    def _agent_index_pose(self, agent):
        """
        人差し指先端の位置と姿勢を取得
        """
        link_state = p.getLinkState(agent["body"], agent["index_link"], computeForwardKinematics=True, physicsClientId=self.client_id)
        # indexリンクのワールド位置ベクトルをNumPy配列化
        index_pos = np.array(link_state[4])         # worldLinkFramePosition
        # indexリンクのワールド姿勢（クォータニオン, [x,y,z,w]）
        index_orn = link_state[5]                   # worldLinkFrameOrientation (quat)
        # クォータニオン → 回転行列
        # クォータニオンを3x3回転行列（フラットな長さ9のリスト, 行優先）に変換
        R = p.getMatrixFromQuaternion(index_orn)
        # 行列を3x3に整形
        R = np.array(R, dtype=np.float32).reshape(3,3)
        
        # indexローカルの前方ベクトルは +X
        fwd_local = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        # 回転行列でローカル前方ベクトルをワールド座標系へ変換。単位ベクトルのまま。
        fwd_world = R @ fwd_local                   # 世界座標の前方向き 
        
        return index_pos, fwd_world
    
    def _lower_arm_pose(self, agent):
        """
        腕先の位置と姿勢を取得
        """
        link_state = p.getLinkState(agent["body"], agent["lower_arm_link"], computeForwardKinematics=True, physicsClientId=self.client_id)
        # handリンクのワールド位置ベクトルをNumPy配列化
        arm_pos = np.array(link_state[4])         # worldLinkFramePosition
        # handリンクのワールド姿勢（クォータニオン, [x,y,z,w]）
        arm_orn = link_state[5]                   # worldLinkFrameOrientation (quat)
        # クォータニオン → 回転行列
        # クォータニオンを3x3回転行列（フラットな長さ9のリスト, 行優先）に変換
        R = p.getMatrixFromQuaternion(arm_orn)
        # 行列を3x3に整形
        R = np.array(R, dtype=np.float32).reshape(3,3)
        
        fwd_local = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        # 回転行列でローカル前方ベクトルをワールド座標系へ変換。単位ベクトルのまま。
        
        fwd_world = R @ fwd_local                   # 世界座標の前方向き 
        return arm_pos, fwd_world

    def _obj_pos(self, oid: int) -> np.ndarray:
        """オブジェクト(body id)のワールド位置を返す。"""
        if oid in self.all_objects and getattr(self.all_objects[oid], "contained_in", None) is not None:
            return self._obj_pos(self.all_objects[oid].contained_in)
        pos, _ = p.getBasePositionAndOrientation(int(oid), physicsClientId=self.client_id)
        return np.asarray(pos, dtype=np.float32)
    
    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
        """2ベクトル間角度[rad]。長さ極小なら0を返し学習安定性確保。"""
        a = np.array(a, dtype=np.float32); b = np.array(b, dtype=np.float32)
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na < eps or nb < eps:
            return 0
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

    def _get_random_position_in_arena(self, 
                                     margin: float = 0.0,
                                     x_range: tuple[float, float] | None = (-0.55, 1.05),
                                     y_range: tuple[float, float] | None = (-1.0, 1.0)) -> tuple[float, float, float]:
        """アリーナ内のランダムな位置を返す。
        
        引数:
            margin: 壁からの余白（デフォルト: 0.0）
            x_range: X座標の範囲 (min, max)。Noneの場合はアリーナ全体
            y_range: Y座標の範囲 (min, max)。Noneの場合はアリーナ全体
        
        戻り値:
            (x, y, 0.0) のタプル
        """
        xh, yh = self.arena_half_extents
        
        # X範囲の決定
        if x_range is not None:
            x_min = max(-xh + margin, x_range[0])
            x_max = min(xh - margin, x_range[1])
        else:
            x_min = -xh + margin
            x_max = xh - margin
        
        # Y範囲の決定
        if y_range is not None:
            y_min = max(-yh + margin, y_range[0])
            y_max = min(yh - margin, y_range[1])
        else:
            y_min = -yh + margin
            y_max = yh - margin
        
        # ランダムな位置を生成
        x = self.np_random.uniform(x_min, x_max)
        y = self.np_random.uniform(y_min, y_max)
        
        return (float(x), float(y), 0.0)

    @staticmethod
    def angle_bisector_theorem(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
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
