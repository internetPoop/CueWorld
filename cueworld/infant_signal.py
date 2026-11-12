#幼児のシグナルを検知する関数をまとめたクラス
from typing import Optional, Tuple, Dict, Any, List, Union
import math
from collections import deque
import numpy as np
import gymnasium as gym
import pybullet as p

from cueworld.intent import DEFAULT_VOICE_VOLUME_THRESHOLD

PERIODICITY_WINDOW = 5
# dyadic 手高さ salience — 直近窓の手 world Z が帯内かのヒット率
HAND_HEIGHT_WINDOW = 10
HAND_HEIGHT_TOL = 0.4  # 目線より 40cm 下まで許容 [m]
DEFAULT_EYE_HEIGHT = 0.95  # eyes_link を取れない環境向けフォールバック

# 円錐レイキャスト（視線・指差し・リーチ・腕向き共通）
CONE_RING_SPECS = (
    {"angle_deg": 3.0, "num_rays": 4},
    {"angle_deg": 8.0, "num_rays": 8},
)
CONE_MIN_ACCURACY = 0.25  # 最外リング（8°）の確度下限。intent.HAND_TARGET_MASS_MIN と別レイヤ
CONE_RAY_LENGTH = 10.0
# 自分の顔に当たらないように、視線レイの起点を目から前へ出す距離 [m]
GAZE_RAY_ORIGIN_OFFSET = 0.1
# 不採用の発話を「ちがうよ」の対象にする最短の記録フレーム数。無音は数に入らない。
UNADOPTED_WORD_MIN_FRAMES = 3


def cone_ray_accuracy(
    angle_deg: float,
    outer_angle_deg: float,
    min_accuracy: float = CONE_MIN_ACCURACY,
) -> float:
    """レイ角度から確度 [min_accuracy, 1] を返す。中心 1.0、最外角で min_accuracy。"""
    if outer_angle_deg <= 0.0:
        return float(np.clip(min_accuracy, 0.0, 1.0))
    raw = 1.0 - (float(angle_deg) / float(outer_angle_deg))
    return float(np.clip(raw, min_accuracy, 1.0))


class InfantSignalDetector(gym.Wrapper):
    """
    幼児のシグナルを判定して数値化するためのクラス。
    docs/env_plan/infant_signal.md で定義された幼児シグナルを
    環境(base.py)から取得し、[0.0, 1.0]の実数値として判定・数値化を行う。
    base.pyの上にWrapperとして実装する．
    """
    def __init__(self,
        env: gym.Env, 
        history_length: int = 30, 
        phoneme_history_length: int = 20,
        phoneme_interval: int = 10,
        voice_interval: int = 3,
        word_history_length: int = 5,
        # 固定長 5・継続 6 フレーム = 30 ステップ．セグメント認識までの暫定値
        word_interval: int = 36,
        active_signals: Optional[List[str]] = None,
        cone_rings: Optional[Tuple[Dict[str, float], ...]] = None,
        cone_min_accuracy: float = CONE_MIN_ACCURACY,
        cone_ray_length: float = CONE_RAY_LENGTH,
        *args, **kwargs
    ):
        self.env = env
        self.history_length = history_length
        self.cone_rings = tuple(cone_rings) if cone_rings is not None else CONE_RING_SPECS
        self.cone_min_accuracy = float(cone_min_accuracy)
        self.cone_ray_length = float(cone_ray_length)
        # 時間は step 前提（1 env.step ≒ 0.1s）。速度・履歴長は per-step で扱う（物理 timestep から導出しない）。
        
        # 履歴バッファ
        #手，ジェスチャー
        self.hand_poses = deque(maxlen=history_length) #ハンドサインの履歴
        self.hand_world_z = deque(maxlen=history_length)  # D1: 手 hand_link の world Z 履歴
        self.infant_eye_height: Optional[float] = None  # D1: エピソード中固定の目線高さ（reset で再取得）
        self.grab_flags = deque(maxlen=history_length) #掴んでいるかのフラグの履歴
        self.touch_flags = deque(maxlen=history_length) #触っているかのフラグの履歴（開き手で掴める距離）
        ##以下はターゲットの履歴だが，Noneとそれ以外によりそれに関するプロトコルのサインとしても使える
        self.hand_targets = deque(maxlen=history_length) #指差し，掴み，リーチングのターゲット候補とそれらの強さの辞書の履歴(それらをしていない場合はNoneが入る) 

        # 模倣検知用: 関節角・姿勢ラベル履歴
        self.hand_pose_labels = deque(maxlen=history_length)
        self.index_joint_history = deque(maxlen=history_length)
        self.thumb_joint_history = deque(maxlen=history_length)
        self.periodicity_window = kwargs.get("periodicity_window", PERIODICITY_WINDOW)

        #視線
        self.gaze_targets = deque(maxlen=history_length) #視線の対象履歴(物を見ていない場合はNone)
        self.looking_at_parent_eyes = deque(maxlen=history_length) #親の目を見ているかの履歴
        self.looking_at_parent_faces = deque(maxlen=history_length) #親の顔(face_links)を見ているかの履歴

        #音声
        ##音素や単語の履歴
        self.phoneme_history_length = phoneme_history_length
        self.voice_interval = voice_interval
        self.voice_volume_threshold = float(
            getattr(env, "voice_volume_threshold", DEFAULT_VOICE_VOLUME_THRESHOLD)
        )
        
        # シグナル制限
        self.active_signals = active_signals  # Noneなら全てのシグナルを使用、リストなら指定されたシグナルのみ使用
        self._instant_parent_eye_hit = False
        self._instant_parent_face_hit = False
            
        # J1: 視線拒否 2 段判定用 — 幼児の頭回転行動履歴（水平=torso+pan 合算）
        self.infant_horizontal_turn_actions = deque(maxlen=history_length)
        self.infant_head_tilt_actions = deque(maxlen=history_length)

        if self.env.easy_voice:
            self.phoneme_model = None #音素認識モデル
            # 抽象音素 p0..p7．座標は 3 ビット符号から生成する
            from cueworld.language.phonemes import build_phoneme_prototypes, build_phoneme_mapping
            self.phoneme_prototypes = build_phoneme_prototypes()
            self.phoneme_mapping = build_phoneme_mapping()
        else:
            raise NotImplementedError("音素認識モデルはまだ実装されていません")
            #self.phoneme_model =
        self.phoneme_history = deque(maxlen=phoneme_history_length) #音素の履歴
        self.phoneme_scores_history = deque(maxlen=phoneme_history_length) #音素のスコアの履歴
        self.word_history_length = word_history_length
        self.word_interval = word_interval #単語を認識するのを何ステップごとに行うか（セグメント未使用時のフォールバック）
        self.word_history = deque(maxlen=word_history_length) #単語の履歴
        self.word_scores_history = deque(maxlen=word_history_length) #単語のスコアの履歴
        self.voice_segment_seq = 0
        self.voice_unadopted_attempt = False
        self.last_voice_reply = None
        # 切り分けテスト用。判定があった step だけ全語の boosted / raw を残す。
        # 公開の words（採用時だけ）とは別。判定の無い step は空。
        self.last_all_word_scores: Dict[str, float] = {}
        self.last_raw_word_scores: Dict[str, float] = {}
        self._voice_eval_seq_seen = 0
        # 沈黙で閉じた非採用の音素列。親が言い終わるまで保持する（喃語）。
        self.babbling_chunk: Optional[List[str]] = None
        # セグメント終了時に単語を更新する（True 推奨）．False なら word_interval 周期
        self.use_voice_segments = kwargs.get("use_voice_segments", True)
        
        # 単語辞書 (LanguageManagerから取得)
        from cueworld.language.manager import LanguageManager
        self.language_manager = LanguageManager()
        self.word_dictionary = self.language_manager.get_word_to_phonemes_dict()

        from cueworld.language.voice_recognizer import VoiceRecognizer
        self.voice_recognizer = VoiceRecognizer(self.word_dictionary)
        # 後方互換: しきい値・コストを detector 属性としても公開
        self.phoneme_cost_scale = self.voice_recognizer.phoneme_cost_scale
        self.cost_del = self.voice_recognizer.cost_del
        self.cost_del_blank = self.voice_recognizer.cost_del_blank
        self.cost_free_edge = self.voice_recognizer.cost_free_edge
        self.cost_ins = self.voice_recognizer.cost_ins
        self.cost_elongate = self.voice_recognizer.cost_elongate
        self.rejection_threshold = self.voice_recognizer.rejection_threshold
        self.phoneme_match_threshold = self.voice_recognizer.phoneme_match_threshold
        self.phoneme_keep_threshold = self.voice_recognizer.phoneme_keep_threshold
        self.phoneme_ambiguity_margin = self.voice_recognizer.phoneme_ambiguity_margin
        self.word_margin_threshold = self.voice_recognizer.word_margin_threshold
        self.word_sequence_quality_threshold = self.voice_recognizer.word_sequence_quality_threshold
        self.word_length_norm = self.voice_recognizer.word_length_norm
        self.max_absolute_word_cost = self.voice_recognizer.max_absolute_word_cost
        self.last_word_recognition_metrics = {}

        
        ##ボリュームやピッチ
        #TODO ボリュームやピッチは分けて使う必要があるので，base.pyの方で音声のwindowを保存するのとは別に保存する(そのほうが使いやすい)
        voice_history_len = 30  # ≒3s @ 10Hz（1 env.step ≒ 0.1s）
        self.voice_volume_history = deque(maxlen=voice_history_len)
        self.voice_pitch_history = deque(maxlen=voice_history_len)

        #表情
        self.face_expressions = deque(maxlen=history_length) #表情の履歴(0: Calm, 1: Excitement, 2: Distress, 3: Whine, 4: Refusal, 5: Sleepy)
        self.face_expression_accuracies = deque(maxlen=history_length) #表情の正確さの履歴
        # 履歴
        self.protocols = deque(maxlen=history_length)

        self.in_sight_parent = deque(maxlen=history_length) #どこに分類できるかな 


        # 現在の発話セグメント情報
        self.current_voice_segment = []
        self.last_voice_features = None
        self.last_emotion_probs = {
            "Calm": 0.0, "Excitement": 0.0, "Distress": 0.0,
            "Whine": 0.0, "Refusal": 0.0, "Sleepy": 0.0
        }

        # 情動プロトタイプ
        # T, V_avg, P_avg, dP, P_std, T_attack
        # 現状未使用: pitch 履歴は保持するが，情動推定への接続は将来課題
        self.emotion_prototypes = {
            "Calm":       np.array([0.8, 0.25, 0.3,  0.0, 0.01, 0.1]),
            "Excitement": np.array([0.5, 0.50, 0.7, -0.1, 0.02, 0.05]),
            "Distress":   np.array([2.0, 0.80, 0.9,  0.0, 0.05, 0.1]),
            "Whine":      np.array([1.5, 0.50, 0.6, -0.1, 0.03, 0.4]),
            "Refusal":    np.array([0.3, 0.50, 0.45,-0.3, 0.02, 0.05]),
            "Sleepy":     np.array([2.0, 0.15, 0.2, -0.2, 0.01, 1.0])
        }
        # 重み行列 (簡易的に対角行列wとして扱う)
        self.emotion_weights = np.array([1.0, 2.0, 2.0, 1.5, 5.0, 1.0])
        
    def _is_modality_active(self, modality: str) -> bool:
        if self.active_signals is None:
            return True
        
        modality_keys = {
            "gesture": [
                "point_intensity", "open_intensity", "hand_target_stability",
                "holding", "touching",
                "hand_open_close_periodicity", "hand_height_margin", "gesture_targets",
            ],
            "gaze": [
                "gaze_target_stability", "looking_at_parent_eye", "looking_at_parent_face",
                "instant_parent_face_hit", "gaze_targets",
                "gaze_targets_now",
                "sudden_gaze_shift", "gaze_shift_dict",
            ],
            "voice": [
                "voice_target_stability", "voice_best_distance", "babbling_target", "babbling_ratio",
                "babbling_chunk", "babbling_salience",
                "voice_volume", "words", "negative_emotion_surge",
                "voice_unadopted_attempt", "voice_segment_seq", "voice_reply",
            ],
            "expression": ["smile_intensity", "distress_intensity", "surprise_intensity", "expressions", "negative_emotion_surge"],
        }
        keys = modality_keys.get(modality, [])
        return any(k in self.active_signals for k in keys)

    def update(self):
        """シミュレーションの毎ステップ呼び出し、履歴を更新する。"""
        if not hasattr(self.env, "infant") or self.env.infant is None:
            return

        # 判定のあった step だけ残す。毎 step 先頭で空にし、下の音声処理で埋める。
        self.last_all_word_scores = {}
        self.last_raw_word_scores = {}
        
        # 手の姿勢・対象・把持の履歴 #TODO: ここの改造
        if self._is_modality_active("gesture"):
            hand_pose, hand_targets, grab_flag, target_accuracy, touch_flag = self.get_hand_status(self.env.infant)
            self.hand_poses.append(hand_pose)
            if hasattr(hand_targets, "shape"):
                if len(self.hand_targets) > 0 and len(hand_targets) != len(self.hand_targets[0]):
                    print(f"Shape mismatch! target: {len(hand_targets)}, expected: {len(self.hand_targets[0])}")
                    # shape is inhomogeneous!
            self.hand_targets.append(hand_targets)
            self.grab_flags.append(grab_flag)
            self.touch_flags.append(touch_flag)
            self.hand_world_z.append(self._get_hand_world_z(self.env.infant))
            self._ensure_infant_eye_height()
            self._update_hand_periodicity_histories()

        # 視線対象履歴
        if self._is_modality_active("gaze"):
            gaze_targets, looking_at_parent_eye, looking_at_parent_face = self.get_gaze_target()
            self._instant_parent_eye_hit = bool(looking_at_parent_eye)
            self._instant_parent_face_hit = bool(looking_at_parent_face)
            self.gaze_targets.append(gaze_targets)
            self.looking_at_parent_eyes.append(looking_at_parent_eye)
            self.looking_at_parent_faces.append(looking_at_parent_face)
        else:
            self._instant_parent_eye_hit = False
            self._instant_parent_face_hit = False

        # 音素・単語・声の情動の履歴
        if self._is_modality_active("voice"):
            volume = self._get_voice_volume_level()
            self.voice_volume_history.append(volume)
            # enable_emotion=False のとき pitch は voice_mapping に無いので履歴も積まない
            if getattr(self.env, "enable_emotion", False):
                pitch = self._get_voice_pitch_level()
                self.voice_pitch_history.append(pitch)
            if self.env.step_count % self.voice_interval == 0:
                # 窓の各フレームを判定し、コスト最小（同点は最新寄り）をそのティックの 1 ラベルにする。
                phoneme, phoneme_scores = self._pick_window_phoneme()
                self.phoneme_history.append(phoneme)
                self.phoneme_scores_history.append(phoneme_scores)
                if self.use_voice_segments:
                    kept_labels = self._kept_window_labels()
                    finished = self.voice_recognizer.update_segment(
                        volume, phoneme, phoneme_scores, kept_labels=kept_labels
                    )
                    if finished is not None:
                        word, word_scores = finished
                        self._commit_word_observation(word, word_scores)
                        # 単語として採用されたら喃語チャンクは消す。
                        if word is not None:
                            self.babbling_chunk = None
                    # 沈黙閉じで 3〜6 非採用なら recognizer が last_babble_chunk を立てる。
                    babble = self.voice_recognizer.last_babble_chunk
                    if babble is not None:
                        self.babbling_chunk = list(babble)
                        self.voice_recognizer.last_babble_chunk = None
                    # 採用なしの 6 音判定など、commit が無い評価でも密な報酬用にスコアを残す。
                    self._sync_last_word_scores_from_recognizer()
            if (not self.use_voice_segments) and self.env.step_count % self.word_interval == 0:
                word, word_scores = self.get_word(self.phoneme_scores_history)
                self._commit_word_observation(word, word_scores)
                self._sync_last_word_scores_from_recognizer()

        # 表情履歴（enable_emotion=False のときは更新しない）
        if getattr(self.env, "enable_emotion", False) and self._is_modality_active("expression"):
            face_expression, face_expression_accuracy = self.get_face_expression()
            self.face_expressions.append(face_expression)
            self.face_expression_accuracies.append(face_expression_accuracy)
        
    def get_normalized_signals(self, ignored_targets: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        全シグナルを [0.0, 1.0] の連続値（または二値）として取得し、
        親が意図推定やタスク設定に用いる。
        このシグナルは履歴から取得される．
        """
        signals = {}
        # ジェスチャーのシグナル
        if self._is_modality_active("gesture"):
            gesture_signals, gesture_targets = self.get_gesture_signals(ignored_targets)
            signals.update(gesture_signals)
            signals["gesture_targets"] = gesture_targets
            
        
        # 視線シグナル
        if self._is_modality_active("gaze"):
            if hasattr(self, "get_gaze_signals"):
                g_signals, g_targets = self.get_gaze_signals(ignored_targets)
                signals.update(g_signals)
                signals["gaze_targets"] = g_targets
                if len(self.gaze_targets) > 0:
                    signals["gaze_targets_now"] = np.asarray(
                        self.gaze_targets[-1], dtype=float
                    ).copy()
                
        # 表情シグナル（enable_emotion=False のときは出さない）
        if getattr(self.env, "enable_emotion", False) and self._is_modality_active("expression"):
            if hasattr(self, "get_expression_signals"):
                e_signals,expressions = self.get_expression_signals()
                signals.update(e_signals)
                signals["expressions"] = expressions
                
        # 音声シグナル
        if self._is_modality_active("voice"):
            if hasattr(self, "get_voice_signals"):
                v_signals, words = self.get_voice_signals()
                signals.update(v_signals)
                signals["words"] = words
                
        # 拒否シグナル（急激な変化）を常に追加
        signals.update(self.get_rejection_signals())

        if self.active_signals is not None:
            signals = {k: v for k, v in signals.items() if k in self.active_signals}

        return signals
    
    def reset(self, **kwargs):
        """Reset the environment and the detector's internal state."""
        obs, info = self.env.reset(**kwargs)
        
        # 履歴バッファのクリア
        self.hand_poses.clear()
        self.hand_world_z.clear()
        self.infant_eye_height = None
        self.grab_flags.clear()
        self.touch_flags.clear()
        self.hand_targets.clear()
        self.hand_pose_labels.clear()
        self.index_joint_history.clear()
        self.thumb_joint_history.clear()
        self.gaze_targets.clear()
        self.looking_at_parent_eyes.clear()
        self.looking_at_parent_faces.clear()
        self.phoneme_history.clear()
        self.phoneme_scores_history.clear()
        self.word_history.clear()
        self.word_scores_history.clear()
        self.voice_segment_seq = 0
        self.voice_unadopted_attempt = False
        self.last_voice_reply = None
        self.last_all_word_scores = {}
        self.last_raw_word_scores = {}
        self._voice_eval_seq_seen = 0
        self.babbling_chunk = None
        self.voice_volume_history.clear()
        self.voice_pitch_history.clear()
        self.face_expressions.clear()
        self.face_expression_accuracies.clear()
        self.protocols.clear()
        self.in_sight_parent.clear()
        self.infant_horizontal_turn_actions.clear()
        self.infant_head_tilt_actions.clear()
        self._instant_parent_eye_hit = False
        self._instant_parent_face_hit = False
        
        self.current_voice_segment = []
        self.last_voice_features = None
        self.last_emotion_probs = {
            "Calm": 0.0, "Excitement": 0.0, "Distress": 0.0,
            "Whine": 0.0, "Refusal": 0.0, "Sleepy": 0.0
        }
        if hasattr(self, "voice_recognizer"):
            self.voice_recognizer.reset()
        
        return obs, info

    def _compute_preference_reward(self) -> Tuple[float, Dict[str, float]]:
        """選好に基づく報酬。視線モダリティが無効、または親がいないときは 0。"""
        breakdown = {"hold": 0.0, "look": 0.0, "total": 0.0}
        if not self._is_modality_active("gaze"):
            return 0.0, breakdown
        if not hasattr(self.env, "parent") or self.env.parent is None:
            return 0.0, breakdown

        hold_reward = 0.0
        infant = getattr(self.env, "infant", None)
        if infant is not None:
            held_oid = infant.get("held_obj")
            if held_oid is not None and hasattr(self.env, "all_objects"):
                obj = self.env.all_objects.get(held_oid)
                if obj is not None:
                    hold_reward = float(getattr(obj, "preference", 0.0))

        look_reward = 0.0
        if self._instant_parent_eye_hit:
            look_reward = float(getattr(self.env, "parent_preference", 0.0))

        total = hold_reward + look_reward
        breakdown = {"hold": hold_reward, "look": look_reward, "total": total}
        return total, breakdown

    def step(self, action):
        #実際にステップ関数内でupdateを呼び出したりする．
        if isinstance(action, (list, tuple)) and len(action) >= 1:
            self._record_infant_turn_actions(action[0])
        o, r, d, t, i = self.env.step(action)
        self.update()
        pref_reward, pref_breakdown = self._compute_preference_reward()
        r = list(r)
        r[0] = float(r[0]) + pref_reward
        i["infant_signals"] = self.get_normalized_signals() #インフォを信号のキャリーとして扱う
        i["preference_reward"] = pref_breakdown
        return o, r, d, t, i

    def _record_infant_turn_actions(self, infant_action) -> None:
        """J1: 視線拒否用に幼児の水平回転(torso+pan)・tilt 行動を履歴へ記録する。"""
        if infant_action is None:
            return
        try:
            action_arr = np.asarray(infant_action, dtype=np.float32).reshape(-1)
            am = getattr(self.env, "action_map", {})
            horizontal = 0.0
            for key in ("torso_joint", "head_pan_joint"):
                idx = am.get(key)
                if idx is not None and 0 <= idx < len(action_arr):
                    horizontal += float(action_arr[idx])
            self.infant_horizontal_turn_actions.append(horizontal)
            tilt = 0.0
            tilt_idx = am.get("head_tilt_joint")
            if tilt_idx is not None and 0 <= tilt_idx < len(action_arr):
                tilt = float(action_arr[tilt_idx])
            self.infant_head_tilt_actions.append(tilt)
        except Exception:
            self.infant_horizontal_turn_actions.append(0.0)
            self.infant_head_tilt_actions.append(0.0)

    def get_average_hand_target_distribution(self) -> np.ndarray:
        """保持している手・ジェスチャー対象の確率分布の平均分布を計算する"""
        if len(self.hand_targets) == 0:
            return np.array([])
        return np.mean(list(self.hand_targets), axis=0)

    def get_average_gaze_target_distribution(self) -> np.ndarray:
        """保持している視線対象の確率分布の平均分布を計算する"""
        if len(self.gaze_targets) == 0:
            return np.array([])
        return np.mean(list(self.gaze_targets), axis=0)

    # ---------------- 0. 音声シグナル (Voice) ----------------

    def _get_voice_volume_level(self) -> float:
        """直近の infant_voice_window から平均 volume を [0, 1] で返す。"""
        if hasattr(self.env, "infant_voice_window") and len(self.env.infant_voice_window) > 0:
            avg_voice = np.mean(self.env.infant_voice_window, axis=0)
            volume = avg_voice[self.env.voice_mapping["volume"]]
        elif hasattr(self.env, "infant_voice"):
            volume = self.env.infant_voice[self.env.voice_mapping["volume"]]
        else:
            return 0.0
        return float(np.clip(volume, 0.0, 1.0))

    def _get_voice_pitch_level(self) -> float:
        """直近の infant_voice_window から平均 pitch を [-1, 1] で返す。
        pitch は音素軸ではなく韻律要素として別途保持する（現状は履歴のみ．情動推定は未接続）．
        enable_emotion=False のときは voice_mapping に pitch が無いので 0.0。
        """
        if not getattr(self.env, "enable_emotion", False):
            return 0.0
        if "pitch" not in getattr(self.env, "voice_mapping", {}):
            return 0.0
        if hasattr(self.env, "infant_voice_window") and len(self.env.infant_voice_window) > 0:
            avg_voice = np.mean(self.env.infant_voice_window, axis=0)
            pitch = avg_voice[self.env.voice_mapping["pitch"]]
        elif hasattr(self.env, "infant_voice"):
            pitch = self.env.infant_voice[self.env.voice_mapping["pitch"]]
        else:
            return 0.0
        return float(np.clip(pitch, -1.0, 1.0))

    def _pick_window_phoneme(self) -> Tuple[Optional[str], Dict[str, float]]:
        """infant_voice_window の各フレームを get_phoneme し、最小コストの 1 ラベルを返す。

        スコアが無い無声フレームは候補にしない。同点は窓の末尾に近いほう。
        候補が無ければゼロベクトルを get_phoneme する。
        """
        window = list(getattr(self.env, "infant_voice_window", []) or [])
        if not window:
            zeros = np.zeros(getattr(self.env, "voice_dim", 5))
            return self.get_phoneme(zeros)

        best_cost = None
        best_index = -1
        best_phoneme: Optional[str] = None
        best_scores: Dict[str, float] = {}
        for index, frame in enumerate(window):
            phoneme, scores = self.get_phoneme(np.array(frame, copy=True))
            if not scores:
                continue
            cost = min(float(v) for v in scores.values())
            # 同点は最新寄りなので、同じコストなら後ろのフレームで上書きする。
            if best_cost is None or cost < best_cost or (cost == best_cost and index >= best_index):
                best_cost = cost
                best_index = index
                best_phoneme = phoneme
                best_scores = scores
        if best_cost is None:
            zeros = np.zeros(getattr(self.env, "voice_dim", 5))
            return self.get_phoneme(zeros)
        return best_phoneme, best_scores

    def _kept_window_labels(self) -> List[str]:
        """窓の各フレームから、残す閾値を通った音素ラベルを時間順に返す。"""
        window = list(getattr(self.env, "infant_voice_window", []) or [])
        labels: List[str] = []
        for frame in window:
            phoneme, scores = self.get_phoneme(np.array(frame, copy=True))
            if self.voice_recognizer.label_passes_keep(phoneme, scores):
                labels.append(str(phoneme))
        return labels

    def _sync_last_word_scores_from_recognizer(self) -> None:
        """認識器が今ティックで評価したら、全語スコアを detector に残す。"""
        rec = self.voice_recognizer
        if int(getattr(rec, "word_eval_seq", 0) or 0) == int(self._voice_eval_seq_seen):
            return
        self._voice_eval_seq_seen = int(rec.word_eval_seq)
        self.last_all_word_scores = {
            str(k): float(v) for k, v in (rec.last_all_word_scores or {}).items()
        }
        self.last_raw_word_scores = {
            str(k): float(v) for k, v in (rec.last_raw_word_scores or {}).items()
        }

    def _commit_word_observation(self, word, word_scores) -> None:
        """セグメント終了時の単語を公開用の履歴へ入れる。不採用のスコア表は出さない。"""
        metrics = dict(self.voice_recognizer.last_word_recognition_metrics)
        self.last_word_recognition_metrics = metrics
        length = int(metrics.get("length", 0) or 0)
        self.voice_segment_seq += 1
        self.word_history.append(word)
        if self.voice_recognizer.last_speak_now:
            self.last_voice_reply = self.voice_recognizer.last_reply_kind
        else:
            self.last_voice_reply = None
        if word is None:
            self.word_scores_history.append({})
            self.voice_unadopted_attempt = length >= UNADOPTED_WORD_MIN_FRAMES
        else:
            self.word_scores_history.append(word_scores if word_scores is not None else {})
            self.voice_unadopted_attempt = False
        # 切り分けテスト用。不採用でも全語の boosted を残す（公開の words は空のまま）。
        if word_scores:
            self.last_all_word_scores = {str(k): float(v) for k, v in word_scores.items()}
        else:
            rec_scores = getattr(self.voice_recognizer, "last_all_word_scores", None) or {}
            self.last_all_word_scores = {str(k): float(v) for k, v in rec_scores.items()}
        raw_scores = getattr(self.voice_recognizer, "last_raw_word_scores", None) or {}
        self.last_raw_word_scores = {str(k): float(v) for k, v in raw_scores.items()}
        self._voice_eval_seq_seen = int(
            getattr(self.voice_recognizer, "word_eval_seq", self._voice_eval_seq_seen) or 0
        )

    def _compute_voice_target_stability(self) -> Tuple[float, float]:
        """単語スコア履歴から voice_target_stability と voice_best_distance を算出する。"""
        if not hasattr(self, "word_scores_history") or len(self.word_scores_history) == 0:
            return 0.0, 0.0

        latest_word_scores = self.word_scores_history[-1]
        best_word = self.word_history[-1]
        # 不採用は単語ではない。安定度は上げず、距離 0 にもしない。
        if best_word is None:
            return 0.0, float("inf")
        if latest_word_scores is None or len(latest_word_scores) == 0:
            return 0.0, 0.0

        scores = np.array(list(latest_word_scores.values()))
        probs = np.exp(-scores)
        sum_probs = np.sum(probs)
        if sum_probs > 0:
            probs = probs / sum_probs
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            if len(probs) > 1:
                max_entropy = np.log(len(probs))
                entropy_norm = float(np.clip(1.0 - (entropy / max_entropy), 0.0, 1.0))
            else:
                entropy_norm = 1.0
        else:
            entropy_norm = 0.0

        best_dist = float(latest_word_scores.get(best_word, 0.0))
        return entropy_norm, best_dist

    def get_voice_signals(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        音声シグナルを [0.0, 1.0] の連続値として取得する
        最新のDPマッチングの距離を確率変換してエントロピーを求め、確信度を算出する。
        """
        voice_volume = self._get_voice_volume_level()
        entropy_norm, best_dist = self._compute_voice_target_stability()

        # babbling_target / babbling_ratio は後方互換で残す（新契約では使わない）。
        babbling_chunk, babbling_salience = self.get_babbling_chunk()

        v_signals = {
            "voice_target_stability": entropy_norm,
            "voice_best_distance": best_dist,
            "babbling_target": None,
            "babbling_ratio": 0.0,
            "babbling_chunk": babbling_chunk,
            "babbling_salience": babbling_salience,
            "voice_volume": voice_volume,
            "voice_unadopted_attempt": bool(self.voice_unadopted_attempt),
            "voice_segment_seq": int(self.voice_segment_seq),
            "voice_reply": self.last_voice_reply,
        }

        words = {}
        if (
            hasattr(self, "word_scores_history")
            and len(self.word_scores_history) > 0
            and len(self.word_history) > 0
            and self.word_history[-1] is not None
        ):
            latest_word_scores = self.word_scores_history[-1]
            if latest_word_scores is not None:
                words = {str(k): float(v) for k, v in latest_word_scores.items()}

        return v_signals, words

    def get_babbling_chunk(self) -> Tuple[Optional[List[str]], float]:
        """保持中の喃語列と salience（列があれば 1、無ければ 0）を返す。"""
        if self.babbling_chunk:
            return list(self.babbling_chunk), 1.0
        return None, 0.0

    def clear_babbling_chunk(self) -> None:
        """親が言い終わったときに呼ぶ。同じ列で次の往復は始まらない。"""
        self.babbling_chunk = None
        if hasattr(self, "voice_recognizer"):
            self.voice_recognizer.last_babble_chunk = None

    def get_babbling_target(self) -> Tuple[Optional[str], float]:
        """
        直近の phoneme_history から最頻出の有効な音素と、その割合（直近履歴の長さに対する割合）を抽出する。
        無音時は履歴に音素が残っていても None を返す。
        新契約では get_babbling_chunk を使う。後方互換のため残す。
        """
        if self._get_voice_volume_level() < self.voice_volume_threshold:
            return None, 0.0

        if len(self.phoneme_history) == 0:
            return None, 0.0
            
        from collections import Counter
        # None や "unknown" を除外した有効な音素のリスト
        valid_phonemes = [p for p in self.phoneme_history if p is not None and p != "unknown"]
        
        if not valid_phonemes:
            return None, 0.0
            
        # 最も出現頻度が高い音素を一つ選ぶ
        counter = Counter(valid_phonemes)
        most_common_phoneme, count = counter.most_common(1)[0]
        
        # 履歴全体に対する割合を計算
        ratio = float(count) / len(self.phoneme_history)
        
        return most_common_phoneme, ratio

    def get_phoneme(self, avg_voice=None) -> Tuple[Optional[str], Dict[str, float]]:
        """
        現在の音素を取得する
        phonemeがNoneのときは，対応する音韻なし．
        スコアは phoneme_cost_scale で正規化した [0,1] コスト．
        """
        if not self.env.easy_voice:
            if avg_voice[self.env.voice_mapping["volume"]] >= self.voice_volume_threshold:
                return self.phoneme_model(self.env.infant_voice_window)
            return None, {}

        # detector 側で更新されたしきい値を recognizer へ同期
        self._sync_recognizer_params()
        return self.voice_recognizer.get_phoneme(
            avg_voice,
            volume_index=self.env.voice_mapping["volume"],
            phoneme_index=self.env.phoneme_index,
        )

    def _sync_recognizer_params(self) -> None:
        r = self.voice_recognizer
        r.word_dictionary = self.word_dictionary
        r.phoneme_prototypes = self.phoneme_prototypes
        r.phoneme_mapping = self.phoneme_mapping
        r.phoneme_cost_scale = self.phoneme_cost_scale
        r.phoneme_match_threshold = self.phoneme_match_threshold
        if hasattr(self, "phoneme_keep_threshold"):
            r.phoneme_keep_threshold = self.phoneme_keep_threshold
        else:
            self.phoneme_keep_threshold = r.phoneme_keep_threshold
        r.phoneme_ambiguity_margin = self.phoneme_ambiguity_margin
        r.voice_volume_threshold = self.voice_volume_threshold
        r.cost_del = self.cost_del
        r.cost_del_blank = self.cost_del_blank
        r.cost_free_edge = self.cost_free_edge
        r.cost_ins = self.cost_ins
        r.cost_elongate = self.cost_elongate
        r.rejection_threshold = self.rejection_threshold
        r.word_margin_threshold = self.word_margin_threshold
        r.word_sequence_quality_threshold = self.word_sequence_quality_threshold
        r.word_length_norm = self.word_length_norm
        r.max_absolute_word_cost = self.max_absolute_word_cost

    def _compute_sequence_quality(
        self,
        phoneme_history: List[Optional[str]],
        phoneme_scores_history: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """音素列の品質を計算する。"""
        self._sync_recognizer_params()
        return self.voice_recognizer._compute_sequence_quality(
            phoneme_history, phoneme_scores_history
        )

    def get_word(self, phoneme_scores_history, context=None):
        """現在の単語を取得する(拡張版レーベンシュタイン距離)"""
        self._sync_recognizer_params()
        word, word_scores = self.voice_recognizer.get_word(
            self.phoneme_history, phoneme_scores_history, context=context
        )
        self.last_word_recognition_metrics = dict(
            self.voice_recognizer.last_word_recognition_metrics
        )
        return word, word_scores

    # ---------------- 1. 手・腕のシグナル (Hand & Arm) ----------------

    def get_hand_status(self, agent) -> Tuple[np.ndarray, np.ndarray, Optional[int], bool]:
        """
        幼児の手の姿勢、対象、掴んでいる対象を取得する。
        
        Returns:
            hand_pose: 手の姿勢 (type, intensity)
            hand_targets: 手の対象ベクトル
            grab_flag: 掴んでいるか (True/False)
            target_accuracy: 対象の確度
            touch_flag: 触っているか (True/False)。掴める距離に物があるとき True
        """
        # 手の形状の取得
        hand_pose, _, hand_pose_intensity = self.env._get_hand_shape(agent) #TODO: ここで_get_hand_shapeメソッド呼び出すくらいなら，base.pyで呼び出したときにagentの辞書に保存しておけばいいのでは？
        # 掴んでいる対象を取得
        held_object = agent.get("held_obj")
        grab_flag = held_object is not None
        # 触っている対象を取得
        # 触れるの定義は，どうする？: 掴める状態
        touched_object = agent.get("graspable_obj")
        touch_flag = touched_object is not None
        # 手の対象
        #それぞれは，対象と考えられるオブジェクトとその強さの辞書またはベクトル
        trackable_targets = self.trackable_targets
        target_vector = np.zeros(len(trackable_targets), dtype=float)
        target_accuracy = 0.0
        
        if grab_flag:
            if held_object in trackable_targets:
                target_vector[trackable_targets.index(held_object)] = 1.0
            hand_targets = target_vector
            target_accuracy = 1.0
        elif touched_object is not None:
            #触っているオブジェクトは一つに絞れるので，これは指差しとわけた
            if touched_object in trackable_targets:
                target_vector[trackable_targets.index(touched_object)] = 1.0
            hand_targets = target_vector
        elif hand_pose == "open":
            hand_targets = self.get_reach_target(agent)
        elif hand_pose == "pointing":
            hand_targets = self.get_pointing_target(agent)
        else:
            hand_targets = self.get_arm_target(agent) #腕の伸ばした先(名前は仮)
        
        return (hand_pose, hand_pose_intensity), hand_targets, grab_flag, target_accuracy, touch_flag

    def get_gesture_signals(self, ignored_targets: Optional[List[int]] = None) -> Tuple[Dict[str, float], Dict[int, float]]:
        """
        手の状態の履歴から，手のシグナルを数値化する
        返り値は手のシグナルをまとめた辞書と対象の辞書
        """
        if len(self.hand_poses) == 0:
            return {
                "point_intensity": 0.0,
                "open_intensity": 0.0,
                "hand_target_stability": 0.0,
                "holding": 0.0,
                "touching": 0.0,
                "hand_open_close_periodicity": 0.0,
                "hand_height_margin": 0.0,
            }, {}

        # 直近の手の形状と強度
        pose_tuple = self.hand_poses[-1]
        pose_type = pose_tuple[0] if isinstance(pose_tuple, tuple) else pose_tuple
        pose_power = pose_tuple[1] if isinstance(pose_tuple, tuple) else 1.0
        
        point_intensity = float(pose_power) if pose_type == "pointing" else 0.0
        open_intensity = float(pose_power) if pose_type == "open" else 0.0

        # 指差しのターゲットの安定性 (直近のターゲット分布の平均分布のエントロピー)
        target_stability = 0.0
        avg_targets = self.get_average_hand_target_distribution()
        
        if len(avg_targets) > 0:
            avg_targets = avg_targets.copy()
            if ignored_targets:
                for i, tgt_id in enumerate(self.trackable_targets):
                    if tgt_id in ignored_targets and i < len(avg_targets):
                        avg_targets[i] = 0.0
                        
        sum_targets = np.sum(avg_targets)
        if len(avg_targets) > 0 and sum_targets > 0:
            probs = avg_targets / sum_targets
            probs = probs[probs > 0]  # 0の要素を除外
            entropy = -np.sum(probs * np.log(probs))
            
            if len(avg_targets) > 1:
                max_entropy = np.log(len(avg_targets))
                target_stability = float(np.clip(1.0 - (entropy / max_entropy), 0.0, 1.0))
            else:
                target_stability = 1.0

        # 物を持っているか
        holding = 1.0 if self.grab_flags[-1] else 0.0
        # 触っているか（開き手で掴める距離。§6-D2 親手がかり用）
        touching = 1.0 if (len(self.touch_flags) > 0 and self.touch_flags[-1]) else 0.0

        return {
            "point_intensity": point_intensity,
            "open_intensity": open_intensity,
            "hand_target_stability": target_stability,
            "holding": holding,
            "touching": touching,
            "hand_open_close_periodicity": self.compute_hand_open_close_periodicity(),
            "hand_height_margin": self.compute_hand_height_margin(),
        }, avg_targets

    # ---------------- 手の周期性シグナル（gesture 用） ----------------

    def _update_hand_periodicity_histories(self) -> None:
        """手の関節角・姿勢ラベル履歴を更新する（gesture モダリティ）。"""
        if len(self.hand_poses) > 0:
            pose_entry = self.hand_poses[-1]
            if isinstance(pose_entry, tuple):
                self.hand_pose_labels.append(pose_entry[0])
            else:
                self.hand_pose_labels.append(str(pose_entry))

        if not hasattr(self.env, "get_joint_angle"):
            return

        agent = self.env.infant
        try:
            if agent.get("index_joint") is not None:
                self.index_joint_history.append(
                    float(self.env.get_joint_angle(agent, "index_joint"))
                )
            if agent.get("thumb_joint") is not None:
                self.thumb_joint_history.append(
                    float(self.env.get_joint_angle(agent, "thumb_joint"))
                )
        except Exception:
            pass

    def _compute_angle_periodicity(
        self,
        history: deque,
        direction: str = "both",
        vmax: float = 0.15,
    ) -> float:
        """関節角履歴から往復・周期性 [0, 1] を算出する。

        direction:
            "both"     … 双方向の往復（手の開閉など）
            "positive" … 開く方向成分（index 角の増加）
            "negative" … 閉じる方向成分（index 角の減少）
        """
        if len(history) < 5:
            return 0.0

        values = np.array(history, dtype=np.float32)
        velocities = np.diff(values)  # rad/step
        if len(velocities) == 0:
            return 0.0

        periodicity = 0.0
        if len(velocities) >= 4:
            zero_crossings = len(np.where(np.diff(np.sign(velocities)))[0])
            periodicity = float(np.clip(zero_crossings / 4.0, 0.0, 1.0))
        if periodicity <= 0.0:
            return 0.0

        if direction == "positive":
            movement_strength = float(np.clip(velocities[-1] / vmax, 0.0, 1.0))
            if movement_strength < 0.1:
                movement_strength = float(np.clip(np.max(velocities) / vmax, 0.0, 1.0))
        elif direction == "negative":
            movement_strength = float(np.clip(-velocities[-1] / vmax, 0.0, 1.0))
            if movement_strength < 0.1:
                movement_strength = float(np.clip(-np.min(velocities) / vmax, 0.0, 1.0))
        else:
            movement_strength = float(np.clip(abs(velocities[-1]) / vmax, 0.0, 1.0))

        return float(np.clip(periodicity * movement_strength, 0.0, 1.0))

    def compute_hand_open_close_periodicity(self) -> float:
        """手の開閉（往復）の周期性 [0, 1]。開閉開閉と閉開閉開は同一リズムとして扱う。"""
        joint_score = self._compute_angle_periodicity(
            self.index_joint_history, direction="both"
        )
        pose_score = self._compute_pose_alternation_periodicity(
            open_labels={"open", "pointing"},
            closed_labels={"closed", "other"},
        )
        return float(np.clip(max(joint_score, pose_score), 0.0, 1.0))

    def _compute_pose_alternation_periodicity(
        self,
        open_labels: set,
        closed_labels: set,
    ) -> float:
        """姿勢ラベルの open↔closed 交替から開閉リズムを推定する。"""
        if len(self.hand_pose_labels) < 5:
            return 0.0

        labels = list(self.hand_pose_labels)[-self.periodicity_window :]
        alternations = 0
        for i in range(1, len(labels)):
            prev_open = labels[i - 1] in open_labels
            prev_closed = labels[i - 1] in closed_labels
            curr_open = labels[i] in open_labels
            curr_closed = labels[i] in closed_labels
            if (prev_open and curr_closed) or (prev_closed and curr_open):
                alternations += 1

        return float(np.clip(alternations / 3.0, 0.0, 1.0))

    def _get_hand_world_z(self, agent) -> float:
        """手 hand_link の world Z [m] を返す。"""
        try:
            link_state = p.getLinkState(
                agent["body"],
                agent["hand_link"],
                computeForwardKinematics=True,
                physicsClientId=self.env.client_id,
            )
            return float(link_state[4][2])
        except Exception:
            if len(self.hand_world_z) > 0:
                return float(self.hand_world_z[-1])
            return 0.0

    def _ensure_infant_eye_height(self) -> float:
        """エピソード中固定の幼児目線高さ（eyes_link world Z）を 1 回だけ確定する。"""
        if self.infant_eye_height is not None:
            return self.infant_eye_height
        infant = getattr(self.env, "infant", None)
        if infant is None:
            self.infant_eye_height = DEFAULT_EYE_HEIGHT
            return self.infant_eye_height
        try:
            eyes_state = p.getLinkState(
                infant["body"],
                infant["eyes_link"],
                computeForwardKinematics=True,
                physicsClientId=self.env.client_id,
            )
            self.infant_eye_height = float(eyes_state[4][2])
        except Exception:
            self.infant_eye_height = DEFAULT_EYE_HEIGHT
        return self.infant_eye_height

    def compute_hand_height_margin(self) -> float:
        """手が目線帯内に留まっているヒット率 [0, 1]。窓が HAND_HEIGHT_WINDOW 未満なら 0。"""
        if len(self.hand_world_z) < HAND_HEIGHT_WINDOW:
            return 0.0
        window = list(self.hand_world_z)[-HAND_HEIGHT_WINDOW:]
        z_eyes = self._ensure_infant_eye_height()
        hits = sum(1 for z in window if z > z_eyes - HAND_HEIGHT_TOL)
        return float(hits) / float(HAND_HEIGHT_WINDOW)

    def get_reach_target(self, agent) -> np.ndarray:
        """
        腕を伸ばした先に何があるかを取得する
        """
        try:
            hand_pose = self.env._get_palm_pose(agent)
            palm_center, palm_fwd = hand_pose
            
            ray_from = palm_center
            return self._get_targets_by_cone_raycast(agent, ray_from, palm_fwd)
            
        except Exception as e:
            # エラー時は空のベクトルを返す
            print(f"Warning: get_reach_target processing failed: {e}")
            trackable_targets = self.trackable_targets
            return np.zeros(len(trackable_targets), dtype=float)
    
    def get_pointing_target(self, agent) -> np.ndarray:
        """指差しの対象オブジェクトを取得する"""
        try:
            index_pos, point_dir = self.env._agent_index_pose(agent)
            tip_pos = index_pos + point_dir * agent["index_link_size"]
            
            ray_from = tip_pos
            return self._get_targets_by_cone_raycast(agent, ray_from, point_dir)
            
        except Exception as e:
            # エラー時は空のベクトルを返す
            print(f"Warning: get_pointing_target processing failed: {e}")
            trackable_targets = self.trackable_targets
            return np.zeros(len(trackable_targets), dtype=float)
    
    def lower_arm_pose(self, agent):
        """前腕の向きと位置を取得する"""
        link_state = p.getLinkState(agent["body"], agent["lower_arm_link"], physicsClientId=self.env.client_id)
        lower_arm_pos = np.array(link_state[4])
        lower_arm_orn = link_state[5]
        R = np.array(p.getMatrixFromQuaternion(lower_arm_orn)).reshape(3, 3)
        lower_arm_fwd = R @ np.array([0.0, 0.0, -1.0])
        arm_tip = lower_arm_pos + lower_arm_fwd * agent["lower_arm_length"]
        return arm_tip, lower_arm_fwd
    
    def get_arm_target(self, agent: Dict[str, Any]) -> np.ndarray:
        """
        腕の向いているオブジェクトを検知する．
        幼児のジェスチャーが十分に発達していない間，これがメインの幼児のジェスチャー信号になる
        """
        try:
            arm_tip, arm_fwd = self.lower_arm_pose(agent)
            
            # 自分の手を検知してしまうのを防ぐため、起点を少し遠くに設定
            ray_from = arm_tip + arm_fwd * 0.07
            return self._get_targets_by_cone_raycast(agent, ray_from, arm_fwd)
            
        except Exception as e:
            print(f"Warning: get_arm_target processing failed: {e}")
            trackable_targets = self.trackable_targets
            return np.zeros(len(trackable_targets), dtype=float)

    # ---------------- 2. 視線・顔のシグナル (Gaze & Face) ----------------

    def get_gaze_target(self) -> Tuple[np.ndarray, bool, bool]:
        """
        現在の視線の先にあるオブジェクトおよび，その正確性を取得する
        レイキャストを視線ベクトルを中心とする円錐状に複数飛ばす手法
        返り値は (対象ベクトル, 親の目ヒット, 親の顔ヒット)
        """
        # 視線の起点(両目の中点)と方向ベクトル(正面方向)を取得
        eye_pos, forward_vec = self.env._agent_eye_pose(self.env.infant)
        # 視線の起点（少し顔の前に出すことで自身の衝突を避ける）
        ray_from = eye_pos + forward_vec * GAZE_RAY_ORIGIN_OFFSET
        
        return self._get_targets_by_cone_raycast(self.env.infant, ray_from, forward_vec, parent_eye=True)
    
    def get_gaze_signals(self, ignored_targets: Optional[List[int]] = None) -> Tuple[Dict[str, float], Dict[int, float]]:
        """
        幼児の視線のシグナルを数値化する
        返り値は視線のシグナルをまとめた辞書と対象の辞書
        TODO: 幼児の視線交互をどう扱うか
        """
        if len(self.gaze_targets) == 0:
            return {
                "gaze_target_stability": 0.0,
                "looking_at_parent_eye": 0.0,
                "looking_at_parent_face": 0.0,
                "instant_parent_face_hit": 0.0,
            }, {}

        target_stability = 0.0
        avg_targets = self.get_average_gaze_target_distribution()
        
        if len(avg_targets) > 0:
            avg_targets = avg_targets.copy()
            if ignored_targets:
                for i, tgt_id in enumerate(self.trackable_targets):
                    if tgt_id in ignored_targets and i < len(avg_targets):
                        avg_targets[i] = 0.0
                        
        sum_targets = np.sum(avg_targets)
        if len(avg_targets) > 0 and sum_targets > 0:
            probs = avg_targets / sum_targets
            probs = probs[probs > 0]  # 0の要素を除外
            entropy = -np.sum(probs * np.log(probs))
            
            if len(avg_targets) > 1:
                max_entropy = np.log(len(avg_targets))
                target_stability = float(np.clip(1.0 - (entropy / max_entropy), 0.0, 1.0))
            else:
                target_stability = 1.0

        looking_at_parent_eye_ratio = float(np.mean(self.looking_at_parent_eyes)) if len(self.looking_at_parent_eyes) > 0 else 0.0
        looking_at_parent_face_ratio = (
            float(np.mean(self.looking_at_parent_faces))
            if len(self.looking_at_parent_faces) > 0
            else 0.0
        )

        return {
            "gaze_target_stability": target_stability,
            "looking_at_parent_eye": looking_at_parent_eye_ratio,
            "looking_at_parent_face": looking_at_parent_face_ratio,
            "instant_parent_face_hit": 1.0 if self._instant_parent_face_hit else 0.0,
        }, avg_targets

    # ---------------- 3. 表情 (Facial Expression) ----------------
    def get_face_expression(self) -> float:
        """
        幼児の表情を数値化する
        幼児の表情の種類: 0: Calm, 1: Excitement, 2: Distress, 3: Whine, 4: Refusal, 5: Sleepy
        表情の正確さ: [0.0, 1.0]
        """
        #TODO: 実装(現在はダミー)
        face_expression = 0
        face_expression_accuracy = 0.0
        return face_expression, face_expression_accuracy

    def get_smile_intensity(self) -> float:
        if hasattr(self.env, "get_joint_angle"):
            lip_l = self.env.get_joint_angle(self.env.infant, "left_lip_joint")
            return float(np.clip(lip_l / 0.5, 0.0, 1.0))
        return 0.0

    def get_distress_intensity(self) -> float:
        if hasattr(self.env, "get_joint_angle"):
            lip_l = self.env.get_joint_angle(self.env.infant, "left_lip_joint")
            return float(np.clip(-lip_l / 0.5, 0.0, 1.0))
        return 0.0

    def get_surprise_intensity(self) -> float:
        if hasattr(self.env, "get_joint_angle"):
            jaw_opening = abs(self.env.get_joint_angle(self.env.infant, "left_lip_joint"))
            return float(np.clip(jaw_opening / 1.0, 0.0, 1.0))
        return 0.0

    def _compute_sustained_turn_intensity(
        self,
        history: deque,
        action_threshold: float = 0.3,
        sign_ratio: float = 0.8,
        window: Optional[int] = None,
    ) -> float:
        """行動履歴から一方向への回転持続強度 [0.0, 1.0] を算出する。"""
        if window is None:
            window = max(15, self.history_length // 2)
        if len(history) < 5:
            return 0.0
        values = np.array(list(history)[-window:], dtype=np.float32)
        active = values[np.abs(values) > action_threshold]
        if len(active) < 5:
            return 0.0
        pos_ratio = float(np.mean(active > 0))
        neg_ratio = float(np.mean(active < 0))
        dominant_ratio = max(pos_ratio, neg_ratio)
        if dominant_ratio < sign_ratio:
            return 0.0
        return float(np.clip(float(np.mean(np.abs(active))), 0.0, 1.0))

    def get_head_turn_away_intensity(self) -> float:
        """幼児の頭回転行動が直近で同符号に偏っている強度 [0.0, 1.0]"""
        horizontal = self._compute_sustained_turn_intensity(self.infant_horizontal_turn_actions)
        tilt = self._compute_sustained_turn_intensity(self.infant_head_tilt_actions)
        return float(max(horizontal, tilt))

    def get_sudden_gaze_shift(self) -> float:
        """視線の急激な逸らしの強度 [0.0, 1.0]"""
        # 短い履歴だと分布推定が不安定で拒否誤検知しやすい（AR=1 で顕在化）
        if len(self.gaze_targets) < max(15, self.history_length // 2):
            return 0.0
        try:
            half = len(self.gaze_targets) // 2
            early_avg = np.mean(list(self.gaze_targets)[:half], axis=0)
            late_avg = np.mean(list(self.gaze_targets)[half:], axis=0)
            if len(early_avg) == 0 or len(late_avg) == 0:
                return 0.0
            
            # L1ノルムで分布の変化量を計算
            diff = np.linalg.norm(early_avg - late_avg, ord=1)
            return float(np.clip(diff / 1.0, 0.0, 1.0))
        except Exception:
            return 0.0

    def get_gaze_shift_dict(self) -> Dict[int, float]:
        """各ターゲットに対する視線確率の変化量（後半平均 - 前半平均）を辞書で返す。
        正の値は「急激に見た」、負の値は「急激に逸らした」ことを意味する。"""
        if len(self.gaze_targets) < max(15, self.history_length // 2):
            return {}
        try:
            half = len(self.gaze_targets) // 2
            early_avg = np.mean(list(self.gaze_targets)[:half], axis=0)
            late_avg = np.mean(list(self.gaze_targets)[half:], axis=0)
            diff_vec = late_avg - early_avg
            
            trackable_targets = self.trackable_targets
            shift_dict = {}
            for i, obj_id in enumerate(trackable_targets):
                if i < len(diff_vec):
                    shift_dict[obj_id] = float(diff_vec[i])
            # 直近半窓での視線質量（拒否誤検知抑制用）
            late_mass = {}
            for i, obj_id in enumerate(trackable_targets):
                if i < len(late_avg):
                    late_mass[obj_id] = float(late_avg[i])
            self._gaze_late_mass = late_mass
            return shift_dict
        except Exception:
            return {}

    def get_negative_emotion_surge(self) -> float:
        """ネガティブ情動（泣き声や嫌がる表情）の急激な立ち上がり強度 [0.0, 1.0]"""
        # TODO: 表情や声の情動の履歴が本格実装されたら、そこから急変（微分値）を計算する
        return 0.0

    def get_rejection_signals(self) -> Dict[str, float]:
        """拒否シグナル（急激な変化）を取得する"""
        signals = {}
        if self.active_signals is None or "head_turn_away_intensity" in self.active_signals:
            signals["head_turn_away_intensity"] = self.get_head_turn_away_intensity()
        if self.active_signals is None or "sudden_gaze_shift" in self.active_signals:
            signals["sudden_gaze_shift"] = self.get_sudden_gaze_shift()
        if self.active_signals is None or "gaze_shift_dict" in self.active_signals:
            signals["gaze_shift_dict"] = self.get_gaze_shift_dict()
            signals["gaze_late_mass"] = getattr(self, "_gaze_late_mass", {})
        if self.active_signals is None or "negative_emotion_surge" in self.active_signals:
            signals["negative_emotion_surge"] = self.get_negative_emotion_surge()
        
        return signals

    # ------------------------------------共通関数---------------------------------------
    def _classify_gaze_hit(
        self, obj_id: int, link_id: int, accuracy: float
    ) -> Optional[Tuple[int, float]]:
        """
        J1: レイヒットを trackable ターゲット id に分類する。
        把持中の hold_gaze_links は held_obj へ振替。face_links は親 body スロットへ。
        """
        if accuracy <= 0.0:
            return None
        parent = getattr(self.env, "parent", None)
        infant = getattr(self.env, "infant", None)
        if hasattr(self.env, "all_objects") and obj_id in self.env.all_objects:
            return (obj_id, accuracy)
        if parent is not None and obj_id == parent.get("body"):
            face_links = {lid for lid in parent.get("face_links", ()) if lid is not None}
            hold_links = {lid for lid in (parent.get("hold_gaze_links") or ()) if lid is not None}
            if link_id in face_links:
                return (parent["body"], accuracy)
            held = parent.get("held_obj")
            if held is not None and link_id in hold_links:
                return (held, accuracy)
            return None
        if infant is not None and obj_id == infant.get("body"):
            hold_links = {lid for lid in (infant.get("hold_gaze_links") or ()) if lid is not None}
            held = infant.get("held_obj")
            if held is not None and link_id in hold_links:
                return (held, accuracy)
            return None
        if obj_id in self.trackable_targets:
            return (obj_id, accuracy)
        return None

    def _get_targets_by_cone_raycast(self, agent: Dict[str, Any], ray_from: np.ndarray, forward_vec: np.ndarray, ray_length: Optional[float] = None, parent_eye: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, bool]]:
        """
        起点(ray_from)と方向ベクトル(forward_vec)から円錐状にバッチレイキャストを行い、
        対象オブジェクトと正確性の固定長ベクトルを返す内部メソッド。ベクトルの各要素は各オブジェクトに対応して，それぞれの確率を格納する．
        各要素のインデックスは self.get_trackable_targets() の要素に対応する。
        parent_eye=Trueの場合、(対象ベクトル, 目ヒット, 顔ヒット) を返す。
        顔ヒットは target_agent の face_links（目・頭・唇）のいずれか。
        """
        try:
            if ray_length is None:
                ray_length = self.cone_ray_length
            rayToPositions = []
            rayAngles = []
            
            # 1. 中心点のレイ
            rayToPositions.append(ray_from + forward_vec * ray_length)
            rayAngles.append(0.0)
            
            # 2. 周囲のレイを円錐状に飛ばすための基底ベクトル作成
            # up_vec: Z軸(0,0,1)を使って外積で右ベクトルを生成
            up_vec = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(forward_vec, up_vec)) > 0.99:
                 up_vec = np.array([1.0, 0.0, 0.0])
            right_vec = np.cross(forward_vec, up_vec)
            right_vec /= np.linalg.norm(right_vec)
            up_vec = np.cross(right_vec, forward_vec)
            up_vec /= np.linalg.norm(up_vec)
            
            # 半径(角度)の設定：中心から3度と8度ずらした2つの円周上に配置
            rings = list(self.cone_rings)
            
            for ring in rings:
                angle_rad = math.radians(ring["angle_deg"])
                num_rays = ring["num_rays"]
                radius = math.tan(angle_rad) * ray_length
                for i in range(num_rays):
                    phi = 2.0 * math.pi * i / num_rays
                    # 円周上の点のオフセット
                    offset = radius * (math.cos(phi) * right_vec + math.sin(phi) * up_vec)
                    target_pos = ray_from + forward_vec * ray_length + offset
                    rayToPositions.append(target_pos)
                    rayAngles.append(ring["angle_deg"])
            
            # 3. バッチ・レイキャスト実行
            # self.env.client_id を用いてPyBulletのC++側で一括計算
            ray_results = p.rayTestBatch(
                [ray_from] * len(rayToPositions), 
                rayToPositions, 
                physicsClientId=self.env.client_id
            )
            
            # 4. ヒット結果の解析（J1: link-aware。幼児 body も hold リンクなら held_obj へ振替）
            ignore_ids = set()
            if hasattr(self.env, "planeId"):
                ignore_ids.add(self.env.planeId)
            elif hasattr(self.env, "floor"):
                ignore_ids.add(self.env.floor)
            else:
                ignore_ids.add(0) # デフォルトで0を床とみなす場合
                
            hit_candidates_dict = {} # trackable_target_id: max_accuracy
            max_angle = rings[-1]["angle_deg"]
            
            hit_parent_eye = False
            hit_parent_face = False
            target_agent = None
            face_links = set()
            if parent_eye:
                if hasattr(self.env, "parent") and self.env.parent is not None and hasattr(self.env, "infant") and self.env.infant is not None:
                    if agent["body"] == self.env.infant["body"]:
                        target_agent = self.env.parent
                    elif agent["body"] == self.env.parent["body"]:
                        target_agent = self.env.infant
                if target_agent is not None:
                    face_links = {
                        lid for lid in target_agent.get("face_links", ()) if lid is not None
                    }
            
            for i, result in enumerate(ray_results):
                obj_id = result[0]
                link_id = result[1]
                
                if target_agent is not None and obj_id == target_agent["body"]:
                    if link_id == target_agent["eyes_link"]:
                        hit_parent_eye = True
                    if link_id in face_links:
                        hit_parent_face = True
                    
                if obj_id < 0 or obj_id in ignore_ids:
                    continue
                angle_deg = rayAngles[i]
                accuracy = cone_ray_accuracy(
                    angle_deg, max_angle, min_accuracy=self.cone_min_accuracy
                )
                classified = self._classify_gaze_hit(obj_id, link_id, accuracy)
                if classified is None:
                    continue
                tid, acc = classified
                if tid not in hit_candidates_dict or hit_candidates_dict[tid] < acc:
                    hit_candidates_dict[tid] = acc #足し算にせず，最大値を更新することで，真ん中を優先できる
                        
            # 辞書からベクトルへ変換
            trackable_targets = self.trackable_targets
            target_vector = np.zeros(len(trackable_targets), dtype=float)
            for i, obj_id in enumerate(trackable_targets):
                target_vector[i] = hit_candidates_dict.get(obj_id, 0.0)
                
            if parent_eye:
                return target_vector, hit_parent_eye, hit_parent_face
            return target_vector
        
        except Exception as e:
            # エラー時は空のベクトルを返す
            print(f"Warning: _get_targets_by_cone_raycast processing failed: {e}")
            trackable_targets = self.trackable_targets
            empty_vector = np.zeros(len(trackable_targets), dtype=float)
            if parent_eye:
                return empty_vector, False, False
            return empty_vector
    
    def __getattr__(self, name):
        """env へ属性を委譲するフォールバック。Wrapper自身に無い属性を下位環境から引き出す。"""
        # 無限再帰を防ぐ
        if name == "env":
            return super().__getattribute__(name)
        return getattr(self.env, name)

    
