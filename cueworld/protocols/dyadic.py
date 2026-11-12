import random
import numpy as np
import pybullet as p
from typing import Dict, Any, Optional, List, Tuple
from .base import ATTENTION_WAIT_STEPS, BaseProtocol
from cueworld.intent import DEFAULT_VOICE_VOLUME_THRESHOLD
from cueworld.infant_signal import HAND_HEIGHT_TOL

VOICE_CHANNEL = "voice"
MOTOR_CHANNELS = ("arm_raise", "hand_open_close")
# 掲げ IK が必要な腕チャネル（voice は到達待ちしない）
ARM_GESTURE_CHANNELS = ("arm_raise", "hand_open_close")
SALIENCE_GLOBAL_THRESH = 0.2
# arm_raise 成立（ヒット率 8/10）。幼児始動 ON・engagement・match で共通
ARM_RAISE_HIT_RATE_MIN = 0.8
# 切り分けテスト: 帯の下端までの不足 h は HAND_HEIGHT_TOL(0.4 m) を 1、成功瞬間を 2
DENSE_SUCCESS_BONUS = 2.0

# 応答 match / チャネル ON / engagement の強度しきい値（チャネル別・絶対値）。
# voice / hand_open_close の 0.2 は仮値。強度違いデモを目視後にここを書き換える。
MATCH_QUANTITY_THRESHOLDS = {
    VOICE_CHANNEL: 0.2,
    "arm_raise": ARM_RAISE_HIT_RATE_MIN,
    "hand_open_close": 0.2,
}

MOTOR_CHANNEL_CONFIG = {
    "arm_raise": {"duration_steps": 90, "parent_action": "wave_arm"},
    "hand_open_close": {"duration_steps": 60, "parent_action": "animate_hand_open_close"},
}
MOTOR_CYCLE_STEPS = 30

# 喃語エコー: 1 音 6 step（単語と同じ）。30 step の引き伸ばしは使わない。
PHONEME_ECHO_STEPS = 6
# 親が先に出す声の見本（幼児の列がまだ無いとき）。
PARENT_INIT_CHUNK = ["p0", "p1", "p0", "p1"]
# セッション内の最大往復数。
MAX_BABBLE_ROUNDS = 5

# モータ持続: 真似開始からの壁時計上限 / 外れ後の再開待ち / 顔離脱
MOTOR_MAX_STEPS = 90
MOTOR_IDLE_TIMEOUT = ATTENTION_WAIT_STEPS  # 50 step
FACE_AWAY_TIMEOUT = 30

_CHANNEL_ORDER = (VOICE_CHANNEL, "arm_raise", "hand_open_close")
_CHANNEL_PEAK_KEYS = _CHANNEL_ORDER


class DyadicProtocol(BaseProtocol):
    """音声・身体模倣プロトコル（幼児始動・親始動の両方を同一クラスで扱う）"""

    def __init__(self, env: Any, imitating_type: str = "voice", rejection_check: bool = True):
        super().__init__("DyadicProtocol", env, rejection_check=rejection_check)

        self.function_type = "declarative"
        self.relation_type = "dyadic"
        self.is_urgent = False
        self.response_type = True
        self.request_type = True
        self.solo = False
        self.has_target = False
        self.level = 0
        # バッチ E: 腕チャネルの成功1回の典型 T。値は仮。
        self.typical_steps = 105

        self.state_proximities = {
            "init": 0.0,
            "waiting_infant": 0.3,
            "performing_parent_cue": 0.5,
            "echoing": 0.7,
            "waiting_babble": 0.7,
            "imitating": 0.7,
            "completed": 1.0,
        }

        self.state_timer = 0
        self.imitating_type = imitating_type
        self.source_phoneme: Optional[str] = None
        self.voice_volume_threshold = float(
            getattr(env, "voice_volume_threshold", DEFAULT_VOICE_VOLUME_THRESHOLD)
        )
        self.babbling_ratio_threshold = 0.5

        self.active_motor_channels: List[str] = []
        self.motor_timers: Dict[str, int] = {}
        self.active_channels: List[str] = []
        self.salient_channels: List[str] = []
        # D2: 親 cue 固定（context["parent_channels"]）。None ならランダム sample
        self.parent_channels_override: Optional[List[str]] = None

        # D2: 応答 match 注釈（実行時の praise / 状態分岐は変えない）
        self.channel_peak_scores: Dict[str, float] = {}
        self.channel_on: Dict[str, int] = {}
        self.response_matched: int = 0
        self.response_peak_active: bool = False
        self.time_to_engagement: Optional[int] = None
        self._response_wait_elapsed: int = 0
        self._infant_has_responded: bool = False

        # 喃語セッション（babbling_bout.md）
        self.round_count: int = 0
        self.max_rounds: int = MAX_BABBLE_ROUNDS
        self.current_babble_chunk: Optional[List[str]] = None

        # 声・モータの並走レーン
        self.voice_active: bool = False
        self.motor_active: bool = False
        self.motor_wall_timer: int = 0
        self.motor_idle_timer: int = 0
        self._motor_above_now: bool = False

        # 顔離脱タイムアウト（face_links。待ちのあいだだけ進行）
        self.face_away_timer: int = 0

        # 掲げジェスチャ（腕チャネル時）。進入時に固定、退出で破棄
        self.gesture_pos = None
        self.gesture_reached = False
        self.gesture_wait_timer = 0
        self.gesture_forward_dist = 0.35  # 親 torso 前方距離（演出ノブ）
        self.gesture_reach_tol = 0.26
        # T2 較正: seeds 42-44、T=ceil(max(中央値,平均))、timeout=2T
        # 前進系 gesture 到達: T_reach=10 → 20 steps ≒ 2.0s @ action_dt=0.1
        self.gesture_reach_timeout_steps = 20
        # 失敗系（幼児待ち）。バッチ H: 5.0s、リトライは 1 回
        self.waiting_infant_timeout_steps = ATTENTION_WAIT_STEPS
        # バッチ H: 待ちは共同注意と同じ 5 秒に揃える（喃語待ちも同じ）。
        self.waiting_babble_timeout_steps = ATTENTION_WAIT_STEPS
        # 旧名の別名（テスト互換）。waiting_babble と同じ。
        self.waiting_infant_response_timeout_steps = ATTENTION_WAIT_STEPS
        self.motor_max_steps = MOTOR_MAX_STEPS
        self.motor_idle_timeout_steps = MOTOR_IDLE_TIMEOUT
        self.face_away_timeout_steps = FACE_AWAY_TIMEOUT

        # 切り分けテスト用スイッチ（start の context で上書き）
        self.close_on_raise = False
        self._dense_active = False
        self._dense_prev_h = None
        self._dense_done = False
        self._dense_success_pending = False

    @staticmethod
    def channel_tau(name: str) -> float:
        return MATCH_QUANTITY_THRESHOLDS.get(name, SALIENCE_GLOBAL_THRESH)

    @staticmethod
    def compute_channel_salience(infant_signals: Dict[str, Any]) -> Dict[str, float]:
        """
        全チャネルの salience スコアを返す（閾値フィルタなし）。
        hand_open_close は arm_raise が成立しているときだけ非ゼロ。
        """
        # 保持中の喃語列があれば 1、無ければ 0（babbling_bout.md）。
        voice_salience = float(infant_signals.get("babbling_salience", 0.0))
        arm_raise = float(infant_signals.get("hand_height_margin", 0.0))
        hand_open_close = (
            float(infant_signals.get("hand_open_close_periodicity", 0.0))
            if arm_raise >= ARM_RAISE_HIT_RATE_MIN
            else 0.0
        )

        return {
            VOICE_CHANNEL: float(np.clip(voice_salience, 0.0, 1.0)),
            "arm_raise": float(np.clip(arm_raise, 0.0, 1.0)),
            "hand_open_close": float(np.clip(hand_open_close, 0.0, 1.0)),
        }

    @staticmethod
    def get_salient_channels(
        infant_signals: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        """
        模倣対象チャネルの salience を計算し、各自の τ 以上のものを固定順で全部返す。
        クロスチャネルの順位付けはしない。infant_signals は get_normalized_signals() の出力を想定する。
        """
        salience = DyadicProtocol.compute_channel_salience(infant_signals)
        selected = []
        for name in _CHANNEL_ORDER:
            score = float(salience.get(name, 0.0))
            if score >= DyadicProtocol.channel_tau(name):
                selected.append((name, score))
        return selected

    def start(self, context: Dict[str, Any]):
        super().start(context)
        self._reset_motor_state()
        self._reset_match_state()
        self._infant_has_responded = False
        self.active_channels = []
        self.salient_channels = []
        self.source_phoneme = None
        self.round_count = 0
        self.current_babble_chunk = None
        self.voice_active = False
        self.motor_active = False
        self.motor_wall_timer = 0
        self.motor_idle_timer = 0
        self._motor_above_now = False
        self.face_away_timer = 0
        override = context.get("parent_channels")
        self.parent_channels_override = list(override) if override is not None else None
        self.close_on_raise = bool(context.get("close_on_raise", False))
        self._dense_active = False
        self._dense_prev_h = None
        self._dense_done = False
        self._dense_success_pending = False

        babbling_chunk = context.get("babbling_chunk")
        if babbling_chunk is not None:
            babbling_chunk = list(babbling_chunk)

        if self.infant_initiative:
            if babbling_chunk:
                self._begin_echo(babbling_chunk)
            else:
                self.change_state("waiting_infant")
        else:
            self._begin_parent_cue(self._parent_cue_channels())

    def _reset_motor_state(self) -> None:
        self.active_motor_channels = []
        self.motor_timers = {}
        self.gesture_pos = None
        self.gesture_reached = False
        self.gesture_wait_timer = 0

    def _reset_match_state(self) -> None:
        """D2: 応答 match 注釈を初期化する。"""
        self.channel_peak_scores = {name: 0.0 for name in _CHANNEL_PEAK_KEYS}
        self.channel_on = {name: 0 for name in _CHANNEL_PEAK_KEYS}
        self.response_matched = 0
        self.response_peak_active = False
        self.time_to_engagement = None
        self._response_wait_elapsed = 0
        if isinstance(self.debug_info, dict):
            self.debug_info.pop("response_match", None)

    def _needs_gesture_raise(self) -> bool:
        return any(ch in ARM_GESTURE_CHANNELS for ch in self.active_motor_channels)

    def _compute_gesture_pos(self) -> np.ndarray:
        """親 torso 前方・幼児目線高さの固定点を計算する（二項の見せ位置）。"""
        parent = self.env.parent
        infant = self.env.infant
        torso_pos, fwd = self.env._agent_torso_pose(parent)
        direction = np.asarray([fwd[0], fwd[1]], dtype=np.float32)
        dnorm = float(np.linalg.norm(direction))
        if dnorm < 1e-4:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / dnorm
        xy = np.asarray(torso_pos, dtype=np.float32)[:2] + direction * float(
            self.gesture_forward_dist
        )
        eyes_state = p.getLinkState(
            infant["body"],
            infant["eyes_link"],
            computeForwardKinematics=True,
            physicsClientId=self.env.client_id,
        )
        return np.array(
            [float(xy[0]), float(xy[1]), float(eyes_state[4][2])], dtype=np.float32
        )

    def _maybe_init_gesture_pos(self) -> None:
        """腕チャネルがあるときだけ gesture_pos を進入時に固定。"""
        if self._needs_gesture_raise():
            self.gesture_pos = self._compute_gesture_pos()
            self.gesture_reached = False
            self.gesture_wait_timer = 0
        else:
            self.gesture_pos = None
            self.gesture_reached = True  # 到達待ち不要
            self.gesture_wait_timer = 0

    @staticmethod
    def _normalize_channels(channels: List[str]) -> List[str]:
        """open_close 単体は無い。常に arm_raise を先行して付ける。"""
        out: List[str] = []
        for name in channels:
            if name == "hand_open_close" and "arm_raise" not in out:
                out.append("arm_raise")
            if name not in out:
                out.append(name)
        return out

    def _sample_parent_channels(self) -> List[str]:
        # 常に 2 スロット: 50% voice+モータ1本、50% モータ 2 本。
        # open_close を引いたら arm_raise を必ず足す。raise 単独のランダムは無し。
        if random.random() < 0.5:
            channels = [VOICE_CHANNEL, random.choice(MOTOR_CHANNELS)]
        else:
            channels = list(MOTOR_CHANNELS)
        return self._normalize_channels(channels)

    def _parent_cue_channels(self) -> List[str]:
        """親 cue チャネル。override があれば固定、無ければランダム sample。"""
        if self.parent_channels_override is not None:
            return self._normalize_channels(list(self.parent_channels_override))
        return self._sample_parent_channels()

    def _queue_phoneme_echo(self, chunk: List[str]) -> None:
        """列を 1 音 PHONEME_ECHO_STEPS で 1 回だけキューする。"""
        for phoneme in chunk:
            for _ in range(PHONEME_ECHO_STEPS):
                self.utterance_queue.append(phoneme)

    def _clear_held_babble(self) -> None:
        """親が言い終わったら保持中の喃語列を消す。同じ列で次の往復は始まらない。"""
        clearer = getattr(self.env, "clear_babbling_chunk", None)
        if callable(clearer):
            clearer()

    def _begin_echo(self, chunk: List[str]) -> None:
        """幼児の喃語列を 1 回返す。モータレーン／見本のモータは消さない。"""
        self.utterance_queue.clear()
        # 持続レーン中、または親の見本モータが残っているときは腕を消さない。
        if not self.motor_active and self.state != "performing_parent_cue":
            self._reset_motor_state()
        self.current_babble_chunk = list(chunk)
        self.salient_channels = [VOICE_CHANNEL]
        self.voice_active = True
        self._queue_phoneme_echo(self.current_babble_chunk)
        self.change_state("echoing")
        self.state_timer = 0
        print(f"[DyadicProtocol] Echoing babble chunk: {self.current_babble_chunk}")

    def _queue_channels_for_imitation(
        self,
        channels: List[Tuple[str, float]],
        infant_signals: Optional[Dict[str, Any]] = None,
        phoneme: Optional[str] = None,
        babble_chunk: Optional[List[str]] = None,
        *,
        parent_init_voice: bool = False,
    ) -> None:
        names = self._normalize_channels([name for name, _score in channels])
        for name in names:
            if name == VOICE_CHANNEL:
                if parent_init_voice:
                    # 親が先に出す声: p0 p1 p0 p1 を 1 音 6 step で 1 回。
                    self._queue_phoneme_echo(PARENT_INIT_CHUNK)
                    self.source_phoneme = PARENT_INIT_CHUNK[0]
                elif babble_chunk:
                    self._queue_phoneme_echo(list(babble_chunk))
                    self.source_phoneme = babble_chunk[0]
                else:
                    selected_phoneme = (
                        phoneme
                        or self.source_phoneme
                        or "p0"
                    )
                    self.source_phoneme = selected_phoneme
                    self._queue_phoneme_echo([selected_phoneme])
            elif name in MOTOR_CHANNEL_CONFIG:
                if name not in self.active_motor_channels:
                    self.active_motor_channels.append(name)
                self.motor_timers[name] = 0

    def _motor_channels_above(
        self, infant_signals: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        ranked = self.get_salient_channels(infant_signals)
        return [(n, s) for n, s in ranked if n != VOICE_CHANNEL]

    def _motor_above_threshold(self, infant_signals: Dict[str, Any]) -> bool:
        return len(self._motor_channels_above(infant_signals)) > 0

    def _activate_motor(self, infant_signals: Dict[str, Any]) -> None:
        """モータレーンを開く。成功フラグを立て、壁時計を 0 から始める。"""
        if self.motor_active:
            return
        motor_only = self._motor_channels_above(infant_signals)
        if not motor_only:
            return
        self.motor_active = True
        self.motor_wall_timer = 0
        self.motor_idle_timer = 0
        self.session_success = True
        self._infant_has_responded = True
        self._motor_above_now = True
        # 親 cue のモータが既にあればそのまま続ける。無ければ salience から積む。
        if not self.active_motor_channels:
            self.salient_channels = self._normalize_channels(
                [name for name, _ in motor_only]
            )
            self._queue_channels_for_imitation(
                motor_only, infant_signals=infant_signals
            )
            self._maybe_init_gesture_pos()
        print(f"[DyadicProtocol] Motor lane opened: {self.active_motor_channels}")

    def _close_motor_lane(self) -> None:
        self.motor_active = False
        self._motor_above_now = False
        self.motor_idle_timer = 0
        self._reset_motor_state()
        self._try_finish_session()

    def _close_voice_lane(self) -> None:
        self.voice_active = False
        self._try_finish_session()

    def _try_finish_session(self) -> None:
        """声とモータの両レーンが閉じたらセッションを終える。"""
        if self.is_finished:
            return
        if self.voice_active or self.motor_active:
            return
        if self.session_success:
            self._finish_session(True)

    def _begin_infant_imitation_from_salience(
        self,
        ranked_channels: List[Tuple[str, float]],
        infant_signals: Optional[Dict[str, Any]] = None,
    ) -> None:
        """モータ等の salience による模倣開始（持続レーンへ）。後方互換の入口。"""
        signals = infant_signals or {}
        chunk = signals.get("babbling_chunk")
        names = [name for name, _ in ranked_channels]
        if VOICE_CHANNEL in names and chunk:
            self._begin_echo(list(chunk))
            return
        self._activate_motor(signals)

    def _begin_parent_cue(self, channels: List[str]) -> None:
        self.utterance_queue.clear()
        self._reset_motor_state()
        self._reset_match_state()
        self.active_channels = self._normalize_channels(list(channels))
        channel_tuples = [(name, 1.0) for name in self.active_channels]
        self._queue_channels_for_imitation(
            channel_tuples,
            parent_init_voice=VOICE_CHANNEL in self.active_channels,
        )
        self._maybe_init_gesture_pos()
        self.change_state("performing_parent_cue")
        self.state_timer = 0
        # 密な報酬: 見本が最初に始まった瞬間から不足 h を測る
        if not self._dense_active:
            self._dense_active = True
            self._dense_prev_h = self._compute_hand_height_deficit()
        print(f"[DyadicProtocol] Parent cue channels: {self.active_channels}")

    def _find_signal_detector(self) -> Any:
        """ラッパー鎖を辿って InfantSignalDetector（手高さ・目線の計算元）を探す。"""
        env = self.env
        seen = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            if hasattr(env, "_ensure_infant_eye_height") and hasattr(
                env, "_get_hand_world_z"
            ):
                return env
            env = getattr(env, "env", None)
        return None

    def _compute_hand_height_deficit(self) -> float:
        """帯の下端までの不足 h [m]。帯内なら 0。hand_height_margin と同じ帯。"""
        det = self._find_signal_detector()
        if det is None:
            return 0.0
        infant = getattr(det, "infant", None)
        if infant is None:
            infant = getattr(self.env, "infant", None)
        if infant is None:
            return 0.0
        z_eyes = float(det._ensure_infant_eye_height())
        z_hand = float(det._get_hand_world_z(infant))
        return float(max(0.0, z_eyes - HAND_HEIGHT_TOL - z_hand))

    def _arm_raise_above(self, infant_signals: Dict[str, Any]) -> bool:
        return float(infant_signals.get("hand_height_margin", 0.0)) >= ARM_RAISE_HIT_RATE_MIN

    def _mark_dense_raise_success(self) -> None:
        """見本中に arm_raise が初めて 0.8 に達した step。+2 を次の dense 呼び出しへ。"""
        if self._dense_active and not self._dense_done:
            self._dense_done = True
            self._dense_success_pending = True

    def dense_reward_function(self, infant_signals: Dict[str, Any]) -> float:
        """切り分けテスト用。見本開始〜掲げ成功まで、不足 h の減少分 + 成功で +2。

        h = max(0, 目線 − 0.4 − hand_link Z)。0.4 m（HAND_HEIGHT_TOL）を 1 とする。
        """
        if self._dense_success_pending:
            self._dense_success_pending = False
            return float(DENSE_SUCCESS_BONUS)
        if self._dense_done or not self._dense_active:
            return 0.0
        h = self._compute_hand_height_deficit()
        if self._dense_prev_h is None:
            self._dense_prev_h = h
            return 0.0
        delta = (float(self._dense_prev_h) - float(h)) / float(HAND_HEIGHT_TOL)
        self._dense_prev_h = h
        return float(delta)

    def _detect_infant_response(self, infant_signals: Dict[str, Any]) -> bool:
        """親始動後の幼児反応検知（engagement）。

        babbling_chunk がある、またはどれか 1 チャネルの salience が自分の τ 以上なら True。
        チャネル一致は問わない（一致は B 層 response_matched 注釈のみ）。

        旧条件を廃した理由:
        - arm_ext は肘角からの計算で静止時（elbow=0）に 1.0 となり常に真だった
        - orientation_to_parent は姿勢であって応答ではなく、デモでは常時注視で常に真だった
        - hand_movement_size > 0.25 単独も、周期性なしの瞬間動きで誤発火しうる
        """
        if infant_signals.get("babbling_chunk") is not None:
            self._infant_has_responded = True
            return True
        salience = self.compute_channel_salience(infant_signals)
        for name, score in salience.items():
            if score >= self.channel_tau(name):
                self._infant_has_responded = True
                return True
        return False

    def infant_is_engaged(self, infant_signals: Dict[str, Any]) -> bool:
        """反応した、または模倣／エコーの最中。"""
        if self.state in ("imitating", "echoing"):
            return True
        if self.motor_active or self.voice_active:
            return True
        if self._infant_has_responded or self.session_success:
            return True
        return self._detect_infant_response(infant_signals)

    def blocks_interrupt(self, infant_signals: Dict[str, Any]) -> bool:
        # 親の発話キューが残っているあいだ、または幼児が発話区間の中。
        if len(self.utterance_queue) > 0:
            return True
        if self.voice_segment.in_segment:
            return True
        return False

    def _calculate_persistence(self, infant_signals: Dict[str, Any]):
        """往復が 1 つ終わるごとに 2.7 / 5.0 から 1.0 引く。下限は 1.0。"""
        if self.typical_steps is None:
            return
        super()._calculate_persistence(infant_signals)
        decrement = 1.0 * float(self.round_count)
        self.persistence = max(1.0, float(self.persistence) - decrement)

    def _update_response_match(self, infant_signals: Dict[str, Any]) -> None:
        """チャネル別 peak を更新し、親 cue がすべて τ 以上なら matched 注釈を書く。"""
        if not self.response_peak_active:
            return
        self._response_wait_elapsed += 1
        salience = self.compute_channel_salience(infant_signals)
        for name, score in salience.items():
            prev = self.channel_peak_scores.get(name, 0.0)
            if score > prev:
                self.channel_peak_scores[name] = float(score)

        self.channel_on = {}
        for name in _CHANNEL_PEAK_KEYS:
            peak = float(self.channel_peak_scores.get(name, 0.0))
            self.channel_on[name] = int(peak >= self.channel_tau(name))

        if self.active_channels:
            self.response_matched = int(
                all(self.channel_on.get(ch, 0) for ch in self.active_channels)
            )
        else:
            self.response_matched = 0

        self.debug_info["response_match"] = {
            "channel_peak_scores": dict(self.channel_peak_scores),
            "channel_on": dict(self.channel_on),
            "response_matched": self.response_matched,
            "time_to_engagement": self.time_to_engagement,
            "active_channels": list(self.active_channels),
        }

    def _voice_active(self) -> bool:
        return len(self.utterance_queue) > 0

    def _parent_voice_demo_pending(self) -> bool:
        """親始動の声の見本がまだキューに残っているか。"""
        return (
            self.state == "performing_parent_cue"
            and VOICE_CHANNEL in self.active_channels
            and self._voice_active()
        )

    def _channels_in_progress(self) -> List[str]:
        # モータ持続中かつ閾値以上なら、duration に縛られず全部を進行中にする。
        if self.motor_active and self._motor_above_now:
            return list(self.active_motor_channels)
        active = []
        for channel in self.active_motor_channels:
            duration = MOTOR_CHANNEL_CONFIG[channel]["duration_steps"]
            if self.motor_timers.get(channel, 0) < duration:
                active.append(channel)
        return active

    def _has_active_motor(self) -> bool:
        return len(self._channels_in_progress()) > 0

    def _imitation_channels_complete(self) -> bool:
        return not self._voice_active() and not self._has_active_motor()

    def _parent_cue_complete(self) -> bool:
        """見本の声が終わり、モータは duration 完了か持続レーンへ移管済み。"""
        if self._voice_active():
            return False
        if self.motor_active:
            return True
        return not self._has_active_motor()

    def _phase(self, channel: str) -> float:
        timer = self.motor_timers.get(channel, 0)
        return float((timer % MOTOR_CYCLE_STEPS) / MOTOR_CYCLE_STEPS)

    def _update_gesture_reach(self) -> None:
        """掲げ到達を更新。タイムアウトで到達扱いにする（デッドロック回避）。"""
        if self.gesture_pos is None:
            self.gesture_reached = True
            return
        if self.gesture_reached:
            return
        err = self.env.palm_reach_error(self.env.parent, self.gesture_pos)
        if err < float(self.gesture_reach_tol):
            self.gesture_reached = True
            return
        self.gesture_wait_timer += 1
        if self.gesture_wait_timer > self.gesture_reach_timeout_steps:
            self.gesture_reached = True  # 届かなくても進める

    def _tick_motor_timers(self) -> None:
        """腕チャネルは掲げ到達後のみタイマ進行。"""
        self._update_gesture_reach()
        for channel in self._channels_in_progress():
            if channel in ARM_GESTURE_CHANNELS and not self.gesture_reached:
                continue
            self.motor_timers[channel] = self.motor_timers.get(channel, 0) + 1

    def _apply_motor_channels(self, action: np.ndarray) -> None:
        parent = self.env.parent
        in_progress = self._channels_in_progress()
        needs_raise = any(ch in ARM_GESTURE_CHANNELS for ch in in_progress)

        # 腕チャネル時は先に掲げ IK を維持
        if needs_raise:
            if self.gesture_pos is None:
                self.gesture_pos = self._compute_gesture_pos()
            self.env.hand_reach_location(parent, self.gesture_pos, action=action)

        for channel in in_progress:
            phase = self._phase(channel)
            if channel == "arm_raise":
                # 見本も真似も腕を振る（幼児の動き方への合わせは将来）。
                self.env.wave_arm(parent, action=action, phase=phase, gesture=True)
            elif channel == "hand_open_close":
                self.env.animate_hand_open_close(
                    parent, action=action, phase=phase, gesture=True
                )
        self._tick_motor_timers()

    def _apply_utterance_action(self, action: np.ndarray) -> None:
        self._emit_utterance(action)

    def _clear_imitation_queues(self) -> None:
        self.utterance_queue.clear()
        self._reset_motor_state()

    def _finish_session(self, success: bool) -> None:
        self.voice_active = False
        self.motor_active = False
        if success:
            self.change_state(self.success_state)
        else:
            self.change_state(self.failure_state)
        self.is_finished = True

    def _on_echo_complete(self) -> None:
        """1 往復完了: フラグ・往復数・保持列のクリア。"""
        self.round_count += 1
        self.session_success = True
        self._infant_has_responded = True
        self._clear_held_babble()
        self.current_babble_chunk = None
        if self.round_count >= self.max_rounds:
            self.voice_active = False
            self.change_state("waiting_babble")
            self.state_timer = 0
            self._try_finish_session()
            return
        self.change_state("waiting_babble")
        self.state_timer = 0
        self.response_peak_active = True
        self._response_wait_elapsed = 0

    def _face_clock_paused(self, infant_signals: Dict[str, Any]) -> bool:
        """顔時計が止まる条件。見本・オウム返し・発話区間・モータ閾値上。"""
        if self.state == "performing_parent_cue":
            return True
        if self.state == "echoing" and self._voice_active():
            return True
        if self.voice_segment.in_segment:
            return True
        # キャッシュではなく今の信号で見る（外れ直後の 1 step 遅れを避ける）。
        if self.motor_active and self._motor_above_threshold(infant_signals):
            return True
        return False

    def _update_face_clock(self, infant_signals: Dict[str, Any]) -> bool:
        """顔離脱タイムアウト。閉じたら True。"""
        if self._face_clock_paused(infant_signals):
            return False
        face_hit = float(infant_signals.get("instant_parent_face_hit", 0.0)) > 0.0
        if face_hit:
            self.face_away_timer = 0
            return False
        self.face_away_timer += 1
        if self.face_away_timer >= self.face_away_timeout_steps:
            self._finish_session(bool(self.session_success))
            return True
        return False

    def _tick_motor_sustain(self, infant_signals: Dict[str, Any]) -> None:
        """モータ持続レーンの壁時計・idle。閉じたらレーンを落とす。"""
        if not self.motor_active:
            return
        self._motor_above_now = self._motor_above_threshold(infant_signals)
        self.motor_wall_timer += 1
        if self._motor_above_now:
            self.motor_idle_timer = 0
            # 開閉が後から立ったらチャネルを足す。
            for name, score in self._motor_channels_above(infant_signals):
                if name not in self.active_motor_channels:
                    self.active_motor_channels.append(name)
                    self.motor_timers[name] = 0
                    if self.gesture_pos is None:
                        self._maybe_init_gesture_pos()
        else:
            self.motor_idle_timer += 1
        if self.motor_wall_timer >= self.motor_max_steps:
            self._close_motor_lane()
            return
        if self.motor_idle_timer >= self.motor_idle_timeout_steps:
            self._close_motor_lane()

    def _try_start_babble_echo(self, infant_signals: Dict[str, Any]) -> bool:
        """喃語列があればオウム返しを始める。声の見本中は始めない。"""
        if self._parent_voice_demo_pending():
            return False
        if self.state == "echoing" and self._voice_active():
            return False
        if self.round_count >= self.max_rounds:
            return False
        # 往復開始後に声レーンを閉じたら、再開しない。
        if self.round_count > 0 and not self.voice_active:
            return False
        chunk = infant_signals.get("babbling_chunk")
        if not chunk:
            return False
        if self.time_to_engagement is None and self.response_peak_active:
            self.time_to_engagement = self._response_wait_elapsed
            if isinstance(self.debug_info.get("response_match"), dict):
                self.debug_info["response_match"]["time_to_engagement"] = (
                    self.time_to_engagement
                )
        self._begin_echo(list(chunk))
        return True

    def _manage_state(self, infant_signals: Dict[str, Any]) -> None:
        if self.detect_rejection(infant_signals):
            self.change_state(self.failure_state)
            self._clear_imitation_queues()
            self.voice_active = False
            self.motor_active = False
            self.is_finished = True
            return

        if self.is_finished:
            return

        self.state_timer += 1

        # D2: waiting_babble 進入後はプロトコル終了まで peak を更新
        if self.response_peak_active:
            self._update_response_match(infant_signals)

        # 顔離脱（セッション全体を閉じうる）
        if self._update_face_clock(infant_signals):
            return

        # モータ持続レーン
        if self.motor_active:
            self._tick_motor_sustain(infant_signals)
            if self.is_finished:
                return

        if self.state == "init":
            if self.infant_initiative:
                self.change_state("waiting_infant")
            else:
                self._begin_parent_cue(self._parent_cue_channels())

        elif self.state == "waiting_infant":
            # 喃語とモータを同時に見て、出たほうを足す。
            started = False
            if infant_signals.get("babbling_chunk"):
                self._try_start_babble_echo(infant_signals)
                started = True
            if self._motor_above_threshold(infant_signals):
                self._activate_motor(infant_signals)
                started = True
            if started:
                if self.voice_active and self.state == "echoing":
                    pass
                elif self.motor_active and not self.voice_active:
                    # モータだけ先に立った。次の喃語を待つ。
                    self.change_state("waiting_babble")
                    self.state_timer = 0
                    self.response_peak_active = True
                    self._response_wait_elapsed = 0
            elif self.state_timer > self.waiting_infant_timeout_steps:
                action = self.handle_timeout(max_retries=1)
                if action == "RETRY":
                    self.queue_utterance("prompt")
                    self.state_timer = 0
                elif action in ("ESCALATE", "CHANGE_TARGET", "FAILED"):
                    self._finish_session(False)

        elif self.state == "performing_parent_cue":
            # 切り分け: 掲げヒット率で成功／見本終了で失敗。持続レーンにも喃語にも入らない。
            if self.close_on_raise:
                if self._arm_raise_above(infant_signals):
                    self.session_success = True
                    self._infant_has_responded = True
                    self._mark_dense_raise_success()
                    self._finish_session(True)
                    return
                if self._parent_cue_complete():
                    self._dense_done = True
                    self._finish_session(False)
                    return
            else:
                # 見本途中でもモータ閾値以上なら即反応。見本のモータはそのまま続ける。
                if not self.motor_active and self._motor_above_threshold(infant_signals):
                    self._mark_dense_raise_success()
                    self._activate_motor(infant_signals)
                # 声の見本が終わったあと（または声無し）なら、喃語でオウム返し。
                if not self._parent_voice_demo_pending():
                    self._try_start_babble_echo(infant_signals)
                if self._parent_cue_complete():
                    # 見本終了。反応が無ければ喃語／モータ待ちへ。
                    if self.state == "echoing":
                        pass
                    else:
                        if not self.session_success:
                            self._dense_done = True
                        self.change_state("waiting_babble")
                        self.state_timer = 0
                        self.response_peak_active = True
                        self._response_wait_elapsed = 0
                        # 見本を言っただけではフラグは立たない（モータ反応済みなら立っている）。

        elif self.state == "echoing":
            # エコー中もモータを足せる。
            if not self.motor_active and self._motor_above_threshold(infant_signals):
                self._activate_motor(infant_signals)
            # 声キューが空なら 1 往復完了（モータは並走のまま）。
            if not self._voice_active():
                self._on_echo_complete()

        elif self.state == "waiting_babble":
            # モータがまだなら足す。声レーンが開いていれば（または未開始なら）喃語を待つ。
            if not self.motor_active and self._motor_above_threshold(infant_signals):
                self._activate_motor(infant_signals)
            if self._try_start_babble_echo(infant_signals):
                return
            if self.state_timer > self.waiting_babble_timeout_steps:
                # 声の機会が終わってモータだけ残っているときは、壁／idle に任せる。
                if not self.voice_active and self.motor_active and self.session_success:
                    self.state_timer = 0
                elif self.voice_active or self.round_count > 0:
                    # 50 step 喃語が無ければ声レーンを閉じる。リトライで見本を言い直さない。
                    self.voice_active = False
                    self._try_finish_session()
                    if not self.is_finished and self.motor_active:
                        self.state_timer = 0
                    elif not self.is_finished:
                        self._finish_session(bool(self.session_success))
                elif self.motor_active:
                    # 親始動後の初回待ちでモータだけ成功済み。声の機会は閉じる。
                    self.state_timer = 0
                else:
                    self._finish_session(False)

        elif self.state == "imitating":
            # 旧単発経路の互換。持続レーンへ移す。
            if not self.motor_active:
                if self.active_motor_channels:
                    self.motor_active = True
                    self.motor_wall_timer = 0
                    self.motor_idle_timer = 0
                    self.session_success = True
                    self._infant_has_responded = True
                    self.change_state("waiting_babble")
                    self.state_timer = 0
                    self.response_peak_active = True
                elif self._imitation_channels_complete():
                    self.session_success = True
                    self._finish_session(True)

    def _determine_action(self) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)

        self.env.gaze_agent(self.env.parent, self.env.infant, yaw_link="head", action=action)

        motor_playing = self._has_active_motor()
        if motor_playing:
            self._apply_motor_channels(action)
        elif self.motor_active and not self._motor_above_now:
            # 持続中だが閾値未満 → 腕を下ろして待つ。
            self.env.reset_arm_pose(self.env.parent, action=action)
        elif self.state not in ("imitating", "performing_parent_cue", "echoing"):
            self.env.reset_arm_pose(self.env.parent, action=action)

        self._apply_utterance_action(action)
        return action
