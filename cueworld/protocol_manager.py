import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple
from .protocols import (
    ObservanceProtocol, JointAttentionProtocol, BaseProtocol,
    DyadicProtocol, ObjectGivingProtocol, HouseholdSoloProtocol,
    VoiceEchoProtocol,
)
from collections import deque
from .intent import (
    INTENT_KEYS,
    compute_intent_flags,
    parent_dyadic_cue,
    parent_object_cue,
    word_candidate,
    DEFAULT_VOICE_VOLUME_THRESHOLD,
)

# 関わりプロトコルの終了履歴（バッチ C）。件数は仮。
PROTOCOL_HISTORY_MAXLEN = 32
PROTOCOL_HISTORY_INFO_LEN = 8
# 本番の家事間隔（デモ・短い観測の既定 200 とは分ける）。
HOUSEHOLD_INTERVAL_PRODUCTION = 6000
# 親始動の加点（自発性 0・τ=0.2 でなし約3分 / あり約30秒）。
WAIT_BASE_SCORE_DEFAULT = 1.04
CUE_BONUS_DEFAULT = 0.36
# バッチ F: 同じ（遊び, 物）が連続成功した回数。届いたら除外リストへ。
SUCCESS_STREAK_MAX = 3
# ==========================================
# Protocol Manager
# ==========================================

class ProtocolManager(gym.Wrapper):
    """
    全てのプロトコルを統括・管理するマネージャークラス。
    幼児のシグナルから「プロトコルの切り替え」や「リキャストの割り込み」を判断し、
    具体的な内容は各プロトコルクラスへ委譲する。
    infant_signalのラッパーに重ねる.
    """
    def __init__(self, env: gym.Env, base_try_count: int = 30, decay_rate: Optional[float] = None, check_interval: int = 10,
                 tau_response: float = 0.2, tau_initiative: float = 0.2, 
                 persistence_scale: float = 1.0, persistence_bias: float = 0.0,
                 wait_base_score: float = WAIT_BASE_SCORE_DEFAULT,
                 cue_bonus: float = CUE_BONUS_DEFAULT,
                 household_interval_steps: int = 200,
                 fixed_protocol: Optional[str] = None,
                 fixed_context: Optional[Dict[str, Any]] = None,
                 change_target: bool = True,
                 # 既定オフ。逃避・視線・情動の経路は残す。True のとき JA / dyadic / giving が detect_rejection を使う。
                 rejection: bool = False,
                 # 切り分けテスト用。True のとき選好を学習報酬から外し、プロトコルの dense_reward_function を r[0] に足す。
                 dense_reward: bool = False,
                 log_protocol_history: bool = True,
                 *args, **kwargs):
        self.env = env
        self.active_signals = getattr(self.env, "active_signals", None)
        self.check_interval = check_interval  # 判定を行うステップ間隔
        self.check_counter = 0  # 判定用カウンタ
        # household の再開間隔。デモ・短い観測の既定は 200。本番ランは HOUSEHOLD_INTERVAL_PRODUCTION（6000）。
        self.household_interval_steps = household_interval_steps
        self.last_household_end_step = 0
        self.fixed_protocol = fixed_protocol
        self.fixed_context = fixed_context or {}
        self.dense_reward = bool(dense_reward)
        
        # 登録されているプロトコル一覧
        self.eval_mode = bool(kwargs.get("eval_mode", False))
        self.protocols = {
            "observance": ObservanceProtocol(self.env),
            "joint_attention": JointAttentionProtocol(self.env, cue_type="pointing", change_target=change_target, rejection_check=rejection, eval_mode=self.eval_mode),
            "dyadic": DyadicProtocol(self.env, rejection_check=rejection),
            "object_giving": ObjectGivingProtocol(self.env, rejection_check=rejection),
            "household_solo": HouseholdSoloProtocol(self.env),
            # 切り分けテスト用。本番の訴え／親始動の候補には載せない（solo / response_type=False）。
            "voice_echo": VoiceEchoProtocol(self.env, rejection_check=False),
        }

        self.level_clusters = {}
        for name, protocol in self.protocols.items():
            if hasattr(protocol, 'level'):
                if protocol.level not in self.level_clusters:
                    self.level_clusters[protocol.level] = set()
                self.level_clusters[protocol.level].add(name)
        
        # 現在実行中のプロトコル
        self.current_protocol: Optional[BaseProtocol] = None
        self.current_protocol_name: Optional[str] = None
        self.current_infant_initiative: bool = False   # 現在のプロトコルが幼児始動かどうか
        # 関わりの終了ノート。reset（睡眠）では残す。プロトコルへは渡さない。
        self.log_protocol_history = log_protocol_history
        self.protocol_history = deque(maxlen=PROTOCOL_HISTORY_MAXLEN)
        self.current_protocol_start_step = 0
        
        # 初期状態のプロトコル設定
        if self.fixed_protocol:
            self.switch_protocol(self.fixed_protocol, self.fixed_context)
            print(f"[ProtocolManager] Fixed protocol mode enabled: {self.fixed_protocol}")
        else:
            self.switch_protocol("observance", {})
            
        #dyadicプロトコルなどでそちら始動かはcontextで指定する
        # 幼児から親への要求が可能なプロトコルのリスト
        self.response_protocol = [
            name for name, protocol in self.protocols.items()
            if getattr(protocol, "response_type", False) and not getattr(protocol, "solo", False)
        ]
        # 親から幼児への要求が可能なプロトコルのリスト
        self.request_protocol = [
            name for name, protocol in self.protocols.items()
            if getattr(protocol, "request_type", False) and not getattr(protocol, "solo", False)
        ]

        # 判断に使用するもの
        # §6-D2: 訴え行列は意図セット（0/1）× プロトコル。部品の直和は開始判定から外す。
        # signal_keys は検出器の並び（ログ・互換用）。effective な訴え入力は intent_keys。
        self.signal_keys = [
            "point_intensity", "open_intensity",
            "hand_target_stability", "holding", "touching",
            "gaze_target_stability",
            "voice_target_stability",
            "hand_open_close_periodicity",
            "hand_height_margin",
            "babbling_ratio", "voice_volume",
        ]
        self.intent_keys = list(INTENT_KEYS)

        # 意図 → プロトコル重み（初期値。本調整はバッチ E）
        # 完成形は温度(0.2)より十分大きく、弱い腕は薄く。
        intent_weights_dict = {
            "joint_attention": {
                "pointing": 2.0,
                "weak_arm": 0.3,
            },
            "object_giving": {
                "reach": 2.0,
                "word": 2.0,
                "weak_arm": 0.3,
            },
            "dyadic": {
                "dyadic": 2.0,
            },
        }

        self.effective_signals = self.active_signals if self.active_signals is not None else self.signal_keys

        weight_matrix = []
        for p_name in self.response_protocol:
            p_weights = intent_weights_dict.get(p_name, {})
            row = [p_weights.get(intent_key, 0.0) for intent_key in self.intent_keys]
            weight_matrix.append(row)

        self.weights_for_strength_calc = np.array(weight_matrix, dtype=np.float64)
        self._last_intent_meta: Dict[str, Any] = {}

        # しきい値
        self.base_zpd_score = 0.1 #zpdスコアのしきい値．これを基準に現在の最大レベルを決定する．
        #フロンティアブースト(幼児がギリギリできない部分を選びやすくする)
        self.frontier_boost = 0.1

        # 幼児の自発性 (Spontaneity) の計算用履歴
        #コミュニケーションプロトコルの成功率(指数移動平均で)
        self.communication_protocol_scores = {}
        # 幼児が自分から始めて成功したコミュニケーションの成功率(指数移動平均)。親の待ちに使う。
        self.infant_initiative_rate = 0
        # プロトコルのレベルごとの成功率(指数移動平均)
        self.level_scores = [0]*len(self.level_clusters)
        # 現在のプロトコルの合計試行回数(これが基準値を超えないと，上記の記録を有効にしない．)
        self.current_try_count = 0
        # それぞれのプロトコルの試行回数
        self.protocol_try_counts = {name: 0 for name in self.protocols.keys()}
        # それぞれのレベルの試行回数
        self.level_try_counts = [0]*len(self.level_clusters)
        # 試行回数の閾値
        self.base_try_count = base_try_count

        # 各指数移動平均の重み
        if decay_rate is None:
            self.decay_rate = 2.0/(self.base_try_count+1.0)
        else:
            self.decay_rate = decay_rate

        # 確率計算用のハイパーパラメータ
        self.tau_response = tau_response
        self.tau_initiative = tau_initiative
        self.persistence_scale = persistence_scale
        self.persistence_bias = persistence_bias
        self.wait_base_score = wait_base_score
        self.cue_bonus = cue_bonus
        
        # ペナルティ管理用
        self.protocol_penalties = {name: 0.0 for name in self.protocols.keys()}
        self.penalty_decay_rate = 0.05 # 1ステップあたりの減衰量 (判定時にまとめて減衰)
        self.base_penalty_value = 2.0  # 失敗時に付与するペナルティ量
        
        # バッチ F: 除外リストは（遊びの名前, 物の id）。生の信号は隠さない。
        self.blocked_pairs: set[tuple[str, Any]] = set()
        self.success_streak: Dict[str, tuple[Any, int]] = {}

    def reset(self, **kwargs):
        """Reset the environment and the manager's internal state."""
        obs, info = self.env.reset(**kwargs)

        # reset は幼児の睡眠。配置・関節は下位 env.reset が戻す。
        # 消すもの: check_counter, protocol_penalties, blocked_pairs / success_streak（今エピソードの除外）
        # 残すもの: EMA（communication_protocol_scores / infant_initiative_rate / level_scores）と try_count、protocol_history
        # 途中だった関わりはノートに書かない（switch するだけ）。
        if self.fixed_protocol:
            target_obj_id = getattr(self.env, "target_obj_id", None)
            if target_obj_id is not None:
                self.fixed_context["fixed_target"] = self.env.all_objects[target_obj_id]
            self.switch_protocol(self.fixed_protocol, self.fixed_context)
        else:
            self.switch_protocol("observance", {})
        self.check_counter = 0
        self.last_household_end_step = 0
        
        # ペナルティの初期化
        self.protocol_penalties = {name: 0.0 for name in self.protocols.keys()}
        self.blocked_pairs.clear()
        self.success_streak.clear()
        
        info["parent_action"] = np.zeros(self.env.action_space[1].shape, dtype=np.float32)  # 初期の親の行動をゼロベクトルに設定
        
        return obs, info
        
    def set_fixed_protocol(self, protocol_name: str, context: Dict[str, Any] = None):
        """デモ用にプロトコルを固定する（外部からの呼び出し用）"""
        self.fixed_protocol = protocol_name
        self.fixed_context = context or {}
        self.switch_protocol(protocol_name, self.fixed_context)

    def step(self, action):
        o, r, d, t, i = self.env.step(action)
        
        # 対策案 X: 親の認知フィルターとして、無視リストを渡してシグナルを再計算
        if hasattr(self.env, "get_normalized_signals"):
            infant_signals = self.env.get_normalized_signals()
            i["infant_signals"] = infant_signals
        else:
            infant_signals = i.get("infant_signals", {})
            
        i["parent_action"], protocol_reward = self.update(infant_signals)
        # プロトコル報酬は学習信号に使わない（info のみ）
        i["protocol_reward"] = protocol_reward if protocol_reward is not None else 0.0
        if self.dense_reward:
            # 選好を学習報酬から外し、密なプロトコル報酬だけを足す
            r = list(r)
            pref_total = float((i.get("preference_reward") or {}).get("total", 0.0))
            r[0] = float(r[0]) - pref_total
            dense = 0.0
            if self.current_protocol is not None:
                dense = float(self.current_protocol.dense_reward_function(infant_signals))
            r[0] = float(r[0]) + dense
            i["dense_reward"] = dense
            i["preference_reward"] = {"hold": 0.0, "look": 0.0, "total": 0.0}
        if self.log_protocol_history:
            i["protocol_history"] = self._protocol_history_for_info()
        
        if self.fixed_protocol and self.current_protocol and self.current_protocol.is_finished:
            d = True
            
        return o, r, d, t, i

    def switch_protocol(self, protocol_name: str, context: Dict[str, Any]):
        """プロトコルを切り替える"""
        if self.current_protocol is not None:
            self.current_protocol.stop()
            
        # キャッシュ（IK状態・視線回避等）をクリア
        if hasattr(self.env, "clear_cache"):
            self.env.clear_cache()
            
        if protocol_name in self.protocols:
            self.current_protocol = self.protocols[protocol_name]
            self.current_protocol_name = protocol_name
            self.current_protocol.start(context)
        else:
            print(f"Warning: Protocol '{protocol_name}' not found.")
            self.current_protocol = self.protocols["observance"]
            self.current_protocol_name = "observance"
        self.current_protocol_start_step = getattr(self.env, "step_count", 0)

    def _household_is_due(self) -> bool:
        """前回 household 完了から十分 step が経過したか。"""
        step_count = getattr(self.env, "step_count", 0)
        return (step_count - self.last_household_end_step) >= self.household_interval_steps

    def _get_household_task_candidates(self) -> list[dict[str, Any]]:
        """household が実行可能な候補を現在環境から取得する。"""
        protocol = self.protocols.get("household_solo")
        if protocol is None:
            return []
        return protocol._get_task_candidates()

    def _can_start_household(self) -> bool:
        """期限を満たし、かつ household の候補が存在するか。"""
        if not self._household_is_due():
            return False
        return len(self._get_household_task_candidates()) > 0

    def _build_household_context(self, infant_signals: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "infant_initiative": False,
            "infant_signals": infant_signals,
            "global_ignored_targets": self._ignored_oids_for("household_solo"),
        }

    def _maybe_start_household(self, infant_signals: Dict[str, Any]) -> bool:
        """期限と候補が揃っていれば household を開始する。"""
        if not self._can_start_household():
            return False
        self.current_infant_initiative = False
        self.switch_protocol("household_solo", self._build_household_context(infant_signals))
        return True

    def update(self, infant_signals: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """
        環境の毎ステップ呼ばれ、親の次の行動を決定する。
        
        Args:
            infant_signals: infant_signal.py から取得した幼児の各種シグナル（辞書形式を想定）
            
        Returns:
            parent_action: 親の行動
            reward: 報酬
        """
        # 判定頻度の制御: check_interval ステップごとに判定を行う
        self.check_counter += 1
        if self.check_counter >= self.check_interval:
            self.check_counter = 0
            
            # ペナルティの減衰処理 (判定のタイミングでまとめて減衰)
            decay_amount = self.penalty_decay_rate * self.check_interval
            for p_name in self.protocol_penalties:
                if self.protocol_penalties[p_name] > 0:
                    self.protocol_penalties[p_name] = max(0.0, self.protocol_penalties[p_name] - decay_amount)

            # 1. プロトコル切り替え（中断・Rejection）の判定
            if not self.fixed_protocol:
                interrupt_blocked = False
                if self.current_protocol is not None:
                    blocks = getattr(self.current_protocol, "blocks_interrupt", None)
                    if callable(blocks):
                        interrupt_blocked = bool(blocks(infant_signals))
                if interrupt_blocked:
                    should_switch, new_protocol_name, context = False, "", {}
                else:
                    should_switch, new_protocol_name, context = self.check_for_response(infant_signals)
                if should_switch:
                    print(f"Protocol switch triggered! Changing to {new_protocol_name}")
                    if not self.current_protocol.solo:
                        # 離脱時の成功は session_success を見る（割り込みでも成功ありうる）。
                        self._leave_protocol(
                            self.current_protocol_name, end_reason="interrupt"
                        )
                    self.switch_protocol(new_protocol_name, context)
                elif self.current_protocol_name == "observance":
                    if self._maybe_start_household(infant_signals):
                        print("Protocol switch triggered! Changing to household_solo")
                    else:
                        should_switch, new_protocol_name, context = self.check_for_initiative(infant_signals)
                        if should_switch:
                            print(f"Protocol switch triggered! Changing to {new_protocol_name}")
                            self.switch_protocol(new_protocol_name, context)
        # TODO: 親の情報を環境などから取得して辞書にする
        parent_info = {}

        # 2. 現在のプロトコルの進行 (この内部でリキャスト等の処理も行われる)
        parent_action, proximity, success_flag, reward = self.current_protocol(infant_signals, parent_info)

        # プロトコルが自然終了した場合はObservanceに戻るなどの処理
        # 履歴・ペナルティは _leave_protocol に揃える（success_flag 返却は使わない）。
        if self.current_protocol.is_finished:
            finished_protocol_name = self.current_protocol_name
            finished_protocol = self.current_protocol
            session_ok = bool(getattr(finished_protocol, "session_success", False))
            # 固定プロトコルでは is_finished が複数 step 続きうるので 1 回だけ書く。
            if not getattr(finished_protocol, "_pm_leave_done", False):
                setattr(finished_protocol, "_pm_leave_done", True)
                if not getattr(finished_protocol, "solo", False):
                    self._leave_protocol(finished_protocol_name, end_reason="natural")
                self._update_block_list_on_finish(
                    finished_protocol_name, finished_protocol, session_ok
                )

            if not self.fixed_protocol:
                if finished_protocol_name == "household_solo":
                    self.last_household_end_step = getattr(self.env, "step_count", self.last_household_end_step)
                    self.switch_protocol("observance", {})
                elif self._maybe_start_household(infant_signals):
                    pass
                else:
                    self.switch_protocol("observance", {})
            self.current_infant_initiative = False

        return parent_action, reward

    def _leave_protocol(self, protocol_name: str, end_reason: str) -> None:
        """関わりプロトコルの離脱を一箇所で処理する。

        成功かどうかはプロトコルの session_success を見る。
        割り込みでも、フラグがあれば成功として書く。
        ペナルティは成功フラグが偽のときだけ足す。
        """
        protocol = self.protocols[protocol_name]
        if getattr(protocol, "solo", False):
            return
        success = bool(getattr(protocol, "session_success", False))
        if end_reason == "interrupt":
            effective_end_reason = "interrupt"
        elif success:
            effective_end_reason = "success"
        else:
            effective_end_reason = "failure"
        self.save_protocol_history(
            protocol_name, success, end_reason=effective_end_reason
        )
        if not success:
            self.protocol_penalties[protocol_name] += self.base_penalty_value
            print(
                f"[ProtocolManager] Protocol {protocol_name} failed. "
                f"Penalty applied: {self.protocol_penalties[protocol_name]:.2f}"
            )

    def _response_probs(
        self, infant_signals: Dict[str, Any]
    ) -> Tuple[list, np.ndarray]:
        """
        幼児の訴えに対する切替くじの (choices, probs)。
        choices[0] は "maintain"。観測・テストが本番と同じ softmax を見るために切り出す。
        """
        strengths = self._calculate_signal_strength(infant_signals)

        for i, p_name in enumerate(self.response_protocol):
            strengths[i] -= self.protocol_penalties.get(p_name, 0.0)

        # 二項ゲート: JA / giving が実際にくじに乗っているときだけ dyadic を落とす。
        # 拒否・ペナルティ・到達圏で JA/giving が 0 なら二項は残す。
        if "dyadic" in self.response_protocol:
            dyadic_idx = self.response_protocol.index("dyadic")
            object_scores = []
            for i, p_name in enumerate(self.response_protocol):
                if p_name in ("joint_attention", "object_giving"):
                    object_scores.append(float(strengths[i]))
            if object_scores and max(object_scores) > 0.0:
                strengths[dyadic_idx] = 0.0

        current_persistence = getattr(self.current_protocol, "persistence", 0.0)
        if current_persistence is None:
            current_persistence = 0.0
        maintain_score = current_persistence * self.persistence_scale + self.persistence_bias

        all_scores = np.concatenate(([maintain_score], strengths))
        scaled_scores = all_scores / self.tau_response
        scaled_scores -= np.max(scaled_scores)
        probs = np.exp(scaled_scores) / np.sum(np.exp(scaled_scores))
        probs /= np.sum(probs)

        choices = ["maintain"] + self.response_protocol
        return choices, probs

    def check_for_response(self, infant_signals: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
        """
        幼児の動きに合わせてプロトコルを切り替えるか判定する。（Acceptance Path / 対拒否処理）
        この経路で始まったプロトコルは幼児始動（infant_initiative=True）。
        """
        choices, probs = self._response_probs(infant_signals)
        choices, probs = self._filter_blocked_lottery(
            choices, probs, infant_signals, infant_initiated=True
        )
        sampled_protocol_name = np.random.choice(choices, p=probs)

        current_persistence = getattr(self.current_protocol, "persistence", 0.0)
        if current_persistence is None:
            current_persistence = 0.0
        maintain_score = current_persistence * self.persistence_scale + self.persistence_bias
        max_req = float(np.max(probs[1:])) if len(probs) > 1 else 0.0
        print(
            f"Checking response: Current={self.current_protocol_name}, "
            f"MaintainScore={maintain_score:.3f}, MaxReqProb={max_req:.3f}, "
            f"Sampled={sampled_protocol_name}"
        )

        if sampled_protocol_name == "maintain" or sampled_protocol_name == self.current_protocol_name:
            return False, "", {} # 維持、または現在と同じなら切り替え不要

        # 4. 新しいプロトコル用のコンテキストを設定
        context = {}
        self.current_infant_initiative = True
        context["infant_initiative"] = True
        context["infant_signals"] = infant_signals
        context["global_ignored_targets"] = self._ignored_oids_for(sampled_protocol_name)

        meta = self._last_intent_meta
        intent_target_id = self._intent_target_for_protocol(sampled_protocol_name, meta)
        if intent_target_id is not None:
            context["intent_target_id"] = intent_target_id
            context["intent_source"] = meta.get("intent_source")
        
        if sampled_protocol_name == "joint_attention":
            context["intent_type"] = "informative"

        if sampled_protocol_name == "dyadic":
            # salience ランキングは DyadicProtocol 内で計算する。
            # ここでは喃語列など、起動時に既知の情報だけ渡す。
            babbling_chunk = infant_signals.get("babbling_chunk")
            if babbling_chunk is not None:
                context["babbling_chunk"] = list(babbling_chunk)
        
        # 5. プロトコルに切り替える(返り値のTrueに従い、update内でswitch_protocolが呼ばれる)
        return True, sampled_protocol_name, context

    def _softmax_probs(self, scores: np.ndarray, tau: float) -> np.ndarray:
        scaled = np.asarray(scores, dtype=float) / float(tau)
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        probs /= np.sum(probs)
        return probs

    def _initiative_candidates(
        self, infant_signals: Dict[str, Any]
    ) -> Tuple[list, list, list]:
        """親始動の候補名、ZPD、手がかり 0/1。候補が無ければ空リスト。"""
        max_level = -1
        for level, score in enumerate(self.level_scores):
            if self.level_try_counts[level] < self.base_try_count:
                score_l = 0.0
            else:
                score_l = score
            s_i = score_l * (1.0 - score_l)
            if s_i >= self.base_zpd_score:
                max_level = level
        max_level += 1

        object_cue = parent_object_cue(infant_signals)
        dyadic_cue = parent_dyadic_cue(infant_signals)

        names = []
        zpds = []
        cues = []
        for p_name in self.request_protocol:
            protocol = self.protocols[p_name]
            level = getattr(protocol, "level", 0)
            if level > max_level:
                continue
            if p_name == "object_giving" and not self._giving_initiative_allowed(infant_signals):
                continue
            if self.level_try_counts[level] < self.base_try_count:
                level_success = 0.0
            else:
                level_success = self.level_scores[level]
            zpd_score = level_success * (1.0 - level_success)
            if zpd_score < self.base_zpd_score:
                zpd_score += self.frontier_boost
            if protocol.has_target:
                cue = float(object_cue)
            elif p_name == "dyadic":
                cue = float(dyadic_cue)
            else:
                cue = 1.0
            names.append(p_name)
            zpds.append(float(zpd_score))
            cues.append(cue)
        return names, zpds, cues

    def _initiative_probs(
        self, infant_signals: Dict[str, Any]
    ) -> Tuple[list, Optional[np.ndarray]]:
        """
        親始動のくじの (choices, probs)。
        choices[0] は "observance"（待機）。候補が無ければ ([], None)。

        一段目は待機 vs 開始（候補数・ZPD に依存しない）。二段目はどの遊びか。
        返す分布は一段×二段を掛けた結合確率（観測・サンプリングが同じ式を見る）。
        """
        # --- 一段目: 待つか、始めるか ---
        # s_spon = infant_initiative_rate（幼児始動の成功 EMA）。
        # 試行回数が base_try_count 未満なら 0。自発性が高いほど待機スコアが上がり親は始めにくくなる。
        if self.current_try_count >= self.base_try_count:
            s_spon = self.infant_initiative_rate
        else:
            s_spon = 0.0
        wait_score = self.wait_base_score + s_spon  # 既定 1.04 + s_spon

        # 二段目の候補（request_protocol から ZPD・手がかり付きで絞り込み）
        names, zpds, cues = self._initiative_candidates(infant_signals)
        if not names:
            return [], None

        # 手がかり（parent_object_cue / parent_dyadic_cue）が 1 つでも立てば開始スコアに cue_bonus。
        # 指差し・リーチ・単語は手がかりに使わない（訴えは response 経路の仕事）。
        any_cue = any(c > 0.0 for c in cues)
        start_score = self.cue_bonus if any_cue else 0.0  # 既定 0.36 or 0
        p_stage1 = self._softmax_probs(
            np.array([wait_score, start_score], dtype=float),
            self.tau_initiative,  # 既定 0.2
        )
        p_wait = float(p_stage1[0])
        p_start = float(p_stage1[1])

        # --- 二段目: どの遊びか ---
        # 魅力度 = zpd + cue_bonus * 手がかり - ペナルティ
        # zpd = level_success * (1 - level_success)。未踏レベルは frontier_boost (+0.1) 済み。
        attractions = []
        for zpd_score, cue, p_name in zip(zpds, cues, names):
            attraction = zpd_score + self.cue_bonus * cue
            attraction -= self.protocol_penalties.get(p_name, 0.0)
            attractions.append(attraction)
        p_which = self._softmax_probs(
            np.array(attractions, dtype=float),
            self.tau_initiative,
        )

        # --- 結合: [p_wait, p_start * p_which[0], ..., p_start * p_which[n]] ---
        # 候補数が増えても開始頻度 p_start は一段目だけで決まる。
        combined = np.concatenate(([p_wait], p_start * p_which))
        combined = np.maximum(combined, 0.0)
        combined /= np.sum(combined)
        choices = ["observance"] + names
        return choices, combined

    def check_for_initiative(self, infant_signals: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
        """
        親主導でプロトコルを開始するか判定する。（Request Path）
        幼児の自発性（S_spon）や注視状態に基づいて、待機するか何かを始めるかを1回のサンプリングで決定する。
        """
        choices, probs = self._initiative_probs(infant_signals)
        if not choices or probs is None:
            return False, "", {}

        choices, probs = self._filter_blocked_lottery(
            choices, probs, infant_signals, infant_initiated=False
        )
        sampled_protocol_name = np.random.choice(choices, p=probs)
        
        wait_score = self.wait_base_score + (
            self.infant_initiative_rate if self.current_try_count >= self.base_try_count else 0.0
        )
        max_attr = float(np.max(probs[1:])) if len(probs) > 1 else 0.0
        print(
            f"Checking initiative: WaitScore={wait_score:.3f}, "
            f"MaxAttrProb={max_attr:.3f}, Sampled={sampled_protocol_name}"
        )
        
        if sampled_protocol_name == "observance":
            return False, "", {}
            
        context = {}
        self.current_infant_initiative = False
        context["infant_initiative"] = False
        context["infant_signals"] = infant_signals
        context["global_ignored_targets"] = self._ignored_oids_for(sampled_protocol_name)

        return True, sampled_protocol_name, context

    def _voice_volume_threshold(self) -> float:
        detector = getattr(self.env, "voice_volume_threshold", None)
        if detector is not None:
            return float(detector)
        return DEFAULT_VOICE_VOLUME_THRESHOLD

    def _trackable_targets(self) -> list:
        return list(getattr(self.env, "trackable_targets", []) or [])

    def _parent_body_id(self) -> Optional[Any]:
        parent = getattr(self.env, "parent", None)
        if parent is None:
            return None
        return parent.get("body")

    def _resolve_target_from_vector(
        self, target_vec: Any, *, ignore_parent: bool = True
    ) -> Optional[Any]:
        if target_vec is None:
            return None
        arr = np.asarray(target_vec, dtype=float)
        if arr.size == 0 or float(np.max(arr)) <= 0.0:
            return None
        trackable = self._trackable_targets()
        parent_body = self._parent_body_id() if ignore_parent else None
        for idx in np.argsort(arr)[::-1]:
            if float(arr[idx]) <= 0.0:
                break
            if int(idx) >= len(trackable):
                continue
            tid = trackable[int(idx)]
            if ignore_parent and parent_body is not None and tid == parent_body:
                continue
            return tid
        return None

    def _resolve_hand_target_id(self, infant_signals: Dict[str, Any]) -> Optional[Any]:
        return self._resolve_target_from_vector(
            infant_signals.get("gesture_targets", []), ignore_parent=True
        )

    def _resolve_gaze_target_id(self, infant_signals: Dict[str, Any]) -> Optional[Any]:
        return self._resolve_target_from_vector(
            infant_signals.get("gaze_targets", []), ignore_parent=True
        )

    def _find_object_by_concept(self, concept_id: Optional[str]) -> Optional[Any]:
        if not concept_id:
            return None
        all_objects = getattr(self.env, "all_objects", {}) or {}
        active = list(getattr(self.env, "active_objects", []) or [])
        candidates = active if active else list(all_objects.keys())
        for oid in candidates:
            obj = all_objects.get(oid)
            if obj is None:
                continue
            if getattr(obj, "concept_id", None) == concept_id:
                return oid
        return None

    def _in_infant_reach(self, oid: Any) -> bool:
        """幼児が自分で取れるか。親が持っている物は取れないので False。"""
        if oid is None:
            return False
        parent = getattr(self.env, "parent", None) or {}
        if parent.get("held_obj") == oid:
            return False
        infant = getattr(self.env, "infant", None)
        if infant is None:
            return False
        checker = getattr(self.env, "object_in_agent_reach", None)
        if callable(checker):
            return bool(checker(infant, oid))
        return False

    def _has_out_of_reach_object(self) -> bool:
        all_objects = getattr(self.env, "all_objects", {}) or {}
        active = list(getattr(self.env, "active_objects", []) or [])
        candidates = active if active else list(all_objects.keys())
        for oid in candidates:
            obj = all_objects.get(oid)
            if obj is not None and getattr(obj, "is_platform", False):
                continue
            if not self._in_infant_reach(oid):
                return True
        return False

    def _giving_initiative_allowed(self, infant_signals: Dict[str, Any]) -> bool:
        """親始動のくれ: 持っている／届く物しかないときは候補にしない。"""
        if float(infant_signals.get("holding", 0.0)) > 0.0:
            return False
        return self._has_out_of_reach_object()

    def _intent_target_for_protocol(
        self, protocol_name: str, meta: Dict[str, Any]
    ) -> Optional[Any]:
        if protocol_name == "joint_attention":
            return meta.get("hand_target_id")
        if protocol_name == "object_giving":
            if meta.get("hand_target_id") is not None and (
                meta.get("intents", {}).get("reach", 0.0) > 0.0
                or meta.get("intents", {}).get("weak_arm", 0.0) > 0.0
            ):
                return meta.get("hand_target_id")
            return meta.get("word_object_id")
        return None

    def _calculate_signal_strength(self, infant_signals: Dict[str, Any]):
        """
        幼児のシグナルから意図セットを合成し、親に向けた訴えの強度(S)を算出する。

        Returns:
            scores: シグナル強度ベクトル (各プロトコルに対応, self.response_protocolの順序)
        """
        vol_thresh = self._voice_volume_threshold()
        intents = compute_intent_flags(
            infant_signals, voice_volume_threshold=vol_thresh
        )
        word_concept = word_candidate(
            infant_signals, voice_volume_threshold=vol_thresh
        )
        word_object_id = self._find_object_by_concept(word_concept)
        # 単語は場にある concept のときだけ訴え
        if word_object_id is None:
            intents["word"] = 0.0

        hand_target_id = self._resolve_hand_target_id(infant_signals)

        # 到達圏ゲート: 届く物・持っている物はくれにしない
        holding = float(infant_signals.get("holding", 0.0)) > 0.0
        reach_blocked = holding
        if hand_target_id is not None and self._in_infant_reach(hand_target_id):
            reach_blocked = True
        if word_object_id is not None and self._in_infant_reach(word_object_id):
            # 単語だけのくれも届く物なら 0
            intents["word"] = 0.0
            word_object_id = None
        if reach_blocked:
            intents["reach"] = 0.0
            # 弱い腕の giving 寄与も止める（JA 側の weak_arm は残す → 行列で両方に乗るので
            # giving 行だけ弱くしたい。ここでは reach だけ落とし、weak_arm は JA 用に残す。
            # giving の weak_arm を落とすには、到達圏時は weak_arm を giving から外す必要がある。
            # 簡単のため到達圏＋手対象ありのときは weak_arm も 0（届く物へのリーチは訴えにしない）。
            if hand_target_id is not None and self._in_infant_reach(hand_target_id):
                intents["weak_arm"] = 0.0
            if holding:
                intents["weak_arm"] = 0.0
                intents["reach"] = 0.0

        # 指差し優先: 揃った指差しがあるとき単語はくれに足さない
        if intents.get("pointing", 0.0) > 0.0:
            intents["word"] = 0.0

        intent_vec = np.array(
            [float(intents.get(k, 0.0)) for k in self.intent_keys], dtype=np.float64
        )
        scores = self.weights_for_strength_calc @ intent_vec

        intent_source = None
        if intents.get("pointing", 0.0) > 0.0:
            intent_source = "pointing"
        elif intents.get("reach", 0.0) > 0.0:
            intent_source = "reach"
        elif intents.get("weak_arm", 0.0) > 0.0:
            intent_source = "weak_arm"
        elif intents.get("word", 0.0) > 0.0:
            intent_source = "word"
        elif intents.get("dyadic", 0.0) > 0.0:
            intent_source = "dyadic"

        self._last_intent_meta = {
            "intents": dict(intents),
            "hand_target_id": hand_target_id,
            "word_concept": word_concept,
            "word_object_id": word_object_id,
            "intent_source": intent_source,
        }
        return scores

    def _ignored_oids_for(self, protocol_name: str) -> list:
        """その遊びについて除外リストに入っている物だけ。ランダム選択用。"""
        return [oid for (pname, oid) in self.blocked_pairs if pname == protocol_name]

    def _finish_target_id(self, protocol) -> Any:
        tid = self._protocol_target_id(protocol)
        if tid is not None:
            return tid
        return getattr(protocol, "current_item_id", None)

    def _prospective_target_for_lottery(
        self,
        protocol_name: str,
        infant_signals: Dict[str, Any],
        *,
        infant_initiated: bool,
    ) -> Any:
        if infant_initiated:
            return self._intent_target_for_protocol(
                protocol_name, self._last_intent_meta or {}
            )
        return self._resolve_gaze_target_id(infant_signals)

    def _filter_blocked_lottery(
        self,
        choices,
        probs,
        infant_signals: Dict[str, Any],
        *,
        infant_initiated: bool,
    ) -> Tuple[list, np.ndarray]:
        """見込み対象が除外済みの遊びを落とし、確率を正規化し直す。

        見込み対象が None（ランダム選択）なら落とさない。
        """
        keep_names = []
        keep_probs = []
        for name, p in zip(choices, probs):
            if name in ("maintain", "observance"):
                keep_names.append(name)
                keep_probs.append(float(p))
                continue
            target_id = self._prospective_target_for_lottery(
                name, infant_signals, infant_initiated=infant_initiated
            )
            if target_id is not None and (name, target_id) in self.blocked_pairs:
                continue
            keep_names.append(name)
            keep_probs.append(float(p))
        arr = np.asarray(keep_probs, dtype=float)
        total = float(np.sum(arr))
        if len(keep_names) == 0:
            fallback = "maintain" if infant_initiated else "observance"
            return [fallback], np.array([1.0], dtype=float)
        if total <= 0.0:
            arr = np.zeros(len(keep_names), dtype=float)
            for i, name in enumerate(keep_names):
                if name in ("maintain", "observance"):
                    arr[i] = 1.0
            if float(np.sum(arr)) <= 0.0:
                arr[:] = 1.0
            total = float(np.sum(arr))
        arr = arr / total
        return keep_names, arr

    def _update_block_list_on_finish(
        self, protocol_name: str, protocol, success_flag: Optional[bool]
    ) -> None:
        """自然終了時の除外リスト更新。割り込みでは呼ばない。"""
        state = getattr(protocol, "state", "")
        success_state = getattr(protocol, "success_state", "completed")
        failure_state = getattr(protocol, "failure_state", "failure")
        is_success = success_flag is True or (
            success_flag is None and state == success_state
        )
        is_failure = success_flag is False or (
            success_flag is None and state == failure_state
        )
        if is_success:
            self._on_protocol_success(protocol_name, protocol)
            return
        if is_failure:
            self._on_protocol_failure(protocol_name, protocol)

    def _on_protocol_success(self, protocol_name: str, protocol) -> None:
        if self.blocked_pairs:
            self.blocked_pairs.clear()
            self.success_streak.clear()
            return
        target_id = self._finish_target_id(protocol)
        if target_id is None:
            return
        last_target, count = self.success_streak.get(protocol_name, (None, 0))
        if last_target == target_id:
            count += 1
        else:
            count = 1
        if count >= SUCCESS_STREAK_MAX:
            self.blocked_pairs.add((protocol_name, target_id))
            self.success_streak[protocol_name] = (target_id, 0)
        else:
            self.success_streak[protocol_name] = (target_id, count)

    def _on_protocol_failure(self, protocol_name: str, protocol) -> None:
        blocks = getattr(protocol, "blocks_target_on_failure", None)
        should_block = True
        if callable(blocks):
            should_block = bool(blocks())
        target_id = self._finish_target_id(protocol)
        if should_block and target_id is not None:
            self.blocked_pairs.add((protocol_name, target_id))
            self.success_streak.pop(protocol_name, None)

    def _protocol_target_id(self, protocol) -> Any:
        target_obj = getattr(protocol, "target_object", None)
        if target_obj is None:
            return None
        return target_obj.id if hasattr(target_obj, "id") else target_obj

    def _protocol_kind(self, protocol, protocol_name: str):
        if protocol_name == "joint_attention":
            return getattr(protocol, "cue_type", None)
        if protocol_name == "dyadic":
            active = list(getattr(protocol, "active_channels", None) or [])
            if active:
                return active
            return list(getattr(protocol, "salient_channels", None) or [])
        return None

    def _copy_history_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(entry)
        if isinstance(row.get("kind"), list):
            row["kind"] = list(row["kind"])
        return row

    def _protocol_history_for_info(self) -> list:
        recent = list(self.protocol_history)[-PROTOCOL_HISTORY_INFO_LEN:]
        return [self._copy_history_entry(entry) for entry in recent]

    def _append_protocol_history(
        self, protocol_name: str, success_flag: bool, end_reason: str
    ) -> None:
        # 割り込み時、PM の current_infant_initiative は新プロトコル用に既に True。
        # バッファは終了するインスタンスの旗を使う。
        if self.current_protocol_name == protocol_name and self.current_protocol is not None:
            record_protocol = self.current_protocol
        else:
            record_protocol = self.protocols[protocol_name]
        kind = self._protocol_kind(record_protocol, protocol_name)
        if isinstance(kind, list):
            kind = list(kind)
        extra = {}
        history_extra = getattr(record_protocol, "history_extra", None)
        if callable(history_extra):
            extra = dict(history_extra() or {})
        end_step = getattr(self.env, "step_count", 0)
        start_step = self.current_protocol_start_step
        row = {
            "name": protocol_name,
            "success": bool(success_flag),
            "infant_initiative": bool(getattr(record_protocol, "infant_initiative", False)),
            "target_id": self._protocol_target_id(record_protocol),
            "start_step": start_step,
            "end_step": end_step,
            "length": end_step - start_step,
            "kind": kind,
            "end_reason": end_reason,
        }
        row.update(extra)
        self.protocol_history.append(row)

    def save_protocol_history(
        self, protocol_name: str, success_flag: bool, end_reason: Optional[str] = None
    ):
        """
        プロトコルの成功・失敗履歴を保存する
        TODO: 自発的かつ社会的なプロトコルの定義は再考．幼児の信号から始まるからといって必ず自発的かつ社会的とは限らない．プロトコルのfunction_typeを利用するべきかも．
        その場合は，urgentの値を設定しておく必要はないかもしれない．いや，solo protocolへの割り込みの強さに使うか．
        バッチ D: function_type は使わない。正本は訴え／親の経路（check_for_response なら幼児始動、check_for_initiative なら親始動）。
        """
        protocol = self.protocols[protocol_name]
        if getattr(protocol, "solo", False):
            return
        if end_reason is None:
            end_reason = "success" if success_flag else "failure"
        self._append_protocol_history(protocol_name, success_flag, end_reason)
        self.current_try_count += 1
        self.protocol_try_counts[protocol_name] += 1

        # 幼児始動かつ成功の成功率（指数移動平均）。割合ではない。
        # 上がると wait_score が上がり、親は見守りやすくなる。
        # 自発性は離れるプロトコル自身の infant_initiative を見る
        # （割り込み直前に立てた current_infant_initiative は見ない）。
        if getattr(protocol, "infant_initiative", False) and success_flag:
            self.infant_initiative_rate = self.infant_initiative_rate * (1.0 - self.decay_rate) + 1.0 * self.decay_rate
        else:
            self.infant_initiative_rate = self.infant_initiative_rate * (1.0 - self.decay_rate) - 0.0 * self.decay_rate
        
        if not getattr(protocol, "urgency", False):
            # レベルごとの成功率
            level = getattr(protocol, "level", 1)
            self.level_try_counts[level] += 1
            score = self.level_scores[level]
            self.level_scores[level] = score * (1.0 - self.decay_rate) + float(success_flag) * self.decay_rate
            #プロトコルごとの成功率
            score = self.communication_protocol_scores.get(protocol.name, 0) 
            self.communication_protocol_scores[protocol.name] = score * (1.0 - self.decay_rate) + float(success_flag) * self.decay_rate

    def __getattr__(self, name):
        """env へ属性を委譲するフォールバック。Wrapper自身に無い属性を下位環境から引き出す。"""
        # 無限再帰を防ぐ
        if name == "env":
            return super().__getattribute__(name)
        return getattr(self.env, name)
