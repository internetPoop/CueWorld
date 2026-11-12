import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque
from cueworld.language.manager import LanguageManager
from cueworld.intent import DEFAULT_VOICE_VOLUME_THRESHOLD


# J1: 視線拒否 2 段判定の閾値（seed 42–44 計測後に調整可）
REJECTION_TURN_AWAY_TAU = 0.5
# J1 フォールバック: 2 段拒否で fixed 誤拒否 / reject 未検知が残る場合のみ True に戻す
J1_GAZE_REJECTION_HOLDING_FALLBACK = False

# バッチ E: 維持（PM のダミークラス）のスコア。
MAINTAIN_SCORE_EARLY = 2.7  # 未乗車 / 同じ物の完成形 / リトライ中 → 割れやすい
MAINTAIN_SCORE_LATE = 5.0   # 乗車かつ別物（または対象なし）→ ほぼ割れない

# バッチ H: 共同注意の成功ホールド / 待ち・離脱 / 発話セグメントの沈黙。
# 1 step ≒ 0.1 秒。VOICE_SEGMENT_SILENCE_STEPS は VoiceRecognizer.offset_silence_frames と同値。
ATTENTION_HOLD_STEPS = 30
ATTENTION_WAIT_STEPS = 50
VOICE_SEGMENT_SILENCE_STEPS = 4
# バッチ F: 開始時の手放し上限。eval_release_steps（seed 42–44）は把持解除が 1 step。
# 失敗判定の余裕として 30（3秒）のまま置く。
RELEASE_TIMEOUT_STEPS = 30


class AttentionHoldTracker:
    """対象への途切れない注視と、一度見たあとの離脱を数える。

    hold は対象を見ているとき +1、他の物（または虚空）を見たら 0、
    親の顔を見ているあいだは据え置き。
    disengage は一度見たあとだけ進み、対象か親の顔を見たら 0。
    一度も見ていなければ never-look 待ちが ATTENTION_WAIT_STEPS で timeout。
    """

    def __init__(
        self,
        hold_steps: int = ATTENTION_HOLD_STEPS,
        wait_steps: int = ATTENTION_WAIT_STEPS,
    ):
        self.hold_steps = int(hold_steps)
        self.wait_steps = int(wait_steps)
        self.reset()

    def reset(self) -> None:
        self.hold = 0
        self.disengage = 0
        self.never_looked_timer = 0
        self.ever_looked = False

    def step(
        self,
        infant_target: Any,
        object_id: Any,
        parent_id: Any = None,
    ) -> str:
        looking_object = object_id is not None and infant_target == object_id
        looking_parent = parent_id is not None and infant_target == parent_id

        if looking_object:
            self.hold += 1
            self.ever_looked = True
            self.disengage = 0
            self.never_looked_timer = 0
            if self.hold >= self.hold_steps:
                return "success"
            return "pending"

        if looking_parent:
            self.disengage = 0
            if not self.ever_looked:
                self.never_looked_timer += 1
                if self.never_looked_timer >= self.wait_steps:
                    return "timeout"
            return "pending"

        self.hold = 0
        if self.ever_looked:
            self.disengage += 1
            if self.disengage >= self.wait_steps:
                return "timeout"
        else:
            self.never_looked_timer += 1
            if self.never_looked_timer >= self.wait_steps:
                return "timeout"
        return "pending"


class VoiceSegmentTracker:
    """幼児の発話区間。volume が閾値以上で開始、無音が silence_steps 続いたら終了。"""

    def __init__(self, silence_steps: int = VOICE_SEGMENT_SILENCE_STEPS):
        self.silence_steps = int(silence_steps)
        self.reset()

    def reset(self) -> None:
        self.in_segment = False
        self.silence_frames = 0

    def update(self, infant_signals: Optional[Dict[str, Any]], volume_threshold: float) -> None:
        volume = float((infant_signals or {}).get("voice_volume", 0.0) or 0.0)
        if volume >= float(volume_threshold):
            self.silence_frames = 0
            self.in_segment = True
            return
        if self.in_segment:
            self.silence_frames += 1
            if self.silence_frames >= self.silence_steps:
                self.in_segment = False
                self.silence_frames = 0


# target_object_checkは，各プロトコルで定義するようにする．ここでは用意しないことで，hasattrでチェックできるようにする．
class BaseProtocol:
    """
    各プロトコルの基本クラス（インターフェース）。
    実際のプロトコル（共同注意、二項関係など）はこれを継承して具体的なステートマシンを実装する。
    """
    def __init__(self, name: str, env: Any, change_target: bool = True, rejection_check: bool = True, eval_mode: bool = False, reward_mode: bool = True):
        self.name = name
        self.env = env
        self.change_target = change_target #ターゲットを変更するかどうか
        self.rejection_check = rejection_check #プロトコル内で拒否検知を行うかどうか
        self.eval_mode = eval_mode #評価モードかどうか
        self.reward_mode = reward_mode #報酬を計算して返すかどうか
        # 親の行動次元数を環境から取得
        self.action_dim = env.action_num
        self.is_finished = False
        self.state = "init"
        self.success_state = "completed" #TODO: いくつかのプロトコルでタイムアウトなどの処理をcompletedとしているものがある．要修正
        self.failure_state = "failure" 
        self.level = 0 #対応する幼児の発達レベル urgency=Trueのプロトコルには定義しない．
        self.urgency = 0 #プロトコルの緊急度(社会的コミュニケーションかどうかを決定する)
        self.has_target = False #オブジェクトを用意するプロトコルであるかどうか
        self.response_type = False #幼児が自発的に開始できるプロトコルかどうか
        self.request_type = False #親が自発的に開始できるプロトコルかどうか
        # 状態ごとに受理状態(クリア)への近さを定義する辞書。サブクラスで設定する。
        self.state_proximities: Dict[str, float] = {}
        # 幼児がきっかけかどうか
        self.infant_initiative = False
        # セッション内で一度成功したら立てる。PM への書き込みは離脱時にこのフラグを見る。
        self.session_success: bool = False
        self._pm_leave_done: bool = False

        self.target_object = None
        self.target_concept = None
        self.target_word = None

        self.ignored_targets = set()

        # エスカレーション・状態遷移管理用
        self.state_timer = 0
        self.retry_count = 0
        self._history_saved = False
        # バッチ F: 開始時の手放し前段。見守りは腕リセットに任せるのでオフ。
        self.uses_pre_release = True
        self._in_pre_release = None
        self._release_timer = 0
        self._pre_release_failed = False

        # 言語・発話管理
        self.parent_utterance = True #親が発話するかどうか(デモなどで単純かする際には発話を行わないようにするためのフラグ)
        self.language_manager = LanguageManager()
        self.utterance_queue = deque()

        # 持続強さ
        self.persistence: float = 0.0
        # 関わりの典型長さ T（成功1回, step）。None なら基底の維持式は persistence を書き換えない。
        self.typical_steps: Optional[int] = None
        # プロトコル開始時の env.step_count。PM の current_protocol_start_step と同値（履歴バッファ用は PM 側が正本）。
        self.start_step: int = 0
        # バッチ H: 幼児の発話区間（JA の締め・二項の割り込みブロックで共有）
        self.voice_segment = VoiceSegmentTracker()
        
        # デバッグ用
        self.debug_info = {}

    def start(self, context: Dict[str, Any]):
        """プロトコル開始時の初期化処理"""
        self.is_finished = False
        self.state = "init"
        self.state_timer = 0
        self.retry_count = 0
        self._history_saved = False
        self.start_step = getattr(self.env, "step_count", 0)
        self.infant_initiative = context.get("infant_initiative", False)
        self.session_success = False
        self._pm_leave_done = False
        self.infant_signals = context.get("infant_signals", {})
        self.utterance_queue.clear()
        self.voice_segment.reset()
        self.debug_info = {}
        self._in_pre_release = None
        self._release_timer = 0
        self._pre_release_failed = False
        
        # Managerから渡されたグローバルな除外リストで初期化
        self.ignored_targets = set(context.get("global_ignored_targets", []))
        self.target_object = None
        self.target_concept = None
        self.target_word = "あれ"
        
        if self.has_target:
            if context.get("fixed_target") is not None:
                fixed = context["fixed_target"]
                if hasattr(fixed, "id"):
                    self.target_object = fixed
                elif isinstance(fixed, int) and hasattr(self.env, "all_objects"):
                    self.target_object = self.env.all_objects[fixed]
                else:
                    self.target_object = fixed
            elif context.get("intent_target_id") is not None:
                # §6-D2: PM が訴えから決めた対象（手の先 / 単語の concept）
                tid = context["intent_target_id"]
                if tid not in self.ignored_targets and hasattr(self.env, "all_objects") and tid in self.env.all_objects:
                    self.target_object = self.env.all_objects[tid]
                else:
                    self.select_target()
            else:
                self.select_target()
            
        # if self.has_target and self.target_object is not None:
        #     if isinstance(self.target_object, int) and hasattr(self.env.unwrapped, "all_objects"):
        #         if isinstance(self.env.unwrapped.all_objects, dict):
        #             self.target_object = self.env.unwrapped.all_objects[self.target_object]
                
            if self.target_object is not None:
                self.target_concept = self.language_manager.get_concept_for_object(self.target_object)
                self.target_word = self.language_manager.concept_to_human_word.get(self.target_concept, "あれ")
        #print(f"[{self.name}] Protocol started.")

    def select_target(self, infant_signals: Optional[Dict[str, Any]] = None):
        """
        ターゲットオブジェクトを決定する。
        デフォルトでは、幼児の注視しているオブジェクト、またはランダムなアクティブオブジェクトを選ぶ。
        プロトコル固有の条件がある場合は、各サブクラスでオーバーライドする。
        """
        # 再選択時に前のターゲットを除外リストに追加
        if getattr(self, "target_object", None) is not None:
            old_target_id = self.target_object.id if hasattr(self.target_object, "id") else self.target_object
            self.ignored_targets.add(old_target_id)
            
        body_id = None
        gaze_targets = (infant_signals or self.infant_signals).get("gaze_targets", [])
        
        # 1. 幼児の注視対象を優先して取得
        if len(gaze_targets) > 0:
            # 確率の高い順にソートして、ignored_targetsに含まれないものを探す
            sorted_indices = np.argsort(gaze_targets)[::-1]
            for target_idx in sorted_indices:
                if gaze_targets[target_idx] == 0:
                    continue
                cand_id = None
                if hasattr(self.env, "all_objects"):
                    # リスト形式で取得できる場合
                    if isinstance(self.env.all_objects, list) and 0 <= target_idx < len(self.env.all_objects):
                        cand_id = self.env.all_objects[target_idx]
                    # 辞書形式の場合 (キーのリストから取得)
                    elif isinstance(self.env.all_objects, dict):
                        keys = list(self.env.all_objects.keys())
                        if 0 <= target_idx < len(keys):
                            cand_id = keys[target_idx]
                if cand_id is not None and cand_id not in self.ignored_targets:
                    body_id = cand_id
                    break
        
        # 2. 注視対象がない場合、またはランダム選択（親始動など）の場合はアクティブなオブジェクトから選ぶ
        if body_id is None and hasattr(self.env, "active_objects") and len(self.env.active_objects) > 0:
            valid_objects = [obj_id for obj_id in self.env.active_objects if obj_id not in self.ignored_targets]
            if len(valid_objects) > 0:
                body_id = int(np.random.choice(valid_objects))
            else:
                # 全て除外されてしまった場合のフォールバック（フェールセーフ）
                body_id = int(np.random.choice(self.env.active_objects))
            
        # オブジェクトインスタンスへ解決
        if hasattr(self.env.unwrapped, "all_objects") and body_id in self.env.unwrapped.all_objects:
            self.target_object = self.env.unwrapped.all_objects[body_id]
        else:
            self.target_object = body_id

    def queue_utterance(self, template_key: str, concept_id: str = None):
        """
        発話テンプレートから指定されたキーと引数に基づき、音素キューに追加する。
        """
        # self.name (例: "JointAttentionProtocol") からsnake_caseのプロトコル名を推測
        # または、単純にクラス名から取得。ここでは簡易的にプレフィックス等で取得するか、
        # template_key のみで common や他のプロトコルを探せるようにする。
        # 今回はクラス名から "Protocol" を除いてスネークケースにしたものを利用する手もあるが、
        # manager 側で common へのフォールバックがあるので、そのまま self.name の小文字変換等を渡す。
        #TODO: templatesのキーにあわせた名前をプロトコルに与えるべきでは？
        if not self.parent_utterance:
            return
        import re
        protocol_snake = re.sub(r'(?<!^)(?=[A-Z])', '_', self.name.replace('Protocol', '')).lower()
        
        self.debug_info["last_utterance"] = {"template_key": template_key, "concept_id": concept_id}
        
        phonemes_with_duration = self.language_manager.get_template_phonemes(protocol_snake, template_key, concept_id)
        for phoneme, duration in phonemes_with_duration:
            for _ in range(duration):
                self.utterance_queue.append(phoneme)

    def _emit_utterance(self, action: np.ndarray, volume: float = 0.5) -> np.ndarray:
        """
        utterance_queue の先頭音素を action に反映する．
        "none" または phoneme_mapping に無いラベルは無声（volume=0）として扱う．
        """
        if len(self.utterance_queue) == 0:
            return action
        phoneme = self.utterance_queue.popleft()
        mapping = getattr(self.env, "phoneme_mapping", {})
        if phoneme in ("none", None) or phoneme not in mapping:
            if "volume" in self.env.action_map:
                action[self.env.action_map["volume"]] = 0.0
            return action
        v_coords = mapping[phoneme]
        self.env.apply_phoneme_to_action(action, v_coords, volume=volume)
        return action

    def __call__(self, infant_signals: Dict[str, Any], parent_info: Dict[str, Any]) -> Tuple[np.ndarray, float, Optional[bool], Optional[float]]:
        """
        毎ステップ呼ばれる更新処理。
        内部で状態遷移と行動決定を行い、親の行動ベクトル、受理状態への近さ、成功フラグ、および報酬を返す。
        """
        self.voice_segment.update(infant_signals, self._voice_volume_threshold())

        if self._in_pre_release is None:
            self._in_pre_release = bool(self.uses_pre_release) and self._should_pre_release()

        if self._in_pre_release:
            parent = getattr(self.env, "parent", None) or {}
            held = parent.get("held_obj")
            if held is not None:
                parent_action = self._pre_release_action()
                held = parent.get("held_obj")
                if held is not None or self.is_finished:
                    proximity = self.state_proximities.get("init", 0.0)
                    self._calculate_persistence(infant_signals)
                    success_flag = None
                    if self.is_finished and not self._history_saved:
                        self._history_saved = True
                        if self.state == self.success_state:
                            success_flag = True
                        elif self.state == self.failure_state:
                            success_flag = False
                    reward = 0.0
                    if self.reward_mode:
                        reward = self.reward_function(infant_signals)
                    return parent_action, proximity, success_flag, reward
            self._in_pre_release = False
            self._release_timer = 0

        # 1. 状態管理メソッドの呼び出し（状態更新）
        self._manage_state(infant_signals)
        
        # 2. 行動決定メソッドの呼び出し（状態と情報に応じた行動決定）
        #    戻り値は行動ベクトル (np.ndarray) となる
        parent_action = self._determine_action()
            
        # 3. 現在状態の「受理状態への近さ」を取得
        proximity = self.state_proximities.get(self.state, 0.0)

        # 4. 持続強さ(persistence)を計算
        self._calculate_persistence(infant_signals)

        # 5. 終了状態にあり、発話キューが空ならプロトコルを自動終了する（クリーンアップ）
        if self.state in [self.success_state, self.failure_state] and not self.is_finished:
            if len(self.utterance_queue) == 0:
                self.is_finished = True

        # 6. プロトコル終了時に1回だけ結果フラグを返す（履歴保存の重複防止）
        success_flag = None
        if self.is_finished and not self._history_saved:
            self._history_saved = True
            # completedなどの成功ステートならTrue, failureならFalse, それ以外(途中中断等)はNone
            if self.state == self.success_state:
                success_flag = True
            elif self.state == self.failure_state:
                success_flag = False
                
        # 7. 報酬の計算
        reward = 0.0
        if self.reward_mode:
            reward = self.reward_function(infant_signals)
                
        return parent_action, proximity, success_flag, reward

    def _manage_state(self, infant_signals: Dict[str, Any]) -> None:
        """
        状態管理メソッド
        幼児の信号と親の情報を入力とし、それによって状態(self.state)遷移を決定する。
        サブクラスで必ずオーバーライドする。
        """
        raise NotImplementedError

    def _determine_action(self) -> np.ndarray:
        """
        行動決定メソッド
        現在の状態(self.state)が定義する目標に応じた行動をIKなどで求め、
        行動ベクトル (np.ndarray) として返す。
        サブクラスで必ずオーバーライドする。
        """
        raise NotImplementedError

    def _current_target_id(self) -> Optional[Any]:
        target = getattr(self, "target_object", None)
        if target is None:
            return None
        return target.id if hasattr(target, "id") else target

    def _trackable_argmax_id(
        self, targets: Any, *, ignore_parent: bool = True
    ) -> Optional[Any]:
        if targets is None:
            return None
        target_array = np.asarray(targets, dtype=float)
        if target_array.size == 0 or float(np.max(target_array)) <= 0.0:
            return None
        trackable = list(getattr(self.env, "trackable_targets", []) or [])
        parent_body = None
        parent = getattr(self.env, "parent", None)
        if parent is not None:
            parent_body = parent.get("body")
        order = np.argsort(target_array)[::-1]
        for idx in order:
            if float(target_array[int(idx)]) <= 0.0:
                break
            if int(idx) >= len(trackable):
                continue
            tid = trackable[int(idx)]
            if ignore_parent and parent_body is not None and tid == parent_body:
                continue
            return tid
        return None

    def _infant_attending_current_object(self, infant_signals: Dict[str, Any]) -> bool:
        """提示対象への見る / 触る / 持つ。parent_object_cue の「どれか物」ではなく対象一致。"""
        current_id = self._current_target_id()
        if current_id is None:
            return False
        gaze_id = self._trackable_argmax_id(
            infant_signals.get("gaze_targets", []), ignore_parent=True
        )
        if gaze_id == current_id:
            return True
        infant = getattr(self.env, "infant", None) or {}
        if infant.get("held_obj") == current_id:
            return True
        hand_id = self._trackable_argmax_id(
            infant_signals.get("gesture_targets", []), ignore_parent=True
        )
        if float(infant_signals.get("holding", 0.0)) > 0.0 and hand_id == current_id:
            return True
        if float(infant_signals.get("touching", 0.0)) > 0.0 and (
            hand_id == current_id or gaze_id == current_id
        ):
            return True
        return False

    def _voice_volume_threshold(self) -> float:
        detector = getattr(self.env, "voice_volume_threshold", None)
        if detector is not None:
            return float(detector)
        return float(DEFAULT_VOICE_VOLUME_THRESHOLD)

    def blocks_interrupt(self, infant_signals: Dict[str, Any]) -> bool:
        """締めの発話中など、PM が割り込みくじを飛ばすべきか。既定はしない。"""
        return False

    def keeps_held_object(self, held_id: Any) -> bool:
        """親が持っている物を、この遊びの開始時に持ったままでよいか。既定は手放す。"""
        return False

    def blocks_target_on_failure(self) -> bool:
        """失敗終了したとき、その（遊び, 物）を除外リストに入れるか。手放し失敗は入れない。"""
        return not bool(self._pre_release_failed)

    def _should_pre_release(self) -> bool:
        parent = getattr(self.env, "parent", None) or {}
        held = parent.get("held_obj")
        if held is None:
            return False
        return not self.keeps_held_object(held)

    def _pre_release_action(self) -> np.ndarray:
        """今の手の位置で指を開く。上限を超えたら失敗。held_obj はコードで消さない。"""
        action = np.zeros(self.action_dim, dtype=np.float32)
        parent = getattr(self.env, "parent", None) or {}
        self._release_timer += 1
        if self._release_timer > RELEASE_TIMEOUT_STEPS:
            self._pre_release_failed = True
            self.fail_on_error_timeout()
            return action
        opener = getattr(self.env, "open_hand", None)
        if callable(opener):
            opener(parent, action=action)
        return action

    def history_extra(self) -> Dict[str, Any]:
        """履歴 1 件に足すプロトコル固有の欄。既定は空。"""
        return {}

    def infant_is_engaged(self, infant_signals: Dict[str, Any]) -> bool:
        """幼児がこの遊びに乗っているか。JA / くれ / 二項がオーバーライドする。"""
        return False

    def complete_form_same_target(self, infant_signals: Dict[str, Any]) -> bool:
        """完成形（指差し・リーチ・弱い腕）の手が今の対象と同じか。"""
        from cueworld.intent import pointing_is_complete, reach_is_complete, weak_arm

        current_id = self._current_target_id()
        if current_id is None:
            return False
        if not (
            pointing_is_complete(infant_signals)
            or reach_is_complete(infant_signals)
            or weak_arm(infant_signals)
        ):
            return False
        hand_id = self._trackable_argmax_id(
            infant_signals.get("gesture_targets", []), ignore_parent=True
        )
        return hand_id == current_id

    def _calculate_persistence(self, infant_signals: Dict[str, Any]):
        """
        持続強さ(persistence)を計算する。

        typical_steps がある関わりプロトコルは、乗った証拠と retry_count で
        未乗車／同じ物／リトライ中 2.7、乗車かつ別物 5.0 の 2 値（バッチ E）。
        typical_steps が None なら persistence を書き換えない（見守り 1.0・家事 5.0 のオーバーライドが有効）。
        経過時間の半分は使わない。proximity も使わない。
        """
        if self.typical_steps is None:
            return
        if self.retry_count >= 1:
            self.persistence = MAINTAIN_SCORE_EARLY
            return
        if not self.infant_is_engaged(infant_signals):
            self.persistence = MAINTAIN_SCORE_EARLY
            return
        if self.complete_form_same_target(infant_signals):
            self.persistence = MAINTAIN_SCORE_EARLY
            return
        self.persistence = MAINTAIN_SCORE_LATE
        
    def stop(self):
        """プロトコル中断・終了時のクリーンアップ処理"""
        self.is_finished = True
        print(f"[{self.name}] Protocol stopped.")

    def change_state(self, new_state: str):
        """
        状態を遷移させ、滞在タイマーとリトライカウンタをリセットする。
        """
        self.state = new_state
        self.state_timer = 0
        #TODO ここ確認
        #self.retry_count = 0

    def handle_timeout(self, max_retries: int = 3) -> str:
        """
        失敗系（幼児の行動待ち）タイムアウト時のエスカレーションレベルを判定し返す。

        親モータの張り付き防止（エラー対策）には使わない。そちらは fail_on_error_timeout。
        """
        self.retry_count += 1
        if self.retry_count > max_retries:
            return "FAILED"
        elif self.retry_count == 2:
            return "ESCALATE"
        elif self.retry_count == 3:
            if self.change_target:
                return "CHANGE_TARGET"
            else: 
                return "FAILED"
        else:
            return "RETRY"

    def fail_on_error_timeout(self) -> None:
        """
        エラー対策タイムアウト: 親のみの行動が上限以内に成功せず進行が止まる場合に即失敗終了。

        retry_count / handle_timeout の梯子は触らない。
        """
        print(f"[{self.name}] Error timeout in state={self.state} (timer={self.state_timer})")
        self.change_state(self.failure_state)
        self.is_finished = True

    def detect_rejection(self, infant_signals: Dict[str, Any]) -> bool:
        """
        幼児の急激な拒否シグナルを検知し、即時中断すべきかを判定する。
        """
        if not self.rejection_check:
            return False

        gaze_shift_dict = infant_signals.get("gaze_shift_dict", {})
        negative_surge = infant_signals.get("negative_emotion_surge", 0.0)

        # 1. 情動的な拒否 (急に泣き出した)
        if negative_surge > 0.6:
            print(f"[{self.name}] Rejection detected: Negative Emotion ({negative_surge:.2f})")
            return True

        # 2. 注意の拒否 (インタラクションの対象から強く顔を背けた)
        # 現在のターゲット（対象オブジェクト、および親自身）を「有効なターゲット群」とする
        valid_target_ids = []
        if getattr(self, "target_object", None) is not None:
            if hasattr(self.target_object, "id"):
                valid_target_ids.append(self.target_object.id)
            else:
                valid_target_ids.append(self.target_object)
                
        # プロトコル実行中は親も有効なターゲット
        if hasattr(self.env, "parent") and self.env.parent is not None:
            valid_target_ids.append(self.env.parent["body"])

        # 有効なターゲット全体への視線の変化量の合計を計算
        # 親から対象物へ視線を移しただけなら、親の負の変化と対象物の正の変化が相殺されて拒否にならない
        total_valid_shift = sum(gaze_shift_dict.get(tid, 0.0) for tid in valid_target_ids)
        
        # 有効なターゲット群全体から急激に目を逸らした場合を拒否候補とみなす
        # AR=1 では reach / 把头振りの過渡で視線分布が大きく揺れ、旧 -0.5 だと誤検知しやすい
        # 直近半窓でも有効ターゲットを見ていれば「逸らした」とはみなさない
        # J1: 2 段拒否 — 候補に加え head_turn_away が伴うときのみ確定
        # J1 フォールバック: J1_GAZE_REJECTION_HOLDING_FALLBACK=True で把持中は視線拒否をスキップ
        parent = getattr(self.env, "parent", None)
        target_oid = None
        if getattr(self, "target_object", None) is not None:
            target_oid = (
                self.target_object.id
                if hasattr(self.target_object, "id")
                else self.target_object
            )
        parent_holding_target = (
            parent is not None
            and target_oid is not None
            and parent.get("held_obj") == target_oid
        )
        gaze_reject_candidate = total_valid_shift < -0.95
        head_turn_away = float(infant_signals.get("head_turn_away_intensity", 0.0))
        has_gaze_history = len(gaze_shift_dict) > 0
        if has_gaze_history and not gaze_reject_candidate:
            late_mass = infant_signals.get("gaze_late_mass", {}) or {}
            late_valid = sum(float(late_mass.get(tid, 0.0)) for tid in valid_target_ids)
            if late_valid < 0.15 and head_turn_away > REJECTION_TURN_AWAY_TAU:
                gaze_reject_candidate = True
        if gaze_reject_candidate and not (J1_GAZE_REJECTION_HOLDING_FALLBACK and parent_holding_target):
            late_mass = infant_signals.get("gaze_late_mass", {}) or {}
            late_valid = sum(float(late_mass.get(tid, 0.0)) for tid in valid_target_ids)
            if late_valid < 0.15 and head_turn_away > REJECTION_TURN_AWAY_TAU:
                print(f"[{self.name}] Rejection detected: Sudden Gaze Shift away from valid targets ({total_valid_shift:.2f})")
                return True

        return False

    def reward_function(self, infant_signals: Dict[str, Any]) -> float:
        """
        報酬関数
        プロトコルの目的に応じて、幼児のシグナルと親の情報から報酬を計算して返す。
        サブクラスでオーバーライドする。
        """
        return 0.0

    def dense_reward_function(self, infant_signals: Dict[str, Any]) -> float:
        """切り分けテスト用の密な報酬。既定は 0。学習へ足すのは ProtocolManager(dense_reward=True) のときだけ。"""
        return 0.0