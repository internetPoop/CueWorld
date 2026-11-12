# ImitationProtocol: 音声以外の模倣 実装計画

本ドキュメントは `cueworld/docs/` と現行コード（`protocols/imitation.py`, `infant_signal.py`, `parent_tools.py`, `protocol_manager.py`）を踏まえ、**音声模倣に加えて身体動作の模倣**を `ImitationProtocol` に実装するための手順をまとめたものである。

---

## 0. 設計原則（スコープの境界）

### 0.1 二項関係 vs 三項関係の分担

| 関係 | 担当プロトコル | 模倣の対象例 |
|------|----------------|-------------|
| **二項（dyadic）** | `ImitationProtocol` | バブリング、手振り、うなずき、手の開閉など **相手（幼児）向け** の動き |
| **三項（triadic）** | `JointAttentionProtocol` 等 | 対象物への **指差し**、物に向かって **腕を伸ばす** リーチ、物を見せる行為 |

**指差し・対象物へのリーチなど、ターゲットオブジェクトを伴うジェスチャは本プロトコルでは扱わない。** 共同注意など三項関係のプロトコルが担う。どのプロトコルを起動するかは **`protocol_manager` が幼児シグナルと重み行列に基づいて選択する**ため、`ImitationProtocol` 内部で「三項シグナルが高いから模倣を中止する」といった判定は **行わない**。

`has_target = False` のまま維持し、三項向けツール（`point_object` 等）は使わない。

### 0.2 全関節の自由模倣は対象外

幼児の全関節角を逐次コピーするような **自由模倣はスコープに含めない**（設計・実装ともに考慮不要）。

代わりに、あらかじめ定義した **離散モータチャネル**（手振り、うなずき、手の開閉など）へのマッピングのみを行う。

### 0.3 始動パターンごとの最終目標

| 始動 | 最終目標 |
|------|----------|
| **幼児始動** | 幼児の動きのうち **目立つ 1〜2 個** を検出し、親がそれらを模倣する |
| **親始動** | **ランダムに選んだ 2 要素**（発声・筋肉動作など）を組み合わせて行動し、幼児の反応を待つ |

### 0.4 検知は「状態」ではなく「変化」と「周期性」

模倣のトリガーおよび顕著性スコアは、**現在の姿勢の強度**（例: 手が開いている）ではなく、**動きの変化**（例: 閉じた状態から開く）と **周期性**（例: 手振り・うなずきの繰り返し）に基づく。

| 検知の種類 | 対応するプリミティブ例 | 既存シグナル | 要新規実装 |
|------------|------------------------|--------------|------------|
| **変化（transition）** | 発声の開始（音声チャンク） | 一部（音声） | — |
| **周期性（periodicity）** | 手振り、うなずき、**手の開閉** | `hand_movement_regularity` | 頭のうなずき周期性、手の開閉周期性 |

**対象外**: 腕を伸ばすだけの静的ポーズ（`arm_raise` 等）は模倣プリミティブに含めない。

### 0.5 音声模倣と行動模倣の再生の違い

幼児始動時、親が行う模倣の **時間構造** はモダリティで異なる。

| モダリティ | 幼児の行為 | 親の模倣の仕方 | 終了条件 |
|------------|------------|----------------|----------|
| **音声** (`voice`) | ある **音声の塊**（バブリングの一連） | 検出した音素列を **その塊として再生**（`_queue_phoneme_pattern` で音素＋間を積む） | `utterance_queue` が空になる |
| **周期性** (`arm_wave`, `head_nod`, `hand_open_close`) | しばらく続く周期的な動き（手の開閉の繰り返し含む） | 検出後、**一定ステップ数だけ周期運動を継続**（sin 往復・開閉ループなど） | `duration_steps` 経過（複数周期） |

要点:

- **音声**: 幼児の「そのときの発声チャンク」を真似て **再生し終わったら完了**。継続的なループはしない（現行の `utterance_queue` 方式を踏襲）。
- **行動（周期性）**: 幼児が手振り・うなずき・**手の開閉の繰り返し**を続けているのに対し、親も **しばらく同種の周期運動を続ける**（`duration_steps` で長さを規定。例: 90 ステップ ≒ 数周期分）。手の開閉も一回の遷移アニメではなく、**開→閉→開…のループ**として再生する。

親始動時も同様: 音声は音素列の塊、モータ（手振り・うなずき・手開閉含む）は `duration_steps` 分の周期運動を継続する。

---

## 1. 環境の理解（関連コンポーネント）

### 1.1 全体アーキテクチャ

```
BaseEnv (base.py)
  └─ InfantSignalDetector (infant_signal.py)  … 幼児行動 → 数値シグナル
       └─ ParentTools (parent_tools.py)       … 親の IK・視線・接近
            └─ ProtocolManager               … プロトコル選択・実行
                 └─ ImitationProtocol       … 模倣タスクのステートマシン
```

- プロトコルは `BaseProtocol` を継承し、毎ステップ `_manage_state`（状態遷移）と `_determine_action`（親の行動ベクトル生成）を実行する。
- 模倣は **二項関係（dyadic）** の宣言的コミュニケーションであり、オブジェクトターゲットは持たない（`has_target = False`）。

### 1.2 現行 `ImitationProtocol` の音声模倣フロー

| 始動 | フロー |
|------|--------|
| **幼児始動** (`infant_initiative=True`) | `waiting_infant` → バブリング検知 → `imitating`（親が同音素を発話） → `completed` |
| **親始動** (`infant_initiative=False`) | `vocalizing`（親が音素を発話） → `waiting_infant_response` → 幼児反応 → `acknowledging`（praise） → `completed` |

**検知（音声）**

- 幼児バブリング: `babbling_target` + `voice_volume` + `babbling_ratio` + `voice_target_stability`
- 親の模倣出力: `utterance_queue` に音素列を積み、`_determine_action` で `v_1`, `v_2`, `volume` を設定

**親の身体動作（現状）**

- 常に幼児を見る（`gaze_agent`）
- 腕は `reset_arm_pose` で初期姿勢に戻すのみ
- **`imitating_type` 引数は保存されるが未使用**

### 1.3 幼児シグナルと模倣プロトコルでの扱い

#### 本プロトコルで使わないシグナル（三項・静的）

| シグナル | 備考 |
|----------|------|
| `point_intensity`, `hand_target_stability`, `gesture_targets` | 三項関係向け。模倣の起動は `protocol_manager` が共同注意等へ振り分ける |
| `arm_ext` | リーチ・静的伸展。模倣プリミティブの対象外 |

`ImitationProtocol` 内でこれらを読んで実行を打ち切る処理は **不要**。

#### 現状あるが模倣検知には不十分（静的・瞬間値）

| シグナル | 現状の意味 | 問題点 |
|----------|------------|--------|
| `open_intensity` | 現在「手が開いている」強度 | **開閉の繰り返しリズム**を検出できない |
| `fist_intensity` | 現在「握っている」強度 | **開閉の繰り返しリズム**を検出できない |

`hand_poses` 履歴はあるが、開閉の **周期性スコア**には未変換。

#### 既存で活用可能（動き・周期性）

| シグナル | 意味 | 模倣での用途 |
|----------|------|-------------|
| `hand_movement_size` | 手の速度ノルム | 手振りの **動きの強さ** |
| `hand_movement_regularity` | Y 軸速度の反転回数 | 手振りの **周期性** |
| `babbling_ratio`, `voice_volume` 等 | 音声の反復・音量 | 発声の **変化・周期性**（既存） |
| `orientation_to_parent` | 親への朝向 | 親始動後の **反応判定** のみ |

#### 新規実装が必要なシグナル（既存モダリティへ統合）

| 新シグナル | 所属モダリティ | 検知内容 | 対応プリミティブ |
|------------|----------------|----------|------------------|
| `hand_open_close_periodicity` | **gesture** | 手の開閉（往復）の周期性 [0,1] | `hand_open_close` |

開閉開閉…と閉開閉開…は **位相がずれた同一リズム**のため、検知も模倣チャネルも 1 つに統合する。
| `head_nod_periodicity` | **gaze** | 頭ピッチの往復・周期性 [0,1] | `head_nod` |

`imitation` という独立モダリティは設けない。手の動きは gesture、頭の動きは gaze に割り当てる。

### 1.4 親ツール（模倣で使うもの）

| メソッド | 模倣での用途 | 親の動きの性質 |
|----------|-------------|----------------|
| `gaze_agent` | 幼児への視線（常時） | — |
| `reset_arm_pose` | 中立姿勢への復帰 | — |
| `wave_arm`（新規） | 手振り | **周期性**（肩ピッチ往復） |
| `nod_head`（新規） | うなずき | **周期性**（頭ピッチ往復） |
| `animate_hand_open_close`（新規） | 手の開閉ループ | **周期性**（phase で開く相・閉じる相を交互にループ） |

静的ポーズの維持（腕を伸ばしたまま等）は行わない。親も幼児と同様 **周期のある動き** でお手本を見せる。手の開閉は `duration_steps` の間、開く相・閉じる相を交互にループする。

**使わないもの**: `point_object`, `hand_reach_object`, `gaze_object`, `raise_arm`（いずれも三項 or 静的伸展）

---

## 2. 実装方針

### 2.1 モータチャネルと検知の対応（周期性中心）

各モータプリミティブは **周期性検知** と 1 対 1 で対応させる。音声のみ **変化（チャンク再生）** を持つ。

| チャネル ID | 親が行う動き | 検知タイプ | 幼児側シグナル（顕著性） | 状態 |
|-------------|-------------|------------|--------------------------|------|
| `voice` | 発声（音素列） | 変化＋周期 | `babbling_ratio`, `voice_volume`, `voice_target_stability` | 既存 |
| `arm_wave` | 手振り | **周期性** | `hand_movement_size` × `hand_movement_regularity` | 既存 |
| `head_nod` | うなずき | **周期性** | `head_nod_periodicity`（新規） | 要実装 |
| `hand_open_close` | 手の開閉の繰り返し | **周期性** | `hand_open_close_periodicity`（新規） | 要実装 |

**スコープ外（削除）**: `arm_raise` など腕を伸ばすだけの静的動作はプリミティブに含めない。

`imitating_type` はデバッグ・デモ用。本番は幼児始動で salience 自動選択、親始動でランダム 2 要素。

### 2.1.1 新規シグナルの算出方針（`infant_signal.py`）

**手の開閉周期性**（`hand_open_close_periodicity`）

手振り・うなずきと同様、**一回の遷移**ではなく **開閉の往復リズム**を検出する。`index_joint` の双方向の角速度往復と、姿勢ラベルの `open`↔`closed` 交替を併用する。

```python
# index_joint の速度符号反転回数 → 開閉の往復（位相に依存しない）
index_vel = diff(index_joint_history)
hand_open_close_periodicity = zero_crossings(index_vel) * clip(abs(index_vel[-1]) / vmax, 0, 1)

# 姿勢ラベルの open↔closed 交替回数も併用
alternations = count_pose_switches(hand_pose_labels, {"open", "closed"})
```

- 静的に「開いている」「握っている」だけではスコア 0
- **開閉開閉** も **閉開閉開** も同じスコア（位相の違いは salience に影響しない）
- 模倣チャネルも `hand_open` / `hand_close` ではなく **`hand_open_close` 1 本**に統合

**うなずき周期性**（`head_nod_periodicity`）

頭ピッチ角の履歴（新規 `head_pitch_history`）を追加し、手振りと同様の手法を適用する。

```python
# head_tilt_joint の履歴から
pitch_vel = diff(head_pitch_history)
zero_crossings = count_sign_changes(pitch_vel)
head_nod_periodicity = clip(zero_crossings / K, 0, 1)
# 動きの強さ: abs(pitch_vel[-1]) を掛け合わせ
head_nod_salience = head_nod_periodicity * clip(abs(pitch_vel[-1]) / vmax, 0, 1)
```

**顕著性の原則**: 各チャネルの salience = **変化の大きさ**（または周期の強度）× モダリティ固有のゲイン。静的 `*_intensity` は salience 計算に使わない。

### 2.1.2 シグナル計算と salience ランキングの分担

| 層 | 担当 | 内容 |
|----|------|------|
| **生シグナル** | `infant_signal.py` | モダリティ別の客観的計測値（`hand_open_close_periodicity` は gesture、`head_nod_periodicity` は gaze など） |
| **プロトコル選択** | `protocol_manager.py` | 重み付き和で「模倣プロトコルを起動するか」を判定 |
| **模倣チャネル選択** | `ImitationProtocol.get_salient_channels()` | 起動後に「何を真似るか」を salience ランキングで決定 |

`salient_motor_channels` は `get_normalized_signals()` には含めない。模倣プロトコルが選択された後、`waiting_infant` などの状態遷移の中で `get_salient_channels(infant_signals)` を呼ぶ。

**理由**: salience の合成（音声 × 手振り × 手開閉を一括ランク付け）は模倣タスク固有の解釈であり、他プロトコル（共同注意・物体遊び等）では不要。一方、周期性シグナル自体は `protocol_manager` の `weights_dict["imitation"]` でも参照するため、`infant_signal.py` に残す。

### 2.2 幼児始動: 目立つ 1〜2 個の模倣

**フロー**

```
waiting_infant
  → ImitationProtocol.get_salient_channels(infant_signals) でランク付け
  → 上位 1〜2 個を motor_queue / utterance_queue に積む
  → imitating（親が順に再生）
  → completed
```

**顕著性スコア（`ImitationProtocol.get_salient_channels` 内）**

```python
salience = {
    # 音声: voice モダリティの既存シグナルから合成
    "voice":        babbling_ratio * voice_volume * (1 - voice_target_stability),

    # 周期性: gesture / gaze モダリティのシグナルから合成
    "arm_wave":     hand_movement_size * hand_movement_regularity,
    "head_nod":     head_nod_periodicity,            # gaze
    "hand_open_close": hand_open_close_periodicity,  # gesture
}

# しきい値以上のものだけ残し、上位 max 2 件
top_k = top_k_above_threshold(salience, k=2, thresh=0.2)
```

プロトコルが起動された時点で模倣対象とみなす。三項シグナルによる salience のマスクは **行わない**（起動判断は `protocol_manager` 側の責務）。

- 1 個だけがしきい値を超える場合は 1 個のみ模倣
- 音声と身体が同時に目立つ場合は **最大 2 チャネル**（例: バブリング + 手振り）
- 複数チャネルがある場合: **音声は塊再生後に終了**、**周期性モータは `duration_steps` の間ずっと継続**（同時並行可）

### 2.2.1 再生キューへの積み方（幼児始動）

```python
# 音声: 検出した音素の塊をそのままキューに（現行踏襲）
if "voice" in top_k:
    self._queue_phoneme_pattern(babbling_target)

# 周期性モータ: チャネル名だけ積み、再生中は phase を回し続ける
if "arm_wave" in top_k:
    self.motor_queue.append(("arm_wave",))  # duration_steps まで wave_arm を継続
if "hand_open_close" in top_k:
    self.motor_queue.append(("hand_open_close",))  # duration_steps まで開閉ループを継続
```

### 2.3 親始動: ランダム 2 要素の組み合わせ

親が先に「お手本」を見せ、幼児の反応を待つ。

**要素プール**

```python
MOTOR_CHANNELS = ["arm_wave", "head_nod", "hand_open_close"]
VOICE_CHANNEL  = "voice"
```

**毎回のサンプリング**

```python
# 例: 発声 + モータ 1 つ、または モータ 2 つ
if random() < 0.5:
    channels = ["voice", random.choice(MOTOR_CHANNELS)]
else:
    channels = random.sample(MOTOR_CHANNELS, k=2)
```

- `voice` が選ばれた場合: ランダム音素列（またはテンプレート）を `utterance_queue` に積む
- モータが選ばれた場合: 対応プリミティブを `motor_queue` に積む
- 2 要素を **同時並行** で `_determine_action` に反映（発声しながら手を振る、など）

**幼児の成功反応**: 現行 `_detect_infant_response` と同様、厳密な同型模倣は要求しない。バブリング・手の動き・腕伸展・親への orienting のいずれかで十分。

### 2.4 状態マシン

```
共通:
  init
    → waiting_infant        （幼児始動）
    → performing_parent_cue （親始動: 2 要素を再生）
  performing_parent_cue → waiting_infant_response
  waiting_infant_response → acknowledging → completed
  imitating → completed     （幼児始動: 1〜2 チャネル模倣完了後）

キュー:
  utterance_queue  … 音声チャネル
  motor_queue      … モータチャネル（arm_wave 等）
```

後方互換のため `vocalizing` は `performing_parent_cue` のエイリアスとして当面残してよい。

---

## 3. 新規データ構造

### 3.0 チャネル定義の置き場所

モータチャネルはすべて **周期性検知 + `duration_steps` 継続再生**であり、音声のみ `utterance_queue` によるチャンク再生が例外。

| モジュール | 定義内容 |
|------------|----------|
| `infant_signal.py` | `PERIODICITY_WINDOW`、周期性シグナル計算（gesture / gaze に統合） |
| `imitation.py` | `VOICE_CHANNEL`, `MOTOR_CHANNELS`, `SALIENCE_GLOBAL_THRESH`, `MOTOR_CHANNEL_CONFIG`（Step 3）, `get_salient_channels()` |

### 3.1 モータチャネルと親ツール（`imitation.py` で定義予定）

```python
# 音声以外のモータはすべて周期性 + duration_steps 継続
MOTOR_CHANNEL_CONFIG = {
    "arm_wave":    {"duration_steps": 90, "parent_action": "wave_arm"},
    "head_nod":    {"duration_steps": 60, "parent_action": "nod_head"},
    "hand_open_close": {"duration_steps": 60, "parent_action": "animate_hand_open_close"},
}
# 音声は utterance_queue でチャンク再生（上記には含めない）
```

各チャネルは **親が実行するレシピ**（周期アニメーション）であり、幼児の関節角の直接コピーではない。

### 3.2 プロトコル内キュー

```python
self.utterance_queue = deque()   # 音声（既存）
self.motor_queue = deque()       # ("arm_wave",) などタプル
self.motor_timer = 0
self.active_channels = []      # 親始動時に選んだ 2 要素
self.salient_channels = []     # 幼児始動時に検出した 1〜2 要素
```

### 3.3 幼児シグナル拡張（`infant_signal.py`）

**追加する履歴バッファ**

```python
self.head_pitch_history = deque(maxlen=history_length)   # gaze: head_tilt_joint 角
self.hand_pose_labels   = deque(maxlen=history_length)   # gesture: 姿勢ラベル
self.index_joint_history = deque(maxlen=history_length)  # gesture
self.thumb_joint_history  = deque(maxlen=history_length) # gesture
```

履歴更新は `update()` 内でモダリティに応じて行う:

- `gesture` 有効時 → `_update_hand_periodicity_histories()`
- `gaze` 有効時 → `_update_head_periodicity_histories()`

**追加する計算メソッド（`infant_signal.py`）**

| メソッド | 出力 | 公開先 |
|----------|------|--------|
| `compute_hand_open_close_periodicity()` | `hand_open_close_periodicity` [0,1] | `get_gesture_signals()` |
| `compute_head_nod_periodicity()` | `head_nod_periodicity` [0,1] | `get_gaze_signals()` |

**模倣チャネル salience ランキング（`imitation.py`）**

| メソッド | 出力 |
|----------|------|
| `ImitationProtocol.get_salient_channels(infant_signals, max_channels=2)` | `[(channel, salience), ...]` |

`get_normalized_signals()` には周期性シグナルが gesture / gaze 経由で含まれる。`salient_motor_channels` は含めない。

---

## 4. 実装手順（詳細）

### Step 1: プリミティブ定義と顕著性検出

| 作業 | ファイル | 内容 |
|------|----------|------|
| 1-1 | `infant_signal.py` | `PERIODICITY_WINDOW`、履歴バッファ（gesture / gaze 別更新） |
| 1-2 | `infant_signal.py` | `compute_hand_open_close_periodicity()` → `get_gesture_signals()` に統合 |
| 1-3 | `infant_signal.py` | `compute_head_nod_periodicity()` → `get_gaze_signals()` に統合 |
| 1-4 | `imitation.py` | `get_salient_channels()`（salience ランキング） |
| 1-5 | `docs/infant_signal.md` | モダリティ分担・役割分担の説明 |

**変化・周期性のしきい値（初期値）**

```python
PERIODICITY_WINDOW = 5          # 周期性検出に見る履歴ステップ数（infant_signal.py）
SALIENCE_GLOBAL_THRESH = 0.2    # salience がこれ未満なら模倣トリガーにしない（imitation.py）
```

### Step 2: 親ツールにモータ実行メソッドを追加

| 作業 | ファイル | 内容 |
|------|----------|------|
| 2-1 | `parent_tools.py` | `wave_arm(agent, action, phase)` — 周期性 |
| 2-2 | `parent_tools.py` | `nod_head(agent, action, phase)` — 周期性 |
| 2-3 | `parent_tools.py` | `animate_hand_open_close(agent, action, phase)` — 開閉ループ（phase で開く相・閉じる相を交互に） |

| 2-4 | `docs/parent_tools.md` | 模倣用モータメソッドの説明 |

模倣中は `reset_arm_pose` を呼ばない。複数チャネル同時実行時は action ベクトルへ加算的に書き込む（`_apply_joint_target`）。

### Step 3: `ImitationProtocol` のリファクタリング

| 作業 | 内容 |
|------|------|
| 3-1 | 幼児始動: `get_salient_channels(infant_signals)` で 1〜2 チャネルを `motor_queue` / `utterance_queue` に積む |
| 3-2 | 親始動: `start()` で 2 要素をランダムサンプリングし `active_channels` に保存 |
| 3-3 | `_manage_state`: `waiting_infant` で salience 検出 → `imitating` |
| 3-4 | `_determine_action`: 音声キュー + モータキューを並行適用 |

```python
def _determine_action(self):
    action = np.zeros(self.action_dim, dtype=np.float32)
    self.env.gaze_agent(self.env.parent, self.env.infant, yaw_link="head", action=action)

    if self._has_active_motor():
        self._apply_motor_channels(action)
    else:
        self.env.reset_arm_pose(self.env.parent, action=action)

    self._apply_utterance_action(action)
    return action
```

| 3-5 | 完了判定: 音声は `utterance_queue` 空、モータは各チャネルの `duration_steps` 経過 |

### Step 4: `protocol_manager.py` の連携

| 作業 | 内容 |
|------|------|
| 4-1 | `check_for_response` の imitation 分岐: `target_phoneme` 等を context に渡す（salience はプロトコル内で計算） |
| 4-2 | `weights_dict["imitation"]`: gesture / gaze / voice の周期性シグナルを重みに使用 |

```python
"imitation": {
  # 音声（voice モダリティ）
  "voice_volume": 0.4, "babbling_ratio": 0.5, "voice_target_stability": -0.4,
  # 手振り・手開閉（gesture モダリティ）
  "hand_movement_size": 0.20, "hand_movement_regularity": 0.20,
  "hand_open_close_periodicity": 0.15,
  # うなずき（gaze モダリティ）
  "head_nod_periodicity": 0.15,
  "orientation_to_parent": 0.10, "proximity_intensity": 0.10,
}
```

### Step 5: 言語テンプレート（任意）

親始動で `voice` が選ばれたときの音素列。既存 `prompt` / ランダム音素で足りる場合は追加不要。

### Step 6: テスト・デモ

| 作業 | 内容 |
|------|------|
| 6-1 | `cueworld/demo/demo_imitation_motor.py`（新規） |
| 6-2 | 幼児スクリプト: **手の開閉繰り返し** / **手振り（周期）** / **バブリング+手振り** など |
| 6-3 | 親始動デモ: ランダム 2 要素（例: 発声塊+手振り継続、手開閉ループ+うなずき継続） |
| 6-4 | `docs/demo.md` に「案F: 二項モータ模倣」を追記 |

---

## 5. ファイル変更一覧

| ファイル | 変更種別 | 概要 |
|----------|----------|------|
| `protocols/imitation.py` | 修正 | `get_salient_channels()` / salience 1〜2 / 親始動 2 要素 / キュー並行 |
| `infant_signal.py` | 修正 | 周期性シグナルを gesture / gaze に統合 |
| `parent_tools.py` | 修正 | `wave_arm`, `nod_head`, `animate_hand_open_close` |
| `protocol_manager.py` | 修正 | context 渡し・重み（プロトコル選択はここで完結） |
| `docs/infant_signal.md` | 修正 | 新シグナル説明 |
| `cueworld/demo/demo_imitation_motor.py` | 新規 | デモ |

---

## 6. 実装順序（推奨）

```
[Step 1] infant_signal.py（gesture/gaze 周期性）+ imitation.py（get_salient_channels）
    ↓
[Step 2] parent_tools（wave_arm から。指差し系は実装しない）
    ↓
[Step 3] imitation.py
         - 幼児始動: salience 上位 1〜2 チャネル模倣
         - 親始動: ランダム 2 要素
    ↓
[Step 4] protocol_manager 連携
    ↓
[Step 6] デモ（手開閉ループ → 手振り継続 → 音声塊+手振り同時）
    ↓
[追加] head_nod 周期性
```

**MVP（最初の縦スライス）**

1. **周期性（手開閉）**: 幼児が手を開閉し続ける → `hand_open_close_periodicity` → 親が **60 ステップ程度** 開閉ループを継続 → `completed`
2. **周期性（手振り）**: 幼児が手を振る → salience で `arm_wave` → 親が **90 ステップ程度** `wave_arm` を継続 → `completed`
3. **音声+周期**: バブリング塊の再生と手振り継続が同時に走り、それぞれの終了条件で止まる

---

## 7. 注意点

### 7.1 三項関係との分担（protocol_manager 側）

指差し・物へのリーチが強いときは **`protocol_manager` が `joint_attention` 等を選択**する。`ImitationProtocol` が起動したら、内部では salience に基づき模倣を実行するだけでよい。プロトコル内での三項シグナル監視・打ち切りは **実装しない**。

### 7.2 音声とモータの終了タイミングの違い

`imitating` 状態では、音声キューが先に空になってもモータは `motor_timer < duration_steps` の間 **動き続ける**。逆も同様。`completed` へは **積んだチャネルがすべて各自の終了条件を満たしたとき**。

### 7.3 静的強度と変化の混同を避ける

`open_intensity` / `fist_intensity` は割り込み重み計算など別用途に残してよいが、**模倣の salience には使わない**。「開いている」≠「開閉を繰り返している」。

### 7.4 親始動の 2 要素の同時実行

`voice` + `arm_wave` など、1 ステップの `action` ベクトルに両方の成分を書き込む。キュー長・duration が異なる場合は、短い方が終わっても長い方は継続する。

### 7.5 視線

模倣中も基本は `gaze_agent(parent, infant)`。対象物を見る `gaze_object` は使わない。

### 7.6 `_manage_state` の分岐

`archive/todo.md` と同様、タイムアウト待ちと salience 検知は独立 `if` で並列判定を検討する。

---

## 8. 成功判定

| モード | 成功条件 |
|--------|----------|
| 幼児始動 | 積んだ各チャネルが終了: 音声＝キュー空、モータ＝`duration_steps` 継続完了（手開閉含む） |
| 親始動 | 2 要素のお手本再生後、幼児が `_detect_infant_response` を満たす |

評価モードでは、模倣チャネル数（1 or 2）と幼児反応の有無を記録する。対象物一致率の評価は **行わない**（三項の領域）。

---

## 9. 参考: `_apply_motor_channels` の骨格

```python
def _apply_motor_channels(self, action):
    for channel in self._channels_in_progress():
        phase = self._phase(channel)  # 0..1 over duration
        if channel == "arm_wave":
            self.env.wave_arm(self.env.parent, action=action, phase=phase)
        elif channel == "head_nod":
            self.env.nod_head(self.env.parent, action=action, phase=phase)
        elif channel == "hand_open_close":
            self.env.animate_hand_open_close(self.env.parent, action=action, phase=phase)
    self.motor_timer += 1
```

---

## 10. 関連ドキュメント

- [protocol.md](docs/protocol.md) — プロトコル全般（模倣 vs 共同注意の分担）
- [infant_signal.md](docs/infant_signal.md) — 幼児シグナル定義
- [parent_tools.md](docs/parent_tools.md) — 親の行動生成
- [base.md](base.md) — BaseEnv のメソッド一覧


