# プロトコル共通契約 (`protocols/base.py`)

プロトコルは「親が幼児と何をしているか」を表す状態機械（FSM）である。
`ProtocolManager` が 1 つを選んで走らせ、プロトコルは毎 step **親の行動ベクトル 1 本**を返す。

本ドキュメントは `BaseProtocol` の**共通契約の正本**である。個々の FSM は各ファイルを参照。

| プロトコル | 内容 |
|-----------|------|
| [dyadic.md](dyadic.md) | 二項的やりとり（模倣） |
| [joint_attention.md](joint_attention.md) | 共同注意 |
| [object_giving.md](object_giving.md) | 物を渡す |
| [household_solo.md](household_solo.md) | 家事（幼児を相手にしない） |
| [observance.md](observance.md) | 見守り |

親の意思決定（どれをいつ選ぶか、割り込むか）は [../protocol_manager.md](../protocol_manager.md)。

条件分岐だけの状態遷移図は [automata/](automata/README.md)。

---

## 1. インターフェース

```python
protocol.start(context)                                   # 開始時に 1 回
action, proximity, success_flag, reward = protocol(infant_signals, parent_info)   # 毎 step
```

### `start(context)`

`ProtocolManager` が渡す `context` の中身。

| キー | 意味 |
|------|------|
| `infant_initiative` | 幼児の訴えから始まったか |
| `infant_signals` | 開始時点のシグナル（ターゲット選択に使う） |
| `global_ignored_targets` | PM が持つ除外リスト。`ignored_targets` の初期値になる |
| `fixed_target` | デモ・テストで対象を固定するとき |
| `intent_target_id` | PM が訴えから決めた対象 body id |

`has_target = True` のプロトコルは、この順でターゲットを決める。

1. `fixed_target` があればそれ
2. `intent_target_id` があり、除外リストに無く、`all_objects` に存在すればそれ
3. どちらもなければ `select_target()`

決まったら `target_concept`（概念 ID）と `target_word`（日本語表記、ログ用）も引く。

### `__call__(infant_signals, parent_info)`

毎 step の処理順。

1. `voice_segment` を更新（幼児の発話区間追跡）
2. **開始時の手放し**（[§4](#4-開始時の手放し)）。手放し中はここで打ち切って返る
3. `_manage_state(infant_signals)` — 状態遷移。サブクラスが実装
4. `_determine_action()` — 行動ベクトル生成。サブクラスが実装
5. `proximity = state_proximities[state]` を引く
6. `_calculate_persistence(infant_signals)`（[§5](#5-persistence)）
7. 終了状態にあり、かつ**発話キューが空**なら `is_finished = True`
8. `success_flag` を 1 回だけ返す（`success_state` なら `True`、`failure_state` なら `False`、それ以外は `None`）
9. `reward_function(infant_signals)`（`reward_mode=True` のとき）

**7 が重要**: 状態が `completed` / `failure` に入っても、発話キューが残っている限り終わらない。締めの「すごいね」を言い切ってから終わる。PM の割り込みもこの間は保留されうる（[§8](#8-pm-へ渡すフック)）。

返り値の意味。

| 位置 | 名前 | 意味 |
|------|------|------|
| 1 | `action` | 親の行動ベクトル（`env.action_num` 次元） |
| 2 | `proximity` | 現在状態の「受理状態への近さ」[0, 1]。PM が進捗の目安に使う |
| 3 | `success_flag` | 終了時に 1 回だけ `True` / `False`。それ以外は `None` |
| 4 | `reward` | 報酬。既定の `reward_function` は 0.0 を返す |

---

## 2. サブクラスが定義するもの

### 必須

| メンバ | 内容 |
|--------|------|
| `_manage_state(infant_signals)` | 状態遷移。未実装なら `NotImplementedError` |
| `_determine_action()` | 行動ベクトル生成。未実装なら `NotImplementedError` |
| `state_proximities` | `状態名 → [0, 1]`。定義されていない状態は 0.0 |

### 分類フラグ

`ProtocolManager` がカタログを絞り込むのに使う。

| フラグ | 意味 |
|--------|------|
| `level` | 対応する発達段階。PM の候補フィルタに使う |
| `urgency` | 緊急度。社会的コミュニケーションかどうかの区別 |
| `has_target` | 物体を対象に取るか |
| `response_type` | 幼児の訴えに応じて始められるか |
| `request_type` | 親から自発的に始められるか |
| `typical_steps` | 成功 1 回の典型長さ [step]。`None` なら基底の persistence 式が働かない |

現行カタログの値。

| プロトコル | `level` | `has_target` | `response_type` | `request_type` | `typical_steps` |
|-----------|---------|--------------|-----------------|----------------|-----------------|
| `DyadicProtocol` | 0 | ✗ | ✓ | ✓ | 105 |
| `ObjectGivingProtocol` | 1 | ✓ | ✓ | ✓ | 175 |
| `JointAttentionProtocol` | 2 | ✓ | ✓ | ✓ | 130 |
| `HouseholdSoloProtocol` | 0 | ✗ | ✗ | ✗ | `None`（`persistence = 5.0` を直接設定） |
| `ObservanceProtocol` | 0 | ✗ | ✗ | ✗ | `None`（`persistence = 1.0` を直接設定） |

### 任意のオーバーライド

| メソッド | 既定 | 用途 |
|----------|------|------|
| `select_target(infant_signals)` | 視線の最大 → アクティブ物からランダム | 対象の選び方 |
| `infant_is_engaged(signals)` | `False` | 幼児が乗っているか。persistence が読む |
| `keeps_held_object(held_id)` | `False`（手放す） | 開始時に持っている物を持ち続けてよいか |
| `blocks_interrupt(signals)` | `False` | PM の割り込みくじを飛ばすか |
| `blocks_target_on_failure()` | 手放し失敗でなければ `True` | 失敗時に（遊び, 物）を除外リストへ入れるか |
| `history_extra()` | `{}` | 履歴 1 件に足す固有の欄 |
| `reward_function(signals)` | `0.0` | 報酬 |

---

## 3. ターゲット選択

`select_target()` の既定手順。

1. 前のターゲットがあれば `ignored_targets` に追加する（再選択は必ず別の物になる）
2. `gaze_targets` を降順に見て、確度が 0 でなく除外リストに無い最初の物
3. 見つからなければ `active_objects` から除外リストを引いてランダム
4. それも空なら（フェールセーフ）除外を無視して `active_objects` からランダム

`ignored_targets` はプロトコル内で閉じたローカルな除外リストで、開始時に PM のグローバル除外リストで初期化される。

### 対象 ID の解決ヘルパー

- `_current_target_id()`: `target_object` から body id を取り出す。インスタンスでも int でもよい
- `_trackable_argmax_id(targets, ignore_parent=True)`: 対象ベクトルの最大要素に対応する body id。全要素 0 なら `None`。`ignore_parent=True` なら親をスキップして次点を返す
- `_infant_attending_current_object(signals)`: 幼児が**今の対象に**注意を向けているか。視線が一致 / 幼児が持っている / 手が一致して `holding` / 手か視線が一致して `touching`、のいずれか

`_infant_attending_current_object` は `intent.parent_object_cue`（「どれか物に注意している」）とは違い、**対象の一致まで見る**。

---

## 4. 開始時の手放し

`uses_pre_release = True`（既定）のプロトコルは、開始時に親が物を持っていたら**まず手放してから**本体の FSM に入る。

- `keeps_held_object(held_id)` が `True` を返す物は持ったまま進む。共同注意で「その物を見せる」場合や、家事で「今運んでいる物」がこれに当たる。
- 手放しは `env.open_hand` を呼んで指を開くだけ。`held_obj` をコードで消すことはしない（`BaseEnv` の解放処理に任せる）。
- 手放している間は状態遷移も行動生成も走らない。`proximity` は `state_proximities["init"]` を返す。
- `RELEASE_TIMEOUT_STEPS`（30 step ≒ 3 秒）以内に離れなければ `_pre_release_failed = True` を立てて即失敗終了する。この失敗は `blocks_target_on_failure()` が `False` を返すので、**除外リストには入らない**（物のせいではないため）。

`ObservanceProtocol` だけ `uses_pre_release = False`。見守りは毎 step `reset_arm_pose` を呼ぶので、その過程で指が開く。

---

## 5. persistence

`persistence` は「この遊びをどれだけ守りたいか」を表す 1 つの数値で、PM が割り込みくじの重みに使う。

`typical_steps` が設定されているプロトコル（関わり系の 3 つ）は、`_calculate_persistence` が毎 step **2 値**で上書きする。

| 条件（上から順に判定） | `persistence` |
|-----------------------|---------------|
| `retry_count >= 1`（リトライ中） | 2.7（`MAINTAIN_SCORE_EARLY`） |
| 幼児が乗っていない（`infant_is_engaged` が偽） | 2.7 |
| 完成形の手が**今の対象と同じ**（`complete_form_same_target`） | 2.7 |
| 上のどれでもない（乗っていて、別の物を訴えている／訴えていない） | 5.0（`MAINTAIN_SCORE_LATE`） |

- **経過時間も `proximity` も使わない。** 途中まで進んだから守る、ではなく「乗っているか」だけで決まる。
- 3 番目の条件が入っているのは、**今やっている物を幼児が訴えているなら、割り込ませて応じたほうがよい**から。同じ物を指差し続けているのに親が別の段取りを続けるのは不自然。
- `typical_steps` が `None` のプロトコル（家事・見守り）は基底式が何もしないので、各プロトコルが `persistence` を直接設定する（家事 5.0、見守り 1.0）。

`complete_form_same_target` は `intent.py` の完成形判定（指差し / リーチ / 弱い腕）を再利用し、`gesture_targets` の最大が現在の対象と一致するかを見る（[../intent.md](../intent.md)）。

---

## 6. タイムアウトとエスカレーション

2 種類のタイムアウトがあり、**用途が違う**。

### `handle_timeout(max_retries=3) -> str`

**幼児の行動を待って空振りしたとき**に呼ぶ。`retry_count` を進めて、次に何をするかを文字列で返す。

| `retry_count` | 返り値 | 意味 |
|---------------|--------|------|
| 1 | `"RETRY"` | 同じキューをもう一度 |
| 2 | `"ESCALATE"` | キューを強める |
| 3 | `"CHANGE_TARGET"`（`change_target=True` のとき）／ `"FAILED"` | 対象を変える |
| `> max_retries` | `"FAILED"` | 諦める |

返り値をどう解釈するかはプロトコル側の自由で、実際に `"ESCALATE"` を使い分けているのは共同注意だけ。多くは `max_retries=1` を渡して「1 回リトライしたら失敗」にしている。

### `fail_on_error_timeout()`

**親のモータ動作が上限内に終わらないとき**に呼ぶ。IK が届かない、接近が収束しない、といった進行不能の検出用。

- 即座に `failure_state` へ遷移して `is_finished = True`
- `retry_count` には触らない。エスカレーションの梯子とは無関係

「幼児が応じてくれない」と「親が動けない」を混ぜないための分離。前者はやりとりの結果として意味があるが、後者は単なる不具合なので学習の材料にしてはいけない。

### `change_state(new_state)`

状態を変えて `state_timer` を 0 に戻す。`retry_count` は**リセットしない**（意図的。エスカレーションが状態遷移でリセットされると梯子が登れない）。

---

## 7. 拒否検知

`detect_rejection(infant_signals) -> bool`。`rejection_check=False` なら常に `False`。

`ProtocolManager` はコンストラクタ引数 `rejection`（既定 `False`）を、共同注意・二項・受け渡しの `rejection_check` に渡す。既定ではこの 3 経路は走らない。`rejection=True` のとき、その 3 本は毎 step 呼び、どれかが立てば拒否になる。見守りと家事は呼ばない。

2 つの経路があり、どれかが立てば拒否。

### 1. 情動的な拒否

`negative_emotion_surge > 0.6`。現行では常に 0.0 なので発火しない（[../infant_signal.md](../infant_signal.md)）。

### 2. 注意の拒否（2 段判定）

「有効なターゲット群」＝ 現在の対象物 ＋ 親の body。

まず**候補**になるかを見る。

- `gaze_shift_dict` の有効ターゲット分の合計が `-0.95` 未満（急激に目を逸らした）
- または、視線履歴があり、直近半窓の有効ターゲットへの視線質量が 0.15 未満で、かつ `head_turn_away_intensity > 0.5`

候補になったら、**確定条件**として次を要求する。

- 直近半窓の有効ターゲットへの視線質量が 0.15 未満（いまも見ていない）
- かつ `head_turn_away_intensity > 0.5`

視線だけでは確定しない 2 段構えにしてあるのは、リーチや首振りの過渡で視線分布が大きく揺れ、1 段だと誤検知が多かったため。

有効ターゲットを**合計**で見るのは、親から対象物へ視線を移しただけのとき、親の負の変化と物の正の変化が相殺されて拒否にならないようにするため。

---

## 8. PM へ渡すフック

### `blocks_interrupt(infant_signals)`

`True` を返す間、PM は割り込みくじを引かない。締めの発話中など、途中で切ると不自然な区間を守るために使う。オーバーライドしているのは共同注意と二項のみ。

### `keeps_held_object(held_id)`

[§4](#4-開始時の手放し) 参照。

### `blocks_target_on_failure()`

失敗終了したとき、その（遊び, 物）の組を PM のグローバル除外リストへ入れるか。既定は入れる。ただし開始時の手放しに失敗した場合だけは入れない。

### `history_extra()`

PM が履歴を 1 件保存するときに足す固有の欄。評価スクリプトが読む。

---

## 9. 共通ユーティリティ

### `AttentionHoldTracker`

対象への**途切れない注視**と、一度見たあとの**離脱**を数える。共同注意の成功判定に使う。

毎 step `step(infant_target, object_id, parent_id)` を呼ぶと `"success"` / `"timeout"` / `"pending"` を返す。

| 幼児が見ているもの | hold | disengage | 結果 |
|-------------------|------|-----------|------|
| 対象物 | +1 | 0 にリセット | `hold >= 30` なら `"success"` |
| 親の顔 | **据え置き** | 0 にリセット | 一度も見ていなければ never-look タイマーが進む |
| それ以外 | 0 にリセット | 一度でも見ていれば +1 | `disengage >= 50` なら `"timeout"` |

**親の顔を見ている間は hold を据え置く**のが要点。共同注意では幼児が対象と親の顔を交互に見るのが自然で、それを離脱と数えてはいけない。ただし一度も対象を見ていないなら、親の顔を見ていても待ち時間は進む。

| 定数 | 値 | 意味 |
|------|-----|------|
| `ATTENTION_HOLD_STEPS` | 30（≒3 秒） | 成功に必要な連続注視 |
| `ATTENTION_WAIT_STEPS` | 50（≒5 秒） | 待ち／離脱の上限 |

### `VoiceSegmentTracker`

幼児の発話区間。`voice_volume` が閾値以上で開始し、無音が `VOICE_SEGMENT_SILENCE_STEPS`（4 フレーム）続いたら終了する。`VoiceRecognizer.offset_silence_frames` と同じ値で、認識器の区間と揃えてある（[../language.md §6](../language.md#6-発話区間の切り出し)）。

すべてのプロトコルが `self.voice_segment` として持ち、`__call__` の先頭で毎 step 更新される。共同注意の締め判定と、二項の割り込みブロックが読む。

### 発話

`queue_utterance(template_key, concept_id)` と `_emit_utterance(action, volume)`。テンプレートの展開規則と送出の契約は [../language.md §3](../language.md#3-テンプレート)。

---

## 10. 注意事項

**`success_state` の名前は統一されていない。**
既定は `"completed"` だが、一部のプロトコルはタイムアウトの結末も `"completed"` に落としている。`success_flag = True` が「幼児が応じた」を意味するとは限らないので、評価では `history_extra()` の内容も見ること。

**`retry_count` は `change_state` でリセットされない。**
意図的だが、状態を往復するタイプの FSM では想定より早く `"FAILED"` に到達しうる。

**`negative_emotion_surge` 経路の拒否は発火しない。**
シグナルが常に 0.0 のため（[../infant_signal.md](../infant_signal.md)）。情動を実装するまで、拒否は逃避と視線・首振りだけで判定される。

**対象を持っているかの判定は 2 経路。**
`_infant_attending_current_object` は、先に `infant["held_obj"]` が今の対象と一致するかを見る。加えて、シグナルの `holding > 0` かつ手の対象が今の対象であるときも真になる。`holding` は `held_obj` があるとき 1.0（[../infant_signal.md §9](../infant_signal.md#9-注意事項)）。

**`J1_GAZE_REJECTION_HOLDING_FALLBACK` は `False` 固定。**
親が対象を持っている間の視線拒否をスキップするフォールバック。2 段判定を入れて不要になったので無効化してあるが、誤拒否が再発したときのために残してある。

**`level` の使い分けが未整理。**
`urgency` は全プロトコルで 0 のまま。緊急度による分岐は実装されていない。
