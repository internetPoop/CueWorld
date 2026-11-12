# 共同注意 (`protocols/joint_attention.py`)

親が対象物を指定し、幼児がその物を見るまでを扱う三項関係のプロトコル。ラベリング（物の名前を教える）と感情共有を同じ FSM で扱う。

共通契約は [README.md](README.md)。

| 属性 | 値 |
|------|-----|
| `level` | 2（カタログ中で最も高い） |
| `has_target` | `True` |
| `response_type` / `request_type` | どちらも `True` |
| `typical_steps` | 130 |

---

## 1. 2 つの軸

挙動は独立した 2 つのパラメータで決まる。

### `cue_type` — どうやって注意を向けさせるか

| 値 | 親の動作 |
|----|---------|
| `"looking"` | 対象物を見るだけ |
| `"pointing"` | 対象物を指差しながら見る |
| `"shake"` | 対象物を取りに行って掴み、幼児の前で掲げて見せる。手首は振らない |

### `intent_type` — 何のために向けさせるか

| 値 | 発話 |
|----|------|
| `"informative"` | `declare_object`（「これ ＋ 対象語」）。名前を教える |
| `"sharing"` | `praise`（「すごい」）。感情を共有する |
| `"simple"` | 発話しない（`parent_utterance = False`） |

どちらも `start` の `context` で上書きできる（デモや固定プロトコル用）。

---

## 2. 状態遷移

```
init
 ├─ 幼児始動 ────────────────────→ approach
 ├─ cue_type == "shake" ─────────→ fetching → approach
 └─ その他 ──────────────────────→ approach

approach ──(shake)──→ shaking
         └(その他)──→ looking / pointing

looking / pointing / shaking ──(注視 3 秒)──→ post_shared_attention ──(言葉の往復 / 待ち)──→ completed
                             └─(5 秒待ち/離脱 ×2)──→ failure
```

`state_proximities`: `init` 0.0 / `fetching` 0.15 / `approach` 0.2 / `looking`・`pointing`・`shaking` 0.4 / `post_shared_attention` 0.8 / `completed` 1.0 / `failure` 0.0。

### `fetching`（shake のみ）

`grasp_object` で対象を掴む。掴めたら `first_hand_status` の親エントリを消して `approach` へ（[../parent_tools.md §4](../parent_tools.md#4-腕と手)）。

`fetch_timeout_steps = 116`（≒11.6 秒）を超えたら `fail_on_error_timeout()` で即失敗。**これは幼児の応答待ちではなく親のモータが進まない場合の保険**なので、`retry_count` の梯子には乗らない。

### `approach`

| `cue_type` | 使うツール | 進む条件 |
|-----------|-----------|---------|
| `"shake"` | `approach_infant(approach_dist = shake_stand_dist)` | 距離が `shake_stand_dist + 0.1` 未満 |
| その他 | `approach_infant_for_ja(approach_dist = 4.0)` | 距離が 4.2 未満 |

`approach_infant_for_ja` は**親が幼児と対象物の間に入らないように迂回する**（[../parent_tools.md §3](../parent_tools.md#3-移動)）。`approach_dist = 4.0` は実質「近づかない」で、位置調整のためだけに呼んでいる。指差しは離れていても見えるので、近づく必要がない。

`approach_timeout_steps = 52`（≒5.2 秒）を超えたら即失敗。

### `looking` / `pointing`

毎 step 次の 3 つを行う。

1. `_ensure_cue_started()` — 初回だけ発話をキューし、注視トラッカーをリセット
2. `_maybe_react_to_words()` — 幼児の単語に反応（[§4](#4-単語への反応)）。ホールド成立前は seq だけ消費し返事しない
3. `_update_cue_attention()` — 注視トラッカーを 1 step 進める（[§3](#3-成功判定)）

### `shaking`

対象を落としてしまったら `fetching` に戻る（**注視トラッカーはリセットしない**）。持っている間は、

- `show_pos`（掲げ点）への到達を判定し、到達後だけ `shake_timer` を進める
- 単語反応と注視トラッカーは `looking` と同じ

掲げ点は `compute_show_position` の既定（目からの水平距離）を **`shaking` に入るときに 1 回だけ**計算して固定する。抜けるときに破棄する。渡しの掲げも同じ既定を使う。

XY は幼児の**目**から親の方向へ水平に進んだ点で、Z は計算した瞬間の目の高さ。距離は `show_distance_from_eyes`（`GAZE_RAY_ORIGIN_OFFSET` 0.1 m に対象の半幅を 2 つ足す）。視線レイは目の 0.1 m 前から出るので、胴体基準の `place_dist` だと点が目の中に入り、レイが物体を素通りして何も見ていないことになる。半幅の 1 つは手前の面をその起点の外へ出す分、もう 1 つは手が目標より顔側へ寄る分。`env.show_dist` を渡したときは、その値を目からの水平距離として使う。

| 定数 | 値 | 意味 |
|------|-----|------|
| `shake_stand_dist` | `env.stand_dist` | 見せるために立つ距離 |
| `show_dist` | 目から `0.1 + 2 × 半幅`（`env.show_dist` で上書き可） | 掲げ点。目のすぐ前で、親の顔より手前 |
| `show_reach_tol` | 0.12 m | 掲げ到達の判定 |
| `shake_steps` | 60 | 未使用（[§7](#7-注意事項)） |

### `post_shared_attention`

**最初の注視が成立したあとのセッション**。合図は続けたまま顔だけ幼児へ向ける。`session_success` はこの進入時に立つ。詳細は [../joint_attention_session.md](../joint_attention_session.md)。

毎 step:

1. 返事のキューが空なら、言い終えた往復を 1 つ数える（最大 5 で `completed`）
2. 注視トラッカーを進める。成立後の離脱 50 step は合図を強めず `completed`（成功のまま）
3. ホールドが続いているあいだだけ単語に返事する
4. 次の言葉を待つ（親のキューが空で、プロトコルの声区間の外だけ `voice_wait_timer` を進める。50 step で `completed`）。声が出続けたまま返事が無い 50 step も見切って `completed`

視線が切れたら、次の返事の前に再び 30 step 見る。合図は強めない。

---

## 3. 成功判定

成功フラグは**視線だけ**で決める。`AttentionHoldTracker`（[README.md §9](README.md#9-共通ユーティリティ)）に毎 step、幼児の視線対象を渡す。

- 対象物を **30 step（3 秒）途切れずに**見たら `session_success` を立て、`post_shared_attention` へ
- 親の顔を見ている間はカウントを据え置く（対象と親の顔を交互に見るのは共同注意として自然）
- それ以外を見たらカウントを 0 に戻し、離脱タイマーを進める
- **成立前**に一度も見ないまま、または離脱したまま 50 step（5 秒）経ったらタイムアウト（下の強化）
- **成立後**の同じ離脱は成功のままセッションを閉じる

`get_infant_target` は `gaze_targets_now`（その step のベクトル）の最大要素を返す。キーが無い、または全要素 0 なら `None`。**親を除外しない**（`ignore_parent=False`）ので、親の顔を見ていれば親の body id が返り、トラッカーが据え置きに分岐する。

### タイムアウトの扱い（`_handle_cue_timeout`）— 成立前のみ

`retry_count` が 0 なら 1 回だけ強化して、2 回目で失敗。**対象は変えない。**

| `cue_type` | 強化の内容 |
|-----------|-----------|
| `"looking"` | `cue_type` を `"pointing"` に変えて `pointing` 状態へ。発話を再キュー |
| `"pointing"` / `"shake"` | 同じ合図のまま発話をもう一度。見せる合図は取りに戻らない |

見るだけ → 指差す、という強化の梯子がここにある。

---

## 4. 単語への反応

`_maybe_react_to_words` は、認識器が出した `voice_reply` を読む。プロトコル実行中は対象の概念 ID を認識器へ渡す。返事はホールドが成立しているあいだだけ積む。

| `voice_reply` | 列に残す種類 | 発話 |
|------|-------------|------|
| `"sugoi"` | `"correct"` | キューを空にして `praise_with_target`（「すごい ＋ 対象語」）。その場 |
| `"oshii"` | `"near"` | `recast_with_target`（「そうだね ＋ 対象語」）。黙ったとき |
| `"chigauyo"` | `"wrong"` または `"unadopted"` | `correct_with_target`（「ちがうよ ＋ 対象語」）。黙ったとき |
| 無し | 変えない | 発話しない |

- 1 まとまり（認識器の発話区間）につき親は 1 回。途中で「すごい」が出たらそれだけ。「すごい」が無ければ閉じたときに「おしい」か「ちがうよ」。まとまりが閉じたら認識器とプロトコルの「もう言った」を戻す。
- 「すごい」は残っている合図や「ちがうよ」を消してから入れる。
- 往復が 1 つ終わるのは、返事を最後まで言い終えたとき。消された途中の返事は `word_match` 列にも往復数にも入れない。
- `infant_spoke` は返事を積むときに立つ。
- **成功フラグは視線だけ。** 単語が合っていても間違っていても、最初のホールドで立つ。
- `looking` / `pointing` / `shaking` / `post_shared_attention` のすべてで呼ばれるが、成立前は seq の消費だけ。

---

## 5. ターゲット選択

`select_target` をオーバーライドしている。

**幼児始動で、かつ手の先が使えるとき**は手の先を対象にする。

- 「使える」＝ `pointing_is_complete` または `weak_arm`（[../intent.md](../intent.md)）が真で、`gesture_targets` の最大（親を除く）が除外リストに無いこと
- 弱い腕まで許すのは、指差しが未発達な幼児でも腕の向きで対象を示せるようにするため

それ以外は基底の `select_target`（視線 → ランダム）に委譲する。

`_pointing_is_reliable` も定義されているが、`_hand_target_is_usable` のほうが実際に使われる。

---

## 6. PM との関係

### `infant_is_engaged`

`gaze_targets`（30 step 平均）の最大が現在の対象と一致していれば真。**成功条件の 3 秒ホールドとは独立**で、平均の一致でよい。親の顔が平均で一番なら偽。persistence の判定に使う。

### `keeps_held_object`

`cue_type == "shake"` かつ持っている物が対象そのものなら `True`。開始時に手放さない。他のケースでは手放す。

### `blocks_interrupt`

`post_shared_attention` のときだけ、次のいずれかで割り込みを止める。

- 親の発話キューが残っている
- 幼児の声が出ている（プロトコルの声区間）

### 維持

言い終えた言葉の往復ごとに、2.7 / 5.0 から 1.0 引く。下限は 1.0。

### `history_extra`

`intent_type` / `infant_spoke` / `word_match`（言い終えた種類の列） / `emotion_reacted`（常に `None`）を履歴に残す。

---

## 7. 注意事項

**`shake_steps` と `shake_reach_timeout_steps` は使われていない。**
`shake_timer` は到達後に進むだけで、上限判定に使われていない。掲げて見せる動作は注視が成立するかタイムアウトするまで続く。`shake_reach_timeout_steps = 24` も参照されていない（掲げに届かなくても見せる位置への移動は続ける設計になったため）。

**`approach_dist = 4.0` は「近づかない」の意味。**
非 shake の接近はアリーナの対角より長い距離を指定しているので、前進はほぼ発生せず、`approach_infant_for_ja` の迂回ロジックだけが働く。距離判定の閾値 4.2 も同じ理由。

**`reward_function` の返り値は環境に届かない。**
`BaseEnv` の `reward` は常に 0 で、プロトコルの報酬は接続されていない（[../env.md §13](../env.md#13-注意事項)）。値を使うには PM の返り値から拾う必要がある。

**`eval_mode` の `delta_sum` はインデックスの取り方が `trackable_targets` と違う。**
`get_eval` は `all_objects.keys()` の順で対象のインデックスを引くが、`gaze_targets` の並びは `trackable_targets`（全物体 ＋ 親）。物体だけなら先頭部分が一致するので実害は出にくいが、並びの前提が 2 か所で違う。

**`sharing` 意図の表情表現は未実装。**
`intent_type == "sharing"` でも `praise` を言うだけで、笑顔などの動作は入らない。

**`_pointing_is_reliable` は呼ばれていない。**
`_hand_target_is_usable` に置き換わった。

**`fetching` に戻ったときトラッカーをリセットしない。**
落として拾い直すあいだも注視の蓄積は保たれる。これは意図的で、親が手元でもたついている間に幼児の注視が切れた扱いになるのを避けている。
