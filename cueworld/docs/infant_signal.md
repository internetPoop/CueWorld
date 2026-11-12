# 幼児シグナル (`infant_signal.py`)

`InfantSignalDetector` は `BaseEnv` の上に載る `gym.Wrapper` で、幼児の身体状態を**親が観測できる形の数値**へ変換する。
本ドキュメントは**シグナルキーの契約（定義・値域・ゼロ条件）の正本**である。

親はここで作られた辞書しか見ない。物体 ID や関節角を直接読むことはしない。
「どのシグナルの組み合わせを訴えとみなすか」は [intent.md](intent.md)、「訴えをどう処理するか」は [protocol_manager.md](protocol_manager.md)。

---

## 1. 位置づけ

```python
env = InfantSignalDetector(ParentTools(BaseEnv(...)))
obs, r, d, t, info = env.step(action)
signals = info["infant_signals"]
```

- `step()` は下位 `env.step()` を呼んだ**あと**に `update()` で履歴を伸ばし、`get_normalized_signals()` の結果を `info["infant_signals"]` に入れる。したがってシグナルは**その step の行動を反映した値**。
- `reset()` はすべての履歴バッファと認識器の状態をクリアする。
- `__getattr__` で `BaseEnv` の属性・メソッドをそのまま引ける（`self.trackable_targets` は `env.trackable_targets`）。

### モダリティと `active_signals`

シグナルは gesture / gaze / voice / expression の 4 モダリティに分かれる。
`active_signals`（キー名のリスト）を渡すと、そのキーを含むモダリティだけを計算・出力する。`None`（既定）なら全部。

履歴の更新自体もモダリティ単位でスキップされるので、途中で `active_signals` を変えても履歴は埋まらない。
拒否シグナル（[§7](#7-拒否シグナル)）だけは常に計算される。

各表の **使用** 列は、`intent.py` / プロトコル / `ProtocolManager` が実行時にそのキーを読むかどうか。テスト・デモ・`ProtocolManager.signal_keys`（ログ・互換用の並び）は含めない。

### 履歴長

| バッファ | 長さ | 実時間 |
|----------|------|--------|
| 手・視線・表情・行動履歴 | `history_length = 30` | ≒ 3 秒 |
| 音素・音素スコア | `phoneme_history_length = 20` | — |
| 単語・単語スコア | `word_history_length = 5` | — |
| 音量・ピッチ | 30 | ≒ 3 秒 |

音素の更新は `voice_interval = 3` step ごと。窓の各フレームを判定し、コスト最小（同点は最新寄り）を 1 ラベルにする。単語は有声採取が 10 フレーム以上でゲートを満たした時点、または無声で区間が終わった時点で更新される（[language.md §6](language.md#6-発話区間の切り出し)）。10 フレーム未満の無音閉じは公開しない。

### 時間の契約

この節は **シグナルが有効とみなされるまでの時間の畳み込み** の正本。プロトコルの `persistence`（割り込まれにくさ）は対象外（[protocol/README.md §5](protocol/README.md#5-persistence)）。

- 十分な長さの履歴は検出器が持つ。式はすべて検出器に置く。窓長と「その畳み込みを使うか」は、読み手によって変わるものだけ読み手が決める（ヘルパに `n` を渡す）。
- シグナル自身の性質（`*_stability`、周期性、喃語比）は公開キーにしてよい。ヒット率・連続カウントなど「有効とみなす長さ」は、読み手のスケールが揃えられるなら揃えて公開キーにする。揃えられないものだけヘルパ。
- 窓付き公開キーに対し、読み手は**閾値かそのまま**だけ。再平均・step 間差分・そのキーの連続カウントは禁止。変化が欲しいときは瞬間側を見る。
- step 内のセンサ融合（円錐レイ、`infant_voice_window` の平均 → その step の volume）は「瞬間」であって、step 間の窓ではない。

読み分け（公開キーを増やす実装は、この契約をキー表へ落としてから）:

| キー | 公開するもの | 読み手 |
|------|--------------|--------|
| `gaze_targets` | 30 step 平均 | 対象選択、`infant_is_engaged`、PM 見込み対象 |
| `gaze_targets_now` | その step のベクトル | JA の `get_infant_target` / `AttentionHoldTracker` |
| `looking_at_parent_eye` | 30 step ヒット率 | `intent.parent_dyadic_cue`（`>= 0.6`） |
| （公開しない） | その step の目ヒット | 選好報酬の `I_look`（内部） |
| `gesture_targets` | 30 step 平均のみ | 瞬間キーは足さない |
| `hand_height_margin` | 10 step ヒット率 | 二項 `arm_raise` |
| 5a（未実装） | 「holding かつ目線帯」の 10 step ヒット率 | 持って掲げ続け。高さ窓は増やさない。このキーの上に連続も積まない |

---

## 2. 対象ベクトルの共通契約

「幼児が何に向いているか」は、すべて **`env.trackable_targets` と同じ長さ・同じ並びの float ベクトル**で表す。
並びは「全物体の body id ＋ 親の body id」で、エピソード中は変わらない（[env.md §11](env.md#11-gym-api-と-step-の処理順)）。

値は各対象の**確度**で、確率分布ではない（合計は 1 にならない）。どの対象にも当たらなければ全要素 0。

### 円錐レイキャスト

視線・指差し・リーチ・腕向きはすべて同じ `_get_targets_by_cone_raycast` を通る。

- 起点から前方 10 m へ、**中心 1 本 + 3° リング 4 本 + 8° リング 8 本の計 13 本**を一括レイキャスト。
- 確度 = `clip(1 - 角度 / 8.0, CONE_MIN_ACCURACY, 1)`（既定 `CONE_MIN_ACCURACY = 0.25`）。中心が 1.0、3° リングが 0.625、8° リングが 0.25。`InfantSignalDetector` の `cone_min_accuracy` / `cone_rings` / `cone_ray_length` で差し替え可能。
- 同じ対象に複数本当たったら最大値を採る。
- 床（body id 0）は無視する。

起点と方向はモダリティごとに違う。

| 対象ベクトル | 起点 | 方向 |
|--------------|------|------|
| 視線 | 目の位置 + 前方 0.1 m（`GAZE_RAY_ORIGIN_OFFSET`） | 目の前方 |
| リーチ（開き手） | palm 中心 | palm 前方 |
| 指差し | 人差し指の先端 | 指の方向 |
| 腕向き（その他の手形） | 前腕先端 + 前方 0.07 m | 前腕の向き |

起点を少し前へずらすのは、自分の身体に当たるのを避けるため。

### ヒットの分類（`_classify_gaze_hit`）

当たったのが物体でない場合の振り分け。

| 当たった先 | 帰属先 |
|-----------|--------|
| 物体 | その物体 |
| 親の顔（`face_links`: 目・頭・唇） | 親の body |
| 親の腕〜手（`hold_gaze_links`）で、親が何か持っている | **親の把持物** |
| 親のその他のリンク（胴・肩） | 無視 |
| 幼児自身の腕〜手で、幼児が何か持っている | 幼児の把持物 |
| その他 | 無視 |

**把持中の物へ視線クレジットを振り替える**のがこの関数の要点。親が物を持って見せているとき、幼児が親の手元を見ていれば「その物を見ている」と数える。逆に胴体に当たっただけでは「親を見ている」にはならない。

---

## 3. 安定性シグナルの共通式

`hand_target_stability` / `gaze_target_stability` / `voice_target_stability` はいずれも**正規化エントロピーの補数**で、「1 つに絞れているか」を [0, 1] で表す。

```
p = 分布 / 分布の総和          （0 の要素は除く）
H = -Σ p log p
stability = clip(1 - H / log(要素数), 0, 1)
```

- 全要素 0（＝何にも向いていない）なら 0.0。
- 要素が 1 つしかないなら 1.0。**追跡対象が 1 つだけの構成では常に 1.0 になる**ので注意。
- 手・視線は**履歴 30 step の平均分布**に対して計算する。瞬間値ではない。
- `ignored_targets` を渡すと、該当対象の質量を 0 にしてから計算する。「もう相手にしない物」を安定性の計算から外すために使う。

---

## 4. ジェスチャシグナル

手の形は `BaseEnv._get_hand_shape` の 4 分類（`open` / `closed` / `pointing` / `other`。[env.md §7](env.md#7-接触把持受け渡し)）。

| キー | 定義 | ゼロ条件 | 使用 |
|------|------|----------|------|
| `point_intensity` | 手の形が `pointing` のときだけ非ゼロ。値はその形の強度 [0, 1]（理想 1、形の端 0） | 形が `pointing` でない | `intent.pointing_is_complete`（`> 0`） |
| `open_intensity` | 手の形が `open` のときだけ非ゼロ。値はその形の強度 [0, 1] | 形が `open` でない | `intent.reach_is_complete`（`> 0`） |
| `hand_target_stability` | 手の対象分布の安定性（[§3](#3-安定性シグナルの共通式)） | 履歴なし／全対象 0 | `intent.hand_target_is_clear` |
| `holding` | `held_obj` があるとき 1.0 | 持っていない | `intent`（reach / weak_arm の除外、`parent_object_cue`）。PM のくれ到達圏ゲート。[protocol/README.md §5](protocol/README.md#5-persistence) |
| `touching` | `graspable_obj` があるとき 1.0（手形は見ない） | 把持範囲に物がない | `intent.parent_object_cue`。[protocol/README.md §5](protocol/README.md#5-persistence) |
| `hand_open_close_periodicity` | 手の開閉リズム（下記） | 往復が検出できない | `DyadicProtocol.compute_channel_salience`（`hand_open_close`） |
| `hand_height_margin` | 直近 10 step のうち手が目線帯にあった割合 | 履歴 10 未満 | `DyadicProtocol.compute_channel_salience`（`arm_raise`） |
| `gesture_targets` | 手の対象ベクトル（履歴平均、`ignored_targets` 適用済み） | — | `intent.hand_target_is_clear`。JA の対象選択。PM `_resolve_hand_target_id`。[protocol/README.md §5](protocol/README.md#5-persistence) |

`point_intensity` / `open_intensity` の値は `_get_hand_shape` の第3戻り（強度）をその形のときだけ載せる。`intent.py` はこれらを **`> 0` の二値ゲート**として使う（強度の大きさで閾値判定はしていない）。
- 物を持って掲げ続けているというシグナルもほしいかも

手の形 `closed` はシグナルキーにはしない。物を掴む・離す判定と、開閉リズムの「閉じ」側のラベル、開いても指差してもいないときの前腕の向きに使っている。

### 手の対象ベクトルの決まり方

手の形によって、どの経路で対象を求めるかが変わる。

1. 把持可能距離に物がある → その物 1 点に 1.0（`touching`）。手形は見ない
2. 手の形が `open` → palm 前方のレイキャスト
3. 手の形が `pointing` → 指先のレイキャスト
4. それ以外（`closed` / `other`） → 前腕の向きのレイキャスト

4 番目があるのは、指差しが未発達な幼児でも「腕がどっちを向いているか」を親が読めるようにするため。

### 周期性

`_compute_angle_periodicity(履歴, 方向, vmax=0.15)` が共通の骨格。

```
periodicity = clip(角速度の符号反転回数 / 4, 0, 1)     # 履歴 5 未満なら 0
（periodicity == 0 なら即 0）
strength    = clip(|最新角速度| / 0.15, 0, 1)
結果        = periodicity × strength
```

**往復していること（周期性）と、いま動いていること（強度）の積**。ゆっくり大きく動かすだけでも、小刻みに止まっているだけでも上がらない。

`hand_open_close_periodicity` は 2 つの推定の大きいほうを採る。

- 人差し指の関節角履歴に上式を適用したもの
- 直近 5 フレームの手形ラベルが `{open, pointing}` と `{closed, other}` の間で交替した回数 / 3

関節が細かく動かない場合でもラベル交替で拾えるようにしてある。

### 手の高さ

`hand_height_margin` は「手を目線の高さまで挙げているか」を割合で返す。

- 帯の下端は `目線高さ - 0.4 m`（`HAND_HEIGHT_TOL`）。上限は無い。
- 目線高さ（`eyes_link` の world Z）はエピソード開始時に一度だけ確定させ、以後固定する。首を動かしても基準は動かない。
- 窓は直近 10 step（`HAND_HEIGHT_WINDOW`）。それ未満なら 0.0。

`DyadicProtocol` の `arm_raise` チャネルはこの値をそのまま salience として使う。

---

## 5. 視線シグナル

| キー | 定義 | ゼロ条件 | 使用 |
|------|------|----------|------|
| `gaze_target_stability` | 視線対象分布の安定性（[§3](#3-安定性シグナルの共通式)） | 履歴なし／全対象 0 | `intent.parent_object_cue`（`>= 0.6`） |
| `looking_at_parent_eye` | 直近 30 step のうち親の `eyes_link` にレイが当たった割合 | 一度も当たっていない | `intent.parent_dyadic_cue`（`>= 0.6`） |
| `gaze_targets` | 視線対象ベクトル（履歴平均、`ignored_targets` 適用済み） | — | `BaseProtocol.select_target`。JA `infant_is_engaged`。PM `_resolve_gaze_target_id`。[protocol/README.md §5](protocol/README.md#5-persistence) |
| `gaze_targets_now` | その step の視線対象ベクトル（履歴の最後の要素。`ignored_targets` はかけない） | 履歴なし → キーを出さない。全要素 0 は「何も見ていない」 | JA `get_infant_target` / `AttentionHoldTracker` |

`looking_at_parent_eye` は**割合**であって瞬間値ではない。「いま目を見ている」ではなく「直近 3 秒どれくらい目を見ていたか」。`intent.parent_dyadic_cue` はこれが 0.6 以上（＝3 秒中 1.8 秒）を「目を見ている」とみなす。選好の `I_look` はこのキーを使わない（[§10](#10-選好報酬)）。窓キーの step 間差分は取らない。

---

## 6. 音声シグナル

| キー | 定義 | ゼロ条件 | 使用 |
|------|------|----------|------|
| `voice_volume` | 発声窓の平均 volume を `[0, 1]` にクリップ | 無声 | `intent.word_candidate`。二項 voice salience。`VoiceSegmentTracker` |
| `voice_target_stability` | 最新の単語スコア分布の安定性 | 単語スコア履歴なし | `intent.word_candidate`。二項 voice salience（`1 -` で喃語らしさ） |
| `voice_best_distance` | 採用された単語のスコア（小さいほど良い） | 発話がまだ無い → 0.0。不採用 → 無限大 | `intent.word_candidate` |
| `babbling_target` | 音素履歴の最頻出音素（`None` / `"unknown"` を除く） | いま無声／有効音素なし。`None` を返す | 二項の模倣音素・反応検知。PM が dyadic context に渡す |
| `babbling_ratio` | その音素の出現数 / 音素履歴長 | 同上 | 二項 voice salience |
| `words` | 採用された発話の `概念 ID → 下駄後の生コスト`（小さいほど良い） | 未発話・不採用 → `{}` | `intent.word_candidate`。JA `_maybe_react_to_words` |
| `voice_unadopted_attempt` | 不採用で、畳んだ音が 3 以上 | 未発話・採用・記録 2 以下 → 偽 | 長さの記録。JA の返事は `voice_reply` |
| `voice_segment_seq` | 単語の公開、または黙って 3 音以上を判定するたびに増える番号 | まだ公開していない → 0 | JA が区間の切れ目を見る |
| `voice_reply` | いま言う返事。`sugoi` / `oshii` / `chigauyo` / `None` | 返事が無い → `None` | JA `_maybe_react_to_words` |
| `negative_emotion_surge` | 常に 0.0（[§9](#9-注意事項)） | 常に | `BaseProtocol.detect_rejection` が読むが常に 0.0 なので発火しない |

`voice_target_stability` は単語スコアを `exp(-score)` で確率化してからエントロピーを取る。他の安定性と同じく「1 語に絞れているか」の指標。

声の大きさと高さは、それぞれ直近 30 step の履歴として残す。続いた割合のキーは出さない。声の感情は、この履歴をあとで読む。

`babbling_target` は**現在無声なら履歴が残っていても `None`**。「いま喃語を出している」という現在形の意味を持たせるため。`DyadicProtocol` の voice チャネルはこれを模倣対象の音素として使う。

音素・単語の推定そのもの（プロトタイプ距離、DP マッチング、採択条件）は [language.md](language.md)。
- babblingの定義は今のままでいいのか(異なる母音の並びが繰り返されるけど単語には判定されないとかのほうがいいのでは？)
- 音声の感情の定義．声の大きさやピッチで判定したいので，それらを使って求める．

---

## 7. 拒否シグナル

`active_signals` の設定にかかわらず毎 step 計算され、シグナル辞書に必ず入る。

| キー | 定義 | ゼロ条件 | 使用 |
|------|------|----------|------|
| `head_turn_away_intensity` | 幼児の回転行動が直近で同符号に偏っている強度 | 履歴 5 未満／有効な行動 5 未満／偏りが 80% 未満 | `BaseProtocol.detect_rejection` |
| `sudden_gaze_shift` | 視線分布の前半平均と後半平均の L1 距離 | 視線履歴 15 未満 | **未使用**（拒否は `gaze_shift_dict` を読む） |
| `gaze_shift_dict` | 対象ごとの `後半平均 - 前半平均`。正なら急に見た、負なら急に逸らした | 視線履歴 15 未満 → `{}` | `BaseProtocol.detect_rejection` |
| `gaze_late_mass` | 対象ごとの後半平均視線質量 | 同上 | `BaseProtocol.detect_rejection` |
| `negative_emotion_surge` | 常に 0.0 | 常に | `detect_rejection` が読むが常に 0.0 なので発火しない |

`head_turn_away_intensity` は**幼児の行動ベクトル**（`torso_joint + head_pan_joint` の合算、および `head_tilt_joint`）の履歴から計算する。関節角ではなく行動を見るのは、可動域の端に張り付いても「回そうとし続けている」ことを検出するため。判定は「絶対値 0.3 を超える行動が 5 個以上あり、その 80% 以上が同符号」で、値は有効行動の平均絶対値。

`gaze_shift_dict` と `gaze_late_mass` を対で使うのは、視線を逸らしたのか単に一度も見ていなかったのかを区別するため。プロトコルの拒否判定（[protocol/README.md](protocol/README.md)）が両方を読む。

シグナル自体は毎 step 計算する。`ProtocolManager` の `rejection` は既定 `False` で、共同注意・二項・受け渡しは `rejection=True` のときだけ毎 step `detect_rejection` を呼ぶ。視線の 2 段は、そのとき生きている。確認に使うのは `head_turn_away_intensity` だけである。情動の `negative_emotion_surge` は常に 0.0 なので、オンにしてもその経路は発火しない。見守りと家事は呼ばない。既定をオフにしているのは、オンにすると首振りの過渡で過剰に失敗終了するためで、再設計はまだである。


---

## 8. 身体シグナル

身体モダリティは出さない。親との距離、近さ、向き、接近、`body_targets` は計算しない。

---

## 9. 注意事項

**`holding` / `touching` は `agent` 辞書を正しく読む。**
`get_hand_status` は `agent.get("held_obj")` と `agent.get("graspable_obj")` を読む。`holding` は `held_obj` の有無、`touching` は `graspable_obj` の有無（手形は見ない）。把持中はだいたい両方 1.0。

**円錐確度の下限クリップ。**
最外リング（既定 8°）の確度は `CONE_MIN_ACCURACY`（既定 0.25）未満にならない。`_classify_gaze_hit` は確度 0 以下を捨てるが、8° ヒットは 0.25 で残る。親の `face_links` ヒットも視線ベクトルの親スロットに載る。

**手形 intensity は `_get_hand_shape` の第3戻り。**
理想姿勢を 1、その形と判定される一番端を 0 とする [0, 1]。reset 直後の開き手（関節 0）は `open_intensity == 1` になり得る。対象がはっきりしていれば `intent.reach_is_complete` の `> 0` ゲートを通る。

**`touching` は指先の接触ではない。**
`BaseEnv` の `agent["graspable_obj"]`（palm 基準の把持可能判定）を読む。意味は「手が掴める位置に物がある」。手形は見ない。

**表情シグナルは出力されない。**
`get_face_expression` はダミー（常に `(0, 0.0)`）で、`get_expression_signals` は定義されていない。`expressions` キーも出ない。メソッドはあるがどこからも呼ばれない。

| キー | 状態 |
|------|------|
| `smile_intensity` | メソッドあり・未呼び出し。キーも出ない |
| `distress_intensity` | 同上 |
| `surprise_intensity` | 同上 |
| `expressions` | `get_expression_signals` 未定義のため出ない |

**不採用の発話は単語として出さない。**
認識器が採択しなかったとき、`words` は空、`voice_target_stability` は 0、`voice_best_distance` は無限大になる。距離 0 にはしない。畳んだ音が 3 以上なら `voice_unadopted_attempt` が真になる。1 音か 2 音で黙ったときは公開しないので、単語も不採用も出さない。発話がまだ無いあいだは安定度 0、距離 0、`words` 空。単語を公開したときと、黙って 3 音以上を判定したときに `voice_segment_seq` が増える。対象があるときの返事は `voice_reply` に入る。

**シグナルは env より 1 段遅れない。**
`update()` は `env.step()` の直後に走るので、その step の行動が反映された値が同じ `info` で返る。ただし履歴平均を取るシグナル（安定性、割合、周期性）は当然過去 30 step を含む。

---

## 10. 選好報酬

`InfantSignalDetector.step()` が下位 `env.step()` のあと `reward[0]` に足す。`info["infant_signals"]` には載せない。

**式**（視線モダリティが有効で、親がいるときのみ）:

```
r_pref = I_hold × obj.preference + I_look × env.parent_preference
```

- `I_hold`: 幼児が何かを把持中なら `1`、そうでなければ `0`。振るは見ない。`obj` は `held_obj` のクラス `preference`。
- `I_look`: **その step** のレイキャストで親の `eyes_link` に当たったら `1`、そうでなければ `0`。`looking_at_parent_eye`（30 step 平均）とは別。

**ゼロ条件**

- 視線モダリティが `active_signals` で無効 → **把持分も含めて 0**。
- `num_agents == 1` または親が無い → 0。

プロトコルの `reward_function` は学習信号に使わない。`info["protocol_reward"]` にだけ載る（[protocol_manager.md](protocol_manager.md)）。
