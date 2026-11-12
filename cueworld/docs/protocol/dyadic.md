# 二項的やりとり (`protocols/dyadic.py`)

物を介さず、親と幼児が**同じリズムを交わす**やりとり。幼児の喃語や身振りを親が真似る（幼児始動）か、親が身振りを見せて幼児が真似るのを待つ（親始動）。

`DyadicProtocol` は 1 クラスで両方向を扱う。共通契約は [README.md](README.md)。

| 属性 | 値 |
|------|-----|
| `level` | 0（最も早い発達段階） |
| `has_target` | `False`（物を使わない） |
| `response_type` / `request_type` | どちらも `True` |
| `typical_steps` | 105 |

---

## 1. チャネル

やりとりの単位を**チャネル**と呼ぶ。3 本ある。

| チャネル | 幼児側の salience | 親側の動作 | 持続 |
|----------|------------------|-----------|------|
| `voice` | 保持中の喃語列があれば 1、無ければ 0 | 列を 1 音 6 step で 1 回返す | 発話キューが空になるまで |
| `arm_raise` | 手を目線帯まで挙げているか | 掲げ + 腕を振る | 閾値のあいだ持続（上限 90 step） |
| `hand_open_close` | 手の開閉リズム | 掲げ + 指の開閉 | 閾値のあいだ持続（掲げと同じレーン） |

モータは掲げと開閉を**1 つのレーン**として扱う。どちらかが閾値以上ならレーンは開いたまま。開閉だけの指定は `_normalize_channels` が掲げを前に足す。

モータチャネルの 1 周期は `MOTOR_CYCLE_STEPS = 30` step（≒3 秒）。見本再生の duration は `arm_raise` 90 step、`hand_open_close` 60 step。持続レーンに移ったあとは duration に縛られない。

### salience の計算（`compute_channel_salience`）

```
voice           = babbling_salience（保持中の喃語列があれば 1）
arm_raise       = hand_height_margin
hand_open_close = hand_open_close_periodicity   （ただし arm_raise >= 0.8 のときだけ。それ以外は 0）
```

- `hand_open_close` に `arm_raise` の前提を付けているのは、手を下ろしたままの開閉を親が見られないため。**見えていないものは模倣しない**。

### 閾値 τ

チャネルごとに独立した閾値を持つ（`MATCH_QUANTITY_THRESHOLDS`）。クロスチャネルの順位付けはしない。

| チャネル | τ |
|----------|---|
| `voice` | 0.2 |
| `arm_raise` | 0.8（`ARM_RAISE_HIT_RATE_MIN`。直近 10 step 中 8 step 挙げている） |
| `hand_open_close` | 0.2 |

`get_salient_channels` は τ を超えたチャネルを**固定順**（voice → arm_raise → hand_open_close）で全部返す。強いチャネルを 1 本選ぶ、ということはしない。

### 正規化（`_normalize_channels`）

`hand_open_close` が入っているのに `arm_raise` が無ければ、`arm_raise` を**前に**足す。手を挙げてから開閉するという順序を保証する。「開閉だけ」は存在しない。

---

## 2. 声レーンとモータレーン

セッションは**声**と**モータ**の 2 レーンを並走できる。閉じたレーンの親の動作だけが止まり、セッションが終わるのは**両方閉じたとき**（または顔離脱で全体を閉じたとき）。

| レーン | 開くとき | 閉じるとき |
|--------|----------|------------|
| 声 (`voice_active`) | 幼児の喃語列のオウム返しを始めたとき | 最大 5 往復、または次の喃語待ちが約 5 秒切れたとき |
| モータ (`motor_active`) | 幼児の掲げ／開閉が閾値以上になったとき | 真似開始から 90 step、または閾値を外れてから約 5 秒戻らないとき |

成功フラグ (`session_success`) は、最初のオウム返しを言い終えたとき、またはモータレーンを開いたときに立つ。見本を出しただけでは立たない。

---

## 3. 状態遷移

`infant_initiative`（`start` の `context` で渡る）で入口が分かれる。違いは**見本を先に出すかどうか**だけ。モダリティの時計は始動で揃える。片方の途中でもう片方が出たら足す。

```
init
 ├─ 幼児始動 ─→ waiting_infant
 │                  ├─(喃語)──→ echoing ──(完了)──→ waiting_babble ⇄ echoing（最大 5）
 │                  └─(モータ)─→ motor レーン開 + waiting_babble（喃語待ち）
 └─ 親始動  ─→ performing_parent_cue
                    ├─(見本中のモータ反応)─→ motor レーン開（見本の腕はそのまま）
                    ├─(声の見本終了後の喃語)─→ echoing
                    └─(見本完了)──→ waiting_babble
```

`state_proximities` は `init` 0.0 / `waiting_infant` 0.3 / `performing_parent_cue` 0.5 / `echoing`・`waiting_babble`・`imitating` 0.7 / `completed` 1.0。

### 幼児始動

1. `start` の `context` に `babbling_chunk` があれば、待ちを飛ばしてオウム返しへ
2. 無ければ `waiting_infant` で毎 step、喃語列とモータ閾値の**両方**を見る
3. 喃語があればオウム返し（声レーン）。モータが閾値以上なら真似開始（モータレーン）。両方同時でも、片方の途中でもう片方が出ても足す
4. どちらも約 5 秒無ければ「ねえ」を 1 回言ってリトライ。2 回目で失敗

### 親始動

1. `_parent_cue_channels()` でチャネルを決める（[§4](#4-親のキュー選択)）
2. `performing_parent_cue` で見本を再生する
3. **声の見本が残っているあいだ**は、幼児の喃語を反応にしない。言い切ったあと（または声無し見本）なら、喃語でオウム返しを始める
4. **モータの閾値**は、声の見本の有無にかかわらず即座に反応。見本のモータはそのまま続ける。90 step の壁時計はその反応時点から
5. 見本が全部終わったあとは `waiting_babble`。反応が無ければ約 5 秒で失敗。見本のやり直しも「ねえ」も無い

### 反応の検出（`_detect_infant_response`）

次のどちらかで反応とみなす。

- `babbling_chunk` がある
- どれか 1 チャネルの salience が自分の τ 以上

**親が見せたチャネルと一致している必要はない。** 一致は評価用の注釈（[§7](#7-応答マッチ注釈)）としてだけ記録する。「すごい」は今回出さない。

相手の目を見ることは、反応にも成功にも入れない。親が二項を始めてよいかの手がかり（`parent_dyadic_cue` / `looking_at_parent_eye`）のまま。

---

## 4. 親のキュー選択

`_sample_parent_channels()` は**常に 2 スロット**を返す。

- 50%: `voice` + モータ 1 本（ランダム）→ 正規化後、25% が声と掲げ、25% が声と掲げと開閉
- 50%: モータ 2 本（`arm_raise` と `hand_open_close`）

`hand_open_close` を引いたら `_normalize_channels` が `arm_raise` を足す。**`arm_raise` 単独をランダムに引くことはない。** 掲げ単独は `context["parent_channels"]` で明示したときだけ。

声の見本は幼児の音ではなく、固定の `p0 p1 p0 p1` を 1 音 6 step で 1 回。

---

## 5. モータの持続真似

閾値以上のあいだ、親は真似を続ける。掲げは腕を振り、開閉は指を動かす。**見本も真似も同じ腕振り**（幼児の動き方そのものに合わせる真似は将来）。

| 条件 | 親の動作 | 時計 |
|------|----------|------|
| 閾値以上 | 掲げ＋腕振り／開閉を続行 | `motor_wall_timer += 1`、idle は 0 |
| 閾値未満 | 腕を下ろす | wall と idle の両方を進める |
| wall ≥ 90 | モータレーンを閉じる | — |
| idle ≥ 50 | モータレーンを閉じる | — |
| 外れ後 50 step 以内に戻る | 真似を再開 | idle を 0 に戻す |

成功は真似を始めた時点。手を下ろしている時間も 90 step に含まれる。

---

## 6. 顔離脱タイムアウト

枠が外れたときの閉じ。成功条件ではない。

- 見る範囲は親の `face_links`（目・頭・唇）。瞬間値は `instant_parent_face_hit`
- 連続 30 step（約 3 秒）顔から離れたら、セッション全体を閉じる
- フラグが既にあれば成功、無ければ失敗。最初の待ちで顔が離れたときは「ねえ」を言わずに失敗
- 顔に戻った step で、離れていた時間だけゼロに戻る。モダリティの待ちは戻らない

顔の時計が**止まる**条件:

- 親が見本を出しているあいだ（`performing_parent_cue`）
- 親がオウム返しを言っているあいだ（`echoing` かつ発話キューが残っている）
- 幼児が発話区間の中（`voice_segment.in_segment`）
- モータが閾値以上のあいだ

進むのは、次の喃語かもう一度の掲げを**待っているあいだ**だけ。

開始の手がかり `looking_at_parent_eye`（目だけ、30 step 平均）は変えない。顔の平均キー `looking_at_parent_face` も検出器に出すが、タイムアウトの判定は瞬間値。

---

## 7. 掲げ動作

### 掲げ位置（`_compute_gesture_pos`）

- XY: 親の胴体前方 `gesture_forward_dist = 0.35` m
- Z: **計算した瞬間**の幼児の `eyes_link` の高さ

チャネルに入るときに 1 回だけ計算して固定する。幼児が動いても追従しない（[../parent_tools.md §5](../parent_tools.md#5-見せる置く)と同じ設計）。

### 到達待ち（`_update_gesture_reach`）

腕チャネルは**掲げ位置に到達するまでタイマーを進めない**。到達していないうちにリズムを始めると、幼児から見えない位置で手が動くことになる。

- 到達判定: `palm_reach_error < gesture_reach_tol = 0.26` m
- 到達しなくても `gesture_reach_timeout_steps = 20`（≒2 秒）でデッドロックを避けるため強制的に到達扱いにする

`voice` チャネルは到達を待たない。

### 行動の重ね方（`_apply_motor_channels`）

1. 腕チャネルがあれば `hand_reach_location(gesture_pos)` で掲げを維持
2. その上に `wave_arm(gesture=True)`（肩ピッチに加算）や `animate_hand_open_close(gesture=True)`（指を上書き）を重ねる
3. `_tick_motor_timers()` でタイマーを進める

### 行動生成の全体（`_determine_action`）

```
1. gaze_agent(親 → 幼児, yaw_link="head")       毎 step 必ず
2. モータが進行中なら _apply_motor_channels
   持続中だが閾値未満なら reset_arm_pose
   そうでなく、かつ imitating / performing_parent_cue / echoing のいずれでもなければ reset_arm_pose
3. _emit_utterance                              発話キューを 1 音素消化
```

---

## 8. 応答マッチ注釈

ピーク窓が開いてからプロトコル終了まで、各チャネルの salience のピークを記録し続ける。実行時の分岐（「すごい」を言うか、成功か失敗か）には**一切影響しない**。評価用の注釈である。

| 記録 | 意味 |
|------|------|
| `channel_peak_scores` | チャネルごとの salience 最大値 |
| `channel_on` | ピークが τ を超えたか（0/1） |
| `response_matched` | **親が見せた全チャネル**が `channel_on` になったか（0/1） |
| `time_to_engagement` | ピーク窓を開いてから反応を検出するまでの step 数 |

`debug_info["response_match"]` に入り、PM が履歴へ保存する。

---

## 9. 発話パターン

喃語のオウム返しは、列を 1 音 `PHONEME_ECHO_STEPS = 6` で 1 回だけキューする。30 step の引き伸ばしと 2 回繰り返しは使わない。親が言い終わったら保持中の列を消す。同じ列で次の往復は始まらない。

親始動の声の見本は `PARENT_INIT_CHUNK = ["p0", "p1", "p0", "p1"]`。

`prompt`（「ねえ」）は幼児始動の最初の待ちのリトライだけ。`praise`（「すごい」）は今回出さない。

---

## 10. PM との関係

### 起動の判定

`intent.dyadic_is_salient` が `DyadicProtocol.compute_channel_salience` と `channel_tau` を**そのまま呼ぶ**。意図レイヤと FSM で判定がずれないよう、閾値と式を 1 か所に集めてある（[../intent.md](../intent.md)）。

### `infant_is_engaged`

次のいずれかで真。

- 状態が `imitating` または `echoing`
- 声レーンまたはモータレーンが開いている
- 一度でも反応を検出した（`_infant_has_responded` / `session_success`）
- いま `_detect_infant_response` が真

### `blocks_interrupt`

次のいずれかの間、PM は割り込みくじを引かない。

- 親の発話キューが残っている
- 幼児が発話区間の途中（`voice_segment.in_segment`）

---

## 11. 注意事項

**`imitating` 状態は旧単発経路の互換。**
モータは持続レーン（`motor_active`）へ移す。新規の成功経路は `waiting_babble` 上のレーン並走。

**`head_nod` チャネルは廃止された。**
以前は首振りもチャネルだったが、`MOTOR_CHANNELS` から外れている。`ParentTools.nod_head` は模倣デモがまだ呼ぶ。うなずきの周期性キーは出さない。二項からはどちらも呼ばれない。

**`solo` / `function_type` / `relation_type` / `is_urgent` は分類ラベルで、分岐には使われない。**

**`imitating_type` コンストラクタ引数は使われない。**
チャネルは salience から決まるので、この引数は読まれない。

**`arm_raise` の τ が 0.8 なのは意図的に高い。**
`hand_height_margin` は直近 10 step のヒット率なので、0.8 は「直近 1 秒のうち 8 割は挙げ続けている」を意味する。一瞬挙げただけでは模倣しない。
