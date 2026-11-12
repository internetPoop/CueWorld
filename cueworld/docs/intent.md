# 意図レイヤ (`intent.py`)

`InfantSignalDetector` が出す連続値の原子シグナルから、**「幼児が何を訴えているか」の 0/1 セット**を合成する層。`ProtocolManager` はこの 0/1 だけを見てプロトコルを選ぶ。

- 入力: `info["infant_signals"]`（[infant_signal.md](infant_signal.md)）
- 出力: 5 つのフラグ（`INTENT_KEYS`）と、単語から引いた概念 ID
- 使い方: [protocol_manager.md](protocol_manager.md)

**なぜ間に層を挟むのか。** 原子シグナルをそのまま重み付き和にすると、部品（手の安定度・声量・腕の伸び）がばらばらに足し込まれて、「どれも中途半端だが合計は大きい」という状態でプロトコルが起動してしまう。意図レイヤは**部品が揃って初めて 1 になる AND 条件**にすることで、これを防ぐ。

---

## 1. 5 つの意図

| キー | 意味 | 条件 |
|------|------|------|
| `pointing` | 揃った指差し | `point_intensity > 0` かつ手の対象がはっきりしている |
| `reach` | 揃ったリーチ | `open_intensity > 0` かつ持っていない かつ手の対象がはっきりしている |
| `weak_arm` | 弱い腕 | 持っていない かつ手の対象がはっきりしている（手の形は問わない） |
| `word` | 単語の訴え | 声があり、尖っていて、既知語に十分近い |
| `dyadic` | 二項的な働きかけ | `DyadicProtocol` のいずれかのチャネルが閾値以上 |

`pointing` と `reach` は `weak_arm` の条件を含むので、これらが立つときは `weak_arm` も立つ。**弱い腕は「形は未熟だが方向は示している」段階**を拾うための、薄い訴えとして用意されている。
- dyadicの開始判定は，目線が合うことを前提にするべき？(幼児始動の場合)

### 手の対象がはっきりしている（`hand_target_is_clear`）

`pointing` / `reach` / `weak_arm` に共通の前提。

| 条件 | 閾値 |
|------|------|
| `hand_target_stability >= HAND_TARGET_STABILITY_MIN` | 0.6 |
| `max(gesture_targets) >= HAND_TARGET_MASS_MIN` | 0.2 |

前者は**手の向きが定まっていること**（時間方向）、後者は**特定の物にレイが集中していること**（空間方向）。両方揃わないと、腕を振り回しているだけで対象を指したことになってしまう。

### 単語（`word_candidate`）

3 つすべてを満たすとき、最良の概念 ID を返す。1 つでも欠ければ `None`。

| 条件 | 閾値 |
|------|------|
| `voice_volume >= voice_volume_threshold` | 0.25（`DEFAULT_VOICE_VOLUME_THRESHOLD`。`BaseEnv.voice_volume_threshold` と同期） |
| `voice_target_stability >= WORD_STABILITY_MIN` | 0.6 |
| `voice_best_distance <= WORD_MATCH_MAX_DISTANCE` | 0.6 |

- 声量: **いま声を出していること**。過去に言った単語が残っているだけでは訴えにしない
- 安定度: 単語の分布が尖っていること。複数の語に同程度で当たっている状態を弾く
- 距離: 認識の質。遠ければ「何か言ったが単語ではない」とみなす

`voice_volume_threshold` は呼び出し側から差し替えられる（PM は `env.voice_volume_threshold` を優先する）。

### 二項（`dyadic_is_salient`）

`DyadicProtocol.compute_channel_salience` を呼び、どれか 1 チャネルがそのチャネルの閾値以上なら真。**判定ロジックをプロトコル側に一本化している**ので、閾値の定義は [protocol/dyadic.md](protocol/dyadic.md) が正。

---

## 2. 親側の手がかり

幼児が親に向けて出しているかどうかを見る、別系統の 0/1。意図フラグには含まれず、PM が親始動の判断（幼児が受け入れそうかどうか）に使う。

### `parent_object_cue`

**物への関心があるか。** 次のいずれかで 1.0。

- `gaze_target_stability >= 0.6`（何かをじっと見ている）
- `touching > 0`
- `holding > 0`

対象が何かは問わない。「どれか物に関わっている」という緩い判定で、物を介した遊び（共同注意・受け渡し）を親から持ちかけてよいかを見る。

対象一致まで要求する `_infant_attending_current_object`（[protocol/README.md §5](protocol/README.md#5-persistence)）とは別物なので混同しないこと。

### `parent_dyadic_cue`

`looking_at_parent_eye >= 0.6` なら 1.0。**親の目を見ているか**。物を介さない二項の遊びを持ちかけてよいかを見る。

---

## 3. PM 側のゲート

`compute_intent_flags` は環境の事実を知らない。**対象 ID の解決と到達圏の判定は PM が行い、そのあとでフラグを落とす**（`_calculate_signal_strength`）。

| ゲート | 落とすもの | 理由 |
|--------|-----------|------|
| 単語の概念が場に無い | `word` | 場に無い物の名前を言っても、渡しようがない |
| 単語の物が幼児の到達圏内 | `word` | 自分で取れる物を「くれ」とは解釈しない |
| 手の対象が幼児の到達圏内 | `reach`, `weak_arm` | 同上 |
| 幼児が何か持っている | `reach`, `weak_arm` | 手が塞がっているのでリーチではない |
| 揃った指差しが立っている | `word` | 指差し優先。同時に喋っていても、それは「くれ」ではなく共同注意の訴えとみなす |

**`weak_arm` の扱いに非対称がある。** 到達圏ゲートで落とすと共同注意側の寄与も一緒に消えるが、行列の重みが両方の行に乗っているため、行ごとに分けられない。現状は「届く物への腕は訴えにしない」として両方落としている（コードにこの経緯のコメントが残っている）。

### 対象 ID の解決

| 名前 | 解決方法 |
|------|---------|
| `hand_target_id` | `gesture_targets` の最大要素（親を除く）に対応する `trackable_targets` の ID |
| `word_object_id` | `word_candidate` の概念 ID と `concept_id` が一致する active な物体 |

到達圏の判定は `env.object_in_agent_reach(infant, oid)`。**親が持っている物は届かない扱い**にする。

### `intent_source`

ゲート後に残ったフラグから 1 つだけ選んで、どの訴えが起動理由になったかを記録する。優先順は `pointing` → `reach` → `weak_arm` → `word` → `dyadic`。プロトコルに渡す `context["intent_source"]` と履歴に載る。

---

## 4. 訴えの強度への変換

ゲート後の 5 次元ベクトルに重み行列を掛けて、プロトコルごとのスコアにする。

| プロトコル | `pointing` | `reach` | `weak_arm` | `word` | `dyadic` |
|-----------|-----------|--------|-----------|-------|---------|
| `joint_attention` | 2.0 | – | 0.3 | – | – |
| `object_giving` | – | 2.0 | 0.3 | 2.0 | – |
| `dyadic` | – | – | – | – | 2.0 |

完成形（`pointing` / `reach` / `word` / `dyadic`）は 2.0 で、softmax の温度 0.2 に対して十分大きい。**弱い腕の 0.3 は、完成形が無いときだけ効く**薄い重み。どちらのプロトコルにも等しく乗るので、弱い腕だけでは共同注意と受け渡しのどちらが選ばれるか決まらない。

このスコアの使い道は [protocol_manager.md](protocol_manager.md)。

---

## 5. 注意事項

**`point_intensity` / `open_intensity` は「> 0」でしか使っていない。**
強度の大きさで閾値判定はしていない。`reach_is_complete` などは「その形かどうか」のゲートとして読む。将来、強度の閾値を上げる使い方をする場合は intent 側の変更が必要。

**`holding` は `held_obj` の有無。**
`get_hand_status` が `agent.get("held_obj")` を読み、あるとき 1.0（[infant_signal.md §9](infant_signal.md#9-注意事項)）。`reach_is_complete` と `weak_arm` の「持っていない」条件、および PM のくれゲートは、この値をそのまま使う。幼児が物を持っているあいだ、リーチと弱い腕の訴えは立たない。

**`signal_keys` は使われていない。**
PM に原子シグナル名の並びが残っているが、訴えの計算には `intent_keys` しか使わない。ログと互換のために残っている。

**閾値の 0.6 は根拠が揃っていない。**
`HAND_TARGET_STABILITY_MIN` / `WORD_STABILITY_MIN` / `PARENT_GAZE_CUE_MIN` / `PARENT_EYE_CUE_MIN` はいずれも 0.6 だが、これは共同注意の判定に合わせた暫定値。
