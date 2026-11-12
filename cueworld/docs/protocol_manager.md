# プロトコルマネージャ (`protocol_manager.py`)

親が「いま何をするか」を決める層。幼児の意図（[intent.md](intent.md)）を受け取り、プロトコル（[protocol/README.md](protocol/README.md)）を選んで走らせる。

本ドキュメントは**切り替えの手続きの正本**である。意図の定義は [intent.md](intent.md)、各プロトコルの中身は `docs/protocol/` が正。

```python
env = ProtocolManager(InfantSignalDetector(ParentTools(BaseEnv(...))))
obs, r, d, t, info = env.step(infant_action)
parent_action = info["parent_action"]
```

---

## 1. step の順序

`step(action)` は次の順で進む。

1. 下位 `env.step(action)`。幼児の行動が環境に入り、シグナルが更新される
2. `get_normalized_signals()` を呼び直して `info["infant_signals"]` を作る
3. `update(infant_signals)` で親の行動を決める（[§2](#2-update-の中身)）
4. 親の行動を `info["parent_action"]` へ、プロトコルの報酬を `reward[0]` に加算
5. `fixed_protocol` モードでプロトコルが終わっていれば `done = True`

**親の行動は同じ step の中で環境に入らない。** 呼び出し側が次の step で `info["parent_action"]` を混ぜて渡す（[env.md §5](env.md#5-行動空間)）。

## 2. `update` の中身

```
check_counter += 1
if check_counter >= check_interval:        # 既定 10 step ごと
    ペナルティを減衰
    if fixed_protocol でない:
        (a) 割り込み判定  check_for_response
        (b) 見守り中なら → household の時計 → 親始動判定 check_for_initiative

現在のプロトコルを 1 step 進める（__call__）
終了していれば後処理（除外リスト・ペナルティ・次のプロトコル）
```

**判定は毎 step ではなく `check_interval`（既定 10 step ≒ 1 秒）ごと。** 親が幼児の様子を見て考え直す間隔にあたる。プロトコルの進行自体は毎 step 行う。

### 判定の優先順位

1. `blocks_interrupt` が真なら**何も判定しない**（[protocol/README.md §8](protocol/README.md#8-pm-へ渡すフック)）
2. 幼児の訴えによる割り込み（`check_for_response`）
3. 見守り中に限り、家事の時計（`_maybe_start_household`）
4. それでも何も起きなければ、見守り中に限り親始動（`check_for_initiative`）

**家事より幼児の訴えが優先**で、**関わりの最中には家事も親始動も割り込まない**。3 と 4 が `observance` のときだけ走るのはそのため。

### 終了時の後処理

プロトコルが `is_finished` を立てたら、

1. 除外リストを更新（[§6](#6-ペナルティと除外リスト)）
2. 失敗状態で終わっていればペナルティを加算
3. 次のプロトコルを決める。家事が終わったなら時計を押して `observance` へ。それ以外は、家事の期限が来ていれば**`observance` を挟まず**家事へ、来ていなければ `observance` へ

---

## 3. 幼児の訴えによる切り替え（`check_for_response`）

この経路で始まったプロトコルは**幼児始動**（`infant_initiative = True`）。

### スコア

意図ベクトル（0/1、5 次元）に重み行列を掛けて、プロトコルごとの訴えの強度を出す。行列と各ゲートは [intent.md §3-4](intent.md#3-pm-側のゲート)。

そこから 2 つの補正を掛ける。

1. **ペナルティを引く**（[§6](#6-ペナルティと除外リスト)）
2. **二項ゲート**: ペナルティ後に `joint_attention` か `object_giving` のスコアが正なら、`dyadic` を 0 にする

二項ゲートは「物への訴えがあるなら物の遊びを優先する」という順序付け。ただし**拒否・ペナルティ・到達圏で物の遊びが 0 に落ちている場合は二項を残す**ので、物の遊びが選べない状況で二項まで消えることはない。

### 維持（`maintain`）

現在のプロトコルを続ける選択肢を、ダミーのクラスとしてくじに混ぜる。

```
maintain_score = persistence * persistence_scale + persistence_bias
```

`persistence` はプロトコル側が計算する（[protocol/README.md §5](protocol/README.md#5-persistence)）。見守り 1.0、家事 5.0、関わりは 2.7 か 5.0 の 2 値。

### くじ

```
choices = ["maintain"] + response_protocol
probs   = softmax([maintain_score, S_1, ..., S_n] / tau_response)     # tau_response = 0.2
```

除外リストで落ちる選択肢を外して正規化し直し（[§6](#6-ペナルティと除外リスト)）、1 回サンプリングする。`maintain` または現在と同じ名前が出たら切り替えない。

**確率的に選ぶのは、同じ状況でも親が毎回同じ反応をしないようにするため。** 温度 0.2 は、完成形の重み 2.0 に対して十分小さく、訴えが立っていればほぼ確実にそちらへ動く。

### 渡す context

| キー | 内容 |
|------|------|
| `infant_initiative` | `True` |
| `infant_signals` | そのときのシグナル |
| `global_ignored_targets` | その遊びについて除外済みの物 ID |
| `intent_target_id` | 訴えから解決した対象（下記） |
| `intent_source` | どの訴えが起動理由か |
| `intent_type` | 共同注意のみ。常に `"informative"` |
| `target_phoneme` | 二項のみ。`babbling_target` があれば |

`intent_target_id` の決め方（`_intent_target_for_protocol`）。

- `joint_attention`: 手の対象（`hand_target_id`）
- `object_giving`: 手の対象があり、かつ `reach` か `weak_arm` が立っているならそれ。無ければ単語から引いた物
- それ以外: なし

### 割り込みは失敗として記録する

切り替えのとき、前のプロトコルが関わり（`solo` でない）なら `end_reason="interrupt"` で失敗として履歴に積む。

---

## 4. 親始動（`check_for_initiative`）

見守り中だけ走る。この経路で始まったプロトコルは**親始動**（`infant_initiative = False`）。

くじは 2 段構え。**「始めるかどうか」と「何を始めるか」を分ける**ことで、候補の数が増えても開始の頻度が変わらないようにしている。

### 一段目: 待つか、始めるか

```
wait_score  = wait_base_score + s_spon        # wait_base_score = 1.04
start_score = cue_bonus if 手がかりが1つでも立つ else 0.0    # cue_bonus = 0.36
p_wait, p_start = softmax([wait_score, start_score] / tau_initiative)   # tau = 0.2
```

- `s_spon` は幼児の自発性（`infant_initiative_rate`、[§7](#7-学習の記録)）。試行回数が `base_try_count` に満たないうちは 0
- **自発性が高いほど待機スコアが上がり、親は自分から始めなくなる。** 幼児が自分で訴えられるようになったら親が引く、という「待つ行動の創発」がここで起きる
- 手がかり（`parent_object_cue` / `parent_dyadic_cue`）は [intent.md §2](intent.md#2-親側の手がかり)。**指差し・リーチ・単語は手がかりに使わない**（それらは訴えなので response 経路の仕事）

既定値では、自発性 0 で手がかりなしなら約 3 分に 1 回、手がかりありなら約 30 秒に 1 回の頻度で開始する。

### 二段目: どの遊びか

候補は `request_protocol`（`request_type` が真で `solo` でないもの）から、次の条件で絞る。

- レベルが `max_level` 以下（下記）
- `object_giving` は、幼児が何か持っている、または届かない物が 1 つも無ければ候補外（`_giving_initiative_allowed`）

```
zpd  = level_success * (1 - level_success)      # 成功率 0.5 で最大
zpd += frontier_boost if zpd < base_zpd_score   # 未踏レベルへの下駄 (+0.1)
魅力度 = zpd + cue_bonus * 手がかり - ペナルティ
p_which = softmax(魅力度 / tau_initiative)
```

`max_level` は、`zpd >= base_zpd_score`（0.1）を満たす最大のレベル ＋ 1。**「半分くらいできるレベル」の 1 つ上まで**を候補にする（発達の最近接領域）。試行回数が足りないレベルは成功率 0 として扱われるので、`zpd = 0` になり候補の上限に寄与しない。

### 結合

```
probs = [p_wait, p_start * p_which[0], ..., p_start * p_which[n]]
```

一段目と二段目を掛けた結合分布から 1 回だけサンプリングする。除外リストのフィルタは response と同じ。`observance` が出たら何も始めない。

---

## 5. 家事の時計

家事（`household_solo`）は `request_protocol` に入れず、**別建ての時計**で開始する。

| 条件 | 内容 |
|------|------|
| 期限 | 前回家事の**完了**から `household_interval_steps` 以上経過 |
| 候補 | `household_task_candidates` が空でない（[protocol/household_solo.md §1](protocol/household_solo.md#1-タスクの決まり方)） |

`household_interval_steps` のコンストラクタ既定は **200**（デモや短い観測用）。本番ランは `HOUSEHOLD_INTERVAL_PRODUCTION = 6000`（約 10 分）を明示的に渡す。**既定を 6000 にはしない。**

入るタイミングは 2 か所。

- 見守り中で、幼児の訴えによる切り替えが起きなかったとき（親始動の判定より**前**）
- 関わりが終了したとき（`observance` を挟まずに直接）

家事から抜けるときに `last_household_end_step` を更新して時計を押す。

---

## 6. ペナルティと除外リスト

### プロトコルペナルティ（`protocol_penalties`）

失敗状態で終わったプロトコルに `base_penalty_value = 2.0` を加算する。判定のたびに `penalty_decay_rate * check_interval`（= 0.5）ずつ減衰する。

response のスコアからも initiative の魅力度からも引かれるので、**失敗直後のプロトコルはしばらく選ばれにくくなる**。

### 除外リスト（`blocked_pairs`）

単位は**（遊びの名前, 物の ID）のペア**。生のシグナルはマスクしない（幼児が何を訴えているかは見えたまま、親がその組み合わせを選ばなくなるだけ）。

| きっかけ | 動作 |
|---------|------|
| 失敗終了 | そのペアを追加。ただし `blocks_target_on_failure()` が偽なら追加しない（開始時の手放し失敗など） |
| 割り込み | 追加しない |
| 成功終了 | すぐには追加しない。同じペアが `SUCCESS_STREAK_MAX = 3` 回連続で成功したら追加 |
| リストが空でないときの成功終了 | **リストと連続カウントを全部クリア**（家事の成功も含む） |

成功でも 3 回続けば外すのは、**同じ遊びを同じ物で繰り返し続けるのを防ぐため**（飽き）。一度でも別の成功があればリセットするのは、行き詰まりから抜けたら制限を解く、という考え方。

くじでは、その遊びの**見込み対象**がブロック済みなら選択肢ごと落として正規化し直す。見込み対象は、幼児始動なら `intent_target_id`、親始動なら幼児の視線対象。**見込み対象が `None`（対象未定でランダムに選ぶ）なら落とさない。**

`reset()`（幼児の睡眠）でリストと連続カウントは消える。ペナルティも消える。

---

## 7. 学習の記録

関わりプロトコル（`solo` でないもの）が終わるたびに指数移動平均を更新する。家事と見守りは**混ぜない**（幼児の学習課題ではないため）。

```
alpha = 2.0 / (base_try_count + 1.0)          # base_try_count = 30 → alpha ≈ 0.065
新しい値 = 古い値 * (1 - alpha) + 結果 * alpha
```

| 名前 | 更新する結果 | 使い道 |
|------|-------------|--------|
| `level_scores[level]` | 成功なら 1 | 親始動の ZPD |
| `communication_protocol_scores[name]` | 成功なら 1 | 現状は記録のみ |
| `infant_initiative_rate` | **幼児始動かつ成功**なら 1、それ以外（親始動・失敗・割り込み）は 0 | 親始動の待機スコア |

`infant_initiative_rate` は**割合ではない**。幼児始動の成功だけが押し上げ、親始動で成功しても下がる。この値が上がるほど親は待つようになる。

`urgency` が真のプロトコルはレベル・プロトコル別の EMA から外れる（現行カタログには該当なし）。

`decay_rate` はコンストラクタで直接指定もできる。

---

## 8. 関わりの履歴

関わりプロトコルが終わる（成功・失敗・割り込みのいずれでも）たびに、直近 32 件（`PROTOCOL_HISTORY_MAXLEN`）のリングバッファへ 1 件積む。

| 欄 | 内容 |
|----|------|
| `name` | PM のキー |
| `success` | 成功したか |
| `infant_initiative` | **終了するインスタンス側の旗**。割り込みで PM のフラグが次のプロトコル用に書き換わっていても、こちらを使う |
| `target_id` | 終わった時点の対象 |
| `start_step` / `end_step` / `length` | 開始・終了の step と長さ |
| `kind` | 共同注意は `cue_type`、二項は使ったチャネル（`active_channels`、空なら `salient_channels`）、受け渡しは `None` |
| `end_reason` | `"success"` / `"failure"` / `"interrupt"` |

さらに `history_extra()` の中身が展開される（[protocol/README.md §8](protocol/README.md#8-pm-へ渡すフック)）。共同注意なら `intent_type` / `infant_spoke` / `word_match` / `emotion_reacted`。

`log_protocol_history`（既定オン）が真なら、`step` の `info["protocol_history"]` に直近 8 件のコピーが載る。**プロトコルには渡さない**（`parent_info` は空のまま）。`reset()` でもバッファは残るが、途中だった関わりは書かない。

---

## 9. 固定プロトコルモード

`fixed_protocol` を指定すると、切り替え判定をすべて飛ばしてそのプロトコルだけを走らせる。デモと評価用。

- `reset()` のたびに `fixed_context` で再開始する。`env.target_obj_id` が設定されていれば `fixed_context["fixed_target"]` に入れて対象を固定する
- プロトコルが終わったら `done = True` を返す
- `set_fixed_protocol(name, context)` で外から切り替えられる

---

## 10. 注意事項

**プロトコルのインスタンスは使い回される。**
`self.protocols` は 5 つのインスタンスを持ち続け、切り替えのたびに `start(context)` を呼び直す。`start` でリセットし忘れた属性は前回の値を引きずる。

**`switch_protocol` は `env.clear_cache()` を呼ぶ。**
IK や視線のキャッシュが残らないようにしている。プロトコルをまたぐ状態を `ParentTools` 側に置くと消える。→　別に良い

**`signal_keys` は使われていない。**
原子シグナル名の並びが残っているが、訴えの計算に使われるのは `intent_keys` だけ。`active_signals` から作る `effective_signals` も同様に参照されていない。

**判定のたびに print する。**
`check_for_response` / `check_for_initiative` は毎回スコアを標準出力に出す。切り替えが起きたときも出す。長いランでは出力が大量になる。

**`__getattr__` で下位環境へ委譲している。**
PM に無い属性はすべて下位の env に流れる。`self.eval_mode` などはこの経路で読んでいる。**属性名の打ち間違いが `AttributeError` ではなく下位環境の別の値を拾う形で通ってしまう**ことがある。

**プロトコルの報酬は学習信号に使わない。**
`reward_function` の返り値は `info["protocol_reward"]` にだけ載せ、`reward[0]` には足さない。選好報酬は `InfantSignalDetector` が `reward[0]` に足す（[infant_signal.md §10](infant_signal.md#10-選好報酬)）。`BaseEnv` 単体の報酬は常に 0。

**レベル別 EMA の配列長はカタログ構成に依存する。**
`level_scores` / `level_try_counts` の長さは `level_clusters`（登録プロトコルの `level` の種類数）で決まる。現行は 0（見守り・家事・二項）/ 1（受け渡し）/ 2（共同注意）の 3 段。`level` を飛ばした値のプロトコルを足すと、レベル番号で添字を引く箇所が壊れる。
