# 家事・単独行動 (`protocols/household_solo.py`)

親が幼児と関わらずに家事をする。色の合う board へ item を 1 つ運んで置く。

共通契約は [README.md](README.md)。

| 属性 | 値 |
|------|-----|
| `relation_type` | `"solo"` / `solo` = `True` |
| `has_target` | `False`（対象は board と item 側。`target_object` は使わない） |
| `parent_utterance` | `False`（家事中は喋らない） |
| `persistence` | 常に 5.0（[§5](#5-pm-との関係)） |

**このプロトコルの役割は「親が常に幼児を見ているわけではない」状態を作ること。** 幼児から見れば、親が別のことに集中している状況で、どうやって注意を引くかという問題になる。

---

## 1. タスクの決まり方

タスクの候補は `BaseEnv` が組み立てる（`household_task_candidates`）。プロトコルは Wrapper を遡って `household_task_candidates` を持つ層を探し、そこから選ぶ。

### スキーマ（`household_schemas.py`）

`TASK_SCHEMAS` は色ごとの 4 つ。

| `task_type` | ラベル | `category` |
|-------------|--------|-----------|
| `collect_red` | 赤い物を集める | `red` |
| `collect_blue` | 青い物を集める | `blue` |
| `collect_green` | 緑の物を集める | `green` |
| `collect_yellow` | 黄色い物を集める | `yellow` |

### 候補の組み立て（`BaseEnv._build_household_task_candidates`）

各カテゴリについて、

- **board**: そのカテゴリの `ColorBoard` が active であること
- **item**: 色が一致し、`is_platform` でなく、`put_on` が未設定で、どの board にも乗っておらず、誰にも持たれていないこと

両方が 1 つ以上あれば、`(task_type, board_id, item_ids)` を board ごとに候補へ追加する。候補が 1 つも無ければ、プロトコルは開始直後に `completed` になって終わる（何もせず終了）。

`context["fixed_task"]` を渡すとその `task_type` に限定できる。実行不能なら開始せず終わる。

---

## 2. item の選び方（foraging）

`moving_to_item` に入るたびに `select_next_item` で次の item を確率的に選ぶ。決め打ちではなく、距離と典型性で重み付けしたソフトマックス。

各候補 item のロジット。

```
dist_score = exp(-距離 / temperature)
logit      = distance_weight * log(dist_score) + typicality_weight * log(典型性)
確率        = softmax(logit)
```

- 距離: 親の body から item までの距離
- 典型性: クラス名から引くスコア（cube 1.0 / sphere 0.85 / cylinder 0.8 / cone 0.75、未登録は 0.7）
- `distance_weight` / `typicality_weight` / `temperature`: `household_style` ごとの重みと温度

### `household_style`（`HOUSEHOLD_STYLE_PARAMS`）

| style | `distance_weight` | `typicality_weight` | `temperature` |
|-------|------------------|--------------------|--------------|
| `committed`（既定） | 2.0 | 0.5 | 0.8 |
| `opportunistic` | 3.0 | 0.2 | 0.5 |
| `monitoring` | 1.0 | 0.8 | 1.0 |
| `interrupted_but_returning` | 1.0 | 1.0 | 1.0 |

`opportunistic` は近い物に飛びつく、`monitoring` は距離を気にせず典型的な物を選ぶ、という性格の違いを表す。**現状 `context["household_style"]` で指定しない限り `committed` 固定**で、確率的な割り当ては入っていない。

ロジットを softmax に通して選び、選択理由（各候補の距離・典型性・確率）を `item_selection_history` に残す。選ぶたびに標準出力へも表示する。

---

## 3. 状態遷移

```
init
 └→ (タスク選択 → item 選択) →┬→ hesitating ────┐
                              ├→ inspecting_item ┤
                              └──────────────────┴→ moving_to_item
                                                        ↓ 距離 < 0.6
                                                   grasping_item
                                                        ↓ ┬→ inspecting_board ─┐
                                                          └────────────────────┴→ moving_to_board
                                                                                     ↓ 距離 < 0.45
                                                                                placing_on_board
                                                                                     ↓ board に乗った
                                                                                 check_done → completed
```

`state_proximities`: `init` 0.0 / `inspecting_item` 0.2 / `hesitating` 0.22 / `moving_to_item` 0.3 / `grasping_item` 0.5 / `inspecting_board` 0.65 / `moving_to_board` 0.7 / `placing_on_board` 0.85 / `check_done` 0.9 / `completed` 1.0。

### `moving_to_item`

`approach_object(approach_dist=0.5)` で item へ近づく。距離が 0.6 を切ったら `grasping_item`。

### `grasping_item`

`grasp_object(approach_dist=0.5)`。掴めたら `first_hand_status` を丸ごとクリアして先へ進む。100 step 掴めなければ `_enter_moving_to_item()` で**別の item を選び直す**（失敗しても諦めず、次の物へ行く）。

### `moving_to_board`

`approach_object(approach_dist=0.32)` ＋ `grip_object`（運搬中の落下防止）。距離 0.45 未満で `placing_on_board`。

### `placing_on_board`

`_place_on_board_action`。

1. board まで 0.4 より遠ければ、近づきながら握り続ける
2. 近ければ board 中央の 0.18 m 上へ `hand_reach_location` し、指の目標角を `-3.14` にして**強制的に開く**

手放した物は簡易物理で落ち、board の上に重なれば `is_platform` の判定で `objects_on_board` / `put_on` が更新される（[../env.md §7](../env.md#7-接触把持受け渡し)）。

`_is_item_done` が真になったら `check_done`。200 step 経っても乗らなければ、手放し済みなら item を選び直し、まだ持っていれば `moving_to_board` からやり直す。

### `check_done`

`items_placed` を 1 増やして即 `completed`。**1 個置いたら終わり**で、候補が残っていても続けない。

---

## 4. inspection と hesitation

家事の途中に「立ち止まって見る」「迷う」を挟むための状態。いずれも接近せず、首だけ動かす。

| 状態 | 入る確率 | 長さ | 動作 |
|------|---------|------|------|
| `hesitating` | `hesitate_prob` | `hesitate_duration`（25） | 選んだ item と別候補を 20 step 周期（10 step ずつ）で交互に見る |
| `inspecting_item` | `inspect_item_prob` | `inspect_item_duration`（40） | 選んだ item を見る |
| `inspecting_board` | `inspect_board_prob` | `inspect_board_duration`（35） | board を見る |

判定の順序（`_enter_post_selection_state`）は hesitate が先で、外れたら inspect を判定する。両方外れれば直接 `moving_to_item`。`inspecting_board` は掴んだ直後に判定する。

**既定はすべて確率 0（オフ）。** `HOUSEHOLD_INSPECTION_PARAMS` の値は `start` の `context` から上書きできる。デモ用のプリセット `HOUSEHOLD_INSPECTION_DEMO_ON` は item / board の inspection を必ず出し、hesitate は出さない。

入った記録は `inspection_history` に残る（種別・対象・長さ・迷った先の候補）。

---

## 5. PM との関係

### `persistence` は常に 5.0

`_calculate_persistence` をオーバーライドして固定値を返す。基底の「典型ステップ数と幼児の関与から計算する」式は使わない。

PM の割り込み閾値と比べて十分大きいので、**家事中はほとんど割り込めない**。幼児が強い信号を出したときだけ中断される。親が家事に没頭している状況を作るためのパラメータ。

### `keeps_held_object`

運んでいる item そのものなら `True`。割り込まれても持ったままにする。

### 割り込み後の扱い

`blocks_interrupt` はオーバーライドしていないので、割り込み自体は禁止していない。閾値の高さだけで抑えている。

---

## 6. 注意事項

**`check_done` で必ず終わるので `items_placed` は 0 か 1 しか取らない。**
「集める」という名前だが、1 回の起動で運ぶのは 1 個。複数個を続けて運ぶ実装は入っていない。

**`select_next_item` は毎回 print する。**
候補ごとの距離・典型性・確率を標準出力に出す。長いエピソードでは出力が大量になる。

**`grasping_item` の失敗で `first_hand_status` を全消しする。**
他のプロトコルは親のエントリだけを消すが、ここは辞書ごとクリアしている。

**`rejection_check` は既定のまま。**
`BaseProtocol.__init__` の既定値で作られるが、`_manage_state` から `detect_rejection` を呼んでいない。幼児の拒否では中断しない（そもそも幼児に対して何もしていない）。

**`hesitating` の遷移先は常に `moving_to_item`。**
`_hesitation_next_state` は引数で変えられる作りだが、呼び出しは 1 か所で既定値しか使われていない。

**`_place_on_board_action` は指の角度を直接書いている。**
`action_map` の `index_joint` / `other_fingers` に `-3.14` を代入する。`open_hand` などのツールを経由していないので、他のツールと重ねると打ち消し合う可能性がある。
