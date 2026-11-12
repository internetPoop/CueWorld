# 物の受け渡し (`protocols/object_giving.py`)

親が対象物を取ってきて、幼児の到達圏内に置き、幼児が拾うまでを扱う。

共通契約は [README.md](README.md)。

| 属性 | 値 |
|------|-----|
| `level` | 1 |
| `has_target` | `True` |
| `response_type` / `request_type` | どちらも `True` |
| `typical_steps` | 175（カタログ中で最長） |

**手渡し（`hand_over`）は使わない。** 親子を結ぶ線上・幼児側の点に置いて、幼児に拾わせる。`BaseEnv` の受け渡しが片方向にしか動かないため（[../env.md §13](../env.md#13-注意事項)）、置くことで代替している。

---

## 1. 状態遷移

```
init → fetching → approach → placing → waiting_infant → reaction → completed
                     ↑           ↑           │
                     │           │           ├─ RETRY: 発話し直して待ち直す
                     │           │           ├─ ESCALATE: ↓
                     │           │           └─ CHANGE_TARGET / FAILED
                     │           │
        fetching → approach → showing → placing   （ESCALATE 経路）
```

`state_proximities`: `init` 0.0 / `fetching` 0.2 / `approach` 0.4 / `showing` 0.5 / `placing` 0.6 / `waiting_infant` 0.8 / `reaction` 0.9 / `completed` 1.0。

### `fetching`

`grasp_object` で対象を掴む。既に持っていれば即 `approach`。掴めた時点で `first_hand_status` の親エントリを消す。

`fetch_timeout_steps = 116`（≒11.6 秒）を超えたら `fail_on_error_timeout()`。

### `approach`

`approach_infant(approach_dist = env.stand_dist)` で幼児に近づき、運搬中の落下を防ぐため毎 step `grip_object` を重ねる。

距離が `stand_dist + 0.1` を切ったら、

- 通常（`escalated == False`）: 置き点を計算して `placing` へ。同時に `present`（「どうぞ ＋ 対象語」）をキュー
- ESCALATE 経路（`escalated == True`）: 掲げ点を計算して `showing` へ。発話も `present`

`approach_timeout_steps = 52`（≒5.2 秒）を超えたら即失敗。

`change_state("approach")` は `last_distance` を 10.0 に戻すオーバーライドを持つ。前の状態で残った古い距離のせいで、入った瞬間に条件を満たしてしまうのを防ぐ。

### `showing`（ESCALATE 専用）

固定した掲げ点へ `show_object` で持ち上げ、顔は幼児へ向ける。

| 条件 | 遷移 |
|------|------|
| 幼児が対象を持った | `reaction`（掲げている最中に取られたら成功） |
| 親が対象を落とした | `fetching`（`escalated` は維持されるので、拾い直したあとまた `showing` に来る） |
| 掲げ点に届き、かつ `state_timer >= show_min_steps`（30） | `placing` |
| `state_timer > show_timeout_steps`（24） | 届かなくても `placing` |

`show_min_steps = 30` は 1 単語ぶんの長さ（`PHONEME_DURATION_FRAMES` 6 × 5 音素）。**「どうぞ」と言い切るあいだは掲げたままにする**ための下限。

掲げ点は `showing` に入るときに 1 回だけ計算して固定し、抜けるときに破棄する（ライブ追従しない。[../parent_tools.md §5](../parent_tools.md#5-見せる置く)）。距離は共同注意の見せる合図と同じ `compute_show_position` の既定で、幼児の目から `0.1 + 2 × 半幅`。置く位置の `place_dist` とは別。`env.show_dist` があれば、その値を目からの水平距離として使う。

### `placing`

`place_object_at` で置き点へ手を伸ばし、XY が揃ったら手放す。顔は対象物へ向ける。

- 手放す前に**幼児が 0.25 m 以上動いたら**置き点を計算し直す。幼児が寄ってきたり離れたりしても到達圏に入るように。
- 手放したあと `release_settle_steps = 8` step 待って落下と静止を見届ける。
- そのあと物と幼児の距離を測り、`reach_radius + 0.03` 以内なら `waiting_infant`。**外なら即失敗**（`fail_on_error_timeout`）。IK が頭打ちして手前に落ちる場合があるため、置いた結果が到達圏に入らなければ待っても無駄という判断。
- 手放す前に `placing_timeout_steps = 84`（≒8.4 秒）を超えたら即失敗。

手放したあとは腕を戻しつつ幼児を見る。

### `waiting_infant`

幼児が対象を持ったら `reaction`。**この状態だけが「幼児の応答待ち」なので、唯一 `handle_timeout` の梯子に乗る。**

`waiting_infant_timeout_steps = 34`（≒3.4 秒）ごとに `handle_timeout(max_retries=3)`。

| 返り値 | 動作 |
|--------|------|
| `RETRY` | `present` を言い直して `state_timer` を 0 に戻す（「ほら、どうぞ」） |
| `ESCALATE` | `escalated = True` にして `fetching` へ。置いた物を拾い直し、今度は掲げて見せてから置く |
| `CHANGE_TARGET` | 別の対象を選んで `fetching` へ |
| `FAILED` | 失敗して終了 |

`escalated` は `placing` に入るときに `False` に戻る（1 回きり）。2 度目以降のタイムアウトは通常経路で置き直す。

置くだけでは気づかない幼児に対して、掲げて見せるという**より強い提示**に上げるのがこの梯子の意図。

### `reaction`

`praise_with_target`（「すごい ＋ 対象語」）を 1 回だけキューして、発話キューが空になったら成功。幼児を見ながら腕を戻す。

---

## 2. PM との関係

### `infant_is_engaged`

基底の `_infant_attending_current_object`（[README.md §5](README.md#5-persistence)）をそのまま使う。**対象物を見ている / 触っている / 持っている**のいずれかで真。「どれか物に関心がある」ではなく対象一致を要求する。

### `keeps_held_object`

持っている物が対象そのものなら `True`。開始時に手放さない（そのまま `fetching` を飛ばして進める）。

### 割り込み

`blocks_interrupt` はオーバーライドしていない。基底の既定（常に `False`）なので、いつでも割り込める。

---

## 3. 注意事項

**`showing` から `placing` へは `present` を再キューしない。**
`_prepare_placing_from_show` は発話を出さない。掲げるときに一度言っているため。

**エラータイムアウトと応答待ちタイムアウトの区別。**
`fetching` / `approach` / `placing` の各タイムアウトは、親のモータが張り付いた場合の保険で `fail_on_error_timeout()` に落ちる。滞在時間の計測対象からも外れる（[README.md §6](README.md#6-タイムアウトとエスカレーション)）。`retry_count` の梯子に乗るのは `waiting_infant` だけ。

**`hold_height = 0.15` は高さを合わせていない。**
置き点の Z は固定値で、手放したあとは簡易物理で落ちる前提（[../env.md §10](../env.md#10-簡易物理)）。

**`place_pos` 未設定のフォールバックがある。**
`placing` で `place_pos` が `None` かつ未把持でもない場合、接近しながら握り続ける経路に落ちる。通常は到達しない。

**`_get_agent_base_xy` は PyBullet を直接呼ぶ。**
他のプロトコルは `env` 経由の距離関数を使うが、ここだけ `p.getBasePositionAndOrientation` を直接叩いている。
