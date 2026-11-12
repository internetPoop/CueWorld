# 親の行動ツール (`parent_tools.py`)

`ParentTools` は `BaseEnv` の上に載る `gym.Wrapper` で、親の意思決定（プロトコル）が使うモータプリミティブを提供する。
本ドキュメントは**ツールの契約（前提・引数・書き込み先・返り値）の正本**である。

プロトコルは「どの物へ近づき、掴み、掲げ、振るか」だけを決め、関節をどう動かすかは一切書かない。

現行カタログの 5 プロトコルが実際に呼ぶツールだけを本文に載せる。使われていないものは [§8](#8-使われていないツール) にまとめる。

---

## 1. すべてのツールに共通する契約

**ツールは行動ベクトルへ書き込むだけで、何も実行しない。**

```python
action = np.zeros(env.action_num, dtype=np.float32)
env.gaze_object(env.parent, oid, yaw_link="head", action=action)
env.reset_arm_pose(env.parent, action=action)
# → action をプロトコルが返し、ProtocolManager が env.step((infant_action, action)) に渡す
```

- 書き込み先は `env.action_map` のインデックス。第 1 引数の `agent`（`env.parent` / `env.infant`）が対象。
- **`action` は破壊的に書き換えられる**。同じ `action` に複数のツールを重ねて呼ぶのが通常の使い方で、後から呼んだツールが同じインデックスを上書きする。呼ぶ順序が意味を持つ。
- `action` を省略すると新しい辞書を作って返す。プロトコルからは常に `np.ndarray` を渡す。

### 書き込む値は「残差」であって正規化値ではない

- 関節は `目標角 - 現在角` [rad]、前進は `現在距離 - 目標距離` [m] を書く。
- `BaseEnv` 側が `[-1, 1]` にクリップしてから step 幅を掛ける（[env.md §5](env.md#5-行動空間)）。
- したがって**ゲイン 1・飽和つきの P 制御**として振る舞う。遠ければ毎 step 最大速度、近づくと自然に減速して止まる。プロトコル側で速度を作る必要はない。
- 逆に、`3.14` や `-3.14` のような大きな値は「その方向へ最大速度」という意味の慣用句として使われる（指の開閉など）。

### 状態を持つツール

`ParentTools` はインスタンス変数にキャッシュを持つ。`reset()` ですべてクリアされる。

| 属性 | 用途 |
|------|------|
| `ik`, `ik_target`, `ik_count` | agent ごとの IK 解のキャッシュ。目標が 0.01 m 以上動くと再計算 |
| `shake_t` | `shake_object` の内部位相カウンタ |
| `first_hand_status` | `grasp_object` の掴み直し検出 |
| `_place_xy_best`, `_place_xy_stagnant`, `_place_target`, `_place_releasing` | `place_object_at` の手放し判定 |

**`clear_cache()` を呼ぶタイミングはプロトコルの責任**。とくに `grasp_object` で物を掴んだあとは `env.first_hand_status = {}`（または該当エントリの削除）が必要で、忘れると次の掴みが成立しない。

---

## 2. 視線

### `gaze_agent(agent_from, agent_to, yaw_link, action=None) -> (yaw, pitch)`
### `gaze_object(agent, oid, yaw_link, action=None, option=None) -> (yaw, pitch)`

相手の `eyes_link` / 物体の中心に視線を向ける。

`yaw_link` で「何で横を向くか」を選ぶ。

| `yaw_link` | 挙動 |
|------------|------|
| `"head"` | 首（`head_pan_joint`）で向く。胴体はそのまま |
| `"torso"` | 腰（`torso_joint`）で向く。首の pan は 0 へ戻す |

- pitch は常に `head_tilt_joint` に書く。`option="yaw_only"` / `"pitch_only"` で片方を殺せる。
- yaw 誤差は「胴体前方と胴体→対象の XY 角」から求める。目の前方ではなく胴体を基準にするのは、大きく下を向いたときに目の前方 XY が潰れて `atan2` が暴れるため。
- **近接時の振動対策**が入っている。pitch 誤差が 0.35 rad 未満ならゲイン 0.3 を掛け、yaw / pitch とも 1.5° 未満なら 0 にする。目が tilt 軸より約 25 cm 前にあるため、近距離で下を向くと 1 step の pitch が誤差の符号を反転させて首が振動する。
- 返り値の `(yaw, pitch)` は**書き込み後の残差**。プロトコルは `abs(yaw) < 閾値` で「向き終わった」と判定する。

---

## 3. 移動

### `approach_infant(agent, infant, action=None, thresh=None, approach_dist=0) -> (yaw, distance)`
### `approach_object(agent, oid, action=None, thresh=None, approach_dist=0) -> (yaw, distance)`

「向いてから進む」の 2 段。

1. `gaze_*(yaw_link="torso")` で腰を対象へ向ける
2. yaw 残差が `thresh`（幼児向け既定 20°、物体向け既定 15°）以内なら、`forward` に `現在距離 - approach_dist` を書く

- 前進が始まるのは向き終わってから。斜めに歩くことはない。
- `distance` は向き終わるまで `None`。「まだ向いている途中」と「もう着いた」を返り値で区別できる。
- `approach_dist` は呼び出し側が渡す。物を掴むなら `env.grasp_approach_dist`、渡すために立つなら `env.stand_dist`（[env.md §8](env.md#8-到達圏と配置距離)）。

### `approach_infant_for_ja(agent, infant, oid, action=None, thresh=None, approach_dist=1.0, min_angle_deg=30.0) -> (yaw, distance)`

共同注意用。**親が幼児と対象物の間に入らないように迂回して**近づく。

- 親が幼児より対象物に近く、かつ「幼児→対象」と「親→幼児」のなす角が `min_angle_deg` 未満のときだけ迂回が働く。そうでなければ `approach_infant` にそのまま委譲する。
- 迂回先は、幼児から見て対象方向から `min_angle_deg` だけ回した方向、距離 `approach_dist` の点。回す向きは親がいま直線のどちら側にいるかで決める（遠回りしない）。
- 返り値の `distance` は迂回目標までではなく**幼児までの実距離**。プロトコルの到達判定がずれないようにするため。

### `_dist_body_to(agent, target_pos) -> float`

base から目標点までの XY 距離 [m]。プロトコルが距離を直接見たいときに使う。

---

## 4. 腕と手

### `hand_reach_location(agent, target_pos, action=None, ik_every=0, oid=-1)`

IK で手先（`grasp_link`）を `target_pos` へ運ぶ。

- 対象が 0.01 m 以上動いたら IK を再計算し、それ以外はキャッシュを使う。`ik_every > 0` を渡すと、その周期でも再計算する。
- **腰は現在角に固定して IK を解く**（可動域を現在値で潰す）。IK が勝手に体を回すと、視線ツールが書いた腰の値と喧嘩するため。
- 対象の関節は `joints_for_reach`（腰・肩ヨー・肩ピッチ・肘・手首ロール・手首ピッチ）。
- **必ず指を開く**（`index_joint` と `other_fingers` に `-3.14`）。物を持ったまま運びたいときは、このあとに `grip_object` を重ねて上書きすること。
- `oid` を渡すと `roll_wrist_joint` をその物体に手のひらが向く角に合わせる。

### `palm_reach_error(agent, target_pos) -> float`

palm 中心と目標点の 3D 距離 [m]。「掲げ終わったか」の判定に使う。

### `grasp_object(agent, oid, action=None, approach_dist=0)`

近づく → 手を伸ばす → 掴む、をひとつにまとめたもの。

1. `approach_object(approach_dist)` で接近
2. yaw 残差 10° 以内かつ距離が `approach_dist + 0.01` 以内になったら `hand_reach_object`
3. `env.detect_grasp_object` が対象と一致したら指を閉じる（`3.14`）

`first_hand_status` に「掴み始めたときの指の状態」を記録する。すでに閉じていた場合は一度開くまで閉じ直さない。`BaseEnv` の把持成立条件が `open → closed` の遷移だから（[env.md §7](env.md#7-接触把持受け渡し)）。

**掴んだあとは呼び出し側が `first_hand_status` の該当エントリを消すこと。**

### `grip_object(agent, action=None)`

指を閉じ続けるだけ（`index_joint` と `other_fingers` に `3.14`）。把持の維持に使う。`hand_reach_location` / `shake_object` のあとに重ねる。

### `point_object(agent, oid, action=None, ik_every=0)`

指差し。IK の対象リンクが `point_link`、関節が `joints_for_point`（腰・肩ヨー・肩ピッチ・肘）で、手首は 0 へ戻す。指の形は `index_joint = -3.14`（伸ばす）、`other_fingers = 3.14`（閉じる）で、`BaseEnv` の `pointing` 判定に合わせてある。

### `reset_arm_pose(agent, action=None)`

肩・肘・手首・指を角度 0 へ戻す差分を書く。1 step で戻るわけではなく、step 幅の制限で徐々に戻る。

**待機状態のプロトコルは毎 step これを呼ぶ**のが慣習。腕が前の状態のまま固まるのを防ぐ。

---

## 5. 見せる・置く

### `compute_show_position(agent, target_agent, show_dist=None, from_eyes=True, target_object=None) -> np.ndarray`

物を掲げる目標点。Z は**計算した瞬間**の相手の `eyes_link` の高さ。

- 既定（`from_eyes=True`）: XY は相手の目から `agent` 方向へ水平に `show_dist`。`show_dist` が `None` なら `show_distance_from_eyes(target_object)`（`0.1 + 2 × 半幅`）。共同注意の見せる合図と、渡しの掲げ（`ObjectGivingProtocol` の `showing`）がこれを使う。
- `from_eyes=False`: XY は相手の base から `show_dist`。`show_dist` が `None` なら `env.place_dist`。

**ライブ追従はしない。** プロトコルが状態に入るときに 1 回だけ計算して固定するのが前提。毎 step 計算し直すと、相手が動くたびに IK の目標が動いて手が追いかけ続けてしまう。

### `show_object(agent, target_pos, action=None, reach_tol=0.12) -> (action, reached)`

`compute_show_position` で決めた固定点へ掲げる。中身は `hand_reach_location` ＋ `grip_object`（掲げる間は握り続ける）で、`reached` は palm 中心と `target_pos` の 3D 距離が `reach_tol` 未満かどうか。

接近も視線も含まないので、呼び出し側が別途 `gaze_agent` などを重ねる。`ObjectGivingProtocol` の `showing` が使う。`JointAttentionProtocol` の見せる合図は `hand_reach_location` と `grip_object` を直接重ね、手首は振らない。

### `reach_place_position(target_agent, from_agent, place_dist=0.35, hold_height=0.15) -> np.ndarray`

物を置く狙い点。XY は `target_agent` の base から `from_agent` の方向へ `place_dist` 進んだ点で、アリーナ内にクランプされる。Z は `hold_height` 固定で、**高さは合わせない**（手放したあと簡易物理で落ちる前提。[env.md §10](env.md#10-簡易物理)）。

`place_dist` に `env.place_dist` を渡せば、置いた物が相手の到達圏に入る。

### `place_object_at(agent, target_pos, action=None, release_tol=0.15, release_slack_tol=0.20, stagnant_steps=30) -> (action, released)`

目標点へ手を伸ばし、XY が揃ったら手放す。`ObjectGivingProtocol` の中核。

- `hand_reach_location` は指を開くので、XY が合うまでは `grip_object` で上書きして握り続ける。
- 手放し条件は 2 つ。
  1. palm の **XY** 誤差が `release_tol` 未満
  2. 一度も `release_tol` 未満に届かないまま、`release_slack_tol` 未満の状態が `stagnant_steps` 連続
- 2 つ目は IK が 0.17 m 付近で頭打ちする配置への保険。`release_tol` を単純に上げると、うまくいく場合まで早く手放して物が到達圏の外に落ちる。
- 一度手放し判定に入ったら、`held_obj` が消えるまで開き続ける。指は step 幅の制限でゆっくり開くので、途中で `grip` に戻すと閉じたままになってしまう。
- `released` は「今 step で手放し指令を出したか」。実際に離れたかは `agent["held_obj"]` で確認する。

---

## 6. リズムのあるジェスチャ

いずれも `phase`（0.0〜1.0）を呼び出し側が管理し、正弦波の位相として渡す。プロトコルがステップカウンタから位相を作る。

### `shake_object(agent, action=None, shake_period=20, shake_amp=0.5, target_pos=None)`

把持済みの物を振る。**把持していることが前提**。

- `shake_t` を内部で進めて位相を作る（`phase` 引数は取らない）。1 往復 `shake_period` step。
- `target_pos` を渡すと、その点へ IK で手を保持したまま `pitch_wrist_joint` を振る（上書き。加算すると IK と二重になる）。渡さなければ手首だけを振る。
- 指は閉じたまま維持する。
- 振れ幅が `BaseEnv` の `shake_move_threshold` を超えれば、幼児側の観測に `object_sound` / `object_effect` が立つ（[env.md §9](env.md#9-振る検出)）。

### `wave_arm(agent, action=None, phase=0.0, gesture=False, amplitude_scale=1.0)`

手振り。

- `gesture=False`: 腕全体を「上げた姿勢」に固定し、`shoulder_pitch_joint` を `-0.8 ± 0.35` で往復させる。
- `gesture=True`: **`shoulder_pitch_joint` に正弦波を加算するだけ**。`hand_reach_location` で掲げたあとに重ねて使う（掲げ位置を保ったまま腕を振る）。

### `animate_hand_open_close(agent, action=None, phase=0.0, gesture=False, amplitude_scale=1.0)`

手の開閉。指の目標角を `-1.2`（開）と `1.2`（閉）の間で `0.5(1 - cos(2πφ))` に従って動かす。位相の起点によらず必ず往復になる。

- `gesture=False`: 腕も開閉に合う姿勢へ寄せる。
- `gesture=True`: **指だけ**を動かす（上書き）。掲げの上に重ねる用。

`gesture=True` の 2 つは `DyadicProtocol` が「掲げた姿勢を保ったままリズムを見せる」ために使う。

---

## 7. 加算と上書き

`_apply_joint_target(agent, action, joint_name, target_angle, additive=True)` が内部の書き込み口。

- `additive=True`（既定）: `action[idx] += 目標角 - 現在角`。複数チャネルを同時に走らせるときに互いを潰さない。
- `additive=False`: 上書き。IK が既に書いた値を確定的に置き換えたいときに使う（`shake_object` の手首、`animate_hand_open_close(gesture=True)` の指）。

**同じ `action` に複数のツールを重ねるときは、加算なのか上書きなのかを意識すること。** `hand_reach_location` は上書き、`wave_arm(gesture=True)` は加算、`grip_object` は上書き。

---

## 8. 使われていないツール

現行カタログの 5 プロトコルからは呼ばれない。動作は保証されていない。

| ツール | 備考 |
|--------|------|
| `open_close_object` / `put_in_object` | 容器の開閉・収納。現行カタログの物体はすべて `is_openable=False` / `can_contain=False` なので発火しない（[object.md](object.md)） |
| `hand_over` | 手渡し。`BaseEnv` 側の受け渡しが親→幼児しか動かない（[env.md §13](env.md#13-注意事項)）。`ObjectGivingProtocol` は `place_object_at` を使う |
| `nod_head` | うなずき。`DyadicProtocol` の `head_nod` チャネルが廃止されたため未使用 |
| `open_hand` | 指を開くだけ |
| `approach_location` | 座標へ近づく |
| `is_within_reach` | `env.object_in_agent_reach` のほうが使われる |
| `check_ray_visibility` / `find_visible_cell_for_target` / `get_grid_rc_from_pos` / `get_cell_center` | 遮蔽回避のためのセル探索。`approach_infant_for_ja` の角度ベースの迂回に置き換わった |
| `approach_object_for_visibility` / `approach_infant_for_visibility` | 上記のセル探索を使う接近 |
| `hand_reach_object` | `hand_reach_location` に物体位置を渡すだけ。`grasp_object` から内部的に呼ばれる |

---

## 9. 注意事項

**IK は腰を固定して解く。**
腰角を現在値で可動域ごと潰してから `calculateInverseKinematics` を呼ぶ。副作用として `agent["lower_limits"]` / `upper_limits` の腰の要素が書き換わったまま残る。IK 呼び出しのたびに現在値で上書きされるので実害は出ていないが、この配列を他の目的で読むときは注意。

**`hand_reach_location` は必ず指を開く。**
物を持ったまま運ぶ経路（掲げ・置く・振る）では、後段で `grip_object` を重ねることが前提になっている。順序を逆にすると物を落とす。

**視線ツールのデッドゾーンは 1.5°。**
これより細かく向きを合わせることはできない。プロトコルの「向いた」判定の閾値をこれより小さくしてはいけない。

**`grasp_object` はデバッグ出力を毎 step 標準出力へ出す。**
`print(f"graspable_obj: ...")` が残っている。長時間の実験ではログが膨らむ。

**`_place_*` のキャッシュは agent ごと・目標ごと。**
目標点が 0.01 m 以上動くと停滞カウンタがリセットされる。`place_object_at` に毎 step 微妙に違う目標を渡すと、2 つ目の手放し条件が永久に成立しない。目標点はプロトコル側で固定して渡すこと。

**`ParentTools` は親専用ではない。**
すべてのツールが `agent` を引数に取るので、幼児にも使える。デモやテストで幼児をスクリプト駆動するときに使われる。
