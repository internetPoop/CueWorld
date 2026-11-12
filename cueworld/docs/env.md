# 環境 (`base.py` / `BaseEnv`)

`cueworld` は、親（大人）と幼児（子ども）の社会的インタラクションを構成論的に研究するための PyBullet ベースの 3D 環境である。
その土台が `cueworld/base.py` の `BaseEnv`（Gymnasium 準拠）で、本ドキュメントはその**仕様の正本**である。

`BaseEnv` は「物理と身体の保証」だけを持つ。幼児の行動の解釈、親の意思決定、遊びの進行はすべて Wrapper 側にある。

本書に書くのは、上に載るモジュールが依存してよい契約（次元・単位・値域・不変条件・処理順）である。
実装されている振る舞いだけを書く。コード上に残るが現在使われていない経路は [§13 注意事項](#13-注意事項) にまとめる。

---

## 1. スタック構成

環境は下から順に Wrapper を重ねて使う。各層の正本は次のとおり。

| 層 | 実体 | 正本 |
|----|------|------|
| 環境本体 | `BaseEnv` | 本書 |
| 初期配置 | `BaseEnv._reset_bodies_*` | [position.md](position.md) |
| 物体の定義 | `objects/` | [object.md](object.md) |
| 語彙・発話・音声認識 | `language/` | [language.md](language.md) |
| 親の行動生成ツール | `ParentTools`（`gym.Wrapper`） | [parent_tools.md](parent_tools.md) |
| 幼児シグナル | `InfantSignalDetector`（`gym.Wrapper`） | [infant_signal.md](infant_signal.md) |
| 訴えの意図フラグ | `intent.py` | [intent.md](intent.md) |
| 遊びの進行（FSM） | `protocols/` | [protocol/README.md](protocol/README.md) |
| 親の意思決定 | `ProtocolManager`（`gym.Wrapper`） | [protocol_manager.md](protocol_manager.md) |

典型的な積み方は `BaseEnv` → `ParentTools` → `InfantSignalDetector` → `ProtocolManager`。
上位 Wrapper は `__getattr__` 経由で `BaseEnv` のメソッドをそのまま呼ぶ（例: プロトコルが `self.env.gaze_object(...)` と書ける）。

---

## 2. 時間

- **1 env.step ≒ 人間の約 0.1 秒**。これが設計上の正の単位で、時間はすべて step 数で数える。
- `action_repeat` は **1 固定**。`1` 以外を渡すとコンストラクタが `ValueError` を投げる。
- 物理 timestep は `1/240` s に設定されるが、**通常の `step()` では物理エンジンを進めない**（[§13](#13-注意事項)）。step 数と物理時間を対応づけてはいけない。
- タイムアウト等の定数は step 数で持ち、実秒はコメントの換算（`steps × 0.1`）として添えるのが本プロジェクトの慣習。`env.action_dt` のような属性は存在しない。
- `step_count` は `step()` の先頭で +1 される。`reset()` で 0 に戻る。
- `sleepiness` は毎 step +1 される。`done` は `sleepiness >= sleepiness_steps`（既定 100000）。`reset()` で `sleepiness` を 0 に戻す。

---

## 3. 空間

### 座標系

- 右手系。**ワールド**の +Z が上。
- **ボディ局所**は前方 = +X。前進・カメラ前方・palm 向きはすべてこのボディ局所 +X 基準で計算する。
- ワールドの +X / +Y に「右」「前」といった意味はない。グリッド配置の「+X が右の列」は配置レイアウト上の呼び方であり、ボディ局所の +X（前方）とは別物。混同しないこと。

### アリーナ

- `arena_half_extents = (xh, yh)`（既定 `(2.5, 2.0)`）の矩形 `[-xh, +xh] × [-yh, +yh]`。
- 床は `plane.urdf`。四方に静的 BOX の壁（`wall_thickness` 既定 0.01、`wall_height` 既定 1.0）を張る。壁は `static_bodies` / `wall_body_ids` に登録され、可視性・透明度を実行時に切り替えられる（`set_wall_visual_alpha` / `set_wall_visual_visible`）。
- エージェントの base は `z = 0.0` 固定（`useFixedBase=True`）。可動小物は `bbox_half_extents[2]`（物体ごとの床接触高さ。立方体 0.06 m など）に配置される。`fixed_placements` の `z` 省略時フォールバックは `DEFAULT_FIXED_PLACEMENT_Z`（0.05）。
- `arena_scale = sqrt((2xh)^2 + (2yh)^2)`。観測 `parent_position` の正規化に使う。
- どこに何を置くか（`placement_method` の 3 方式、`layout_grid`、予約行、`fixed_placements`）は [position.md](position.md)。

上から見た XY 平面（+Z は画面外＝上方向）。原点 `(0, 0)` はアリーナ中心。

```
                        +Y
                         ↑
           X = -xh       │       X = +xh
         (-2.5)          │          (+2.5)
              ┌──────────┼──────────┐  ← 壁 y = +yh + t  (yh=+2.0)
              │          │          │
              │    col=0 │    col=3 │
         -X ←─┤   (左列) │   (右列) ├─→ +X
              │          │          │
              │      (0, 0)         │  ← アリーナ中心
              │          │          │
              └──────────┼──────────┘  ← 壁 y = -yh - t  (yh=-2.0)
                         ↓
                        -Y

  内側の playable 矩形:  x ∈ [-2.5, +2.5] ,  y ∈ [-2.0, +2.0]   (既定値)
  壁は playable 矩形の外側。移動クリップは x,y ∈ [-xh + margin, +xh - margin]（margin 既定 0.35）。
  arena_scale = √(5² + 4²) ≈ 6.40 m（対角線。parent_position の正規化に使用）。
```

グリッド配置（`placement_method="grid"`）では、行 row が Y 方向・列 col が X 方向に対応する（[position.md](position.md)）。**ここでの +X / +Y はワールド座標の列・行の呼び方**であり、ボディ局所の前方 +X とは別物。

```
  既定: arena_rows=3 × arena_cols=4  (layout_grid 未指定時)

  cell_w = 2·xh / cols = 1.25 m     cell_h = 2·yh / rows ≈ 1.33 m
  セル (r, c) の中心:  cx = -xh + cell_w·(c+0.5) ,  cy = -yh + cell_h·(r+0.5)
  インデックス: idx = r · arena_cols + c

        col →   0           1           2           3
              (x≈-1.88)   (x≈-0.63)   (x≈+0.63)   (x≈+1.88)
      ┌───────────┬───────────┬───────────┬───────────┐  y≈+1.33
 r=2  │  idx  8   │  idx  9   │  idx 10   │  idx 11   │  予約行 (+Y 側)
      │           │           │           │           │  固定物専用
      ├───────────┼───────────┼───────────┼───────────┤  y≈ 0
 r=1  │  idx  4   │  idx  5   │  idx  6   │  idx  7   │  中間行
      │  幼児 i   │  obj  x   │  obj  x   │  親  p    │  (既定の配置候補)
      ├───────────┼───────────┼───────────┼───────────┤  y≈-1.33
 r=0  │  idx  0   │  idx  1   │  idx  2   │  idx  3   │  予約行 (-Y 側)
      │           │           │           │           │  固定物専用
      └───────────┴───────────┴───────────┴───────────┘
        ↑ row=0 は -Y 側                              row=2 は +Y 側 ↑
```

### 移動（キネマティック）

- 前進は物理ではなくキネマティックなスライド。`compute_forward_dxdy` がボディ局所 +X の XY 投影に沿って `(dx, dy)` を求め、`_try_kinematic_slide` が base の XY だけを書き換える。Z と姿勢は維持。
- アリーナ内へクリップされる（`_clamp_xy_inside_arena`、余白 `wall_thickness + 1e-3`）。
- **後退はできない**。`forward` 行動が 0 以下なら移動しない。
- 向きを変えるのは `torso_joint`（腰ヨー）。base の姿勢そのものは回さない。

---

## 4. 身体

- URDF は親 `cue_robot_parent.urdf` / 幼児 `cue_robot_infant.urdf`。親子は同寸法で、見た目は `torso_link` の色で区別する。
- `num_agents` は 1（幼児のみ）または 2（親＋幼児）。1 のとき `self.parent` は `None`。
- `_create_agent()` が返す辞書（`self.parent` / `self.infant`）が身体のメタ情報。関節 index、リンク index、把持関連の実測値（`palm_half`, `palm_thickness`, `index_link_size`, `lower_arm_length`）、関節可動域（`*_limits`）、把持状態（`held_obj`, `held_obj_mass`, `held_obj_prev_local`）、把持候補（`graspable_obj`, `graspable_link`）を持つ。

主なリンク：

| キー | 用途 |
|------|------|
| `eyes_link` | カメラ視点、視線の起点 |
| `gaze_link` | `gaze_link_2` を使用。視線判定の補助 |
| `face_links` | 目・頭・唇。視線が顔に当たったときの分類に使う |
| `hold_gaze_links` | 腕〜手のリンク群。把持中に視線が手に当たったら把持物へ振り替える |
| `torso_link` | 前進方向、振る検出のローカル座標基準 |
| `palm_link` / `hand_link` | 把持判定 |
| `index_finger_tip_link` | （旧指先接触判定用。現行は未使用） |

- 関節角は `set_joint_angle` で **直接書き込む**（`resetJointState`）。モータ制御ではない。
- `torso_joint` と `roll_wrist_joint` は ±π でラップアラウンドする。それ以外の関節は URDF の可動域にクリップされる。

---

## 5. 行動空間

`easy_voice=True`（唯一の実装）のとき、1 エージェントの行動は連続値 `[-1, 1]`。次元は `enable_emotion` で変わる。

**既定（`enable_emotion=False`）: 15 次元。** `lip` と発声 `pitch` を含まない。

| index | 名前 | 内容 |
|-------|------|------|
| 0 | `torso_joint` | 腰ヨー（向き変え） |
| 1 | `head_pan_joint` | 首ヨー |
| 2 | `head_tilt_joint` | 首ピッチ |
| 3 | `shoulder_yaw_joint` | 肩ヨー |
| 4 | `shoulder_pitch_joint` | 肩ピッチ |
| 5 | `elbow_joint` | 肘 |
| 6 | `roll_wrist_joint` | 手首ロール |
| 7 | `pitch_wrist_joint` | 手首ピッチ |
| 8 | `index_joint` | 人差し指 |
| 9 | `other_fingers` | 親指・小指（`thumb_joint` と `little_finger_joint` を逆符号で同時に動かす） |
| 10 | `forward` | 前進（>0 のみ有効） |
| 11 | `volume` | 発声の音量 |
| 12–14 | `v_1`, `v_2`, `v_3` | 音素の識別軸。調音的な意味はない |

**`enable_emotion=True`: 17 次元。** 上記に加え、`other_fingers` の直後に `lip`（唇・左右対称に開閉）、`volume` の直後に `pitch`（発声のピッチ・韻律。音素軸には含めない）が入る。以降の index は 2 つずれる。

- **各行動値は「1 step あたりの変位」の係数**であり、目標角度ではない。実際の変位は `|action| × step_size`。`step_size` は幼児 `infant_joint_mappings` / 親 `parent_joint_mappings` に別々に持つ（幼児のほうが小さい＝発達的に遅い動きの初期値）。
- 腕は片腕（左腕）のみ。
- `action_space = Tuple([幼児, 親])`。`step()` には常に長さ 2 のリスト／タプルを渡す。長さが違えば `ValueError`。
- 幼児は `infant_action_names` で制御次元を制限できる。制限時、幼児側に渡すベクトルはそのリストの順序・次元になり、`step()` 内でフル次元へ展開される。親は常にフル次元。
- `non_linear_action=True` のとき、幼児の行動ベクトルだけ 3 乗される（小さい入力をより小さく）。

### 発声の扱い（親と幼児で非対称）

- **幼児**: `infant_voice[k] += action × 0.5` と**積算**される。音素軸は `[-1, 1]` にクリップ。`volume` だけ `[0, 1]` にクリップ（0 未満にはならない）。`enable_emotion=True` のときだけ `pitch` も同様に積算する。同一エピソード内では 0 行動を入れても値は戻らない。`reset()` で `infant_voice` と `infant_voice_window` を 0 / 空に戻す。
- **親**: `parent_voice[k] = clip(action, -1, 1)` と**毎 step 上書き**される（`volume` だけ `[0, 1]`）。`reset()` では明示的にクリアしない（次の親 `step()` まで前エピソード末の値が残りうる）。
- どちらも更新後の値が `infant_voice_window` / `parent_voice_window`（`maxlen = voice_window_size`、既定 3）に積まれる。幼児シグナル側はこの窓を読む。
- 音素座標を行動に載せるヘルパーが `apply_phoneme_to_action(action, coords, volume=...)`。`v_1..v_3` と `volume` を書き込む。音素座標の定義は [language.md](language.md)。
- **有声閾値**: `voice_volume_threshold`（既定 `DEFAULT_VOICE_VOLUME_THRESHOLD` = 0.25）。`volume >= threshold` で有声。全層（観測・認識・intent・プロトコル）で同じ値を使う。`BaseEnv` コンストラクタで上書き可能。

---

## 6. 観測空間

観測は `(幼児観測 Dict, None)` のタプル。親の観測は現状実装しない予定。

| キー | 形 | 内容 |
|------|-----|------|
| `image` | `(view_height, view_width, 3)` uint8 | 幼児視点 RGB |
| `infant_voice` | `(4,)` / `(5,)` float32 | 幼児の発声ベクトル（下記） |
| `parent_voice` | `(4,)` / `(5,)` float32 | 親の発声ベクトル（下記） |
| `parent_position` | `(2,)` float32 | 幼児から見た親の相対位置 `[rel_x, rel_y]`（`arena_scale` で正規化） |
| `object_sound` | `(2,)` float32 | 振る → 音チャネル |
| `object_effect` | `(2,)` float32 | 振る → 効果チャネル |
| `sleepiness` | `(1,)` float32 | 眠気の進行 `[0, 1]`（`sleepiness / sleepiness_steps`） |
| `joint_angles` | `(12,)` / `(13,)` float32 | 幼児の関節自己受容（full。下記） |
| `tactile` | `(2,)` float32 | 手が把持範囲にある物体の触覚ベクトル |
| `holding` | `(1,)` float32 | 把持中なら `1.0`、未把持なら `0.0` |
| `fatigue` | `(9,)` / `(10,)` float32 | 部位疲労の正規化値 `fatigue / H`（[§6.1](#61-fatigue)） |
| `fatigue_lock` | `(9,)` / `(10,)` float32 | 部位ロック中なら `1.0`、それ以外 `0.0` |
| `is_first` | 0/1 | `reset()` の返り値のみ `True` |
| `is_terminal` | 0/1 | `reset()` では `False`、`step()` では `done` |

発声ベクトルの並びは `enable_emotion` で変わる。

- **既定（`False`）**: `voice_mapping` = `volume, v_1, v_2, v_3`（`voice_dim = 4`、`phoneme_index = 1`）
- **`True`**: `voice_mapping` = `volume, pitch, v_1, v_2, v_3`（`voice_dim = 5`、`phoneme_index = 2`）

**`joint_angles` の並び**

`action_map` の関節スロット順。`forward` と音声は含めない。`torso_joint` と `roll_wrist_joint` だけ `(sin θ, cos θ)`（θ は `get_joint_angle` のラジアン）。それ以外は URDF 可動域で `[-1, 1]` に正規化。`other_fingers` は `little_finger_joint`。`enable_emotion=True` のときだけ末尾に `lip`（`left_lip_joint`）が付く。

- **既定（`False`、12 次元）**: `[sin torso, cos torso, head_pan, head_tilt, shoulder_yaw, shoulder_pitch, elbow, sin roll_wrist, cos roll_wrist, pitch_wrist, index, other_fingers]`
- **`True`（13 次元）**: 上記の末尾に `lip`

`infant_action_names` を指定したとき、実装はそのリスト順のうち上記関節スロットだけを観測に載せる（デバッグ用。契約の正本は full）。

**ゼロ化の条件**（ここが観測の主要な契約）:

- `infant_voice`: `volume < voice_volume_threshold`（既定 0.25）なら全要素 0。有声は `volume >= threshold`（ちょうど 0.25 も有声）。観測の `volume` 次元は `[0, 1]`（`pitch` / 音素は `[-1, 1]`）。
- `parent_voice`: `volume < voice_volume_threshold` なら全要素 0。有声は `volume >= threshold`。
- `parent_position`: **親が有声のときだけ**中身が入る。`parent_voice` がゼロ化される条件では `[0, 0]`。親の位置を常時観測できるわけではない。
- `object_sound` / `object_effect`: その env step 中に振る判定が立たなければ `[0, 0]`（[§9](#9-振る検出)）。
- `sleepiness`: `0` は眠くない（`reset` 直後）。`1` は `sleepiness >= sleepiness_steps` の終了直前。
- `tactile`: 手が把持範囲に `is_graspable=True` な物体がなければ `[0, 0]`。非ゼロは接触（`graspable_obj` の `tactile_vec`）。
- `holding`: 未把持なら `0`。`touching` キーは作らない（触覚の非ゼロが接触を表す）。
- `joint_angles`: クリップ関節の `0` は可動域の中点（rest 姿勢とは限らない）。ラップ関節の rest は `(0, 1)`。
- `fatigue` / `fatigue_lock`: `enable_fatigue=False` のときは常に `0`。キーは常に存在する。

### 6.1 fatigue

幼児のみ。`enable_fatigue`（既定 `False`）で有効化。セットポイントは関節角・`infant_voice[volume]` の **0**（URDF rest。後で URDF を改造する可能性あり）。

**対象チャネル（観測・内部状態の並び）**

既定（`enable_emotion=False`）: `head_pan_joint, head_tilt_joint, shoulder_yaw_joint, shoulder_pitch_joint, elbow_joint, pitch_wrist_joint, index_joint, other_fingers, volume`（9 チャネル。`lip` 無し）。

`enable_emotion=True`: 上記の `other_fingers` と `volume` のあいだに `lip` が入る（10 チャネル）。

**対象外**: `torso_joint`, `roll_wrist_joint`（ラップ）、`forward`, `pitch`, `v_1..v_3`。

**ずれ（正規化）**: 関節は `|q| / max(|lo|, |hi|)`。`other_fingers` は thumb / little の正規化ずれの max。`lip`（感情オン時のみ）は `|left_lip|` の正規化。`volume` は `|infant_voice[volume]|`。

**蓄積ゲイン**: 肩 yaw/pitch・肘・volume は `1.0`。頭・`pitch_wrist`・指・唇は `0.5`。
**状態機械**（チャネルごとに独立）:

- 通常: 毎 step `fatigue += gain × ずれ`、同時に `fatigue -= 0.3`（回復）。`fatigue >= H`（既定 30）でロック。
- ロック中: 蓄積しない。回復のみ。action によるそのチャネルの操作は無効。`step_size` で 0 へ引き戻す。`fatigue <= L`（既定 10）で解除。

親には fatigue を付けない。選好は観測に出さない（[infant_signal.md](infant_signal.md)）。

### カメラ

- 視点は `eyes_link` のワールド位置から前方（ボディ局所 +X）へ `eyes_radius` だけ進めた点。注視点は視点 + 前方ベクトル。up は ワールド +Z 固定（ロールは考慮しない）。
- `view_fov`（既定 90°）、`view_near` / `view_far`（既定 0.01 / 5.0）。
- `ssaa > 1` のとき内部を `ssaa` 倍の解像度で描画し、等分割ブロックの面積平均で縮小する（`ssaa` は整数倍のみ）。
- レンダラは EGL が使えれば `ER_BULLET_HARDWARE_OPENGL`、無ければ TinyRenderer。
- `return_views=True` のとき `info["views"]["infant_rgb"]` に画像を入れる。`output_parent_view=True` を併用すると `parent_rgb` も入る。**観測用の画像とは別に再レンダリングする**ので、その分だけ遅くなる。

---

## 7. 接触・把持・受け渡し

### 手の形

`index_joint` と `thumb_joint` の角度だけで判定する（`_get_hand_shape`）。

| 形 | 条件 |
|----|------|
| `open` | `index < 1.57` かつ `thumb > -1.57` |
| `closed` | `index >= 1.57` かつ `thumb <= -1.57` |
| `pointing` | `index < 1.57` かつ `thumb < -1.57` |
| `other` | 上記以外 |

境界は `HAND_SHAPE_BOUNDARY`（1.57）。`_get_hand_shape` は `(hand_state, grip_power, intensity)` を返す。第3戻り `intensity` はその形の理想姿勢を 1、その形と判定される一番端を 0 とする [0, 1]（`classify_hand_shape`）。`grip_power` は `closed` のとき `thumb_angle`、それ以外 0。

### 把持候補（touching の正本）

毎 step、`_apply_agent_action` 内で `detect_grasp_object` を 1 回実行し、結果を agent 辞書に書く。

| キー | 内容 |
|------|------|
| `graspable_obj` | 手が把持範囲にある最も近い `is_graspable=True` な物体の body id。なければ `None` |
| `graspable_link` | 上記のリンク id |

手形は見ない。`InfantSignalDetector` の `touching` シグナルは `graspable_obj is not None` から導出する（[infant_signal.md §4](infant_signal.md#4-ジェスチャシグナル)）。実際の掴みは [§7 把持](#把持) の open→closed 遷移で `held_obj` に昇格する。

### 把持

- 把持が成立するのは **手の形が `open` → `closed` に遷移した瞬間**だけ。閉じ続けている間に新しく掴むことはない。
- 対象の探索は `detect_grasp_object`: `is_graspable=True` かつ `active_objects` のみ。 palm 中心の周囲 `grasp_distance` の AABB で候補を絞り、`hand_link` との最近接距離が `grasp_distance` 以内、かつ palm 前方ベクトルと接触方向のなす角が `grasp_dir_threshold`（既定 30°）以内、**または** 距離が `grasp_contact_distance` 未満のものから最も近いものを選ぶ。
- 把持すると対象の **mass が 0 になり**、毎サブステップ `palm 中心 + palm 前方 × grasp_object_offset`（既定 0.075 m）へスナップされる。姿勢は元のまま。物理的な拘束ではない。
- 手の形が `closed` でなくなった瞬間に離す。mass を元に戻し、離した位置で
  - 重なっている `is_platform` な物体があれば、その `objects_on_board` に登録し、物体側に `put_on` を立てる（家事の「board に載せる」はこの経路）
  - `can_contain` かつ開いている容器があれば contain 候補として `step()` に返す（現行カタログでは発生しない）

### 受け渡し

`_update_handover` は、同じ step 内で**親と幼児の両方**が transfer フラグを立てたときだけ呼ばれる。現行の実装では親→幼児しか成立しない（[§13](#13-注意事項)）。`ObjectGivingProtocol` は受け渡しを使わず、幼児の到達圏に**置く**（`place_object_at`）ことで渡す。

---

## 8. 到達圏と配置距離

距離は `BaseEnv` が正本で、初回 `reset()` のロード直後に `_compute_interaction_distances()` が実測から確定させる。コンストラクタで明示した値はそのまま使われる（自動計算しない）。

| 属性 | 自動計算 | 用途 |
|------|----------|------|
| `grasp_contact_distance` | `palm_half + max_graspable_object_half` | 角度判定を免除する距離（margin なしの接触距離）。`detect_grasp_object` が palm 方向角の代わりに距離だけで候補に入れる閾値。 |
| `grasp_distance` | `grasp_contact_distance + grasp_margin`（既定 margin 0.12） | 把持検出の半径。[§7 把持](#7-接触把持受け渡し) の `detect_grasp_object` が palm 中心の AABB 探索・`hand_link` との最近接判定に使う。手が open→closed に遷移した瞬間、この距離以内の物体だけが掴み候補になる。 |
| `measured_floor_reach_xy` | 床置き掴み姿勢（`FLOOR_REACH_JOINTS`）に一度だけ関節を入れて FK し、base→palm の XY を実測した値 | `reach_radius` / `grasp_approach_dist` 自動計算の基準。URDF ロード後の腕長実測で、以降の到達・配置距離を派生させる。デバッグ出力（`measured_floor_xy`）にも出る。 |
| `reach_radius` | `measured_floor_reach_xy × reach_factor`（既定 0.85） | エージェント base からの**床 XY 到達圏**（高さは見ない）。`object_in_agent_reach` の閾値。「幼児が自分で取れるか」の判定、`ProtocolManager` の giving ゲート（届く物では始めない）、`ObjectGivingProtocol` の置き結果チェック（`reach_radius + 0.03` 以内）に使う。 |
| `place_dist` | `max(0.05, reach_radius - place_margin)`（既定 margin 0.10） | 物を**置く** XY 距離。`ParentTools.reach_place_position` が、相手 base から自分方向へ `place_dist` 進んだ点を狙い点にする。`ObjectGivingProtocol` の `place_object_at` がこれ。掲げ（共同注意の見せる合図と、渡しの `showing`）は `compute_show_position` の既定で、目から `0.1 + 2 × 半幅`。`place_dist` は使わない。 |
| `grasp_approach_dist` | `min(reach_radius + grasp_approach_delta, measured_floor_reach_xy - 0.02)`（既定 delta 0.05） | 物体を**掴む**ために止まる床 XY 距離。`ParentTools.approach_object` の `approach_dist`。物体 base からこの距離まで近づいたら前進を止める。JA の grasp フェーズでも使う。 |
| `stand_dist` | `place_dist + grasp_approach_dist` | 物を**渡す／見せる**ために幼児から止まる床 XY 距離。`ParentTools.approach_infant` の `approach_dist`。`ObjectGivingProtocol` の接近、`JointAttentionProtocol` の `shake_stand_dist` 既定がこれ。立ち位置から腕を `grasp_approach_dist` 伸ばせば、相手の到達圏内（`place_dist`）に届く想定。 |

自動計算に使うチューニング定数（コンストラクタ引数）:

| 定数 | 既定 | 用途 |
|------|------|------|
| `grasp_margin` | 0.12 | `grasp_distance` の余裕。`grasp_contact_distance` に加え、把持検出を少し甘くする。 |
| `reach_factor` | 0.85 | FK 実測腕長に掛けて `reach_radius` にする係数。実測より保守的（短め）に到達圏を設定する。 |
| `place_margin` | 0.10 | `reach_radius` から引いて `place_dist` にする余裕。置いた物が相手の到達圏内に確実に入るようにする。 |
| `grasp_approach_delta` | 0.05 | `reach_radius` に足して `grasp_approach_dist` の目標値にする。到達圏の外側で止まりつつ腕を伸ばして掴む距離。 |

- `grasp_approach_dist` に上限（`measured_floor_reach_xy - 0.02`）があるのは、到達圏の外に立ちすぎると place も grasp もできなくなるため。
- `object_in_agent_reach(agent, oid)` は **床 XY 距離のみ**で `reach_radius` 以内かを返す。高さは見ない。「幼児が自分で取れるか」の判定はこれを使う。
- `margin`（コンストラクタ引数）はアリーナ配置の余白であり、`grasp_margin` とは別物。名前が似ているので注意。
- 確定値は初回 reset 時に `[interaction_distances] ...` の 1 行で標準出力に出る。

---

## 9. 振る検出

- 対象は `can_shake` が真な物体を**把持している間**だけ。
- 判定は `_update_kinematic_held_objects` の中で行う。把持物のワールド座標を**胴体ローカル座標**に変換し、前サブステップからの変位が `shake_move_threshold`（既定 0.002 m）を超えたら振ったとみなす。胴体ローカルで測るので、歩いて移動しただけでは発火しない。
- 発火すると `shake_effect(oid)` が物体の `shake_sound` / `shake_effect_vec` を観測チャネル `object_sound` / `object_effect` に書き込む。
- チャネルは **毎 env step の先頭でゼロクリア**される。同じ step 内で複数回発火したら最後の値が残る。
- ベクトルの値（形 → 音、色 → 効果）は [object.md](object.md)。
- 把持を開始／解除した時点で前サブステップ座標（`held_obj_prev_local`）はリセットされる。掴んだ直後の 1 サブステップは発火しない。

---

## 10. 簡易物理

`_apply_simple_physics()` が毎 step 1 回、落下だけを手で処理する。

- 落下量は **`fall_per_step` [m/step]**（コンストラクタ引数、既定 **0.01**）。物理 timestep からは導出しない。デモで 0.05 / 0.10 / 0.22 を試す場合は `demo_object_giving.py --fall-per-step` または `BaseEnv(fall_per_step=...)`。
- 床に相当する高さは物体の `bbox_half_extents[2]`。配置 z も同じ値（`_object_rest_z`）。
- 対象外: 把持中の物体、`move_type is None`（固定物）、`put_on` が立っている物体（board に載っている）、`contained_in` が立っている物体。
- 転がり・跳ね返り・傾きは起きない。姿勢は変化しない。

---

## 11. Gym API と `step()` の処理順

### `reset(seed, options)`

1. 接続確認（切れていれば再接続し、未ロード扱いにする）
2. **初回のみ**: `resetSimulation` → 物理パラメータ設定 → `_load_urdfs()`（床・壁・エージェント・物体プール・固定物）→ `_compute_interaction_distances()` → `trackable_targets` を確定
3. `_reset_bodies()`（配置方式ごとの分岐は [position.md](position.md)）
4. `step_count` と `sleepiness` を 0 に、fatigue を 0 に、`infant_voice` / `infant_voice_window` をリセット、振るチャネルと振る検出履歴をリセット
5. 観測を作り、`is_first=True` を立てて返す

**物体プールとエージェントの生成は初回 reset の 1 回だけ**。以降の reset は位置と状態を戻すだけで、body id は変わらない。

### `step(action)`

1. `step_count += 1`、`sleepiness` を +1
2. `trackable_targets` は初回ロード時に確定済み（毎 step 更新しない）
3. 入力の検証（長さ 2）と、幼児行動のフル次元への展開
4. `object_sound` / `object_effect` をゼロクリア
5. 幼児 → 親の順に `_apply_agent_action()`
   - 関節・前進・発声の適用
   - `_get_hand_shape()` → `detect_grasp_object()` → agent に `graspable_obj` / `graspable_link` を書く → `_update_grasp()`（この中で把持物のスナップと振る検出）
6. 両者が transfer フラグを立てていれば `_update_handover()`、そうでなければ contain 判定
7. `_apply_simple_physics()`
8. `done = sleepiness >= sleepiness_steps`
9. `info` の組み立て（`debug_info` / `return_views` に応じて）
10. 観測の生成（この時点の状態が観測になる）

戻り値は `(obs, reward, done, truncated, info)`。`BaseEnv` 単体の `reward` は **常に `[0.0, 0.0]`**。選好報酬は `InfantSignalDetector` が `reward[0]` に足す（[infant_signal.md](infant_signal.md)）。`truncated` は常に `False`。

### その他

- `get_info(action)` は `debug_info=True` のときだけ呼ばれ、`{"step_count": ...}` のみを返す。
- `get_all_trackable_targets()` は「全物体の body id ＋ 親の body id」を返す。視線・ジェスチャの対象分布はこの並び順のベクトルとして表現される。順序は `all_objects` の挿入順（＝ロード順）で、エピソードをまたいで変わらない。
- `render()` は GUI 時の可視化用。ヘッドレスでは案内を出すだけ。
- `close()` は EGL プラグインをアンロードして切断する。

---

## 12. 上位モジュール向けのフック

`BaseEnv` が「事実」として提供し、Wrapper 側が読むだけのもの。

### 家事タスク候補

| 属性 | 中身 | 更新タイミング |
|------|------|----------------|
| `household_boards_by_category` | `色 → [board の body id]` | 初回ロード時（`_index_household_boards`） |
| `household_items_by_category` | `色 → [item の body id]` | 毎 reset（アクティブ選択・配置の後） |
| `household_task_candidates` | `[{"task_type", "board_id", "item_ids"}]` | 毎 reset |

`household_task_candidates` は「いま成立する組み合わせ」だけを列挙したもの。次のいずれかに当たる item は候補から外れる（`_is_eligible_household_item`）。

- アクティブでない
- `is_platform`（board 自身）
- すでに `put_on` が立っている
- どこかの board の `objects_on_board` に入っている
- 親または幼児が把持している

対応する board が無い色、または残り item が無い色は、そもそもタスクとして列挙されない。**候補が空**とは、このリストが空であることを指す。プロトコル側はこのリストを読むだけで、物体を自分で走査しない。タスク種別（`TASK_SCHEMAS`）と選び方は [protocol/household_solo.md](protocol/household_solo.md)。

### 到達圏の問い合わせ

`object_in_agent_reach(agent, oid)`（[§8](#8-到達圏と配置距離)）。`ProtocolManager` の「届く物では『くれ』を始めない」ゲートと、`ObjectGivingProtocol` の置き場所計算が使う。

### 発声

`apply_phoneme_to_action(action, phoneme_coords, volume=0.5)`（[§5](#5-行動空間)）。プロトコルは音素キューを 1 フレームずつこれで行動ベクトルへ載せる。

---

## 13. 注意事項

**物理エンジンは進んでいない。**
`p.stepSimulation()` は初回ロード時の距離計測（`_measure_floor_reach_xy` 内の FK 更新）にだけ呼ばれ、`step()` からは呼ばれない。重力と timestep は設定されているが効いていない。エージェントの関節・base、把持物の位置はすべて `resetJointState` / `resetBasePositionAndOrientation` による直接書き込みで、落下だけが [§10](#10-簡易物理) の手処理で起きる。

**移動の衝突判定は無効。**
`_try_kinematic_slide` はアリーナ内へのクリップだけを行い、貫通チェックのコードはコメントアウトされている。エージェント同士や物体をすり抜ける。

**幼児→親の受け渡しは成立しない。**
`_update_handover` は「幼児が持っていれば親へ移す」→「親が持っていれば幼児へ移す」を続けて実行するため、幼児が持っていた物は親を経由して幼児に戻る。実質的に親→幼児の一方向しか動かない。現行カタログでは `ObjectGivingProtocol` が place 方式を使うのでこの経路には乗らない。

**幼児の発声は step 内で積算される。**
`infant_voice` は毎 step の行動を `× 0.5` して足し込む。同一エピソード内では 0 行動を入れても値は戻らない（無音にするには負の `volume` 行動で下げる）。`reset()` では `infant_voice` と `infant_voice_window` を 0 / 空に戻す（[§5](#5-行動空間)）。

**`parent_position` は親が喋っているときしか入らない。**
沈黙中は `[0, 0]`。原点にいるという意味ではない。

**観測空間の宣言と実際の返り値がずれている。**
`observation_space` の 2 要素目は長さ 0 の `Box` だが、`reset()` / `step()` が返すタプルの 2 要素目は `None`。

**`BaseEnv` の報酬は常に 0。**
選好報酬は `InfantSignalDetector` が `reward[0]` に足す。プロトコルの `reward_function` は `info["protocol_reward"]` にだけ載せ、学習信号には使わない（[protocol_manager.md](protocol_manager.md)）。

**`is_food` は現行カタログでは未使用。**
物体属性として残るが、判定コードパスは動かない。エピソード終了は `sleepiness` のみ。

**contain / open のコードパスは残っているが無効。**
`contain_effect` / `open_effect` / `close_effect` と、把持解除時の容器判定は残っている。ただし現行カタログの物体はすべて `is_containable=False` / `can_contain=False` / `is_openable=False` なので発火しない。

**`_apply_deadzone` と `foward_by_motor` は未使用。**
前者はどこからも呼ばれない。後者は `True` にすると `NotImplementedError`。

**`easy_voice=False` は未実装。**
`False` を渡すと `NotImplementedError`。行動・音声の次元は `easy_voice=True` 前提の値（さらに `enable_emotion` で変わる）。

**モードフラグ**

| フラグ | 効果 |
|--------|------|
| `gui` | PyBullet GUI を開く。`False` はヘッドレス（EGL / TinyRenderer） |
| `reverse_mode` | 親と幼児の URDF を入れ替える（役割の逆転実験用） |
| `parent_invisible_mode` | 親を画面外に置き、配置処理から外す。幼児単独の観測を取るときに使う |
| `eval_mode` | `BaseEnv` 自身は保持するだけ。`ProtocolManager` 経由で `JointAttentionProtocol` の評価用挙動に渡る |
| `enable_emotion` | 既定 `False`。オフのとき `lip` / 発声 `pitch` を行動・観測・fatigue から次元ごと除外し、表情・声の感情シグナルを出さない。共同注意の `intent_type="sharing"` は `ValueError`。オンにすると 17 / 5 / 13 / 10 次元に戻る |
| `debug_info` | `info` に `get_info()` の内容を載せる |
| `return_views` | `info["views"]` に RGB を載せる（追加のレンダリングコスト） |

---

## 14. 設計上の判断メモ

コードからは読み取れないが、変更すると研究の前提が崩れるもの。

- **テーブルを置かない**: 親と幼児がテーブルを挟んで物を操作する構成にすると、指差しと物を掴む動作が幾何的に区別しにくくなる。幼児の可動域を制限すれば同等の条件は作れる、と判断している。
- **キネマティックな移動**: 歩行制御の難しさは本研究の対象ではない。インタラクションそのものに焦点を当てるため、前進をコマンド化している。
- **把持を簡単にしてある**: 形状が複雑だと把持の失敗が増え、それは扱いたい問題ではない。物体をプリミティブに限り、`grasp_margin` を広めに取っている（4 形状で把持成功率に偏りがないことは確認済み）。
- **片腕のみ**: 両手動作は扱わない。
