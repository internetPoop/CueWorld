# 初期配置アルゴリズム（`_reset_bodies_*`）

`BaseEnv` の初期配置は、コンストラクタ引数 `placement_method` により 3 方式から選択される。エピソードごとに `reset()` から `_reset_bodies()` が呼ばれ、エージェントとオブジェクトの位置が決定される。

本ドキュメントは**「どこに置くか」の正本**である。座標系・アリーナ寸法・壁・`reset()` 全体の処理順は [env.md](env.md) を参照（ここでは再掲しない）。物体そのものの寸法と属性は [object.md](object.md)。

| `placement_method` | 概要 |
|---|---|
| `"grid"`（デフォルト） | アリーナを行×列のセルに分割し、セル中心付近に配置 |
| `"random"` | アリーナ内のランダム座標に配置（`layout_grid` は無視） |
| `"geometric"` | 幼児・オブジェクト・親の相対角度・距離制約に基づく配置 |

---

## 全体フロー

```
__init__
  ├─ placement_method の検証
  └─ grid の場合: _setup_layout_grid → _initialize_cells → _setup_fixed_placements

reset()  ← 初回のみ _load_urdfs()（env.md §11）
  └─ _reset_bodies()
       ├─ grid      → _reset_bodies_grid()
       ├─ random    → _reset_bodies_random()
       └─ geometric → _reset_bodies_geometric()
```

物体プールの生成（`_load_urdfs`）は初回 `reset()` の 1 回だけで、body id はエピソードをまたいで変わらない。以降の reset がやるのは位置と状態のリセットだけ。

### アクティブ選択（`_select_active_objects`、grid / random）

1. 初回 reset 時に `move_type is None` のオブジェクトを `_fixed_active_objects` として固定登録
2. 移動可能オブジェクト（`move_type is not None`）から `num_objects - len(固定)` 個をランダムサンプリング（非復元）
3. `active_objects = 固定 + サンプリング分`

`move_type` はオブジェクトクラス側の属性（`BaseObject` の既定は `None`、可動プリミティブは `1`）。固定オブジェクトはロード時に `move_type = None` が上書きされ、`initial_pos` / `initial_orn` に記録した座標へ毎エピソード戻る。

非アクティブなオブジェクトは画面外へ退避され、把持・接触・視線の対象にならない。

配置が確定したあと `_finalize_active_after_placement()` が、実際に床へ置けた id だけを `active_objects` に残し、家事タスク候補（[env.md §12](env.md#12-上位モジュール向けのフック)）を組み直す。

---

## セル座標（grid のみ）

グリッドは配置専用の区切りで、行（row）が Y 方向、列（col）が X 方向に対応する。ここでの「+X が右の列」はレイアウト上の呼び方であり、ボディ局所の前方 +X とは別物（[env.md §3](env.md#3-空間)）。

```
cell_w = (2 * xh) / arena_cols
cell_h = (2 * yh) / arena_rows

セル (r, c) の中心:
  cx = -xh + cell_w * (c + 0.5)
  cy = -yh + cell_h * (r + 0.5)

インデックス: idx = r * arena_cols + c
```

### 予約行（row=0 / row=max）

**row=0** と **row=max**（`arena_rows - 1`）は固定オブジェクト専用に予約される。

| 影響 | 内容 |
|---|---|
| 可動オブジェクトの配置候補 | 中間行（`1 .. arena_rows-2`）のみ。`layout_grid` で `'x'` を指定しても予約行は除外 |
| 固定オブジェクトの Y 制約 | `fixed_placements` の `y` は row=0 または row=max の帯域内である必要がある |

帯域の計算:

```
row r の Y 帯域: [ -yh + cell_h * r ,  -yh + cell_h * (r + 1) ]
```

---

## 共通パラメータ

| パラメータ | デフォルト | 用途 |
|---|---|---|
| `margin` | `0.35` | 壁内側との最小余白 [m] |
| `min_dist` | `0.35` | オブジェクト同士・親幼児間の最小距離 [m] |
| `num_agents` | `2` | `1` のとき親は生成・配置しない |
| `num_objects` | `3` | エピソードに出現するオブジェクト数 |
| `random_parent_pos` | `False` | grid 時、親セルを候補からランダム選択 |
| `parent_invisible_mode` | `False` | 親を `(0, 0, -100)` に配置（画面外） |
| `fixed_placements` | `None` | 固定オブジェクトのワールド座標リスト（grid のみ） |
| `infant_obj_angle_range` | `None` → geometric 内で `(-30°, +30°)` | 幼児→オブジェクトの方位角範囲 [rad]（kwargs） |

> 注: クラスコメントに `first_distance` / `band_width` の記述があるが、現行コンストラクタ引数・配置ロジックでは未使用。

---

## 固定オブジェクト（`fixed_placements`）

`placement_method="grid"` のときのみ使用可能。`layout_grid` とは独立したリストで、ワールド座標を直接指定する。

### 形式

各要素は辞書:

```python
{"class": "Board", "x": 0.0, "y": 1.95, "z": 0.9, "yaw": 0.0}
```

| キー | 必須 | 説明 |
|---|---|---|
| `class`（または `cls`） | ✓ | `cueworld.objects.items` のクラス名（例: `"Board"`, `"RedBoard"`）。`items_legacy.py` のクラス名を書いてもロードに失敗して警告になる |
| `x` | ✓ | ワールド X [m] |
| `y` | ✓ | ワールド Y [m]。row=0 または row=max の帯域内 |
| `z` | — | 高さ [m]（省略時は `DEFAULT_FIXED_PLACEMENT_Z` = 0.05） |
| `yaw` | — | ヨー角 [度]（デフォルト `0`） |

### 制約

- `y` は `_is_fixed_placement_y_allowed()` により、row=0 または row=max の帯域内のみ許可
- `placement_method` が `"grid"` 以外のとき `fixed_placements` を渡すと `ValueError`
- ロード時に `move_type = None` が設定され、毎エピソード `initial_pos` / `initial_orn` へ再配置

### Board を壁代わりに使う例

`board.urdf` は Y 方向に 1.0 m 展開（薄い X 厚）。X 方向に並べて壁にする場合は `yaw=90` にする。

```python
arena_half_extents = (2.5, 2.0)
arena_rows = 3
yh = 2.0
cell_h = (2 * yh) / arena_rows

fixed_placements = [
    # 北側（row=max 帯域）に壁沿い Board
    {"class": "Board", "x": -2.0, "y": yh - 0.05, "z": 0.9, "yaw": 90.0},
    {"class": "Board", "x": -1.0, "y": yh - 0.05, "z": 0.9, "yaw": 90.0},
    {"class": "Board", "x":  0.0, "y": yh - 0.05, "z": 0.9, "yaw": 90.0},
    {"class": "Board", "x":  1.0, "y": yh - 0.05, "z": 0.9, "yaw": 90.0},
    {"class": "Board", "x":  2.0, "y": yh - 0.05, "z": 0.9, "yaw": 90.0},
    # 南側（row=0 帯域）も同様
    {"class": "Board", "x": -2.0, "y": -yh + 0.05, "z": 0.9, "yaw": 90.0},
    # ...
]

layout_grid = [
    ["_", "_", "_", "_"],   # row=0: 固定専用
    ["i", "x", "x", "p"],   # 中間行: エージェント・可動オブジェクト
    ["_", "_", "_", "_"],   # row=max: 固定専用
]
```

デモ実装: `cueworld/demo/demo_initialize_pos.py`

---

## 1. グリッド配置（`placement_method="grid"`）

### 1.1 レイアウト定義

#### `layout_grid` 未指定（従来デフォルト）

`arena_rows`（デフォルト 3）× `arena_cols`（デフォルト 4）のグリッドを使用。

| 役割 | 候補セル |
|---|---|
| 幼児 | 中央行・最左列 `(mid_row, 0)` |
| 親 | 中央行・最右列 `(mid_row, arena_cols-1)` |
| オブジェクト | 中間行 × 列 `1 .. arena_cols-2`（row=0/max と左右端列を除く） |

`random_parent_pos=True` のとき、親候補は**全行の最右列**に拡大。

#### `layout_grid` 指定

二次元リストでセルごとに役割を定義（小文字に正規化）。**固定オブジェクトはここでは指定しない**（`fixed_placements` を使用）。

| セル文字 | 意味 |
|---|---|
| `'i'` | 幼児配置候補 |
| `'p'` | 親配置候補 |
| `'x'` | オブジェクト配置候補（row=0/max は自動除外） |
| `'_'` / `''` | 使用しない |
| 複合（例 `'xp'`） | 各文字の候補に追加 |

必須条件:

- 幼児候補 `'i'` が 1 つ以上
- `num_agents == 2` のとき親候補 `'p'` が 1 つ以上

### 1.2 エピソードごとの配置手順（`_reset_bodies_grid`）

1. **幼児**: `default_infant_cell_idx`（候補リスト先頭）のセル中心 + 微小ランダムオフセット（±10% セル幅/高さ）へ配置。ヨー角 `0`（+X 方向）。
2. **親**（`num_agents == 2`）:
   - `parent_invisible_mode`: `(0, 0, -100)`
   - それ以外: `random_parent_pos` なら未使用セルから候補をランダム選択、否则デフォルトセル。幼児の位置を向くようヨー角を `atan2(dy, dx)` で設定。
3. **関節リセット**: 全関節 `0`、把持状態クリア。
4. **オブジェクト**:
   - `_select_active_objects()` でアクティブ集合を決定
   - 非アクティブは画面外へ
   - 固定オブジェクト（`move_type is None` かつ `initial_pos` あり）: 記録済みワールド座標へ
   - 移動可能オブジェクト: 未使用のオブジェクト候補セルをシャッフルし、順にセル中心 + 微小オフセットへ配置

**使用済みセル管理**: 幼児・親が使ったセルは可動オブジェクト配置から除外。

**2 回目以降の reset**: 固定オブジェクトは `_first_reset_done_grid_placed` フラグにより再移動をスキップ（速度リセットのみ）。

---

## 2. ランダム配置（`placement_method="random"`）

`layout_grid` / `fixed_placements` / セル分割は使わない。

### 2.1 ランダム座標生成（`_get_random_position_in_arena`）

アリーナ内の一様乱数。デフォルトのサンプリング範囲:

- X: `[-0.55, 1.05]` ∩ アリーナ内（`margin` 考慮）
- Y: `[-1.0, 1.0]` ∩ アリーナ内

`margin` 引数で壁からの余白を追加指定可能。

### 2.2 配置手順（`_reset_bodies_random`）

1. **幼児**: `_get_random_position_in_arena()` で配置。ヨー `0`。
2. **親**（`num_agents == 2`）:
   - `parent_invisible_mode`: 画面外
   - それ以外: 最大 50 回試行し、幼児から `min_dist` 以上離れた位置を探す。幼児方向を向く。
3. **関節リセット**
4. **オブジェクト**:
   - `_select_active_objects()`
   - 配置済み座標リストに幼児（と親）を登録
   - 各アクティブオブジェクトについて:
     - 固定オブジェクト: `initial_pos` の XY を使用
     - 移動可能: 最大 100 回、`_get_random_position_in_arena()` で候補を生成し、既存配置との距離が `min_dist` 以上なら採用

---

## 3. 幾何学的配置（`placement_method="geometric"`）

`_select_active_objects()` は呼ばず、`_reset_bodies_geometric()` 内で直接 `active_objects` を構築する。評価・実験向けの制約付き配置。

### 3.1 固定パラメータ（メソッド内ローカル）

| 変数 | 値 |
|---|---|
| `infant_pos` | `[-2.15, 0, 0.0]`（固定、−X 側） |
| `infant_obj_dist_range` | `(0.9, 1.5)` [m] |
| `obj_parent_dist_range` | `(0.9, 1.5)` [m] |
| `infant_parent_angle_range` | `(-30°, +30°)` |
| `infant_parent_dist_range` | `(0.9, 1.5)` [m] |
| `obj_parent_angle_range` | `(-120°, +120°)` |
| `shielding_thresh` | `15°` |
| `xpi_angle_thresh` | `120°` |
| `max_parent_trials` | `100` |

`infant_obj_angle_range` はコンストラクタ kwargs（未指定時 `(-30°, +30°)`）。

### 3.2 配置手順

#### Step 1: 幼児

固定位置 `[-2.15, 0, 0]`、ヨー `0`。

#### Step 2: ターゲットオブジェクト（`num_objects >= 1`）

1. 幼児を原点とした極座標で角度・距離をサンプリング
2. 直交座標に変換後、`_clamp_xy_inside_arena(margin)` でクリップ
3. オブジェクトプールから 1 体をランダム選択し `target_obj_id` として配置
4. `active_objects = [target_obj_id]`

#### Step 3: 親（`num_agents == 2` かつ `parent_invisible_mode=False`）

3 つの分岐があり、主に使われるのは **`obj_parent_angle_range` が指定されている場合**（デフォルトの `(-120°, +120°)`）:

1. オブジェクトを基準に、オブジェクト→親の角度・距離をサンプリングして親位置を決定
2. 導出された幼児→親の角度・距離が `infant_parent_*_range` 内か検証
3. 幼児→オブジェクトと同符号の角度制約（遮蔽回避のため、符号が逆なら距離条件で代替）
4. **遮蔽チェック**: 幼児→親と幼児→オブジェクトの角度差が `shielding_thresh` 以下 **かつ** 親距離 ≤ オブジェクト距離 なら不採用
5. **∠XPI チェック**: オブジェクト(X)・親(P)・幼児(I) の角度 ∠XPI < `xpi_angle_thresh`
6. アリーナ内（`margin`）に収まるか

100 回試行で見つからない場合はフォールバック: 幼児の +X 方向に `infant_parent_dist_range` の上限距離。

**親の向き**: オブジェクトがある場合、∠IPX の二等分線方向（`angle_bisector_theorem`）へヨーを設定。オブジェクトなしなら `π`。

#### Step 4: 残りオブジェクト（`num_objects > 1`）

1. ターゲット以外から `num_objects - 1` 体をランダム選択し `active_objects` に追加
2. 各オブジェクトについて:
   - **優先**: 幼児を基準とした角度・距離サンプリング（ターゲットと同様の範囲）、アリーナ内 + `min_dist` チェック（最大 100 回）
   - **フォールバック**: `_get_random_position_in_arena(margin)` で距離のみ満たす配置（最大 100 回）
   - **最終フォールバック**: 警告を出しランダム配置

---

## 補助関数

### `_clamp_xy_inside_arena(x, y, margin)`

`(x, y)` を `[-xh+margin, +xh-margin] × [-yh+margin, +yh-margin]` にクリップ。

### `_get_random_position_in_arena(margin, x_range, y_range)`

アリーナと指定範囲の交差領域から一様乱数。戻り値 `(x, y, 0.0)`。

### `angle_bisector_theorem(A, B, C)`

点 B における ∠ABC の二等分線上の点 D（辺 AC 上）を返す。親の視線方向計算に使用。

### `_setup_fixed_placements(fixed_placements)`

固定オブジェクトリストの検証・正規化。Y 帯域チェックを行い `self.fixed_placements` に格納。

### `_reserved_grid_rows()`

`{0, arena_rows - 1}` を返す。予約行の判定に使用。

---

## 向き（ヨー角）のまとめ

| 対象 | grid | random | geometric |
|---|---|---|---|
| 幼児 | `0`（+X） | `0` | `0` |
| 親 | 幼児方向 `atan2` | 幼児方向 `atan2` | ∠IPX 二等分線方向（オブジェクトあり）/ `π`（なし） |
| 可動オブジェクト | 単位クォータニオン | 同左 | 単位クォータニオン |
| 固定オブジェクト | `fixed_placements` の `yaw`（度→rad） | — | — |

---

## 関連ソース

| 処理 | 関数 |
|---|---|
| 配置方式分岐 | `_reset_bodies` |
| グリッド初期化 | `_setup_layout_grid`, `_initialize_cells` |
| 固定オブジェクト検証 | `_setup_fixed_placements`, `_is_fixed_placement_y_allowed` |
| 予約行除外 | `_reserved_grid_rows`, `_object_cell_candidates_without_reserved_rows` |
| グリッド配置 | `_reset_bodies_grid` |
| ランダム配置 | `_reset_bodies_random` |
| 幾何配置 | `_reset_bodies_geometric` |
| URDF 読込・固定オブジェクト | `_load_urdfs` |
| アクティブ選択 | `_select_active_objects` |
| ランダム座標 | `_get_random_position_in_arena` |
| デモ | `cueworld/demo/demo_initialize_pos.py` |
