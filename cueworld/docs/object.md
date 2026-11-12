# 環境内オブジェクト (`objects/`)

`cueworld/objects/` は、シミュレーション空間に置かれる物体の識別属性と物理寸法、アフォーダンスを定義する。
本ドキュメントは**物体側の値の正本**である。

- 物体をどう掴むか・どこに置けるか・振ると何が起きるかという**環境側の判定**は [env.md](env.md)。
- どこに配置されるかは [position.md](position.md)。
- 物体の `concept_id` から単語・音素へ展開する規則は [language.md](language.md)。

物体は `BaseObject`（`objects/base.py`）を継承し、`objects/items.py` でクラスとして定義される。
`BaseEnv` は初回 `reset()` で `object_counts` に従ってインスタンスを生成し、URDF をロードして `body_id` を割り当てる（`all_objects: body_id → インスタンス`）。

---

## 1. カタログ

現行カタログは **可動プリミティブ 16 種**と **board 5 種**のみ。

### 可動プリミティブ（`PrimitiveObject`）

4 色 × 4 形 = 16 クラス。クラス名は `RedCube`, `BlueSphere`, `GreenCone`, `YellowCylinder` のように「色 + 形」。

| 属性 | 値 |
|------|-----|
| `color` | `red` / `blue` / `green` / `yellow` |
| `shape_type` | `cube` / `sphere` / `cylinder` / `cone` |
| `concept_id` | `f"{color}_{shape_type}"`（例: `red_cube`） |
| `urdf_file` | `f"{color}_{shape_type}.urdf"` |
| `is_graspable` | `True` |
| `can_shake` | `True` |
| `move_type` | `1`（可動。毎エピソード再配置される） |
| `is_platform` | `False` |
| `texture` | `"smooth"` |
| `preference` | `1.0`（選好の重み。報酬計算のみ。観測には出ない） |

識別は `concept_id`（色 + 形の複合 ID）で行う。色だけ・形だけで一意にはならない。

### 色ボード（`ColorBoard`）

`RedBoard` / `BlueBoard` / `GreenBoard` / `YellowBoard` の 4 種。

| 属性 | 値 |
|------|-----|
| `color` | 対応する色 |
| `concept_id` | `f"{color}_board"` |
| `urdf_file` | `f"board_{color}.urdf"` |
| `accepts_category` | 対応する色（受け付ける item の色） |
| `is_platform` | `True` |
| `is_graspable` | `False` |
| `move_type` | `None`（固定物。`fixed_placements` でワールド座標を指定する） |
| `objects_on_board` | 載っている物体 id のリスト（実行時に更新） |

`accepts_category` は家事タスクの「どの色の物をどの board に集めるか」の対応づけに使う。**判定は色の一致のみ**。

### 汎用ボード（`Board`）

`accepts_category` が `None`、`color` も `None`。色タスクの候補には入らない。壁面ボードとして配置に使う。

`ColorBoard` は `Board` を継承しているので、上記の `is_platform` / `is_graspable` / `move_type` は共通。

---

## 2. 寸法

`bbox_half_extents` は `(x, y, z)` の半長 [m]。環境はこれを次の 2 か所で読む。

- 把持距離 `grasp_distance` の自動計算（**掴める物体** `is_graspable=True` の最大半長を使う。[env.md §8](env.md#8-到達圏と配置距離)）
- 簡易物理の着地高さ（`bbox_half_extents[2]`。[env.md §10](env.md#10-簡易物理)）

| 形 | `bbox_half_extents` |
|----|---------------------|
| `cube` | `[0.06, 0.06, 0.06]` |
| `sphere` | `[0.06, 0.06, 0.06]` |
| `cylinder` | `[0.05, 0.05, 0.06]` |
| `cone` | `[0.06, 0.06, 0.065]` |
| `Board` / `ColorBoard` | `[0.01, 0.4, 0.5]` |

プリミティブの `size` は `max(bbox_half_extents) × 2.0` として自動的に入る。

Board は X に薄く Y に広い板。壁沿いに並べるときは `yaw=90` で向きを変える（[position.md](position.md)）。

---

## 3. アフォーダンス

残っているアフォーダンスは **振る → 音 / 効果チャネル**と**触覚ベクトル**。

物体は 2 本の 2 次元固定ベクトルを持ち、振る判定が立つと環境がそのまま観測チャネルへ書き込む。
判定条件（把持中・胴体ローカルでの変位・毎 step のゼロクリア）は [env.md §9](env.md#9-振る検出)。
触覚は `tactile_vec` を物体が持ち、幼児の手が把持範囲にあるとき観測 `tactile` に載る（[env.md §6](env.md#6-観測空間)）。

### `shake_sound`（形 → 音）

観測キー `object_sound` に載る。

| `shape_type` | ベクトル |
|--------------|----------|
| `cube` | `[1.0, 0.0]` |
| `sphere` | `[0.0, 1.0]` |
| `cylinder` | `[-1.0, 0.0]` |
| `cone` | `[0.0, -1.0]` |

### `shake_effect_vec`（色 → 効果）

観測キー `object_effect` に載る。

| `color` | ベクトル |
|---------|----------|
| `red` | `[1.0, 0.0]` |
| `blue` | `[0.0, 1.0]` |
| `green` | `[-1.0, 0.0]` |
| `yellow` | `[0.0, -1.0]` |

どちらも単位円上の 4 点で、ノルムは 1。無音・無効果の `[0, 0]` と必ず区別できる。

**音を形に、効果を色に割り当てているのは意図的**。board の判定と語彙の色成分が色に寄っているため、振る音まで色に載せると手がかりが色に偏る。
`shake_effect_vec` は観測チャネルの値であって、board の受け入れ判定や語彙の色同一性には影響しない。

### `tactile_vec`（触覚）

観測キー `tactile` に載る（手が `graspable_obj` のときその物体のベクトルをコピー。非接触は `[0, 0]`）。

- `concept_id`（色 × 形）ごとに一意。単位円上、ゼロから離す。
- 形クラスタ中心は `shake_sound` と同じ方位（cube 0°、sphere 90°、cylinder 180°、cone -90°）。
- 色オフセット（全形共通）: red -15°、blue -5°、green +5°、yellow +15°。
- 同じ形のクラスは近く、別形より遠い。
- board は `is_graspable=False` のため観測に乗らない。`tactile_vec` は既定 `[0, 0]`。

割り当ては `objects/items.py` の `tactile_vec_for_concept` が正本。

---

## 4. 実行時に書き換わる属性

インスタンスは静的な定義だけでなく、エピソード中の状態も持つ。

| 属性 | 意味 | 書き換える場所 |
|------|------|----------------|
| `id` | PyBullet の body id | 初回ロード時 |
| `put_on` | どの platform に載っているか（body id / `None`） | 把持解除時（[env.md §7](env.md#7-接触把持受け渡し)） |
| `objects_on_board` | その platform に載っている物体 id | 同上 |
| `initial_pos` / `initial_orn` | 固定物のワールド座標 | `fixed_placements` のロード時 |
| `contained_in` / `contents` | 容器関係 | 現行カタログでは常に空（[§6](#6-注意事項)） |
| `lid_link_id` / `lid_joint` | 蓋のリンク・関節 | ロード時に探索。現行 URDF には無いので `None` |

`put_on` が立っている物体は簡易物理の落下対象から外れ、家事タスクの候補からも外れる。

---

## 5. オブジェクト集合の決まり方

- `object_counts`（`クラス名 → 個数`、既定 `{"RedCube": 3}`）に従って初回 `reset()` でプールを作る。プール生成は 1 回だけで、body id はエピソードをまたいで変わらない。
- 固定物（board 等）は `fixed_placements` で別に指定する。ロード時に `move_type = None` が上書きされる。
- 毎エピソード、そのうち `num_objects` 個が `active_objects` として選ばれる。非アクティブな物体は画面外に置かれ、把持判定・接触判定・視線の対象にならない。
- 選び方の詳細は [position.md](position.md)。

---

## 6. 注意事項

**contain / open は無効化されている。**
`is_containable` は `BaseObject` の既定が `False` で、`PrimitiveObject` も明示的に `False`。`can_contain` / `is_openable` も既定 `False` のまま。したがって「容器に入れる」「蓋を開ける」は現行カタログでは発生しない。`objects/base.py` と `base.py` にコードパスは残っている。

**`BaseObject` には現行カタログが使わないフィールドが残っている。**
`is_food`, `dangerous`, `is_toy`, `is_laundry`, `target_category`, `visual_shape`, `open` などは、いずれも現行の 21 クラスでは既定値のままで、判定にも使われない。

**旧オブジェクト定義は `items_legacy.py` にある。**
`ToyCar`, `Banana`, `Box`, `Refrigerator` などは互換参照用に退避してあり、`items.py` からはロードされない。URDF / OBJ 資産は `objects/assets/` に残っているが、`object_counts` や `fixed_placements` でこれらのクラス名を指定しても `items.py` に無いためロードに失敗する（警告を出してスキップ）。

**`accepts_category` の意味が `BaseObject` のコメントと食い違う。**
`objects/base.py` のコメントは「属性名（例: `is_toy`）」と書いてあるが、実際に入るのは色文字列（`"red"` など）で、判定も色一致。

**`preference`（選好）**
可動プリミティブは `1.0`（`PrimitiveObject`）。Board / `ColorBoard` は `0.0`（`BaseObject` 既定）。観測には出ない。報酬は [infant_signal.md §10](infant_signal.md#10-選好報酬) が正本。親への選好は `BaseEnv.parent_preference`（既定 `1.2`）。

---

## 7. 設計上の判断メモ

- **プリミティブに限る理由**: 複雑な形状は親の把持失敗を増やすが、把持の難しさは本研究の対象ではない。4 形状で把持成功率に偏りがないことは実測で確認済み（各 30 試行、いずれも 30/30）。
- **形ごとに物理を変えない**: 球だけ転がる、円錐だけ倒れる、といった差は付けていない。対象が動くと共同注意が不安定になり、デモが形ごとに分岐してしまうため。
- **色と形で役割を分ける**: 識別は色 × 形の複合 ID、振る音は形、振る効果は色、board の受け入れは色。「何の物体か」と「何が起きるか」を分けている。
